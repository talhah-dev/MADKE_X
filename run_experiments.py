#!/usr/bin/env python
"""Unified MADKE / MADKE-X experiment runner.

``--mode offline``
    Deterministic scripted agents (no LLM, no API key).  Verifies the SHA-256
    digests of the five raw dataset files, runs the unpatched baseline
    ("MADKE (re-run)") and MADKE-X over all 500 questions of every dataset (and
    reports the 160-question subset), checks the pipeline invariants and writes
    the JSON / Markdown material for Tables 3-9 to ``results/offline/``.

``--mode live``
    Any OpenAI-compatible endpoint (vLLM / Ollama / llama.cpp / OpenAI /
    Azure).  Runs both systems with per-question checkpointing (interrupted runs
    resume automatically) and writes Table 10 (F1, EM, Cons, Co2, support rate,
    paired McNemar p-values) to ``results/live/<run_tag>/``.

Examples::

    python run_experiments.py --mode offline
    python run_experiments.py --mode live --base_url http://localhost:8000/v1 --api_key EMPTY \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ --end_index 500 --concurrency 8
"""

import argparse
import json
import os
import platform
import re
import sys
import threading
import time
import tracemalloc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from debate import CODE_VERSION, legacy_config, madke_x_config, run_debate, select_prompt
from models.openai_compat_agent import OpenAICompatAgent, preflight
from models.scripted_agent import ScriptedAgent
from parser import add_agent_backend_args
from utils.calculate_tokens import count_tokens, tokenizer_name
from utils.data_utils import (
    DATASETS,
    append_jsonl,
    build_bm25_index,
    build_pool,
    legacy_pool,
    load_dataset,
    read_jsonl_data,
    sha256_text,
    task_type,
    verify_data_checksums,
    write_json_data,
    write_text,
)
from utils.evaluation_utils import (
    SUPPORT_STATUSES,
    extract_final_answer,
    is_verdict_answer,
    normalize_text,
    parse_evidence_ids,
    parse_evidence_ids_legacy,
    parse_judge_verdict,
    parse_judge_verdict_legacy,
    support_check,
)
from utils.metrics_calculator import (
    average_rounds,
    bootstrap_ci,
    compute_co2,
    compute_co2_original,
    compute_cons,
    compute_cons_original,
    compute_late_agreement,
    compute_em_f1,
    coverage_and_mrr,
    expected_random_coverage,
    expected_random_reciprocal_rank,
    first_hit_rank,
    has_span_answer,
    is_completed,
    is_correct,
    paired_difference_ci,
    paired_em_vectors,
    paired_mcnemar,
    passage_contains_any,
    support_statistics,
    verification_statistics,
)
from utils.prompt_utils import format_evidence_pool

SYSTEM_LABEL = {"madke_legacy": "MADKE (re-run, unpatched)", "madke_x": "MADKE-X"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def pct(x, nd=1):
    return "{:.{nd}f}".format(100.0 * x, nd=nd)


def num(x, nd=2):
    return "{:.{nd}f}".format(x, nd=nd)


def ci(lo_hi, nd=1):
    return "[{:.{nd}f}, {:.{nd}f}]".format(lo_hi[0], lo_hi[1], nd=nd)


def pval(p):
    return "<0.001" if p < 0.001 else "{:.3f}".format(p)


def md_table(headers, rows):
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def slugify(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text or "model").strip("_") or "model"


def log(msg):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), msg), flush=True)


def make_agents(args, dataset_name):
    if args.mode == "offline":
        def mk(name):
            return ScriptedAgent(name, seed=args.seed, fail_rate=args.fail_rate, dataset_name=dataset_name)
    else:
        def mk(name):
            return OpenAICompatAgent(args, name)
    return [mk("Agent_{}".format(i)) for i in range(args.agent_num)], mk("Judge"), mk("Summarizer")


def failed_record(data, index, cfg, error):
    return {"index": index, "question": data["question"], "answer": data["answer"], "llm_answers": [],
            "final_answer": None, "Summary": "", "stop_round": None, "status": "failed",
            "error": {"stage": "runner", "agent": None, "round": None, "error": error, "attempts": None},
            "trace": {"config": cfg.to_dict(), "pool_size": 0, "pool_manifest": [], "rounds": [], "judge_verdicts": [],
                      "union_selected_ids": [], "cited_ids": [], "cited_valid_ids": [], "cited_invalid_ids": [],
                      "support_status": None, "initial_support_status": None, "verification": {"attempted": False},
                      "summariser_binding_ok": None, "calls": {}, "tokens": {"prompt": 0, "completion": 0},
                      "failure": {"stage": "runner", "error": error}}}


# ---------------------------------------------------------------------------
# Running a configuration (with optional per-question checkpointing)
# ---------------------------------------------------------------------------

class LiveRunAborted(RuntimeError):
    """Raised when too many consecutive questions hit failed model calls."""


def record_had_call_failure(rec):
    """True if a question failed (MADKE-X) or carried a failed call on (legacy)."""
    return rec.get("status") == "failed" or (rec.get("trace") or {}).get("call_failures", 0) > 0


def run_records(data, indices, cfg, prompt, agents, checkpoint=None, concurrency=1, retry_failed=False, label="",
                max_consecutive_failures=0, bm25_index=None):
    debaters, judge, summarizer = agents
    done = {}
    if checkpoint and os.path.exists(checkpoint):
        # Records written by another version of the pipeline are never resumed: results of
        # different versions must not be mixed in one table.
        current, stale = [], 0
        for rec in read_jsonl_data(checkpoint, verbose=False):
            version = ((rec.get("trace") or {}).get("config") or {}).get("code_version")
            if version != CODE_VERSION:
                stale += 1
            else:
                current.append(rec)
        if stale:
            backup = "{}.stale-{}".format(checkpoint, time.strftime("%Y%m%d-%H%M%S"))
            os.replace(checkpoint, backup)
            with open(checkpoint, "w", encoding="utf-8") as fw:
                for rec in current:
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log("{}: {} checkpointed questions were written by a different code version (current {}); they are "
                "not resumed and were moved to {}".format(label, stale, CODE_VERSION, backup))
        kept = []
        for rec in current:
            if retry_failed and rec.get("status") == "failed":
                continue
            if rec.get("index") in indices:
                done[rec["index"]] = rec
                kept.append(rec)
        if retry_failed:
            with open(checkpoint, "w", encoding="utf-8") as fw:
                for rec in kept:
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if done:
            log("{} resume: {} questions already checkpointed in {}".format(label, len(done), checkpoint))

    todo = [i for i in indices if i not in done]
    results = dict(done)
    lock = threading.Lock()
    t0 = time.time()
    progress = {"n": 0, "consecutive_failures": 0}
    stop = threading.Event()

    def work(i):
        if stop.is_set():
            return None
        try:
            return run_debate(data[i], debaters, judge, summarizer, prompt, cfg, index=i, bm25_index=bm25_index)
        except Exception as ex:  # never lose a run because of one question
            return failed_record(data[i], i, cfg, "runner exception {}: {}".format(type(ex).__name__, ex))

    pending = []   # failed records of the current failure streak (kept only if the streak ends)

    def commit(rec):
        results[rec["index"]] = rec
        if checkpoint:
            append_jsonl(checkpoint, rec)
        progress["n"] += 1

    def store(rec):
        if rec is None:
            return
        with lock:
            if max_consecutive_failures:
                if record_had_call_failure(rec):
                    pending.append(rec)
                    if len(pending) >= max_consecutive_failures:
                        stop.set()
                        last = rec.get("error") or {}
                        raise LiveRunAborted(
                            "{}: {} consecutive questions had failed model calls (last error: {}). The run was "
                            "stopped and these questions were NOT checkpointed, so a re-run will retry them. "
                            "Check that the model server is running and --base_url/--model are correct.".format(
                                label, len(pending), last.get("error") or "call failure carried on (legacy)"))
                    return
                for old in pending:   # isolated failures: keep them
                    commit(old)
                pending.clear()
            commit(rec)
            n = progress["n"]
            if label and (n % 25 == 0 or n == len(todo)):
                elapsed = time.time() - t0
                log("{}: {}/{} done ({:.1f}s, {:.2f} s/q)".format(label, n, len(todo), elapsed, elapsed / max(n, 1)))

    if todo:
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(work, i) for i in todo]
                try:
                    for fut in as_completed(futures):
                        store(fut.result())
                except LiveRunAborted:
                    for f in futures:
                        f.cancel()
                    raise
        else:
            for i in todo:
                store(work(i))
    with lock:   # a short failure streak at the very end is genuine: keep it
        for old in pending:
            commit(old)
        pending.clear()
    return [results[i] for i in indices if i in results]


# ---------------------------------------------------------------------------
# Pool analysis: data only, no agents involved
# ---------------------------------------------------------------------------

def _answers(q):
    return q["answer"] if isinstance(q["answer"], list) else [q["answer"]]


def _best_time(fn, repeat=3):
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def _peak_memory(fn):
    tracemalloc.start()
    keep = fn()                       # the result stays alive while the peak is read
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del keep
    return peak


def pool_analysis(dataset_name, data, indices, wiki_k, google_k, rrf_k, n_boot=1000, seed=0, timing=True,
                  bm25_index=None):
    """Compare the pool of the release (wiki 10 + Google 5, listed as retrieved) with the pool of
    MADKE-X (wiki ``wiki_k`` + Google ``google_k``, de-duplicated, RRF order).

    Answer-string measures (coverage, ordering) use whole-word matching of the normalised gold
    answers and exclude questions whose gold answer is a verdict (yes / no / FEVER label).
    Timing is measured without memory tracing (best of three passes); memory in a separate pass.
    """
    extractive = task_type(dataset_name) == "extractive"
    qs = [data[i] for i in indices]
    n = len(qs)

    def build_x():
        return [build_pool(q["question"], q["wiki_retrieval"], q["google_retrieval"], w=wiki_k, g=google_k,
                           bm25_index=bm25_index, rrf_k=rrf_k)
                for q in qs]

    def build_original():             # what the released code does: two list slices
        return [q["wiki_retrieval"][:10] + q["google_retrieval"][:5] for q in qs]

    cost = None
    if timing:
        cost = {"index_build_ms": 1000.0 * _best_time(lambda: build_bm25_index(data, wiki_k, google_k)),
                "index_passages": bm25_index.n_docs if bm25_index is not None else None,
                "x_build_ms_per_q": 1000.0 * _best_time(build_x) / n,
                "original_build_ms_per_q": 1000.0 * _best_time(build_original) / n,
                "x_peak_memory_kb": _peak_memory(build_x) / 1024.0,
                "original_peak_memory_kb": _peak_memory(build_original) / 1024.0}

    agg = Counter()
    ranks = {k: [] for k in ("original", "specified", "interleaved", "bm25", "rrf")}
    rand = Counter()
    cover = Counter()
    hit3_original, hit3_rrf = [], []
    n_span = 0
    for q, (pool, meta) in zip(qs, build_x()):
        lpool, _ = legacy_pool(q["wiki_retrieval"], q["google_retrieval"])
        spec_pool, _ = build_pool(q["question"], q["wiki_retrieval"], q["google_retrieval"], w=wiki_k, g=google_k,
                                  dedup=False, rrf=False)
        agg["original_passages"] += len(lpool)
        agg["specified_passages"] += len(spec_pool)
        agg["x_passages"] += len(pool)
        agg["dups_removed"] += meta["duplicates_removed"]
        agg["questions_with_dups"] += 1 if meta["duplicates_removed"] else 0
        agg["original_listing_tokens"] += count_tokens(format_evidence_pool(lpool))
        agg["specified_listing_tokens"] += count_tokens(format_evidence_pool(spec_pool))
        agg["x_listing_tokens"] += count_tokens(format_evidence_pool(pool))

        answers = _answers(q)
        if not extractive or not has_span_answer(answers):
            continue
        n_span += 1
        n_hits = sum(1 for p in pool if passage_contains_any(p["contents"], answers))
        orders = {
            "original": lpool,
            "specified": spec_pool,
            "interleaved": sorted(pool, key=lambda p: (p["source_rank"], 0 if p["source"] == "wiki" else 1)),
            "bm25": sorted(pool, key=lambda p: p["bm25_rank"]),
            "rrf": pool,
        }
        for name, ordered in orders.items():
            ranks[name].append(first_hit_rank([p["contents"] for p in ordered], answers))
        for k in (1, 3, 5):
            rand["cov@{}".format(k)] += expected_random_coverage(len(pool), n_hits, k)
        rand["mrr"] += expected_random_reciprocal_rank(len(pool), n_hits)
        cover["original"] += 1 if ranks["original"][-1] is not None else 0
        cover["specified"] += 1 if ranks["specified"][-1] is not None else 0
        cover["x"] += 1 if n_hits > 0 else 0
        hit3_original.append(1 if (ranks["original"][-1] or 99) <= 3 else 0)
        hit3_rrf.append(1 if (ranks["rrf"][-1] or 99) <= 3 else 0)

    out = {
        "n": n,
        "task_type": task_type(dataset_name),
        "tokenizer": tokenizer_name(),
        "passages": {"original": agg["original_passages"] / n, "specified": agg["specified_passages"] / n,
                     "x": agg["x_passages"] / n},
        "listing_tokens": {"original": agg["original_listing_tokens"] / n,
                           "specified": agg["specified_listing_tokens"] / n,
                           "x": agg["x_listing_tokens"] / n,
                           "reduction_pct": (100.0 * (1.0 - agg["x_listing_tokens"] / agg["original_listing_tokens"])
                                             if agg["original_listing_tokens"] else 0.0)},
        "duplicates_removed": agg["dups_removed"],
        "questions_with_duplicates": agg["questions_with_dups"],
        "cost": cost,
        "n_span": n_span if extractive else 0,
        "n_verdict_gold": (n - n_span) if extractive else n,
        "coverage_pct": None, "ordering": None, "paired_cov3": None,
    }
    if extractive and n_span:
        out["coverage_pct"] = {k: 100.0 * cover[k] / n_span for k in ("original", "specified", "x")}
        ordering = {name: coverage_and_mrr(r) for name, r in ranks.items()}
        ordering["random"] = {"n": n_span, "cov@1": rand["cov@1"] / n_span, "cov@3": rand["cov@3"] / n_span,
                              "cov@5": rand["cov@5"] / n_span, "mrr": rand["mrr"] / n_span,
                              "any_hit": cover["x"] / n_span}
        out["ordering"] = ordering
        mc = paired_mcnemar(hit3_original, hit3_rrf)
        out["paired_cov3"] = {
            "n": n_span, "original_only": mc["b_a_only"], "rrf_only": mc["c_b_only"],
            "gain_pts": 100.0 * (sum(hit3_rrf) - sum(hit3_original)) / n_span,
            "ci95": paired_difference_ci(hit3_original, hit3_rrf, n_boot=n_boot, seed=seed),
            "p_value": mc["p_value"],
        }
    return out


# ---------------------------------------------------------------------------
# Interface behaviour on canonical reply formats (deterministic)
# ---------------------------------------------------------------------------

# Each case is (name, reply): the reply is the text the parser receives.  The pool has ten passages.
EVIDENCE_FORMAT_CASES = [
    ("separate brackets", "[2], [0] and [7]"),
    ("list in one bracket", "[2, 0, 7]"),
    ("repeated identifier", "[2] is the key passage. Selection: [2] [0] [7]"),
    ("passages discussed first, selection last", "[1] and [5] are off topic. Helpful: [2] [0] [7]"),
    ("more than three listed", "[2] [0] [7] [1] [5]"),
    ("identifier outside the pool", "[2] [0] [12]"),
    ("lower-case not-found marker", "[3] is unrelated. [no found]"),
]

# Each case is (name, judge reply, verdict a reader would take from it).
VERDICT_FORMAT_CASES = [
    ("plain", "[Yes]", "yes"),
    ("quoted", '["Yes"]', "yes"),
    ("lower case", "[yes]", "yes"),
    ("trailing punctuation", "[Yes.]", "yes"),
    ("answers echoed before the verdict", "Agent 0: [Yes]. Agent 1: [No]. Inconsistent: [No]", "no"),
]


def interface_format_tables(pool_size=10, cap=3):
    """Behaviour of the released parsers and of the MADKE-X parsers on canonical reply formats."""
    evidence = []
    for name, reply in EVIDENCE_FORMAT_CASES:
        new = parse_evidence_ids(reply, pool_size, cap=cap)
        old = parse_evidence_ids_legacy(reply, pool_size)
        evidence.append({"format": name, "reply": reply, "pool_size": pool_size,
                         "released_delivered": list(old.ids), "x_delivered": list(new.ids),
                         "x_not_found": bool(new.not_found), "x_dropped": list(new.dropped),
                         "x_ignored": list(new.ignored), "x_out_of_range": list(new.out_of_range),
                         "same": list(old.ids) == list(new.ids)})
    verdict = []
    for name, reply, intended in VERDICT_FORMAT_CASES:
        new = parse_judge_verdict(reply)
        old = parse_judge_verdict_legacy(reply)
        verdict.append({"format": name, "reply": reply, "intended": intended, "released": old, "x": new,
                        "released_correct": old == intended, "x_correct": new == intended})
    return {"evidence": evidence, "verdict": verdict}


# ---------------------------------------------------------------------------
# Verification stress test (scripted, offline)
# ---------------------------------------------------------------------------

def verification_stress(args, dataset_name, data, indices, bm25_index=None):
    """Run MADKE-X with a careless scripted summariser that always cites the first listed passage.

    Many first answers then fail the support check, so the bounded verification step is exercised
    on a large share of the questions.  The counts describe the mechanism, not a language model.
    """
    cfg = madke_x_config(dataset_name, agent_num=args.agent_num, max_round=args.max_round,
                         wiki_k=args.wiki_k, google_k=args.google_k, cap=args.cap, rrf_k=args.rrf_k)

    def mk(name, **kw):
        return ScriptedAgent(name, seed=args.seed, dataset_name=dataset_name, **kw)

    agents = ([mk("Agent_{}".format(i)) for i in range(args.agent_num)], mk("Judge"), mk("Summarizer", cite_style="first"))
    recs = run_records(data, indices, cfg, select_prompt(dataset_name), agents, bm25_index=bm25_index)
    c = Counter()
    for r in recs:
        if not is_completed(r):
            continue
        tr = r["trace"]
        ver = tr["verification"]
        first_answer = extract_final_answer(r["Summary"])
        if tr["initial_support_status"] == "not_applicable":
            c["not_applicable"] += 1
            continue
        c["applicable"] += 1
        if tr["initial_support_status"] == "supported":
            c["initial_supported"] += 1
            if r["final_answer"] != first_answer or ver.get("attempted"):
                c["supported_answer_touched"] += 1          # must stay 0
        if tr["support_status"] == "supported":
            c["final_supported"] += 1
        if ver.get("attempted"):
            c["verification_calls"] += 1
            if ver.get("adopted"):
                c["adopted"] += 1
                if normalize_text(ver.get("answer")) != normalize_text(first_answer):
                    c["adopted_new_answer"] += 1
            elif ver.get("abstained"):
                c["abstained"] += 1
            else:
                c["rejected"] += 1
    inv = check_invariants(recs, cfg, data, indices, bm25_index=bm25_index)
    keys = ("applicable", "not_applicable", "initial_supported", "final_supported", "verification_calls", "adopted",
            "adopted_new_answer", "abstained", "rejected", "supported_answer_touched")
    out = {k: c[k] for k in keys}
    out.update({"n": len(recs), "invariants_all_passed": inv["all_passed"],
                "initial_support_rate": c["initial_supported"] / c["applicable"] if c["applicable"] else 0.0,
                "final_support_rate": c["final_supported"] / c["applicable"] if c["applicable"] else 0.0})
    return out


# ---------------------------------------------------------------------------
# Invariants (MADKE-X records)
# ---------------------------------------------------------------------------

def _rebuild_pool(rec, cfg, data, bm25_index=None):
    q = data[rec["index"]]
    if cfg.evidence_type == "no_evidence":
        return []
    if cfg.legacy:
        return legacy_pool(q["wiki_retrieval"], q["google_retrieval"], cfg.wiki_k, cfg.google_k)[0]
    return build_pool(q["question"], q["wiki_retrieval"], q["google_retrieval"], w=cfg.wiki_k, g=cfg.google_k,
                      bm25_index=bm25_index, dedup=cfg.dedup, rrf=cfg.rrf, rrf_k=cfg.rrf_k)[0]


def inv_status(rec, cfg, pool):
    if rec["status"] not in ("completed", "failed"):
        return False, "status={}".format(rec["status"])
    if rec["status"] == "completed":
        if not isinstance(rec.get("final_answer"), str) or not rec.get("Summary"):
            return False, "completed without answer/summary"
    else:
        if rec.get("final_answer") is not None or not rec.get("error"):
            return False, "failed record carries an answer or no error"
    return True, None


def inv_pool(rec, cfg, pool):
    tr = rec["trace"]
    man = tr["pool_manifest"]
    if tr["pool_size"] != len(man) or len(pool) != len(man):
        return False, "manifest size mismatch"
    if len(man) > cfg.wiki_k + cfg.google_k:
        return False, "pool larger than w+g"
    for i, (m, p) in enumerate(zip(man, pool)):
        if m["index"] != i or m["source"] not in ("wiki", "google"):
            return False, "bad manifest entry {}".format(i)
        k = cfg.wiki_k if m["source"] == "wiki" else cfg.google_k
        if not 1 <= m["source_rank"] <= k:
            return False, "source_rank out of range"
        if m["sha256"] != p["sha256"] or m["sha256"] != sha256_text(p["contents"]):
            return False, "manifest hash not reproducible"
    if cfg.dedup and len({normalize_text(p["contents"]) for p in pool}) != len(pool):
        return False, "duplicate passages after dedup"
    if cfg.rrf:
        scores = [m["rrf_score"] for m in man]
        if any(scores[i] < scores[i + 1] - 1e-12 for i in range(len(scores) - 1)):
            return False, "rrf_score not non-increasing"
    return True, None


def inv_evidence_ids(rec, cfg, pool):
    n = len(pool)
    for rt in rec["trace"]["rounds"]:
        for name, a in rt["agents"].items():
            sel, drop, ign, oor = a["selected_ids"], a["dropped_ids"], a["ignored_ids"], a["out_of_range_ids"]
            if len(sel) > cfg.cap or len(set(sel)) != len(sel):
                return False, "{} r{}: cap/duplicate violated {}".format(name, rt["round"], sel)
            if any(not 0 <= i < n for i in sel + drop + ign):
                return False, "{} r{}: in-range list contains out-of-range id".format(name, rt["round"])
            if set(sel) & set(drop):
                return False, "{} r{}: dropped overlaps selected".format(name, rt["round"])
            if any(0 <= i < n for i in oor):
                return False, "{} r{}: out_of_range contains in-range id".format(name, rt["round"])
            if a["not_found"] and sel:
                return False, "{} r{}: not_found with selection".format(name, rt["round"])
    return True, None


def inv_judge(rec, cfg, pool):
    tr = rec["trace"]
    v = tr["judge_verdicts"]
    if any(x not in ("yes", "no", "unparsed") for x in v):
        return False, "unknown verdict"
    if rec["status"] == "completed":
        if len(v) != len(tr["rounds"]) or len(v) != rec["stop_round"] + 1:
            return False, "verdict/round count mismatch"
        if "yes" in v and v.index("yes") != len(v) - 1:
            return False, "debate continued after agreement"
        if "yes" not in v and len(v) != cfg.max_round:
            return False, "stopped early without agreement"
    return True, None


def inv_citations_support(rec, cfg, pool):
    if rec["status"] != "completed":
        return True, None
    tr = rec["trace"]
    union = set(tr["union_selected_ids"])
    if any(not (0 <= c < len(pool) and c in union) for c in tr["cited_valid_ids"]):
        return False, "cited_valid_ids outside pool/union"
    if tr["support_status"] not in SUPPORT_STATUSES:
        return False, "unknown support status"
    if cfg.task_type == "classification" or is_verdict_answer(rec["final_answer"]):
        return (tr["support_status"] == "not_applicable", "verdict answers must be not_applicable")
    expected = support_check(rec["final_answer"], tr["cited_valid_ids"], pool)
    if expected != tr["support_status"]:
        return False, "support status not reproducible ({} vs {})".format(expected, tr["support_status"])
    return True, None


def inv_verification(rec, cfg, pool):
    if rec["status"] != "completed":
        return True, None
    tr = rec["trace"]
    ver = tr["verification"]
    calls = tr["calls"].get("verifier", 0)
    if calls > 1 or calls != int(bool(ver.get("attempted"))):
        return False, "verifier called {} times".format(calls)
    if ver.get("attempted"):
        if not (cfg.verify and tr["initial_support_status"] not in ("supported", "not_applicable")):
            return False, "verification attempted without trigger"
        if ver.get("adopted"):
            if tr["support_status"] != "supported" or rec["final_answer"] != ver["answer"]:
                return False, "adopted answer not supported / not final"
        else:
            if rec["final_answer"] != extract_final_answer(rec["Summary"]) or tr["support_status"] != tr["initial_support_status"]:
                return False, "non-adopted verification changed the answer"
    elif tr["support_status"] != tr["initial_support_status"]:
        return False, "support status changed without verification"
    return True, None


def inv_summariser_binding(rec, cfg, pool):
    if rec["status"] != "completed":
        return True, None
    return (rec["trace"]["summariser_binding_ok"] is True, "question slot does not hold the question")


def inv_calls(rec, cfg, pool):
    if rec["status"] != "completed":
        return True, None
    tr = rec["trace"]
    c = tr["calls"]
    rounds = len(tr["judge_verdicts"])
    if c.get("summariser") != 1 or c.get("judge") != rounds or c.get("debater") != cfg.agent_num * rounds:
        return False, "call accounting mismatch {}".format(c)
    if pool and cfg.use_self_choose and c.get("evidence") != c.get("debater"):
        return False, "evidence calls != debater calls"
    return True, None


INVARIANTS = [
    ("I1_status_and_answer", inv_status),
    ("I2_pool_manifest_dedup_rrf", inv_pool),
    ("I3_evidence_ids_bounds_cap_order", inv_evidence_ids),
    ("I4_judge_verdicts_early_stop", inv_judge),
    ("I5_citations_support_reproducible", inv_citations_support),
    ("I6_bounded_verification", inv_verification),
    ("I7_summariser_binding", inv_summariser_binding),
    ("I8_call_accounting", inv_calls),
]


def check_invariants(records, cfg, data, indices, bm25_index=None):
    per = {name: {"checked": 0, "passed": 0, "failures": []} for name, _ in INVARIANTS}
    for rec in records:
        pool = _rebuild_pool(rec, cfg, data, bm25_index)
        for name, fn in INVARIANTS:
            ok, detail = fn(rec, cfg, pool)
            per[name]["checked"] += 1
            if ok:
                per[name]["passed"] += 1
            elif len(per[name]["failures"]) < 20:
                per[name]["failures"].append({"index": rec["index"], "detail": detail})
    dataset_level = {}
    got = [r["index"] for r in records]
    dataset_level["D1_all_indices_processed"] = (got == list(indices))
    completed = [r for r in records if is_completed(r)]
    cons = compute_cons(records)["count"]
    co2 = compute_co2(records)["count"]
    manual_cons = sum(1 for r in completed if "yes" in r["trace"]["judge_verdicts"])
    manual_co2 = sum(1 for r in completed if "yes" in r["trace"]["judge_verdicts"] and is_correct(r))
    dataset_level["D2_cons_co2_consistent"] = (cons == manual_cons and co2 == manual_co2)
    dataset_level["D3_only_completed_scored"] = (compute_em_f1(records)["total"] == len(completed))
    n_checks = sum(v["checked"] for v in per.values()) + len(dataset_level)
    n_passed = sum(v["passed"] for v in per.values()) + sum(1 for v in dataset_level.values() if v)
    return {"n_records": len(records), "n_checks": n_checks, "n_passed": n_passed,
            "pass_rate": n_passed / n_checks if n_checks else 1.0, "all_passed": n_checks == n_passed,
            "per_invariant": per, "dataset_level": dataset_level}


# ---------------------------------------------------------------------------
# Metrics per configuration
# ---------------------------------------------------------------------------

def config_metrics(records, cfg, data, n_boot, seed):
    em = compute_em_f1(records)
    cons = compute_cons(records)
    co2 = compute_co2(records)
    cons_orig = compute_cons_original(records, max_round=cfg.max_round)
    co2_orig = compute_co2_original(records, max_round=cfg.max_round)
    cons_true = compute_cons(records, robust=True)      # what the judge replies say, read robustly
    co2_true = compute_co2(records, robust=True)
    late = compute_late_agreement(records)
    supp = support_statistics(records)
    ver = verification_statistics(records)

    diag = Counter()
    stop_dist = Counter()
    tokens = Counter()
    for rec in records:
        tr = rec["trace"]
        pool_size = tr["pool_size"]
        if is_completed(rec):
            stop_dist[len(tr["judge_verdicts"])] += 1
        tokens["prompt"] += tr["tokens"]["prompt"]
        tokens["completion"] += tr["tokens"]["completion"]
        tokens["calls"] += sum(tr["calls"].values()) if tr["calls"] else 0
        for rt in tr["rounds"]:
            for a in rt["agents"].values():
                diag["evidence_replies"] += 1
                diag["with_dropped"] += 1 if a["dropped_ids"] else 0
                diag["with_out_of_range"] += 1 if a["out_of_range_ids"] else 0
                diag["with_ignored"] += 1 if a["ignored_ids"] else 0
                diag["not_found"] += 1 if a["not_found"] else 0
                if a.get("intended_ids") is not None:   # scripted agents state the selection they mean
                    diag["intent_replies"] += 1
                    diag["delivered_set_ok"] += 1 if set(a["selected_ids"]) == set(a["intended_ids"]) else 0
                    diag["delivered_exact_ok"] += 1 if list(a["selected_ids"]) == list(a["intended_ids"]) else 0
            if rt.get("judge_verdict_robust") is not None:
                diag["judge_replies"] += 1
                if rt["judge_verdict_robust"] == "unparsed":
                    diag["judge_unparsed"] += 1
                if rt["judge_verdict_robust"] == "yes":
                    diag["judge_yes_robust"] += 1
                if rt["judge_verdict_robust"] == "yes" and rt["judge_verdict_legacy"] != "yes":
                    diag["judge_agreement_missed_by_legacy"] += 1
                if rt["judge_verdict_robust"] != "yes" and rt["judge_verdict_legacy"] == "yes":
                    diag["judge_false_agreement_legacy"] += 1
        # did the debate stop in the round in which the judge reply first states agreement
        # (robust reading of the recorded replies), or in the last round if it never does?
        robust = [rt.get("judge_verdict_robust") for rt in tr["rounds"] if rt.get("judge_verdict_robust") is not None]
        if is_completed(rec) and robust:
            diag["debates"] += 1
            first_yes = next((i + 1 for i, v in enumerate(robust) if v == "yes"), None)
            if len(robust) == (first_yes if first_yes is not None else cfg.max_round):
                diag["stopped_at_agreement_round"] += 1
        # re-parse the recorded evidence replies with both parsers
        for key in (k for k in rec if re.fullmatch(r"Round_\d+", k)):
            for ck, reply in rec[key].items():
                if ck.startswith("Choose_Evi_") and isinstance(reply, str) and reply:
                    new_ids = parse_evidence_ids(reply, pool_size).ids
                    old_ids = parse_evidence_ids_legacy(reply, pool_size).ids
                    if new_ids != old_ids:
                        diag["parser_disagreement"] += 1
                    if new_ids and old_ids and new_ids != old_ids and set(new_ids) == set(old_ids):
                        diag["parser_order_only"] += 1
        if rec["status"] == "failed":
            diag["failed"] += 1
        if is_completed(rec) and tr.get("summariser_binding_ok") is True:
            diag["binding_ok"] += 1
        if is_completed(rec) and tr.get("call_failures", 0) > 0:
            diag["scored_despite_failed_call"] += 1   # C8: an API error entered the scored answer (legacy only)

    n = len(records)
    return {
        "system": SYSTEM_LABEL.get(cfg.name, cfg.name),
        "config": cfg.to_dict(),
        "n": n, "n_completed": em["total"], "n_failed": em["n_failed"],
        "exact": em["exact"], "exact_ci95": bootstrap_ci(em["per_question_em"], n_boot=n_boot, seed=seed, scale=100.0),
        "f1": em["f1"], "f1_ci95": bootstrap_ci(em["per_question_f1"], n_boot=n_boot, seed=seed, scale=100.0),
        "cons": cons["cons"], "cons_count": cons["count"],
        "cons_original_formula": cons_orig["cons_original"], "cons_original_count": cons_orig["count"],
        "co2": co2["co2"], "co2_count": co2["count"],
        "cons_robust": cons_true["cons"], "cons_robust_count": cons_true["count"],
        "co2_robust": co2_true["co2"], "co2_robust_count": co2_true["count"],
        "co2_original_formula": co2_orig["co2_original"], "co2_original_count": co2_orig["count"],
        "late_agreement": late["late_agreement"], "late_agreement_count": late["count"],
        "avg_rounds": average_rounds(records),
        "stop_round_distribution": {str(k): v for k, v in sorted(stop_dist.items())},
        "support": supp, "verification": ver,
        "diagnostics": dict(diag),
        "avg_prompt_tokens": tokens["prompt"] / n if n else 0.0,
        "avg_completion_tokens": tokens["completion"] / n if n else 0.0,
        "avg_calls": tokens["calls"] / n if n else 0.0,
    }


def paired_block(legacy_records, x_records):
    xs, ys, _ = paired_em_vectors(legacy_records, x_records)
    mc = paired_mcnemar(xs, ys)
    return {"n_paired": mc["n"], "legacy_only_correct": mc["b_a_only"], "x_only_correct": mc["c_b_only"], "p_value": mc["p_value"]}


# ---------------------------------------------------------------------------
# Markdown tables
# ---------------------------------------------------------------------------

def render_tables(results, title, n_label):
    """``results[dataset] = {"legacy": metrics, "x": metrics, "pool": ..., "paired": ..., "invariants": ...}``"""
    out = ["## {} ({})".format(title, n_label), ""]
    datasets = list(results.keys())

    # Table 3 -- interface layer
    rows = []
    for d in datasets:
        for key in ("legacy", "x"):
            m = results[d][key]
            dg = m["diagnostics"]
            rows.append([d, m["system"], m["n"], m["n_completed"], m["n_failed"], dg.get("scored_despite_failed_call", 0),
                         dg.get("binding_ok", 0), dg.get("evidence_replies", 0),
                         dg.get("delivered_set_ok", 0) if dg.get("intent_replies") else "-",
                         dg.get("delivered_exact_ok", 0) if dg.get("intent_replies") else "-",
                         dg.get("with_dropped", 0), dg.get("with_out_of_range", 0), dg.get("not_found", 0),
                         dg.get("parser_order_only", 0), dg.get("parser_disagreement", 0) - dg.get("parser_order_only", 0),
                         dg.get("judge_replies", 0), dg.get("judge_yes_robust", 0), dg.get("judge_agreement_missed_by_legacy", 0),
                         dg.get("judge_false_agreement_legacy", 0), dg.get("stopped_at_agreement_round", 0),
                         dg.get("judge_unparsed", 0)])
    out += ["### Table 3. Failure-aware interface layer (C1-C4, C7-C8)", "",
            md_table(["Dataset", "System", "Questions", "Completed", "Failed", "Scored despite failed call (C8)", "Summariser binding OK (C1)",
                      "Evidence replies", "Delivered set = intended (C3)", "Delivered list = intended, in order (C3)",
                      "w/ dropped (>k)", "w/ out-of-range", "No Found", "Released parser: order differs (C3)",
                      "Released parser: set differs (C3)", "Judge replies", "Replies stating agreement",
                      "Agreements missed by released parser (C4)", "False agreements of released parser (C4)",
                      "Debates stopped at the agreement round", "Unparsed verdicts"], rows), ""]

    # Table 4 -- Cons / Co2
    rows = []
    for d in datasets:
        for key in ("legacy", "x"):
            m = results[d][key]
            rows.append([d, m["system"], m["n_completed"], pct(m["cons_robust"]), pct(m["cons"]), pct(m["cons_original_formula"]),
                         pct(m["co2_robust"]), pct(m["co2"]), pct(m["co2_original_formula"]), pct(m["late_agreement"]),
                         num(m["avg_rounds"]), json.dumps(m["stop_round_distribution"])])
    out += ["### Table 4. Debate statistics `Cons` (consistency ratio) and `Co2` (consistent and correct ratio; correct = exact match) (C5, C6)", "",
            md_table(["Dataset", "System", "Completed", "Cons: judge replies read robustly", "Cons: pipeline verdicts",
                      "Cons: release formula", "Co2: judge replies read robustly", "Co2: pipeline verdicts", "Co2: release formula",
                      "Agreement after round 1", "Avg rounds", "Stop-round distribution"], rows), ""]

    # Table 5 -- pool
    rows = []
    for d in datasets:
        p = results[d]["pool"]
        cost = p.get("cost") or {}
        cov = p.get("coverage_pct") or {}
        rows.append([d, num(p["passages"]["original"], 2), num(p["passages"]["specified"], 2), num(p["passages"]["x"], 2),
                     p["duplicates_removed"], p["questions_with_duplicates"],
                     num(p["listing_tokens"]["original"], 0), num(p["listing_tokens"]["x"], 0),
                     num(p["listing_tokens"]["reduction_pct"], 1),
                     num(cov["original"], 1) if cov else "n/a", num(cov["x"], 1) if cov else "n/a",
                     num(cost["original_build_ms_per_q"], 4) if cost else "-", num(cost["x_build_ms_per_q"], 3) if cost else "-",
                     num(cost["x_peak_memory_kb"], 0) if cost else "-"])
    out += ["### Table 5. Evidence governor: pool size, de-duplication, coverage, cost (tokenizer: {})".format(results[datasets[0]]["pool"]["tokenizer"]), "",
            md_table(["Dataset", "Passages (10+5)", "Passages (5+5)", "Passages (MADKE-X)", "Duplicates removed", "Questions affected",
                      "Listing tokens (10+5)", "Listing tokens (MADKE-X)", "Reduction %", "Coverage % (10+5)", "Coverage % (MADKE-X)",
                      "Build ms/q (slice)", "Build ms/q (MADKE-X)", "Peak memory KB (MADKE-X)"], rows), ""]

    # Table 6 -- ordering
    rows = []
    for d in datasets:
        p = results[d]["pool"]
        o = p["ordering"]
        if o is None:
            rows.append([d, "n/a (verdict labels)", "-", "-", "-", "-", "-"])
            continue
        for name, lab in (("original", "original (wiki 10, then Google 5)"), ("interleaved", "interleaved"), ("bm25", "BM25 only"),
                          ("rrf", "RRF (rho + beta)"), ("random", "random (expected)")):
            m = o[name]
            rows.append([d, lab, p["n_span"], pct(m["cov@1"]), pct(m["cov@3"]), pct(m["cov@5"]), num(m["mrr"], 3)])
        pc = p["paired_cov3"]
        rows.append([d, "paired cov@3, RRF vs original: +{} pts {} (RRF only {}, original only {}, McNemar p {})".format(
            num(pc["gain_pts"], 1), ci(pc["ci95"]), pc["rrf_only"], pc["original_only"], "{:.2g}".format(pc["p_value"])), "", "", "", "", ""])
    out += ["### Table 6. Question-only BM25 + RRF ordering quality (gold answer present, as whole words, in the first k passages)", "",
            md_table(["Dataset", "Ordering", "n", "cov@1", "cov@3", "cov@5", "MRR"], rows), ""]

    # Table 7 -- trace / support
    rows = []
    for d in datasets:
        for key in ("legacy", "x"):
            m = results[d][key]
            s = m["support"]
            dist = s["distribution"]
            rows.append([d, m["system"], s["n_completed"], s["citations_parsed"], s["citations_valid"], dist.get("supported", 0),
                         dist.get("cited_but_answer_not_found", 0), dist.get("no_citation", 0), dist.get("empty_answer", 0),
                         dist.get("not_applicable", 0), pct(s["support_rate"])])
    out += ["### Table 7. Decision trace: `{E#}` citations and lexical support status (Algorithm 3)", "",
            md_table(["Dataset", "System", "Completed", "Citations parsed", "Citations valid (pool ∩ selected)", "supported",
                      "cited_but_answer_not_found", "no_citation", "empty_answer", "not_applicable", "Support rate"], rows), ""]

    # Table 8 -- verification
    rows = []
    for d in datasets:
        m = results[d]["x"]
        v = m["verification"]
        rows.append([d, v["n_applicable"], v["attempted"], v["adopted"], v["abstained"], v["attempted"] - v["adopted"] - v["abstained"] if v["attempted"] else 0,
                     pct(v["initial_support_rate"]), pct(v["final_support_rate"]), num(m["avg_calls"], 2)])
    out += ["### Table 8. Bounded verification (Algorithm 4, MADKE-X)", "",
            md_table(["Dataset", "Applicable", "Verification calls", "Adopted (s' supported)", "Abstained [Unsupported]", "Rejected (not supported)",
                      "Support rate before", "Support rate after", "Avg LLM calls / question"], rows), ""]

    # Table 9 -- paired comparison
    rows = []
    for d in datasets:
        l, x, pr = results[d]["legacy"], results[d]["x"], results[d]["paired"]
        rows.append([d, "{} {}".format(num(l["exact"], 1), ci(l["exact_ci95"])), "{} {}".format(num(x["exact"], 1), ci(x["exact_ci95"])),
                     "{} {}".format(num(l["f1"], 1), ci(l["f1_ci95"])), "{} {}".format(num(x["f1"], 1), ci(x["f1_ci95"])),
                     pr["n_paired"], pr["legacy_only_correct"], pr["x_only_correct"], pval(pr["p_value"]),
                     num(l["avg_prompt_tokens"] + l["avg_completion_tokens"], 0), num(x["avg_prompt_tokens"] + x["avg_completion_tokens"], 0)])
    out += ["### Table 9. Paired comparison MADKE (re-run) vs MADKE-X: EM/F1 with 95% bootstrap CIs and exact McNemar test (EM)", "",
            md_table(["Dataset", "EM legacy [CI]", "EM MADKE-X [CI]", "F1 legacy [CI]", "F1 MADKE-X [CI]", "Paired n", "Legacy-only correct (b)",
                      "X-only correct (c)", "McNemar p", "Tokens/q legacy", "Tokens/q MADKE-X"], rows), ""]

    # Invariants
    rows = []
    for d in datasets:
        inv = results[d]["invariants"]
        rows.append([d, inv["n_records"], inv["n_checks"], inv["n_passed"], pct(inv["pass_rate"]), "PASS" if inv["all_passed"] else "FAIL"])
    out += ["### Invariant verification (MADKE-X records)", "", md_table(["Dataset", "Records", "Checks", "Passed", "Pass rate", "Result"], rows), ""]
    return "\n".join(out)


def render_table10(results, tag):
    datasets = list(results.keys())
    rows = []
    for d in datasets:
        for key in ("legacy", "x"):
            m = results[d][key]
            pr = results[d]["paired"]
            rows.append([d, m["system"], m["n"], m["n_failed"], m["diagnostics"].get("scored_despite_failed_call", 0), "{} {}".format(num(m["f1"], 1), ci(m["f1_ci95"])),
                         "{} {}".format(num(m["exact"], 1), ci(m["exact_ci95"])), pct(m["cons"]), pct(m["co2"]),
                         pct(m["support"]["support_rate"]) if m["support"]["n_applicable"] else "n/a",
                         pval(pr["p_value"]) if key == "x" else "-", num(m["avg_rounds"])])
    out = ["## Table 10. Live results ({})".format(tag), "",
           md_table(["Dataset", "System", "Questions", "Failed", "Scored despite failed call", "F1 [CI95]", "EM [CI95]", "Cons", "Co2", "Support rate", "McNemar p (EM vs legacy)", "Avg rounds"], rows), ""]
    rows = []
    for d in datasets:
        inv = results[d]["invariants"]
        rows.append([d, inv["n_records"], inv["n_checks"], inv["n_passed"], pct(inv["pass_rate"]), "PASS" if inv["all_passed"] else "FAIL"])
    out += ["### Invariant verification (MADKE-X records)", "", md_table(["Dataset", "Records", "Checks", "Passed", "Pass rate", "Result"], rows), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main flows
# ---------------------------------------------------------------------------

def dataset_results(args, dataset_name, data, indices, out_dir, checkpoint=False, bm25_index=None):
    prompt = select_prompt(dataset_name)
    cfgs = {}
    if args.configs in ("both", "legacy"):
        cfgs["legacy"] = legacy_config(dataset_name, agent_num=args.agent_num, max_round=args.max_round)
    if args.configs in ("both", "madke_x"):
        cfgs["x"] = madke_x_config(dataset_name, agent_num=args.agent_num, max_round=args.max_round,
                                   wiki_k=args.wiki_k, google_k=args.google_k, cap=args.cap, rrf_k=args.rrf_k)
    records = {}
    for key, cfg in cfgs.items():
        agents = make_agents(args, dataset_name)
        ckpt = os.path.join(out_dir, dataset_name, cfg.name + ".jsonl") if checkpoint else None
        t0 = time.time()
        recs = run_records(data, indices, cfg, prompt, agents, checkpoint=ckpt, concurrency=args.concurrency,
                           retry_failed=args.retry_failed, label="{}/{}".format(dataset_name, cfg.name),
                           max_consecutive_failures=(args.max_consecutive_failures if args.mode == "live" else 0),
                           bm25_index=bm25_index)
        log("{} {}: {} questions in {:.1f}s ({} failed)".format(dataset_name, cfg.name, len(recs), time.time() - t0,
                                                              sum(1 for r in recs if r["status"] == "failed")))
        write_json_data(os.path.join(out_dir, dataset_name, cfg.name + ".json"), recs, verbose=False)
        records[key] = (cfg, recs)
    return records


def summarise(args, dataset_name, data, indices, records, pool_stats, out_dir, subset=None, bm25_index=None):
    res = {}
    sub = (lambda recs: [r for r in recs if r["index"] in subset]) if subset is not None else (lambda recs: recs)
    for key, (cfg, recs) in records.items():
        res[key] = config_metrics(sub(recs), cfg, data, args.n_boot, args.seed)
    if "legacy" in records and "x" in records:
        res["paired"] = paired_block(sub(records["legacy"][1]), sub(records["x"][1]))
    else:
        res["paired"] = {"n_paired": 0, "legacy_only_correct": 0, "x_only_correct": 0, "p_value": 1.0}
    res["pool"] = pool_stats
    x_cfg, x_recs = records.get("x", records.get("legacy"))
    inv_indices = [i for i in indices if subset is None or i in subset]
    res["invariants"] = check_invariants(sub(x_recs), x_cfg, data, inv_indices, bm25_index=bm25_index)
    return res


def main(argv=None):
    parser = argparse.ArgumentParser(description="MADKE-X unified experiment runner")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--configs", type=str, default="both", choices=["both", "legacy", "madke_x"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=500)
    parser.add_argument("--subset", type=int, default=160, help="Also report the first N questions (offline).")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default=None, help="Sub-folder for live runs (default: model slug).")
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--skip_checksums", action="store_true")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--agent_num", type=int, default=2)
    parser.add_argument("--max_round", type=int, default=3)
    parser.add_argument("--wiki_k", type=int, default=5)
    parser.add_argument("--google_k", type=int, default=5)
    parser.add_argument("--cap", type=int, default=3)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel questions (live mode).")
    parser.add_argument("--retry_failed", action="store_true", help="Re-run checkpointed questions that failed.")
    parser.add_argument("--max_consecutive_failures", type=int, default=5,
                        help="Live mode: stop the run after this many consecutive questions with failed calls (0 = never).")
    parser.add_argument("--skip_preflight", action="store_true", help="Live mode: do not test the endpoint before running.")
    parser.add_argument("--fail_demo", type=int, default=100,
                        help="Offline: questions per dataset for the injected-failure demonstration (0 disables).")
    parser.add_argument("--fail_demo_rate", type=float, default=0.02, help="Per-call failure rate of that demonstration.")
    parser.add_argument("--no_stress", dest="stress", action="store_false",
                        help="Offline: skip the verification stress test (careless scripted summariser).")
    add_agent_backend_args(parser)
    parser.set_defaults(mode="offline")
    args = parser.parse_args(argv)

    if args.mode == "live" and not args.model and not (args.endpoint and args.deployment_id):
        parser.error("--mode live requires --model (and --base_url) or --endpoint/--deployment_id")

    if args.mode == "live" and not args.skip_preflight:
        ok, msg = preflight(args)
        log(("PREFLIGHT OK: " if ok else "PREFLIGHT FAILED: ") + msg)
        if not ok:
            log("No live run was started. Start the model server first (see MADKE_X_Kaggle_Colab.ipynb).")
            return 2

    try:
        return _run(args)
    except LiveRunAborted as ex:
        log("LIVE RUN ABORTED: {}".format(ex))
        return 3


def _run(args):
    t_start = time.time()
    tag = args.run_tag or (slugify(args.model) if args.mode == "live" else "offline")
    out_dir = args.out_dir or (os.path.join("results", "offline") if args.mode == "offline" else os.path.join("results", "live", tag))
    os.makedirs(out_dir, exist_ok=True)

    # 1. integrity of the read-only data
    checks = verify_data_checksums(args.data_root, args.datasets, strict=not args.skip_checksums)
    write_json_data(os.path.join(out_dir, "checksums.json"), checks, verbose=False)
    for name, c in checks.items():
        log("SHA-256 {:<15} {}".format(name, "OK" if c["ok"] else "MISMATCH"))

    all_results, subset_results, fail_demo, stress = {}, {}, {}, {}
    for dataset_name in args.datasets:
        data = load_dataset(dataset_name, args.data_root, verify=False)
        end = min(args.end_index, len(data))
        indices = list(range(args.start_index, end))
        log("=== {} : {} questions [{}:{}] ===".format(dataset_name, len(indices), args.start_index, end))

        # reference statistics over the whole file, independent of the evaluated index range
        bm25_index = build_bm25_index(data, args.wiki_k, args.google_k)
        pool_stats = pool_analysis(dataset_name, data, indices, args.wiki_k, args.google_k, args.rrf_k,
                                   n_boot=args.n_boot, seed=args.seed, timing=(args.mode == "offline"),
                                   bm25_index=bm25_index)
        write_json_data(os.path.join(out_dir, dataset_name, "pool_analysis.json"), pool_stats, verbose=False)

        records = dataset_results(args, dataset_name, data, indices, out_dir, checkpoint=(args.mode == "live"),
                                  bm25_index=bm25_index)
        all_results[dataset_name] = summarise(args, dataset_name, data, indices, records, pool_stats, out_dir,
                                              bm25_index=bm25_index)
        write_json_data(os.path.join(out_dir, dataset_name, "invariants.json"), all_results[dataset_name]["invariants"], verbose=False)
        inv = all_results[dataset_name]["invariants"]
        log("{} invariants: {}/{} passed -> {}".format(dataset_name, inv["n_passed"], inv["n_checks"], "PASS" if inv["all_passed"] else "FAIL"))

        if args.mode == "offline" and args.subset and args.subset < len(indices):
            sub_idx = set(indices[:args.subset])
            sub_pool = pool_analysis(dataset_name, data, indices[:args.subset], args.wiki_k, args.google_k, args.rrf_k,
                                     n_boot=args.n_boot, seed=args.seed, timing=False, bm25_index=bm25_index)
            subset_results[dataset_name] = summarise(args, dataset_name, data, indices, records, sub_pool, out_dir,
                                                     subset=sub_idx, bm25_index=bm25_index)

        if args.mode == "offline" and args.fail_demo:
            demo_idx = indices[:args.fail_demo]
            demo = {}
            fr = args.fail_demo_rate
            for key, cfg in (("legacy", legacy_config(dataset_name)), ("x", madke_x_config(dataset_name))):
                agents = [ScriptedAgent("Agent_{}".format(i), seed=args.seed, fail_rate=fr, dataset_name=dataset_name) for i in range(args.agent_num)]
                judge = ScriptedAgent("Judge", seed=args.seed, fail_rate=fr, dataset_name=dataset_name)
                summ = ScriptedAgent("Summarizer", seed=args.seed, fail_rate=fr, dataset_name=dataset_name)
                recs = run_records(data, demo_idx, cfg, select_prompt(dataset_name), (agents, judge, summ),
                                   bm25_index=bm25_index)
                m = config_metrics(recs, cfg, data, 200, args.seed)
                demo[key] = {"system": m["system"], "n": m["n"], "n_failed": m["n_failed"], "n_completed": m["n_completed"],
                             "scored_despite_failed_call": m["diagnostics"].get("scored_despite_failed_call", 0),
                             "exact_over_completed": m["exact"], "fail_rate": fr}
            fail_demo[dataset_name] = demo

        if args.mode == "offline" and args.stress and task_type(dataset_name) == "extractive":
            stress[dataset_name] = verification_stress(args, dataset_name, data, indices, bm25_index=bm25_index)
            st = stress[dataset_name]
            log("{} verification stress: support {:.1f}% -> {:.1f}% ({} calls, {} adopted, {} supported answers touched)".format(
                dataset_name, 100 * st["initial_support_rate"], 100 * st["final_support_rate"], st["verification_calls"],
                st["adopted"], st["supported_answer_touched"]))

    # 3. write tables
    tables = {"mode": args.mode, "code_version": CODE_VERSION, "args": vars(args),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "processor": platform.processor(), "tokenizer": tokenizer_name()},
              "elapsed_sec": time.time() - t_start, "full": all_results}
    if args.mode == "offline":
        tables["subset_{}".format(args.subset)] = subset_results
        tables["failure_injection_demo"] = fail_demo
        tables["verification_stress"] = stress
        tables["interface_formats"] = interface_format_tables(cap=args.cap)
        md = ["# MADKE-X offline verification tables", "",
              "Generated by `python run_experiments.py --mode offline` with deterministic scripted agents (seed {}). "
              "EM/F1 here characterise the *pipeline*, not an LLM; see Table 10 for live results.".format(args.seed), "",
              render_tables(all_results, "Full evaluation", "{} questions per dataset".format(len(indices)))]
        if subset_results:
            md += [render_tables(subset_results, "Subset", "first {} questions per dataset".format(args.subset))]
        if fail_demo:
            rows = []
            for d, demo in fail_demo.items():
                for key in ("legacy", "x"):
                    m = demo[key]
                    rows.append([d, m["system"], m["n"], m["n_completed"], m["n_failed"], m["scored_despite_failed_call"], num(m["exact_over_completed"], 1)])
            md += ["## Failure injection demonstration (C7/C8): scripted agents with a {:.0%} per-call failure rate".format(args.fail_demo_rate), "",
                   md_table(["Dataset", "System", "Questions", "Completed", "Marked failed (C8)", "Scored despite failed call(s)", "EM over completed"], rows), ""]
        if stress:
            rows = [[d, st["applicable"], st["not_applicable"], pct(st["initial_support_rate"]), st["verification_calls"], st["adopted"],
                     st["adopted_new_answer"], st["abstained"], st["rejected"], pct(st["final_support_rate"]), st["supported_answer_touched"]]
                    for d, st in stress.items()]
            md += ["## Verification stress test: careless scripted summariser (always cites the first listed passage)", "",
                   md_table(["Dataset", "Applicable", "Verdict answers (n/a)", "Supported before", "Verification calls", "Adopted",
                             "of which new answer", "Abstained", "Rejected", "Supported after", "Supported answers touched"], rows), ""]
        fmt = tables["interface_formats"]
        md += ["## Parser behaviour on canonical reply formats", "",
               md_table(["Reply format", "Reply", "Released parser delivers", "MADKE-X delivers", "MADKE-X records"],
                        [[e["format"], "`{}`".format(e["reply"]), e["released_delivered"] or "nothing",
                          ("not found" if e["x_not_found"] else e["x_delivered"]),
                          "dropped {} / ignored {} / out-of-range {}".format(e["x_dropped"], e["x_ignored"], e["x_out_of_range"])]
                         for e in fmt["evidence"]]), "",
               md_table(["Verdict format", "Judge reply", "Intended", "Released parser", "MADKE-X parser"],
                        [[v["format"], "`{}`".format(v["reply"]), v["intended"], v["released"], v["x"]] for v in fmt["verdict"]]), ""]
        write_text(os.path.join(out_dir, "TABLES.md"), "\n".join(md))
        write_json_data(os.path.join(out_dir, "tables.json"), tables, verbose=False)
        all_pass = all(r["invariants"]["all_passed"] for r in all_results.values())
        log("Offline run finished in {:.1f}s. Invariants: {}. Tables -> {}".format(time.time() - t_start, "ALL PASSED" if all_pass else "FAILURES", os.path.join(out_dir, "TABLES.md")))
        return 0 if all_pass else 1

    md = ["# MADKE-X live results", "", "Model: `{}`  base_url: `{}`  generated by `run_experiments.py --mode live`.".format(args.model or args.deployment_id, args.base_url), "",
          render_table10(all_results, tag)]
    write_text(os.path.join(out_dir, "TABLE10.md"), "\n".join(md))
    write_json_data(os.path.join(out_dir, "table10.json"), tables, verbose=False)
    log("Live run finished in {:.1f}s. Table 10 -> {}".format(time.time() - t_start, os.path.join(out_dir, "TABLE10.md")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
