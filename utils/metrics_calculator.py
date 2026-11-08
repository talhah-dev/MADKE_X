"""Automatic metrics for MADKE / MADKE-X prediction records (no paid grading).

Implemented here (all computed in pure Python):

* EM / F1 (SQuAD definitions via :mod:`utils.evaluation_utils`)
* ``Cons`` and ``Co2`` -- repaired definitions (fixes C5 and C6)
* the *original* ``Cons`` formula (for the ablation table)
* support rate and support-status distribution (Algorithm 3 outputs)
* verification statistics (Algorithm 4 outputs)
* paired exact McNemar test, percentile bootstrap confidence intervals
* retrieval ordering quality: ``cov@k`` and MRR

A *record* is one question's output dict as produced by :func:`debate.run_debate`
(status, final_answer, trace.judge_verdicts, ...).  Legacy records produced by
the original ``madke.py`` are also accepted: their judge verdicts are recovered
from ``Round_k["Judge"]`` with the robust parser.
"""

import math
import random
import re

from utils.evaluation_utils import (
    compute_exact,
    compute_f1,
    extract_answer,
    extract_choice,
    normalize_text,
    parse_judge_verdict,
    parse_judge_verdict_legacy,
)

# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------

def is_completed(record):
    return record.get("status", "completed") == "completed"


def record_prediction(record):
    """Final predicted answer string of a record (``None`` for failed questions)."""
    if not is_completed(record):
        return None
    if record.get("final_answer") is not None:
        return record["final_answer"]
    answers = record.get("llm_answers", "")
    if isinstance(answers, list):
        return answers[-1] if answers else ""
    return answers or ""


def record_references(record):
    answer = record.get("answer", [])
    return answer if isinstance(answer, list) else [answer]


def _round_keys(record):
    keys = [k for k in record if re.fullmatch(r"Round_\d+", k)]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def judge_verdicts(record, legacy_parser=False):
    """Judge verdicts per round (``"yes"``/``"no"``/``"unparsed"``), round 1 first."""
    trace = record.get("trace") or {}
    if "judge_verdicts" in trace and not legacy_parser:
        return list(trace["judge_verdicts"])
    verdicts = []
    for key in _round_keys(record):
        judge_reply = record[key].get("Judge", "") if isinstance(record[key], dict) else ""
        verdicts.append(parse_judge_verdict_legacy(judge_reply) if legacy_parser else parse_judge_verdict(judge_reply))
    return verdicts


# ---------------------------------------------------------------------------
# EM / F1
# ---------------------------------------------------------------------------

def em_f1_for_prediction(prediction, references):
    refs = references or [""]
    em = max(compute_exact(ref, prediction) for ref in refs)
    f1 = max(compute_f1(ref, prediction) for ref in refs)
    return em, f1


def compute_em_f1(records):
    """EM/F1 over completed records.  Failed records are excluded, never scored as ``""``."""
    ems, f1s, idx = [], [], []
    for record in records:
        if not is_completed(record):
            continue
        em, f1 = em_f1_for_prediction(record_prediction(record) or "", record_references(record))
        ems.append(float(em))
        f1s.append(float(f1))
        idx.append(record.get("index"))
    n = len(ems)
    return {
        "exact": 100.0 * sum(ems) / n if n else 0.0,
        "f1": 100.0 * sum(f1s) / n if n else 0.0,
        "total": n,
        "n_failed": sum(1 for r in records if not is_completed(r)),
        "per_question_em": ems,
        "per_question_f1": f1s,
        "indices": idx,
    }


# ---------------------------------------------------------------------------
# Cons / Co2 (fix C5, C6)
# ---------------------------------------------------------------------------

def compute_cons(records, legacy_parser=False):
    """``Cons``: proportion of completed questions with a ``yes`` verdict in ANY round ``1..R_q``.

    Includes the final round ``R`` (the original implementation never counted a
    final-round agreement because it tested ``"[Yes]" in <dict>``).
    """
    completed = [r for r in records if is_completed(r)]
    hits = 0
    for r in completed:
        if any(v == "yes" for v in judge_verdicts(r, legacy_parser)):
            hits += 1
    return {"cons": hits / len(completed) if completed else 0.0, "count": hits, "n_completed": len(completed)}


def compute_co2(records, legacy_parser=False):
    """``Co2``: completed questions with ``v_1 != yes`` and ``v_{R_q} == yes`` (agreement reached in round >= 2)."""
    completed = [r for r in records if is_completed(r)]
    hits = 0
    for r in completed:
        v = judge_verdicts(r, legacy_parser)
        if len(v) >= 2 and v[0] != "yes" and v[-1] == "yes":
            hits += 1
    return {"co2": hits / len(completed) if completed else 0.0, "count": hits, "n_completed": len(completed)}


def compute_cons_original(records, max_round=3):
    """The ORIGINAL (buggy) ``Cons`` of ``evaluation.calculate_acc_c``.

    A question counted as consistent iff it stopped before the last round, or
    ``"[Yes]" in data["Round_<last>"]`` -- which tests dict *keys* and is
    therefore always ``False``: final-round agreements are never counted.
    Reproduced here for the C5 ablation table.
    """
    hits, n = 0, 0
    for r in records:
        if "stop_round" not in r:
            continue
        n += 1
        stop_round = r["stop_round"]
        if stop_round < max_round - 1:
            hits += 1
            continue
        if "[Yes]" in r.get("Round_" + str(stop_round), {}):  # dict membership -> keys
            hits += 1
    return {"cons_original": hits / n if n else 0.0, "count": hits, "n": n}


def average_rounds(records):
    completed = [r for r in records if is_completed(r)]
    if not completed:
        return 0.0
    return sum(len(judge_verdicts(r)) for r in completed) / len(completed)


# ---------------------------------------------------------------------------
# Support / verification statistics
# ---------------------------------------------------------------------------

def support_statistics(records):
    completed = [r for r in records if is_completed(r)]
    dist = {}
    applicable = 0
    supported = 0
    cited_total = cited_valid = 0
    for r in completed:
        trace = r.get("trace") or {}
        status = trace.get("support_status", "not_applicable")
        dist[status] = dist.get(status, 0) + 1
        if status != "not_applicable":
            applicable += 1
            if status == "supported":
                supported += 1
        cited_total += len(trace.get("cited_ids", []) or [])
        cited_valid += len(trace.get("cited_valid_ids", []) or [])
    return {
        "support_rate": supported / applicable if applicable else 0.0,
        "n_supported": supported,
        "n_applicable": applicable,
        "n_completed": len(completed),
        "distribution": dist,
        "citations_parsed": cited_total,
        "citations_valid": cited_valid,
    }


def verification_statistics(records):
    attempted = adopted = abstained = 0
    initial_supported = final_supported = applicable = 0
    for r in records:
        if not is_completed(r):
            continue
        trace = r.get("trace") or {}
        ver = trace.get("verification") or {}
        if trace.get("support_status", "not_applicable") != "not_applicable":
            applicable += 1
            if trace.get("initial_support_status", trace.get("support_status")) == "supported":
                initial_supported += 1
            if trace.get("support_status") == "supported":
                final_supported += 1
        if ver.get("attempted"):
            attempted += 1
            if ver.get("adopted"):
                adopted += 1
            if ver.get("abstained"):
                abstained += 1
    return {
        "attempted": attempted,
        "adopted": adopted,
        "abstained": abstained,
        "initial_supported": initial_supported,
        "final_supported": final_supported,
        "n_applicable": applicable,
        "initial_support_rate": initial_supported / applicable if applicable else 0.0,
        "final_support_rate": final_supported / applicable if applicable else 0.0,
    }


# ---------------------------------------------------------------------------
# Statistics: exact McNemar and bootstrap CI
# ---------------------------------------------------------------------------

def mcnemar_exact(b, c):
    """Two-sided exact (binomial) McNemar p-value for discordant counts ``b``, ``c``."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_mcnemar(correct_a, correct_b):
    """Paired McNemar between two aligned 0/1 correctness lists (A vs B)."""
    assert len(correct_a) == len(correct_b)
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)   # A right, B wrong
    c = sum(1 for x, y in zip(correct_a, correct_b) if not x and y)   # A wrong, B right
    return {"n": len(correct_a), "b_a_only": b, "c_b_only": c, "p_value": mcnemar_exact(b, c)}


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05, scale=1.0):
    """Percentile bootstrap CI of the mean (deterministic given ``seed``)."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int(math.floor((alpha / 2.0) * (n_boot - 1)))]
    hi = means[int(math.ceil((1.0 - alpha / 2.0) * (n_boot - 1)))]
    return (scale * lo, scale * hi)


def paired_em_vectors(records_a, records_b):
    """Align two record lists on ``index`` and return paired EM 0/1 vectors over questions completed in both."""
    by_idx_b = {r.get("index"): r for r in records_b}
    xs, ys, idx = [], [], []
    for ra in records_a:
        rb = by_idx_b.get(ra.get("index"))
        if rb is None or not (is_completed(ra) and is_completed(rb)):
            continue
        em_a, _ = em_f1_for_prediction(record_prediction(ra) or "", record_references(ra))
        em_b, _ = em_f1_for_prediction(record_prediction(rb) or "", record_references(rb))
        xs.append(int(em_a))
        ys.append(int(em_b))
        idx.append(ra.get("index"))
    return xs, ys, idx


# ---------------------------------------------------------------------------
# Retrieval ordering quality (cov@k, MRR)
# ---------------------------------------------------------------------------

def passage_contains_any(passage_text, answers):
    norm_passage = normalize_text(passage_text)
    for a in answers:
        na = normalize_text(a)
        if na and na in norm_passage:
            return True
    return False


def first_hit_rank(ordered_texts, answers):
    """1-based rank of the first passage lexically containing a gold answer (``None`` if absent)."""
    for rank, text in enumerate(ordered_texts, start=1):
        if passage_contains_any(text, answers):
            return rank
    return None


def coverage_and_mrr(hit_ranks, ks=(1, 3, 5)):
    n = len(hit_ranks)
    out = {"n": n}
    for k in ks:
        out["cov@{}".format(k)] = (sum(1 for r in hit_ranks if r is not None and r <= k) / n) if n else 0.0
    out["mrr"] = (sum((1.0 / r) for r in hit_ranks if r is not None) / n) if n else 0.0
    out["any_hit"] = (sum(1 for r in hit_ranks if r is not None) / n) if n else 0.0
    return out


# ---------------------------------------------------------------------------
# Backward-compatible helpers from the original module
# ---------------------------------------------------------------------------

def extract_answer_mc(response):
    pattern = re.compile(r"\((.*?)\)")
    response = (response or "").replace("\n", "").replace('"', "")
    return re.findall(pattern, response)


__all__ = [
    "extract_answer", "extract_answer_mc", "extract_choice",
    "compute_em_f1", "compute_cons", "compute_co2", "compute_cons_original", "average_rounds",
    "support_statistics", "verification_statistics",
    "mcnemar_exact", "paired_mcnemar", "bootstrap_ci", "paired_em_vectors",
    "first_hit_rank", "coverage_and_mrr", "judge_verdicts", "record_prediction", "is_completed",
]
