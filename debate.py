"""MADKE / MADKE-X debate loop (Algorithms 1-4) as a reusable function.

``run_debate`` executes one question: evidence governor -> R rounds of
(evidence selection, debater reply) x N agents -> judge -> early stop ->
summariser (optionally citing ``{E#}``) -> lexical support check -> optional
bounded verification.  Every question returns a record that is backward
compatible with the original output (``question``, ``answer``, ``llm_answers``,
``stop_round``, ``Round_k``, ``Summary``) and adds ``status``, ``final_answer``,
``error`` and the audit ``trace`` (Section 4.3).

``DebateConfig(legacy=True)`` reproduces the *unpatched* MADKE behaviour
(10+5 pool, no de-duplication / re-ranking, reversed evidence parser, literal
``[Yes]`` judge match, swapped summariser arguments and error strings scored
as answers) so that the "MADKE (re-run)" baseline can be executed on exactly
the same questions as MADKE-X.
"""

import time
from dataclasses import asdict, dataclass

import prompts.fever_prompt as fever_prompt
import prompts.hotpotqa_prompt as hotpot_prompt
import prompts.medmc_prompt as medmc_prompt
import prompts.nq_prompt as nq_prompt
import prompts.triviaqa_prompt as tqa_prompt
import prompts.wikimultihopqa_prompt as wikimultihopqa_prompt
from models.openai_compat_agent import build_agent
from utils.calculate_tokens import count_tokens
from utils.data_utils import build_pool, legacy_pool, pool_manifest, select_evidence_lists, task_type
from utils.evaluation_utils import (
    extract_answer,
    extract_final_answer,
    normalize_text,
    parse_citations,
    support_check,
)
from utils.prompt_utils import (
    call_agent,
    format_citable_evidence,
    format_debate,
    judge_consistency,
    self_choose_evidence,
    summary_final_answer,
    verify_answer,
)


def select_prompt(dataset_name):
    if dataset_name == "triviaqa":
        return tqa_prompt
    if dataset_name == "hotpotqa":
        return hotpot_prompt
    if dataset_name == "nq":
        return nq_prompt
    if dataset_name in ("fever", "feverous"):
        return fever_prompt
    if dataset_name == "wikimultihopqa":
        return wikimultihopqa_prompt
    if dataset_name == "mmlu_med":
        return medmc_prompt
    raise ValueError("Prompt type not exist for dataset '{}'".format(dataset_name))


@dataclass
class DebateConfig:
    dataset_name: str = "triviaqa"
    agent_num: int = 2
    max_round: int = 3
    wiki_k: int = 5
    google_k: int = 5
    cap: int = 3
    evidence_type: str = "all_evidence"
    use_self_choose: bool = True
    dedup: bool = True
    rrf: bool = True
    rrf_k: int = 60
    cite: bool = False
    verify: bool = False
    legacy: bool = False
    name: str = "madke_x"

    @property
    def task_type(self):
        return task_type(self.dataset_name)

    def to_dict(self):
        d = asdict(self)
        d["task_type"] = self.task_type
        return d

    @classmethod
    def from_args(cls, args):
        return cls(dataset_name=args.dataset_name, agent_num=args.agent_num, max_round=args.max_round,
                   wiki_k=args.wiki_k, google_k=args.google_k, cap=getattr(args, "cap", 3),
                   evidence_type=args.evidence_type, use_self_choose=args.use_self_choose,
                   dedup=not getattr(args, "no_dedup", False), rrf=not getattr(args, "no_rrf", False),
                   rrf_k=getattr(args, "rrf_k", 60), cite=getattr(args, "cite", False),
                   verify=getattr(args, "verify", False), legacy=getattr(args, "legacy", False),
                   name="madke_legacy" if getattr(args, "legacy", False) else "madke_x")


def legacy_config(dataset_name, **overrides):
    """Unpatched MADKE: 10+5 pool, no dedup/RRF, buggy parsers, swapped summariser arguments."""
    cfg = dict(dataset_name=dataset_name, wiki_k=10, google_k=5, dedup=False, rrf=False,
               cite=False, verify=False, legacy=True, name="madke_legacy")
    cfg.update(overrides)
    return DebateConfig(**cfg)


def madke_x_config(dataset_name, **overrides):
    """MADKE-X: 5+5 de-duplicated RRF pool, robust parsers, {E#} citations, bounded verification."""
    cfg = dict(dataset_name=dataset_name, wiki_k=5, google_k=5, dedup=True, rrf=True,
               cite=True, verify=True, legacy=False, name="madke_x")
    cfg.update(overrides)
    return DebateConfig(**cfg)


def build_agents(args, agent_num, mode=None, dataset_name=None):
    """Create ``(debaters, judge, summarizer)`` for ``--mode offline`` (scripted) or ``live``."""
    if dataset_name is not None and hasattr(args, "dataset_name"):
        args.dataset_name = dataset_name
    debaters = [build_agent(args, "Agent_{}".format(i), mode) for i in range(agent_num)]
    judge = build_agent(args, "Judge", mode)
    summarizer = build_agent(args, "Summarizer", mode)
    return debaters, judge, summarizer


_EMPTY_PARSE = {"selected_ids": [], "dropped_ids": [], "ignored_ids": [], "out_of_range_ids": [], "not_found": False}


def run_debate(data, debaters, judge, summarizer, prompt, cfg, index=None):
    """Run the full debate for one question and return its record (see module docstring)."""
    t0 = time.time()
    question = data["question"]
    answer = data["answer"]

    # ------------------------------------------------------------------ pool
    wiki, google, w, g = select_evidence_lists(data, cfg.evidence_type, cfg.wiki_k, cfg.google_k)
    if cfg.evidence_type == "no_evidence":
        pool, pool_meta = [], {"candidates": 0, "duplicates_removed": 0}
    elif cfg.legacy:
        pool, pool_meta = legacy_pool(wiki, google, w, g)
    else:
        pool, pool_meta = build_pool(question, wiki, google, w=w, g=g, dedup=cfg.dedup, rrf=cfg.rrf, rrf_k=cfg.rrf_k)

    record = {"index": index, "question": question, "answer": answer, "llm_answers": [], "final_answer": None,
              "Summary": "", "stop_round": None, "status": "completed", "error": None}
    trace = {
        "config": cfg.to_dict(),
        "pool_size": len(pool),
        "pool_meta": pool_meta,
        "pool_manifest": pool_manifest(pool),
        "rounds": [],
        "judge_verdicts": [],
        "union_selected_ids": [],
        "cited_ids": [], "cited_valid_ids": [], "cited_invalid_ids": [], "cite_none_flag": False,
        "initial_support_status": None, "support_status": None,
        "verification": {"attempted": False},
        "summariser_binding_ok": None,
        "calls": {"evidence": 0, "debater": 0, "judge": 0, "summariser": 0, "verifier": 0},
        "tokens": {"prompt": 0, "completion": 0},
        "server_usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "call_failures": 0,   # failed calls that were carried on (legacy mode only; MADKE-X stops at the first)
        "failure": None,
    }

    def account(stage, user_prompt, reply, usage=None):
        trace["calls"][stage] += 1
        trace["tokens"]["prompt"] += count_tokens(user_prompt or "")
        trace["tokens"]["completion"] += count_tokens(reply or "")
        if usage:
            trace["server_usage"]["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            trace["server_usage"]["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)

    def fail(stage, res, agent_name=None, round_num=None):
        record["status"] = "failed"
        record["error"] = {"stage": stage, "agent": agent_name, "round": None if round_num is None else round_num + 1,
                           "error": res.get("error"), "attempts": res.get("attempts")}
        trace["failure"] = record["error"]

    history = {agent.agent_name: [] for agent in debaters}
    union_selected = []
    failed = False

    for round_num in range(cfg.max_round):
        record["stop_round"] = round_num
        round_key = "Round_{}".format(round_num)
        record[round_key] = {}
        round_trace = {"round": round_num + 1, "agents": {}, "judge_verdict": None,
                       "judge_verdict_robust": None, "judge_verdict_legacy": None}
        replies = []

        for num, agent in enumerate(debaters):
            prev = history[agent.agent_name][-1] if history[agent.agent_name] else ""

            # --- evidence selection (Algorithm 1)
            if cfg.evidence_type == "no_evidence" or not pool:
                sel = {"ok": True, "reply": "", "chosen_text": "", "ids": [], "parse": dict(_EMPTY_PARSE),
                       "attempts": 0, "error": None, "usage": {}, "prompt": None}
            else:
                sel = self_choose_evidence(agent, question, prev, pool, prompt, cfg.use_self_choose, round_num,
                                           cap=cfg.cap, legacy=cfg.legacy)
                account("evidence", sel.get("prompt"), sel.get("reply"), sel.get("usage"))
            if not sel["ok"]:
                if cfg.legacy:   # original behaviour: carry on with the error text
                    trace["call_failures"] += 1
                    sel.update(chosen_text="", ids=[], parse=dict(_EMPTY_PARSE), reply=sel.get("error") or "Error")
                else:
                    fail("evidence", sel, agent.agent_name, round_num)
                    failed = True
                    break
            for i in sel["ids"]:
                if i not in union_selected:
                    union_selected.append(i)

            # --- debater
            if round_num == 0:
                user_prompt = prompt.debater_first_round_prompt.format(question=question, evidences=sel["chosen_text"])
            else:
                own = "Here is your historical answer: " + prev
                others = "Answers from other Agents:\n"
                for name in history:
                    if name != agent.agent_name and history[name]:
                        others += "({}) {}".format(name, history[name][-1])
                user_prompt = prompt.debater_other_round_prompt.format(question=question, evidences=sel["chosen_text"],
                                                                       answer_from_other_agents=others,
                                                                       your_historical_answer=own)
            messages = [{"role": "system", "content": prompt.debater_system_prompt},
                        {"role": "user", "content": user_prompt}]
            res = call_agent(agent, messages)
            account("debater", user_prompt, res.text, res.usage)
            if not res.ok:
                if cfg.legacy:
                    trace["call_failures"] += 1
                    reply = res.error or "Error"
                else:
                    fail("debater", res.to_dict(), agent.agent_name, round_num)
                    failed = True
                    break
            else:
                reply = res.text
            history[agent.agent_name].append(reply)
            replies.append(reply)

            record[round_key]["User_{}".format(num)] = user_prompt
            record[round_key]["Choose_Evi_{}".format(num)] = sel["reply"]
            record[round_key]["Agent_{}".format(num)] = reply
            agent_trace = dict(sel["parse"] or _EMPTY_PARSE)
            agent_trace.update({"answer": extract_final_answer(reply), "attempts": res.attempts})
            round_trace["agents"][agent.agent_name] = agent_trace

        if failed:
            trace["rounds"].append(round_trace)
            break

        # --- judge (fix C4: last [Yes]/[No], exact round of agreement)
        debate_text = format_debate(replies)
        jres = judge_consistency(judge, question, debate_text, prompt, legacy=cfg.legacy)
        account("judge", jres.get("prompt"), jres.get("reply"), jres.get("usage"))
        if not jres["ok"]:
            if cfg.legacy:
                trace["call_failures"] += 1
                jres.update(verdict="no", agree=False, reply=jres.get("error") or "Error",
                            verdict_robust="unparsed", verdict_legacy="no")
            else:
                trace["rounds"].append(round_trace)
                fail("judge", jres, "Judge", round_num)
                break
        record[round_key]["Judge"] = jres["reply"]
        round_trace["judge_verdict"] = jres["verdict"]
        round_trace["judge_verdict_robust"] = jres.get("verdict_robust")
        round_trace["judge_verdict_legacy"] = jres.get("verdict_legacy")
        trace["judge_verdicts"].append(jres["verdict"])
        trace["rounds"].append(round_trace)

        if not (jres["agree"] or round_num == cfg.max_round - 1):
            continue

        # --- summariser (fix C1: keyword binding) -------------------------------
        trace["union_selected_ids"] = list(union_selected)
        evidence_text = format_citable_evidence(pool, union_selected) if cfg.cite else None
        sres = summary_final_answer(summarizer, question=question, debate=debate_text, evidence=evidence_text,
                                    prompt=prompt, cite=cfg.cite, legacy_swap=cfg.legacy)
        account("summariser", sres.get("prompt"), sres.get("reply"), sres.get("usage"))
        trace["summariser_binding_ok"] = sres.get("question_slot_is_question")
        if not sres["ok"]:
            if cfg.legacy:
                trace["call_failures"] += 1
                summary = sres.get("error") or "Error"
            else:
                fail("summariser", sres, "Summarizer", round_num)
                break
        else:
            summary = sres["reply"]

        record["Summary"] = summary
        record["llm_answers"] = extract_answer(summary)
        final_answer = extract_final_answer(summary)
        record["final_answer"] = final_answer

        cited, none_flag = parse_citations(summary)
        valid = [c for c in cited if 0 <= c < len(pool) and c in union_selected]
        trace["cited_ids"] = cited
        trace["cited_valid_ids"] = valid
        trace["cited_invalid_ids"] = [c for c in cited if c not in valid]
        trace["cite_none_flag"] = none_flag
        if cfg.task_type == "classification":
            status = "not_applicable"
        else:
            status = support_check(final_answer, valid, pool)
        trace["initial_support_status"] = status
        trace["support_status"] = status

        # --- bounded verification (Algorithm 4) ----------------------------------
        if cfg.verify and cfg.task_type == "extractive" and status != "supported":
            vres = verify_answer(summarizer, question=question, debate=debate_text,
                                 evidence=format_citable_evidence(pool, union_selected), prompt=prompt)
            account("verifier", vres.get("prompt"), vres.get("reply"), vres.get("usage"))
            ver = {"attempted": True, "ok": vres["ok"], "reply": vres.get("reply"), "error": vres.get("error"),
                   "attempts": vres.get("attempts"), "adopted": False, "abstained": False, "answer": None,
                   "cited_ids": [], "cited_valid_ids": [], "support_status": None}
            if vres["ok"]:
                v_answer = extract_final_answer(vres["reply"])
                v_cited, v_none = parse_citations(vres["reply"])
                v_valid = [c for c in v_cited if 0 <= c < len(pool) and c in union_selected]
                v_status = support_check(v_answer, v_valid, pool)
                ver.update(answer=v_answer, cited_ids=v_cited, cited_valid_ids=v_valid, support_status=v_status,
                           abstained=(normalize_text(v_answer) == "unsupported") or (v_none and not v_valid))
                if v_status == "supported":
                    ver["adopted"] = True
                    record["final_answer"] = v_answer
                    record["llm_answers"] = extract_answer(vres["reply"])
                    record["Summary_verified"] = vres["reply"]
                    trace["support_status"] = "supported"
                    trace["cited_ids"] = v_cited
                    trace["cited_valid_ids"] = v_valid
                    trace["cited_invalid_ids"] = [c for c in v_cited if c not in v_valid]
                    trace["cite_none_flag"] = v_none
            trace["verification"] = ver
        break

    if record["status"] == "failed":
        record["final_answer"] = None
        record["llm_answers"] = []
    trace["elapsed_sec"] = time.time() - t0
    record["trace"] = trace
    return record
