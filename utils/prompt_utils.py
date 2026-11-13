"""Prompt construction and agent-call wrappers for the debate stages.

All summariser arguments are bound BY KEYWORD (fix C1): the original code
called ``summary_final_answer(Summarizer, all_agent_answer, question, prompt)``
against the signature ``(summary_agent, question, all_agent_answer, prompt)``,
which swapped the question and the debate transcript in every summary prompt.
``summary_final_answer`` now takes keyword-only arguments ``question``,
``debate`` and ``evidence``.

Every wrapper returns a dict with at least ``ok``, ``reply``, ``error`` and
``attempts`` so that the debate loop can mark a question as ``failed`` when a
call does not succeed (fix C7/C8) instead of scoring an error string.
"""

import re

from models.base import AgentResult
from utils.evaluation_utils import (
    parse_evidence_ids,
    parse_evidence_ids_legacy,
    parse_judge_verdict,
    parse_judge_verdict_legacy,
)

# ---------------------------------------------------------------------------
# Formatting helpers (formats are shared with models.scripted_agent)
# ---------------------------------------------------------------------------

def _clean(text):
    return (text or "").replace("\n", " - ")


def format_evidence_pool(pool):
    """``Evidence: [0] ...\\n[1] ...`` listing shown to the evidence-selection step."""
    out = "Evidence: "
    for i, evi in enumerate(pool):
        out += "[{}] {}\n".format(i, _clean(evi["contents"]))
    return out


def format_chosen_evidence(pool, ids):
    """``Evidence: (2) ...`` listing of the selected passages shown to a debater."""
    if not ids:
        return ""
    out = "Evidence: "
    for i in ids:
        out += "({}) {}\n".format(i, _clean(pool[i]["contents"]))
    return out


def format_citable_evidence(pool, ids):
    """``{E2} ...`` listing used by the citing summariser / verifier (0-based pool index)."""
    if not ids:
        return "Evidence (cite with {E#}): none\n"
    out = "Evidence (cite with {E#}):\n"
    for i in ids:
        out += "{{E{}}} {}\n".format(i, _clean(pool[i]["contents"]))
    return out


def format_debate(agent_replies):
    """``[Agent 0] reply\\n[Agent 1] reply\\n`` transcript (original format)."""
    return "".join("[Agent {}] {}\n".format(i, reply) for i, reply in enumerate(agent_replies))


def call_agent(agent, messages):
    """Call an agent and always obtain an :class:`AgentResult`."""
    try:
        return AgentResult.from_value(agent(messages))
    except Exception as ex:  # defensive: never let an agent exception kill the run
        return AgentResult(ok=False, text="", error="agent raised {}: {}".format(type(ex).__name__, ex))


def _result_dict(result, **extra):
    out = {"ok": result.ok, "reply": result.text, "error": result.error, "attempts": result.attempts,
           "latency": result.latency, "usage": result.usage, "meta": getattr(result, "meta", None) or {}}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Stage 1: evidence selection
# ---------------------------------------------------------------------------

def self_choose_evidence(agent, question, history_response, evi_pool, prompt, use_self_choose, round_num,
                         cap=3, legacy=False):
    """Let ``agent`` select up to ``cap`` passages from ``evi_pool``.

    Returns a dict with ``ok``, ``reply``, ``chosen_text`` (evidence block for
    the debater prompt), ``ids`` and the full ``parse`` record
    ``{selected_ids, dropped_ids, ignored_ids, out_of_range_ids, not_found}``.
    """
    all_evi_pool = format_evidence_pool(evi_pool)

    if not use_self_choose:
        ids = list(range(len(evi_pool)))
        return {"ok": True, "reply": "Self choose evidence not be used, return {} evidence of all evidence pool.".format(len(evi_pool)),
                "error": None, "attempts": 0, "latency": 0.0, "usage": {}, "chosen_text": all_evi_pool, "ids": ids,
                "parse": {"selected_ids": ids, "dropped_ids": [], "ignored_ids": [], "out_of_range_ids": [], "not_found": False},
                "prompt": None}

    if round_num == 0:
        evi_prompt = prompt.choose_evidence_prompt.format(evidences=all_evi_pool, question=question)
    else:
        evi_prompt = prompt.other_round_choose_evidence_prompt.format(evidences=all_evi_pool,
                                                                      your_historical_answer=history_response,
                                                                      question=question)
    instance = [{"role": "system", "content": prompt.choose_evidence_system_prompt},
                {"role": "user", "content": evi_prompt}]
    result = call_agent(agent, instance)
    if not result.ok:
        return _result_dict(result, chosen_text="", ids=[], parse=None, prompt=evi_prompt)

    if legacy:
        parsed = parse_evidence_ids_legacy(result.text, len(evi_pool))
    else:
        parsed = parse_evidence_ids(result.text, len(evi_pool), cap=cap)

    chosen_text = format_chosen_evidence(evi_pool, parsed.ids)
    if legacy and not parsed.ids and not parsed.not_found:
        chosen_text = "Evidence: "            # the release prints the bare label when it could read no identifier
    parse = {"selected_ids": list(parsed.ids), "dropped_ids": list(parsed.dropped), "ignored_ids": list(parsed.ignored),
             "out_of_range_ids": list(parsed.out_of_range), "not_found": bool(parsed.not_found)}
    return _result_dict(result, chosen_text=chosen_text, ids=list(parsed.ids), parse=parse, prompt=evi_prompt)


# ---------------------------------------------------------------------------
# Stage 2: judge
# ---------------------------------------------------------------------------

def judge_consistency(judge_agent, question, all_agent_answer, prompt, legacy=False):
    """Ask the judge whether the agents agree.  ``verdict`` in {yes, no, unparsed}; ``agree`` is bool."""
    judge_prompt = prompt.judge_prompt.format(question=question, all_answers_from_agents=all_agent_answer)
    judge_instance = [{"role": "system", "content": prompt.judge_system_prompt},
                      {"role": "user", "content": judge_prompt}]
    result = call_agent(judge_agent, judge_instance)
    if not result.ok:
        return _result_dict(result, verdict="unparsed", agree=False, prompt=judge_prompt)
    verdict = parse_judge_verdict_legacy(result.text) if legacy else parse_judge_verdict(result.text)
    return _result_dict(result, verdict=verdict, agree=(verdict == "yes"), prompt=judge_prompt,
                        verdict_robust=parse_judge_verdict(result.text), verdict_legacy=parse_judge_verdict_legacy(result.text))


# ---------------------------------------------------------------------------
# Stage 3: summariser (keyword-only binding, fix C1) and Stage 4: verifier
# ---------------------------------------------------------------------------

def _summary_template(prompt, cite):
    if cite and hasattr(prompt, "summary_prompt_cite"):
        return prompt.summary_prompt_cite
    return prompt.summary_prompt


def summary_final_answer(summary_agent, *, question, debate, evidence=None, prompt, cite=False, legacy_swap=False):
    """Summarise the debate into a final ``[answer]``.

    Parameters are keyword-only so that ``question`` and ``debate`` can never be
    swapped.  ``evidence`` is the ``{E#}`` listing (string) shown when ``cite`` is
    enabled.  ``legacy_swap=True`` deliberately reproduces the original C1 bug
    (used only by the unpatched "MADKE (re-run)" baseline).
    """
    template = _summary_template(prompt, cite)
    q_slot, d_slot = (debate, question) if legacy_swap else (question, debate)
    if cite and "{evidence}" in template:
        user_prompt = template.format(question=q_slot, all_answers_from_agents=d_slot, evidence=evidence or "")
    else:
        user_prompt = template.format(question=q_slot, all_answers_from_agents=d_slot)
    instance = [{"role": "system", "content": prompt.summary_system_prompt},
                {"role": "user", "content": user_prompt}]
    result = call_agent(summary_agent, instance)
    binding_ok = _question_slot_holds(user_prompt, question)
    return _result_dict(result, prompt=user_prompt, question_slot_is_question=binding_ok)


def verify_answer(summary_agent, *, question, debate, evidence, prompt):
    """Bounded verification call (Algorithm 4): answer only if a listed passage states it, else abstain."""
    template = getattr(prompt, "verify_prompt", None)
    if template is None:
        from prompts import madke_x_prompts
        template = madke_x_prompts.verify_prompt
    user_prompt = template.format(question=question, all_answers_from_agents=debate, evidence=evidence or "")
    instance = [{"role": "system", "content": prompt.summary_system_prompt},
                {"role": "user", "content": user_prompt}]
    result = call_agent(summary_agent, instance)
    return _result_dict(result, prompt=user_prompt)


def _question_slot_holds(user_prompt, question):
    """True iff the LAST ``Question:`` line of the prompt starts with the dataset question."""
    lines = re.findall(r"^Question:\s*(.*)$", user_prompt or "", flags=re.M)
    if not lines:
        return False
    return lines[-1].strip().startswith((question or "").strip()[:80])


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def get_assistant_message(completion):
    """Original helper kept for compatibility (raw Azure completion dict -> message)."""
    if completion == "Error":
        return {"role": "assistant", "content": "Error"}
    if "choices" not in completion:
        return {"role": "assistant", "content": completion["error"]["message"]}
    if "content" not in completion["choices"][0]["message"]:
        completion["choices"][0]["message"]["content"] = ""
    return completion["choices"][0]["message"]
