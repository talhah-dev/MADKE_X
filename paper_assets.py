#!/usr/bin/env python
"""Generate every table, figure and number of the manuscript from the result files.

    python paper_assets.py

Reads   results/offline/tables.json                 (written by run_experiments.py --mode offline)
        results/live/<run_tag>/table10.json         (written by run_experiments.py --mode live; optional)
Writes  overleaf/generated/numbers.tex              one \\newcommand per number used in the running text
        overleaf/generated/tab_*.tex                the result tables (complete floats)
        overleaf/figures/fig_*.pdf                  the result figures

No number of the manuscript is typed by hand: re-run this script after any new experiment and
compile the paper again.  Cells of the live table stay "--" until a live run of the same code
version exists; nothing is estimated.
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.metrics_calculator import (  # noqa: E402
    mcnemar_sample_size,
    two_proportion_power,
    two_proportion_sample_size,
)

DATASETS = ["triviaqa", "nq", "hotpotqa", "wikimultihopqa", "fever"]
EXTRACTIVE = ["triviaqa", "nq", "hotpotqa", "wikimultihopqa"]
LABEL = {"triviaqa": "TriviaQA", "nq": "NQ", "hotpotqa": "HotpotQA", "wikimultihopqa": "2WikiMQA", "fever": "FEVER"}
SUFFIX = {"triviaqa": "TQA", "nq": "NQ", "hotpotqa": "HQA", "wikimultihopqa": "WMQ", "fever": "FEV"}

# Main results of MADKE as printed in Table 2 of arXiv:2312.04854v2 (transcribed, not re-measured).
# Columns: TriviaQA F1, G4, NQ F1, G4, HotpotQA F1, G4, 2WikiMQA F1, G4, FEVER EM.
PUBLISHED = [
    ("GPT-3.5-Turbo", [
        ("SP", [0.761, 0.778, 0.390, 0.532, 0.331, 0.384, 0.190, 0.210, 0.548]),
        ("CoT", [0.715, 0.772, 0.418, 0.588, 0.333, 0.410, 0.165, 0.190, 0.556]),
        ("SC", [0.756, 0.818, 0.431, 0.622, 0.342, 0.408, 0.175, 0.206, 0.560]),
        ("RAG", [0.768, 0.824, 0.450, 0.436, 0.423, 0.498, 0.204, 0.228, 0.646]),
        ("MAD", [0.712, 0.798, 0.401, 0.648, 0.287, 0.394, 0.133, 0.186, 0.548]),
        ("MADKE", [0.728, 0.830, 0.453, 0.736, 0.405, 0.542, 0.205, 0.274, 0.612]),
    ]),
    ("Qwen1.5-32B-Chat", [
        ("SP", [0.581, 0.632, 0.295, 0.462, 0.311, 0.384, 0.207, 0.220, 0.536]),
        ("CoT", [0.587, 0.644, 0.276, 0.452, 0.315, 0.382, 0.220, 0.228, 0.546]),
        ("SC", [0.583, 0.628, 0.276, 0.442, 0.312, 0.386, 0.203, 0.222, 0.540]),
        ("RAG", [0.446, 0.554, 0.269, 0.436, 0.309, 0.420, 0.183, 0.248, 0.664]),
        ("MAD", [0.630, 0.704, 0.321, 0.566, 0.363, 0.482, 0.256, 0.286, 0.520]),
        ("MADKE", [0.763, 0.838, 0.475, 0.724, 0.450, 0.570, 0.258, 0.314, 0.646]),
    ]),
    ("Qwen1.5-72B-Chat", [
        ("SP", [0.665, 0.706, 0.378, 0.550, 0.332, 0.428, 0.226, 0.260, 0.596]),
        ("CoT", [0.715, 0.776, 0.406, 0.648, 0.431, 0.514, 0.275, 0.324, 0.625]),
        ("SC", [0.659, 0.700, 0.369, 0.548, 0.336, 0.430, 0.207, 0.242, 0.596]),
        ("RAG", [0.662, 0.762, 0.333, 0.710, 0.438, 0.538, 0.217, 0.304, 0.668]),
        ("MAD", [0.703, 0.796, 0.376, 0.660, 0.397, 0.548, 0.274, 0.344, 0.516]),
        ("MADKE", [0.787, 0.874, 0.479, 0.748, 0.495, 0.622, 0.284, 0.350, 0.686]),
    ]),
    ("GPT-4-Turbo", [
        ("SP", [0.828, 0.902, 0.459, 0.692, 0.472, 0.566, 0.265, 0.284, 0.608]),
    ]),
]
PROPORTION_COLUMNS = [1, 3, 5, 7, 8]          # GPT-4 score columns and FEVER EM: shares of questions
PUBLISHED_DATASETS = ["TriviaQA", "NQ", "HotpotQA", "2WikiMQA", "FEVER"]

MODEL_NAMES = {"qwen2.5:7b": "Qwen2.5-7B", "qwen:32b": "Qwen1.5-32B", "qwen2.5_7b": "Qwen2.5-7B", "qwen_32b": "Qwen1.5-32B"}
LIVE_ORDER = ["Qwen2.5-7B", "Qwen1.5-32B"]


# ----------------------------------------------------------------------------- formatting helpers
def f1(x):
    return "{:.1f}".format(x)


def f2(x):
    return "{:.2f}".format(x)


def f3(x):
    return "{:.3f}".format(x)


def pct(x, nd=1):
    """Share in [0, 1] -> percent string."""
    return "{:.{nd}f}".format(100.0 * x, nd=nd)


def share(a, b, nd=1):
    return "{:.{nd}f}".format(100.0 * a / b, nd=nd) if b else "0.0"


def thousands(n):
    return "{:,}".format(int(round(n)))


def p_tex(p):
    """p-value as LaTeX (math mode content, usable in text through \\ensuremath)."""
    if p >= 0.0995:
        return "{:.2f}".format(p)
    if p >= 0.001:
        return "{:.3f}".format(p)
    exp = int(math.floor(math.log10(p)))
    mant = p / 10 ** exp
    if round(mant, 1) >= 10.0:
        mant, exp = 1.0, exp + 1
    return "{:.1f}\\times10^{{{}}}".format(mant, exp)


def holm(pvalues):
    """Holm step-down adjusted p-values, in the order given."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def tt(text):
    """Typewriter text with LaTeX specials escaped."""
    out = []
    for ch in text:
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "\\":
            out.append("\\textbackslash{}")
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        elif ch == '"':
            out.append("\\textquotedbl{}")
        else:
            out.append(ch)
    return "\\texttt{" + "".join(out) + "}"


def idlist(ids):
    return ", ".join(str(i) for i in ids) if ids else "none"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def float_table(label, caption, body, size="\\small", note=None):
    lines = ["% generated by paper_assets.py -- do not edit by hand", "\\begin{table}[!htbp]", "\\centering", size,
             "\\caption{" + caption + "}", "\\label{" + label + "}", body]
    if note:
        lines.append("\\par\\vspace{3pt}{\\footnotesize " + note + "}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def tabular(colspec, header, rows, resize=False, tabcolsep=None):
    out = []
    if tabcolsep:
        out.append("\\setlength{\\tabcolsep}{" + tabcolsep + "}")
    if resize:
        out.append("\\resizebox{\\textwidth}{!}{%")
    out.append("\\begin{tabular}{" + colspec + "}")
    out.append("\\toprule")
    out.extend(header)
    out.append("\\midrule")
    for row in rows:
        out.append(row if row.startswith(("\\midrule", "\\cmidrule", "\\addlinespace")) else row + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}" + ("%\n}" if resize else ""))
    return "\n".join(out)


# ----------------------------------------------------------------------------- numbers
class Numbers:
    def __init__(self):
        self.items = []
        self.seen = set()

    def add(self, name, value):
        name = "V" + name
        if not name.isalpha():
            raise ValueError("macro name must contain letters only: " + name)
        if name in self.seen:
            raise ValueError("duplicate macro " + name)
        self.seen.add(name)
        self.items.append((name, str(value)))

    def render(self):
        lines = ["% generated by paper_assets.py -- do not edit by hand",
                 "% every number of the running text; write \\VName{} when a space follows"]
        for name, value in self.items:
            lines.append("\\newcommand{\\" + name + "}{" + value + "}")
        return "\n".join(lines) + "\n"


def pooled(full, system, key, sub="diagnostics"):
    total = 0
    for d in DATASETS:
        m = full[d][system]
        total += (m[sub].get(key, 0) if sub else m.get(key, 0))
    return total


def build_numbers(t, live_done):
    full = t["full"]
    N = Numbers()
    add = N.add

    n_per = full["triviaqa"]["x"]["n"]
    n_all = sum(full[d]["x"]["n"] for d in DATASETS)
    add("NPerSet", thousands(n_per))
    add("NQuestions", thousands(n_all))
    add("CodeVersion", t["code_version"])
    add("Python", t["environment"]["python"])
    add("Elapsed", "{:.0f}".format(t["elapsed_sec"]))

    # ---- interface layer, pooled over the five datasets
    for sysname, tag in (("legacy", "Rel"), ("x", "X")):
        sel = pooled(full, sysname, "intent_replies")
        set_ok = pooled(full, sysname, "delivered_set_ok")
        exact_ok = pooled(full, sysname, "delivered_exact_ok")
        add("Sel" + tag, thousands(sel))
        add("SetOk" + tag, thousands(set_ok))
        add("SetBad" + tag, thousands(sel - set_ok))
        add("SetPct" + tag, share(set_ok, sel))
        add("OrderOk" + tag, thousands(exact_ok))
        add("OrderPct" + tag, share(exact_ok, sel))
        add("Bind" + tag, thousands(pooled(full, sysname, "binding_ok")))
        debates = pooled(full, sysname, "debates")
        stop_ok = pooled(full, sysname, "stopped_at_agreement_round")
        add("StopOk" + tag, thousands(stop_ok))
        add("StopPct" + tag, share(stop_ok, debates))
        add("StopLate" + tag, thousands(debates - stop_ok))
        yes = pooled(full, sysname, "judge_yes_robust")
        missed = pooled(full, sysname, "judge_agreement_missed_by_legacy")
        add("JudgeReplies" + tag, thousands(pooled(full, sysname, "judge_replies")))
        add("JudgeYes" + tag, thousands(yes))
        add("Missed" + tag, thousands(missed))
        add("MissedPct" + tag, share(missed, yes))
        add("FalseAgree" + tag, thousands(pooled(full, sysname, "judge_false_agreement_legacy")))
        add("Dropped" + tag, thousands(pooled(full, sysname, "with_dropped")))
        add("OutRange" + tag, thousands(pooled(full, sysname, "with_out_of_range")))
        add("NotFound" + tag, thousands(pooled(full, sysname, "not_found")))
        add("Ignored" + tag, thousands(pooled(full, sysname, "with_ignored")))
        add("Unparsed" + tag, thousands(pooled(full, sysname, "judge_unparsed")))
        # debate statistics pooled over all completed debates
        comp = sum(full[d][sysname]["n_completed"] for d in DATASETS)
        for key, name in (("cons_robust_count", "ConsTrue"), ("cons_count", "ConsPipe"), ("cons_original_count", "ConsForm"),
                          ("co2_robust_count", "CoTrue"), ("co2_count", "CoPipe"), ("co2_original_count", "CoForm")):
            add(name + tag, share(pooled(full, sysname, key, sub=None), comp))
        add("Rounds" + tag, f2(sum(full[d][sysname]["avg_rounds"] * full[d][sysname]["n_completed"] for d in DATASETS) / comp))
        add("Calls" + tag, f1(sum(full[d][sysname]["avg_calls"] * full[d][sysname]["n"] for d in DATASETS) / n_all))
        add("InTok" + tag, thousands(sum(full[d][sysname]["avg_prompt_tokens"] * full[d][sysname]["n"] for d in DATASETS) / n_all))
        add("Failed" + tag, thousands(sum(full[d][sysname]["n_failed"] for d in DATASETS)))
    in_rel = sum(full[d]["legacy"]["avg_prompt_tokens"] for d in DATASETS)
    in_x = sum(full[d]["x"]["avg_prompt_tokens"] for d in DATASETS)
    add("InTokRed", f1(100.0 * (1.0 - in_x / in_rel)))
    cons_gap = [100.0 * (full[d]["legacy"]["cons_robust"] - full[d]["legacy"]["cons_original_formula"]) for d in DATASETS]
    add("ConsGapMin", f1(min(cons_gap)))
    add("ConsGapMax", f1(max(cons_gap)))
    x_gap = [100.0 * (full[d]["x"]["cons"] - full[d]["x"]["cons_original_formula"]) for d in DATASETS]
    add("ConsGapXMin", f1(min(x_gap)))
    add("ConsGapXMax", f1(max(x_gap)))

    # ---- per data set
    for d in DATASETS:
        s = SUFFIX[d]
        p = full[d]["pool"]
        add("TokRel" + s, thousands(p["listing_tokens"]["original"]))
        add("TokSpec" + s, thousands(p["listing_tokens"]["specified"]))
        add("TokX" + s, thousands(p["listing_tokens"]["x"]))
        add("TokRed" + s, f1(p["listing_tokens"]["reduction_pct"]))
        add("Dup" + s, p["duplicates_removed"])
        add("ConsTrueRel" + s, pct(full[d]["legacy"]["cons_robust"]))
        add("ConsFormRel" + s, pct(full[d]["legacy"]["cons_original_formula"]))
        add("ConsX" + s, pct(full[d]["x"]["cons"]))
        add("ScriptEmRel" + s, f1(full[d]["legacy"]["exact"]))
        add("ScriptEmX" + s, f1(full[d]["x"]["exact"]))
        add("ScriptP" + s, "\\ensuremath{" + p_tex(full[d]["paired"]["p_value"]) + "}")
    for d in EXTRACTIVE:
        s = SUFFIX[d]
        p = full[d]["pool"]
        o = p["ordering"]
        add("NSpan" + s, p["n_span"])
        add("NVerdictGold" + s, p["n_verdict_gold"])
        for kname, k in (("One", "cov@1"), ("Three", "cov@3"), ("Five", "cov@5")):
            add("Cov" + kname + "Orig" + s, pct(o["original"][k]))
            add("Cov" + kname + "Rrf" + s, pct(o["rrf"][k]))
            add("Cov" + kname + "Bm" + s, pct(o["bm25"][k]))
            add("Cov" + kname + "Inter" + s, pct(o["interleaved"][k]))
            add("Cov" + kname + "Rand" + s, pct(o["random"][k]))
        add("MrrOrig" + s, f3(o["original"]["mrr"]))
        add("MrrRrf" + s, f3(o["rrf"]["mrr"]))
        pc = p["paired_cov3"]
        add("Gain" + s, f1(pc["gain_pts"]))
        add("GainLo" + s, f1(pc["ci95"][0]))
        add("GainHi" + s, f1(pc["ci95"][1]))
        add("RrfOnly" + s, pc["rrf_only"])
        add("OrigOnly" + s, pc["original_only"])
        add("Pval" + s, "\\ensuremath{" + p_tex(pc["p_value"]) + "}")
        add("CoverRel" + s, f1(p["coverage_pct"]["original"]))
        add("CoverX" + s, f1(p["coverage_pct"]["x"]))
        add("CoverLoss" + s, f1(p["coverage_pct"]["original"] - p["coverage_pct"]["x"]))
        st = t["verification_stress"][d]
        add("StressBefore" + s, pct(st["initial_support_rate"]))
        add("StressAfter" + s, pct(st["final_support_rate"]))
        add("SupX" + s, pct(full[d]["x"]["support"]["support_rate"]))

    # ---- how far each ordering is from the best ordering, at worst (coverage columns, points)
    def worst_gap(name):
        gaps = []
        for d in EXTRACTIVE:
            o = full[d]["pool"]["ordering"]
            for k in ("cov@1", "cov@3", "cov@5"):
                best = max(o[n][k] for n in ("original", "interleaved", "bm25", "rrf"))
                gaps.append(100.0 * (best - o[name][k]))
        return max(gaps)
    add("WorstGapRrf", f1(worst_gap("rrf")))
    add("WorstGapBm", f1(worst_gap("bm25")))
    add("WorstGapInter", f1(worst_gap("interleaved")))
    add("WorstGapOrig", f1(worst_gap("original")))

    # ---- evidence governor: ranges
    gains = [full[d]["pool"]["paired_cov3"]["gain_pts"] for d in EXTRACTIVE]
    pvals = [full[d]["pool"]["paired_cov3"]["p_value"] for d in EXTRACTIVE]
    add("GainMin", f1(min(gains)))
    add("GainMax", f1(max(gains)))
    add("PvalMax", "\\ensuremath{" + p_tex(max(pvals)) + "}")
    add("HolmMax", "\\ensuremath{" + p_tex(max(holm(pvals))) + "}")
    reds = [full[d]["pool"]["listing_tokens"]["reduction_pct"] for d in DATASETS]
    add("TokRedMin", f1(min(reds)))
    add("TokRedMax", f1(max(reds)))
    add("TokRedMean", f1(sum(reds) / len(reds)))
    spec_red = [100.0 * (1.0 - full[d]["pool"]["listing_tokens"]["x"] / full[d]["pool"]["listing_tokens"]["specified"]) for d in DATASETS]
    add("TokRedSpecMax", f1(max(spec_red)))
    saved = [full[d]["pool"]["listing_tokens"]["original"] - full[d]["pool"]["listing_tokens"]["x"] for d in DATASETS]
    add("TokSavedMin", thousands(min(saved)))
    add("TokSavedMax", thousands(max(saved)))
    add("TokSavedSixMax", thousands(6 * max(saved)))
    losses = [full[d]["pool"]["coverage_pct"]["original"] - full[d]["pool"]["coverage_pct"]["x"] for d in EXTRACTIVE]
    add("CoverLossMin", f1(min(losses)))
    add("CoverLossMax", f1(max(losses)))
    dups = [full[d]["pool"]["duplicates_removed"] for d in DATASETS]
    add("DupMin", min(dups))
    add("DupMax", max(dups))
    add("DupTotal", sum(dups))
    cost = [full[d]["pool"]["cost"] for d in DATASETS]
    add("BuildXMin", f2(min(c["x_build_ms_per_q"] for c in cost)))
    add("BuildXMax", f2(max(c["x_build_ms_per_q"] for c in cost)))
    add("BuildRelMax", "{:.4f}".format(max(c["original_build_ms_per_q"] for c in cost)))
    add("IndexMsMin", "{:.0f}".format(min(c["index_build_ms"] for c in cost)))
    add("IndexMsMax", "{:.0f}".format(max(c["index_build_ms"] for c in cost)))
    add("IndexNMin", thousands(min(c["index_passages"] for c in cost)))
    add("IndexNMax", thousands(max(c["index_passages"] for c in cost)))
    add("MemXMin", f1(min(c["x_peak_memory_kb"] for c in cost) / 1024.0))
    add("MemXMax", f1(max(c["x_peak_memory_kb"] for c in cost) / 1024.0))
    add("MemRelMax", f2(max(c["original_peak_memory_kb"] for c in cost) / 1024.0))

    # ---- trace, support, verification (standard scripted run)
    sup = [full[d]["x"]["support"]["support_rate"] for d in EXTRACTIVE]
    add("SupXMin", pct(min(sup)))
    add("SupXMax", pct(max(sup)))
    add("CiteParsed", thousands(sum(full[d]["x"]["support"]["citations_parsed"] for d in DATASETS)))
    add("CiteValid", thousands(sum(full[d]["x"]["support"]["citations_valid"] for d in DATASETS)))
    add("VerCalls", sum(full[d]["x"]["verification"]["attempted"] for d in DATASETS))
    add("VerAdopt", sum(full[d]["x"]["verification"]["adopted"] for d in DATASETS))
    add("VerAbst", sum(full[d]["x"]["verification"]["abstained"] for d in DATASETS))
    add("VerdictAnswers", sum(full[d]["x"]["support"]["distribution"].get("not_applicable", 0) for d in EXTRACTIVE))

    # ---- stress test
    st = t["verification_stress"]
    add("StressApplicable", thousands(sum(st[d]["applicable"] for d in EXTRACTIVE)))
    add("StressCalls", thousands(sum(st[d]["verification_calls"] for d in EXTRACTIVE)))
    add("StressAdopt", thousands(sum(st[d]["adopted"] for d in EXTRACTIVE)))
    add("StressNew", sum(st[d]["adopted_new_answer"] for d in EXTRACTIVE))
    add("StressSame", thousands(sum(st[d]["adopted"] - st[d]["adopted_new_answer"] for d in EXTRACTIVE)))
    add("StressAbst", sum(st[d]["abstained"] for d in EXTRACTIVE))
    add("StressRejected", sum(st[d]["rejected"] for d in EXTRACTIVE))
    add("StressTouched", sum(st[d]["supported_answer_touched"] for d in EXTRACTIVE))
    add("StressBeforeMin", pct(min(st[d]["initial_support_rate"] for d in EXTRACTIVE)))
    add("StressBeforeMax", pct(max(st[d]["initial_support_rate"] for d in EXTRACTIVE)))
    add("StressAfterMin", pct(min(st[d]["final_support_rate"] for d in EXTRACTIVE)))
    add("StressAfterMax", pct(max(st[d]["final_support_rate"] for d in EXTRACTIVE)))

    # ---- failure injection
    fd = t["failure_injection_demo"]
    any_d = DATASETS[0]
    add("FailRate", "{:.0f}".format(100.0 * fd[any_d]["x"]["fail_rate"]))
    add("FailN", fd[any_d]["x"]["n"])
    add("FailNAll", thousands(sum(fd[d]["x"]["n"] for d in DATASETS)))
    rel_scored = [fd[d]["legacy"]["scored_despite_failed_call"] for d in DATASETS]
    x_failed = [fd[d]["x"]["n_failed"] for d in DATASETS]
    add("FailScoredRel", sum(rel_scored))
    add("FailScoredRelMin", min(rel_scored))
    add("FailScoredRelMax", max(rel_scored))
    add("FailMarkedRel", sum(fd[d]["legacy"]["n_failed"] for d in DATASETS))
    add("FailFlaggedX", sum(x_failed))
    add("FailFlaggedXMin", min(x_failed))
    add("FailFlaggedXMax", max(x_failed))
    add("FailScoredX", sum(fd[d]["x"]["scored_despite_failed_call"] for d in DATASETS))

    # ---- invariants
    add("InvChecks", thousands(full[any_d]["invariants"]["n_checks"]))
    add("InvChecksAll", thousands(sum(full[d]["invariants"]["n_checks"] for d in DATASETS)))
    add("InvPassedAll", thousands(sum(full[d]["invariants"]["n_passed"] for d in DATASETS)))

    # ---- scripted answers (pipeline exercise only)
    add("ScriptPMin", "\\ensuremath{" + p_tex(min(full[d]["paired"]["p_value"] for d in DATASETS)) + "}")

    # ---- published table: granularity of the reported shares
    cells = [row[c] for _, methods in PUBLISHED for _, row in methods for c in PROPORTION_COLUMNS]
    multiples = [v for v in cells if abs(v * 500 - round(v * 500)) < 1e-6]
    odd = [v for v in cells if abs(v * 500 - round(v * 500)) >= 1e-6]
    add("PubCells", len(cells))
    add("PubMultiples", len(multiples))
    add("PubOdd", ", ".join("{:.3f}".format(v) for v in odd))

    # ---- power analysis of the published margins
    margins = power_margins()
    add("PowN", len(margins))
    add("PowOver", sum(1 for m in margins if m["n_needed"] > 500))
    add("PowOverTwoK", sum(1 for m in margins if m["n_needed"] > 2000))
    add("PowBelow", sum(1 for m in margins if m["margin"] < 0))
    resolvable = [abs(m["margin"]) for m in margins if m["n_needed"] <= 500]
    add("PowMinResolved", f1(100.0 * min(resolvable)) if resolvable else "--")
    add("PowHalfMargin", f1(100.0 * detectable_margin(500)))
    add("PairedNeed", thousands(mcnemar_sample_size(0.044, 0.25)))
    add("PairedNeedHalf", thousands(mcnemar_sample_size(0.044, 0.5)))

    lines = N.render()
    lines += "\\newif\\iflivedone\n" + ("\\livedonetrue\n" if live_done else "\\livedonefalse\n")
    return lines, N


def detectable_margin(n, base=0.5, power=0.8):
    """Smallest difference above ``base`` that an unpaired test with ``n`` questions per method detects."""
    lo, hi = 0.0001, 0.49
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if two_proportion_power(base, base + mid, n) < power:
            lo = mid
        else:
            hi = mid
    return hi


def power_margins():
    """MADKE against its strongest published baseline, on the share metrics (GPT-4 score, FEVER EM)."""
    out = []
    for backbone, methods in PUBLISHED:
        rows = dict(methods)
        if "MADKE" not in rows:
            continue
        for ds, col in zip(PUBLISHED_DATASETS, PROPORTION_COLUMNS):
            best_name, best = max(((name, row[col]) for name, row in methods if name != "MADKE"), key=lambda z: z[1])
            madke = rows["MADKE"][col]
            n_needed = two_proportion_sample_size(madke, best) if madke != best else float("inf")
            out.append({"backbone": backbone, "dataset": ds, "madke": madke, "best": best, "best_name": best_name,
                        "margin": madke - best, "n_needed": n_needed,
                        "power_500": two_proportion_power(madke, best, 500)})
    return out


# ----------------------------------------------------------------------------- tables
def tab_formats(t):
    fmt = t["interface_formats"]
    rows = []
    for e in fmt["evidence"]:
        rec = []
        if e["x_dropped"]:
            rec.append("dropped " + idlist(e["x_dropped"]))
        if e["x_ignored"]:
            rec.append("ignored " + idlist(e["x_ignored"]))
        if e["x_out_of_range"]:
            rec.append("out of range " + idlist(e["x_out_of_range"]))
        delivered = "none (not found)" if e["x_not_found"] else idlist(e["x_delivered"])
        rows.append(" & ".join([e["format"], tt(e["reply"]), idlist(e["released_delivered"]), delivered,
                                "; ".join(rec) if rec else "--"]))
    rr = ">{\\raggedright\\arraybackslash}"
    a = tabular(rr + "p{0.23\\textwidth}" + rr + "p{0.31\\textwidth}" + rr + "p{0.09\\textwidth}" + rr + "p{0.10\\textwidth}"
                + rr + "p{0.15\\textwidth}",
                ["Reply format & Selection reply & Release delivers & MADKE-X delivers & MADKE-X records \\\\"], rows,
                tabcolsep="3.5pt")
    rows = []
    for v in fmt["verdict"]:
        rows.append(" & ".join([v["format"], tt(v["reply"]), v["intended"], v["released"], v["x"]]))
    b = tabular(rr + "p{0.23\\textwidth}" + rr + "p{0.38\\textwidth}" + rr + "p{0.08\\textwidth}" + rr + "p{0.08\\textwidth}"
                + rr + "p{0.10\\textwidth}",
                ["Reply format & Judge reply & Stated & Release & MADKE-X \\\\"], rows, tabcolsep="3.5pt")
    body = ("\\textbf{(a)} Evidence selection (pool of ten passages, at most three delivered)\\\\[3pt]\n" + a +
            "\n\\par\\vspace{8pt}\n\\textbf{(b)} Judge verdict\\\\[3pt]\n" + b)
    cap = ("Behaviour of the released parsers and of the MADKE-X parsers on fixed reply formats. The outcome is "
           "deterministic and does not depend on a model. Identifiers are listed in the order in which the passages are delivered.")
    return float_table("tab:formats", cap, body, size="\\footnotesize")


def tab_interface(t):
    full = t["full"]

    def cells(d_list):
        out = []
        for key_ok, key_n in (("binding_ok", None), ("delivered_set_ok", "intent_replies"),
                              ("delivered_exact_ok", "intent_replies"), ("stopped_at_agreement_round", "debates")):
            for sysname in ("legacy", "x"):
                ok = sum(full[d][sysname]["diagnostics"].get(key_ok, 0) for d in d_list)
                n = (sum(full[d][sysname]["diagnostics"].get(key_n, 0) for d in d_list) if key_n
                     else sum(full[d][sysname]["n_completed"] for d in d_list))
                out.append(share(ok, n))
        return out

    rows = [" & ".join([LABEL[d]] + cells([d])) for d in DATASETS]
    rows.append("\\midrule")
    rows.append(" & ".join(["All five"] + cells(DATASETS)))
    header = [" & \\multicolumn{2}{c}{Summariser prompt} & \\multicolumn{2}{c}{Same passages} & "
              "\\multicolumn{2}{c}{Same passages,} & \\multicolumn{2}{c}{Stopped at round} \\\\",
              " & \\multicolumn{2}{c}{correct} & \\multicolumn{2}{c}{delivered} & "
              "\\multicolumn{2}{c}{written order} & \\multicolumn{2}{c}{of agreement} \\\\",
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
              "Dataset & MADKE & MADKE-X & MADKE & MADKE-X & MADKE & MADKE-X & MADKE & MADKE-X \\\\"]
    body = tabular("lrrrrrrrr", header, rows, resize=True, tabcolsep="4pt")
    cap = ("Interface properties under scripted agents (500 questions per dataset; percent). Summariser prompt: questions "
           "whose summariser prompt holds the question in the question slot. Same passages: selection replies for which "
           "the delivered passages are the ones the reply selects, as a set and as an ordered list. Stopped at round of "
           "agreement: debates that end in the first round whose judge reply states agreement, or in the last round. "
           "MADKE is the release re-run without changes.")
    return float_table("tab:interface", cap, body)


def tab_consistency(t):
    full = t["full"]
    rows = []
    for d in DATASETS:
        for i, (sysname, lab) in enumerate((("legacy", "MADKE"), ("x", "MADKE-X"))):
            m = full[d][sysname]
            rows.append(" & ".join([("\\multirow{2}{*}{" + LABEL[d] + "}") if i == 0 else "", lab,
                                    pct(m["cons_robust"]), pct(m["cons"]), pct(m["cons_original_formula"]),
                                    pct(m["co2_robust"]), pct(m["co2"]), pct(m["co2_original_formula"])]))
        if d != DATASETS[-1]:
            rows.append("\\midrule")
    header = [" & & \\multicolumn{3}{c}{Cons (\\%)} & \\multicolumn{3}{c}{Co2 (\\%)} \\\\",
              "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
              "Dataset & System & \\shortstack{judge\\\\replies} & \\shortstack{stored\\\\verdicts} & \\shortstack{release\\\\formula} & "
              "\\shortstack{judge\\\\replies} & \\shortstack{stored\\\\verdicts} & \\shortstack{release\\\\formula} \\\\"]
    body = tabular("llrrrrrr", header, rows, tabcolsep="4pt")
    cap = ("Debate statistics under scripted agents (500 questions per dataset). \\emph{Judge replies}: agreement as stated "
           "in the judge replies, read with the verdict parser of Eq.~\\eqref{eq:verdict}; this is the reference value. "
           "\\emph{Stored verdicts}: Eq.~\\eqref{eq:cons} applied to the verdicts that steered the debate. "
           "\\emph{Release formula}: the computation of the released evaluation script. Correctness in Co2 is exact match; "
           "the scripted answers are not model answers, so the level of Co2 carries no meaning and only the differences "
           "between the three columns are of interest.")
    return float_table("tab:consistency", cap, body)


def tab_failure(t):
    fd = t["failure_injection_demo"]
    rows = []
    tot = [0, 0, 0, 0, 0]
    for d in DATASETS:
        rel, x = fd[d]["legacy"], fd[d]["x"]
        vals = [rel["n_failed"], rel["scored_despite_failed_call"], x["n_failed"], x["n_completed"], x["scored_despite_failed_call"]]
        tot = [a + b for a, b in zip(tot, vals)]
        rows.append(" & ".join([LABEL[d], str(rel["n"])] + [str(v) for v in vals]))
    rows.append("\\midrule")
    rows.append(" & ".join(["All five", str(sum(fd[d]["x"]["n"] for d in DATASETS))] + [str(v) for v in tot]))
    rate = "{:.0f}".format(100.0 * fd[DATASETS[0]]["x"]["fail_rate"])
    header = [" & & \\multicolumn{2}{c}{MADKE} & \\multicolumn{3}{c}{MADKE-X} \\\\",
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
              "Dataset & Questions & \\shortstack{marked\\\\failed} & \\shortstack{scored although\\\\a call failed} & "
              "\\shortstack{marked\\\\failed} & \\shortstack{completed\\\\and scored} & \\shortstack{scored although\\\\a call failed} \\\\"]
    body = tabular("lrrrrrr", header, rows, resize=True, tabcolsep="6pt")
    cap = ("Fault injection: every scripted call fails with probability " + rate + "\\%. MADKE passes the error text on as "
           "a reply and scores the question; MADKE-X marks the question as failed and excludes it from the scores.")
    return float_table("tab:failure", cap, body)


ORDERINGS = [("original", "Release order"), ("interleaved", "Interleaved"), ("bm25", "BM25 only"), ("rrf", "RRF (MADKE-X)"),
             ("random", "Random (expected)")]


def tab_ordering(t):
    full = t["full"]
    rows = []
    for d in EXTRACTIVE:
        o = full[d]["pool"]["ordering"]
        best = {k: max(o[name][k] for name, _ in ORDERINGS if name != "random") for k in ("cov@1", "cov@3", "cov@5", "mrr")}
        for i, (name, lab) in enumerate(ORDERINGS):
            m = o[name]

            def cell(k, fmt):
                s = fmt(m[k])
                return "\\textbf{" + s + "}" if name != "random" and abs(m[k] - best[k]) < 1e-12 else s
            first = ("\\multirow{5}{*}{\\shortstack[l]{" + LABEL[d] + "\\\\($n=" + str(full[d]["pool"]["n_span"]) + "$)}}") if i == 0 else ""
            rows.append(" & ".join([first, lab, cell("cov@1", pct), cell("cov@3", pct), cell("cov@5", pct), cell("mrr", f3)]))
        if d != EXTRACTIVE[-1]:
            rows.append("\\midrule")
    body = tabular("llrrrr", ["Dataset & Ordering & cov@1 & cov@3 & cov@5 & MRR \\\\"], rows)
    cap = ("Position of the first answer-bearing passage under five orderings (cov@$k$ in percent of the questions with a "
           "span answer; best value per dataset in bold). \\emph{Release order}: ten Wikipedia passages followed by five "
           "Google passages. The other orderings rearrange the MADKE-X pool; \\emph{Random} is the exact expectation over "
           "all orders of that pool.")
    return float_table("tab:ordering", cap, body)


def tab_paired(t):
    full = t["full"]
    pvals = [full[d]["pool"]["paired_cov3"]["p_value"] for d in EXTRACTIVE]
    adj = holm(pvals)
    rows = []
    for d, pa in zip(EXTRACTIVE, adj):
        pc = full[d]["pool"]["paired_cov3"]
        rows.append(" & ".join([LABEL[d], str(pc["n"]), str(pc["rrf_only"]), str(pc["original_only"]),
                                "$+" + f1(pc["gain_pts"]) + "$", "[" + f1(pc["ci95"][0]) + ", " + f1(pc["ci95"][1]) + "]",
                                "$" + p_tex(pc["p_value"]) + "$", "$" + p_tex(pa) + "$"]))
    header = ["Dataset & $n$ & \\shortstack{RRF\\\\only} & \\shortstack{Release\\\\only} & \\shortstack{Gain\\\\(points)} & "
              "\\shortstack{95\\% bootstrap\\\\interval} & \\shortstack{Exact\\\\McNemar $p$} & \\shortstack{Holm-adjusted\\\\$p$} \\\\"]
    body = tabular("lrrrrccc", header, rows, resize=True, tabcolsep="5pt")
    cap = ("Paired comparison of the RRF order with the release order on cov@3. \\emph{RRF only} and \\emph{Release only} "
           "count the questions that have an answer-bearing passage among the first three under one ordering but not the other.")
    return float_table("tab:paired", cap, body)


def tab_pool(t):
    full = t["full"]
    rows = []
    for d in DATASETS:
        p = full[d]["pool"]
        cov = p.get("coverage_pct")
        rows.append(" & ".join([LABEL[d], f2(p["passages"]["original"]), f2(p["passages"]["x"]),
                                thousands(p["listing_tokens"]["original"]), thousands(p["listing_tokens"]["x"]),
                                f1(p["listing_tokens"]["reduction_pct"]),
                                f1(cov["original"]) if cov else "n/a", f1(cov["x"]) if cov else "n/a",
                                str(p["duplicates_removed"])]))
    header = [" & \\multicolumn{2}{c}{Passages} & \\multicolumn{3}{c}{Listing tokens} & \\multicolumn{2}{c}{Coverage (\\%)} & "
              "Duplicates \\\\",
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}",
              "Dataset & MADKE & MADKE-X & MADKE & MADKE-X & $-$\\% & MADKE & MADKE-X & removed \\\\"]
    body = tabular("lrrrrrrrr", header, rows, resize=True, tabcolsep="5pt")
    cap = ("Evidence pool of the release (ten Wikipedia and five Google passages) and of MADKE-X (five and five, "
           "de-duplicated) on the shipped files; means over 500 questions. Listing tokens: size of the evidence listing of "
           "one selection call (cl100k\\_base tokenizer). Coverage: share of the questions with a span answer whose pool "
           "holds an answer-bearing passage; it is not defined for the verdict labels of FEVER.")
    return float_table("tab:pool", cap, body)


def tab_stress(t):
    st = t["verification_stress"]
    rows = []
    for d in EXTRACTIVE:
        s = st[d]
        rows.append(" & ".join([LABEL[d], str(s["applicable"]), pct(s["initial_support_rate"]), str(s["verification_calls"]),
                                str(s["adopted"] - s["adopted_new_answer"]), str(s["adopted_new_answer"]), str(s["abstained"]),
                                pct(s["final_support_rate"]), str(s["supported_answer_touched"])]))
    header = [" & & & & \\multicolumn{3}{c}{Outcome of the call} & & \\\\",
              "\\cmidrule(lr){5-7}",
              "Dataset & \\shortstack{Appli-\\\\cable} & \\shortstack{Supported\\\\at first (\\%)} & "
              "\\shortstack{Verification\\\\calls} & \\shortstack{adopted,\\\\same answer} & "
              "\\shortstack{adopted,\\\\new answer} & abstained & \\shortstack{Supported at\\\\the end (\\%)} & "
              "\\shortstack{Supported\\\\answers changed} \\\\"]
    body = tabular("lrrrrrrrr", header, rows, resize=True, tabcolsep="5pt")
    cap = ("Stress test of the bounded verification step: a scripted summariser that always cites the first listed passage, "
           "whether or not it contains the answer (500 questions per dataset). Applicable: completed questions whose final "
           "answer is not a verdict.")
    return float_table("tab:stress", cap, body)


INVARIANT_TEXT = [
    ("I1_status_and_answer", "I1", "A record is either completed, with an answer and a summary, or failed, with an error and no answer."),
    ("I2_pool_manifest_dedup_rrf", "I2", "The pool rebuilt from the raw file equals the manifest (indices, sources, ranks, digests); it holds at most $w+g$ passages, no duplicate, and is sorted by fused score."),
    ("I3_evidence_ids_bounds_cap_order", "I3", "Delivered identifiers lie in the pool, are distinct and at most $k$; dropped and out-of-range identifiers are disjoint from them."),
    ("I4_judge_verdicts_early_stop", "I4", "One verdict per executed round; the debate ends in the first round with verdict yes, or after $R$ rounds."),
    ("I5_citations_support_reproducible", "I5", "Valid citations point into the pool and to selected passages; recomputing the support status from the trace gives the stored status."),
    ("I6_bounded_verification", "I6", "At most one verification call, made only for an unsupported answer; an adopted answer is supported; otherwise answer and status are unchanged."),
    ("I7_summariser_binding", "I7", "The question slot of the summariser prompt holds the question."),
    ("I8_call_accounting", "I8", "One summariser call, one judge call per round, $A$ selection and $A$ answer calls per round."),
]
DATASET_INVARIANT_TEXT = [
    ("D1_all_indices_processed", "D1", "Every question of the requested index range has exactly one record."),
    ("D2_cons_co2_consistent", "D2", "Cons and Co2 equal a direct count over the stored verdicts and answers."),
    ("D3_only_completed_scored", "D3", "EM and F1 are computed over the completed questions only."),
]


def tab_invariants(t):
    full = t["full"]
    rows = []
    for key, short, text in INVARIANT_TEXT:
        checked = sum(full[d]["invariants"]["per_invariant"][key]["checked"] for d in DATASETS)
        passed = sum(full[d]["invariants"]["per_invariant"][key]["passed"] for d in DATASETS)
        rows.append(" & ".join([short, text, thousands(checked), thousands(passed)]))
    rows.append("\\midrule")
    for key, short, text in DATASET_INVARIANT_TEXT:
        passed = sum(1 for d in DATASETS if full[d]["invariants"]["dataset_level"][key])
        rows.append(" & ".join([short, text, str(len(DATASETS)), str(passed)]))
    body = tabular("p{0.05\\textwidth}p{0.66\\textwidth}rr", ["ID & Invariant & Checked & Passed \\\\"], rows, tabcolsep="5pt")
    cap = ("Invariants checked on the MADKE-X records of all five datasets (I1--I8 once per question, D1--D3 once per dataset).")
    return float_table("tab:invariants", cap, body)


def tab_published():
    rows = []
    for bi, (backbone, methods) in enumerate(PUBLISHED):
        for i, (name, row) in enumerate(methods):
            first = ("\\multirow{" + str(len(methods)) + "}{*}{" + backbone + "}") if i == 0 else ""
            rows.append(" & ".join([first, name] + ["{:.3f}".format(v) for v in row]))
        if bi != len(PUBLISHED) - 1:
            rows.append("\\midrule")
    header = [" & & \\multicolumn{2}{c}{TriviaQA} & \\multicolumn{2}{c}{NQ} & \\multicolumn{2}{c}{HotpotQA} & "
              "\\multicolumn{2}{c}{2WikiMQA} & FEVER \\\\",
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-11}",
              "Backbone & Method & F1 & G4 & F1 & G4 & F1 & G4 & F1 & G4 & EM \\\\"]
    body = tabular("llccccccccc", header, rows, resize=True, tabcolsep="4pt")
    cap = ("Results published for MADKE and its baselines \\cite{wang2025madke}, copied from Table~2 of the arXiv version "
           "(v2) without re-measurement. F1 and GPT-4 score (G4) for the four question-answering datasets, exact match for "
           "FEVER. SP: standard prompting; CoT: chain of thought; SC: self-consistency; RAG: retrieval-augmented "
           "generation; MAD: multi-agent debate.")
    return float_table("tab:published", cap, body, size="\\footnotesize")


def load_live(pattern, code_version):
    """Live runs of the current code version, keyed by model name."""
    runs = {}
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if data.get("mode") != "live" or data.get("code_version") != code_version:
            print("  live run ignored (other mode or code version):", path)
            continue
        raw = (data.get("args") or {}).get("model") or os.path.basename(os.path.dirname(path))
        name = MODEL_NAMES.get(raw, MODEL_NAMES.get(os.path.basename(os.path.dirname(path)), raw))
        runs[name] = data
    return runs


def tab_live(runs):
    order = [m for m in LIVE_ORDER if m in runs] + [m for m in runs if m not in LIVE_ORDER]
    models = order if order else LIVE_ORDER
    rows = []
    for mi, model in enumerate(LIVE_ORDER if not runs else models):
        data = runs.get(model)
        for i, (sysname, lab) in enumerate((("legacy", "MADKE"), ("x", "MADKE-X"))):
            first = ("\\multirow{2}{*}{" + model + "}") if i == 0 else ""
            if data is None:
                rows.append(" & ".join([first, lab] + ["--"] * 13))
                continue
            full = data["full"]
            ns = [full[d][sysname]["n_completed"] for d in DATASETS if d in full]
            failed = sum(full[d][sysname]["n_failed"] for d in DATASETS if d in full)
            cells = [str(min(ns)) if min(ns) == max(ns) else "{}--{}".format(min(ns), max(ns)), str(failed)]
            for d in EXTRACTIVE:
                cells += [f1(full[d][sysname]["exact"]), f1(full[d][sysname]["f1"])] if d in full else ["--", "--"]
            cells.append(f1(full["fever"][sysname]["exact"]) if "fever" in full else "--")
            done = [d for d in DATASETS if d in full]
            if len(done) == len(DATASETS):      # the averages are defined over all five datasets only
                cells.append(f1(100.0 * sum(full[d][sysname]["cons"] for d in done) / len(done)))
                cells.append(f1(100.0 * sum(full[d][sysname]["co2"] for d in done) / len(done)))
            else:
                cells += ["--", "--"]
            rows.append(" & ".join([first, lab] + cells))
        if mi != len(models) - 1:
            rows.append("\\midrule")
    header = [" & & & & \\multicolumn{2}{c}{TriviaQA} & \\multicolumn{2}{c}{NQ} & \\multicolumn{2}{c}{HotpotQA} & "
              "\\multicolumn{2}{c}{2WikiMQA} & FEVER & & \\\\",
              "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\\cmidrule(lr){13-13}",
              "Backbone & System & $n$ & failed & EM & F1 & EM & F1 & EM & F1 & EM & F1 & EM & Cons & Co2 \\\\"]
    body = tabular("llrrccccccccccc", header, rows, resize=True, tabcolsep="3pt")
    cap = ("Live comparison of MADKE (release re-run) and MADKE-X on identical questions, with Qwen2.5-7B-Instruct and "
           "Qwen1.5-32B-Chat as backbones. $n$: completed questions per dataset; failed: questions excluded because a model call failed; EM and F1 in percent over the completed "
           "questions; Cons and Co2 by Eq.~\\eqref{eq:cons}, averaged over the five datasets. ``--'': the run had not "
           "finished when this version of the manuscript was prepared; no value is estimated.")
    return float_table("tab:live", cap, body)


# ----------------------------------------------------------------------------- figures
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_X, C_REL = "#2a78d6", "#eb6834"            # MADKE-X (categorical slot 1), released MADKE (slot 2)
C_BEFORE, C_AFTER = "#86b6ef", "#2a78d6"     # one hue, two steps (before -> after)


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.labelsize": 8.5, "xtick.color": INK2, "ytick.color": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
        "legend.fontsize": 8.5, "legend.handletextpad": 0.3, "legend.columnspacing": 1.4, "legend.borderaxespad": 0.0,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "figure.dpi": 150,
    })
    return plt


def dot(ax, x, y, color, hollow=False, size=52, zorder=4):
    if hollow:
        ax.scatter([x], [y], s=size - 14, facecolors="white", edgecolors=color, linewidths=1.6, zorder=zorder)
    else:
        ax.scatter([x], [y], s=size, color=color, edgecolors="white", linewidths=1.5, zorder=zorder)


def link(ax, x0, x1, y):
    ax.plot([x0, x1], [y, y], color=AXIS, lw=2, solid_capstyle="round", zorder=2)


def legend_handles(entries):
    from matplotlib.lines import Line2D
    handles = []
    for label, color, kind in entries:
        if kind == "hollow":
            handles.append(Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor="white", markeredgecolor=color,
                                  markeredgewidth=1.6, label=label))
        elif kind == "tick":
            handles.append(Line2D([], [], marker="|", ls="", markersize=9, markeredgewidth=1.6, color=color, label=label))
        else:
            handles.append(Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=color, markeredgecolor="white",
                                  markeredgewidth=1.0, label=label))
    return handles


def rows_axis(ax, labels, xlabel, xlim):
    n = len(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_ylim(n - 0.45, -0.6)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", pad=6)


def fig_interface(t, path, plt):
    full = t["full"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.55})

    props = [("Summariser prompt\ncorrect", "binding_ok", None), ("Same passages\ndelivered", "delivered_set_ok", "intent_replies"),
             ("Same passages,\nwritten order", "delivered_exact_ok", "intent_replies"),
             ("Stopped at round\nof agreement", "stopped_at_agreement_round", "debates")]
    for y, (_, key, nkey) in enumerate(props):
        vals = {}
        for sysname in ("legacy", "x"):
            ok = sum(full[d][sysname]["diagnostics"].get(key, 0) for d in DATASETS)
            n = (sum(full[d][sysname]["diagnostics"].get(nkey, 0) for d in DATASETS) if nkey
                 else sum(full[d][sysname]["n_completed"] for d in DATASETS))
            vals[sysname] = 100.0 * ok / n
        link(a, vals["legacy"], vals["x"], y)
        dot(a, vals["legacy"], y, C_REL)
        dot(a, vals["x"], y, C_X)
        a.text(vals["legacy"], y - 0.27, "{:.1f}".format(vals["legacy"]), ha="center", va="bottom", fontsize=8, color=INK2)
    rows_axis(a, [p[0] for p in props], "Share of cases (%), five datasets pooled", (-4, 104))
    a.set_xticks([0, 25, 50, 75, 100])
    a.set_title("(a) Interface properties")

    for y, d in enumerate(DATASETS):
        rel, x = full[d]["legacy"], full[d]["x"]
        truth, formula, xv = 100 * rel["cons_robust"], 100 * rel["cons_original_formula"], 100 * x["cons"]
        link(b, formula, truth, y - 0.13)
        dot(b, formula, y - 0.13, C_REL, hollow=True)
        dot(b, truth, y - 0.13, C_REL)
        dot(b, xv, y + 0.2, C_X)
    rows_axis(b, [LABEL[d] for d in DATASETS], "Cons (%)", (30, 102))
    b.set_title("(b) Consistency ratio")
    fig.legend(handles=legend_handles([("MADKE (release)", C_REL, "dot"), ("MADKE-X", C_X, "dot"),
                                       ("MADKE, as counted by the release formula (panel b)", C_REL, "hollow")]),
               loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3)
    fig.savefig(path)
    plt.close(fig)


def fig_ordering(t, path, plt):
    full = t["full"]
    fig, ax = plt.subplots(figsize=(6.2, 2.35))
    for y, d in enumerate(EXTRACTIVE):
        o = full[d]["pool"]["ordering"]
        orig, rrf, rand = 100 * o["original"]["cov@3"], 100 * o["rrf"]["cov@3"], 100 * o["random"]["cov@3"]
        link(ax, orig, rrf, y)
        ax.plot([rand, rand], [y - 0.24, y + 0.24], color=INK, lw=1.6, zorder=3, solid_capstyle="butt")
        dot(ax, orig, y, C_REL)
        dot(ax, rrf, y, C_X)
        ax.text(rrf + 2.2, y, "+{:.1f}".format(full[d]["pool"]["paired_cov3"]["gain_pts"]), ha="left", va="center",
                fontsize=8, color=INK2)
    rows_axis(ax, [LABEL[d] for d in EXTRACTIVE], "Questions with an answer-bearing passage among the first three (%)", (0, 90))
    ax.legend(handles=legend_handles([("Release order", C_REL, "dot"), ("RRF order (MADKE-X)", C_X, "dot"),
                                      ("Random order (expected)", INK, "tick")]),
              loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=3)
    fig.savefig(path)
    plt.close(fig)


def fig_cost(t, path, plt):
    full = t["full"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.5})
    off = 0.17
    for y, d in enumerate(DATASETS):
        lt = full[d]["pool"]["listing_tokens"]
        for val, color, dy in ((lt["original"], C_REL, -off), (lt["x"], C_X, off)):
            a.plot([0, val], [y + dy, y + dy], color=color, lw=2, solid_capstyle="butt", zorder=3)
            dot(a, val, y + dy, color, size=40)
        a.text(lt["original"] + 70, y - off, "\u2212{:.1f}%".format(lt["reduction_pct"]), ha="left", va="center", fontsize=8, color=INK2)
    rows_axis(a, [LABEL[d] for d in DATASETS], "Listing tokens per selection call", (0, 2300))
    a.set_xticks([0, 500, 1000, 1500, 2000])
    a.set_xticklabels(["0", "500", "1,000", "1,500", "2,000"])
    a.set_title("(a) Size of the evidence listing")
    fig.legend(handles=legend_handles([("MADKE (10 + 5 passages)", C_REL, "dot"), ("MADKE-X (5 + 5, de-duplicated)", C_X, "dot")]),
               loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2)

    for y, d in enumerate(EXTRACTIVE):
        cov = full[d]["pool"]["coverage_pct"]
        dot(b, cov["original"], y - off, C_REL)
        dot(b, cov["x"], y + off, C_X)
        b.text(cov["original"] + 2.6, y, "\u2212{:.1f}".format(cov["original"] - cov["x"]), ha="left", va="center", fontsize=8, color=INK2)
    rows_axis(b, [LABEL[d] for d in EXTRACTIVE], "Questions with an answer in the pool (%)", (20, 100))
    b.set_title("(b) Coverage of the pool")
    fig.savefig(path)
    plt.close(fig)


def fig_verification(t, path, plt):
    st = t["verification_stress"]
    fig, ax = plt.subplots(figsize=(6.2, 2.35))
    for y, d in enumerate(EXTRACTIVE):
        before, after = 100 * st[d]["initial_support_rate"], 100 * st[d]["final_support_rate"]
        link(ax, before, after, y)
        dot(ax, before, y, C_BEFORE)
        dot(ax, after, y, C_AFTER)
        ax.text(before - 2.4, y, "{:.1f}".format(before), ha="right", va="center", fontsize=8, color=INK2)
    rows_axis(ax, [LABEL[d] for d in EXTRACTIVE], "Answers supported by a cited passage (%)", (0, 104))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.legend(handles=legend_handles([("Before verification (careless citations)", C_BEFORE, "dot"),
                                      ("After the verification call", C_AFTER, "dot")]),
              loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_power(path, plt):
    margins = sorted(power_margins(), key=lambda m: m["n_needed"])
    short = {"GPT-3.5-Turbo": "GPT-3.5", "Qwen1.5-32B-Chat": "Qwen-32B", "Qwen1.5-72B-Chat": "Qwen-72B"}
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    labels = []
    for y, m in enumerate(margins):
        n = min(m["n_needed"], 2e5)
        dot(ax, n, y, INK2, hollow=(m["margin"] < 0), size=46)
        labels.append("{} \u00b7 {} ({}{:.1f})".format(short[m["backbone"]], m["dataset"], "+" if m["margin"] >= 0 else "\u2212",
                                                   100 * abs(m["margin"])))
    ax.axvline(500, color=INK, lw=1.0, zorder=1)
    ax.text(500 * 1.08, -0.62, "500 questions", ha="left", va="top", fontsize=8, color=INK2)
    ax.set_xscale("log")
    rows_axis(ax, labels, "Questions per method needed for 80% power (two-sided test, 5% level)", (60, 2.2e5))
    ax.set_ylim(len(margins) - 0.3, -0.7)
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(["100", "1,000", "10,000", "100,000"])
    ax.minorticks_off()
    ax.legend(handles=legend_handles([("MADKE above its strongest baseline", INK2, "dot"),
                                      ("MADKE below its strongest baseline", INK2, "hollow")]),
              loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the tables, figures and numbers of the manuscript.")
    ap.add_argument("--offline", default=os.path.join("results", "offline", "tables.json"))
    ap.add_argument("--live_glob", default=os.path.join("results", "live", "*", "table10.json"))
    ap.add_argument("--out", default="overleaf")
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args(argv)

    with open(args.offline, encoding="utf-8") as fh:
        t = json.load(fh)
    if t.get("mode") != "offline":
        raise SystemExit("{} is not an offline result file".format(args.offline))
    for d in DATASETS:
        if not t["full"][d]["invariants"]["all_passed"]:
            raise SystemExit("invariants failed for {}: the manuscript is not generated from a failing run".format(d))

    runs = load_live(args.live_glob, t["code_version"])
    live_done = bool(runs) and all(all(d in r["full"] for d in DATASETS) for r in runs.values())
    gen = os.path.join(args.out, "generated")
    numbers, N = build_numbers(t, live_done)
    write(os.path.join(gen, "numbers.tex"), numbers)
    tables = {
        "tab_formats": tab_formats(t), "tab_interface": tab_interface(t), "tab_consistency": tab_consistency(t),
        "tab_failure": tab_failure(t), "tab_ordering": tab_ordering(t), "tab_paired": tab_paired(t),
        "tab_pool": tab_pool(t), "tab_stress": tab_stress(t), "tab_invariants": tab_invariants(t), "tab_published": tab_published(),
        "tab_live": tab_live(runs),
    }
    for name, text in tables.items():
        write(os.path.join(gen, name + ".tex"), text)
    print("code version {} | {} numbers | {} tables | live runs: {}".format(
        t["code_version"], len(N.items), len(tables), ", ".join(runs) if runs else "none (cells stay --)"))

    if not args.no_figures:
        plt = setup_matplotlib()
        figdir = os.path.join(args.out, "figures")
        os.makedirs(figdir, exist_ok=True)
        fig_interface(t, os.path.join(figdir, "fig_interface.pdf"), plt)
        fig_ordering(t, os.path.join(figdir, "fig_ordering.pdf"), plt)
        fig_cost(t, os.path.join(figdir, "fig_cost.pdf"), plt)
        fig_verification(t, os.path.join(figdir, "fig_verification.pdf"), plt)
        fig_power(os.path.join(figdir, "fig_power.pdf"), plt)
        print("figures written to", figdir)
    return N


if __name__ == "__main__":
    main()
