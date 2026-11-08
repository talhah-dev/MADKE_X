"""Evaluate MADKE / MADKE-X prediction files WITHOUT paid grading.

Computes EM, F1, ``Cons``, ``Co2`` (repaired definitions, fixes C5/C6), the
original ``Cons`` formula for comparison, support statistics (Algorithm 3),
verification statistics (Algorithm 4) and -- with ``--compare_path`` -- a paired
exact McNemar test on EM.  Failed questions (``status == "failed"``) are
reported but never scored as wrong answers (fix C8).

GPT-4 grading (``--eval_metrics gpt4_score``) is OPTIONAL and only runs when an
endpoint is configured; it is not needed for any table of the paper.

Example::

    python evaluation.py --prediction_path results/predictions/nq/nq_madke_x_live.json \
        --compare_path results/predictions/nq/nq_madke_legacy_live.json --evaluation_path results/eval_nq.json
"""

import argparse
import json
import warnings

from parser import add_agent_backend_args
from utils.data_utils import read_json_data, read_jsonl_data, write_json_data
from utils.metrics_calculator import (
    average_rounds,
    bootstrap_ci,
    compute_co2,
    compute_cons,
    compute_cons_original,
    compute_em_f1,
    is_completed,
    paired_em_vectors,
    paired_mcnemar,
    record_prediction,
    support_statistics,
    verification_statistics,
)

warnings.filterwarnings("ignore")


def load_predictions(path):
    return read_json_data(path) if path.endswith(".json") else read_jsonl_data(path)


def evaluate_records(records, max_round=3, n_boot=1000, seed=0):
    """All automatic metrics for one prediction file."""
    em_f1 = compute_em_f1(records)
    cons = compute_cons(records)
    co2 = compute_co2(records)
    cons_orig = compute_cons_original(records, max_round=max_round)
    support = support_statistics(records)
    verification = verification_statistics(records)
    em_ci = bootstrap_ci(em_f1["per_question_em"], n_boot=n_boot, seed=seed, scale=100.0)
    f1_ci = bootstrap_ci(em_f1["per_question_f1"], n_boot=n_boot, seed=seed, scale=100.0)
    return {
        "n_records": len(records),
        "n_completed": em_f1["total"],
        "n_failed": em_f1["n_failed"],
        "exact": em_f1["exact"], "exact_ci95": em_ci,
        "f1": em_f1["f1"], "f1_ci95": f1_ci,
        "cons": cons["cons"], "cons_count": cons["count"],
        "co2": co2["co2"], "co2_count": co2["count"],
        "cons_original_formula": cons_orig["cons_original"],
        "avg_rounds": average_rounds(records),
        "support_rate": support["support_rate"], "support_distribution": support["distribution"],
        "verification": verification,
    }


def calculate_em_f1(result_data):
    """Original API: EM/F1 of a prediction list (now excludes failed questions)."""
    scores = compute_em_f1(result_data)
    out = {"exact": scores["exact"], "f1": scores["f1"], "total": scores["total"]}
    print(out)
    return out


def calculate_gpt4_score(args, result_data):
    """OPTIONAL GPT-4 grading (requires a configured endpoint).  Not used by any paper table."""
    from models.gpt4_agent import GPT4
    from prompts.eval_gpt4_prompt import eval_prompt, eval_system_prompt
    from utils.evaluation_utils import extract_final_answer

    eval_gpt4 = GPT4(args, "Eval")
    eval_process, acc = [], []
    for result in result_data:
        if not is_completed(result):
            continue
        evaluation_answer = record_prediction(result) or "No Answer"
        eval_user_prompt = eval_prompt.format(question=result["question"],
                                              reference_answers=", ".join(result["answer"]),
                                              evaluation_answer=evaluation_answer)
        res = eval_gpt4([{"role": "system", "content": eval_system_prompt},
                         {"role": "user", "content": eval_user_prompt}])
        if not res.ok:
            eval_process.append({"question": result["question"], "status": "failed", "error": res.error})
            continue
        verdict = extract_final_answer(res.text)
        acc.append(1.0 if "true" in verdict.lower() else 0.0)
        eval_process.append({"question": result["question"], "truth_answer": result["answer"],
                             "llm_answer": evaluation_answer, "llm_summary": result.get("Summary", ""),
                             "eval_gpt4": res.text})
    mean = sum(acc) / len(acc) if acc else 0.0
    print("gpt4 acc: {} over {} graded answers".format(mean, len(acc)))
    return eval_process, {"gpt4_score": mean, "graded": len(acc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic evaluation (EM/F1/Cons/Co2/support) of prediction files")
    parser.add_argument("--eval_metrics", type=str, default="all", choices=["all", "em_f1", "cons_co2", "support", "gpt4_score"])
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--round_num", "--max_round", dest="max_round", type=int, default=3)
    parser.add_argument("--prediction_path", type=str, required=True)
    parser.add_argument("--compare_path", type=str, default=None, help="Second prediction file for a paired McNemar test (EM).")
    parser.add_argument("--evaluation_path", type=str, default=None, help="Where to write the metrics json.")
    parser.add_argument("--n_boot", type=int, default=1000)
    add_agent_backend_args(parser)
    args = parser.parse_args()

    predict_data = load_predictions(args.prediction_path)
    report = {"prediction_path": args.prediction_path}

    if args.eval_metrics == "gpt4_score":
        if not (args.model or (args.endpoint and args.deployment_id)):
            raise SystemExit("gpt4_score needs --model/--base_url or --endpoint/--deployment_id; "
                             "all paper metrics are available without it (use --eval_metrics all).")
        eval_process, gpt4 = calculate_gpt4_score(args, predict_data)
        report.update(gpt4)
        report["eval_process"] = eval_process
    else:
        metrics = evaluate_records(predict_data, max_round=args.max_round, n_boot=args.n_boot, seed=args.seed)
        if args.eval_metrics == "em_f1":
            metrics = {k: metrics[k] for k in ("n_completed", "n_failed", "exact", "exact_ci95", "f1", "f1_ci95")}
        elif args.eval_metrics == "cons_co2":
            metrics = {k: metrics[k] for k in ("n_completed", "cons", "co2", "cons_original_formula", "avg_rounds")}
        elif args.eval_metrics == "support":
            metrics = {k: metrics[k] for k in ("n_completed", "support_rate", "support_distribution", "verification")}
        report.update(metrics)
        if args.compare_path:
            other = load_predictions(args.compare_path)
            xs, ys, _ = paired_em_vectors(other, predict_data)
            report["paired_mcnemar_em_vs_compare"] = paired_mcnemar(xs, ys)
            report["compare_metrics"] = evaluate_records(other, max_round=args.max_round, n_boot=args.n_boot, seed=args.seed)

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if args.evaluation_path:
        write_json_data(args.evaluation_path, report)
