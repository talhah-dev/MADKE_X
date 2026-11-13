# MADKE-X: Traceable, Evidence-Governed Multi-Agent Debate for Knowledge-Intensive QA

This repository extends **MADKE** (_Learning to Break:_ Knowledge-Enhanced Reasoning in Multi-Agent Debate System,
Neurocomputing 2025, [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231224018344)]) with the
**MADKE-X** components: a failure-aware interface layer, an evidence governor (5+5 de-duplicated pool ordered by
question-only BM25 + RRF), a decision trace with `{E#}` citations and a lexical support check, and a bounded
verification step. The debate itself (agents, debate prompts, round budget) is unchanged. Everything runs
**without a paid API key**:

* `--mode offline` -- deterministic scripted agents exercise every code path on all 500 questions x 5 datasets in about 3 minutes;
* `--mode live` -- any OpenAI-compatible server (Ollama / vLLM / llama.cpp / OpenAI / Azure), e.g. Qwen on free Kaggle / Colab GPUs.

EM, F1, `Cons`, `Co2`, support rate, paired McNemar tests and bootstrap intervals are computed in pure Python
(`utils/metrics_calculator.py`); GPT-4 grading (`models/gpt4_agent.py`) is optional and unused by the tables.

**Code version 1.2.0.** Every record stores the version that wrote it, and a run never resumes checkpoints of
another version, so results of different versions are not mixed.

## Quick start

```shell
# no third-party package is required to run the experiments (see requirements.txt for the optional ones)
python -m unittest discover -s tests -v            # 30 unit tests of the parsers, pool, metrics, debate loop
python run_experiments.py --mode offline           # tables + invariants  -> results/offline/TABLES.md, tables.json
python paper_assets.py                             # tables, figures and numbers of the paper -> overleaf/generated, overleaf/figures
python madke.py --dataset_name nq --mode offline --cite --verify --end_index 20   # one dataset, one configuration
```

Live run against a model server (see `MADKE_X_Kaggle_Colab.ipynb` for a ready Kaggle / Colab version):

```shell
ollama serve &                                     # or any other OpenAI-compatible server
ollama pull qwen2.5:7b
python run_experiments.py --mode live --base_url http://127.0.0.1:11434/v1 --api_key ollama \
    --model qwen2.5:7b --end_index 500 --concurrency 4 --max_tokens 512      # -> results/live/<model>/TABLE10.md
```

Live runs checkpoint every question (`results/live/<model>/<dataset>/<system>.jsonl`); re-running the same command
resumes, and `--retry_failed` re-runs questions whose calls failed. Any prediction file can be scored again with

```shell
python evaluation.py --prediction_path results/live/<model>/nq/madke_x.json --compare_path results/live/<model>/nq/madke_legacy.json
```

After a live run has finished, `python paper_assets.py` fills the live table of the paper from
`results/live/*/table10.json`. Cells of runs that do not exist stay `--`; nothing is estimated.

## Data integrity (read-only)

The five raw files under `data/` are never modified. Their SHA-256 digests are pinned in `DATA_CHECKSUMS.sha256`
and in `utils/data_utils.DATA_SHA256`; every run verifies them. The digests are computed with line endings
normalised to LF, so a Windows checkout (git `core.autocrlf`, CRLF) and a Linux / Kaggle / Colab copy give the
same result; on Linux `sha256sum -c DATA_CHECKSUMS.sha256` verifies the files directly.

Live runs first send one test request to the endpoint (preflight) and do not start if it fails. A run also stops
after `--max_consecutive_failures` (default 5) questions in a row whose model calls failed, without checkpointing
them, so an unreachable server can never produce an "empty" result table.

## What MADKE-X changes (Section 4 of the paper)

| Finding | MADKE-X | Code |
|---|---|---|
| C1 swapped summariser arguments | arguments bound by keyword (`question=`, `debate=`, `evidence=`) | `utils/prompt_utils.summary_final_answer` |
| C2 hard-coded `[59:]` slice | explicit `--start_index/--end_index` | `madke.py`, `parser.py` |
| C3 selection parser reads only single-integer brackets | Algorithm 1: integers in any bracket group; the selection is the **end** of the reply (last `k=3` distinct in-range ids, delivered in written order); dropped / ignored / out-of-range ids are recorded | `utils/evaluation_utils.parse_evidence_ids` |
| C4 literal `"Yes"` test | last `[Yes]`/`[No]` group, any case, quotes and punctuation stripped; stop in the round of agreement | `utils/evaluation_utils.parse_judge_verdict` |
| C5/C6 `Cons` never counts the last round | `Cons` = agreement in any executed round; `Co2` = consistent **and correct** (exact match); late agreement reported separately | `utils/metrics_calculator.compute_cons / compute_co2 / compute_late_agreement` |
| C7/C8 unbounded retries, error text as reply | `AgentResult(ok, text, error, attempts)`, bounded retries, verified TLS (`--ca_bundle`, `--insecure`); failed questions are marked `failed` and never scored | `models/openai_compat_agent.py`, `debate.py` |
| P1-P3 pool 10+5, duplicates, Wikipedia first | evidence governor (Algorithm 2): `W[:5]+G[:5]`, de-duplication on normalised text, question-only BM25 with statistics of the whole benchmark file, RRF `1/(60+rho)+1/(60+beta)` | `utils/data_utils.build_pool / build_bm25_index` |
| X1 no link from answer to evidence | `{E#}` citations, lexical support check on whole words (Algorithm 3; verdict answers are `not_applicable`), decision `trace` per question | `prompts/madke_x_prompts.py`, `utils/evaluation_utils.support_check`, `debate.py` |
| -- | bounded verification: at most one extra summariser call, adopted iff supported | `utils/prompt_utils.verify_answer`, `debate.py` |
| R1 point estimates only | paired exact McNemar, bootstrap intervals, Holm correction, power analysis | `utils/metrics_calculator.py`, `paper_assets.py` |

`--legacy` (or `debate.legacy_config`) reproduces the behaviour of the **release** (10+5 pool, the original selection
parser and verdict test, swapped summariser arguments, error strings carried into the answer), so that the
"MADKE (re-run)" baseline answers exactly the same questions as MADKE-X.

The formal equations of every stage, their symbols and their mapping onto the code are given in
[`METHODOLOGY_EQUATIONS.md`](METHODOLOGY_EQUATIONS.md).

## Repository layout

```
debate.py                  debate loop (Algorithms 1-4), DebateConfig, legacy_config / madke_x_config, CODE_VERSION
madke.py / parser.py       single-dataset CLI and all flags
evaluation.py              automatic metrics for prediction files (+ optional GPT-4 grading)
run_experiments.py         offline / live experiment runner, invariants, result tables
paper_assets.py            generates every table, figure and number of the paper from the result files
models/                    base.AgentResult, openai_compat_agent (Ollama/vLLM/OpenAI/Azure), scripted_agent (offline), thin ChatGPT/Qwen/GPT4 wrappers
prompts/                   original per-dataset prompts + madke_x_prompts ({E#} summariser, verification)
utils/                     data_utils (checksums, BM25, build_pool), evaluation_utils (parsers, support check, EM/F1),
                           metrics_calculator (Cons/Co2/McNemar/bootstrap/cov@k/power), prompt_utils, calculate_tokens
tests/test_madke_x.py      unit tests
results/offline/           output of `run_experiments.py --mode offline` (TABLES.md, tables.json, per-question records)
overleaf/                  the manuscript (LaTeX); `generated/` and `figures/fig_*.pdf` are written by paper_assets.py
```

## Original MADKE instructions

The original Azure-based commands still work through the compatible flags:

```shell
python madke.py --dataset_name triviaqa --endpoint YOUR_ENDPOINT --deployment_id YOUR_DEPLOYMENT_ID --api_key YOUR_API_KEY --legacy
```

## Citation

MADKE-X builds on MADKE; please cite the original paper:

```bibtex
@article{WANG2025129063,
title = {Learning to break: Knowledge-enhanced reasoning in multi-agent debate system},
journal = {Neurocomputing},
volume = {618},
pages = {129063},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.129063},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224018344},
author = {Haotian Wang and Xiyuan Du and Weijiang Yu and Qianglong Chen and Kun Zhu and Zheng Chu and Lian Yan and Yi Guan}
}
```
