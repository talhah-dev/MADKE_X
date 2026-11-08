# MADKE-X: Traceable, Evidence-Governed Multi-Agent Debate for Knowledge-Intensive QA

This repository extends **MADKE** (_Learning to Break:_ Knowledge-Enhanced Reasoning in Multi-Agent Debate System,
Neurocomputing 2025, [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231224018344)]) with the
**MADKE-X** upgrades: a failure-aware interface layer (eight repaired defects C1-C8), an evidence governor
(5+5 de-duplicated pool ordered by question-only BM25 + RRF), a decision trace with `{E#}` citations and a
lexical support check, and a bounded verification step. Everything runs **without a paid API key**:

* `--mode offline` -- deterministic scripted agents reproduce every code path on all 500 questions x 5 datasets in ~2 min;
* `--mode live` -- any OpenAI-compatible server (vLLM / Ollama / llama.cpp / OpenAI / Azure), e.g. Qwen on free Kaggle / Colab GPUs.

EM, F1, `Cons`, `Co2`, support rate, paired McNemar tests and bootstrap CIs are computed in pure Python
(`utils/metrics_calculator.py`); GPT-4 grading (`models/gpt4_agent.py`) is optional and unused by the tables.

## Quick start

```shell
# no third-party dependency is required (tqdm / tiktoken are optional, see requirements.txt)
python -m unittest discover -s tests -v            # 20 unit tests of the parsers, pool, metrics, debate loop
python run_experiments.py --mode offline           # Tables 3-9 + invariants  -> results/offline/TABLES.md
python madke.py --dataset_name nq --mode offline --cite --verify --end_index 20   # one dataset, one config
```

Live run against a local model server (see `MADKE_X_Kaggle_Colab.ipynb` for a zero-setup Kaggle / Colab version):

```shell
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --dtype half --max-model-len 4096
python run_experiments.py --mode live --base_url http://localhost:8000/v1 --api_key EMPTY \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ --end_index 500 --concurrency 8      # Table 10 -> results/live/<model>/TABLE10.md
```

Live runs checkpoint every question (`results/live/<model>/<dataset>/<system>.jsonl`); re-running the same command
resumes, `--retry_failed` re-runs questions whose calls failed. Any prediction file can be re-scored with

```shell
python evaluation.py --prediction_path results/live/<model>/nq/madke_x.json --compare_path results/live/<model>/nq/madke_legacy.json
```

## Data integrity (read-only)

The five raw files under `data/` are never modified. Their SHA-256 digests are pinned in `DATA_CHECKSUMS.sha256`
and in `utils/data_utils.DATA_SHA256`; every run verifies them (`sha256sum -c DATA_CHECKSUMS.sha256` does the same).

## What changed (Section 4 of the paper)

| Step | Component | Code |
|---|---|---|
| 4.1 | C1 summariser arguments bound by keyword (`question=`, `debate=`, `evidence=`) | `utils/prompt_utils.summary_final_answer` |
| 4.1 | C2 no hard-coded `[59:]` slice; `--start_index/--end_index` | `madke.py`, `parser.py` |
| 4.1 | C3 order-preserving evidence-ID parser (Algorithm 1), exact bounds, dedup, cap `k=3` | `utils/evaluation_utils.parse_evidence_ids` |
| 4.1 | C4 last `[Yes]`/`[No]` verdict (case/quote-insensitive), early stop on the exact round | `utils/evaluation_utils.parse_judge_verdict` |
| 4.1 | C5/C6 repaired `Cons` (any round incl. final) and `Co2` (`v_1 != yes`, `v_{R_q} == yes`) | `utils/metrics_calculator.compute_cons/compute_co2` |
| 4.1 | C7/C8 failure-aware client `AgentResult(ok, text, error, attempts)`, bounded retries, verified TLS (`--ca_bundle`, `--insecure`); failed questions marked `failed`, never scored | `models/openai_compat_agent.py`, `debate.py` |
| 4.2 | Evidence governor: `W[:5]+G[:5]`, normalised-text dedup, question-only BM25, RRF `1/(60+rho)+1/(60+beta)` (Algorithm 2) | `utils/data_utils.build_pool` |
| 4.3 | `{E#}` citations, lexical support check (Algorithm 3), audit `trace` per question | `prompts/madke_x_prompts.py`, `utils/evaluation_utils.support_check`, `debate.py` |
| 4.4 | Bounded verification: at most one extra summariser call, adopted iff supported (Algorithm 4) | `utils/prompt_utils.verify_answer`, `debate.py` |
| 5 | Unified runner (offline / live, checkpoints, Tables 3-10), Kaggle/Colab notebook | `run_experiments.py`, `MADKE_X_Kaggle_Colab.ipynb` |

`--legacy` (or `debate.legacy_config`) reproduces the **unpatched** MADKE behaviour (10+5 pool, reversed parser,
literal `[Yes]` match, swapped summariser arguments, error strings carried into the answer) so that the
"MADKE (re-run)" baseline is evaluated on exactly the same questions as MADKE-X.

The formal equations of every stage, their symbols and their mapping onto the code are given in
[`METHODOLOGY_EQUATIONS.md`](METHODOLOGY_EQUATIONS.md).

## Repository layout

```
debate.py                  debate loop (Algorithms 1-4), DebateConfig, legacy_config / madke_x_config
madke.py / parser.py       single-dataset CLI and all flags
evaluation.py              automatic metrics for prediction files (+ optional GPT-4 grading)
run_experiments.py         offline / live experiment runner, invariants, Tables 3-10
models/                    base.AgentResult, openai_compat_agent (vLLM/Ollama/OpenAI/Azure), scripted_agent (offline), thin ChatGPT/Qwen/GPT4 wrappers
prompts/                   original per-dataset prompts + madke_x_prompts ({E#} summariser, verification)
utils/                     data_utils (checksums, BM25, build_pool), evaluation_utils (parsers, support check, EM/F1),
                           metrics_calculator (Cons/Co2/McNemar/bootstrap/cov@k), prompt_utils, calculate_tokens
tests/test_madke_x.py      unit tests
results/offline/           output of `run_experiments.py --mode offline` (TABLES.md, tables.json, per-question records)
```

## Original MADKE instructions

The original Azure-based commands still work through the compatible flags:

```shell
python madke.py --dataset_name triviaqa --endpoint YOUR_ENDPOINT --deployment_id YOUR_DEPLOYMENT_ID --api_key YOUR_API_KEY --legacy
```

## Citation
If you find this work helpful, we kindly request that citations refer to the arXiv version:
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
