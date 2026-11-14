# EXPECTED RESULTS: NOT REAL RESULTS

> **Note for anyone reading this file:** these are *expected* results only, written for learning
> before the experiments were run. No number here was measured. This file is temporary and will be
> deleted once the real results are available.

---


> **Every accuracy figure below is an illustrative RANGE or SCENARIO, not a measurement.**
> Nothing in this file may be copied into the paper. Delete this file after the real run.
> Real numbers come only from the Kaggle result files via `python paper_assets.py`.

Setting assumed: 500 questions per dataset, two backbones (Qwen 7B, Qwen 32B),
each run with two systems (MADKE re-run, MADKE-X). That is 4 runs per dataset.

---

## To-do 3: Section 8.2 Models and infrastructure (what it will say)

- Hardware: Kaggle, 2 x Tesla T4, Ollama; temperature 0.5; 2 agents; 3 rounds.
- Run time at 500 questions (from the pilot speeds, both systems together):
  - Qwen 7B: about 40–50 s per question per dataset pair -> **about 6–7 hours per dataset**
  - Qwen 32B: about 95 s per question -> **about 13 hours per dataset**

## To-do 4: Table 15 (main live table), illustrative

### Part A: efficiency (confident, follows from the offline tests)

| Measure (per question) | MADKE (re-run) | MADKE-X | Expected change |
|---|---|---|---|
| Debate rounds | about 2.2–2.4 | about 1.5–1.7 | about −30% |
| Model calls | about 12 | about 9 | about −27% |
| Input tokens | about 12,000 | about 6,500 | about −46% |
| Answers scored from failed calls | some (> 0) | 0 | fixed |
| Cons (consistency) | lower | higher (often 90%+) | release formula misses final-round agreement |
| Support rate (answer in cited passage) | not measured | likely 70–95% | new in MADKE-X |

### Part B: accuracy (UNCERTAIN, three scenarios)

Typical EM range for open models of this size with retrieved passages
(a rough range, not a prediction for this code):

| Dataset | Qwen 7B, typical EM | Qwen 32B, typical EM |
|---|---|---|
| TriviaQA | 50–65 | 60–72 |
| NQ | 30–45 | 38–50 |
| HotpotQA | 25–38 | 32–45 |
| 2WikiMultihopQA | 18–32 | 25–40 |
| FEVER (accuracy) | 55–70 | 65–78 |

How MADKE-X might compare with MADKE on the same model:

| Scenario | MADKE-X vs MADKE (EM) | How it would be written |
|---|---|---|
| Better | +2 to +6 points (larger on HotpotQA/2Wiki, thanks to better passage order) | "MADKE-X improves EM by x points (p < 0.05) at 27% fewer calls" |
| Equal (quite likely) | within ±2 points | "Same accuracy at lower cost, with traceable answers" |
| Worse | −1 to −3 points (smaller pool loses some answer passages) | "A small accuracy cost in exchange for 46% fewer tokens" |

Why "better" is possible: offline, the right passage appears in the first three far more often
(TriviaQA 52% -> 76%, HotpotQA 21.7% -> 41.5%). If agents use that, EM rises.
Why "worse" is possible: the 5+5 pool contains an answer passage slightly less often
(for example HotpotQA 51.6% -> 48.2%).

## To-do 5: Section 8.3 Results (what the text will say)

One paragraph per scenario above, plus: McNemar p-values per dataset, a pooled test over all
datasets, and 95% bootstrap intervals.

## To-do 6: Section 8.5 Analyses (what will be measured)

- Supported vs unsupported answers: expected that supported answers are correct more often.
- How often the verification step fires (offline: about 0.5% of questions) and how often it helps.
- 7B vs 32B: the larger model is expected to gain more from better passage order.

## To-do 7: Discussion and to-do 8: Conclusion (one-sentence versions)

- Better: "MADKE-X answers more questions correctly while using fewer calls."
- Equal: "MADKE-X matches MADKE's accuracy at lower cost and makes every answer traceable."
- Worse: "MADKE-X trades a small loss in accuracy for large savings and full traceability."

## To-dos 1–2: Abstract and Introduction

One sentence picked from the matching scenario above.

---

## Comparison with the old MADKE paper

Not directly comparable: the old paper used GPT-3.5/GPT-4 and a GPT-4 judge.
The fair comparison is MADKE vs MADKE-X on the same Qwen model (Table 15).

---

## Added: GPT-3.5 Turbo, GPT-4 Turbo and Qwen 72B

### IMPORTANT before buying: GPT-3.5 Turbo and GPT-4 Turbo no longer exist

OpenAI shut down `gpt-3.5-turbo` (incl. `-0125`) and `gpt-4-turbo` (incl. `-2024-04-09`) on
**23 October 2026** (official OpenAI deprecations page). They can no longer be bought or called.
Any website still selling "GPT-3.5/GPT-4 Turbo" access is either a different model under that name
or not trustworthy, and results from it could not honestly be reported as GPT-3.5/GPT-4.

### Anchor: what the old MADKE paper published (REAL published numbers, Table 16 of the paper)

These are the numbers the original authors reported for MADKE (F1 for QA datasets, EM for FEVER).
They are real, but they come from the authors' own setup, so a re-run will not match them exactly.

| Backbone | TriviaQA F1 | NQ F1 | HotpotQA F1 | 2Wiki F1 | FEVER EM |
|---|---|---|---|---|---|
| GPT-3.5 Turbo (MADKE) | 72.8 | 45.3 | 40.5 | 20.5 | 61.2 |
| Qwen1.5-32B (MADKE) | 76.3 | 47.5 | 45.0 | 25.8 | 64.6 |
| Qwen1.5-72B (MADKE) | 78.7 | 47.9 | 49.5 | 28.4 | 68.6 |
| GPT-4 Turbo | MADKE not published (only single-prompt: 82.8 / 45.9 / 47.2 / 26.5 / 60.8) | | | | |

### Qwen 72B: EXPECTED (not real)

- MADKE (re-run) with Qwen 72B: expected **somewhere near** the published row above, often a few
  points lower because a 4-bit quantised model and a different server are used.
- MADKE-X vs MADKE on Qwen 72B: same three scenarios as before (better +2 to +6, equal ±2,
  worse −1 to −3). The larger model is the most likely to turn the better passage order into
  better answers, so "equal or better" is the more likely outcome, but this is a guess.
- Efficiency (likely): about −27% calls and −46% input tokens, as for the other models.

### Cost and time for 500 questions x 5 datasets x 2 systems (rough estimates)

Token use per question (offline counts of the real prompts): MADKE about 12,400 input tokens,
MADKE-X about 6,600; real models write more output than the scripted agents, about 2,000–2,500
output tokens per question assumed. Total for the full run: about **48 million input + 11 million
output tokens**.

| Option | Rough cost | Rough time | Note |
|---|---|---|---|
| GPT-3.5 Turbo | — | — | Shut down 23 Oct 2026 |
| GPT-4 Turbo | — | — | Shut down 23 Oct 2026 |
| Qwen 72B on a rented GPU (e.g. 1 x A100 80 GB, about $1.5–2.5 per hour) | about $100–250 | about 50–100 GPU hours | Check the provider's current price first |
| Qwen 72B via a paid API (pay per token) | depends on the provider's price per million tokens | fast, can run in parallel | Use the exact model name you report in the paper |
| Qwen 32B / 7B on Kaggle | free | see "To-do 3" above | Already tested |

Before paying anything: check the current price on the provider's own page, and make sure the
model name you buy is exactly the one written in the paper.

---

## Added: single best guesses for Qwen 32B (NOT real, middle of each range)

| Dataset | Old paper, MADKE (real, published) | MADKE-X best guess (NOT real) | Expected range |
|---|---|---|---|
| TriviaQA (F1) | 76.3% | ~74.5% | 70–79% |
| NQ (F1) | 47.5% | ~46% | 42–50% |
| HotpotQA (F1) | 45.0% | ~44% | 40–48% |
| 2WikiMultihopQA (F1) | 25.8% | ~25.5% | 22–29% |
| FEVER (EM) | 64.6% | ~64% | 60–68% |

Higher is better. The guesses are slightly lower than the old paper only because of the setup,
not the method: the old paper used the full model on strong paid servers, while this run uses a
compressed (4-bit) model on free Kaggle GPUs, which usually costs 1–3 points.
The fair comparison in the paper is MADKE vs MADKE-X on the SAME model and hardware (Table 15).
