# MADKE-X — Formal Methodology, Equations and Code Mapping

This document states the formal definition of every stage of MADKE-X, maps each symbol onto the Python functions
of this repository and gives the measured effect of each equation. The equation numbers are those of the
manuscript (`overleaf/`, Sections 3, 4 and 6).

All numbers below come from the deterministic offline run of **code version 1.2.0**
(`python run_experiments.py --mode offline`, seed 0, 500 questions per dataset; output in
`results/offline/TABLES.md` and `tables.json`). The manuscript takes the same numbers from
`overleaf/generated/numbers.tex`, which `python paper_assets.py` writes; after a new run, regenerate instead of
editing numbers by hand. Scripted agents exercise the pipeline; their answers are not model answers, so EM, F1
and the level of `Co2` in the offline run carry no meaning. Accuracy is measured only by
`run_experiments.py --mode live`.

## Notation

| Symbol | Meaning | Code |
|---|---|---|
| $q$, $G_q$ | question (or claim, for FEVER); its gold answers | `data["question"]`, `data["answer"]` |
| $Q$, $Q_c$ | evaluated questions; those with status *completed* | `records`, `metrics_calculator.is_completed` |
| $A$, $R$, $R_q$ | agents (2); maximum rounds (3); rounds executed for $q$ | `DebateConfig.agent_num`, `.max_round`, `len(trace["judge_verdicts"])` |
| $W$, $G$ | Wikipedia and Google passage lists of $q$, in retrieval order | `data["wiki_retrieval"]`, `data["google_retrieval"]` |
| $w$, $g$ | passages taken from each list (5 and 5) | `DebateConfig.wiki_k`, `.google_k` |
| $E=(E_0,\dots,E_{m-1})$ | ordered evidence pool, size $m$ | `pool`, `trace["pool_size"]` |
| $B$, $N_B$ | reference set for the BM25 statistics; its size | `data_utils.build_bm25_index` |
| $\rho(e)$, $\pi(e)$, $\beta(e)$ | rank in the source list; position in the concatenated list; BM25 rank | `source_rank`, `concat_rank`, `bm25_rank` |
| $\nu(\cdot)$ | lower case, punctuation to spaces, whitespace collapsed | `evaluation_utils.normalize_text` |
| $\mathrm{norm}(\cdot)$ | SQuAD answer normalisation | `evaluation_utils.normalize_answer` |
| $k$ | passages delivered per selection, at most (3) | `DebateConfig.cap` |
| $I_i^{(r)}$, $U_q$ | identifiers delivered to agent $i$ in round $r$; their union | `trace["rounds"][r]["agents"][name]["selected_ids"]`, `trace["union_selected_ids"]` |
| $v_r$ | parsed verdict of round $r$ | `trace["judge_verdicts"]` |
| $\hat y$, $C$, $C^\star$ | summariser answer; cited identifiers; valid ones | `record["final_answer"]`, `trace["cited_ids"]`, `trace["cited_valid_ids"]` |
| $\sigma$ | support status | `trace["support_status"]` |

---

## Eq. (1) — Calls per question (Section 3.1)

$$C_{\max} = R\,(2A+1)+1$$

Two calls per agent and round (selection, answer), one judge call per round, one summariser call: 16 for
$A=2$, $R=3$. Checked per record by invariant I8 (`run_experiments.inv_calls`).

## Eqs. (2)–(4) — Explicit call results, question status, scoring on the completed set (C7, C8)

$$\mathrm{call}(\mathbf m) = (\mathrm{ok},\, s,\, \epsilon,\, \tau), \qquad 1 \le \tau \le T_{\max}+1$$

$$\mathrm{status}(q)=\begin{cases}\text{failed} & \text{if a mandatory call of } q \text{ ends with } \mathrm{ok}=0\\ \text{completed} & \text{otherwise}\end{cases}
\qquad Q_c=\{q\in Q:\mathrm{status}(q)=\text{completed}\}$$

$$\mathrm{EM}=\frac{1}{|Q_c|}\sum_{q\in Q_c}\max_{g\in G_q}\mathbb 1[\mathrm{norm}(\hat y_q)=\mathrm{norm}(g)],\qquad
\mathrm{F1}=\frac{1}{|Q_c|}\sum_{q\in Q_c}\max_{g\in G_q}F_1(\mathrm{norm}(\hat y_q),\mathrm{norm}(g))$$

**Code.** `models/base.AgentResult(ok, text, error, attempts)`; bounded retries with exponential back-off and TLS
verification in `models/openai_compat_agent.OpenAICompatAgent.generate` ($T_{\max}$ = `--max_retries`, default 3);
`debate.run_debate` sets `record["status"]` and leaves the loop at the first failed mandatory call;
`metrics_calculator.compute_em_f1` scores completed records only. A failed verification call does not fail the
question (the step is optional). The release behaviour (error text carried on) is reproduced by
`DebateConfig(legacy=True)` and counted in `trace["call_failures"]`.

**Measured effect.** With 2 % of the scripted calls failing (first 100 questions of each dataset), the release
scored 100 of 500 questions although at least one of their calls had failed and marked none; MADKE-X marked 68
questions as failed and scored none of them.

## Keyword binding of the summariser prompt (C1)

`utils/prompt_utils.summary_final_answer(agent, *, question, debate, evidence, prompt, ...)` takes keyword-only
arguments; a positional call raises `TypeError`. `trace["summariser_binding_ok"]` stores whether the last
`Question:` line of the prompt holds the question (invariant I7).

**Measured effect.** The question slot holds the question in 0 of 2,500 summariser prompts of the release and in
all 2,500 prompts of MADKE-X.

## Eq. (5) — Evidence selection parser (Algorithm 1; C3)

Let $\mathrm{br}(r)=(u_1,\dots,u_\ell)$ be the bracket groups of a selection reply $r$ and $Z(r)=(z_1,\dots,z_n)$
the integers inside them, in written order.

$$\mathrm{NF}(r)=\mathbb 1[\exists j: u_j \text{ matches } \texttt{no[- ]?found} \text{ (any case)}]$$

$$Z^{\mathrm{in}}=(z_j:0\le z_j<m),\qquad D=\mathrm{last}(Z^{\mathrm{in}}),\qquad O=\mathrm{first}\big((z_j: z_j<0\ \vee\ z_j\ge m)\big)$$

$$\mathrm{parse}(r,m,k)=\begin{cases}(\varnothing,\,1,\,D,\,\varnothing,\,O) & \text{if }\mathrm{NF}(r)=1\\ (D_{n_D-k+1:n_D},\,0,\,\varnothing,\,D_{1:n_D-k},\,O) & \text{otherwise}\end{cases}$$

$\mathrm{last}(\cdot)$ keeps every identifier once, at the position of its last occurrence. The result holds the
delivered identifiers, the not-found flag, the ignored, the dropped and the out-of-range identifiers.

The selection is the **end** of the reply: the selection prompt asks the agent to think step by step, so passages
may be mentioned before the choice is stated. The release reads from the end as well. For every reply the release
can read, both parsers deliver the same passages; MADKE-X additionally reads integers inside any bracket group
(a list in one bracket), recognises the not-found marker in any spelling and delivers in written order instead of
reverse order.

**Code.** `evaluation_utils.parse_evidence_ids(reply, pool_size, cap=3)` returns
`EvidenceParse(ids, not_found, ignored, dropped, out_of_range)`; the release parser is
`parse_evidence_ids_legacy`. Scripted agents return the selection they mean in `AgentResult.meta["intended_ids"]`,
stored as `intended_ids` in the trace.

**Measured effect.** The release delivers the intended passages for 92.3 % of 11,324 scripted selection replies;
the other 877 replies are lists in one bracket, for which it delivers nothing. MADKE-X delivers the intended
passages in the written order for all 7,992 replies of its runs and records 935 replies with dropped and 785 with
out-of-range identifiers. Fixed reply formats are compared in `tables.json["interface_formats"]`.

## Eq. (6) — Judge verdict and stopping round (C4)

$$v_r=\psi(j_r)=\begin{cases}\mathrm{strip}(u^\ast) & \text{if } u^\ast \text{ exists}\\ \text{unparsed} & \text{otherwise}\end{cases}
\qquad R_q=\min(\{r:v_r=\text{yes}\}\cup\{R\})$$

$u^\ast$ is the last bracket group whose content, with quotes, punctuation and spaces stripped at both ends and
in lower case, is `yes` or `no`.

**Code.** `evaluation_utils.parse_judge_verdict`; release test: `parse_judge_verdict_legacy`
(`"Yes" in bracket_groups`). Both readings are stored per round (`judge_verdict_robust`, `judge_verdict_legacy`).

**Measured effect.** In the scripted re-run of the release, the literal test misses 1,348 of 3,181 judge replies
that state agreement (42.4 %), and 845 of 2,500 debates run past the round of agreement (66.2 % stop in it).
All 2,500 debates of MADKE-X stop in the round of agreement (invariant I4).

## Eq. (7) — Debate statistics `Cons` and `Co2` (C5, C6)

$$\mathrm{Cons}=\frac{1}{|Q_c|}\sum_{q\in Q_c}\gamma_q,\quad \gamma_q=\mathbb 1[\exists r\le R_q: v_r^{(q)}=\text{yes}]
\qquad
\mathrm{Co2}=\frac{1}{|Q_c|}\sum_{q\in Q_c}\gamma_q\,c_q,\quad c_q=\max_{g\in G_q}\mathbb 1[\mathrm{norm}(\hat y_q)=\mathrm{norm}(g)]$$

`Cons` is the consistency ratio and `Co2` the **consistent-and-correct** ratio of the MADKE paper. Correctness is
exact match here (MADKE uses a GPT-4 judgement on the question-answering datasets), so the level of `Co2` is not
comparable with the published values. `Cons − Co2` is the share of questions on which the agents agreed on a
wrong answer.

**Code.** `metrics_calculator.compute_cons`, `compute_co2` (`robust=True` reads the judge replies with Eq. (6)
also for the release re-run); the formula of the release (`evaluation.calculate_acc_c`: consistent iff the debate
stopped before the last round) is `compute_cons_original` / `compute_co2_original`; `compute_late_agreement`
gives the share of questions with disagreement in round 1 and agreement later (versions up to 1.1.0 reported this
statistic under the name `Co2`).

**Measured effect.** In the re-run of the release, the judge replies state agreement in 89.4 % of the debates,
the verdicts the release recognised give 73.3 %, and its evaluation formula reports 54.6 %. For MADKE-X the
equation equals a direct count on every dataset (invariant D2).

## Eqs. (8)–(11) — Evidence governor (Algorithm 2; P1–P3)

$$E^{(0)}=W_{1:w}\,\|\,G_{1:g},\qquad \rho(e)=\text{rank of } e \text{ in its source list},\qquad \pi(e)=\text{position of } e \text{ in } E^{(0)}$$

$$E^{(1)}=\{e\in E^{(0)}:\nexists e'\in E^{(0)} \text{ with } \pi(e')<\pi(e) \text{ and } \nu(e')=\nu(e)\}$$

$$\mathrm{BM25}(q,e)=\sum_{t\in q}\mathrm{idf}(t)\,\frac{f(t,e)(k_1+1)}{f(t,e)+k_1(1-b+b\,|e|/\bar L_B)},\qquad
\mathrm{idf}(t)=\ln\Big(\frac{N_B-n_t+0.5}{n_t+0.5}+1\Big)$$

$$S_{\mathrm{RRF}}(q,e)=\frac{1}{\kappa+\rho(e)}+\frac{1}{\kappa+\beta(e)},\quad \kappa=60,\qquad
E=\mathrm{sort}_{\downarrow S_{\mathrm{RRF}},\,\uparrow\pi}(E^{(1)})$$

$k_1=1.5$, $b=0.75$. The statistics $N_B$, $n_t$, $\bar L_B$ come from the reference set $B$: the first $w$
Wikipedia and first $g$ Google passages of **every** question of the benchmark file (4,770 to 4,996 passages),
built once per file. No gold answer is used, and the statistics do not depend on the evaluated index range.
$\beta(e)$ is the BM25 rank with ties broken by $\pi(e)$.

**Code.** `data_utils.build_bm25_index(dataset, w, g)`, `data_utils.build_pool(question, wiki, google, w, g,
bm25_index, dedup, rrf, rrf_k)`; manifest in `trace["pool_manifest"]` (index, source, source rank, SHA-256);
invariant I2 rebuilds every pool from the raw file. Release pool: `data_utils.legacy_pool` (10 + 5, listed as
retrieved).

**Measured effect (no agent involved).** Share of questions with an answer-bearing passage among the first three,
release order → RRF order: TriviaQA 52.0 → 76.0 %, NQ 60.4 → 66.4 %, HotpotQA 21.7 → 41.5 %, 2WikiMultihopQA
11.8 → 26.3 % (paired exact McNemar, largest Holm-adjusted $p=0.008$). The evidence listing of a selection call
shrinks by 44.3–44.8 % relative to the default pool of the release; relative to the specified 5 + 5 pool the
saving is at most 0.5 % (de-duplication only). Coverage falls by 0.8–3.4 points; 7–16 duplicates are removed per
dataset; a pool takes 0.64–0.78 ms to build and the reference statistics 180–197 ms per file.

## Eqs. (12)–(14) — Citations, lexical support check, support rate (Algorithm 3; X1)

$$C=\mathrm{first}\big((c:\{Ec\} \text{ occurs in } s)\big),\qquad C^\star=\{c\in C:0\le c<m\ \wedge\ c\in U_q\}$$

$$\sigma=\begin{cases}
\text{not\_applicable} & \text{if } q \text{ is a FEVER claim or } \nu(\hat y)\in\mathcal V\\
\text{empty\_answer} & \text{if } \nu(\hat y)=\varepsilon\\
\text{no\_citation} & \text{if } C^\star=\varnothing\\
\text{supported} & \text{if } \exists c\in C^\star:\nu(\hat y)\sqsubseteq_w\nu(E_c)\\
\text{cited\_but\_answer\_not\_found} & \text{otherwise}
\end{cases}
\qquad \mathrm{SR}=\frac{1}{|Q_c^{\mathrm{app}}|}\sum_{q\in Q_c^{\mathrm{app}}}\mathbb 1[\sigma_q=\text{supported}]$$

$\mathcal V=\{\text{yes, no, supports, refutes, not enough info}\}$; $\sqsubseteq_w$ is containment as a sequence of
whole words (so "Paris" is not found in "comparison").

**Code.** `evaluation_utils.parse_citations`, `support_check`, `contains_phrase`, `is_verdict_answer`;
status handling in `debate.run_debate`; `metrics_calculator.support_statistics`. Invariant I5 recomputes the
status from the trace.

**Measured effect.** All 3,270 citations of the scripted MADKE-X runs resolve to a pool passage that an agent had
selected, and the stored status is reproduced from the trace for all 2,500 records.

## Eqs. (15)–(16) — Bounded verification (Algorithm 4)

$$\text{trigger: } \sigma(\hat y,C^\star,E)\notin\{\text{supported},\text{not\_applicable}\},\qquad
\hat y_{\mathrm{final}}=\begin{cases}\hat y' & \text{if } \sigma(\hat y',C'^\star,E)=\text{supported}\\ \hat y & \text{otherwise}\end{cases}$$

$$C_{\mathrm X}(q)\le R_q(2A+1)+1+1\le C_{\max}+1$$

**Code.** `utils/prompt_utils.verify_answer`, `prompts/madke_x_prompts.verify_prompt`, `debate.run_debate`
(`trace["verification"]`); invariant I6.

**Measured effect.** Stress test with a scripted summariser that always cites the first listed passage
(`tables.json["verification_stress"]`): the first answer is supported for 61.8–74.4 % of the questions; the step
is called 641 times; 628 second replies keep the answer and cite a passage that contains it, 5 give a new
supported answer, 8 abstain; the support rate ends at 99.6 % on every dataset and no supported answer is changed.

## Eqs. (17)–(20) — Measures and statistics (Section 6)

$$\mathrm{cov@}k=\frac{1}{|Q|}\sum_{q\in Q}\mathbb 1[h_q\le k],\qquad \mathrm{MRR}=\frac{1}{|Q|}\sum_{q\in Q}\frac{\mathbb 1[h_q \text{ exists}]}{h_q}$$

$$\Pr[h\le k]=1-\binom{n-a}{k}\Big/\binom{n}{k},\qquad \mathbb E[1/h]=\sum_{j=1}^{n-a+1}\frac1j\binom{n-j}{a-1}\Big/\binom na$$

$$p=\min\Big(1,\ 2\sum_{i=0}^{\min(b,c)}\binom{b+c}{i}2^{-(b+c)}\Big)$$

$$n=\frac{\big(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}+z_{1-\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)}\big)^2}{(p_1-p_2)^2}$$

$h_q$ is the position of the first passage that contains a gold answer as whole words; questions whose gold
answer is a verdict (FEVER, yes/no) are excluded from these measures.

**Code.** `metrics_calculator.first_hit_rank`, `coverage_and_mrr`, `expected_random_coverage`,
`expected_random_reciprocal_rank`, `mcnemar_exact`, `paired_mcnemar`, `bootstrap_ci`, `paired_difference_ci`,
`two_proportion_sample_size`, `two_proportion_power`, `mcnemar_sample_size`; Holm correction in
`paper_assets.holm`; `run_experiments.pool_analysis`.

## Invariants (`run_experiments.check_invariants`)

| ID | Checked on every MADKE-X record |
|---|---|
| I1 | completed with an answer and a summary, or failed with an error and no answer |
| I2 | pool rebuilt from the raw file equals the manifest; at most $w+g$ passages, no duplicate, sorted by fused score |
| I3 | delivered identifiers in the pool, distinct, at most $k$; dropped and out-of-range identifiers disjoint from them |
| I4 | one verdict per executed round; the debate ends in the first round with verdict yes, or after $R$ rounds |
| I5 | valid citations point into the pool and to selected passages; the support status is reproducible from the trace |
| I6 | at most one verification call, only for an unsupported answer; an adopted answer is supported |
| I7 | the question slot of the summariser prompt holds the question |
| I8 | one summariser call, one judge call per round, $A$ selection and $A$ answer calls per round |
| D1–D3 | per dataset: every index has one record; `Cons` / `Co2` equal a direct count; only completed questions are scored |

All 20,015 checks pass in the offline run (4,003 per dataset).
