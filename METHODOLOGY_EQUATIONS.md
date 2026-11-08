# MADKE-X — Formal Methodology, Equations and Code Mapping

This document states the formal definition of every stage of MADKE-X (Section 4 of the manuscript), maps each
symbol onto the Python functions and variables of this repository, and summarises the quantitative effect of each
equation. All quantitative statements refer to the deterministic offline verification run
(`python run_experiments.py --mode offline`, seed 0, 500 questions per dataset, scripted agents; output in
`results/offline/TABLES.md`). The scripted agents exercise the pipeline deterministically; EM and F1 values of that
run therefore characterise the pipeline rather than a language model, whereas live results (Table 10) are
produced by `run_experiments.py --mode live`.

## Global notation

| Symbol | Meaning | Code |
|---|---|---|
| $q$ | a question (or claim, for FEVER) | `data["question"]` |
| $G_q$ | the set of gold answers of $q$ | `data["answer"]` |
| $Q$, $Q_c \subseteq Q$ | evaluated questions; questions whose status is *completed* | `records`, `metrics_calculator.is_completed` |
| $N$ | number of debating agents (2) | `DebateConfig.agent_num` |
| $R$ | maximum number of rounds (3) | `DebateConfig.max_round` |
| $R_q \le R$ | number of rounds actually executed for $q$ | `len(trace["judge_verdicts"])`, `stop_round + 1` |
| $W, G$ | Wikipedia and Google retrieval lists of $q$ (retrieval order) | `data["wiki_retrieval"]`, `data["google_retrieval"]` |
| $P = (E_0, \dots, E_{n-1})$ | the evidence pool, $n = \lvert P \rvert$ | `pool`, `trace["pool_size"]` (0-based indices) |
| $\nu(\cdot)$ | text normalisation: lower-case, punctuation → space, whitespace collapsed | `evaluation_utils.normalize_text` |
| $\mathrm{norm}(\cdot)$ | SQuAD answer normalisation (lower-case, punctuation and articles removed) | `evaluation_utils.normalize_answer` |
| $\hat{y}_q$ | final predicted answer of $q$ | `record["final_answer"]` |
| $\mathbb{1}[\cdot]$ | indicator function | — |

---

## Eq. 1 — Failure-aware interface calls and completed-set scoring (Section 4.1; C7, C8)

Every model call returns an explicit result tuple instead of a bare string:

$$
\mathrm{call}(m) = (\mathrm{ok},\; s,\; \epsilon,\; a), \qquad a \le A_{\max} + 1,
\tag{1a}
$$

where $m$ is the message list, $\mathrm{ok} \in \{0, 1\}$, $s$ is the reply text (empty when $\mathrm{ok}=0$),
$\epsilon$ is the error description and $a$ the number of attempts, bounded by the retry budget $A_{\max}$.
The status of a question is derived from the calls of its mandatory stages (evidence selection, debater,
judge, summariser):

$$
\mathrm{status}(q) =
\begin{cases}
\text{failed} & \text{if } \exists\, \text{mandatory call with } \mathrm{ok} = 0 \text{ after } A_{\max} \text{ retries},\\
\text{completed} & \text{otherwise,}
\end{cases}
\qquad
Q_c = \{\, q \in Q : \mathrm{status}(q) = \text{completed} \,\}.
\tag{1b}
$$

Accuracy metrics are defined on $Q_c$ only, so an error string can never be scored as an answer:

$$
\mathrm{EM} = \frac{1}{\lvert Q_c \rvert} \sum_{q \in Q_c} \max_{g \in G_q} \mathbb{1}\big[\mathrm{norm}(\hat{y}_q) = \mathrm{norm}(g)\big],
\qquad
\mathrm{F1} = \frac{1}{\lvert Q_c \rvert} \sum_{q \in Q_c} \max_{g \in G_q} F_1\big(\mathrm{norm}(\hat{y}_q), \mathrm{norm}(g)\big),
\tag{1c}
$$

with $F_1$ the token-overlap harmonic mean of precision and recall. The number $\lvert Q \setminus Q_c \rvert$ is
reported alongside every table.

**Code mapping.** (1a): `models/base.AgentResult(ok, text, error, attempts)`; retries, exponential back-off and
TLS verification (`--ca_bundle`, `--insecure`) in `models/openai_compat_agent.OpenAICompatAgent.generate`
with $A_{\max}$ = `--max_retries`. (1b): `debate.run_debate` sets `record["status"]` and `record["error"]`
(stage, agent, round, error, attempts) and leaves the debate loop at the first failed mandatory call; the
unpatched baseline (`DebateConfig(legacy=True)`) instead carries the error text on and increments
`trace["call_failures"]`. (1c): `utils/metrics_calculator.compute_em_f1` (`is_completed`,
`record_prediction`), built on `utils/evaluation_utils.compute_exact / compute_f1`.

**Symbols.** $m$ — chat messages; $s$ — reply; $\epsilon$ — error message; $a$ — attempts; $A_{\max}$ — retry
budget (default 3); $Q, Q_c$ — all / completed questions; $G_q$ — gold answers; $\hat{y}_q$ — prediction.

**Quantitative impact.** With a 2 % per-call failure rate injected into 100 questions per dataset, the unpatched
pipeline completed all 100 questions while 16–23 of them (TriviaQA 17, NQ 16, HotpotQA 23, 2WikiMultiHopQA 21,
FEVER 23) were scored although at least one of their calls had failed; the corresponding answers are empty or
derived from an error string and are counted as errors in the denominator. Under Eq. (1b) the same calls mark
12–28 questions per dataset as *failed* (25, 18, 23, 28, 12) and exactly 0 questions are scored despite a
failed call. Without injected failures all 500 questions per dataset complete, so Eq. (1) leaves the clean
tables unchanged while making every reported denominator explicit (invariants I1 and D3,
`run_experiments.check_invariants`).

---

## Eq. 2 — Order-preserving evidence-ID parsing (Section 4.1, Algorithm 1; C3)

Let $B(r) = (b_1, \dots, b_m)$ be the bracket groups `[...]` of a reply $r$ in written order and
$Z(r) = (z_1, \dots, z_\ell)$ the integers contained in those groups, also in written order. With pool size $n$
and cap $k = 3$:

$$
\mathrm{NF}(r) = \mathbb{1}\big[\exists\, j : b_j \text{ matches } \texttt{no[- ]?found}\big],
\qquad
I = \mathrm{dedup}_{\prec}\big(\{ z \in Z(r) : 0 \le z < n \}\big),
\qquad
O = \mathrm{dedup}_{\prec}\big(\{ z \in Z(r) : z < 0 \,\vee\, z \ge n \}\big),
\tag{2a}
$$

$$
\mathrm{parse}(r, n, k) =
\begin{cases}
(\varnothing,\; \mathrm{nf}=1,\; \mathrm{ignored}=I,\; \mathrm{dropped}=\varnothing,\; O) & \text{if } \mathrm{NF}(r) = 1,\\
(I_{1:k},\; \mathrm{nf}=0,\; \mathrm{ignored}=\varnothing,\; \mathrm{dropped}=I_{k+1:},\; O) & \text{otherwise,}
\end{cases}
\tag{2b}
$$

where $\mathrm{dedup}_{\prec}$ keeps the first occurrence in written order. The selection of agent $i$ in round
$r$ is $I_i^{(r)} = I_{1:k}$ and the union of all selections of a question is $U_q = \bigcup_{i, r} I_i^{(r)}$.

**Code mapping.** `utils/evaluation_utils.parse_evidence_ids(reply, pool_size, cap=3)` returns
`EvidenceParse(ids=I[:k], not_found, ignored, dropped=I[k:], out_of_range=O)`; the bounds test `0 <= n < pool_size`
removes the former off-by-one `IndexError` at $z = n$. The per-agent, per-round record is stored in
`trace["rounds"][r]["agents"][name]` as `selected_ids`, `dropped_ids`, `ignored_ids`, `out_of_range_ids`,
`not_found`; $U_q$ is `trace["union_selected_ids"]`. The original behaviour (reversed order, only whole-group
integers, case-sensitive `No Found`) is reproduced by `parse_evidence_ids_legacy` for the baseline.

**Symbols.** $r$ — reply; $B(r)$ — bracket groups; $Z(r)$ — integers; $n$ — pool size; $k$ — cap; $I$ — in-range
ids; $O$ — out-of-range ids; $\mathrm{NF}$ — no-found flag; $I_i^{(r)}$ — selection of agent $i$ in round $r$;
$U_q$ — union of selections.

**Quantitative impact.** Over the 1,248–2,172 evidence-selection replies per dataset produced for MADKE-X,
12.5–13.6 % contained more than $k$ ids (266, 271, 247, 293, 162 replies) whose surplus is now recorded as
*dropped* rather than silently extending the context, and 101–180 replies contained an out-of-range id that is
recorded in $O$ instead of raising an exception or being discarded without trace. Re-parsing the same replies
with the original parser changes the *order* of the selected passages in 987–1,679 replies and the *set* of
selected passages in 20.7–22.3 % of them (408, 439, 425, 484, 260), because comma-separated groups such as
`[2, 0, 7]` are rejected by the original whole-group match. Invariant I3 confirms on all 2,500 questions that
$\lvert I_i^{(r)} \rvert \le k$, ids are unique and in range, and $\mathrm{NF} = 1 \Rightarrow I_i^{(r)} = \varnothing$.

---

## Eq. 3 — Judge verdict, exact-round early stopping, `Cons` and `Co2` (Section 4.1; C4, C5, C6)

For the judge reply $j_r$ of round $r$, the verdict is read from the **last** bracket group whose stripped,
lower-cased content is a verdict token:

$$
v_r = \psi(j_r) =
\begin{cases}
\mathrm{strip}(b_m)\!\downarrow & \text{if } \mathrm{strip}(b_m)\!\downarrow \in \{\text{yes}, \text{no}\} \text{ for the last such } b_m \in B(j_r),\\
\text{unparsed} & \text{otherwise,}
\end{cases}
\qquad
R_q = \min\big(\{ r : v_r = \text{yes} \} \cup \{ R \}\big),
\tag{3a}
$$

where $\mathrm{strip}$ removes surrounding quotes, punctuation and whitespace and $\downarrow$ denotes
lower-casing. The debate stops after round $R_q$ and the summariser is invoked once. The repaired consistency
metrics on $Q_c$ are

$$
\mathrm{Cons} = \frac{1}{\lvert Q_c \rvert} \sum_{q \in Q_c} \mathbb{1}\big[\exists\, r \le R_q : v_r^{(q)} = \text{yes}\big],
\qquad
\mathrm{Co2} = \frac{1}{\lvert Q_c \rvert} \sum_{q \in Q_c} \mathbb{1}\big[v_1^{(q)} \ne \text{yes} \;\wedge\; v_{R_q}^{(q)} = \text{yes}\big].
\tag{3b}
$$

`Cons` includes an agreement reached in the final round $R$; `Co2` requires $R_q \ge 2$ by construction.

**Code mapping.** (3a): `utils/evaluation_utils.parse_judge_verdict` (used by
`utils/prompt_utils.judge_consistency`; `verdict ∈ {yes, no, unparsed}`); the round loop in `debate.run_debate`
breaks when `agree` is true or `round_num == max_round - 1`; verdicts are stored in `trace["judge_verdicts"]`
and `record["stop_round"]` $= R_q - 1$. (3b): `utils/metrics_calculator.compute_cons` and `compute_co2`
(`judge_verdicts(record)`); the original formula, which tested `"[Yes]" in record["Round_<last>"]` on a dict
and therefore never counted a final-round agreement, is kept as `compute_cons_original` for the ablation.
`parse_judge_verdict_legacy` reproduces the original exact `"Yes"` match.

**Symbols.** $j_r$ — judge reply; $b_m$ — last verdict-like bracket group; $v_r$ — verdict of round $r$;
$R_q$ — stopping round; $v_1^{(q)}, v_{R_q}^{(q)}$ — first and last verdict of $q$.

**Quantitative impact.** Among the judge replies of the MADKE-X run, 12.5–29.5 % of the agreements recognised
by Eq. (3a) (183/979, 152/1,061, 182/988, 136/1,086, 184/624) are missed by the original exact `[Yes]` match
because the verdict is quoted, lower-cased or followed by punctuation; in the unpatched run those missed
agreements prolong the debate, which raises the average number of rounds from 1.96–2.17 (MADKE-X) to
2.44–2.53 (baseline) on the four QA datasets and from 1.25 to 1.73 on FEVER. The original `Cons` formula
under-reports consistency by 11.4–20.8 points on the baseline records (46.4 → 66.8 TriviaQA, 44.6 → 63.6 NQ,
42.4 → 63.2 HotpotQA, 37.8 → 58.6 2WikiMultiHopQA, 78.4 → 89.8 FEVER) and by 2.8–6.2 points on MADKE-X records,
the difference being exactly the final-round agreements excluded by the dict-membership test. `Co2` of
Eq. (3b) is 49.8–57.0 % for the baseline and 50.8–56.8 % for MADKE-X on the QA datasets; on FEVER it falls from
41.2 % to 13.2 % because 82.4 % of MADKE-X debates (412/500) already agree in round 1. Invariant I4 verifies on
every record that a `yes` verdict occurs only in the last executed round and that debates without agreement
run exactly $R$ rounds; invariant D2 verifies that `compute_cons` / `compute_co2` equal a direct recount.

---

## Eq. 4 — Evidence governor: 5+5 pool, de-duplication, question-only BM25 and RRF (Section 4.2, Algorithm 2)

The candidate pool is the ordered concatenation of the first $w = 5$ Wikipedia and $g = 5$ Google passages,
each labelled with its source and 1-based source rank $\rho(e)$:

$$
P_0 = W_{1:w} \oplus G_{1:g}, \qquad
\rho(e) = \text{rank of } e \text{ within its own source list}, \qquad
\pi(e) = \text{position of } e \text{ in } P_0 .
\tag{4a}
$$

De-duplication keeps the first passage of every normalised text:

$$
P_1 = \big\{ e \in P_0 : \nexists\, e' \in P_0,\; \pi(e') < \pi(e),\; \nu(e') = \nu(e) \big\}.
\tag{4b}
$$

The question-only lexical score of each remaining passage is Okapi BM25 with $k_1 = 1.5$, $b = 0.75$ and IDF
statistics computed over $P_1$:

$$
\mathrm{BM25}(q, e) = \sum_{t \in q} \ln\!\Big(\frac{\lvert P_1 \rvert - n_t + 0.5}{n_t + 0.5} + 1\Big)\;
\frac{\mathrm{tf}(t, e)\,(k_1 + 1)}{\mathrm{tf}(t, e) + k_1\big(1 - b + b\,\lvert e \rvert / \overline{\lvert e \rvert}\big)},
\tag{4c}
$$

$$
\beta(e) = 1 + \big\lvert \{ e' \in P_1 : \mathrm{BM25}(q, e') > \mathrm{BM25}(q, e) \;\vee\; (\mathrm{BM25}(q, e') = \mathrm{BM25}(q, e) \wedge \pi(e') < \pi(e)) \} \big\rvert,
\tag{4d}
$$

$$
S_{\mathrm{RRF}}(q, e) = \frac{1}{60 + \rho(e)} + \frac{1}{60 + \beta(e)},
\qquad
P = \mathrm{sort}_{\downarrow S_{\mathrm{RRF}},\; \uparrow \pi}(P_1).
\tag{4e}
$$

Ties in $S_{\mathrm{RRF}}$ are broken deterministically by the concatenation position $\pi$. The resulting
0-based position of $e$ in $P$ is the index shown to the agents (`[i]`) and used in citations (`{Ei}`).

**Code mapping.** `utils/data_utils.build_pool(question, wiki_list, google_list, w=5, g=5, bm25_index=None,
dedup=True, rrf=True, rrf_k=60)` returns `(pool, meta)`; per passage: `source`, `source_rank` $= \rho$,
`concat_rank` $= \pi$, `bm25_score`, `bm25_rank` $= \beta$, `rrf_score` $= S_{\mathrm{RRF}}$, `sha256`;
`meta["duplicates_removed"]` $= \lvert P_0 \rvert - \lvert P_1 \rvert$. BM25 is `utils/data_utils.BM25`
(`tokenize` → `\w+` tokens), $\beta$ is `rank_positions`, $\nu$ is `normalize_text`. `w` and `g` are the
`--wiki_k` / `--google_k` flags (the original code default was 10 + 5, kept in `legacy_pool`). The audit
manifest `(index, source, source_rank, sha256)` is `utils/data_utils.pool_manifest` →
`trace["pool_manifest"]`.

**Symbols.** $W, G$ — retrieval lists; $w, g$ — pool budget per source; $P_0, P_1, P$ — candidate,
de-duplicated and re-ordered pool; $\rho(e)$ — source rank; $\pi(e)$ — concatenation position; $\nu$ — text
normalisation; $t$ — question token; $n_t$ — number of pool passages containing $t$; $\mathrm{tf}(t, e)$ — term
frequency; $\lvert e \rvert, \overline{\lvert e \rvert}$ — passage length and mean length; $k_1, b$ — BM25
constants; $\beta(e)$ — lexical rank; $S_{\mathrm{RRF}}$ — fused score with constant 60.

**Quantitative impact.** Replacing the 10 + 5 concatenation by Eq. (4a) reduces the mean pool from 14.5–15.0 to
9.5–10.0 candidates and the pool text from 1,444–1,467 to 789–805 tokens per question (−44.9 % to −45.4 %);
over the whole debate this lowers the per-question prompt-plus-completion volume from 11,511–12,454 tokens
(8,535 on FEVER) to 6,859–7,941 tokens (4,912 on FEVER), i.e. −35.5 % to −42.4 %. Eq. (4b) removes 7–16 exact
duplicates per dataset (affecting 7–15 questions), confirming that duplication within the first 5 + 5 is rare
but non-zero. Eq. (4c–4e) place a passage that lexically contains a gold answer at the first position for 55.2 %
of TriviaQA questions (source order: 41.2 %), 24.4 % of HotpotQA (17.6 %) and 17.0 % of 2WikiMultiHopQA (7.0 %),
with cov@3 rising from 53.8 → 77.2 %, 23.8 → 41.0 % and 12.4 → 26.6 % and MRR from 0.519 → 0.664, 0.245 → 0.335 and
0.133 → 0.230. On NQ, whose Wikipedia list is already well ordered, pure BM25 degrades cov@1 (48.6 → 33.4 %) while
the fusion of Eq. (4e) recovers most of it (40.6 %) and still improves cov@3/cov@5 (61.0 → 64.8 %, 68.4 → 75.8 %),
illustrating why the source rank $\rho$ is retained in the fusion. Pool construction costs 3.8–5.0 ms per
question with a peak allocation of about 2.3 MB for 500 questions (`tracemalloc`). Invariant I2 verifies that
the manifest is reproducible from the raw data (identical SHA-256 per position), contains no duplicate
normalised text, and is sorted by non-increasing $S_{\mathrm{RRF}}$.

---

## Eq. 5 — `{E#}` citations and lexical support check (Section 4.3, Algorithm 3)

Given the summariser reply $s$, the cited indices and their validated subset are

$$
C = \mathrm{dedup}_{\prec}\big(\{ c : \texttt{\{E}c\texttt{\}} \in s \}\big),
\qquad
C^\star = \{ c \in C : 0 \le c < n \;\wedge\; c \in U_q \},
\tag{5a}
$$

and the support status of the extracted answer $\hat{y} = $ last `[...]` group of $s$ is

$$
\sigma(\hat{y}, C^\star, P) =
\begin{cases}
\text{empty\_answer} & \text{if } \nu(\hat{y}) = \varepsilon,\\
\text{no\_citation} & \text{if } C^\star = \varnothing,\\
\text{supported} & \text{if } \exists\, c \in C^\star : \nu(\hat{y}) \sqsubseteq \nu(E_c),\\
\text{cited\_but\_answer\_not\_found} & \text{otherwise,}
\end{cases}
\tag{5b}
$$

where $\sqsubseteq$ denotes the substring relation. For claim-verification data (FEVER) the label is never a
substring of a passage, and $\sigma$ is set to *not\_applicable*. The support rate over the applicable completed
questions $Q_c^{\mathrm{ext}}$ is

$$
\mathrm{SR} = \frac{1}{\lvert Q_c^{\mathrm{ext}} \rvert} \sum_{q \in Q_c^{\mathrm{ext}}} \mathbb{1}\big[\sigma_q = \text{supported}\big].
\tag{5c}
$$

**Code mapping.** The citing summariser prompt is `prompts/madke_x_prompts.summary_prompt_cite` (imported as
`summary_prompt_cite` by every dataset prompt module; FEVER uses `summary_prompt_cite_fever`), selected when
`--cite` is set; the passages $E_c, c \in U_q$ are listed as `{Ec} text` by
`utils/prompt_utils.format_citable_evidence`. (5a): `utils/evaluation_utils.parse_citations` →
`trace["cited_ids"]` $= C$; `trace["cited_valid_ids"]` $= C^\star$; `trace["cited_invalid_ids"]` $= C \setminus C^\star$.
(5b): `utils/evaluation_utils.support_check(answer, cited_ids, pool)` with `answer = extract_final_answer(s)`;
result in `trace["support_status"]` (and `trace["initial_support_status"]` before verification). (5c):
`utils/metrics_calculator.support_statistics`. The full per-question audit record `trace` additionally holds
`pool_manifest`, per-round/per-agent `selected_ids`, `dropped_ids`, `ignored_ids`, `out_of_range_ids`,
`judge_verdicts`, `calls`, `tokens`, `summariser_binding_ok` and `verification`.

**Symbols.** $s$ — summariser reply; $C$ — cited indices; $C^\star$ — cited indices inside the pool and inside
the union of agent selections $U_q$; $\hat{y}$ — final answer; $E_c$ — cited passage; $\sigma$ — support status;
$\mathrm{SR}$ — support rate; $Q_c^{\mathrm{ext}}$ — completed extractive questions.

**Quantitative impact.** Without citations the baseline receives $\sigma =$ *no\_citation* for all 500 completed
questions of every QA dataset ($\mathrm{SR} = 0$ by construction), i.e. no answer of the original system can be
traced to a passage. In the MADKE-X run, 691–820 citations per dataset were parsed and every one of them lay
inside $P \cap U_q$ (invalid citations: 0), and $\mathrm{SR}$ reached 99.2 % (TriviaQA), 99.4 % (NQ), 99.8 %
(HotpotQA) and 99.2 % (2WikiMultiHopQA); the residual 1–4 questions per dataset are *no\_citation* or
*cited\_but\_answer\_not\_found*. These values reflect scripted agents that cite passages which contain their
answer; with a live model the same quantities measure how often the model's answer is actually stated by the
evidence it cites. Invariant I5 recomputes $\sigma$ from the raw data for every record and verifies
$C^\star \subseteq \{0, \dots, n-1\} \cap U_q$.

---

## Eq. 6 — Bounded verification (Section 4.4, Algorithm 4)

When the initial answer is not supported, at most one additional summariser call is made with the union
evidence $E[U_q]$ and an instruction to answer only from a listed passage or abstain:

$$
s' = \mathrm{Summariser}\big(\mathrm{verify\_prompt}(q,\, D,\, E[U_q])\big)
\quad \text{iff} \quad \sigma(\hat{y}, C^\star, P) \ne \text{supported} \;\wedge\; q \text{ is extractive},
\tag{6a}
$$

$$
\hat{y}_{\mathrm{final}} =
\begin{cases}
\hat{y}' & \text{if } \sigma(\hat{y}', C'^\star, P) = \text{supported},\\
\hat{y} & \text{otherwise (including the abstention } \texttt{[Unsupported] \{E-none\}}\text{)},
\end{cases}
\tag{6b}
$$

where $D$ is the debate transcript and $\hat{y}', C'^\star$ are obtained from $s'$ with Eq. (5a). The number of
model calls per question is therefore bounded by

$$
\mathrm{calls}(q) \le 2 N R_q + R_q + 1 + 1 .
\tag{6c}
$$

**Code mapping.** (6a): `utils/prompt_utils.verify_answer` with `prompts/madke_x_prompts.verify_prompt`
(`verify_prompt_fever` is defined but never triggered because FEVER is *not\_applicable*), called from
`debate.run_debate` only when `cfg.verify` is set; the attempt, reply, `answer` $= \hat{y}'$, `cited_ids`,
`cited_valid_ids`, `support_status`, `adopted` and `abstained` flags are stored in `trace["verification"]`.
(6b): `record["final_answer"]` is replaced and `record["Summary_verified"]` added only when `adopted` is true;
otherwise the initial `record["final_answer"]` and `trace["initial_support_status"]` are kept. (6c): the counters
`trace["calls"]` (`evidence`, `debater`, `judge`, `summariser`, `verifier`). A failure of the verification call
itself is recorded (`trace["verification"]["ok"] = False`) and the initial answer is kept, because the mandatory
pipeline has already completed. Aggregates: `utils/metrics_calculator.verification_statistics`.

**Symbols.** $s'$ — verification reply; $\hat{y}'$ — verified answer; $C'^\star$ — its validated citations;
$D$ — debate transcript; $E[U_q]$ — passages selected by any agent in any round; $N, R_q$ — agents and executed
rounds.

**Quantitative impact.** Eq. (6a) was triggered for 4–8 of 500 questions per QA dataset (TriviaQA 8, NQ 4,
HotpotQA 7, 2WikiMultiHopQA 7; FEVER 0), i.e. for exactly the questions whose initial answer was unsupported.
The verifier adopted a supported answer for 4, 1, 6 and 3 of them and abstained with `[Unsupported] {E-none}` for
4, 3, 1 and 4; no verified answer was rejected as unsupported after being proposed. The support rate rose from
98.4 → 99.2 %, 99.2 → 99.4 %, 98.6 → 99.8 % and 98.6 → 99.2 %. The mean number of model calls per question is
10.8–11.9 for MADKE-X versus 13.2–13.7 for the baseline on the QA datasets (7.2 vs 9.7 on FEVER), consistent
with $(2N + 1) R_q + 1$ calls plus the rare verification call: the single bounded verification call is more than
offset by the exact-round early stopping of Eq. (3a). Invariant I6
verifies on every record that the verifier is called at most once, only when Eq. (6a) holds, and that a
non-adopted verification never alters the answer or the support status.

---

## Appendix A — Statistical procedures used in the tables

**Exact McNemar test (paired EM).** For questions completed by both systems, $b$ = number correct only under
the baseline and $c$ = number correct only under MADKE-X; the two-sided exact $p$-value is

$$
p = \min\!\Big(1,\; 2 \sum_{i=0}^{\min(b, c)} \binom{b + c}{i}\, 2^{-(b + c)}\Big).
\tag{A1}
$$

`utils/metrics_calculator.mcnemar_exact`, `paired_mcnemar`, `paired_em_vectors` (alignment on `index`).
In the scripted offline run the paired differences are small and not significant ($p$ = 0.230, 0.291, 0.541,
1.000, 0.851), as expected for agents that do not model answer quality; Table 10 applies the same test to live
model outputs.

**Percentile bootstrap confidence intervals.** For a per-question score vector $x_1, \dots, x_m$, 1,000
resamples with replacement are drawn with `random.Random(seed)`, and the 2.5th and 97.5th percentiles of the
resampled means form the 95 % interval (`utils/metrics_calculator.bootstrap_ci`, reported ×100).

**Retrieval ordering quality.** With $h_q$ the 1-based position of the first pool passage such that
$\nu(g) \sqsubseteq \nu(E)$ for some $g \in G_q$ (undefined if none),

$$
\mathrm{cov@}k = \frac{1}{\lvert Q \rvert} \sum_q \mathbb{1}[h_q \le k], \qquad
\mathrm{MRR} = \frac{1}{\lvert Q \rvert} \sum_q \frac{\mathbb{1}[h_q \text{ defined}]}{h_q} .
\tag{A2}
$$

`utils/metrics_calculator.first_hit_rank`, `coverage_and_mrr`; evaluated in `run_experiments.pool_analysis`
for the source order, BM25-only order, RRF order and the legacy 10 + 5 order.

**Token reduction.** $1 - T_{\mathrm{X}} / T_{\mathrm{legacy}}$ with $T$ the summed token count of the pool text
(Table 5) or of all prompts and completions of a question (Table 9); `utils/calculate_tokens.count_tokens`
(tiktoken `cl100k_base` when installed, otherwise a deterministic regex approximation — the offline tables were
produced with the approximation).

## Appendix B — Constraints C1 and C2

**C1 (summariser binding).** The summariser prompt is instantiated with keyword-only arguments, so the
predicate

$$
\mathrm{bind}(q) = \mathbb{1}\big[\text{the last \texttt{Question:} slot of the summariser prompt starts with } q\big]
\tag{B1}
$$

holds for every completed question of MADKE-X (500/500 on each dataset, invariant I7) and for none of the
baseline questions (0/500), in which the question and the debate transcript are exchanged
(`utils/prompt_utils.summary_final_answer(summary_agent, *, question, debate, evidence, prompt, cite, legacy_swap)`;
`trace["summariser_binding_ok"]`).

**C2 (index range).** The evaluated set is $Q = \{ q_i : i_{\mathrm{start}} \le i < i_{\mathrm{end}} \}$ with
defaults $0 \le i < 500$ (`--start_index`, `--end_index` in `parser.py`, `run_experiments.py`), replacing the
hard-coded `[59:]` slice that silently evaluated 441 questions; invariant D1 verifies that the records cover
exactly this range.
