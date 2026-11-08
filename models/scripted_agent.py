"""Deterministic scripted agents for ``--mode offline``.

The scripted agent plays every role of the debate (evidence selector, debater,
judge, summariser, verifier) using only lexical heuristics over the text that
appears in its own prompt.  Its purpose is to exercise *every* code path of
the MADKE / MADKE-X pipeline deterministically and in seconds -- parsers,
pool construction, early stopping, citations, support check, verification,
failure handling and metric computation -- without any LLM or API key.

It is NOT a model of LLM quality: EM/F1 obtained offline characterise the
pipeline, not an LLM.  Replies are made deliberately *messy* in a seeded,
reproducible way (duplicate ids, out-of-range ids, more than three ids,
quoted / lower-case verdicts, ``[no-found]`` variants) so that the robustness
of the interface layer is measurable (Tables 3-9).

Determinism: every decision is drawn from ``random.Random(sha256(seed|agent|prompt))``.
"""

import hashlib
import random
import re

from models.base import AgentResult, BaseAgent
from utils.evaluation_utils import bracket_groups, extract_final_answer, normalize_text
from utils.data_utils import tokenize

STOPWORDS = set("""a an the of in on at to for from by with and or is are was were be been being who whom whose which
what when where why how did do does done has have had can could would should will shall may might must this that these
those it its as into than then there their they them he she his her him we you your our i me my not no nor so if
about after before over under between during against also both each few more most other some such only own same too
very s t just don now first last name called known one two three four five six seven eight nine ten many much new old
part number year years made make based used""".split())

_POOL_LINE_RE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_CHOSEN_LINE_RE = re.compile(r"^\((\d+)\)\s*(.*)$")
_CITE_LINE_RE = re.compile(r"^\{E(\d+)\}\s*(.*)$")
_AGENT_BLOCK_RE = re.compile(r"\[Agent (\d+)\]\s*(.*?)(?=\n\[Agent \d+\]|\Z)", re.S)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z][\w'\-]*|\d[\d,./]*")
_CONNECTORS = {"of", "the", "de", "la", "le", "von", "van", "der", "and", "du", "del", "di", "da", "y", "for", "in", "on"}
# capitalised function words that must never start or extend a candidate answer span
_CAP_STOP = {"The", "He", "She", "It", "They", "We", "I", "You", "This", "That", "These", "Those", "There", "His", "Her",
             "Its", "Their", "Our", "In", "On", "At", "By", "For", "From", "With", "After", "Before", "During", "When",
             "Where", "Who", "What", "Which", "How", "Why", "And", "But", "Or", "As", "If", "Then", "So", "However",
             "Although", "While", "Since", "Because", "Therefore", "Thus", "Also", "Both", "Each", "Some", "Many", "Most",
             "Other", "Such", "Only", "Later", "Early", "Following", "According", "Despite", "Although", "Meanwhile"}


def _rng(*parts):
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _content_tokens(text):
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def _parse_listed_passages(text, line_re):
    passages = []
    for line in (text or "").split("\n"):
        m = line_re.match(line.strip())
        if m:
            passages.append((int(m.group(1)), m.group(2).strip()))
    return passages


def _parse_question(text):
    """Last 'Question:' line of a prompt (the examples come first)."""
    qs = re.findall(r"^Question:\s*(.*)$", text or "", flags=re.M)
    return qs[-1].strip() if qs else ""


def _parse_agent_blocks(text):
    return [(int(i), body.strip()) for i, body in _AGENT_BLOCK_RE.findall(text or "")]


def _candidate_spans(text):
    """Capitalised-phrase and numeric spans with their sentence context."""
    spans = []
    for sent in _SENT_SPLIT_RE.split(text):
        words = _WORD_RE.findall(sent)
        i = 0
        while i < len(words):
            w = words[i]
            if w[0].isupper() and w not in _CAP_STOP:
                j = i + 1
                while j < len(words) and ((words[j][0].isupper() and words[j] not in _CAP_STOP) or
                                          (words[j].lower() in _CONNECTORS and j + 1 < len(words)
                                           and words[j + 1][0].isupper() and words[j + 1] not in _CAP_STOP)):
                    j += 1
                span = re.sub(r"'s$", "", " ".join(words[i:j]).strip())
                if span:
                    spans.append((span, sent))
                i = j
            elif w[0].isdigit():
                spans.append((w.strip(",."), sent))
                i += 1
            else:
                i += 1
    return spans


class ScriptedAgent(BaseAgent):
    def __init__(self, name="Agent", seed=0, fail_rate=0.0, dataset_name="triviaqa", messy=True):
        super().__init__(name)
        self.seed = seed
        self.fail_rate = float(fail_rate or 0.0)
        self.dataset_name = dataset_name
        self.task_type = "classification" if dataset_name in ("fever", "feverous") else "extractive"
        self.messy = messy
        self.calls = 0

    # ------------------------------------------------------------------
    def generate(self, messages):
        self.calls += 1
        user = ""
        for m in messages:
            if m.get("role") == "user":
                user = m.get("content", "")
        rng = _rng(self.seed, self.agent_name, user)

        if self.fail_rate > 0 and rng.random() < self.fail_rate:
            return AgentResult(ok=False, text="", error="injected failure (scripted agent, fail_rate={})".format(self.fail_rate), attempts=4)

        low = user.lower()
        if "select evidence from the evidence pool" in low:
            text = self._choose_evidence(user, rng)
        elif "evaluate the consistency" in low:
            text = self._judge(user, rng)
        elif "verification step" in low:
            text = self._verify(user, rng)
        elif "summarize the final answer" in low:
            text = self._summarise(user, rng)
        else:
            text = self._debate(user, rng)
        prompt_tokens = sum(len(tokenize(m.get("content", ""))) for m in messages)
        return AgentResult(ok=True, text=text, attempts=1,
                           usage={"prompt_tokens": prompt_tokens, "completion_tokens": len(tokenize(text))})

    # ------------------------------------------------------------------
    # Evidence selection
    # ------------------------------------------------------------------
    def _choose_evidence(self, user, rng):
        question = _parse_question(user)
        passages = _parse_listed_passages(user, _POOL_LINE_RE)
        pool_size = len(passages)
        q_tokens = set(_content_tokens(question))
        scored = []
        for idx, text in passages:
            overlap = len(q_tokens & set(tokenize(text)))
            scored.append((overlap, idx))
        # agent-specific perspective: Agent_1 prefers google-side passages on ties
        scored.sort(key=lambda t: (-t[0], t[1] if self.agent_name.endswith("0") else -t[1]))
        helpful = [idx for overlap, idx in scored if overlap > 0]

        if not helpful:
            variant = rng.choice(["[No Found]", "[no found]", "[No-Found]", "[NO FOUND]"]) if self.messy else "[No Found]"
            return ("Let's think step by step! None of the passages mention the entities of the question, "
                    "so the evidence pool does not contain the information needed. {}".format(variant))

        top = helpful[:3]
        if not self.messy:
            return "The most helpful evidence is {}.".format(", ".join("[{}]".format(i) for i in top))

        style = rng.random()
        if style < 0.55:
            body = ", ".join("[{}]".format(i) for i in top)
            return "Let's think step by step! The passages that mention the question entities are {}. These are the most helpful.".format(body)
        if style < 0.70:  # duplicates
            body = " ".join("[{}]".format(i) for i in top + top[:1])
            return "Relevant evidence: {}. Passage [{}] is the most important one.".format(body, top[0])
        if style < 0.82:  # more than three ids -> dropped
            extra = [idx for _, idx in scored if idx not in top][:2]
            body = " ".join("[{}]".format(i) for i in top + extra)
            return "Several passages help: {}.".format(body)
        if style < 0.92:  # out-of-range id
            body = " ".join("[{}]".format(i) for i in top[:2] + [pool_size + rng.randint(0, 5)])
            return "Useful evidence: {}.".format(body)
        # comma-separated single group (legacy parser cannot read it)
        return "The helpful passages are [{}].".format(", ".join(str(i) for i in top))

    # ------------------------------------------------------------------
    # Debater
    # ------------------------------------------------------------------
    def _debate(self, user, rng):
        question = _parse_question(user)
        passages = _parse_listed_passages(user, _CHOSEN_LINE_RE)
        other_answers = []
        m = re.search(r"Answers from other Agents:\n(.*?)(?:\n\n|\nHere is your historical answer:)", user, re.S)
        if m:
            for _, body in re.findall(r"\((Agent_\d+)\)\s*(.*?)(?=\(Agent_\d+\)|\Z)", m.group(1), re.S):
                ans = extract_final_answer(body)
                if ans:
                    other_answers.append(ans)
        hist = re.search(r"Here is your historical answer:\s*(.*)", user, re.S)
        own_prev = extract_final_answer(hist.group(1).split("\n\nQuestion:")[0]) if hist else ""

        if self.task_type == "classification":
            answer = self._verdict_label(question, passages, rng)
        else:
            answer = self._propose_answer(question, passages, rng)

        if other_answers:
            evidence_text = normalize_text(" ".join(t for _, t in passages))
            for oa in other_answers:
                if self.task_type == "classification":
                    # converge to the other agent when it holds the stronger (supported) label
                    if answer == "NOT ENOUGH INFO" and oa.upper() in ("SUPPORTS", "REFUTES"):
                        answer = oa.upper()
                elif normalize_text(oa) and normalize_text(oa) in evidence_text:
                    answer = oa  # the other agent's answer is grounded in my own evidence: adopt it
            if own_prev and not answer:
                answer = own_prev
        if not answer:
            answer = own_prev or ("NOT ENOUGH INFO" if self.task_type == "classification" else "unknown")

        cited = " ".join("({})".format(i) for i, _ in passages[:2])
        reasoning = "Let's think step by step! The provided evidence {} is relevant to the question.".format(cited) if passages else \
            "Let's think step by step! No evidence was provided, so the answer relies on prior knowledge."
        if other_answers:
            reasoning += " Considering the other agents' answers, "
            reasoning += "I keep my answer." if answer == own_prev else "I update my answer."
        return "{} Therefore, the answer is [{}].".format(reasoning, answer)

    def _propose_answer(self, question, passages, rng):
        q_norm = normalize_text(question)
        q_tokens = set(_content_tokens(question))
        scores = {}
        first_seen = {}
        display = {}
        for pos, (_, text) in enumerate(passages):
            body = text.split(" - ", 1)[1] if text.startswith('"') and " - " in text else text
            weight = 1.0 / (1.0 + pos)
            for span, sent in _candidate_spans(body):
                key = normalize_text(span)
                if not key or len(key) < 2 or key in q_norm or key in STOPWORDS:
                    continue
                if len(key.split()) > 6:
                    continue
                sent_overlap = len(q_tokens & set(tokenize(sent))) / (len(q_tokens) or 1)
                scores[key] = scores.get(key, 0.0) + weight * (1.0 + 2.0 * sent_overlap)
                first_seen.setdefault(key, (pos, len(first_seen)))
                display.setdefault(key, span)
        if not scores:
            return ""
        ranked = sorted(scores, key=lambda k: (-scores[k], first_seen[k]))
        pick = 0
        if len(ranked) > 1 and rng.random() < 0.3:  # agent-specific disagreement
            pick = 1
        return display[ranked[pick]]

    def _verdict_label(self, claim, passages, rng):
        c_tokens = set(_content_tokens(claim))
        if not c_tokens or not passages:
            return "NOT ENOUGH INFO"
        best_ratio, best_sent = 0.0, ""
        for _, text in passages:
            for sent in _SENT_SPLIT_RE.split(text):
                ratio = len(c_tokens & set(tokenize(sent))) / len(c_tokens)
                if ratio > best_ratio:
                    best_ratio, best_sent = ratio, sent
        hi, lo = (0.6, 0.35) if self.agent_name.endswith("0") else (0.5, 0.3)
        if best_ratio >= hi:
            negated = re.search(r"\b(not|never|no|n't|neither|nor)\b", best_sent.lower()) is not None
            return "REFUTES" if negated else "SUPPORTS"
        if best_ratio >= lo:
            return "SUPPORTS" if rng.random() < 0.8 else "REFUTES"
        return "NOT ENOUGH INFO"

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------
    def _judge(self, user, rng):
        blocks = _parse_agent_blocks(user)
        answers = [normalize_text(extract_final_answer(body)) for _, body in blocks]
        agree = len(answers) > 0 and all(a == answers[0] and a != "" for a in answers)
        verdict = "Yes" if agree else "No"
        listing = " ".join("Agent {} answered [{}].".format(i, extract_final_answer(body) or "none") for i, body in blocks)
        if not self.messy:
            return "{} The answers are {}. [{}]".format(listing, "consistent" if agree else "inconsistent", verdict)
        style = rng.random()
        if style < 0.6:
            return "Let's think step by step! {} The answers are {}. [{}]".format(listing, "consistent" if agree else "inconsistent", verdict)
        if style < 0.75:
            return 'Let\'s think step by step! {} Therefore, the verdict is ["{}"].'.format(listing, verdict)
        if style < 0.9:
            return "{} Verdict: [{}]".format(listing, verdict.lower())
        return "{} Final judgement: [{}.]".format(listing, verdict)

    # ------------------------------------------------------------------
    # Summariser / verifier
    # ------------------------------------------------------------------
    @staticmethod
    def _majority(blocks):
        counts, display = {}, {}
        for _, body in blocks:
            ans = extract_final_answer(body)
            key = normalize_text(ans)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, ans)
        if not counts:
            return ""
        best = sorted(counts, key=lambda k: (-counts[k], list(display).index(k)))[0]
        return display[best]

    def _summarise(self, user, rng):
        blocks = _parse_agent_blocks(user)
        answer = self._majority(blocks)
        cite_passages = _parse_listed_passages(user, _CITE_LINE_RE)
        cite_mode = "{E" in user
        if not answer:
            text = "Let's think step by step! The agents did not provide a clear answer. Therefore, the final answer is []."
            return text + (" {E-none}" if cite_mode else "")
        reply = "Let's think step by step! Based on the agents' responses, the agents converge on the same answer. Therefore, the final answer is [{}].".format(answer)
        if cite_mode:
            na = normalize_text(answer)
            supporting = [i for i, t in cite_passages if na and na in normalize_text(t)]
            if supporting:
                reply += " Supporting evidence: " + " ".join("{{E{}}}".format(i) for i in supporting[:2])
            elif cite_passages and rng.random() < 0.5:
                reply += " Supporting evidence: {{E{}}}".format(cite_passages[0][0])   # over-citation (not grounded)
            else:
                reply += " {E-none}"
        return reply

    def _verify(self, user, rng):
        blocks = _parse_agent_blocks(user)
        passages = _parse_listed_passages(user, _CITE_LINE_RE)
        candidates = []
        for _, body in blocks:
            ans = extract_final_answer(body)
            if ans and normalize_text(ans) not in [normalize_text(c) for c in candidates]:
                candidates.append(ans)
        for cand in candidates:
            nc = normalize_text(cand)
            hits = [i for i, t in passages if nc and nc in normalize_text(t)]
            if hits:
                return "Verification: passage {{E{}}} states the answer. Therefore, the final answer is [{}]. {{E{}}}".format(hits[0], cand, hits[0])
        # fall back to an extractive candidate that IS stated by a listed passage
        question = _parse_question(user)
        if self.task_type == "extractive" and passages:
            proposal = self._propose_answer(question, passages, rng)
            np_ = normalize_text(proposal)
            hits = [i for i, t in passages if np_ and np_ in normalize_text(t)]
            if hits and rng.random() < 0.5:
                return "Verification: passage {{E{}}} states the answer. Therefore, the final answer is [{}]. {{E{}}}".format(hits[0], proposal, hits[0])
        return "Verification: none of the listed passages states the agents' answer. [Unsupported] {E-none}"
