"""Answer normalisation, EM/F1 scoring and the MADKE-X interface-layer parsers.

This module keeps the original SQuAD-style EM/F1 helpers (``normalize_answer``,
``compute_exact``, ``compute_f1``, ``get_em_f1_scores``, ``extract_answer``) and
adds the failure-aware parsers introduced by MADKE-X:

* :func:`parse_evidence_ids`   -- Algorithm 1 (order-preserving evidence ID parser, fix C3)
* :func:`parse_judge_verdict`  -- robust judge verdict parser (fix C4)
* :func:`extract_final_answer` -- last ``[answer]`` group of a summariser reply
* :func:`parse_citations`      -- ``{E#}`` citation parser (Section 4.3)
* :func:`support_check`        -- Algorithm 3 (lexical support check)

The ``*_legacy`` variants reproduce the behaviour of the original MADKE code so
that the unpatched baseline ("MADKE (re-run)") can be executed faithfully.
"""

import re
import string
import warnings
from collections import Counter, namedtuple

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_UNICODE_PUNCT = "‘’“”–—…«»·"
_PUNCT_TO_SPACE = str.maketrans({ch: " " for ch in string.punctuation + _UNICODE_PUNCT})


def normalize_text(s):
    """Lower-case, replace punctuation by spaces and collapse whitespace.

    Used for passage de-duplication (Algorithm 2) and for the lexical support
    check (Algorithm 3).  Punctuation is mapped to a space (rather than deleted)
    so that ``"Campbell-Bannerman"`` and ``"Campbell Bannerman"`` normalise to the
    same string while word boundaries are preserved.
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.lower().translate(_PUNCT_TO_SPACE).split())


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace (SQuAD)."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if s is None:
        s = ""
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_em_f1_scores(examples, references):
    """Compute the exact and f1 scores from the examples and the model predictions."""
    exact_scores = {}
    f1_scores = {}

    if len(examples) != len(references):
        warnings.warn("The length of the prediction and reference are not the same")
        assert len(examples) < len(references), "prediction should be a subset"

    for idx, prediction in examples.items():
        reference = references[idx]
        assert isinstance(reference, list), reference

        exact_scores[idx] = max(compute_exact(a, prediction) for a in reference)
        f1_scores[idx] = max(compute_f1(a, prediction) for a in reference)

    total = len(exact_scores)
    return {
        "exact": 100.0 * sum(exact_scores.values()) / total if total else 0.0,
        "f1": 100.0 * sum(f1_scores.values()) / total if total else 0.0,
        "total": len(examples),
    }


# ---------------------------------------------------------------------------
# Bracket helpers
# ---------------------------------------------------------------------------

BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
NO_FOUND_RE = re.compile(r"no[- ]?found", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")
CITATION_RE = re.compile(r"\{\s*E\s*(?:(-?\s*none)|(\d+))\s*\}", re.IGNORECASE)
_EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def bracket_groups(text):
    """Return the contents of all ``[...]`` groups of ``text`` in written order."""
    if not text:
        return []
    return BRACKET_RE.findall(text.replace("\n", " "))


def extract_answer(response):
    """Original MADKE helper: all bracketed groups of ``response`` (list)."""
    pattern = re.compile(r"\[(.*?)\]")
    response = (response or "").replace("\n", "").replace('"', "")
    predictions = re.findall(pattern, response)
    return predictions


def extract_final_answer(response):
    """Return the LAST ``[answer]`` group of a reply with quotes/whitespace stripped.

    ``{E#}`` citations use curly braces and therefore never collide with the
    answer brackets.  Returns ``""`` when no bracket group is present.
    """
    groups = bracket_groups(response)
    if not groups:
        return ""
    answer = groups[-1].strip().strip("\"'`“”‘’").strip()
    return answer


def extract_choice(response):
    """Extract choice from response of multi-choice questions."""
    pattern = r"\[(\w)\]"
    matches = re.findall(pattern, response or "")
    answer = ""
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            answer = solution
            break
    return answer


# ---------------------------------------------------------------------------
# Algorithm 1: order-preserving evidence ID parser (fix C3)
# ---------------------------------------------------------------------------

EvidenceParse = namedtuple("EvidenceParse", ["ids", "not_found", "ignored", "dropped", "out_of_range"])


def parse_evidence_ids(reply, pool_size, cap=3):
    """Parse evidence ids from an agent reply (Algorithm 1).

    The selection is read as the *final statement* of the reply, the convention the
    pipeline also uses for answers (last ``[answer]`` group) and judge verdicts (last
    ``[Yes]`` / ``[No]`` group).  The selection prompt asks the agent to reason step by
    step, so passages may be mentioned in brackets before the selection is stated.

    Parameters
    ----------
    reply : str
        Raw agent reply.
    pool_size : int
        Number of passages in the evidence pool (valid ids are ``0 .. pool_size-1``).
    cap : int
        Maximum number of ids to keep (``k`` in the paper, default 3).

    Returns
    -------
    EvidenceParse(ids, not_found, ignored, dropped, out_of_range)
        * ``ids``          -- the LAST ``k`` distinct in-range ids of the reply, in written
                              order (each id at the position of its last occurrence)
        * ``not_found``    -- ``True`` iff any bracket group matches ``no[- ]?found``
        * ``ignored``      -- the distinct in-range ids when ``not_found`` is set (else ``[]``)
        * ``dropped``      -- distinct in-range ids mentioned before the kept ones (beyond the cap)
        * ``out_of_range`` -- integers outside ``[0, pool_size)`` in written order

    For every reply the released parser can read (groups that hold exactly one integer,
    ``[No Found]`` spelled exactly so) the delivered SET is the one the release delivers;
    the delivered ORDER is the written order instead of its reverse.
    """
    groups = bracket_groups(reply)
    not_found = False
    mentions, out_of_range = [], []
    seen_out = set()

    for group in groups:
        if NO_FOUND_RE.search(group):
            not_found = True
            continue
        for tok in INT_RE.findall(group):
            n = int(tok)
            if 0 <= n < pool_size:
                mentions.append(n)
            elif n not in seen_out:
                seen_out.add(n)
                out_of_range.append(n)

    # distinct in-range ids, ordered by the position of their LAST occurrence
    distinct, seen = [], set()
    for n in reversed(mentions):
        if n not in seen:
            seen.add(n)
            distinct.append(n)
    distinct.reverse()

    if not_found:
        return EvidenceParse([], True, distinct, [], out_of_range)
    cap = max(int(cap), 0)
    cut = max(len(distinct) - cap, 0)
    return EvidenceParse(distinct[cut:], False, [], distinct[:cut], out_of_range)


def parse_evidence_ids_legacy(reply, pool_size):
    """Reproduce the original MADKE evidence parser (for the unpatched re-run).

    Original behaviour: all ``[...]`` groups are collected with ``re.findall`` on
    the raw reply (groups spanning a newline are missed), the list is
    *reversed*, ``"No Found"`` is matched case-sensitively as a whole group, only
    groups that are exactly one integer string are accepted and the loop breaks
    once a fourth id has been appended (so at most three passages are shown).
    """
    pattern = re.compile(r"\[(.*?)\]")
    choose_ids = re.findall(pattern, reply or "")
    choose_ids.reverse()
    evi_num = [str(i) for i in range(pool_size)]
    if "No Found" in choose_ids:
        return EvidenceParse([], True, [], [], [])
    valid = []
    shown = []
    for evi_id in choose_ids:
        if evi_id not in evi_num or evi_id in valid:
            continue
        valid.append(evi_id)
        if len(valid) > 3:
            break
        shown.append(int(evi_id))
    return EvidenceParse(shown, False, [], [], [])


# ---------------------------------------------------------------------------
# Judge verdict parser (fix C4)
# ---------------------------------------------------------------------------

def parse_judge_verdict(reply):
    """Return ``"yes"``, ``"no"`` or ``"unparsed"`` from the LAST ``[Yes]``/``[No]`` group.

    Matching is case-insensitive and surrounding quotes / punctuation are
    stripped, so ``["Yes"]``, ``[yes.]`` and ``[ No ]`` are all recognised.
    Bracket groups that are not a verdict (e.g. echoed agent answers such as
    ``[Paris]``) are skipped while scanning backwards.
    """
    for group in reversed(bracket_groups(reply)):
        token = _EDGE_PUNCT_RE.sub("", group).lower()
        if token in ("yes", "no"):
            return token
    return "unparsed"


def parse_judge_verdict_legacy(reply):
    """Original behaviour: ``True`` iff some bracket group is exactly ``"Yes"``."""
    pattern = re.compile(r"\[(.*?)\]")
    result = re.findall(pattern, reply or "")
    return "yes" if "Yes" in result else "no"


# ---------------------------------------------------------------------------
# {E#} citations and Algorithm 3: lexical support check
# ---------------------------------------------------------------------------

def parse_citations(reply):
    """Parse ``{E#}`` citations.

    Returns ``(ids, none_flag)`` where ``ids`` are the cited pool indices in
    written order without duplicates and ``none_flag`` is ``True`` when the reply
    contains ``{E-none}`` (explicit abstention).
    """
    ids, seen, none_flag = [], set(), False
    for none_tok, num in CITATION_RE.findall(reply or ""):
        if none_tok:
            none_flag = True
            continue
        n = int(num)
        if n not in seen:
            seen.add(n)
            ids.append(n)
    return ids, none_flag


SUPPORT_STATUSES = ("supported", "cited_but_answer_not_found", "no_citation", "empty_answer", "not_applicable")

# Answers that are verdicts rather than answer spans.  A verdict never occurs in a
# passage as the statement of an answer, so the lexical support check (and every
# coverage measure built on answer strings) is not applicable to them.
VERDICT_ANSWERS = ("yes", "no", "supports", "refutes", "not enough info")


def contains_phrase(norm_phrase, norm_text):
    """True iff ``norm_phrase`` occurs in ``norm_text`` as a sequence of WHOLE words.

    Both arguments must already be normalised with :func:`normalize_text` (words
    separated by single spaces).  Whole-word matching prevents accidental hits
    such as ``"paris"`` inside ``"comparison"`` or ``"5"`` inside ``"1957"``.
    """
    if not norm_phrase:
        return False
    return (" " + norm_phrase + " ") in (" " + norm_text + " ")


def is_verdict_answer(answer):
    """True iff the normalised answer is a verdict (yes / no / FEVER label), not an answer span."""
    return normalize_text(answer) in VERDICT_ANSWERS


def _passage_text(passage):
    if isinstance(passage, dict):
        return passage.get("contents", "")
    return passage if isinstance(passage, str) else str(passage)


def support_check(answer, cited_ids, pool):
    """Algorithm 3: lexical support check.

    Returns one of
      * ``"empty_answer"``               -- normalised ``answer`` is empty
      * ``"no_citation"``                -- ``cited_ids`` is empty
      * ``"supported"``                  -- the normalised answer occurs, as a sequence of
                                            whole words, in the normalised text of at least
                                            one cited passage
      * ``"cited_but_answer_not_found"`` -- otherwise

    Verdict answers (``is_verdict_answer``) are reported as ``"not_applicable"`` by
    the caller (``debate.run_debate``); this function itself is purely lexical.
    """
    norm_answer = normalize_text(answer)
    if not norm_answer:
        return "empty_answer"
    if not cited_ids:
        return "no_citation"
    for i in cited_ids:
        if 0 <= i < len(pool) and contains_phrase(norm_answer, normalize_text(_passage_text(pool[i]))):
            return "supported"
    return "cited_but_answer_not_found"
