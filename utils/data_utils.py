"""Data loading, integrity checks and the MADKE-X Evidence Governor.

Section 4.2 / Algorithm 2 of the paper is implemented by :func:`build_pool`:

1. ``W[:w] + G[:g]`` pool (paper default 5+5) with per-passage source and
   1-based source rank ``rho(e)``;
2. de-duplication on normalised text (first occurrence kept);
3. question-only BM25 lexical rank ``beta(e)`` and Reciprocal Rank Fusion
   ``S_RRF(q, e) = 1/(60 + rho(e)) + 1/(60 + beta(e))`` with deterministic
   tie-breaking on the original concatenation order.

The raw dataset files under ``data/`` are treated as strictly read-only: they
are only ever opened in read mode and their SHA-256 digests are pinned in
:data:`DATA_SHA256` and verified by :func:`verify_data_checksums`.
"""

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter

from utils.evaluation_utils import normalize_text

# ---------------------------------------------------------------------------
# Dataset registry and integrity
# ---------------------------------------------------------------------------

DATASETS = ("triviaqa", "nq", "hotpotqa", "wikimultihopqa", "fever")

DATA_FILES = {
    "triviaqa": os.path.join("data", "triviaqa", "triviaqa_test_all_retrieval_500.json"),
    "nq": os.path.join("data", "nq", "nq_test_all_retrieval_500.json"),
    "hotpotqa": os.path.join("data", "hotpotqa", "hotpotqa_dev_all_retrieval_500.json"),
    "wikimultihopqa": os.path.join("data", "wikimultihopqa", "wikimultihopqa_dev_all_retrieval_500.json"),
    "fever": os.path.join("data", "fever", "fever_dev_all_retrieval_500.json"),
}

# SHA-256 digests of the five raw dataset files (pinned; see DATA_CHECKSUMS.sha256).
DATA_SHA256 = {
    "triviaqa": "6b6a0c65c21d82757351365d1cf0a0c4d629f783b73273a6249152ad6409666e",
    "nq": "f1494bf6f1f97e2faac8312dbb81efdfe67a5a1e77057b8f6fbb5177addf7f2f",
    "hotpotqa": "839d2c1faa09e491eb4260ce9ff786c370744a5fbc03e2238873fd5f92773594",
    "wikimultihopqa": "6aa08ee7c6639fa72d21119dbca25dec1e2bf31a4f5da9796a99385afbd87ed6",
    "fever": "3fa23b42177c53a2f6e76f65786a238d64445cd3d1606fde7dfe3d923065d279",
}

CLASSIFICATION_DATASETS = ("fever", "feverous")


def task_type(dataset_name):
    """``"classification"`` for claim verification datasets, else ``"extractive"``."""
    return "classification" if dataset_name in CLASSIFICATION_DATASETS else "extractive"


def sha256_file(file_path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(file_path, "rb") as fr:
        for chunk in iter(lambda: fr.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def dataset_path(dataset_name, data_root="."):
    if dataset_name not in DATA_FILES:
        raise KeyError("Unknown dataset '{}'. Known: {}".format(dataset_name, ", ".join(DATASETS)))
    return os.path.join(data_root, DATA_FILES[dataset_name])


def verify_data_checksums(data_root=".", datasets=None, strict=True):
    """Verify the SHA-256 digest of every raw dataset file.

    Returns ``{dataset: {"path", "expected", "actual", "ok"}}``.  With
    ``strict=True`` a mismatch or missing file raises ``RuntimeError``.
    """
    report = {}
    for name in datasets or DATASETS:
        path = dataset_path(name, data_root)
        entry = {"path": path, "expected": DATA_SHA256[name], "actual": None, "ok": False}
        if os.path.isfile(path):
            entry["actual"] = sha256_file(path)
            entry["ok"] = entry["actual"] == entry["expected"]
        report[name] = entry
        if strict and not entry["ok"]:
            raise RuntimeError(
                "Data integrity check failed for '{}' ({}): expected sha256 {} got {}".format(
                    name, path, entry["expected"], entry["actual"]))
    return report


def load_dataset(dataset_name, data_root=".", verify=True):
    """Load one raw dataset (read-only) and optionally verify its checksum."""
    if verify:
        verify_data_checksums(data_root, [dataset_name], strict=True)
    return read_json_data(dataset_path(dataset_name, data_root), verbose=False)


# ---------------------------------------------------------------------------
# Generic IO helpers (original API preserved)
# ---------------------------------------------------------------------------

def read_json_data(file_path, verbose=True):
    """Read json data, [{}, {}, ...]"""
    if verbose:
        print("Load json data from path: {}".format(file_path))
    with open(file_path, "r", encoding="utf-8") as fr:
        dataset = json.load(fr)
    return dataset


def read_jsonl_data(file_path, verbose=True):
    """Read jsonl data."""
    if verbose:
        print("Load jsonl data from path: {}".format(file_path))
    dataset = []
    with open(file_path, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset


def read_csv_data(file_path):
    """Read csv data."""
    print("Load csv data from path: {}".format(file_path))
    dataset = []
    with open(file_path, "r", encoding="utf-8") as fr:
        reader = csv.reader(fr)
        for line in reader:
            dataset.append(line)
    return dataset


def _ensure_parent(file_path):
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def write_json_data(file_path, dataset, verbose=True):
    """Write json data, [{}, {}, ...]"""
    if verbose:
        print("Write json data to path: {}".format(file_path))
    _ensure_parent(file_path)
    with open(file_path, "w", encoding="utf-8") as fw:
        json.dump(dataset, fw, indent=4, ensure_ascii=False)


def append_jsonl(file_path, record):
    _ensure_parent(file_path)
    with open(file_path, "a", encoding="utf-8") as fw:
        fw.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_text(file_path, text):
    _ensure_parent(file_path)
    with open(file_path, "w", encoding="utf-8") as fw:
        fw.write(text)


def print_args(args):
    for key in args.__dict__:
        print(key + ": " + str(args.__dict__[key]))


# ---------------------------------------------------------------------------
# Tokenisation and BM25 (standard library only)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    """Lower-cased alphanumeric tokens (used by BM25)."""
    return _TOKEN_RE.findall((text or "").lower())


class BM25:
    """Okapi BM25 over a list of tokenised documents.

    ``score(query_tokens)`` returns one score per indexed document::

        BM25(q, e) = sum_{t in q} idf(t) * tf(t, e) * (k1 + 1)
                     / (tf(t, e) + k1 * (1 - b + b * |e| / avgdl))

        idf(t) = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )
    """

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = [Counter(toks) for toks in docs_tokens]
        self.doc_lens = [len(toks) for toks in docs_tokens]
        self.n_docs = len(docs_tokens)
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0
        df = Counter()
        for counter in self.docs:
            df.update(counter.keys())
        self.idf = {t: math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}

    def score(self, query_tokens):
        scores = []
        for counter, dl in zip(self.docs, self.doc_lens):
            s = 0.0
            if self.avgdl > 0:
                norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            else:
                norm = self.k1
            for t in query_tokens:
                tf = counter.get(t)
                if not tf:
                    continue
                s += self.idf.get(t, 0.0) * tf * (self.k1 + 1.0) / (tf + norm)
            scores.append(s)
        return scores


def rank_positions(values, higher_is_better=True):
    """1-based position ranks with deterministic tie-breaking on list index."""
    order = sorted(range(len(values)), key=lambda i: ((-values[i] if higher_is_better else values[i]), i))
    ranks = [0] * len(values)
    for pos, i in enumerate(order):
        ranks[i] = pos + 1
    return ranks


# ---------------------------------------------------------------------------
# Algorithm 2: Evidence Governor
# ---------------------------------------------------------------------------

def _passage_contents(item):
    if isinstance(item, dict):
        return item.get("contents", "") or ""
    return item if isinstance(item, str) else str(item)


def _passage_id(item):
    if isinstance(item, dict):
        return item.get("id")
    return None


def _take(items, k, source):
    out = []
    for rank, item in enumerate(list(items or [])[:k], start=1):
        out.append({
            "contents": _passage_contents(item),
            "source": source,
            "source_rank": rank,       # rho(e), 1-based within its source list
            "orig_id": _passage_id(item),
        })
    return out


def build_pool(question, wiki_list, google_list, w=5, g=5, bm25_index=None,
               dedup=True, rrf=True, rrf_k=60):
    """Build the evidence pool for ``question`` (Algorithm 2).

    Parameters
    ----------
    question : str
    wiki_list, google_list : list
        Retrieved passages (dicts with ``contents``) in retrieval order.
    w, g : int
        Number of wiki / google passages taken (paper default 5 + 5).
    bm25_index : BM25 or None
        Optional pre-built index whose IDF statistics are used instead of the
        pool-local statistics (must index exactly the candidate pool in order).
    dedup : bool
        Remove passages whose normalised text was already seen (keep first).
    rrf : bool
        Re-order the pool by decreasing ``S_RRF``; when ``False`` the
        concatenation order is kept (scores are still recorded).
    rrf_k : int
        RRF constant (60 in the paper).

    Returns a list of passage dicts with keys ``index`` (final 0-based pool
    position), ``contents``, ``source``, ``source_rank`` (rho), ``concat_rank``,
    ``bm25_score``, ``bm25_rank`` (beta), ``rrf_score``, ``sha256``, ``orig_id``.
    """
    candidates = _take(wiki_list, w, "wiki") + _take(google_list, g, "google")
    for pos, p in enumerate(candidates, start=1):
        p["concat_rank"] = pos
        p["norm_text"] = normalize_text(p["contents"])

    if dedup:
        seen, kept, removed = set(), [], 0
        for p in candidates:
            if p["norm_text"] in seen:
                removed += 1
                continue
            seen.add(p["norm_text"])
            kept.append(p)
        candidates = kept
    else:
        removed = 0

    q_tokens = tokenize(question)
    docs_tokens = [tokenize(p["contents"]) for p in candidates]
    index = bm25_index if bm25_index is not None else BM25(docs_tokens)
    scores = index.score(q_tokens) if candidates else []
    betas = rank_positions(scores, higher_is_better=True)

    for p, s, beta in zip(candidates, scores, betas):
        p["bm25_score"] = float(s)
        p["bm25_rank"] = beta
        p["rrf_score"] = 1.0 / (rrf_k + p["source_rank"]) + 1.0 / (rrf_k + beta)

    if rrf:
        candidates.sort(key=lambda p: (-p["rrf_score"], p["concat_rank"]))

    pool = []
    for i, p in enumerate(candidates):
        pool.append({
            "index": i,
            "contents": p["contents"],
            "source": p["source"],
            "source_rank": p["source_rank"],
            "concat_rank": p["concat_rank"],
            "bm25_score": p["bm25_score"],
            "bm25_rank": p["bm25_rank"],
            "rrf_score": p["rrf_score"],
            "sha256": sha256_text(p["contents"]),
            "orig_id": p["orig_id"],
        })
    pool_meta = {"candidates": len(_take(wiki_list, w, "wiki")) + len(_take(google_list, g, "google")),
                 "duplicates_removed": removed}
    return pool, pool_meta


def legacy_pool(wiki_list, google_list, w=10, g=5):
    """Original MADKE pool: ``W[:10] + G[:5]`` concatenation, no de-duplication, no re-ranking."""
    pool, meta = build_pool("", wiki_list, google_list, w=w, g=g, dedup=False, rrf=False)
    return pool, meta


def pool_manifest(pool):
    """Audit manifest: ``(index, source, source_rank, sha256)`` per passage."""
    return [{"index": p["index"], "source": p["source"], "source_rank": p["source_rank"],
             "sha256": p["sha256"], "bm25_rank": p.get("bm25_rank"), "rrf_score": p.get("rrf_score")}
            for p in pool]


def select_evidence_lists(data, evidence_type, wiki_k, google_k):
    """Map the original ``--evidence_type`` switch onto (wiki_list, google_list, w, g)."""
    wiki = data.get("wiki_retrieval", []) or []
    google = data.get("google_retrieval", []) or []
    if evidence_type == "all_evidence":
        return wiki, google, wiki_k, google_k
    if evidence_type == "only_wiki_5":
        return wiki, [], wiki_k, 0
    if evidence_type == "only_google_5":
        return [], google, 0, google_k
    return [], [], 0, 0
