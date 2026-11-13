"""Unit tests for the MADKE-X interface layer, evidence governor, metrics and debate loop.

Run with::  python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debate import legacy_config, madke_x_config, run_debate, select_prompt  # noqa: E402
from models.base import AgentResult  # noqa: E402
from models.scripted_agent import ScriptedAgent  # noqa: E402
from utils.data_utils import BM25, build_bm25_index, build_pool, legacy_pool, tokenize  # noqa: E402
from utils.evaluation_utils import (  # noqa: E402
    contains_phrase,
    extract_final_answer,
    is_verdict_answer,
    parse_citations,
    parse_evidence_ids,
    parse_evidence_ids_legacy,
    parse_judge_verdict,
    parse_judge_verdict_legacy,
    support_check,
)
from utils.metrics_calculator import (  # noqa: E402
    expected_random_coverage,
    expected_random_reciprocal_rank,
    has_span_answer,
    mcnemar_sample_size,
    passage_contains_any,
    two_proportion_power,
    two_proportion_sample_size,
    bootstrap_ci,
    compute_co2,
    compute_co2_original,
    compute_cons,
    compute_cons_original,
    compute_late_agreement,
    compute_em_f1,
    mcnemar_exact,
    paired_mcnemar,
)
from utils.prompt_utils import summary_final_answer  # noqa: E402


class TestAlgorithm1(unittest.TestCase):
    def test_final_statement_dedup_cap_bounds(self):
        # the selection is the final statement: the last three distinct ids, in written order
        p = parse_evidence_ids("[2] then [0], [7] and [2] again, [12], [1] [5]", pool_size=10, cap=3)
        self.assertEqual(p.ids, [2, 1, 5])
        self.assertFalse(p.not_found)
        self.assertEqual(p.dropped, [0, 7])
        self.assertEqual(p.out_of_range, [12])
        self.assertEqual(p.ignored, [])
        # passages discussed first, selection last
        p = parse_evidence_ids("[1] and [5] are off topic. Helpful: [2] [0] [7]", 10)
        self.assertEqual((p.ids, p.dropped), ([2, 0, 7], [1, 5]))
        # an id that is repeated keeps the position of its last occurrence
        p = parse_evidence_ids("[2] is the key passage. Selection: [2] [0] [7]", 10)
        self.assertEqual((p.ids, p.dropped), ([2, 0, 7], []))
        self.assertEqual(parse_evidence_ids("[4] [6]", 10, cap=0).dropped, [4, 6])

    def test_same_passages_as_the_release_in_written_order(self):
        # whenever the released parser can read the reply, the delivered SET is identical
        for reply in ("[2], [0] and [7]", "[2] [0] [7] [1] [5]", "[1] and [5] are off topic. Helpful: [2] [0] [7]",
                      "[2] is the key passage. Selection: [2] [0] [7]", "[2] [0] [12]", "[4]", "no brackets"):
            new, old = parse_evidence_ids(reply, 10).ids, parse_evidence_ids_legacy(reply, 10).ids
            self.assertEqual(new, list(reversed(old)), reply)

    def test_boundary_n_equals_pool_size(self):
        p = parse_evidence_ids("[9] [10]", pool_size=10)
        self.assertEqual(p.ids, [9])
        self.assertEqual(p.out_of_range, [10])

    def test_no_found_variants(self):
        for reply in ("[No Found]", "[no found]", "[NO-FOUND]", "[3] ... [no-found]"):
            p = parse_evidence_ids(reply, 10)
            self.assertTrue(p.not_found, reply)
            self.assertEqual(p.ids, [])
        self.assertEqual(parse_evidence_ids("[3] [no-found]", 10).ignored, [3])

    def test_comma_group_and_negative(self):
        self.assertEqual(parse_evidence_ids("[2, 0, 7]", 10).ids, [2, 0, 7])
        self.assertEqual(parse_evidence_ids("[-1] [0]", 10).out_of_range, [-1])

    def test_legacy_parser_reverses_and_loses_groups(self):
        self.assertEqual(parse_evidence_ids_legacy("[2] [0] [7]", 10).ids, [7, 0, 2])
        self.assertEqual(parse_evidence_ids_legacy("[2, 0, 7]", 10).ids, [])
        self.assertFalse(parse_evidence_ids_legacy("[no found]", 10).not_found)


class TestJudgeAndAnswerParsers(unittest.TestCase):
    def test_last_bracket_with_quotes_and_case(self):
        self.assertEqual(parse_judge_verdict('Agent 0 said [Paris]. Verdict ["Yes"].'), "yes")
        self.assertEqual(parse_judge_verdict("[yes]"), "yes")
        self.assertEqual(parse_judge_verdict("[No.]"), "no")
        self.assertEqual(parse_judge_verdict("[Yes] but finally [No]"), "no")
        self.assertEqual(parse_judge_verdict("no brackets at all"), "unparsed")
        self.assertEqual(parse_judge_verdict_legacy('["Yes"]'), "no")

    def test_final_answer_and_citations(self):
        self.assertEqual(extract_final_answer('[Agent 0] said [x]. Final answer is ["Chicago"]. {E0} {E3}'), "Chicago")
        self.assertEqual(parse_citations("{E0} {E3} {E0} {E-none} {e 7}"), ([0, 3, 7], True))
        self.assertEqual(extract_final_answer("nothing"), "")


class TestAlgorithm3(unittest.TestCase):
    pool = [{"contents": "The Rotary Club was founded in Chicago, Illinois, in 1905."}, {"contents": "Unrelated text."}]

    def test_statuses(self):
        self.assertEqual(support_check("Chicago", [0], self.pool), "supported")
        self.assertEqual(support_check("Chicago", [1], self.pool), "cited_but_answer_not_found")
        self.assertEqual(support_check("Chicago", [], self.pool), "no_citation")
        self.assertEqual(support_check("  ", [0], self.pool), "empty_answer")
        self.assertEqual(support_check("Chicago", [99], self.pool), "cited_but_answer_not_found")

    def test_normalisation_is_punctuation_insensitive(self):
        pool = [{"contents": "Sir Henry Campbell-Bannerman became Prime Minister."}]
        self.assertEqual(support_check("Campbell Bannerman", [0], pool), "supported")

    def test_whole_word_matching(self):
        pool = [{"contents": "A comparison of the two towns was published in 1957."}]
        self.assertEqual(support_check("Paris", [0], pool), "cited_but_answer_not_found")   # not inside "comparison"
        self.assertEqual(support_check("5", [0], pool), "cited_but_answer_not_found")       # not inside "1957"
        self.assertEqual(support_check("1957", [0], pool), "supported")
        self.assertEqual(support_check("two towns", [0], pool), "supported")
        self.assertTrue(contains_phrase("two towns", "a comparison of the two towns"))
        self.assertFalse(contains_phrase("", "anything"))

    def test_verdict_answers(self):
        for a in ("yes", "No.", "SUPPORTS", "Refutes", "NOT ENOUGH INFO"):
            self.assertTrue(is_verdict_answer(a), a)
        self.assertFalse(is_verdict_answer("Chicago"))
        self.assertFalse(has_span_answer(["yes"]))
        self.assertTrue(has_span_answer(["Chicago", "City of Chicago"]))
        self.assertTrue(passage_contains_any("Founded in Chicago, Illinois.", ["chicago"]))
        self.assertFalse(passage_contains_any("A comparison of towns.", ["Paris"]))


class TestAlgorithm2(unittest.TestCase):
    def setUp(self):
        self.wiki = [{"id": "w%d" % i, "contents": t} for i, t in enumerate([
            "Paris is the capital of France.", "Berlin is the capital of Germany.", "Duplicate text here.",
            "Madrid is the capital of Spain.", "Rome is the capital of Italy.", "Sixth wiki passage not used."])]
        self.google = [{"id": i, "contents": t} for i, t in enumerate([
            "duplicate   TEXT here!", "France's capital city is Paris, on the Seine.", "Lisbon is in Portugal."])]

    def test_pool_config_dedup_ranks(self):
        pool, meta = build_pool("What is the capital of France?", self.wiki, self.google, w=5, g=5)
        self.assertEqual(meta["candidates"], 8)
        self.assertEqual(meta["duplicates_removed"], 1)
        self.assertEqual(len(pool), 7)
        texts = [p["contents"] for p in pool]
        self.assertIn("Duplicate text here.", texts)
        self.assertNotIn("duplicate   TEXT here!", texts)   # first occurrence kept
        self.assertEqual([p["index"] for p in pool], list(range(7)))
        for p in pool:
            self.assertIn(p["source"], ("wiki", "google"))
            self.assertGreaterEqual(p["source_rank"], 1)
            self.assertAlmostEqual(p["rrf_score"], 1 / (60 + p["source_rank"]) + 1 / (60 + p["bm25_rank"]))
        scores = [p["rrf_score"] for p in pool]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # the two passages about Paris/France must lead
        self.assertTrue(all("Paris" in t for t in texts[:2]))

    def test_deterministic_tie_break(self):
        pool_a, _ = build_pool("zzz", self.wiki, self.google)
        pool_b, _ = build_pool("zzz", self.wiki, self.google)
        self.assertEqual([p["sha256"] for p in pool_a], [p["sha256"] for p in pool_b])

    def test_legacy_pool(self):
        pool, meta = legacy_pool(self.wiki, self.google)
        self.assertEqual(len(pool), 9)
        self.assertEqual(meta["duplicates_removed"], 0)

    def test_bm25(self):
        idx = BM25([tokenize("Paris France capital"), tokenize("Berlin Germany capital")])
        s = idx.score(tokenize("France"))
        self.assertGreater(s[0], s[1])

    def test_dataset_level_index(self):
        dataset = [{"question": "q1", "wiki_retrieval": self.wiki, "google_retrieval": self.google},
                   {"question": "q2", "wiki_retrieval": [{"contents": "The capital of Peru is Lima."}] * 7,
                    "google_retrieval": [{"contents": "Capital cities of the world."}]}]
        idx = build_bm25_index(dataset, w=5, g=5)
        self.assertEqual(idx.n_docs, 5 + 3 + 5 + 1)                      # first w + first g passages of every question
        with self.assertRaises(ValueError):
            idx.score(tokenize("capital"))                               # reference index keeps statistics only
        # "capital" occurs in most reference passages, "seine" in one: the rare term must weigh more
        self.assertGreater(idx.idf["seine"], idx.idf["capital"])
        pool, meta = build_pool("Which capital lies on the Seine?", self.wiki, self.google, bm25_index=idx)
        self.assertEqual(meta["bm25_reference"], {"scope": "dataset", "n_passages": 14})
        seine = [p for p in pool if "Seine" in p["contents"]][0]
        self.assertEqual(seine["bm25_rank"], 1)                          # best lexical match
        self.assertLessEqual(seine["index"], 1)                          # fused with source rank: first or second
        _, meta_local = build_pool("Which capital lies on the Seine?", self.wiki, self.google)
        self.assertEqual(meta_local["bm25_reference"]["scope"], "pool")


def _rec(idx, verdicts, status="completed", answer="x", gold=("x",), stop_round=None):
    return {"index": idx, "status": status, "final_answer": answer if status == "completed" else None, "answer": list(gold),
            "stop_round": len(verdicts) - 1 if stop_round is None else stop_round,
            "trace": {"judge_verdicts": verdicts}, **{"Round_%d" % i: {"Judge": "[%s]" % v.capitalize()} for i, v in enumerate(verdicts)}}


class TestMetrics(unittest.TestCase):
    def test_cons_co2(self):
        recs = [_rec(0, ["yes"]), _rec(1, ["no", "yes"]), _rec(2, ["no", "no", "yes"]), _rec(3, ["no", "no", "no"]),
                _rec(4, ["no"], status="failed")]
        self.assertAlmostEqual(compute_cons(recs)["cons"], 3 / 4)
        self.assertAlmostEqual(compute_co2(recs)["co2"], 3 / 4)               # all three consistent answers are correct
        self.assertAlmostEqual(compute_late_agreement(recs)["late_agreement"], 2 / 4)
        # the formula of the release never counts an agreement of the final round (dict membership bug)
        self.assertAlmostEqual(compute_cons_original(recs, max_round=3)["cons_original"], 2 / 4)
        self.assertAlmostEqual(compute_co2_original(recs, max_round=3)["co2_original"], 2 / 4)

    def test_co2_is_consistent_and_correct(self):
        recs = [_rec(0, ["yes"]), _rec(1, ["no", "yes"], answer="wrong"), _rec(2, ["no", "no", "yes"]),
                _rec(3, ["no", "no", "no"])]
        self.assertAlmostEqual(compute_cons(recs)["cons"], 3 / 4)
        self.assertAlmostEqual(compute_co2(recs)["co2"], 2 / 4)               # question 1 is consistent but wrong
        self.assertAlmostEqual(compute_co2_original(recs)["co2_original"], 1 / 4)
        # robust reading of an unpatched record: the pipeline missed the agreement, the judge reply states it
        missed = _rec(4, ["no", "no", "no"])
        missed["trace"]["rounds"] = [{"judge_verdict": "no", "judge_verdict_robust": v} for v in ("no", "yes", "yes")]
        self.assertAlmostEqual(compute_cons([missed])["cons"], 0.0)
        self.assertAlmostEqual(compute_cons([missed], robust=True)["cons"], 1.0)
        self.assertAlmostEqual(compute_co2([missed], robust=True)["co2"], 1.0)

    def test_em_f1_excludes_failed(self):
        recs = [_rec(0, ["yes"], answer="x"), _rec(1, ["yes"], answer="y"), _rec(2, ["yes"], status="failed")]
        m = compute_em_f1(recs)
        self.assertEqual(m["total"], 2)
        self.assertEqual(m["n_failed"], 1)
        self.assertAlmostEqual(m["exact"], 50.0)

    def test_mcnemar_and_bootstrap(self):
        self.assertAlmostEqual(mcnemar_exact(5, 1), 2 * 7 / 64)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(paired_mcnemar([1, 1, 1, 0, 0], [0, 1, 1, 1, 0])["p_value"], 1.0)
        a = bootstrap_ci([0, 1, 1, 0, 1], n_boot=200, seed=1)
        b = bootstrap_ci([0, 1, 1, 0, 1], n_boot=200, seed=1)
        self.assertEqual(a, b)
        self.assertLessEqual(a[0], 0.6)
        self.assertGreaterEqual(a[1], 0.6)


class TestDebateLoop(unittest.TestCase):
    data = {"question": "In which city were Rotary Clubs set up in 1905?", "answer": ["Chicago"],
            "wiki_retrieval": [{"id": "1", "contents": "\"Rotary International\"\nThe first Rotary Club was formed when attorney Paul Harris called together a meeting of businessmen in Chicago, Illinois, in 1905."},
                               {"id": "2", "contents": "\"Chicago\"\nChicago is the most populous city in Illinois."},
                               {"id": "3", "contents": "\"Service club\"\nA service club is a voluntary non-profit organization."}],
            "google_retrieval": [{"id": 0, "contents": "Rotary Clubs were set up in 1905 in the city of Chicago."},
                                 # same text as wiki passage 3 up to case / whitespace / punctuation -> removed by dedup
                                 {"id": 1, "contents": "\"service club\"   A SERVICE CLUB is a voluntary, non profit organization"}]}

    def _agents(self, fail_rate=0.0):
        mk = lambda n: ScriptedAgent(n, seed=3, fail_rate=fail_rate, dataset_name="triviaqa")  # noqa: E731
        return [mk("Agent_0"), mk("Agent_1")], mk("Judge"), mk("Summarizer")

    def test_madke_x_record_has_trace(self):
        debaters, judge, summ = self._agents()
        rec = run_debate(self.data, debaters, judge, summ, select_prompt("triviaqa"), madke_x_config("triviaqa"), index=0)
        self.assertEqual(rec["status"], "completed")
        tr = rec["trace"]
        self.assertEqual(tr["pool_size"], 4)   # one google duplicate of the wiki passage removed
        self.assertEqual(len(tr["pool_manifest"]), 4)
        self.assertTrue(tr["summariser_binding_ok"])
        self.assertIn(tr["support_status"], ("supported", "cited_but_answer_not_found", "no_citation", "empty_answer"))
        self.assertLessEqual(tr["calls"]["verifier"], 1)
        for rt in tr["rounds"]:
            for a in rt["agents"].values():
                self.assertLessEqual(len(a["selected_ids"]), 3)
                self.assertTrue(all(0 <= i < 4 for i in a["selected_ids"]))
                self.assertEqual(a["selected_ids"], a["intended_ids"])   # delivered = what the script meant to select
        if "yes" in tr["judge_verdicts"]:
            self.assertEqual(tr["judge_verdicts"].index("yes"), len(tr["judge_verdicts"]) - 1)

    def test_legacy_swaps_summariser_arguments(self):
        debaters, judge, summ = self._agents()
        rec = run_debate(self.data, debaters, judge, summ, select_prompt("triviaqa"), legacy_config("triviaqa"), index=0)
        self.assertEqual(rec["status"], "completed")
        self.assertFalse(rec["trace"]["summariser_binding_ok"])
        self.assertEqual(rec["trace"]["pool_size"], 5)

    def test_failure_marks_question_failed(self):
        debaters, judge, summ = self._agents(fail_rate=1.0)
        rec = run_debate(self.data, debaters, judge, summ, select_prompt("triviaqa"), madke_x_config("triviaqa"), index=0)
        self.assertEqual(rec["status"], "failed")
        self.assertIsNone(rec["final_answer"])
        self.assertEqual(rec["error"]["stage"], "evidence")
        self.assertEqual(compute_em_f1([rec])["total"], 0)

    def test_summary_keyword_binding(self):
        class Echo:
            agent_name = "Echo"

            def __call__(self, messages):
                return AgentResult(ok=True, text=messages[-1]["content"])
        res = summary_final_answer(Echo(), question="Q?", debate="[Agent 0] a\n", prompt=select_prompt("nq"))
        self.assertTrue(res["question_slot_is_question"])
        self.assertIn("Question: Q?", res["reply"])
        with self.assertRaises(TypeError):
            summary_final_answer(Echo(), "Q?", "[Agent 0] a\n", select_prompt("nq"))  # positional use is rejected


class TestAnalysisHelpers(unittest.TestCase):
    def test_random_order_expectations(self):
        self.assertAlmostEqual(expected_random_coverage(10, 2, 3), 1 - 56 / 120)
        self.assertEqual(expected_random_coverage(10, 0, 3), 0.0)
        self.assertEqual(expected_random_coverage(3, 1, 5), 1.0)
        self.assertAlmostEqual(expected_random_reciprocal_rank(6, 2), 0.58)       # checked by enumeration
        self.assertAlmostEqual(expected_random_reciprocal_rank(4, 4), 1.0)

    def test_power_functions(self):
        n = two_proportion_sample_size(0.736, 0.648)
        self.assertTrue(425 < n < 440)
        self.assertAlmostEqual(two_proportion_power(0.736, 0.648, n), 0.8, places=2)
        self.assertTrue(1000 < mcnemar_sample_size(0.044, 0.25) < 1025)

    def test_interface_format_tables(self):
        from run_experiments import interface_format_tables
        fmt = interface_format_tables()
        ev = {e["format"]: e for e in fmt["evidence"]}
        self.assertEqual(ev["separate brackets"]["x_delivered"], [2, 0, 7])
        self.assertEqual(ev["separate brackets"]["released_delivered"], [7, 0, 2])
        self.assertEqual(ev["list in one bracket"]["released_delivered"], [])
        self.assertEqual(ev["list in one bracket"]["x_delivered"], [2, 0, 7])
        self.assertEqual(ev["passages discussed first, selection last"]["x_delivered"], [2, 0, 7])
        self.assertEqual(ev["passages discussed first, selection last"]["x_dropped"], [1, 5])
        self.assertEqual(ev["more than three listed"]["released_delivered"], [5, 1, 7])
        self.assertEqual(ev["more than three listed"]["x_delivered"], [7, 1, 5])
        self.assertEqual(ev["more than three listed"]["x_dropped"], [2, 0])
        self.assertEqual(ev["identifier outside the pool"]["x_out_of_range"], [12])
        self.assertTrue(ev["lower-case not-found marker"]["x_not_found"])
        self.assertEqual(ev["lower-case not-found marker"]["released_delivered"], [3])
        for v in fmt["verdict"]:
            self.assertTrue(v["x_correct"], v)
        self.assertEqual(sum(1 for v in fmt["verdict"] if v["released_correct"]), 1)


class TestVerdictAnswersAndResume(unittest.TestCase):
    data = {"question": "Are both rivers in Europe?", "answer": ["yes"],
            "wiki_retrieval": [{"id": "1", "contents": "The Rhine is a river in Europe."},
                               {"id": "2", "contents": "The Danube is a river in Europe."}],
            "google_retrieval": [{"id": 0, "contents": "Both the Rhine and the Danube flow through Europe."}]}

    class Fixed:
        """Agent that always selects passage 0 and answers / summarises with a fixed verdict."""

        def __init__(self, name):
            self.agent_name = name
            self.calls = []

        def __call__(self, messages):
            user = messages[-1]["content"].lower()
            self.calls.append(user[:40])
            if "select evidence from the evidence pool" in user:
                return AgentResult(ok=True, text="[0]")
            if "evaluate the consistency" in user:
                return AgentResult(ok=True, text="[Yes]")
            if "verification step" in user:
                raise AssertionError("verification must not be called for a verdict answer")
            return AgentResult(ok=True, text="Both are in Europe. Therefore, the answer is [yes]. {E0}")

    def test_verdict_answer_is_not_applicable_and_not_verified(self):
        mk = self.Fixed
        rec = run_debate(self.data, [mk("Agent_0"), mk("Agent_1")], mk("Judge"), mk("Summarizer"),
                         select_prompt("hotpotqa"), madke_x_config("hotpotqa"), index=0)
        self.assertEqual(rec["status"], "completed")
        self.assertEqual(rec["final_answer"], "yes")
        self.assertEqual(rec["trace"]["support_status"], "not_applicable")
        self.assertFalse(rec["trace"]["verification"]["attempted"])
        self.assertEqual(rec["trace"]["calls"]["verifier"], 0)
        self.assertEqual(rec["trace"]["config"]["code_version"], __import__("debate").CODE_VERSION)

    def test_checkpoints_of_another_version_are_not_resumed(self):
        import json
        import tempfile
        from run_experiments import run_records
        mk = self.Fixed
        cfg = madke_x_config("hotpotqa")
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "madke_x.jsonl")
            stale = {"index": 0, "status": "completed", "final_answer": "old", "trace": {"config": {"code_version": "0.0.1"}}}
            with open(ckpt, "w", encoding="utf-8") as fw:
                fw.write(json.dumps(stale) + "\n")
            agents = ([mk("Agent_0"), mk("Agent_1")], mk("Judge"), mk("Summarizer"))
            recs = run_records([self.data], [0], cfg, select_prompt("hotpotqa"), agents, checkpoint=ckpt)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["final_answer"], "yes")                      # re-run, not resumed
            self.assertTrue(any(".stale-" in f for f in os.listdir(tmp)))
            recs2 = run_records([self.data], [0], cfg, select_prompt("hotpotqa"), agents, checkpoint=ckpt)
            self.assertEqual(recs2[0]["final_answer"], "yes")                     # now resumed from the new checkpoint
            with open(ckpt, encoding="utf-8") as fr:
                self.assertEqual(len(fr.readlines()), 1)


if __name__ == "__main__":
    unittest.main()
