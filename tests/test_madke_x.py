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
from utils.data_utils import BM25, build_pool, legacy_pool, tokenize  # noqa: E402
from utils.evaluation_utils import (  # noqa: E402
    extract_final_answer,
    parse_citations,
    parse_evidence_ids,
    parse_evidence_ids_legacy,
    parse_judge_verdict,
    parse_judge_verdict_legacy,
    support_check,
)
from utils.metrics_calculator import (  # noqa: E402
    bootstrap_ci,
    compute_co2,
    compute_cons,
    compute_cons_original,
    compute_em_f1,
    mcnemar_exact,
    paired_mcnemar,
)
from utils.prompt_utils import summary_final_answer  # noqa: E402


class TestAlgorithm1(unittest.TestCase):
    def test_order_dedup_cap_bounds(self):
        p = parse_evidence_ids("[2] then [0], [7] and [2] again, [12], [1] [5]", pool_size=10, cap=3)
        self.assertEqual(p.ids, [2, 0, 7])
        self.assertFalse(p.not_found)
        self.assertEqual(p.dropped, [1, 5])
        self.assertEqual(p.out_of_range, [12])
        self.assertEqual(p.ignored, [])

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


def _rec(idx, verdicts, status="completed", answer="x", gold=("x",), stop_round=None):
    return {"index": idx, "status": status, "final_answer": answer if status == "completed" else None, "answer": list(gold),
            "stop_round": len(verdicts) - 1 if stop_round is None else stop_round,
            "trace": {"judge_verdicts": verdicts}, **{"Round_%d" % i: {"Judge": "[%s]" % v.capitalize()} for i, v in enumerate(verdicts)}}


class TestMetrics(unittest.TestCase):
    def test_cons_co2(self):
        recs = [_rec(0, ["yes"]), _rec(1, ["no", "yes"]), _rec(2, ["no", "no", "yes"]), _rec(3, ["no", "no", "no"]),
                _rec(4, ["no"], status="failed")]
        self.assertAlmostEqual(compute_cons(recs)["cons"], 3 / 4)
        self.assertAlmostEqual(compute_co2(recs)["co2"], 2 / 4)
        # original formula never counts final-round agreement (dict membership bug)
        self.assertAlmostEqual(compute_cons_original(recs[:4], max_round=3)["cons_original"], 2 / 4)

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


if __name__ == "__main__":
    unittest.main()
