"""Counterexamples for failures, abstentions and unlabeled audit inputs."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import audit_agent_v2 as auditor
import run_3way_comparison as threeway
import run_improvement_benchmark as improvement
import scaled_benchmark as scaled
from audit_outcomes import (
    guard_grounded_decision, label_from_value, parse_audit_response,
    prediction_from_status, score_outcomes,
)


CHUNK = {"section": "1910.22", "citation": "29 CFR 1910.22", "heading": "Walking surfaces", "text": "Keep walking surfaces clean."}
VERIFIER = SimpleNamespace(valid_sections={"1910.22"})


def rows():
    return [{"Log_ID": f"case-{i}", "Equipment_ID": "F1", "Equipment_Type": "Forklift",
             "Location": "Walkway", "Operator_Shift_Hours": "4",
             "Reported_Incident": "Synthetic observation", "Is_Violation": str(i % 2),
             "Violation_Type": "housekeeping" if i % 2 else "none",
             "Expected_Citation": "1910.22" if i % 2 else "NONE"} for i in range(8)]


def memory():
    return SimpleNamespace(episodes=[], retrieve_contrastive_precedents=lambda *a, **k: {},
                           retrieve_pitfalls=lambda *a, **k: [])


def pillars():
    return SimpleNamespace(gaps={}, pillars={}, pillar_vectors={})


class EmptyResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def json(self):
        return {"message": {"content": ""}}


class FakeSession(EmptyResponse):
    def post(self, *args, **kwargs):
        return EmptyResponse()


class OutcomeContractTests(unittest.TestCase):
    def test_binary_decisions_only(self):
        self.assertEqual(prediction_from_status("CLEAR"), 0)
        self.assertEqual(prediction_from_status("VIOLATION"), 1)
        for status in ("REVIEW", "UNPARSED", "INPUT_ERROR", "INFERENCE_ERROR", "CLEARANCE", None, ""):
            with self.subTest(status=status):
                self.assertIsNone(prediction_from_status(status))

    def test_absent_labels_remain_absent_and_invalid_labels_fail(self):
        for value in (None, "", "  ", float("nan")):
            self.assertIsNone(label_from_value(value))
        for value in (0, 1, "0", "1", " 1 ", 1.0, np.int64(1), np.float64(0)):
            self.assertIn(label_from_value(value), (0, 1))
        for value in (2, -1, "negative", float("inf")):
            with self.assertRaises(ValueError):
                label_from_value(value)

    def test_outage_is_neither_clearance_nor_false_alarm(self):
        result = score_outcomes([{"status": "UNPARSED", "predicted": 0, "actual": label} for label in [0, 1] * 4])
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["true_negatives"], 0)
        self.assertEqual(result["false_positives"], 0)
        self.assertEqual(result["false_negatives"], 0)
        self.assertEqual(result["unresolved_positive_cases"], 4)
        self.assertEqual(result["unresolved_negative_cases"], 4)
        self.assertEqual(result["inference_error_count"], 8)
        self.assertEqual(result["strict_errors"], 8)
        self.assertEqual(result["automation_coverage"], 0)
        self.assertIsNone(result["decided_accuracy"])

    def test_unlabeled_event_cannot_earn_accuracy(self):
        result = auditor.compute_metrics([{"status": "CLEAR", "predicted": 0, "actual": None}])
        self.assertEqual(result["n_labeled"], 0)
        self.assertEqual(result["n_unlabeled"], 1)
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["true_negatives"], 0)

    def test_review_and_failures_have_separate_denominators(self):
        result = score_outcomes([
            {"status": "VIOLATION", "actual": 1},
            {"status": "VIOLATION", "actual": 0},
            {"status": "CLEAR", "actual": 1},
            {"status": "CLEAR", "actual": 0},
            {"status": "UNPARSED", "actual": 1},
            {"status": "REVIEW", "actual": 0},
            {"status": "CLEAR", "actual": None},
            {"status": "INPUT_ERROR", "actual": None},
        ])
        self.assertEqual(result["n_labeled"], 6)
        self.assertEqual(result["n_unlabeled"], 2)
        self.assertEqual(result["accuracy"], .3333)
        self.assertEqual(result["decided_accuracy"], .5)
        self.assertEqual(result["labeled_decision_coverage"], .6667)
        self.assertEqual(result["automation_coverage"], .625)
        self.assertEqual(result["review_count"], 1)
        self.assertEqual(result["input_error_count"], 1)
        self.assertEqual(result["inference_error_count"], 1)
        self.assertEqual(result["missed_positive_cases"], 2)
        self.assertEqual(result["recall"], .3333)

    def test_empty_cohort_has_no_performance_claim(self):
        result = score_outcomes([])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["automation_coverage"])

    def test_review_reply_and_raw_response_are_preserved(self):
        raw = "STATUS: REVIEW\nCITATION: NONE\nREASON: Required fact is absent."
        parsed = parse_audit_response(raw)
        self.assertEqual(parsed["status"], "REVIEW")
        self.assertEqual(parsed["raw_response"], raw)

    def test_crlf_and_trailing_whitespace_preserve_valid_reply(self):
        raw = "STATUS: CLEAR  \r\nCITATION: NONE\r\nREASON: Controls confirmed.  \r\n\r\n"
        self.assertEqual(parse_audit_response(raw)["status"], "CLEAR")
        self.assertEqual(parse_audit_response(raw)["raw_response"], raw)

    def test_paired_comparison_joins_ids_instead_of_position(self):
        left = [{"log_id":"A", "status":"CLEAR", "actual":0}, {"log_id":"B", "status":"CLEAR", "actual":1}]
        right = [{"log_id":"B", "status":"VIOLATION", "actual":1}, {"log_id":"A", "status":"CLEAR", "actual":0}]
        result = scaled.exact_mcnemar(left, right)
        self.assertEqual(result["left_only_correct"], 0)
        self.assertEqual(result["right_only_correct"], 1)
        self.assertEqual(result["discordant_pairs"], 1)

    def test_paired_comparison_rejects_missing_duplicate_or_unlabeled_cases(self):
        valid = [{"log_id":"A", "status":"CLEAR", "actual":0}]
        invalid = [[], valid * 2, [{"log_id":"B", "actual":0}],
                   [{"actual":0}], [{"log_id":"A", "actual":None}], [{"log_id":"A", "actual":1}]]
        for cases in invalid:
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                scaled.exact_mcnemar(valid, cases)

    def test_partial_or_conflicting_reply_is_not_a_clearance(self):
        for text in (
            "STATUS: CLEAR",
            "STATUS: CLEARANCE\nCITATION: NONE\nREASON: text",
            "STATUS: CLEAR\nCITATION: NONE\nREASON:",
            "STATUS: CLEAR\nSTATUS: VIOLATION\nCITATION: NONE\nREASON: text",
            "STATUS: CLEAR\nCITATION: UNKNOWN\nREASON: text",
            "STATUS: VIOLATION\nCITATION: 1910.22 or 1910.333\nREASON: text",
            "STATUS: VIOLATION\nCITATION: not 1910.22\nREASON: text",
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_audit_response(text)["status"], "UNPARSED")

    def test_explicit_authority_prefix_and_subsection_parse(self):
        for cite, expected in (("29 CFR §1910.22(a)(1)", "1910.22"), ("49 U.S.C. 21103", "21103")):
            parsed = parse_audit_response(f"STATUS: VIOLATION\nCITATION: {cite}\nREASON: Synthetic check.")
            self.assertEqual(parsed["status"], "VIOLATION")
            self.assertEqual(parsed["citation"], expected)

    def test_example_or_contradiction_outside_frame_cannot_earn_clearance(self):
        replies = (
            "Example response only:\nSTATUS: CLEAR\nCITATION: NONE\nREASON: This is an example.\nActual assessment: REVIEW because evidence is missing.",
            "STATUS: CLEAR\nCITATION: NONE\nREASON: No hazard.\nFINAL STATUS: VIOLATION",
            "```\nSTATUS: CLEAR\nCITATION: NONE\nREASON: Example answer, not an assessment.\n```\nI cannot assess this event.",
            "```\nSTATUS: CLEAR\nCITATION: NONE\nREASON: No hazard.\n```",
            "STATUS: CLEAR\nCITATION: NONE\nREASON: No hazard.\nNOTE: Required evidence is missing.",
        )
        for raw in replies:
            with self.subTest(raw=raw):
                parsed = parse_audit_response(raw)
                guarded = guard_grounded_decision(parsed, {"1910.22"}, {"1910.22"})
                metrics = score_outcomes([dict(guarded, actual=0)])
                self.assertEqual(parsed["status"], "UNPARSED")
                self.assertEqual(guarded["status"], "UNPARSED")
                self.assertEqual(parsed["raw_response"], raw)
                self.assertEqual(metrics["accuracy"], 0)
                self.assertEqual(metrics["true_negatives"], 0)

    def test_complete_frame_preserves_think_stripping_and_blank_lines(self):
        raw = "<think>Consider STATUS: VIOLATION as a hypothesis.\nNeed facts.</think>\n\nSTATUS: REVIEW\n\nCITATION: NONE\nREASON: A required fact is absent.\n\n"
        parsed = parse_audit_response(raw)
        self.assertEqual(parsed["status"], "REVIEW")
        self.assertEqual(parsed["raw_response"], raw)

    def test_out_of_order_response_fields_are_unparsed(self):
        raw = "CITATION: NONE\nSTATUS: CLEAR\nREASON: Wrong frame order."
        self.assertEqual(parse_audit_response(raw)["status"], "UNPARSED")

    def test_guard_retains_rejected_decision_and_requires_evidence(self):
        parsed = parse_audit_response("STATUS: VIOLATION\nCITATION: 1910.999\nREASON: Unsupported claim.")
        guarded = guard_grounded_decision(parsed, {"1910.22"}, {"1910.22"})
        self.assertEqual(guarded["status"], "REVIEW")
        self.assertEqual(guarded["unguarded_citation"], "1910.999")
        self.assertEqual(parsed["status"], "VIOLATION")
        clear = parse_audit_response("STATUS: CLEAR\nCITATION: NONE\nREASON: No breach.")
        self.assertEqual(guard_grounded_decision(clear, set())["status"], "REVIEW")
        self.assertEqual(guard_grounded_decision(clear, {"1910.22"})["status"], "CLEAR")

    def test_historical_replay_is_explicit_and_keeps_old_fields(self):
        data = json.loads((ROOT / "json/finetune_evaluation_results.json").read_text())
        records = data["strategies"]["base"]["results"]
        old = auditor.compute_historical_metrics(records)
        new = auditor.compute_metrics(records)
        self.assertNotIn("scoring_schema", old)
        self.assertEqual(new["scoring_schema"], "selective_binary_v2")
        self.assertEqual(new["accuracy"], 0)
        self.assertEqual(new["false_positives"], 0)
        self.assertEqual(new["unresolved_count"], len(records))


class RunnerRegressionTests(unittest.TestCase):
    def assert_outage(self, results, metrics):
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r["status"] == "UNPARSED" for r in results))
        self.assertTrue(all(r["predicted"] is None for r in results))
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["true_negatives"], 0)
        self.assertEqual(metrics["unresolved_count"], 8)

    def test_continuous_runner_balanced_outage(self):
        with patch.object(auditor, "retrieve_triangulated_adaptive", return_value=[CHUNK]), patch.object(auditor, "ask_async", AsyncMock(return_value="")), patch.object(auditor.aiohttp, "ClientSession", FakeSession):
            result, _ = asyncio.run(auditor.run_continuous_audit(rows(), [CHUNK], np.ones((1, 1)), memory(), pillars(), VERIFIER, None,
                enable_few_shot=False, enable_self_reflection=False, ingest_memory=False, adapt_pillars=False))
        self.assert_outage(result, auditor.compute_metrics(result))

    def test_continuous_runner_unlabeled_input_and_exception(self):
        row = rows()[0]
        row.pop("Is_Violation")
        with patch.object(auditor, "retrieve_triangulated_adaptive", return_value=[CHUNK]), patch.object(auditor, "ask_async", AsyncMock(side_effect=RuntimeError("synthetic transport failure"))), patch.object(auditor.aiohttp, "ClientSession", FakeSession):
            result, _ = asyncio.run(auditor.run_continuous_audit([row], [CHUNK], np.ones((1, 1)), memory(), pillars(), VERIFIER, None,
                enable_few_shot=False, enable_self_reflection=False, ingest_memory=False, adapt_pillars=False))
        self.assertIsNone(result[0]["actual"])
        self.assertIsNone(result[0]["predicted"])
        self.assertIsNone(auditor.compute_metrics(result)["accuracy"])

    def test_final_guard_follows_critique(self):
        bad = parse_audit_response("STATUS: VIOLATION\nCITATION: 1910.999\nREASON: Synthetic critique error.")
        critic = SimpleNamespace(critique_and_reground_async=AsyncMock(return_value=(bad, True, {})))
        raw = "STATUS: REVIEW\nCITATION: NONE\nREASON: Need review."
        with patch.object(auditor, "retrieve_triangulated_adaptive", return_value=[CHUNK]), patch.object(auditor, "ask_async", AsyncMock(return_value=raw)), patch.object(auditor.aiohttp, "ClientSession", FakeSession):
            result, _ = asyncio.run(auditor.run_continuous_audit(rows()[:1], [CHUNK], np.ones((1, 1)), memory(), pillars(), VERIFIER, critic,
                enable_few_shot=False, ingest_memory=False, adapt_pillars=False))
        self.assertEqual(result[0]["status"], "REVIEW")
        self.assertIsNone(result[0]["predicted"])
        self.assertEqual(result[0]["unguarded_citation"], "1910.999")
        self.assertEqual(result[0]["initial_response"]["status"], "REVIEW")

    def test_scaled_runner_balanced_outage(self):
        with patch.object(scaled, "retrieve_triangulated_adaptive", return_value=[CHUNK]), patch.object(scaled, "ask_model", AsyncMock(return_value="")), patch.object(scaled.aiohttp, "ClientSession", FakeSession):
            result = asyncio.run(scaled.evaluate_strategy("static_rag", rows(), [CHUNK], np.ones((1, 1)), memory(), pillars(), VERIFIER, None, "mock", 1))
        self.assert_outage(result["results"], result["metrics"])

    def test_threeway_ungrounded_balanced_outage(self):
        with patch.object(threeway.aiohttp, "ClientSession", FakeSession):
            result, metrics = asyncio.run(threeway.audit_ungrounded(rows(), VERIFIER, 1))
        self.assert_outage(result, metrics)

    def test_improvement_runner_balanced_outage(self):
        with patch.object(improvement, "embed_texts", return_value=np.ones((1, 1))), patch.object(improvement.aiohttp, "ClientSession", FakeSession):
            result, metrics = asyncio.run(improvement.run_audit_pass(rows(), [CHUNK], np.ones((1, 1)), memory(), pillars(), VERIFIER, None, is_baseline=True))
        self.assert_outage(result, metrics)

    def test_hybrid_review_is_not_an_invented_false_alert(self):
        review = SimpleNamespace(status="REVIEW", citation=None, to_dict=lambda: {"status":"REVIEW", "citation":None, "reason":"uncertain"})
        with patch.object(scaled, "evaluate_event", return_value=review):
            result = scaled.evaluate_hybrid(rows(), [CHUNK], VERIFIER)
        self.assertEqual(result["metrics"]["accuracy"], 0)
        self.assertEqual(result["metrics"]["false_positives"], 0)
        self.assertEqual(result["metrics"]["false_negatives"], 0)
        self.assertEqual(result["metrics"]["review_count"], 8)
        self.assertEqual(result["metrics"]["unresolved_count"], 8)


if __name__ == "__main__":
    unittest.main()
