"""A rejected citation can be investigated without becoming a trusted decision."""
import asyncio
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import audit_agent_v2 as auditor
from audit_outcomes import guard_grounded_decision, parse_audit_response
from learning_policy import assess_for_learning


HIT = {"section": "1910.333", "citation": "29 CFR 1910.333",
       "heading": "Electrical work", "text": "Synthetic excerpt for a contract test."}
VALID = {"1910.22", "1910.333"}
PROPOSAL = "STATUS: VIOLATION\nCITATION: 1910.22\nREASON: Synthetic observation."


class OfflineSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class RetrievalGapObservationTests(unittest.TestCase):
    def run_case(self, reply, critique=None):
        row = {"Log_ID": "fixture-gap", "Equipment_ID": "TEST", "Equipment_Type": "Inspection",
               "Location": "Walkway", "Operator_Shift_Hours": 8,
               "Reported_Incident": "Synthetic observation"}
        bank = SimpleNamespace(gaps={}, pillars={}, pillar_vectors={})

        def record(section, event):
            self.assertIs(event, row)
            bank.gaps.setdefault(section, {"count": 0})["count"] += 1

        bank.record_retrieval_gap = record
        memory = SimpleNamespace(episodes=[])
        critic = None if critique is None else SimpleNamespace(
            critique_and_reground_async=AsyncMock(return_value=(critique, False, "Synthetic critique.")))
        with patch.object(auditor, "retrieve_triangulated_adaptive", return_value=[HIT]), \
             patch.object(auditor, "ask_async", AsyncMock(return_value=reply)), \
             patch.object(auditor.aiohttp, "ClientSession", OfflineSession), \
             contextlib.redirect_stdout(io.StringIO()):
            results, metrics = asyncio.run(auditor.run_continuous_audit(
                [row], [HIT], None, memory, bank, SimpleNamespace(valid_sections=VALID), critic,
                enable_few_shot=False, enable_self_reflection=critique is not None,
                ingest_memory=False, adapt_pillars=False))
        self.assertEqual(memory.episodes, [])
        self.assertEqual(metrics["pillars_promoted"], [])
        return results[0], metrics

    def test_valid_unretrieved_proposal_is_recorded_but_stays_review(self):
        result, metrics = self.run_case(PROPOSAL)
        self.assertEqual(result["status"], "REVIEW")
        self.assertIsNone(result["predicted"])
        self.assertIsNone(result["citation"])
        self.assertEqual(result["unguarded_citation"], "1910.22")
        self.assertEqual(metrics["retrieval_gaps_seen"], {"1910.22": 1})
        self.assertFalse(assess_for_learning({}, result["status"], result["citation"], True).trusted)

    def test_initial_and_guarded_critique_proposal_count_once(self):
        guarded = guard_grounded_decision(parse_audit_response(PROPOSAL), {HIT["section"]}, VALID)
        result, metrics = self.run_case(PROPOSAL, guarded)
        self.assertEqual(result["status"], "REVIEW")
        self.assertEqual(metrics["retrieval_gaps_seen"], {"1910.22": 1})

    def test_new_guarded_critique_proposal_is_preserved(self):
        guarded = guard_grounded_decision(parse_audit_response(PROPOSAL), {HIT["section"]}, VALID)
        result, metrics = self.run_case("", guarded)
        self.assertEqual(result["initial_response"]["status"], "UNPARSED")
        self.assertEqual(result["status"], "REVIEW")
        self.assertEqual(metrics["retrieval_gaps_seen"], {"1910.22": 1})

    def test_missing_or_invalid_proposals_do_not_create_gaps(self):
        replies = ["", "STATUS: VIOLATION\nCITATION: 1910.22",
                   "STATUS: VIOLATION\nCITATION: 1910.999\nREASON: Unknown section.",
                   "STATUS: REVIEW\nCITATION: NONE\nREASON: Missing evidence.",
                   "STATUS: CLEAR\nCITATION: 1910.22\nREASON: Contradictory citation."]
        for reply in replies:
            with self.subTest(reply=reply):
                _, metrics = self.run_case(reply)
                self.assertEqual(metrics["retrieval_gaps_seen"], {})

    def test_retrieved_proposal_does_not_create_a_gap(self):
        result, metrics = self.run_case(PROPOSAL.replace("1910.22", HIT["section"]))
        self.assertEqual(result["status"], "VIOLATION")
        self.assertEqual(metrics["retrieval_gaps_seen"], {})

    def test_initial_proposal_is_diagnostic_when_critique_changes_the_answer(self):
        critique = parse_audit_response("STATUS: CLEAR\nCITATION: NONE\nREASON: Synthetic corrected reply.")
        result, metrics = self.run_case(PROPOSAL, critique)
        self.assertEqual(result["status"], "CLEAR")
        self.assertIsNone(result["citation"])
        self.assertEqual(metrics["retrieval_gaps_seen"], {"1910.22": 1})


if __name__ == "__main__":
    unittest.main()
