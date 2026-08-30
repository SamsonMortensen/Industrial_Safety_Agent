import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from autonomous_investigator import (
    AutonomousInvestigator,
    HybridRetriever,
    Observation,
    TriggerEngine,
)
from hybrid_policy import evaluate_event


def electrical_observation():
    return Observation.from_mapping(
        {
            "event_id": "electrical-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "source_id": "camera-1",
            "summary": "Crane boom moved four feet inside an energized overhead conductor approach area.",
            "equipment": ["crane"],
            "changed": True,
            "human_flag": True,
        }
    )


def chunk(section, text):
    return {
        "citation": f"29 CFR {section}",
        "section": section,
        "heading": "Test authority",
        "text": text,
    }


class AutonomousGroundingGateTests(unittest.TestCase):
    def build(self, chunks):
        return AutonomousInvestigator(
            HybridRetriever(chunks),
            trigger_engine=TriggerEngine(sample_rate=0.0),
            policy_fn=evaluate_event,
            candidate_limit=len(chunks),
        )

    def test_policy_citation_is_blocked_when_retrieval_does_not_support_it(self):
        investigator = self.build(
            [chunk("1910.22", "Walking-working surfaces must be kept clean and dry.")]
        )
        result = investigator.investigate(electrical_observation())
        self.assertEqual(result.status, "REVIEW")
        self.assertIsNone(result.citation)
        self.assertIn("not present in the retrieved candidate set", result.reason)

    def test_policy_citation_is_allowed_when_retrieval_supports_it(self):
        investigator = self.build(
            [
                chunk(
                    "1910.333",
                    "Energized conductors require safe work practices and approach distance.",
                )
            ]
        )
        result = investigator.investigate(electrical_observation())
        self.assertEqual((result.status, result.citation), ("VIOLATION", "1910.333"))


if __name__ == "__main__":
    unittest.main()
