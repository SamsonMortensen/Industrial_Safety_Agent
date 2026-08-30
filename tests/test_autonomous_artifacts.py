import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AutonomousArtifactTests(unittest.TestCase):
    def test_camera_example_contains_evidence_not_ground_truth(self):
        example = json.loads(
            (ROOT / "data" / "example_camera_observation.json").read_text(
                encoding="utf-8"
            )
        )
        forbidden = {"Is_Violation", "Violation_Type", "Expected_Section"}
        self.assertTrue(forbidden.isdisjoint(example))
        self.assertTrue(example["relations"])
        self.assertTrue(example["evidence_refs"])

    def test_hybrid_benchmark_is_locked_to_current_dataset(self):
        report = json.loads(
            (ROOT / "json" / "autonomous_retrieval_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            report["provenance"]["dataset_sha256"],
            sha256(ROOT / "daily_yard_log.csv"),
        )
        self.assertFalse(report["provenance"]["labels_used_to_form_queries"])
        self.assertEqual(report["applicability_reranked"]["cases"], 144)
        self.assertGreaterEqual(report["applicability_reranked"]["recall_at_4"], 0.60)
        self.assertGreaterEqual(report["applicability_reranked"]["recall_at_16"], 0.95)

    def test_demo_decision_is_supported_by_retrieved_authority(self):
        report = json.loads(
            (ROOT / "json" / "autonomous_demo_result.json").read_text(encoding="utf-8")
        )
        result = report["results"][0]
        retrieved = {candidate["section"] for candidate in result["candidates"]}
        self.assertTrue(result["trigger"]["triggered"])
        self.assertEqual(
            (result["status"], result["citation"]), ("VIOLATION", "1910.333")
        )
        self.assertIn(result["citation"], retrieved)


if __name__ == "__main__":
    unittest.main()
