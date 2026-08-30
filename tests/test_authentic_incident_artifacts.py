import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class AuthenticIncidentArtifactTests(unittest.TestCase):
    def test_osha_profile_is_deidentified_and_provenanced(self):
        profile = json.loads(
            (ROOT / "json" / "osha_sir_profile.json").read_text(encoding="utf-8")
        )
        self.assertEqual(profile["records"], 105996)
        self.assertEqual(profile["date_start"], "2015-01-01")
        self.assertEqual(profile["date_end"], "2025-11-30")
        self.assertEqual(profile["narrative_coverage"], 1.0)
        self.assertTrue(profile["provenance"]["structured_identifier_fields_removed"])
        self.assertFalse(profile["provenance"]["narrative_redaction_performed"])
        self.assertFalse(profile["provenance"]["oiics_codes_used_for_queries"])
        self.assertFalse(profile["provenance"]["outcomes_used_for_queries"])
        self.assertEqual(
            profile["provenance"]["source_sha256"],
            "406ffe7bc0758a447ad1650171b9c3f6990f901e8b73f167dad7cf60a97f13cd",
        )

    def test_authentic_benchmark_reports_coverage_not_accuracy(self):
        report = json.loads(
            (ROOT / "json" / "authentic_incident_retrieval_hybrid.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["coverage"]["cases"], 500)
        self.assertEqual(report["coverage"]["trigger_rate"], 1.0)
        self.assertEqual(report["coverage"]["nonempty_retrieval_rate"], 1.0)
        self.assertFalse(report["provenance"]["regulation_labels_available"])
        self.assertEqual(report["grounding_gate"]["unsupported_citations_allowed"], 0)
        self.assertTrue(all("narrative" not in record for record in report["records"]))
        self.assertTrue(
            any("not legal accuracy" in item for item in report["limitations"])
        )


if __name__ == "__main__":
    unittest.main()
