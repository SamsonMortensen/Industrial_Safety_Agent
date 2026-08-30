import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from osha_incident_pipeline import (
    PII_FIELDS_REMOVED,
    build_profile,
    normalize_row,
    read_rows,
)


def sample_row(**overrides):
    row = {
        "ID": "123",
        "UPA": "private-key",
        "EventDate": "02/03/2025",
        "Employer": "Example Employer",
        "Address1": "1 Private Road",
        "Address2": "Suite 2",
        "City": "Example City",
        "State": "WA",
        "Zip": "99999",
        "Latitude": "47.1",
        "Longitude": "-122.1",
        "Primary NAICS": "482111",
        "Hospitalized": "1",
        "Amputation": "0",
        "Loss of Eye": "0",
        "Inspection": "987",
        "Final Narrative": "Worker was struck by moving rail equipment.",
        "Nature": "10",
        "NatureTitle": "Traumatic injuries",
        "Part of Body": "20",
        "Part of Body Title": "Multiple parts",
        "Event": "30",
        "EventTitle": "Contact with objects",
        "Source": "40",
        "SourceTitle": "Vehicles",
        "Secondary Source": "",
        "Secondary Source Title": "",
        "FederalState": "Federal",
    }
    row.update(overrides)
    return row


class IncidentPipelineTests(unittest.TestCase):
    def test_normalized_observation_removes_establishment_identifiers(self):
        normalized = normalize_row(sample_row())
        rendered = str(normalized)
        for forbidden in (
            "Example Employer",
            "1 Private Road",
            "Example City",
            "47.1",
            "-122.1",
            "private-key",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(normalized["human_flag"])
        self.assertEqual(normalized["source_id"], "osha-severe-injury-report")
        self.assertEqual(
            set(PII_FIELDS_REMOVED),
            {
                "Employer",
                "Address1",
                "Address2",
                "City",
                "Zip",
                "Latitude",
                "Longitude",
                "UPA",
            },
        )

    def test_profile_separates_outcomes_from_query_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            source.write_text("placeholder", encoding="utf-8")
            profile = build_profile(
                [sample_row(), sample_row(ID="124", Amputation="1")], source
            )
        self.assertEqual(profile["records"], 2)
        self.assertEqual(profile["target_domain_records"]["rail_transportation"], 2)
        self.assertFalse(profile["provenance"]["outcomes_used_for_queries"])
        self.assertTrue(profile["provenance"]["structured_identifier_fields_removed"])
        self.assertFalse(profile["provenance"]["narrative_redaction_performed"])

    def test_missing_columns_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "Final Narrative"])
                writer.writeheader()
                writer.writerow({"ID": "1", "Final Narrative": "text"})
            with self.assertRaisesRegex(ValueError, "missing columns"):
                list(read_rows(path))


if __name__ == "__main__":
    unittest.main()
