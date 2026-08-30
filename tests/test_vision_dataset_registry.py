import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from incident_dataset_registry import load_registry as load_incident_registry
from incident_dataset_registry import select_by_content
from vision_dataset_registry import eligible_datasets, load_registry, validate_registry
from vision_manifest import (
    assign_group_splits,
    read_manifest,
    validate_no_group_leakage,
)


class VisionDatasetRegistryTests(unittest.TestCase):
    def test_registry_is_valid_and_ids_are_unique(self):
        payload = load_registry(ROOT / "data" / "vision_datasets.json")
        ids = [item["id"] for item in payload["datasets"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(validate_registry(payload), [])

    def test_commercial_gate_excludes_research_and_unresolved_datasets(self):
        payload = load_registry(ROOT / "data" / "vision_datasets.json")
        selected = eligible_datasets(payload, "commercial")
        self.assertEqual(
            [item["id"] for item in selected], ["onal_dandil_safe_unsafe_video"]
        )

    def test_research_gate_still_excludes_unresolved_rights(self):
        payload = load_registry(ROOT / "data" / "vision_datasets.json")
        selected = {item["id"] for item in eligible_datasets(payload, "research")}
        self.assertIn("sh17_ppe", selected)
        self.assertIn("isafetybench", selected)
        self.assertNotIn("shwd_helmet", selected)


class IncidentDatasetRegistryTests(unittest.TestCase):
    def test_authentic_incident_registry_is_valid(self):
        payload = load_incident_registry(ROOT / "data" / "incident_text_datasets.json")
        self.assertGreaterEqual(len(payload["datasets"]), 7)
        self.assertTrue(
            all(item["prohibited_inferences"] for item in payload["datasets"])
        )

    def test_rail_filter_returns_fra(self):
        payload = load_incident_registry(ROOT / "data" / "incident_text_datasets.json")
        selected = {item["id"] for item in select_by_content(payload, "rail")}
        self.assertIn("fra_accident_incident", selected)


class VisionManifestTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for group_number in range(12):
            label = "unsafe" if group_number % 2 else "safe"
            for frame_number in range(3):
                self.rows.append(
                    {
                        "sample_id": f"g{group_number}-f{frame_number}",
                        "source_path": f"clip-{group_number}.mp4",
                        "label": label,
                        "group_id": f"camera-a-day-{group_number}",
                        "site_id": "site-a",
                        "camera_id": "camera-a",
                        "captured_at": f"2026-01-{group_number + 1:02d}T12:00:00Z",
                    }
                )

    def test_group_split_is_deterministic_and_leak_free(self):
        first = assign_group_splits(self.rows, seed=21)
        second = assign_group_splits(self.rows, seed=21)
        self.assertEqual(first, second)
        validate_no_group_leakage(first)
        self.assertEqual(
            {row["split"] for row in first}, {"train", "validation", "test"}
        )

    def test_manifest_requires_grouping_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "label"])
                writer.writeheader()
                writer.writerow({"sample_id": "one", "label": "safe"})
            with self.assertRaisesRegex(ValueError, "Manifest missing columns"):
                read_manifest(path)


if __name__ == "__main__":
    unittest.main()
