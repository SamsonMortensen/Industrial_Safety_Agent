"""Keep the publication comparison matched and honest about its boundaries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from report_retrieval_comparison import check, compare, current_capture, read, validate_capture
import capture_retrieval_run as capture


def sequential_sum(values, start=0):
    """Emulate Python 3.11's finite floating sum without changing retrieval."""
    for value in values:
        start += value
    return start


class RetrievalComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = read(ROOT / "data/retrieval_comparison/before.json")
        cls.after = read(ROOT / "data/retrieval_comparison/after.json")

    def test_saved_captures_accept_python311_float_recomputation(self):
        with patch.object(capture, "sum", sequential_sum, create=True):
            validate_capture(self.before)
            validate_capture(self.after)

    def test_fresh_capture_accepts_python311_metrics_with_exact_same_records(self):
        fresh = copy.deepcopy(current_capture(ROOT, self.after))
        with patch.object(capture, "sum", sequential_sum, create=True):
            fresh["metrics"] = {cohort: capture.summarize([r for r in fresh["records"]
                                if cohort == "all144" or r["cohort"] == cohort])
                                for cohort in fresh["metrics"]}
        self.assertNotEqual(fresh["metrics"], self.after["metrics"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text(json.dumps(fresh), encoding="utf-8")
            check(ROOT, fresh=path)

    def test_fresh_record_change_is_not_covered_by_float_tolerance(self):
        fresh = copy.deepcopy(current_capture(ROOT, self.after))
        fresh["records"][0]["queries"][0] += " changed"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text(json.dumps(fresh), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "after-run: records"):
                check(ROOT, fresh=path)

    def test_metric_counts_and_meaningful_float_changes_still_fail(self):
        for key, value in (("cases", 64.0), ("hits_at_16", 62),
                           ("mean_reciprocal_rank_at_16", .49)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.after)
                changed["metrics"]["primary64"][key] = value
                with self.assertRaisesRegex(ValueError, "metrics do not match"):
                    validate_capture(changed)

    def test_matched_gain_and_first_rank_regressions_are_both_retained(self):
        result = compare(self.before, self.after)
        self.assertEqual(result["metrics"]["primary64"]["after"]["hits_at_16"], 63)
        self.assertEqual(result["metrics"]["all144"]["paired_cutoffs"]["16"], {"gained": 14, "lost": 0})
        self.assertEqual(result["metrics"]["all144"]["paired_cutoffs"]["1"], {"gained": 23, "lost": 4})
        self.assertEqual(len(result["remaining_misses"]), 4)
        self.assertEqual(len(result["rank_regressions"]), 15)

    def test_changed_corpus_cannot_be_called_matched(self):
        after = copy.deepcopy(self.after)
        after["input_and_code_sha256_lf"]["json/regulations.json"] = "changed"
        with self.assertRaisesRegex(ValueError, "keep dataset and corpus fixed"):
            compare(self.before, after)

    def test_changed_incident_cannot_be_called_matched(self):
        after = copy.deepcopy(self.after)
        after["records"][0]["incident"] = "different evidence"
        with self.assertRaisesRegex(ValueError, "text, or cohort changed"):
            compare(self.before, after)

    def test_duplicate_cases_are_rejected(self):
        after = copy.deepcopy(self.after)
        after["records"][1]["event_id"] = after["records"][0]["event_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_capture(after)

    def test_rank_tampering_is_detected_from_candidate_list(self):
        for value in (99, float(self.after["records"][0]["rank"]), True):
            with self.subTest(value=value):
                after = copy.deepcopy(self.after)
                after["records"][0]["rank"] = value
                with self.assertRaisesRegex(ValueError, "Rank does not match"):
                    validate_capture(after)

    def test_protected_evaluation_flags_are_rejected(self):
        for field in ("independent_evaluation_used", "calibration_reserve_used"):
            with self.subTest(field=field):
                after = copy.deepcopy(self.after)
                after[field] = True
                with self.assertRaisesRegex(ValueError, "Only development"):
                    validate_capture(after)

    def test_saved_narrative_metrics_and_figures_match(self):
        result = check(ROOT)
        self.assertEqual(len(result["cases"]), 144)


if __name__ == "__main__":
    unittest.main()
