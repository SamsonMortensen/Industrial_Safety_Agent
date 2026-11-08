"""Numerical, publication-boundary and stale-report regressions."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import public_results as report
from export_public_development import export_run, read_completed, digest


def row(case, group, target, predicted, score=.5):
    return {"case": case, "group": group, "target": target,
            "predictions": {"probe": predicted}, "scores": {"probe": score}}


def sequential_sum(values, start=0):
    """Emulate the pre-3.12 floating sum used by the supported Linux runtime."""
    for value in values:
        start += value
    return start


class PublicResultsTests(unittest.TestCase):
    def test_recomputation_accepts_python311_accumulation_rounding(self):
        with patch.object(report, "sum", sequential_sum, create=True):
            data = report.check(ROOT)
        saved = report.read(ROOT / "docs/results/metrics.json")
        # Exercise an actual cross-version difference, not an unchanged value.
        self.assertNotEqual(data["person"]["round2"]["score_diagnostics"]["frozen_first_pilot"]["group_weighted_brier"],
                            saved["person"]["round2"]["score_diagnostics"]["frozen_first_pilot"]["group_weighted_brier"])

    def test_metric_tolerance_keeps_counts_structure_and_large_changes_exact(self):
        self.assertTrue(report.metrics_equal({"score": .5}, {"score": .5 + 1e-15}))
        for old, new in ((1, 1.0), (1, True), (1, 2), ({"count": 1}, {"other": 1}),
                         ([1], [1, 2]), (.5, .500001), (float("nan"), float("nan")),
                         (float("inf"), float("inf"))):
            with self.subTest(old=old, new=new):
                self.assertFalse(report.metrics_equal(old, new))

    def test_tiny_saved_metric_edit_still_fails_evidence_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            needed = list(report.manifest_inputs(ROOT)) + report.artifact_paths() + ["docs/results/manifest.json"]
            for relative in needed:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, path)
            data = report.read(root / "docs/results/metrics.json")
            data["person"]["round2"]["score_diagnostics"]["frozen_first_pilot"]["group_weighted_brier"] += 1e-15
            (root / "docs/results/metrics.json").write_text(report.json_text(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                report.check(root)

    def test_public_text_fingerprints_survive_windows_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_bytes(b'{\n  "value": 1\n}\n')
            expected = report.sha256(path)
            path.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
            self.assertEqual(report.sha256(path), expected)

    def test_group_bounds_do_not_count_camera_views_as_independent(self):
        rows = [row(str(i), f"g{i // 2}", i % 2, i % 2) for i in range(8)]
        m = report.classification(rows, "probe")
        self.assertEqual(m["source_groups"], 4)
        self.assertAlmostEqual(m["group_exact_95"][0], .025 ** .25)
        self.assertLess(m["group_exact_95"][0], .025 ** .125)

    def test_exact_bounds_include_empty_successes_and_documented_example(self):
        self.assertEqual(report.exact_group_interval(0, 4)[0], 0)
        self.assertAlmostEqual(report.exact_group_interval(0, 4)[1], 1 - .025 ** .25)
        # Published SciPy proportion_ci example: k=7, n=50.
        lo, hi = report.exact_group_interval(7, 50)
        self.assertAlmostEqual(lo, .05819170033997342)
        self.assertAlmostEqual(hi, .26739600249700846)

    def test_group_weighting_and_missing_class_denominator(self):
        rows = [row("one", "g1", 1, 0, 0)] + [row(str(i), "g2", 1, 1, 1) for i in range(3)]
        self.assertEqual(report.probability_metrics(rows, "probe")["group_weighted_brier"], .5)
        self.assertIsNone(report.classification(rows, "probe")["balanced_accuracy"])
        self.assertIsNone(report.classification(rows, "probe")["false_alert_rate"])

    def test_unparsed_predictions_remain_strict_errors(self):
        m = report.binary_counts([0, 1], [None, None], strict_unparsed=True)
        self.assertEqual(m["confusion"]["false_positive"], 1)
        self.assertEqual(m["confusion"]["false_negative"], 1)
        self.assertEqual(m["accuracy"], 0)
        with self.assertRaises(ValueError):
            report.binary_counts([0], [None])

    def test_invalid_scores_and_duplicate_cases_fail(self):
        for score in (float("nan"), float("inf"), -.1, 1.1):
            with self.assertRaises(ValueError):
                report.probability_metrics([row("a", "g", 1, 1, score)], "probe")
        with self.assertRaises(ValueError):
            report.classification([row("a", "g", 1, 1)] * 2, "probe")

    def test_committed_public_results_and_interpretation(self):
        data = report.check(ROOT)
        self.assertEqual(data["person"]["round3"]["metrics"]["round3_classifier"]["source_groups"], 4)
        changed = copy.deepcopy(data)
        changed["person"]["round3"]["metrics"]["round3_classifier"]["confusion"]["false_positive"] = 1
        with self.assertRaisesRegex(ValueError, "interpretation"):
            report.validate_historical_scope(changed)

    def test_snapshot_has_only_public_prediction_fields(self):
        snapshot = report.read(ROOT / "data/public_development_runs.json")
        allowed = {"case", "group", "target", "predictions", "scores"}
        for run in snapshot["runs"]:
            for item in run["rows"]:
                self.assertEqual(set(item), allowed)
            self.assertIs(run["independent_evaluation_used"], False)
        raw = json.dumps(snapshot).lower()
        for forbidden in ("c:\\", "source_path", "sample_id", "approver", "approval", "reviewer", ".mp4"):
            self.assertNotIn(forbidden, raw)

    def test_modified_figure_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            needed = list(report.manifest_inputs(ROOT)) + report.artifact_paths() + ["docs/results/manifest.json"]
            for relative in needed:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, path)
            (root / "docs/results/person_development.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                report.check(root)

    def test_completed_export_rejects_independent_data_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results.json").write_text(json.dumps({"independent_evaluation_used": True, "model_promoted": False}))
            (root / "predictions.json").write_text("[]")
            (root / "protocol.json").write_text("{}")
            status = {"status": "complete", "completed_at_utc": "2026-10-01T00:00:00+00:00",
                      "artifacts_sha256": {p: digest(root / p) for p in ("results.json", "predictions.json", "protocol.json")}}
            (root / "status.json").write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "Independent"):
                read_completed(root)
            (root / "predictions.json").write_text("[{}]")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_completed(root)

    def test_export_omits_source_secrets_and_nonvalidation_rows(self):
        snapshot = report.read(ROOT / "data/public_development_runs.json")["runs"][0]
        metrics, rows = {}, []
        for model in snapshot["models"]:
            m = report.classification(snapshot["rows"], model)
            metrics[model] = {"confusion": m["confusion"], "source_groups": m["source_groups"],
                              "all_views_correct_groups": m["fully_correct_groups"],
                              "all_views_correct_group_rate_95_exact_interval": m["group_exact_95"]}
        for r in snapshot["rows"]:
            rows.append({"sample_id": r["case"], "group_id": r["group"], "target": r["target"],
                         "partition": "development_validation", "source_path": "PRIVATE_MEDIA_PATH",
                         "reviewer": "PRIVATE_REVIEWER", "trained_prediction": r["predictions"]["trained_logistic_probe"],
                         "trained_score_uncalibrated": r["scores"]["trained_logistic_probe"],
                         "zero_shot_prediction": r["predictions"]["frozen_zero_shot_baseline"]})
        rows += [{"partition": "development_train", "target": 1, "private_note": "PRIVATE_TRAINING"}]
        result = {"schema": "approved_person_development_results_v1", "model_promoted": False,
                  "independent_evaluation_used": False, "development_validation": metrics,
                  "split_counts": snapshot["coverage"], "approval": {"approver": "PRIVATE_REVIEWER"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, content in (("results.json", result), ("predictions.json", rows),
                                      ("protocol.json", {"encoder": snapshot["encoder"], "path": "PRIVATE_PROTOCOL"})):
                (root / filename).write_text(json.dumps(content))
            (root / "status.json").write_text(json.dumps({"status": "complete", "completed_at_utc": "2026-10-01T00:00:00+00:00",
                "artifacts_sha256": {p: digest(root / p) for p in ("results.json", "predictions.json", "protocol.json")}}))
            exported = export_run(root, "pilot", result["schema"], "development_validation")
            self.assertEqual(len(exported["rows"]), 4)
            self.assertNotIn("PRIVATE", json.dumps(exported))
            self.assertNotIn("approval", json.dumps(exported))


if __name__ == "__main__":
    unittest.main()
