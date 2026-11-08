"""Publication boundaries use generated records, never private run artifacts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import export_public_development as exporter
from public_results import classification, probability_metrics

MARKER = "SYNTHETIC_PRIVATE_MARKER"
SPECS = {
    "pilot": ("approved_person_development_results_v1", "development_validation"),
    "round2": ("expanded_person_development_results_v1", "new_validation_metrics"),
    "round3": ("person_round3_results_v1", "validation_metrics"),
    "calibration": (None, "development_validation_metrics"),
}
ENCODER = {"model": "google/siglip2-base-patch16-224",
           "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"}


def fixture(run_id):
    models = exporter.EXPECTED_MODELS[run_id]
    public_rows = []
    for i, (target, score) in enumerate(zip((0, 1, 1, 0), (.1, .8, .2, .1))):
        public_rows.append({
            "case": f"v{i + 1:03}", "group": f"g{i // 2 + 1:02}", "target": target,
            "predictions": {model: 0 if model in ("train_majority_baseline", "training_majority")
                            else int(score >= .5) for model in models},
            "scores": {model: score for model in exporter.EXPECTED_SCORES[run_id]},
        })
    metrics = {}
    for model in models:
        summary = classification(public_rows, model)
        clean = {"confusion": summary["confusion"], "source_groups": summary["source_groups"],
                 "all_views_correct_groups": summary["fully_correct_groups"],
                 "all_views_correct_group_rate_95_exact_interval": summary["group_exact_95"]}
        metrics[model] = ({"classification_at_fixed_half": clean,
                           **probability_metrics(public_rows, model)} if run_id == "calibration" else clean)
    rows = []
    for public in public_rows:
        row = {"sample_id": public["case"], "group_id": public["group"],
               "target": public["target"], "partition": "development_validation",
               "reviewer": MARKER, "source_path": MARKER}
        if run_id == "pilot":
            row.update(trained_prediction=public["predictions"]["trained_logistic_probe"],
                       zero_shot_prediction=public["predictions"]["frozen_zero_shot_baseline"],
                       trained_score_uncalibrated=public["scores"]["trained_logistic_probe"])
        elif run_id == "calibration":
            row["scores"] = deepcopy(public["scores"])
        else:
            row["predictions"] = deepcopy(public["predictions"])
            if run_id == "round3":
                row["scores_uncalibrated"] = deepcopy(public["scores"])
            else:
                row.update(frozen_score_uncalibrated=public["scores"]["frozen_first_pilot"],
                           expanded_score_uncalibrated=public["scores"]["expanded_classifier"])
        rows.append(row)
    rows.extend({"partition": "development_train", "target": target, "private_note": MARKER}
                for target in (0, 0, 1, 1))
    schema, metric_key = SPECS[run_id]
    roles = (("classifier_fit", "temperature_fit", "development_validation") if run_id == "calibration"
             else ("development_train", "development_validation"))
    result = {"schema": schema, metric_key: metrics,
              "coverage": {role: {"clips": 4, "groups": 2, "private_note": MARKER} for role in roles},
              "independent_evaluation_used": False, "model_promoted": False,
              "approval": {"reviewer": MARKER}}
    if run_id == "calibration":
        result.update(temperature={"temperature": 1.25}, calibrated_model_accepted=False)
    return {"result": result, "rows": rows, "protocol": {"encoder": ENCODER, "private_note": MARKER},
            "completed_at_utc": "2026-10-01T12:30:15.123456+00:00", "expected_rows": public_rows}


def export_fixture(run_id, data):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, value in (("results.json", data["result"]), ("predictions.json", data["rows"]),
                            ("protocol.json", data["protocol"])):
            (root / name).write_text(json.dumps(value), encoding="utf-8")
        status = {"status": "complete", "completed_at_utc": data["completed_at_utc"],
                  "private_note": MARKER, "artifacts_sha256": {name: exporter.digest(root / name)
                  for name in ("results.json", "predictions.json", "protocol.json")}}
        (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
        return exporter.export_run(root, run_id, *SPECS[run_id])


def set_score(data, run_id, value):
    row = data["rows"][0]
    if run_id == "pilot":
        row["trained_score_uncalibrated"] = value
    elif run_id == "round2":
        row["frozen_score_uncalibrated"] = value
    else:
        field = "scores_uncalibrated" if run_id == "round3" else "scores"
        row[field][exporter.EXPECTED_SCORES[run_id][0]] = value


class PublicExportPrivacyTests(unittest.TestCase):
    def test_each_valid_run_preserves_anonymous_numeric_output(self):
        for run_id in SPECS:
            with self.subTest(run=run_id):
                data = fixture(run_id)
                output = export_fixture(run_id, data)
                self.assertEqual(output["rows"], data["expected_rows"])
                self.assertEqual(output["provenance"]["completed_at_utc"], data["completed_at_utc"])
                self.assertNotIn(MARKER, json.dumps(output))
                for row in output["rows"]:
                    self.assertEqual(set(row["scores"]), set(exporter.EXPECTED_SCORES[run_id]))

    def test_extra_nested_score_metadata_cannot_reach_public_export(self):
        for run_id, field in (("round3", "scores_uncalibrated"), ("calibration", "scores")):
            with self.subTest(run=run_id):
                data = fixture(run_id)
                data["rows"][0][field]["private_metadata"] = {"note": MARKER}
                output = export_fixture(run_id, data)
                self.assertEqual(output["rows"], data["expected_rows"])
                self.assertNotIn(MARKER, json.dumps(output))

    def test_scores_reject_text_nested_data_boolean_and_nonfinite_values(self):
        for run_id in SPECS:
            for value in (MARKER, {"note": MARKER}, True, float("nan"), float("inf"), -.01, 1.01):
                with self.subTest(run=run_id, value_type=type(value).__name__):
                    data = fixture(run_id)
                    set_score(data, run_id, value)
                    with self.assertRaisesRegex(ValueError, "Score") as error:
                        export_fixture(run_id, data)
                    self.assertNotIn(MARKER, str(error.exception))

    def test_completion_time_rejects_free_text_and_non_utc_values(self):
        for value in (MARKER, {"note": MARKER}, True, "2026-10-01", "2026-10-01T12:30:15",
                      "2026-10-01T12:30:15-07:00", "2026-02-30T12:30:15+00:00",
                      "2026-10-01T12:30:15+00:00 " + MARKER):
            with self.subTest(value_type=type(value).__name__):
                data = fixture("round3")
                data["completed_at_utc"] = value
                with self.assertRaisesRegex(ValueError, "UTC ISO timestamp") as error:
                    export_fixture("round3", data)
                self.assertNotIn(MARKER, str(error.exception))

    def test_valid_utc_timestamp_forms_are_preserved(self):
        for value in ("2026-10-01T12:30:15Z", "2026-10-01T12:30:15+00:00",
                      "2026-10-01T12:30:15.123456+00:00"):
            data = fixture("pilot")
            data["completed_at_utc"] = value
            self.assertEqual(export_fixture("pilot", data)["provenance"]["completed_at_utc"], value)

    def test_temperature_rejects_private_text_boolean_and_invalid_numbers(self):
        for value in (MARKER, {"note": MARKER}, True, float("nan"), float("inf"), 0, -1):
            data = fixture("calibration")
            data["result"]["temperature"]["temperature"] = value
            with self.subTest(value_type=type(value).__name__), self.assertRaisesRegex(ValueError, "Temperature"):
                export_fixture("calibration", data)

    def test_labels_and_predictions_require_binary_integers(self):
        for run_id in SPECS:
            data = fixture(run_id)
            data["rows"][0]["target"] = False
            with self.subTest(run=run_id), self.assertRaisesRegex(ValueError, "binary integers"):
                export_fixture(run_id, data)
        for run_id in ("pilot", "round2", "round3"):
            data = fixture(run_id)
            if run_id == "pilot":
                data["rows"][0]["trained_prediction"] = False
            else:
                data["rows"][0]["predictions"][exporter.EXPECTED_MODELS[run_id][0]] = False
            with self.subTest(run=run_id), self.assertRaisesRegex(ValueError, "binary integers"):
                export_fixture(run_id, data)

    def test_coverage_requires_integer_counts(self):
        for value in (True, "4", {"note": MARKER}, 4.0, -1):
            data = fixture("pilot")
            data["result"]["coverage"]["development_train"]["clips"] = value
            with self.subTest(value_type=type(value).__name__), self.assertRaisesRegex(ValueError, "Coverage counts"):
                export_fixture("pilot", data)


if __name__ == "__main__":
    unittest.main()
