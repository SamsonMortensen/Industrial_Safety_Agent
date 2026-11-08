"""Export anonymous development predictions, never media or review receipts.

Explicit completed runs only; no directory crawling or independent-evaluation input.
Source hashes identify the saved artifacts, not a certification of their labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

from public_results import classification, probability_metrics

EXPECTED_MODELS = {
    "pilot": ("train_majority_baseline", "frozen_zero_shot_baseline", "trained_logistic_probe"),
    "round2": ("training_majority", "fixed_prompt_baseline", "frozen_first_pilot", "expanded_classifier"),
    "round3": ("training_majority", "fixed_prompt_baseline", "frozen_first_pilot", "frozen_round2", "round3_classifier"),
    "calibration": ("previous_expanded_classifier", "new_probe_uncalibrated", "new_probe_temperature_scaled"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_completed(root):
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError("Only completed development runs can be exported")
    for name in ("results.json", "predictions.json", "protocol.json"):
        if digest(root / name) != status["artifacts_sha256"][name]:
            raise ValueError(f"Completion hash mismatch: {name}")
    result = json.loads((root / "results.json").read_text(encoding="utf-8"))
    if result.get("independent_evaluation_used") is not False:
        raise ValueError("Independent evaluation is outside this export's scope")
    if result.get("model_promoted") is not False:
        raise ValueError("This export is for unpromoted development experiments")
    rows = json.loads((root / "predictions.json").read_text(encoding="utf-8"))
    return result, rows, {
        "completed_at_utc": status["completed_at_utc"],
        "source_sha256": {name: digest(root / name) for name in
                          ("status.json", "results.json", "predictions.json", "protocol.json")},
    }


def export_run(root, run_id, schema, metric_key):
    result, rows, provenance = read_completed(root)
    if schema is not None and result.get("schema") != schema:
        raise ValueError(f"Unsupported schema for {run_id}")
    metrics = result[metric_key]
    if tuple(metrics) != EXPECTED_MODELS[run_id]:
        raise ValueError("Model set changed; review the export contract before publishing")
    selected = [r for r in rows if r.get("partition", r.get("role")) == "development_validation"]
    if not selected or len({r["sample_id"] for r in selected}) != len(selected):
        raise ValueError("Validation rows must be nonempty and unique")
    groups = {g: f"g{i + 1:02}" for i, g in enumerate(sorted({r["group_id"] for r in selected}))}
    public_rows = []
    for i, row in enumerate(selected):
        if run_id == "pilot":
            # The pilot's training majority is derived from training targets only.
            train = [r["target"] for r in rows if r["partition"] == "development_train"]
            majority = int(sum(train) > len(train) / 2)
            predictions = {"train_majority_baseline": majority,
                           "frozen_zero_shot_baseline": row["zero_shot_prediction"],
                           "trained_logistic_probe": row["trained_prediction"]}
            scores = {"trained_logistic_probe": row["trained_score_uncalibrated"]}
        elif run_id == "calibration":
            scores = {k: row["scores"][k] for k in metrics}
            predictions = {k: int(v >= .5) for k, v in scores.items()}
        else:
            predictions = {k: row["predictions"][k] for k in metrics}
            scores = (row["scores_uncalibrated"] if run_id == "round3" else {
                "frozen_first_pilot": row["frozen_score_uncalibrated"],
                "expanded_classifier": row["expanded_score_uncalibrated"]})
        public_rows.append({"case": f"v{i + 1:03}", "group": groups[row["group_id"]],
                            "target": row.get("target", row.get("person")),
                            "predictions": predictions, "scores": scores})

    # Verify that the anonymous export reproduces the saved report before writing it.
    for model, saved in metrics.items():
        actual = classification(public_rows, model)
        expected = saved.get("classification_at_fixed_half", saved)
        if actual["confusion"] != expected["confusion"]:
            raise ValueError(f"Confusion counts changed: {run_id}/{model}")
        if actual["source_groups"] != expected["source_groups"]:
            raise ValueError("Source group denominator changed")
        if actual["fully_correct_groups"] != expected["all_views_correct_groups"]:
            raise ValueError("Fully correct group count changed")
        if any(abs(a - b) > 1e-10 for a, b in zip(actual["group_exact_95"],
                expected["all_views_correct_group_rate_95_exact_interval"])):
            raise ValueError("Group confidence bounds changed")
        if run_id == "calibration":
            p = probability_metrics(public_rows, model)
            for key in ("group_weighted_brier", "group_weighted_log_loss"):
                if abs(p[key] - saved[key]) > 1e-10:
                    raise ValueError(f"Probability metric changed: {key}")

    coverage = result.get("coverage", result.get("split_counts"))
    # Numeric fields only. Do not carry free-form source text or approval identities.
    clean_coverage = {}
    roles = ("classifier_fit", "temperature_fit", "development_validation") if run_id == "calibration" else ("development_train", "development_validation")
    for role in roles:
        counts = coverage[role]
        clean_coverage[role] = {"clips": int(counts["clips"]),
                                "groups": int(counts.get("groups", counts.get("source_groups")))}
    exported = {"id": run_id, "provenance": provenance, "coverage": clean_coverage,
                "evidence_type": "selected_synthetic_development",
                "independent_evaluation_used": False, "model_promoted": False,
                "models": list(metrics), "rows": public_rows}
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    encoder = protocol["encoder"]
    exported["encoder"] = {"model": encoder["model"], "revision": encoder["revision"]}
    if exported["encoder"] != {"model": "google/siglip2-base-patch16-224",
                                "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"}:
        raise ValueError("Encoder changed; review the historical report")
    if run_id == "calibration":
        if result["calibrated_model_accepted"] is not False:
            raise ValueError("Expected the unaccepted calibration experiment")
        exported["temperature"] = result["temperature"]["temperature"]
        exported["validation_previously_exposed"] = True
        exported["calibrator_accepted"] = False
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "round2", "round3", "calibration"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    specs = [("pilot", "approved_person_development_results_v1", "development_validation"),
             ("round2", "expanded_person_development_results_v1", "new_validation_metrics"),
             ("round3", "person_round3_results_v1", "validation_metrics"),
             ("calibration", None, "development_validation_metrics")]
    result = {"schema": "public_development_snapshot_v1",
              "scope": "Recompute historical development metrics; not retraining or independent evaluation.",
              "group_ids_are_local_to_each_run": True,
              "runs": [export_run(getattr(args, name), name, schema, key)
                       for name, schema, key in specs]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Exported four completed development runs with anonymous validation rows.")


if __name__ == "__main__":
    main()
