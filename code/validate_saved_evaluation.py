"""Recompute a saved matched evaluation without loading either language model."""

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / "json" / "finetune_evaluation_results.json"
DEFAULT_HOLDOUT = ROOT / "data" / "scaled_benchmark_holdout.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_agent_v2 import compute_historical_metrics, compute_metrics
from audit_outcomes import SCORING_SCHEMA, format_metric, label_from_value, prediction_from_status
from scaled_benchmark import exact_mcnemar, wilson_interval
from train_lora_dpo import file_sha256


# Compatibility is limited to this exact published historical report. New runs
# use the versioned scoring contract; archived metrics are never rewritten.
LEGACY_REPORT_SHA256_LF = "f4397c84dddd5ec7e9db56b9a7da9da677eb57f12ee94fa26ed33c009048667b"


def validate_case_binding(report, holdout):
    with holdout.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = {row["Log_ID"]: row for row in rows}
    if not rows or len(targets) != len(rows) or "" in targets:
        raise ValueError("Holdout cases must have unique nonempty IDs.")
    order = list(targets)
    for strategy in ("base", "qlora_sft"):
        results = report["strategies"][strategy]["results"]
        by_id = {result.get("log_id"): result for result in results}
        if len(by_id) != len(results) or set(by_id) != set(targets):
            raise ValueError("Saved case IDs do not match the holdout exactly.")
        for case, result in by_id.items():
            target = targets[case]
            actual = label_from_value(target.get("Is_Violation"))
            expected = target.get("Expected_Citation", "NONE")
            if actual is None or result.get("actual") != actual or result.get("expected_citation") != expected:
                raise ValueError("Saved targets do not match the holdout labels and citations.")
            if result.get("status") not in {"CLEAR", "VIOLATION", "REVIEW", "UNPARSED", "INPUT_ERROR", "INFERENCE_ERROR"}:
                raise ValueError("Saved output has an unsupported status.")
            if result.get("predicted") != prediction_from_status(result.get("status")):
                raise ValueError("Saved prediction contradicts its status.")
            citation_correct = (result.get("citation") == expected if expected != "NONE"
                                else result.get("citation") is None and result.get("status") == "CLEAR")
            if result.get("citation_correct") is not citation_correct:
                raise ValueError("Saved citation correctness contradicts the case evidence.")
        report["strategies"][strategy]["results"] = [by_id[case] for case in order]


def recompute_strategy(payload, historical=False):
    results = payload["results"]
    for result in results:
        result["audited_by"] = "model"
        expected = result.get("expected_citation", "NONE")
        result["expected_section"] = expected if expected != "NONE" else None

    metrics = compute_historical_metrics(results) if historical else compute_metrics(results)
    violations = [result for result in results if result["actual"] == 1]
    correct = sum(result.get("predicted") == result["actual"] for result in results)
    metrics.update(
        {
            "citation_correctness": round(
                sum(bool(result.get("citation_correct")) for result in violations)
                / len(violations),
                4,
            ) if violations else None,
            "all_case_verdict_and_citation_accuracy": round(
                sum(
                    result.get("predicted") == result["actual"]
                    and bool(result.get("citation_correct"))
                    for result in results
                )
                / len(results),
                4,
            ),
            "accuracy_95pct_wilson": wilson_interval(correct, len(results)),
            "runtime_seconds": payload["metrics"].get("runtime_seconds"),
            "peak_gpu_memory_mib": payload["metrics"].get("peak_gpu_memory_mib"),
        }
    )
    payload["metrics"] = metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate saved fine-tuning outputs and metrics."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--adapter", type=Path, help="Also verify a locally available adapter configuration; weights are not included in the repository")
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    historical = hashlib.sha256(args.report.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == LEGACY_REPORT_SHA256_LF
    if not historical and any(
        report["strategies"][strategy]["metrics"].get("scoring_schema") != SCORING_SCHEMA
        for strategy in ("base", "qlora_sft")
    ):
        raise ValueError("New reports require metrics with the current scoring schema; legacy compatibility is limited to the frozen published report.")
    validate_case_binding(report, args.holdout)
    original = copy.deepcopy(report)
    for strategy in ("base", "qlora_sft"):
        recompute_strategy(report["strategies"][strategy], historical=historical)
        if report["strategies"][strategy]["metrics"] != original["strategies"][strategy]["metrics"]:
            raise ValueError(f"Saved {strategy} metrics do not match the per-case outputs.")

    expected_holdout_hash = file_sha256(args.holdout)
    if args.adapter is not None:
        expected_adapter_hash = file_sha256(args.adapter / "adapter_config.json")
        if report["provenance"].get("adapter_config_sha256") != expected_adapter_hash:
            raise ValueError("The saved report does not match this adapter configuration.")
    if report["provenance"].get("holdout_sha256") != expected_holdout_hash:
        raise ValueError("The saved report does not match this holdout file.")

    report["comparison"] = exact_mcnemar(
        report["strategies"]["base"]["results"],
        report["strategies"]["qlora_sft"]["results"],
    )
    if report["comparison"] != original["comparison"]:
        raise ValueError("Saved paired comparison does not match the per-case outputs.")
    print("Saved metrics and holdout hash verified; no files changed.")
    print("Historical scoring compatibility: frozen published report only." if historical else "Current versioned scoring verified.")
    if args.adapter is None:
        print("Adapter configuration not checked; supply --adapter to verify it. Model weights are not verified by this command.")

    for strategy in ("base", "qlora_sft"):
        metrics = report["strategies"][strategy]["metrics"]
        print(
            f"{strategy:10} accuracy={format_metric(metrics['accuracy'])} "
            f"f1={format_metric(metrics['f1'])} citation={format_metric(metrics['citation_correctness'])} "
            f"joint={format_metric(metrics['all_case_verdict_and_citation_accuracy'])}"
        )
    print(json.dumps(report["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
