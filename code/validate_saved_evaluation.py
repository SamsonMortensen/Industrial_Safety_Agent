"""Recompute a saved matched evaluation without loading either language model."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / "json" / "finetune_evaluation_results.json"
DEFAULT_ADAPTER = ROOT / "models" / "compliance-auditor-qwen25-3b-lora"
DEFAULT_HOLDOUT = ROOT / "data" / "scaled_benchmark_holdout.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_agent_v2 import compute_metrics
from scaled_benchmark import exact_mcnemar, wilson_interval
from train_lora_dpo import file_sha256


def recompute_strategy(payload):
    results = payload["results"]
    for result in results:
        result["audited_by"] = "model"
        expected = result.get("expected_citation", "NONE")
        result["expected_section"] = expected if expected != "NONE" else None

    metrics = compute_metrics(results)
    violations = [result for result in results if result["actual"] == 1]
    correct = sum(result.get("predicted") == result["actual"] for result in results)
    metrics.update(
        {
            "citation_correctness": round(
                sum(bool(result.get("citation_correct")) for result in violations)
                / len(violations),
                4,
            ),
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
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    for strategy in ("base", "qlora_sft"):
        recompute_strategy(report["strategies"][strategy])

    expected_adapter_hash = file_sha256(args.adapter / "adapter_config.json")
    expected_holdout_hash = file_sha256(args.holdout)
    if report["provenance"].get("adapter_config_sha256") != expected_adapter_hash:
        raise ValueError("The saved report does not match this adapter configuration.")
    if report["provenance"].get("holdout_sha256") != expected_holdout_hash:
        raise ValueError("The saved report does not match this holdout file.")

    report["provenance"].update(
        {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "adapter": "models/compliance-auditor-qwen25-3b-lora",
            "holdout": "data/scaled_benchmark_holdout.csv",
            "metrics_recomputed_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics_recomputed_from_saved_outputs": True,
        }
    )
    report["comparison"] = exact_mcnemar(
        report["strategies"]["base"]["results"],
        report["strategies"]["qlora_sft"]["results"],
    )
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for strategy in ("base", "qlora_sft"):
        metrics = report["strategies"][strategy]["metrics"]
        print(
            f"{strategy:10} accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} citation={metrics['citation_correctness']:.4f} "
            f"joint={metrics['all_case_verdict_and_citation_accuracy']:.4f}"
        )
    print(json.dumps(report["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
