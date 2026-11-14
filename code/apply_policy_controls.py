"""Apply deterministic safety-policy controls to a saved model audit.

This is a post-processing evaluation. It does not rerun the language model.
Unresolved policy decisions remain REVIEW. The saved model verdict is retained
as diagnostic evidence, not used to turn missing evidence into a clearance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from hybrid_policy import evaluate_event
from audit_outcomes import label_from_value, prediction_from_record, prediction_from_status, score_outcomes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def citation_matches_target(result: dict[str, Any]) -> bool:
    """A matching absence is not a correct violation citation."""
    expected = result.get("expected_section")
    return (
        isinstance(expected, str)
        and re.fullmatch(r"(?:\d{3,4}\.\d+|21103)(?:\([a-z0-9]+\))*", expected) is not None
        and result.get("citation") == expected
    )


def score(results: list[dict[str, Any]]) -> dict[str, Any]:
    results = [dict(row, actual=label_from_value(row.get("actual")),
                    predicted=prediction_from_record(row)) for row in results]
    tp = sum(r["actual"] == 1 and r["predicted"] == 1 for r in results)
    correct_citations = sum(
        r["actual"] == 1
        and r["predicted"] == 1
        and citation_matches_target(r)
        for r in results
    )
    joint_correct = sum(
        (r["actual"] == 0 and r["predicted"] == 0)
        or (
            r["actual"] == 1
            and r["predicted"] == 1
            and citation_matches_target(r)
        )
        for r in results
    )
    metrics = score_outcomes(results)
    labeled = metrics["n_labeled"]
    return {
        **metrics,
        "citation_correctness": round(safe_div(correct_citations, tp), 4),
        "correct_citations_on_true_positives": correct_citations,
        "joint_status_and_citation_accuracy": round(
            joint_correct / labeled, 4
        ) if labeled else None,
    }


def per_hazard(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results = [dict(row, actual=label_from_value(row.get("actual")),
                    predicted=prediction_from_record(row)) for row in results]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result["actual"] == 1:
            grouped[str(result.get("actual_type", "unknown"))].append(result)

    output = {}
    for hazard, rows in sorted(grouped.items()):
        detected = sum(r["predicted"] == 1 for r in rows)
        citations = sum(
            r["predicted"] == 1 and citation_matches_target(r)
            for r in rows
        )
        output[hazard] = {
            "cases": len(rows),
            "detected": detected,
            "recall": round(safe_div(detected, len(rows)), 4),
            "correct_citations": citations,
        }
    return output


def apply_controls(
    raw_report_path: Path, dataset_path: Path, output_path: Path
) -> dict[str, Any]:
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    rows = {row["Log_ID"]: row for row in input_rows}
    if len(rows) != len(input_rows) or "" in rows:
        raise ValueError("Policy input cases must have unique nonempty IDs.")
    saved_ids = [result.get("log_id") for result in raw_report["results"]]
    if len(set(saved_ids)) != len(saved_ids) or any(case not in rows for case in saved_ids):
        raise ValueError("Saved output cases must be unique and match the policy inputs.")

    controlled_results = []
    baseline_results = []
    policy_controlled = 0
    overrides = 0
    for raw in raw_report["results"]:
        result = dict(raw)
        row = rows[result["log_id"]]
        decision = evaluate_event(row)
        result["model_status"] = raw.get("status")
        result["model_citation"] = raw.get("citation")
        result["model_reason"] = raw.get("reason")
        result["actual"] = label_from_value(row.get("Is_Violation"))
        result["actual_type"] = row.get("Violation_Type", "unknown")
        result["expected_section"] = row.get("Expected_Section", row.get("Expected_Citation"))
        baseline_results.append(dict(result, predicted=prediction_from_record(raw)))

        if decision.status == "REVIEW":
            overrides += int(raw.get("status") != "REVIEW")
            result.update(status="REVIEW", citation=None, citation_raw="NONE",
                          reason=decision.reason, predicted=None,
                          decision_source="policy_review", policy_rule=decision.rule,
                          policy_evidence=decision.evidence, confidence=None,
                          citation_exists=None, citation_was_retrieved=None)
        else:
            policy_controlled += 1
            changed = decision.status != raw.get(
                "status"
            ) or decision.citation != raw.get("citation")
            overrides += int(changed)
            result.update(
                {
                    "status": decision.status,
                    "citation": decision.citation,
                    "citation_raw": decision.citation or "NONE",
                    "reason": decision.reason,
                    "predicted": prediction_from_status(decision.status),
                    "decision_source": "deterministic_policy",
                    "policy_rule": decision.rule,
                    "policy_evidence": decision.evidence,
                    "citation_exists": True if decision.citation else None,
                    "citation_was_retrieved": (
                        decision.citation in raw.get("retrieved", [])
                        if decision.citation
                        else None
                    ),
                    "confidence": None,
                }
            )
        controlled_results.append(result)

    report = {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "method": "deterministic policy control over saved model outputs",
            "model_was_rerun": False,
            "raw_report": str(raw_report_path.relative_to(ROOT)),
            "raw_report_sha256": sha256(raw_report_path),
            "dataset": str(dataset_path.relative_to(ROOT)),
            "dataset_sha256": sha256(dataset_path),
            "policy": "code/hybrid_policy.py",
            "policy_sha256": sha256(ROOT / "code" / "hybrid_policy.py"),
            "raw_run_id": raw_report.get("metrics", {}).get("run_id"),
        },
        "raw_model_metrics": score(baseline_results),
        "historical_raw_model_metrics": raw_report.get("metrics", {}),
        "controlled_metrics": {
            **score(controlled_results),
            "policy_controlled_cases": policy_controlled,
            "model_fallback_cases": 0,
            "policy_review_cases": len(controlled_results) - policy_controlled,
            "policy_overrides": overrides,
        },
        "per_hazard": per_hazard(controlled_results),
        "results": controlled_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "json" / "full_audit_results.json"
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "daily_yard_log.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "json" / "full_audit_policy_controlled.json",
    )
    args = parser.parse_args()
    report = apply_controls(
        args.raw.resolve(), args.dataset.resolve(), args.out.resolve()
    )
    print(json.dumps(report["controlled_metrics"], indent=2))


if __name__ == "__main__":
    main()
