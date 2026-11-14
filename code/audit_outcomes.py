"""Outcome conversion and metrics that keep unresolved decisions visible.

CLEAR and VIOLATION are the only binary decisions. A review, invalid response,
or inference failure is unresolved. Strict accuracy counts an unresolved labeled
case as incorrect, without pretending that it was a positive or negative alert.
"""

from __future__ import annotations

import math
import re
from numbers import Real
from typing import Any, Iterable, Mapping

SCORING_SCHEMA = "selective_binary_v2"


def format_metric(value: float | None, spec: str = ".4f") -> str:
    """Show an undefined rate as n/a; keep the saved metric unchanged."""
    return "n/a" if value is None else format(value, spec)


def strictly_correct(result: Mapping[str, Any]) -> bool:
    """A missing label or unresolved decision cannot count as a correct case."""
    actual = label_from_value(result.get("actual"))
    return actual is not None and prediction_from_record(result) == actual


def parse_audit_response(text: Any) -> dict[str, Any]:
    """Parse one complete three-field reply; malformed replies stay unresolved."""
    raw = text if isinstance(text, str) else ""
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"<think>.*?</think>", "", normalized, flags=re.DOTALL | re.I).strip()
    # Searching for three valid fields inside a larger reply can mistake a
    # quoted example for the answer and ignore a contradictory final verdict.
    # Only blank lines may surround or separate the complete ordered frame.
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    envelope = len(lines) == 3 and all(
        re.fullmatch(rf"{name}:[ \t]*[^\r\n]+", line, re.I)
        for name, line in zip(("STATUS", "CITATION", "REASON"), lines)
    )
    cleaned = "\n".join(lines)
    fields = {
        name: re.findall(rf"^{name}:[ \t]*([^\r\n]*)$", cleaned, re.I | re.M)
        for name in ("STATUS", "CITATION", "REASON")
    }
    unique = all(len(values) == 1 and values[0].strip() for values in fields.values())
    status = fields["STATUS"][0].strip().upper() if len(fields["STATUS"]) == 1 else "UNPARSED"
    cite = fields["CITATION"][0].strip() if len(fields["CITATION"]) == 1 else "NONE"
    reason = fields["REASON"][0].strip() if len(fields["REASON"]) == 1 else ""
    section = re.fullmatch(
        r"(?:(?:(?:29|49)\s*C\.?\s*F\.?\s*R\.?\s*)?(?:§\s*)?(\d{3,4}\.\d+)"
        r"|(?:49\s*U\.?\s*S\.?\s*C\.?\s*)?(?:§\s*)?(21103))(?:\([a-z0-9]+\))*",
        cite, re.I,
    )
    valid_citation = cite.upper() == "NONE" or section is not None
    valid = envelope and unique and status in {"CLEAR", "VIOLATION", "REVIEW"} and valid_citation and "<think" not in cleaned.lower()
    return {
        "status": status if valid else "UNPARSED",
        "citation_raw": cite,
        "citation": (section.group(1) or section.group(2)) if section else None,
        "reason": reason,
        "raw_response": raw,
    }


def prediction_from_status(status: Any) -> int | None:
    """Return a binary decision only for an explicit supported verdict."""
    return {"CLEAR": 0, "VIOLATION": 1}.get(str(status).strip().upper())


def label_from_value(value: Any) -> int | None:
    """Preserve absent labels; reject values that cannot be a binary target."""
    if value is None or isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, str):
        value = value.strip()
        if value in {"0", "1"}:
            return int(value)
    elif isinstance(value, Real):
        if math.isnan(value):
            return None
        if value in (0, 1):
            return int(value)
    raise ValueError("The actual label must be 0, 1, or missing.")


def prediction_from_record(result: Mapping[str, Any]) -> int | None:
    """A stored unresolved status cannot be overridden by stale predicted=0."""
    if "status" in result:
        return prediction_from_status(result["status"])
    # Metrics-only records may contain a binary prediction without a verdict.
    return label_from_value(result.get("predicted"))


def guard_grounded_decision(
    parsed: Mapping[str, Any],
    retrieved_sections: Iterable[str],
    valid_sections: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Enforce the evidence boundary after any model or critique pass.

    This checks evidence availability and citation membership, not whether a
    cited rule entails the answer. Preserve the unguarded response for diagnosis.
    """
    result = dict(parsed)
    status = str(result.get("status", "UNPARSED")).strip().upper()
    allowed = set(retrieved_sections)
    citation = result.get("citation")
    problem = None
    if status in {"CLEAR", "VIOLATION"} and not allowed:
        problem = "No governing excerpts were available; the event requires review."
    elif status == "VIOLATION" and (
        citation not in allowed or valid_sections is not None and citation not in set(valid_sections)
    ):
        problem = "The proposed violation lacks a valid citation in the supplied excerpts."
    elif status == "CLEAR" and citation is not None:
        problem = "The clearance contradicts its nonempty citation and requires review."
    if problem is not None:
        result.update(
            unguarded_status=result.get("status"),
            unguarded_citation=citation,
            unguarded_citation_raw=result.get("citation_raw"),
            unguarded_reason=result.get("reason"),
            status="REVIEW", citation=None, citation_raw="NONE", reason=problem,
            decision_source="grounding_guard",
        )
    return result


def score_outcomes(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score labeled cases strictly and describe unresolved/unlabeled coverage.

    Confusion cells contain explicit binary decisions only. Recall includes
    unresolved positives in its denominator; ``missed_positive_cases`` exposes
    that total. Review and inference failures remain separate counts. Rates
    for accuracy and coverage are None without an eligible denominator. For
    labeled cohorts, precision/recall/F1 use zero for a zero denominator; for a
    wholly unlabeled cohort all supervised metrics are None.
    """
    rows = list(results)
    outcomes = [(label_from_value(r.get("actual")), prediction_from_record(r)) for r in rows]
    labeled = [(actual, pred) for actual, pred in outcomes if actual is not None]
    decided = [(actual, pred) for actual, pred in labeled if pred is not None]
    tp = sum(actual == 1 and pred == 1 for actual, pred in labeled)
    fp = sum(actual == 0 and pred == 1 for actual, pred in labeled)
    fn = sum(actual == 1 and pred == 0 for actual, pred in labeled)
    tn = sum(actual == 0 and pred == 0 for actual, pred in labeled)
    unresolved_positive = sum(actual == 1 and pred is None for actual, pred in labeled)
    unresolved_negative = sum(actual == 0 and pred is None for actual, pred in labeled)
    unresolved = sum(pred is None for _, pred in outcomes)
    reviews = sum(str(r.get("status", "")).strip().upper() == "REVIEW" for r in rows)
    statuses = [str(r.get("status", "")).strip().upper() for r in rows]
    input_errors = statuses.count("INPUT_ERROR")
    unparsed = statuses.count("UNPARSED")
    # Unrecognized outcomes are failures, rather than silently becoming reviews.
    inference_errors = sum(
        pred is None and str(row.get("status", "")).strip().upper() not in {"REVIEW", "INPUT_ERROR"}
        for row, (_, pred) in zip(rows, outcomes)
    )

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    precision = tp / (tp + fp) if tp + fp else 0.0
    positives = tp + fn + unresolved_positive
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "scoring_schema": SCORING_SCHEMA,
        "n": len(rows),
        "n_labeled": len(labeled),
        "n_unlabeled": len(rows) - len(labeled),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4) if labeled else None,
        "recall": round(recall, 4) if labeled else None,
        "f1": round(f1, 4) if labeled else None,
        "accuracy": ratio(tp + tn, len(labeled)),
        "strict_correct": tp + tn,
        "strict_errors": len(labeled) - tp - tn,
        "decided_labeled_cases": len(decided),
        "decided_accuracy": ratio(tp + tn, len(decided)),
        "labeled_decision_coverage": ratio(len(decided), len(labeled)),
        "automation_coverage": ratio(len(rows) - unresolved, len(rows)),
        "unresolved_count": unresolved,
        "unresolved_positive_cases": unresolved_positive,
        "unresolved_negative_cases": unresolved_negative,
        "missed_positive_cases": fn + unresolved_positive,
        "review_count": reviews,
        "input_error_count": input_errors,
        "inference_error_count": inference_errors,
        "unparsed": unparsed,
    }
