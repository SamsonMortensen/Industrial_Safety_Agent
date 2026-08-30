"""Trust policy for autonomous compliance learning.

Grounded citations are necessary, but they do not prove that an audit decision is
correct. This module keeps model-generated decisions out of trusted memory until
they agree with labeled ground truth or an explicit reviewer decision.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Set

EXPECTED_CITATIONS: Dict[str, Set[Optional[str]]] = {
    "none": {None},
    "duty_limit": {"21103"},
    "fatigue": {"21103"},
    "housekeeping": {"1910.22"},
    "spill": {"1910.22"},
    "electrical_clearance": {"1910.333"},
    "electrical": {"1910.333"},
}

APPROVED_REVIEW_STATES = {"approved", "confirmed", "verified"}


@dataclass(frozen=True)
class LearningDecision:
    """Explain whether one episode may influence later audits."""

    trusted: bool
    feedback_type: str
    verification_source: str
    reason: str
    correct_verdict: Optional[str] = None
    correct_citation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_binary_label(value: Any) -> Optional[int]:
    """Return a binary label, or ``None`` when the event is unlabeled."""

    if value is None or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "violation"}:
        return 1
    if normalized in {"0", "false", "no", "clear"}:
        return 0
    return None


def _normalized_citation(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return None if normalized.upper() in {"", "NONE", "N/A"} else normalized


def assess_for_learning(
    row: Dict[str, Any],
    verdict: str,
    citation: Optional[str],
    is_grounded: bool,
) -> LearningDecision:
    """Apply the promotion gate for episodic memory and training data."""

    predicted = 1 if verdict == "VIOLATION" else 0 if verdict == "CLEAR" else None
    citation = _normalized_citation(citation)

    review_state = str(row.get("Review_Status", "")).strip().lower()
    if review_state in APPROVED_REVIEW_STATES:
        reviewed_verdict = str(row.get("Reviewed_Verdict", "")).strip().upper()
        reviewed_citation = _normalized_citation(row.get("Reviewed_Citation"))
        matches_review = verdict == reviewed_verdict and citation == reviewed_citation
        if matches_review and is_grounded:
            return LearningDecision(
                True,
                "verified",
                "human_review",
                "The result matches the approved reviewer decision and is grounded.",
                reviewed_verdict,
                reviewed_citation,
            )
        return LearningDecision(
            False,
            "pitfall",
            "human_review",
            "The result conflicts with the approved reviewer decision or lacks grounding.",
            reviewed_verdict,
            reviewed_citation,
        )

    actual = parse_binary_label(row.get("Is_Violation"))
    violation_type = str(row.get("Violation_Type", "none")).strip().lower()
    expected_field = (
        "Expected_Citation"
        if "Expected_Citation" in row
        else "Expected_Section"
        if "Expected_Section" in row
        else None
    )
    explicit_expected = (
        _normalized_citation(row.get(expected_field)) if expected_field else None
    )
    has_explicit_expected = expected_field is not None
    if actual is not None and (
        has_explicit_expected or violation_type in EXPECTED_CITATIONS
    ):
        expected_citations = (
            {explicit_expected}
            if has_explicit_expected
            else EXPECTED_CITATIONS[violation_type]
        )
        expected_verdict = "VIOLATION" if actual else "CLEAR"
        expected_citation = next(iter(expected_citations))
        matches_label = predicted == actual and citation in expected_citations
        if matches_label and is_grounded:
            return LearningDecision(
                True,
                "verified",
                "ground_truth",
                "The result matches labeled ground truth, the governing citation, and retrieved law.",
                expected_verdict,
                expected_citation,
            )
        return LearningDecision(
            False,
            "pitfall",
            "ground_truth",
            "The result conflicts with labeled ground truth, the governing citation, or retrieved law.",
            expected_verdict,
            expected_citation,
        )

    return LearningDecision(
        False,
        "quarantine",
        "model_only",
        "No independent label or approved review confirms this model-generated decision.",
    )


def episode_is_trusted(episode: Dict[str, Any]) -> bool:
    """Return whether a stored episode is eligible for reuse or training."""

    decision = assess_for_learning(
        row=episode.get("row", {}),
        verdict=str(episode.get("verdict", "")),
        citation=episode.get("citation"),
        is_grounded=bool(episode.get("is_grounded", False)),
    )
    return decision.trusted
