"""Scope-aware deterministic controls for high-consequence audit decisions.

The policy separates classification from narration. It selects a governing
section only when the event contains both the rule's scope and its measurable
hazard condition. Ambiguous events are sent to human review instead of being
forced into a binary answer. A language model may explain these decisions, but
it is not allowed to change their status or citation.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PolicyDecision:
    status: str
    citation: Optional[str]
    reason: str
    rule: Optional[str]
    evidence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _contains(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _combined_text(row: Dict[str, Any]) -> str:
    return " ".join(
        str(row.get(field, ""))
        for field in ("Equipment_Type", "Location", "Reported_Incident")
    ).lower()


def _decision(
    status: str,
    citation: Optional[str],
    reason: str,
    rule: str,
    evidence: List[str],
) -> PolicyDecision:
    return PolicyDecision(status, citation, reason, rule, evidence)


def evaluate_event(row: Dict[str, Any]) -> PolicyDecision:
    """Return a conservative policy verdict for one operational event."""

    text = _combined_text(row)
    incident = str(row.get("Reported_Incident", "")).lower()
    if _contains(incident, (
        "does not show", "doesn't show", "cannot determine", "cannot confirm",
        "can't determine", "can't confirm", "unknown", "uncertain", "unclear",
        "not known", "not confirmed", "not verified", "unverified",
        "whether", "might be", "may be", "suspected",
    )):
        return PolicyDecision(
            "REVIEW", None,
            "The observation explicitly leaves a relevant fact uncertain; confirm it before assigning a citation.",
            "uncertain_evidence", [],
        )
    try:
        shift_hours = float(row.get("Operator_Shift_Hours"))
        if not math.isfinite(shift_hours) or shift_hours < 0:
            shift_hours = None
    except (TypeError, ValueError):
        shift_hours = None

    candidates: List[PolicyDecision] = []

    passenger_scope = _contains(
        text, ("commuter", "intercity", "passenger train")
    ) and _contains(text, ("train employee", "covered service", "passenger train"))
    if passenger_scope:
        if shift_hours is None:
            return PolicyDecision(
                "REVIEW", None, "A valid duty duration is needed for the covered employee.",
                "missing_duty_duration", [],
            )
        if shift_hours > 12.0:
            candidates.append(
                _decision(
                    "VIOLATION",
                    "228.405",
                    "A passenger train employee remained in covered service beyond 12 consecutive hours.",
                    "passenger_hours_of_service",
                    ["passenger-rail scope", f"shift={shift_hours:.2f} hours"],
                )
            )
        else:
            candidates.append(
                _decision(
                    "CLEAR",
                    None,
                    "The covered passenger-rail duty period did not exceed 12 consecutive hours.",
                    "passenger_hours_of_service",
                    ["passenger-rail scope", f"shift={shift_hours:.2f} hours"],
                )
            )

    train_employee_scope = (
        not passenger_scope
        and _contains(
            text,
            (
                "train employee",
                "switch employee",
                "switch crew",
                "switch engine",
                "covered switching service",
            ),
        )
        and _contains(
            text,
            (
                "freight",
                "switch",
                "classification yard",
                "covered service",
                "duty",
                "relief",
            ),
        )
    )
    if train_employee_scope:
        if shift_hours is None:
            return PolicyDecision(
                "REVIEW", None, "A valid duty duration is needed for the covered employee.",
                "missing_duty_duration", [],
            )
        if shift_hours > 12.0:
            candidates.append(
                _decision(
                    "VIOLATION",
                    "21103",
                    "A covered train employee remained on duty beyond 12 consecutive hours.",
                    "train_employee_hours_of_service",
                    [
                        "freight or switching train-employee scope",
                        f"shift={shift_hours:.2f} hours",
                    ],
                )
            )
        else:
            candidates.append(
                _decision(
                    "CLEAR",
                    None,
                    "The covered train employee's duty period did not exceed 12 consecutive hours.",
                    "train_employee_hours_of_service",
                    [
                        "freight or switching train-employee scope",
                        f"shift={shift_hours:.2f} hours",
                    ],
                )
            )

    walking_surface = _contains(
        text,
        (
            "walkway",
            "walking route",
            "crosswalk",
            "footpath",
            "pedestrian",
            "personnel aisle",
        ),
    )
    fluid = _contains(
        incident,
        (
            "oil",
            "fluid",
            "leak",
            "spill",
            "grease",
            "coolant",
            "diesel",
            "rainwater",
            "wash water",
            "slick",
            "sheen",
        ),
    )
    surface_control = _contains(
        incident,
        (
            "cleaned",
            "cleanup",
            "dried",
            "left it dry",
            "contained",
            "captured",
            "separator",
            "drained",
            "drip pan",
            "sump",
            "barricaded",
            "alternate dry",
            "sealed",
            "without releasing",
            "curbed treatment bay",
            "remained inside",
        ),
    )
    if walking_surface and (fluid or surface_control):
        controlled = surface_control and "uncontained" not in incident
        candidates.append(
            _decision(
                "CLEAR" if controlled else "VIOLATION",
                None if controlled else "1910.22",
                "The walking surface was isolated or restored before employee use."
                if controlled
                else "An uncontained contaminant remained on an active employee walking surface.",
                "walking_surface_housekeeping",
                ["employee walking surface", "fluid or slip condition"],
            )
        )

    electrical_context = _contains(
        text,
        (
            "energized",
            "live line",
            "live 25",
            "live overhead",
            "live bus",
            "catenary",
            "busbar",
            "conductor",
            "voltage",
        ),
    )
    if electrical_context:
        controlled = _contains(
            incident,
            (
                "de-energized",
                "lockout",
                "locked out",
                "grounded",
                "safe distance",
                "outside",
                "barrier",
                "stopped at",
                "approved envelope",
            ),
        )
        exposed = _contains(
            incident,
            (
                "inside",
                "within",
                "four feet",
                "six feet",
                "approach",
                "prohibited",
                "unguarded",
                "exposed",
                "drifted toward",
                "beside",
            ),
        )
        if controlled or exposed:
            candidates.append(
                _decision(
                    "CLEAR" if controlled else "VIOLATION",
                    None if controlled else "1910.333",
                    "The event states that energy isolation or an engineered approach control was maintained."
                    if controlled
                    else "Equipment or personnel entered the restricted area around exposed energized parts.",
                    "electrical_work_practices",
                    [
                        "energized electrical context",
                        "control maintained" if controlled else "restricted approach",
                    ],
                )
            )

    sling_context = _contains(
        text,
        (
            "sling",
            "rigging",
            "wire rope",
            "chain link",
            "chain inspection",
            "eye fitting",
            "end attachment",
        ),
    )
    if sling_context:
        controlled = _contains(
            incident,
            (
                "removed from service",
                "tagged out",
                "replaced",
                "undamaged",
                "no visible damage",
                "approved",
                "acceptable",
                "within capacity",
                "within its marked",
                "legible rating",
                "legibly tagged",
            ),
        )
        damaged = _contains(
            incident,
            (
                "broken",
                "melted",
                "crushed",
                "stretched",
                "abraded",
                "kinked",
                "heat-damaged",
                "cracked",
                "elongated",
                "missing",
                "rejection",
            ),
        )
        if controlled or damaged:
            candidates.append(
                _decision(
                    "CLEAR" if controlled else "VIOLATION",
                    None if controlled else "1910.184",
                    "The sling was verified serviceable or removed before use."
                    if controlled
                    else "Damaged or unidentified sling components remained in lifting service.",
                    "slings",
                    [
                        "lifting sling or rigging",
                        "serviceable control"
                        if controlled
                        else "damage while in service",
                    ],
                )
            )

    aisle_context = _contains(
        text, ("aisle", "passage", "route to the exit", "employee access")
    )
    material_context = _contains(
        text,
        (
            "pallet",
            "cargo",
            "freight",
            "container",
            "crate",
            "parts",
            "load",
            "material",
            "racks",
        ),
    )
    if aisle_context and material_context:
        controlled = _contains(
            incident,
            (
                "stayed clear",
                "stayed unobstructed",
                "remained open",
                "preserved",
                "inside",
                "without extending",
                "without projecting",
                "restored full",
                "behind the floor line",
            ),
        )
        obstructed = _contains(
            incident,
            (
                "blocked",
                "obstruct",
                "projected",
                "across",
                "narrowed",
                "prevented",
                "closed",
                "spilled into",
                "eliminated",
            ),
        )
        if controlled or obstructed:
            candidates.append(
                _decision(
                    "CLEAR" if controlled else "VIOLATION",
                    None if controlled else "1910.176",
                    "Stored material remained within its boundary and preserved aisle clearance."
                    if controlled
                    else "Stored material obstructed a permanent aisle or passageway.",
                    "materials_handling_aisles",
                    [
                        "materials storage",
                        "clear aisle" if controlled else "obstructed passage",
                    ],
                )
            )

    dockboard_context = _contains(
        text,
        ("dockboard", "dock plate", "bridge plate", "dock leveler", "portable plate"),
    ) or (
        "loading dock" in text
        and _contains(incident, ("plate", "anchoring", "bearing"))
    )
    if dockboard_context:
        unsafe = _contains(
            incident,
            (
                "unsecured",
                "no anchoring",
                "never secured",
                "without positive securing",
                "loose",
                "slid",
                "shifted",
                "moved",
                "slipped",
                "crept",
                "inadequate anchoring",
                "lacked safe handholds",
            ),
        )
        controlled = not unsafe and _contains(
            incident,
            (
                "secured",
                "anchored",
                "anchors were engaged",
                "locked",
                "positive restraint",
                "positive securing",
                "wheel restraints",
                "confirmed",
                "capacity",
                "adequate bearing",
                "bearing, and anchoring",
            ),
        )
        if controlled or unsafe:
            candidates.append(
                _decision(
                    "CLEAR" if controlled else "VIOLATION",
                    None if controlled else "1910.26",
                    "The dockboard was secured and its capacity or bearing was verified."
                    if controlled
                    else "A portable dockboard was used without adequate securing against displacement.",
                    "dockboards",
                    [
                        "dockboard transfer",
                        "secured control"
                        if controlled
                        else "movement or missing securing",
                    ],
                )
            )

    fall_context = _contains(
        text, ("platform", "catwalk", "edge", "drop", "opening", "scaffold")
    )
    if fall_context:
        controlled = _contains(
            incident,
            (
                "complete guardrail",
                "inspected guardrail",
                "self-closing gate",
                "connected restraint",
                "travel-restraint",
                "fall-arrest",
                "guardrails enclosed",
                "edge protection",
                "protected every open",
                "temporary rails",
                "guarded work",
                "enclosed the work area",
                "prevented the employee from reaching",
            ),
        )
        exposed = _contains(
            incident,
            (
                "without",
                "missing",
                "absent",
                "lacked",
                "unprotected",
                "open edge",
                "open-sided",
                "no guardrail",
                "no restraint",
            ),
        )
        if controlled or exposed:
            candidates.append(
                _decision(
                    "CLEAR" if controlled else "VIOLATION",
                    None if controlled else "1910.28",
                    "A guardrail, restraint, or fall-arrest system controlled the elevated exposure."
                    if controlled
                    else "Work occurred at an elevated unprotected edge without a fall-protection system.",
                    "fall_protection",
                    [
                        "elevated exposure",
                        "fall control" if controlled else "missing fall control",
                    ],
                )
            )

    if not candidates:
        return PolicyDecision(
            "REVIEW",
            None,
            "The event does not contain enough scoped evidence for an automatic federal citation.",
            None,
            [],
        )

    unique = {(item.status, item.citation) for item in candidates}
    if len(unique) > 1:
        return PolicyDecision(
            "REVIEW",
            None,
            "The event contains multiple or conflicting regulated conditions and requires review.",
            None,
            [evidence for item in candidates for evidence in item.evidence],
        )
    return candidates[0]
