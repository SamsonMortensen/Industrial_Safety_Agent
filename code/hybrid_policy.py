"""Scope-aware deterministic controls for high-consequence audit decisions.

The policy separates classification from narration. It selects a governing
section only when the event contains both the rule's scope and its measurable
hazard condition. Ambiguous events are sent to human review instead of being
forced into a binary answer. A language model may explain these decisions, but
it is not allowed to change their status or citation.
"""

from dataclasses import asdict, dataclass
import math
import re
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


def _normalized_text(value: Any) -> str:
    """Normalize matching text while retaining the original event evidence."""
    text = str(value).lower().translate(str.maketrans({
        "\u2019": "'", "\u2010": "-", "\u2011": "-", "\u2013": "-",
    }))
    return " ".join(text.split())


def _combined_text(row: Dict[str, Any]) -> str:
    return " ".join(
        _normalized_text(row.get(field, ""))
        for field in ("Equipment_Type", "Location", "Reported_Incident")
    )


def _decision(
    status: str,
    citation: Optional[str],
    reason: str,
    rule: str,
    evidence: List[str],
) -> PolicyDecision:
    return PolicyDecision(status, citation, reason, rule, evidence)


@dataclass(frozen=True)
class ControlAssertion:
    """A bounded language check, not a probability or a general language parser."""

    state: str
    evidence: tuple[str, ...]


def _control_matches(incident: str, rule: str):
    # Boundaries avoid matching 'grounded' in 'ungrounded' or 'contained' in
    # 'uncontained'. Plural nouns such as barriers still match their control.
    terms = sorted(CONTROL_TERMS[rule], key=len, reverse=True)
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?:s)?(?!\w)"
    return list(re.finditer(pattern, incident))


CONTROL_TERMS = {'walking_surface_housekeeping': ('cleaned',
                                  'cleanup',
                                  'dried',
                                  'left it dry',
                                  'contained',
                                  'captured',
                                  'separator',
                                  'drained',
                                  'drip pan',
                                  'sump',
                                  'barricaded',
                                  'alternate dry',
                                  'sealed',
                                  'without releasing',
                                  'curbed treatment bay',
                                  'remained inside'),
 'electrical_work_practices': ('de-energized',
                               'lockout',
                               'locked out',
                               'grounded',
                               'safe distance',
                               'outside',
                               'barrier',
                               'stopped at',
                               'approved envelope'),
 'slings': ('removed from service',
            'tagged out',
            'replaced',
            'undamaged',
            'no visible damage',
            'approved',
            'acceptable',
            'within capacity',
            'within its marked',
            'legible rating',
            'legibly tagged'),
 'materials_handling_aisles': ('stayed clear',
                               'stayed unobstructed',
                               'remained open',
                               'preserved',
                               'inside',
                               'without extending',
                               'without projecting',
                               'restored full',
                               'behind the floor line'),
 'fall_protection': ('complete guardrail',
                     'inspected guardrail',
                     'self-closing gate',
                     'connected restraint',
                     'travel-restraint',
                     'fall-arrest',
                     'guardrails enclosed',
                     'edge protection',
                     'protected every open',
                     'temporary rails',
                     'guarded work',
                     'enclosed the work area',
                     'prevented the employee from reaching'),
 'dockboards': ('secured',
                'anchored',
                'anchors were engaged',
                'locked',
                'positive restraint',
                'positive securing',
                'wheel restraints',
                'confirmed',
                'capacity',
                'adequate bearing',
                'bearing, and anchoring')}

CONTROL_SUBJECTS = {'walking_surface_housekeeping': ('\\b(?:walkway|walking|crosswalk|footpath|aisle|spill|leak|oil|fluid|runoff|water|lubricant|hose|containment)\\b',),
 'electrical_work_practices': ('\\b(?:circuit|line|conductor|voltage|energized|crane|boom|substation|electrical|lockout|equipment|approach|barrier)\\b',),
 'slings': ('\\b(?:sling|slings|rigging|chain|rope|rigger|hardware|fitting)\\b',),
 'materials_handling_aisles': ('\\b(?:aisle|passage|exit|access|pallets?|cargo|freight|racks?|loads?|storage|container|staging)\\b',),
 'dockboards': ('\\b(?:dockboard|dockboards|plate|leveler|restraints?|forklift|bearing|anchoring)\\b',),
 'fall_protection': ('\\b(?:platform|edge|fall|guardrails?|restraint|rails|scaffold|elevated|opening|anchorage|worker|employee|work)\\b',)}


CLAUSE_BOUNDARY = re.compile(
    r"[.;!?]|\b(?:but|however|whereas|although|while|yet)\b|"
    r",(?=\s*(?:the|a|an|it|they|he|she)\b)|"
    r"\band\s+(?=(?:the|a|an|it|they|he|she|was|were|is|are)\b)"
)
DENIAL = re.compile(
    r"\b(?:not|never|no|without|neither|nor|lacks?|lacked|lacking|missing|absent|"
    r"failed\s+to|fails?\s+to|unable\s+to|cannot|can't|isn't|wasn't|weren't|"
    r"aren't|hasn't|haven't|hadn't|didn't|don't|doesn't|won't|wouldn't|couldn't|shouldn't|mustn't)\b"
)
NONASSERTED = re.compile(
    r"\b(?:will|would|should|could|must|may|might|needs?|needed|requires?|required|"
    r"planned|plans?|planning|promised|scheduled|requested|pending|awaiting|"
    r"intends?|intended|if|unless)\b"
)
POST_DENIAL = re.compile(
    r"^\s*(?:(?:system|control|equipment|protection|method|barrier|was|were|is|are|"
    r"remained|found|reported|confirmed|has|have|had|been|being|temporarily)\s+){0,5}"
    r"(?:not\b|never\b|absent\b|missing\b|unavailable\b|disconnected\b|failed\b|"
    r"removed\b|dismantled\b|detached\b|disabled\b|bypassed\b|unhooked\b|disengaged\b)"
)
POST_NONASSERTED = re.compile(
    r"^\s*(?:(?:system|control|equipment|protection|was|were|is|are|remained)\s+){0,4}"
    r"(?:available\b|nearby\b|stored\b|planned\b|pending\b|scheduled\b|requested\b|"
    r"required\b|needed\b|to\s+be\b|will\b|would\b|should\b|could\b|must\b|may\b|might\b)"
)
LATE_CONTROL = re.compile(
    r"\b(?:later|eventually|subsequently|tomorrow)\b|\bnext\s+(?:shift|day|week)\b|"
    r"\bafter\s+(?:the\s+)?(?:incident|exposure|lift|crossing|work\s+(?:began|started)|"
    r"employees?\s+(?:entered|crossed|slipped)|people\s+(?:entered|crossed))\b"
)
OTHER_OBJECT = re.compile(r"\b(?:another|different|other|spare|second|unrelated|elsewhere|adjacent|nearby)\b")
INCOMPLETE_CONTROL = re.compile(r"\b(?:partly|partially|incompletely|barely)\b|\bonly\s+in\s+part\b")
# A completed cleanup does not clear a later spill on the same walking route.
# This covers an explicit renewed exposure, not general narrative chronology.
RENEWED_WALKWAY_SPILL = re.compile(
    r"\b(?:fresh|new|additional)\s+(?:oil|fluid|grease|coolant|diesel|water)\b"
    r"[^.;!?]{0,45}\b(?:spilled|leaked|covered|reached|spread|flowed)\b"
    r"[^.;!?]{0,65}\b(?:walkway|walking\s+route|crosswalk|footpath|aisle)\b"
)
CONTINUING_HAZARD = {
    "walking_surface_housekeeping": re.compile(
        r"\b(?:remained|remains|still|continued)\b.{0,45}\b(?:oily|slick|slippery|wet|uncontained|spill|oil)\b"
    ),
    "electrical_work_practices": re.compile(
        r"\b(?:remained|remains|still)\b.{0,35}\b(?:energized|live|exposed|inside)\b"
    ),
    "slings": re.compile(
        r"\b(?:damaged|broken|cracked|defective|abraded)\b.{0,70}"
        r"\b(?:remained|remains|still|continued|kept)\b.{0,40}\b(?:active|use|service|lift|load)\b"
    ),
    "materials_handling_aisles": re.compile(
        r"\b(?:remained|remains|still|continued)\b.{0,35}\b(?:blocked|obstructed|narrowed|closed)\b"
    ),
    "dockboards": re.compile(
        r"\b(?:remained|remains|still|continued)\b.{0,35}\b(?:loose|unsecured|sliding|moving|shifting)\b"
    ),
    "fall_protection": re.compile(
        r"\b(?:remained|remains|still|continued)\b.{0,35}\b(?:unprotected|unguarded|disconnected|exposed)\b"
    ),
}
CONTROL_ACTIONS = {
    "cleaned", "dried", "contained", "captured", "drained", "barricaded", "sealed",
    "de-energized", "locked out", "grounded", "removed from service", "tagged out",
    "replaced", "approved", "preserved", "secured", "anchored", "locked", "confirmed",
}


def _scope_matches(text: str, rule: str) -> bool:
    return any(re.search(pattern, text) for pattern in CONTROL_SUBJECTS[rule])


def _explicit_other_subject(prefix: str, suffix: str, control: str, rule: str) -> bool:
    """Reject an explicitly different controlled object, even near a hazard noun.

    For example, a truck cleaned beside a walkway is not a cleaned walkway.
    This checks common passive and direct-object constructions; ambiguous
    language remains outside the supported automatic clearance contract.
    """
    if control not in CONTROL_ACTIONS:
        return False
    subject = re.search(
        r"(?:^|,|\band\b)([^,;]+?)\s+(?:was|were|is|are|has\s+been|had\s+been)\s*$",
        prefix,
    )
    if subject:
        words = subject.group(1).strip()
        if words not in {"it", "they", "this", "these"}:
            return not _scope_matches(words, rule)
    # An active verb followed by a noun phrase has an explicit object.
    obj = re.match(r"\s+(?:the|a|an)\s+(.+?)(?=\s+(?:before|after|beside|near|while|and|at|in|on)\b|$)", suffix)
    if not subject and obj:
        return not _scope_matches(obj.group(1), rule)
    return False


def _control_assertion(incident: str, rule: str) -> ControlAssertion:
    """Require an affirmative, current control tied to this hazard's evidence.

    Scope is deliberately bounded. Clauses about an unrelated object, promised
    controls, explicit denials and contradictory accounts do not establish a
    maintained control. Cases outside these language patterns need review; no
    absence of a negation is treated as proof that an arbitrary control works.
    """
    clauses = []
    begin = 0
    for boundary in CLAUSE_BOUNDARY.finditer(incident):
        clauses.append((begin, boundary.start(), incident[begin:boundary.start()]))
        begin = boundary.end()
    clauses.append((begin, len(incident), incident[begin:]))
    matches = _control_matches(incident, rule)
    if not matches:
        return ControlAssertion("unknown", ())
    if CONTINUING_HAZARD[rule].search(incident):
        return ControlAssertion("unknown", ("A continuing hazard contradicts a maintained control.",))
    states = []
    evidence = []
    for match in matches:
        start, stop, clause = next(
            c for c in clauses if c[0] <= match.start() < c[1]
        )
        prefix = incident[start:match.start()]
        suffix = incident[match.end():stop]
        # The words *inside* a control phrase can be affirmative: e.g. 'no
        # visible damage' and 'without releasing'. Only its surrounding scope
        # is checked for denial.
        assertion_prefix = re.sub(r"\bnot\s+only\b", "", prefix)
        if DENIAL.search(assertion_prefix) or POST_DENIAL.search(suffix):
            state = "denied"
        elif (NONASSERTED.search(prefix) or POST_NONASSERTED.search(suffix)
              or LATE_CONTROL.search(clause) or OTHER_OBJECT.search(clause)
              or INCOMPLETE_CONTROL.search(clause)
              or (rule == "walking_surface_housekeeping"
                  and RENEWED_WALKWAY_SPILL.search(incident[match.end():]))
              or _explicit_other_subject(prefix, suffix, match.group(), rule)):
            state = "unknown"
        else:
            scoped = _scope_matches(clause, rule)
            # A continuation ('and dried', ', locked out') can inherit the
            # subject of its sentence. A separate sentence cannot silently
            # reuse a different event's scope.
            sentence_start = max(incident.rfind(p, 0, match.start()) for p in '.;!?') + 1
            sentence = incident[sentence_start:stop]
            continuation = bool(re.match(r"\s*(?:was|were|is|are|it|they|and|,)\b", clause))
            if not scoped and continuation:
                scoped = _scope_matches(sentence, rule)
            # These very narrow phrases identify their object intrinsically.
            if rule == "fall_protection" and match.group().startswith(("fall-arrest", "travel-restraint", "guardrail")):
                scoped = True
            state = "affirmed" if scoped else "unknown"
        states.append(state)
        evidence.append(f"{state}: {match.group()}")
    if "denied" in states:
        return ControlAssertion("denied", tuple(evidence))
    if "unknown" in states:
        return ControlAssertion("unknown", tuple(evidence))
    return ControlAssertion("affirmed", tuple(evidence))


def evaluate_event(row: Dict[str, Any]) -> PolicyDecision:
    """Return a conservative policy verdict for one operational event."""

    text = _combined_text(row)
    incident = _normalized_text(row.get("Reported_Incident", ""))
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
    surface_control = _control_matches(incident, "walking_surface_housekeeping")
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
        controlled = _control_matches(incident, "electrical_work_practices")
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
        controlled = _control_matches(incident, "slings")
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
        controlled = _control_matches(incident, "materials_handling_aisles")
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
        controlled = not unsafe and _control_matches(incident, "dockboards")
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
        controlled = _control_matches(incident, "fall_protection")
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

    # A recognized keyword is only a proposal for a control. Validate its
    # assertion before allowing it to clear an otherwise hazardous event.
    for index, candidate in enumerate(candidates):
        if candidate.status == "CLEAR" and candidate.rule in CONTROL_TERMS:
            assertion = _control_assertion(incident, candidate.rule)
            if assertion.state != "affirmed":
                candidates[index] = PolicyDecision(
                    "REVIEW", None,
                    "A relevant control is denied, not yet established, or not tied to this event; review the observation.",
                    candidate.rule, list(assertion.evidence),
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
