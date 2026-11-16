"""Give an initial assessment and useful evidence while a decision is unresolved.

These helpers add search hypotheses and practical follow-up to the interactive
workflow. They do not alter observations, assign labels, or decide legal scope.
The historical retrieval benchmark continues to use its original planner.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from autonomous_investigator import Observation, QueryPlanner


PROFILE = "interactive_investigation_support_v1"
_STOP_WORDS = frozenset("a an and are as at be been by can could for from has have in is it its of on or that the their then there these this to was were what when which who with would".split())
_EQUIPMENT = r"\b(?:charger|equipment|machine|motor|pump|cabinet|container|load|object|box|boxes)\b"
_DAMAGE = r"\b(?:broken|damage[ds]?|frayed|defective|worn)\b"
_LIQUID = r"\b(?:spills?|leaks?|leaking|oil|wet)\b"


def _text(observation: Observation | str) -> str:
    return observation.evidence_text() if isinstance(observation, Observation) else str(observation)


def _charger(text: str) -> bool:
    # Explicitly saying the object is not a charger is not a charging observation.
    without_denials = re.sub(r"\b(?:no|not\s+(?:an?|the)|without\s+(?:an?|the))\s+(?:battery\s+)?chargers?\b", "", text, flags=re.I)
    return bool(re.search(r"\bchargers?\b", without_denials, re.I))


def _raised(text: str) -> bool:
    # A floor-level object below a platform is not an object on that platform.
    text = re.sub(r"\b(?:on|at)\s+(?:the\s+)?(?:floor|ground)(?:\s+level)?\s+(?:below|beneath|under|beside|near)\s+(?:(?:an?|the|raised|elevated)\s+)*(?:platform|shelf|shelving|rack)\b", "floor level", text, flags=re.I)
    text = re.sub(r"\b(?:not\s+(?:on\s+(?:(?:an?|the)\s+)?)?|no\s+)(?:(?:raised|elevated)\s+)?(?:platform|shelf|rack)\b|\bnot\s+(?:raised|elevated|overhead)\b", "", text, flags=re.I)
    return _affirmed(text, r"\b(?:elevated|raised|platform|shelf|shelving|rack|overhead)\b|\b(?:feet|foot|ft)\s+(?:up|above|from\s+the\s+ground)\b")


def _movement(text: str) -> bool:
    text = re.sub(r"\b(?:cannot|can't|can\s+not)\s+(?:(?:be\s+)?moved\s+easily|slide|tip|wobble)\b|\bnot\s+(?:unsecured|unstable|loose|sliding|tipping|wobbling)\b", "", text, flags=re.I)
    # Missing bolts alone do not establish movement or defeat another restraint.
    return _affirmed(text, r"\b(?:unsecured|unstable|loose|sliding|tipping|wobbling)\b|\bnot\s+(?:secured|fastened|anchored)\b|\b(?:can\s+be\s+)?moved\s+easily\b|\bcan\s+(?:slide|tip|wobble)\b")


def _affirmed(text: str, pattern: str) -> bool:
    """Recognize stated conditions without turning denials into observations.

    This bounded language check only chooses follow-up wording. It does not
    decide whether a hazard is present or a control is adequate.
    """
    for clause in re.split(r"[.!?;]|\b(?:but|however|yet|except)\b", text, flags=re.I):
        for match in re.finditer(pattern, clause, re.I):
            before, after = clause[:match.start()], clause[match.end():]
            denied = re.search(r"\b(?:no|without|not|never)\s+(?:(?:any|visible|reported|signs?|evidence|was|were|is|are|has|have|had|been|of|longer|oil|hydraulic|fluid|coolant|liquid|chemical|water|spill|leak|damage|broken|wires|sling|or|and)\s+){0,9}$", before, re.I)
            free_of = re.search(r"\b(?:free|clear)\s+of\s+(?:(?:any|oil|liquid|damage|broken)\s+){0,3}$", before, re.I)
            no_evidence = re.search(r"\b(?:no|without)\s+(?:reported\s+|visible\s+)?(?:evidence|signs?|indication|confirmation)\s+(?:of|that)\b[^,]*$", before, re.I)
            uncertain = re.search(r"\b(?:if|whether|unknown|uncertain|suspected|possible)\b[^,]*$", before, re.I)
            absent = re.match(r"(?:\s+[a-z-]+){0,3}\s+(?:not|never)\s+(?:present|found|seen|observed|detected|confirmed)\b", after, re.I)
            if not (denied or free_of or no_evidence or uncertain or absent):
                return True
    return False


def _focused_hypotheses(text: str) -> list[str]:
    queries = []
    if _charger(text):
        queries.append("charger electrical equipment mounting fastening firmly secured surface installation")
        queries.append("charger charging apparatus installation protection from damage charging area")
    equipment = bool(re.search(_EQUIPMENT, text, re.I))
    if _raised(text) and equipment and (_movement(text) or _charger(text)):
        queries.append("elevated equipment stability falling objects protection exposure below")
    return queries


def focused_queries(observation: Observation | str) -> list[str]:
    """Return separate search hypotheses, never observation facts or verdicts."""
    return QueryPlanner._deduplicate(_focused_hypotheses(_text(observation)))


class FocusedQueryPlanner(QueryPlanner):
    """Keep the event query and add a few separate, inspectable search hypotheses."""

    def __init__(self, max_queries: int = 8):
        if isinstance(max_queries, bool) or not isinstance(max_queries, int) or not 1 <= max_queries <= 8:
            raise ValueError("The interactive planner accepts one to eight queries")
        super().__init__(max_queries=max_queries)

    def plan(self, observation: Observation) -> list[str]:
        original = super().plan(observation)
        hypotheses = focused_queries(observation)
        # Reserve the first query for the original observation. Search vocabulary
        # is never written back into the evidence used by the policy or reasoner.
        return self._deduplicate([*original[:1], *hypotheses, *original[1:]])[: self.max_queries]


def _query_weights(query: str) -> dict[str, float]:
    weights = {word: 1.0 for word in re.findall(r"[a-z]+", query.casefold()) if len(word) > 2 and word not in _STOP_WORDS}
    if _charger(query):
        weights.update({word: 3.0 for word in ("charger", "chargers", "charging", "electric", "electrical", "mounting", "mounted", "secured", "fastening")})
    if _raised(query) and _movement(query):
        weights.update({word: 2.0 for word in ("falling", "objects", "stable", "stability", "secured")})
    return weights


def relevant_excerpt(text: str, query: str, max_chars: int = 650) -> dict[str, Any]:
    """Select a verbatim window; offsets are Python character indices in ``text``.

    The score only ranks excerpt relevance. It is not confidence, legal
    applicability, or evidence that a violation occurred.
    """
    if not isinstance(text, str) or not isinstance(query, str):
        raise TypeError("Excerpt text and query must be strings")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 40 <= max_chars <= 2000:
        raise ValueError("Excerpt length must be between 40 and 2,000 characters")
    weights = _query_weights(query)
    words = list(re.finditer(r"[a-z]+", text, flags=re.I))
    hits = [(match.start(), match.end(), weights[match.group().casefold()]) for match in words if match.group().casefold() in weights]
    starts = {0}
    for start, _, _ in hits:
        proposed = max(0, start - 80)
        # Start after the previous sentence when it is close enough to retain
        # the match. Leave paragraph labels and wording intact.
        boundary = max(text.rfind(". ", max(0, proposed - 120), start), text.rfind("\n", max(0, proposed - 120), start))
        if boundary >= 0:
            proposed = boundary + (2 if text[boundary:boundary + 2] == ". " else 1)
        starts.add(proposed)
    best = (0.0, 0, min(len(text), max_chars))
    for start in sorted(starts):
        end = min(len(text), start + max_chars)
        counts: dict[str, int] = {}
        for match in words:
            word = match.group().casefold()
            if start <= match.start() and match.end() <= end and word in weights:
                counts[word] = counts.get(word, 0) + 1
        score = sum(weights[word] * min(count, 2) for word, count in counts.items())
        if score > best[0]:
            best = (score, start, end)
    score, start, end = best
    return {"text": text[start:end], "start": start, "end": end, "score": round(score, 4)}


def _candidate_section(candidate: Any) -> str:
    if isinstance(candidate, Mapping):
        return str(candidate.get("section") or "")
    return str(getattr(candidate, "section", ""))


def _long_shift(observation: Observation | str, text: str) -> bool:
    if isinstance(observation, Observation):
        hours = observation.measurements.get("operator_shift_hours", observation.measurements.get("shift_hours"))
        try:
            return 12 <= float(hours) < float("inf")
        except (TypeError, ValueError):
            pass
    return bool(re.search(r"\b(?:1[2-9]|2[0-4])\s*[- ]?\s*(?:hour|hr)s?\s+shift\b|\bshift\s+(?:of\s+)?(?:1[2-9]|2[0-4])\s*(?:hours?|hrs?)\b", text, re.I))


def build_investigation_support(
    observation: Observation | str,
    status: str,
    candidates: Sequence[Any] = (),
    decision_source: str = "",
) -> dict[str, Any]:
    """Offer a useful first assessment, followed by at most two important gaps.

    Precautions are conditional incident-response guidance. They do not turn
    a search match into a legal verdict or approve an equipment modification.
    """
    text = _text(observation)
    lower = text.casefold()
    status = str(status).upper()
    concerns: list[str] = []
    precautions: list[str] = []
    questions: list[str] = []
    notes: list[str] = []
    sections = {_candidate_section(candidate) for candidate in candidates}
    bolts_mentioned = bool(re.search(r"\bbolts?|bolted|bolting\b", lower))
    bolts_missing = _affirmed(text, r"\b(?:not\s+bolted|unbolted)\b|\b(?:no|missing|absent)\s+(?:mounting\s+)?bolts?\b|\bbolts?\s+(?:(?:is|are|was|were)\s+)?(?:missing|absent)\b")

    if status == "NO_ACTION":
        assessment = "No investigation trigger fired for this observation. That is a routing result, not a finding that the area is safe."
        precautions = ["Continue the usual checks and report a new condition or change."]
    elif status == "CLEAR":
        assessment = "The scoped check supports the reported control. Keep that control in place and check again if the conditions change."
        precautions = ["Confirm that the reported control still covers the activity; this result does not clear unrelated hazards."]
    elif _charger(text) and _raised(text) and _movement(text):
        assessment = "An elevated charger that can move deserves prompt attention: it could slide or fall from its support. The first check is stability and the area below, even while the exact compliance finding remains open."
        concerns = ["Potential falling equipment if the charger can slide or tip.", "The installation may need secure mounting or a restraint appropriate to its design."]
        if bolts_mentioned:
            concerns[-1] += " Bolts are not the only possible method."
        precautions = ["If it is unstable, keep people out of the potential fall zone and notify the responsible supervisor or safety person.", "Have an authorized person assess how to secure or relocate it; do not try to move heavy or energized equipment without the right training and procedure."]
        questions = ["Can people pass or work where the charger could fall?", "Is any restraint or other secure mounting keeping it from sliding or tipping?"]
        notes = ["Equipment height alone does not establish a worker fall-protection issue; employee exposure and the installation still matter."]
    elif _raised(text) and _movement(text) and re.search(_EQUIPMENT, lower):
        assessment = "The reported movement of elevated equipment or material warrants a stability check. A falling object could affect anyone below, but the observation does not by itself establish their exposure."
        concerns = ["Possible sliding, tipping or falling equipment or material."]
        precautions = ["If instability is present, keep people clear of the possible fall zone and report it to the responsible person.", "Use the site's authorized handling procedure for securing or relocating the item."]
        questions = ["Is anyone exposed below or beside the item?", "What is keeping it from sliding or tipping?"]
    elif _charger(text):
        assessment = "Check whether the charger is stable, protected from damage, and being used as intended."
        if bolts_missing:
            assessment += " A missing bolt alone does not settle whether the installation is unsafe."
        concerns = ["Possible movement, cable damage or unsuitable placement if the installation is not stable."]
        precautions = ["Report visible instability or damage and have the responsible person check it before continued use."]
        questions = ["Is the charger moving, damaged, or exposed to being struck?", "Is its current placement and restraint consistent with its installation instructions?"]
    elif re.search(r"\b(?:energized|electrical|electric|conductor|cabinet|voltage)\b", lower):
        assessment = "The reported electrical condition needs a focused exposure check. Unconfirmed isolation should not be treated as proof that equipment is de-energized."
        concerns = ["Potential contact with hazardous electrical energy if the equipment is energized and accessible."]
        precautions = ["Keep people from approaching exposed parts and have an authorized qualified person verify the energy state and required controls."]
        questions = ["Can anyone contact or approach the potentially energized parts?", "Has an authorized person verified the isolation or safe working arrangement?"]
    elif re.search(r"\b(?:sling|lifting|lift)\b", lower) and re.search(_DAMAGE, lower):
        if _affirmed(text, _DAMAGE):
            assessment = "Reported damage to lifting equipment warrants attention before further use. The immediate concern is loss of the load or equipment failure."
            concerns = ["Possible failure of damaged lifting equipment."]
            precautions = ["Keep people clear of any suspended load and use the site's safe-stop and inspection procedure.", "Have the responsible qualified person assess the reported damage before further use."]
            questions = ["Is the item with reported damage still carrying a load or available for use?"]
        else:
            assessment = "The description does not establish damage to the lifting equipment. Check its condition and suitability for the lift through the normal inspection process."
            precautions = ["Maintain the normal lifting and inspection controls; reassess if a defect is found or conditions change."]
    elif re.search(_LIQUID, lower):
        if _affirmed(text, _LIQUID):
            assessment = "Check where the liquid is reaching and whether the reported containment is working. A spill in a travel path can create a slip exposure; a contained leak is a different condition."
            concerns = ["Possible slip or contact exposure if liquid reaches an occupied area."]
            precautions = ["If liquid reaches the walking route, keep people clear while the responsible person arranges suitable containment and cleanup."]
            questions = ["Has the liquid reached a walking route or other occupied area?", "Is containment holding, or is the leak continuing beyond it?"]
        else:
            assessment = "The description does not establish liquid on the walking route. Continue the usual surface checks and reassess if a leak or spill appears."
            precautions = ["Keep normal housekeeping checks in place; this does not clear unrelated hazards in the area."]
    elif _long_shift(observation, text):
        assessment = "A long operator shift merits a fatigue and task-fitness check. Shift length alone does not establish a railroad hours-of-service violation; the person's actual duties matter."
        concerns = ["Possible fatigue affecting attention or coordination, if those effects are present."]
        precautions = ["If fatigue is affecting safe work, report it and arrange relief or a pause through the site's procedure."]
        questions = ["What duties is the operator performing, and is fatigue affecting the work?"]
    elif re.search(r"\b(?:machine|machinery|rotating|guarding|unguarded|pinch|nip)\b", lower):
        assessment = "Check the reported machine condition against the task and the protection around moving parts. This workflow has not established whether the machine is adequately guarded."
        concerns = ["Possible contact or entanglement with moving parts if access is not controlled."]
        precautions = ["Keep people clear of accessible moving parts and report missing or damaged protection.", "Use the site's authorized isolation and inspection procedure; do not reach into or adjust moving equipment."]
        questions = ["Can anyone reach the moving parts during the reported task?", "What guarding or isolation is in place for that task?"]
        if "1910.212" not in sections:
            notes = ["General machine-guarding authority was not retrieved. This limited evidence set cannot settle the guarding question."]
    else:
        assessment = "Start with the reported condition and who could be affected. The current check has not resolved it, so pass that observation and the available evidence to the responsible person for a focused review."
        concerns = ["The applicable requirement or a fact needed for the decision remains unresolved."]
        precautions = ["If an immediate hazard is visible, keep people clear and use the site's reporting or safe-stop procedure.", "Record the condition and any control already in place without recreating the exposure."]
        questions = ["What is happening now, and who or what could be exposed?", "What control is already in place, and is it working?"]

    if _charger(text) and status not in {"CLEAR", "NO_ACTION"}:
        if "1910.303" in sections:
            notes.append("The retrieved electrical-installation evidence is relevant to mounting and use; check its requirements against the actual installation.")
            if bolts_mentioned:
                notes.append("The mounting text does not establish that every charger must be bolted down.")
        if "1910.178" in sections:
            notes.append("The retrieved truck-charging provisions are conditional on the equipment and activity; charging-only exceptions need to be checked before applying the full paragraph.")
    if status in {"ERROR", "UNKNOWN", "FAILED"} or "error" in decision_source.casefold() or "failure" in decision_source.casefold():
        assessment = "The automated check did not complete. " + assessment
        notes.append("A failed check provides no compliance conclusion; keep its result out of trusted labels and learning memory.")
    return {"profile": PROFILE, "first_assessment": assessment, "possible_concerns": concerns,
            "immediate_precautions": precautions, "priority_clarifications": questions[:2],
            "evidence_notes": notes, "final_legal_verdict": False, "automatic_label": False}
