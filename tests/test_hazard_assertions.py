"""Hazard words must describe the condition and exposure being decided."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from hybrid_policy import evaluate_event


def decide(text):
    return evaluate_event({"Equipment_Type": "Forklift", "Location": "Operations Yard",
                           "Operator_Shift_Hours": 8, "Reported_Incident": text})


# These examples are authored regression cases, not independently reviewed
# field labels. REVIEW withholds the unsupported automatic citation.
DENIED_HAZARDS = (
    "The sling has no broken wires and remains in service.",
    "The sling has no visible broken outer wires and remains in service.",
    "Broken wires were not found on the sling.",
    "The sling is not cracked and remains in use.",
    "No oil spill is present on the walkway.",
    "No hydraulic fluid spill is present on the pedestrian walkway.",
    "The oil spill was not observed on the walkway.",
    "No leak is present on the walkway.",
    "Pallets did not obstruct the aisle.",
    "Freight never blocked the aisle.",
    "The dockboard never shifted under the forklift.",
    "The dockboard is not loose during crossing.",
    "The conductor was not energized while the crane worked within four feet.",
    "The crane was not within the approach boundary of the energized conductor.",
    "No wires were broken on the sling, which remained in service.",
    "No sling wires were broken during the lift.",
    "No hydraulic oil was spilled on the walkway.",
    "The walkway was free of oil.",
    "An oil-free walkway remained open.",
)
UNSCOPED_ELECTRICAL_CONDITIONS = (
    "Panel left open with energised parts exposed to the walkway.",
    "Panel left open with energized parts exposed to the walkway.",
    "Electrical equipment has exposed energized parts beside the walkway.",
    "A worker reported an open panel with exposed energized parts.",
    "The crane is parked inside a maintenance building. An energized conductor is nearby.",
    "A worker is inside the office. Electrical equipment beside the walkway remains energized.",
    "The crane moved through the yard. An energized overhead conductor is beside the walkway.",
    "The crane is parked inside a maintenance building beside an energized conductor.",
)
EQUIPMENT_WITHOUT_PERSON_FALL_EXPOSURE = (
    "A turret machine charger is on a platform six feet above the ground without mounting bolts.",
    "A charger is on an elevated platform with mounting bolts missing.",
    "A charger is on a platform six feet above the aisle and can be moved easily.",
    "A fixed ladder to the ramp platform is in use with two rungs missing.",
    "No worker is at the open edge of the platform with missing guardrails.",
    "A worker inspected the charger on the platform without mounting bolts.",
    "The elevated catwalk had a missing rail and no personal fall protection.",
    "The access platform lacked both guardrails and an approved fall-restraint connection.",
    "A charger sits on an elevated work platform without guardrails.",
    "A work light is on a platform with missing rails.",
    "A worker in the office reported a platform with missing guardrails.",
    "A worker remains in the office. A remote platform has missing guardrails.",
    "A worker stood on the enclosed platform. A remote platform has missing guardrails.",
    "The worker did not stand at the open edge of the platform without guardrails.",
)
POSITIVE_HAZARDS = (
    ("The sling has broken wires and remains in active lifting service.", "1910.184"),
    ("No injuries occurred and broken sling wires remained in an active lift.", "1910.184"),
    ("No oil spill was present, but the sling had broken wires in an active lift.", "1910.184"),
    ("No wires were broken on the spare sling; the lifting sling had cracked fittings.", "1910.184"),
    ("Oil spilled across the active employee walkway.", "1910.22"),
    ("No injuries occurred and oil spilled across the walkway.", "1910.22"),
    ("No oil spill was present; coolant leaked onto the walkway.", "1910.22"),
    ("Standing oily runoff spread over the pedestrian route with no containment.", "1910.22"),
    ("Palletised freight left obstructing the designated traffic aisle.", "1910.176"),
    ("No injuries occurred while pallets obstructed the aisle.", "1910.176"),
    ("A dockboard shifted under the forklift with no securing device fitted.", "1910.26"),
    ("No injuries occurred while the dockboard shifted under the forklift.", "1910.26"),
    ("A crane boom worked within four feet of an energized overhead conductor.", "1910.333"),
    ("Spreader boom worked within four feet of the energised catenary.", "1910.333"),
    ("No injuries occurred while the crane worked within four feet of an energized overhead conductor.", "1910.333"),
    ("Equipment crossed the minimum approach boundary beside the live busbar.", "1910.333"),
    ("The boom tip passed inside the posted live-line clearance boundary.", "1910.333"),
    ("A worker stood at an open edge without fall protection.", "1910.28"),
    ("Work proceeded beside a ten-foot drop with no guardrail or arrest system.", "1910.28"),
    ("The charger had no mounting bolts. A worker stood at the open platform edge without fall protection.", "1910.28"),
    ("No injuries occurred while the employee worked at an open edge without fall protection.", "1910.28"),
    ("A technician stepped onto an open catwalk without fall protection.", "1910.28"),
    ("The missing midrail left an employee exposed at the elevated platform edge.", "1910.28"),
    ("Maintenance continued beside an open-sided platform without a protective system.", "1910.28"),
    ("A worker stood on an elevated work platform without guardrails.", "1910.28"),
    ("A worker crossed an unprotected edge eight feet above the lower level.", "1910.28"),
    ("A work light stood in the office. A worker stood at the open platform edge without fall protection.", "1910.28"),
    ("No wires were broken on the spare sling. The lifting sling had cracked fittings.", "1910.184"),
    ("No hydraulic oil was spilled on the walkway, but coolant leaked onto it across the walkway.", "1910.22"),
    ("The walkway was free of oil, but coolant spilled across the walkway.", "1910.22"),
)
UNSUPPORTED_OUTSIDE_CONTROLS = (
    "A crane is parked inside a maintenance building. An energized conductor is outside.",
    "A crane stood beside an energized conductor outside the building.",
    "An energized conductor is outside the approach boundary.",
    "A worker described an energized conductor outside the warehouse.",
)
EXPLICIT_OUTSIDE_CONTROLS = (
    "The crane remained outside the established approach boundary of the energized conductor.",
    "The boom route stayed outside the engineering-approved live-line envelope.",
    "The operator kept the crane outside the marked approach boundary of the energized line.",
)


class HazardAssertionTests(unittest.TestCase):
    def test_denied_hazards_do_not_supply_a_violation_or_clearance(self):
        for text in DENIED_HAZARDS:
            with self.subTest(text=text):
                outcome = decide(text)
                self.assertEqual((outcome.status, outcome.citation), ("REVIEW", None))

    def test_open_panel_does_not_establish_work_or_approach_exposure(self):
        for text in UNSCOPED_ELECTRICAL_CONDITIONS:
            with self.subTest(text=text):
                # Background equipment/location labels cannot invent the
                # employee or moving equipment missing from the incident.
                outcome = evaluate_event({"Equipment_Type": "Gantry Crane",
                    "Location": "Energized Catenary Zone", "Reported_Incident": text})
                self.assertEqual((outcome.status, outcome.citation), ("REVIEW", None))

    def test_equipment_height_and_missing_parts_do_not_establish_person_fall_exposure(self):
        for text in EQUIPMENT_WITHOUT_PERSON_FALL_EXPOSURE:
            with self.subTest(text=text):
                outcome = decide(text)
                self.assertEqual((outcome.status, outcome.citation), ("REVIEW", None))

    def test_asserted_hazards_remain_visible_beside_unrelated_negation(self):
        for text, section in POSITIVE_HAZARDS:
            with self.subTest(text=text):
                outcome = decide(text)
                self.assertEqual((outcome.status, outcome.citation), ("VIOLATION", section))

    def test_conditional_hazards_are_not_observed_events(self):
        for text in (
            "If oil spills on the walkway, stop access.",
            "If broken wires are found on the sling, stop the lift.",
            "If the dockboard shifted during crossing, report it.",
        ):
            with self.subTest(text=text):
                self.assertEqual(decide(text).status, "REVIEW")

    def test_substrings_do_not_invent_damage_or_contamination(self):
        for text in ("The unbroken sling remains in service.",
                     "A boiler is beside the employee walkway."):
            with self.subTest(text=text):
                self.assertEqual(decide(text).status, "REVIEW")

    def test_not_only_is_not_a_denial_of_an_asserted_hazard(self):
        outcome = decide("The sling was not only broken but still used in the active lift.")
        self.assertEqual((outcome.status, outcome.citation), ("VIOLATION", "1910.184"))

    def test_no_mounting_bolts_is_not_automatically_a_mounting_violation(self):
        outcome = decide("A turret truck charger is not bolted to its platform but is retained by its factory mounting system.")
        self.assertEqual((outcome.status, outcome.citation), ("REVIEW", None))

    def test_generic_outside_is_not_an_electrical_approach_control(self):
        for text in UNSUPPORTED_OUTSIDE_CONTROLS:
            with self.subTest(text=text):
                outcome = decide(text)
                self.assertEqual((outcome.status, outcome.citation), ("REVIEW", None))

    def test_explicit_actor_outside_approach_constraint_remains_clear(self):
        for text in EXPLICIT_OUTSIDE_CONTROLS:
            with self.subTest(text=text):
                outcome = decide(text)
                self.assertEqual((outcome.status, outcome.citation), ("CLEAR", None))


if __name__ == "__main__":
    unittest.main()
