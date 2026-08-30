import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from hybrid_policy import evaluate_event


class HybridPolicyTests(unittest.TestCase):
    def decide(
        self, incident, equipment="Forklift", location="Operations Yard", shift=8
    ):
        return evaluate_event(
            {
                "Equipment_Type": equipment,
                "Location": location,
                "Operator_Shift_Hours": shift,
                "Reported_Incident": incident,
            }
        )

    def test_freight_train_employee_uses_us_code(self):
        decision = self.decide(
            "Freight train employee remained in covered switching service",
            equipment="Yard Switch Engine",
            location="Classification Yard",
            shift=12.4,
        )
        self.assertEqual((decision.status, decision.citation), ("VIOLATION", "21103"))

    def test_passenger_train_employee_uses_passenger_rule(self):
        decision = self.decide(
            "Commuter train employee remained in covered service",
            equipment="Commuter Passenger Train",
            location="Passenger Terminal Yard",
            shift=12.4,
        )
        self.assertEqual((decision.status, decision.citation), ("VIOLATION", "228.405"))

    def test_ordinary_forklift_shift_is_not_forced_into_rail_law(self):
        decision = self.decide("No incident reported", shift=13.0)
        self.assertEqual(decision.status, "REVIEW")
        self.assertIsNone(decision.citation)

    def test_contained_spill_is_clear(self):
        decision = self.decide(
            "A leak was contained in a drip pan before reaching the walkway",
            location="Marked Pedestrian Walkway",
        )
        self.assertEqual((decision.status, decision.citation), ("CLEAR", None))

    def test_damaged_sling_in_service_is_violation(self):
        decision = self.decide(
            "A wire-rope sling with broken outer wires remained in an active lift",
            equipment="Gantry Crane",
            location="Lift Pad",
        )
        self.assertEqual(
            (decision.status, decision.citation), ("VIOLATION", "1910.184")
        )

    def test_secured_dockboard_is_clear(self):
        decision = self.decide(
            "The rated dockboard was anchored against movement before crossing",
            location="Railcar Loading Dock",
        )
        self.assertEqual((decision.status, decision.citation), ("CLEAR", None))

    def test_unprotected_platform_edge_is_violation(self):
        decision = self.decide(
            "Work proceeded beside a ten-foot drop with no guardrail or arrest system",
            location="Elevated Service Platform",
        )
        self.assertEqual((decision.status, decision.citation), ("VIOLATION", "1910.28"))


if __name__ == "__main__":
    unittest.main()
