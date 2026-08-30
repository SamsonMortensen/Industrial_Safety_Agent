import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from hybrid_policy import evaluate_event
from self_reflection import StatutoryGroundedVerifier


class AuditControlRegressionTests(unittest.TestCase):
    def test_captured_runoff_is_not_an_active_walking_surface_hazard(self):
        decision = evaluate_event(
            {
                "Equipment_Type": "Wash Rack",
                "Location": "Marked Pedestrian Walkway",
                "Operator_Shift_Hours": 8,
                "Reported_Incident": (
                    "Washdown runoff captured by the oil-water separator drain"
                ),
            }
        )
        self.assertEqual((decision.status, decision.citation), ("CLEAR", None))

    def test_duty_time_measurement_section_cannot_stand_alone_as_violation(self):
        verifier = StatutoryGroundedVerifier()
        result = verifier.verify(
            {"status": "VIOLATION", "citation": "228.7"},
            {
                "Equipment_Type": "Gantry Crane",
                "Location": "Lift Pad",
                "Operator_Shift_Hours": 8,
                "Reported_Incident": "Damaged sling removed from service",
            },
            [{"section": "228.7"}],
        )
        self.assertFalse(result["is_valid"])
        self.assertTrue(
            any("cannot independently establish" in issue for issue in result["issues"])
        )


if __name__ == "__main__":
    unittest.main()
