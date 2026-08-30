import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from learning_policy import assess_for_learning, parse_binary_label


class LearningPolicyTests(unittest.TestCase):
    def test_unlabeled_model_output_is_quarantined(self):
        decision = assess_for_learning({}, "CLEAR", None, True)
        self.assertFalse(decision.trusted)
        self.assertEqual(decision.feedback_type, "quarantine")

    def test_grounded_matching_label_is_trusted(self):
        row = {"Is_Violation": "1", "Violation_Type": "spill"}
        decision = assess_for_learning(row, "VIOLATION", "1910.22", True)
        self.assertTrue(decision.trusted)
        self.assertEqual(decision.verification_source, "ground_truth")

    def test_real_but_wrong_citation_is_rejected(self):
        row = {"Is_Violation": "1", "Violation_Type": "spill"}
        decision = assess_for_learning(row, "VIOLATION", "1910.178", True)
        self.assertFalse(decision.trusted)
        self.assertEqual(decision.feedback_type, "pitfall")
        self.assertEqual(decision.correct_verdict, "VIOLATION")
        self.assertEqual(decision.correct_citation, "1910.22")

    def test_approved_human_review_can_promote_an_episode(self):
        row = {
            "Review_Status": "approved",
            "Reviewed_Verdict": "CLEAR",
            "Reviewed_Citation": "NONE",
        }
        decision = assess_for_learning(row, "CLEAR", None, True)
        self.assertTrue(decision.trusted)
        self.assertEqual(decision.verification_source, "human_review")

    def test_unknown_binary_label_remains_unlabeled(self):
        self.assertIsNone(parse_binary_label("maybe"))

    def test_explicit_expected_citation_supports_new_hazard_types(self):
        row = {
            "Is_Violation": "1",
            "Violation_Type": "defective_sling",
            "Expected_Citation": "1910.184",
        }
        decision = assess_for_learning(row, "VIOLATION", "1910.184", True)
        self.assertTrue(decision.trusted)
        self.assertEqual(decision.correct_citation, "1910.184")

    def test_fatigue_ground_truth_uses_train_employee_statute(self):
        row = {"Is_Violation": "1", "Violation_Type": "fatigue"}
        decision = assess_for_learning(row, "VIOLATION", "21103", True)
        self.assertTrue(decision.trusted)
        self.assertEqual(decision.correct_citation, "21103")


if __name__ == "__main__":
    unittest.main()
