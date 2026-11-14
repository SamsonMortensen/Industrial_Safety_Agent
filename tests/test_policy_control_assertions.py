"""Control words cannot establish a control that the observation denies."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import apply_policy_controls
from hybrid_policy import evaluate_event


def decide(incident, location="Operations Yard", equipment="Forklift"):
    return evaluate_event({"Equipment_Type": equipment, "Location": location,
                           "Operator_Shift_Hours": 8, "Reported_Incident": incident})


class ControlAssertionTests(unittest.TestCase):
    def test_audit_paired_controls(self):
        pairs = [
            ("Oil spill on the employee walkway was cleaned.",
             "Oil spill on the employee walkway was not cleaned."),
            ("The circuit was de-energized and grounded before work.",
             "The circuit was not de-energized and not grounded before work."),
            ("Broken sling was removed from service before the lift.",
             "Broken sling was not removed from service and remained in the active lift."),
            ("Work at the open edge proceeded with fall-arrest.",
             "Work at the open edge proceeded without fall-arrest."),
        ]
        for affirmative, negative in pairs:
            with self.subTest(affirmative=affirmative):
                self.assertEqual(decide(affirmative).status, "CLEAR")
                self.assertEqual(decide(negative).status, "REVIEW")
                self.assertIsNone(decide(negative).citation)

    def test_control_denials_include_scope_and_unicode(self):
        cases = [
            "Oil on the walkway has not yet been cleaned.",
            "Oil on the walkway wasn't cleaned.",
            "Oil on the walkway wasn’t cleaned.",
            "Oil on the walkway won't be cleaned.",
            "Oil on the walkway couldn’t be cleaned.",
            "Oil on the walkway shouldn't be described as cleaned.",
            "The oil spills on the walkway haven't been cleaned.",
            "Oil on the walkway was NEVER cleaned.",
            "Oil on the walkway was not CLEANED.\n",
            "No fall-arrest was used at the open edge.",
            "Fall-arrest was missing at the open edge.",
            "Fall-arrest was not connected at the open edge.",
            "The circuit had no lockout before work.",
            "The sling was never removed from service.",
            "The dockboard was not anchored before crossing.",
            "The aisle did not stay clear; cargo was not kept inside the storage boundary.",
        ]
        for incident in cases:
            with self.subTest(incident=incident):
                self.assertNotEqual(decide(incident).status, "CLEAR")

    def test_promised_conditional_or_late_controls_do_not_clear(self):
        cases = [
            "The oil spill on the walkway will be cleaned.",
            "The oil spill on the walkway should be cleaned.",
            "The crew planned to have the oil spill on the walkway cleaned.",
            "If the oil spill on the walkway is cleaned, work can resume.",
            "The oil spill on the walkway was cleaned after employees slipped.",
            "The broken sling was removed from service after the lift.",
            "Fall-arrest will be provided for work at the open edge.",
            "Fall-arrest was available near the open edge.",
            "The oil spill on the walkway is cleaned tomorrow.",
        ]
        for incident in cases:
            with self.subTest(incident=incident):
                self.assertEqual(decide(incident).status, "REVIEW")

    def test_unrelated_object_does_not_supply_control(self):
        cases = [
            "Oil was on the employee walkway. A mechanic cleaned the truck.",
            "An energized conductor was within four feet of the crane. The spare crane was grounded.",
            "Broken sling remained in an active lift. Another sling was removed from service.",
            "Workers were at an open edge. A complete guardrail was stored nearby.",
            "Cargo blocked the aisle. A different passage stayed clear.",
            "The truck was cleaned beside an oil spill on the walkway.",
            "The worker cleaned the truck beside an oil spill on the walkway.",
            "A broken sling was used while the crew replaced the gloves.",
        ]
        for incident in cases:
            with self.subTest(incident=incident):
                self.assertEqual(decide(incident).status, "REVIEW")

    def test_conflicting_and_continuing_exposure_requires_review(self):
        cases = [
            "Oil spill on the walkway was cleaned, but the spill was not cleaned.",
            "The oil spill was cleaned, but the walkway remained slippery.",
            "The sling was removed from service, but the broken sling remained in the active lift.",
            "The dockboard was anchored, but the plate continued moving during crossing.",
            "The platform had complete guardrails, but the edge remained unprotected.",
            "Cargo stayed inside storage; the aisle remained blocked.",
        ]
        for incident in cases:
            with self.subTest(incident=incident):
                self.assertEqual(decide(incident).status, "REVIEW")

    def test_positive_controls_and_unrelated_denial_remain_clear(self):
        cases = [
            "No equipment damage occurred; the oil spill on the walkway was cleaned.",
            "The oil spill on the walkway was cleaned; no equipment damage occurred.",
            "The oil spill on the walkway was cleaned and the truck was not damaged.",
            "The inspected sling had a legible rating tag and no visible damage.",
            "A closed hose passed above the walkway without releasing material.",
            "The circuit was de-energized, locked out, tested, and grounded before work.",
            "The spill area was barricaded and an alternate dry walkway was in service.",
            "A rated dockboard was anchored before crossing.",
            "The  OIL spill on the employee WALKWAY\n was CLEANED.",
            "The oil spill on the walkway was not only cleaned but the walkway was dried.",
        ]
        for incident in cases:
            with self.subTest(incident=incident):
                self.assertEqual(decide(incident).status, "CLEAR")

    def test_removed_or_incomplete_controls_require_review(self):
        cases = [
            "Fall-arrest was removed before work at the open edge.",
            "The complete guardrail was dismantled before work at the open edge.",
            "The complete guardrail had been temporarily removed before work at the open edge.",
            "The connected restraint was detached before work at the open edge.",
            "The oil spill on the employee walkway was only partly cleaned.",
            "The oil spill on the employee walkway was partially cleaned.",
            "The oil spill on the employee walkway was cleaned only in part.",
        ]
        for incident in cases:
            with self.subTest(incident=incident):
                self.assertEqual(decide(incident).status, "REVIEW")

    def test_new_spill_after_cleanup_is_not_cleared_by_the_old_cleanup(self):
        cases = [
            "The oil spill on the employee walkway was cleaned yesterday but fresh oil spilled across the walkway today.",
            "The oil spill on the employee walkway was cleaned. New fluid leaked onto the walkway.",
        ]
        for incident in cases:
            with self.subTest(incident=incident):
                self.assertEqual(decide(incident).status, "REVIEW")

    def test_current_controls_still_support_clearance(self):
        cases = [
            "The oil spill on the employee walkway was fully cleaned before use.",
            "Fresh oil spilled across the walkway. The oil spill on the walkway was cleaned before use.",
            "A damaged sling was removed from service before the lift began.",
            "A complete guardrail system protected every open side of the platform.",
        ]
        for incident in cases:
            with self.subTest(incident=incident):
                self.assertEqual(decide(incident).status, "CLEAR")

    def test_postprocessor_does_not_erase_warning_with_denied_control(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw.json"
            dataset = root / "events.csv"
            raw.write_text(json.dumps({"results": [{"log_id": "fixture", "actual": 1,
                "predicted": 1, "status": "VIOLATION", "citation": "1910.22",
                "reason": "Oil remains on walkway", "expected_section": "1910.22"}]}))
            row = {"Log_ID": "fixture", "Equipment_Type": "Forklift", "Location": "Walkway",
                   "Operator_Shift_Hours": "8", "Reported_Incident": "Oil spill on the employee walkway was not cleaned."}
            with dataset.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
            with patch.object(apply_policy_controls, "ROOT", root), patch.object(apply_policy_controls, "sha256", return_value="fixture_hash"):
                report = apply_policy_controls.apply_controls(raw, dataset, root / "out.json")
            result = report["results"][0]
            self.assertNotEqual(result["status"], "CLEAR")
            self.assertEqual(result["model_status"], "VIOLATION")


if __name__ == "__main__":
    unittest.main()
