"""Check useful first assessments without manufacturing facts or verdicts."""

import copy
from dataclasses import asdict
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from autonomous_investigator import Observation, QueryPlanner
from investigation_support import FocusedQueryPlanner, build_investigation_support, focused_queries, relevant_excerpt

CHARGER = "A turret machine charger is on a platform, six feet from the ground on an end cap aisle. The charger is not bolted down and can be moved easily."


def observation(summary, **values):
    return Observation.from_mapping({"event_id": "synthetic-support-case", "summary": summary, **values})


class FocusedSearchTests(unittest.TestCase):
    def test_charger_search_keeps_original_and_does_not_change_facts(self):
        item = observation(CHARGER, metadata={"Expected_Section": "SENTINEL", "Is_Violation": 1})
        before = copy.deepcopy(asdict(item))
        queries = FocusedQueryPlanner().plan(item)
        self.assertEqual(queries[0], CHARGER)
        self.assertEqual(before, asdict(item))
        self.assertTrue(any("mounting" in query for query in queries[1:]))
        self.assertTrue(any("falling objects" in query for query in queries[1:]))
        self.assertNotIn("SENTINEL", " ".join(queries))
        self.assertFalse(any("1910" in query for query in queries))

    def test_queries_are_bounded_even_with_many_observed_relations(self):
        item = observation(CHARGER, measurements={"measure_" + str(i): i for i in range(20)}, novelty_score=1)
        for limit in (1, 3, 8):
            with self.subTest(limit=limit):
                queries = FocusedQueryPlanner(max_queries=limit).plan(item)
                self.assertLessEqual(len(queries), limit)
                self.assertEqual(queries[0], CHARGER)
                self.assertEqual(len(queries), len(set(queries)))

    def test_invalid_query_budget_is_rejected(self):
        for value in (0, 9, True, 2.5):
            with self.assertRaises(ValueError):
                FocusedQueryPlanner(max_queries=value)

    def test_unrelated_event_keeps_original_planner_queries(self):
        item = observation("A leak was contained in a drip pan before reaching the walkway", equipment=["forklift"])
        self.assertEqual(FocusedQueryPlanner().plan(item), QueryPlanner().plan(item))

    def test_elevated_unsecured_non_charger_gets_stability_search(self):
        queries = FocusedQueryPlanner().plan(observation("An unsecured motor is sliding on a raised shelf"))
        self.assertTrue(any("falling objects" in query for query in queries))
        self.assertFalse(any("charging" in query for query in queries))

    def test_pump_that_can_slide_gets_stability_search(self):
        item = observation("A pump on a raised shelf can slide toward the edge.")
        searches = focused_queries(item)
        self.assertEqual(searches, ["elevated equipment stability falling objects protection exposure below"])
        self.assertIn(searches[0], FocusedQueryPlanner().plan(item))
        self.assertEqual(focused_queries("A pump on a raised shelf cannot slide or tip."), [])

    def test_focused_search_api_does_not_include_original_or_modify_it(self):
        item = observation(CHARGER)
        before = asdict(item)
        searches = focused_queries(item)
        self.assertLessEqual(len(searches), 3)
        self.assertNotIn(CHARGER, searches)
        self.assertIn("mounting", searches[0])
        self.assertEqual(asdict(item), before)

    def test_denied_charger_identity_does_not_add_charging_queries(self):
        for summary in ("This is not a charger; it is a table.", "No charger is present on the platform."):
            with self.subTest(summary=summary):
                queries = FocusedQueryPlanner().plan(observation(summary))
                self.assertFalse(any("charging apparatus" in query for query in queries))


class ExcerptTests(unittest.TestCase):
    def test_later_mounting_passage_is_selected_verbatim_with_offsets(self):
        opening = "Unrelated definitions and introductory material. " * 20
        relevant = "(8) Mounting and cooling of equipment. (i) Electric equipment shall be firmly secured to the surface on which it is mounted."
        text = opening + relevant + " Other provisions follow."
        result = relevant_excerpt(text, CHARGER, max_chars=300)
        self.assertGreater(result["start"], 0)
        self.assertIn("firmly secured", result["text"])
        self.assertEqual(result["text"], text[result["start"]:result["end"]])
        self.assertLessEqual(len(result["text"]), 300)
        self.assertGreater(result["score"], 0)

    def test_unicode_offsets_refer_to_original_text(self):
        text = "Straße — introductory text. " * 25 + "Electric equipment mounting must remain secured."
        result = relevant_excerpt(text, "charger mounting", max_chars=100)
        self.assertEqual(result["text"], text[result["start"]:result["end"]])
        self.assertIn("mounting", result["text"])

    def test_no_match_and_empty_excerpt_remain_honest(self):
        self.assertEqual(relevant_excerpt("", "charger"), {"text": "", "start": 0, "end": 0, "score": 0.0})
        result = relevant_excerpt("Unrelated material.", "charger")
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["text"], "Unrelated material.")

    def test_relevance_score_prefers_mounting_evidence_to_worker_hole_text(self):
        good = relevant_excerpt("Mounting electric equipment: it must be firmly secured to its mounted surface.", CHARGER)
        other = relevant_excerpt("Employees must be protected from falling through a hole four feet above the ground.", CHARGER)
        self.assertGreater(good["score"], other["score"])


class AssessmentTests(unittest.TestCase):
    def test_charger_first_assessment_is_actionable_before_two_questions(self):
        result = build_investigation_support(observation(CHARGER), "REVIEW", [{"section": "1910.303"}, {"section": "1910.178"}])
        self.assertIn("slide or fall", result["first_assessment"])
        self.assertIn("If it is unstable", result["immediate_precautions"][0])
        self.assertTrue(any("authorized" in item for item in result["immediate_precautions"]))
        self.assertEqual(len(result["priority_clarifications"]), 2)
        self.assertFalse(any("model" in question or "manufacturer" in question for question in result["priority_clarifications"]))
        self.assertFalse(result["final_legal_verdict"])
        self.assertFalse(result["automatic_label"])
        self.assertTrue(any("bolts are not" in item.lower() for item in result["possible_concerns"]))
        self.assertTrue(any("charging-only exceptions" in item for item in result["evidence_notes"]))

    def test_equivalent_restraint_does_not_become_movable_charger(self):
        item = observation("The charger is on a platform. It is not bolted down but is secured with a rated restraint and cannot be moved easily.")
        result = build_investigation_support(item, "REVIEW")
        self.assertNotIn("An elevated charger that can move", result["first_assessment"])
        self.assertIn("missing bolt alone", result["first_assessment"])

    def test_floor_charger_does_not_infer_elevation(self):
        for text in ("A loose charger is on the floor, not on an elevated platform.",
                     "The charger is loose on the floor below the platform.",
                     "A loose charger is on the ground beneath the raised platform."):
            with self.subTest(text=text):
                result = build_investigation_support(text, "REVIEW")
                self.assertNotIn("An elevated charger", result["first_assessment"])
                self.assertFalse(any("falling objects" in query for query in focused_queries(text)))

    def test_denied_or_unknown_charger_movement_is_not_asserted(self):
        for text in ("There is no evidence that the charger on a platform can slide or tip.",
                     "It is unknown whether the charger on a raised platform can be moved easily.",
                     "The charger on a raised platform is not unsecured.",
                     "The charger on a platform is not loose.",
                     "There is no indication that the raised charger is unstable."):
            with self.subTest(text=text):
                result = build_investigation_support(text, "REVIEW")
                self.assertNotIn("An elevated charger that can move", result["first_assessment"])
                self.assertTrue(result["immediate_precautions"])
                self.assertLessEqual(len(result["priority_clarifications"]), 2)
                self.assertFalse(result["final_legal_verdict"])

    def test_uncertain_non_charger_movement_does_not_invent_reported_movement(self):
        text = "There is no evidence that the pump on the shelf can slide."
        result = build_investigation_support(text, "REVIEW")
        self.assertNotIn("reported movement", result["first_assessment"])
        self.assertEqual(focused_queries(text), [])

    def test_positive_movement_after_uncertainty_remains_visible(self):
        text = "It was unknown whether the charger was secure, but the charger on the raised platform can now be moved easily."
        result = build_investigation_support(text, "REVIEW")
        self.assertIn("An elevated charger that can move", result["first_assessment"])

    def test_present_bolts_do_not_become_missing_bolts(self):
        for text in ("The charger is bolted down and stable.",
                     "The charger is stable on a platform; its mounting bolts are not missing."):
            with self.subTest(text=text):
                result = build_investigation_support(text, "REVIEW", [{"section": "1910.303"}])
                self.assertNotIn("missing bolt", str(result).lower())
                self.assertNotIn("An elevated charger that can move", result["first_assessment"])

    def test_retaining_bracket_does_not_invent_missing_bolts(self):
        result = build_investigation_support("A charger is secured on a platform with a retaining bracket.", "REVIEW", [{"section": "1910.303"}])
        self.assertNotIn("bolt", str(result).lower())
        self.assertNotIn("An elevated charger that can move", result["first_assessment"])

    def test_pump_movement_receives_useful_first_assessment(self):
        result = build_investigation_support("A pump on an elevated shelf can slide.", "REVIEW")
        self.assertIn("stability check", result["first_assessment"])
        self.assertTrue(any("fall zone" in item for item in result["immediate_precautions"]))

    def test_denied_or_uncertain_sling_damage_is_not_asserted_as_a_fact(self):
        cases = ("The sling has no broken wires.", "The sling is not damaged.", "No signs of damage were observed on the sling.", "Broken wires were not found on the sling.", "It is unknown whether the sling is damaged.", "No wires were broken on the sling, which remained in service.", "No sling wires were broken during the lift.")
        for text in cases:
            with self.subTest(text=text):
                result = build_investigation_support(text, "REVIEW")
                self.assertIn("does not establish damage", result["first_assessment"])
                self.assertEqual(result["possible_concerns"], [])
                self.assertEqual(result["priority_clarifications"], [])
                self.assertFalse(result["final_legal_verdict"])

    def test_denied_liquid_is_not_asserted_as_a_current_spill(self):
        cases = ("There is no oil spill on the walkway.", "No spill was found on the walkway.", "The walkway is not wet.", "Oil was not observed on the walkway.", "The pump is no longer leaking.", "No hydraulic oil was spilled on the walkway.", "The walkway was free of oil.")
        for text in cases:
            with self.subTest(text=text):
                result = build_investigation_support(text, "REVIEW")
                self.assertIn("does not establish liquid", result["first_assessment"])
                self.assertEqual(result["possible_concerns"], [])
                self.assertEqual(result["priority_clarifications"], [])

    def test_positive_damage_after_a_denial_remains_visible(self):
        for text in ("The sling has no broken wires but is frayed.", "No damage was found on one sling; the second sling is broken."):
            with self.subTest(text=text):
                result = build_investigation_support(text, "REVIEW")
                self.assertIn("Reported damage", result["first_assessment"])
                self.assertTrue(result["possible_concerns"])

    def test_positive_spill_after_a_denial_remains_visible(self):
        for text in ("There was no spill before, but oil is now leaking onto the walkway.", "No oil was found in the first aisle; the next walkway is wet."):
            with self.subTest(text=text):
                result = build_investigation_support(text, "REVIEW")
                self.assertIn("where the liquid is reaching", result["first_assessment"])
                self.assertTrue(result["possible_concerns"])

    def test_clear_and_no_action_are_scoped_not_blanket_safety(self):
        clear = build_investigation_support("Leak contained before reaching walkway", "CLEAR")
        quiet = build_investigation_support("Unchanged area", "NO_ACTION")
        self.assertIn("scoped check", clear["first_assessment"])
        self.assertEqual(clear["priority_clarifications"], [])
        self.assertIn("not a finding that the area is safe", quiet["first_assessment"])

    def test_unknown_observation_has_practical_follow_up_without_new_citation(self):
        result = build_investigation_support("An unusual sound was reported near the work area.", "REVIEW")
        self.assertIn("responsible person", result["first_assessment"])
        self.assertTrue(result["immediate_precautions"])
        self.assertLessEqual(len(result["priority_clarifications"]), 2)
        self.assertNotIn("1910", str(result))

    def test_model_failure_has_no_conclusion_or_learning_label(self):
        result = build_investigation_support("Unusual sound near equipment", "ERROR", decision_source="model_failure")
        self.assertIn("did not complete", result["first_assessment"])
        self.assertTrue(any("trusted labels" in item for item in result["evidence_notes"]))
        self.assertFalse(result["automatic_label"])

    def test_long_forklift_shift_raises_fatigue_check_without_railroad_verdict(self):
        item = observation("No incident reported", equipment=["forklift"], measurements={"operator_shift_hours": 13.0})
        result = build_investigation_support(item, "REVIEW")
        self.assertIn("fatigue", result["first_assessment"])
        self.assertIn("does not establish", result["first_assessment"])
        self.assertEqual(len(result["priority_clarifications"]), 1)
        self.assertFalse(result["final_legal_verdict"])

    def test_machine_guard_gap_is_explicit_and_not_filled_with_a_fake_citation(self):
        result = build_investigation_support("An unfamiliar machine has accessible rotating parts", "REVIEW", [{"section": "1910.28"}])
        self.assertIn("moving parts", result["first_assessment"])
        self.assertTrue(any("not retrieved" in note for note in result["evidence_notes"]))
        self.assertTrue(any("do not reach" in item for item in result["immediate_precautions"]))
        self.assertNotIn("1910.212", str(result))

    def test_default_incident_families_receive_relevant_precautions(self):
        cases = [("A crane is beside an energized conductor", "energy"), ("A broken sling remains in a lift", "lifting"), ("A leak reaches a walkway", "liquid")]
        for text, term in cases:
            with self.subTest(text=text):
                result = build_investigation_support(text, "VIOLATION")
                self.assertIn(term, result["first_assessment"] + " ".join(result["possible_concerns"]))
                self.assertLessEqual(len(result["priority_clarifications"]), 2)

    def test_helpers_need_no_network_or_models_and_preserve_observation(self):
        item = observation(CHARGER)
        before = asdict(item)
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
            FocusedQueryPlanner().plan(item)
            result = build_investigation_support(item, "REVIEW")
            relevant_excerpt("Electric equipment shall be firmly secured.", CHARGER)
        self.assertEqual(before, asdict(item))
        for key in ("possible_concerns", "immediate_precautions", "priority_clarifications", "evidence_notes"):
            self.assertIsInstance(result[key], list)
            self.assertTrue(all(isinstance(item, str) for item in result[key]))


if __name__ == "__main__":
    unittest.main()
