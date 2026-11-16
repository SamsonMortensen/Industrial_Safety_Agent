"""Evidence can answer a question without asserting that an incident happened."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from autonomous_investigator import HybridRetriever, Observation, TriggerEngine
from autonomous_investigator import GroundedDecision
from citation_alignment import InteractiveInvestigator, information_request


class CitationIntentTests(unittest.TestCase):
    def test_explicit_questions_and_hypotheticals_do_not_reach_decision_providers(self):
        policy, reasoner = Mock(), Mock()
        investigator = InteractiveInvestigator(
            HybridRetriever([{"section": "1910.22", "citation": "29 CFR 1910.22",
                              "text": "Walking surfaces must be free of oil spills."}]),
            trigger_engine=TriggerEngine(sample_rate=0), policy_fn=policy, reasoner=reasoner,
        )
        for text in (
            "Is an oil spill on the walkway covered by housekeeping?",
            "Which rule covers a cracked lifting sling?",
            "Does a cleaned walkway meet the housekeeping rule?",
            "Hypothetically, an oil spill remains on the walkway.",
            "Suppose oil remains on the employee walkway.",
            "Imagine a sling remains damaged and in use.",
            "Please explain the rule for wet walking surfaces.",
        ):
            with self.subTest(text=text):
                observation = Observation.from_mapping({"summary": text, "human_flag": True})
                result = investigator.investigate(observation)
                self.assertEqual((result.status, result.citation), ("REVIEW", None))
                self.assertEqual(result.decision_source, "information_request")
                self.assertEqual(observation.summary, text)
        policy.assert_not_called()
        reasoner.decide.assert_not_called()

    def test_statements_followed_by_questions_still_report_conditions(self):
        for text in ("Oil remains on the employee walkway. Is this unsafe?",
                     "Can you help? An oil spill remains on the employee walkway.",
                     "A cracked sling is still in use; which rule applies?",
                     "I saw oil on the floor. What should I do?"):
            self.assertFalse(information_request(text))

    def test_structured_report_is_not_erased_by_a_question_title(self):
        policy = Mock(return_value=GroundedDecision("CLEAR", None, "controlled test", "test"))
        investigator = InteractiveInvestigator(HybridRetriever([]), policy_fn=policy,
                                               trigger_engine=TriggerEngine(sample_rate=0))
        observation = Observation.from_mapping({"summary": "Is this unsafe?", "human_flag": True,
                                               "conditions": ["Reported condition in structured input"]})
        result = investigator.investigate(observation)
        self.assertEqual(result.status, "CLEAR")
        policy.assert_called_once()
        policy.reset_mock()
        for title in ("Suppose this occurred", "Please suppose oil remains on the walkway.",
                      "Consider a hypothetical oil spill."):
            hypothetical = Observation.from_mapping({"summary": title, "human_flag": True,
                                                    "conditions": ["Hypothetical condition"]})
            self.assertEqual(investigator.investigate(hypothetical).status, "REVIEW")
        policy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
