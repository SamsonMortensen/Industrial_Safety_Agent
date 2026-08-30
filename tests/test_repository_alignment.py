import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from audit_agent import parse
from audit_agent_v2 import compute_metrics
from generate_yard_log import HAZARDS, audit_labels, corpus_sections, generate
from self_reflection import StatutoryGroundedVerifier


class RepositoryAlignmentTests(unittest.TestCase):
    def test_freight_duty_limit_uses_train_employee_statute(self):
        duty = next(item for item in HAZARDS if item["kind"] == "duty_limit")
        self.assertEqual(duty["section"], "21103")
        self.assertTrue(
            all("employee" in text.lower() for text in duty["bad"] + duty["ok"])
        )

    def test_generated_labels_are_in_the_combined_corpus(self):
        rows = generate(1000)
        self.assertEqual(audit_labels(rows, corpus_sections()), [])

    def test_statute_is_retrievable_and_verifiable(self):
        statutes = json.loads(
            (ROOT / "json" / "statutes.json").read_text(encoding="utf-8")
        )
        self.assertIn("21103", statutes["sections"])
        self.assertIn("21103", StatutoryGroundedVerifier().valid_sections)
        self.assertEqual(
            parse("STATUS: VIOLATION\nCITATION: 49 U.S.C. 21103\nREASON: test")[
                "citation"
            ],
            "21103",
        )

    def test_saved_yard_log_matches_the_generator_contract(self):
        with (ROOT / "daily_yard_log.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1000)
        self.assertEqual(sum(int(row["Is_Violation"]) for row in rows), 144)
        fatigue = [row for row in rows if row["Violation_Type"] == "duty_limit"]
        self.assertEqual(len(fatigue), 9)
        self.assertEqual({row["Expected_Section"] for row in fatigue}, {"21103"})

    def test_readme_and_notebook_do_not_carry_stale_scope_claims(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        notebook = (ROOT / "main.ipynb").read_text(encoding="utf-8")
        self.assertNotIn("\u2014", readme)
        self.assertNotIn("228.7", notebook)
        self.assertIn("49 U.S.C. 21103", readme)
        self.assertIn("Fresh QLoRA result", readme)

    def test_measured_finetune_result_has_locked_holdout(self):
        report = json.loads(
            (ROOT / "json" / "finetune_evaluation_results.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["provenance"]["holdout_cases"], 56)
        self.assertEqual(
            report["strategies"]["qlora_sft"]["metrics"]["accuracy"], 0.8929
        )
        training = json.loads(
            (ROOT / "json" / "finetune_training_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(training["holdout_opened_by_trainer"])

    def test_unparsed_outputs_are_strict_errors_not_metric_crashes(self):
        rows = [
            {
                "status": "UNPARSED",
                "predicted": None,
                "actual": 1,
                "citation": None,
                "citation_exists": None,
                "citation_was_retrieved": None,
            },
            {
                "status": "UNPARSED",
                "predicted": None,
                "actual": 0,
                "citation": None,
                "citation_exists": None,
                "citation_was_retrieved": None,
            },
        ]
        metrics = compute_metrics(rows)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["unparsed"], 2)


if __name__ == "__main__":
    unittest.main()
