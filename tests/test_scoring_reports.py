"""Exercise report output without turning missing labels or citations into credit."""
import asyncio
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import apply_policy_controls as controls
import audit_agent_v2 as auditor
import run_3way_comparison as threeway
import run_improvement_benchmark as improvement
import scaled_benchmark as scaled
import validate_saved_evaluation as saved
from audit_outcomes import format_metric


def record(label=None, status="UNPARSED", number=0):
    return {"log_id": f"fixture-{number}", "actual": label, "status": status,
            "predicted": {"CLEAR": 0, "VIOLATION": 1}.get(status),
            "citation": None, "citation_raw": "NONE", "reason": "Synthetic report check.",
            "citation_exists": None, "citation_was_retrieved": None, "audited_by": "model"}


def metrics(rows):
    return dict(auditor.compute_metrics(rows), critique_corrections=0)


def fixture_services(module, directory):
    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(module, "JSON_DIR", directory))
    stack.enter_context(patch.object(module, "load_corpus_index", return_value=([], [], set())))
    stack.enter_context(patch.object(module, "EpisodicMemoryBank",
                                    side_effect=lambda **kwargs: SimpleNamespace(episodes=[], vectors=None)))
    stack.enter_context(patch.object(module, "AdaptivePillarBank", return_value=SimpleNamespace()))
    stack.enter_context(patch.object(module, "StatutoryGroundedVerifier", return_value=SimpleNamespace()))
    stack.enter_context(patch.object(module, "SelfReflectionCritic", return_value=SimpleNamespace()))
    return stack


class ScoringReportTests(unittest.TestCase):
    def test_undefined_rates_display_as_na_without_changing_values(self):
        self.assertEqual(format_metric(None), "n/a")
        self.assertEqual(format_metric(None, ".2%"), "n/a")
        self.assertEqual(format_metric(0), "0.0000")
        self.assertEqual(format_metric(.5, ".2%"), "50.00%")

    def test_missing_or_invalid_citation_target_cannot_earn_joint_credit(self):
        for target in (None, "", "NONE", "not a section"):
            with self.subTest(target=target):
                row = dict(record(1, "VIOLATION"), citation=target,
                           expected_section=target, actual_type="housekeeping")
                result = controls.score([row])
                hazard = controls.per_hazard([row])["housekeeping"]
                self.assertEqual(result["accuracy"], 1.0)
                self.assertEqual(result["citation_correctness"], 0.0)
                self.assertEqual(result["correct_citations_on_true_positives"], 0)
                self.assertEqual(result["joint_status_and_citation_accuracy"], 0.0)
                self.assertEqual(hazard["correct_citations"], 0)

    def test_explicit_matching_citation_earns_credit(self):
        for section in ("1910.22", "1910.22(a)(1)", "21103"):
            with self.subTest(section=section):
                row = dict(record(1, "VIOLATION"), citation=section,
                           expected_section=section, actual_type="fixture")
                result = controls.score([row])
                self.assertEqual(result["citation_correctness"], 1.0)
                self.assertEqual(result["joint_status_and_citation_accuracy"], 1.0)
                self.assertEqual(controls.per_hazard([row])["fixture"]["correct_citations"], 1)

    def test_per_hazard_uses_the_effective_unresolved_status(self):
        row = dict(record(1, "REVIEW"), predicted=1, citation="1910.22",
                   expected_section="1910.22", actual_type="fixture")
        result = controls.per_hazard([row])["fixture"]
        self.assertEqual(result["detected"], 0)
        self.assertEqual(result["correct_citations"], 0)

    def test_saved_evaluation_cli_accepts_valid_all_negative_cohort(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            holdout = directory / "holdout.csv"
            holdout.write_text("Log_ID,Is_Violation,Expected_Citation\nfixture-0,0,NONE\n", encoding="utf-8")
            row = dict(record(0, "CLEAR"), citation_correct=True, expected_citation="NONE")
            report = {"strategies": {name: {"results": [copy.deepcopy(row)],
                      "metrics": {"runtime_seconds": 0, "peak_gpu_memory_mib": None}}
                      for name in ("base", "qlora_sft")},
                      "provenance": {"holdout_sha256": saved.file_sha256(holdout)}}
            for payload in report["strategies"].values():
                saved.recompute_strategy(payload)
                self.assertIsNone(payload["metrics"]["citation_correctness"])
            report["comparison"] = scaled.exact_mcnemar(
                report["strategies"]["base"]["results"], report["strategies"]["qlora_sft"]["results"])
            path = directory / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            before = path.read_bytes()
            output = io.StringIO()
            with patch.object(sys, "argv", ["validate_saved_evaluation", "--report", str(path),
                                             "--holdout", str(holdout)]), contextlib.redirect_stdout(output):
                self.assertEqual(saved.main(), 0)
            self.assertIn("citation=n/a", output.getvalue())
            self.assertEqual(path.read_bytes(), before)

    def test_scaled_cli_displays_an_undefined_citation_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            output = io.StringIO()
            payload = {"strategies": {"hybrid": {"metrics": dict(metrics([record(0, "CLEAR")]),
                        citation_correctness=None, runtime_seconds=0.0)}}}
            with patch.object(sys, "argv", ["scaled_benchmark", "--out", str(Path(folder) / "report.json")]), \
                 patch.object(scaled, "run_benchmark", AsyncMock(return_value=payload)), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(scaled.main(), 0)
            self.assertIn("citation_correctness=n/a", output.getvalue())

    def test_improvement_report_shows_outage_misses_separately(self):
        rows = [record(i % 2, number=i) for i in range(8)]
        async def fake_pass(**kwargs):
            return copy.deepcopy(rows), metrics(rows)
        with tempfile.TemporaryDirectory() as folder, fixture_services(improvement, Path(folder)), \
             patch.object(improvement, "create_challenging_dataset", return_value=[]), \
             patch.object(improvement, "run_audit_pass", side_effect=fake_pass):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                asyncio.run(improvement.execute_experiment())
        lines = output.getvalue().splitlines()
        self.assertEqual(next(line for line in lines if line.startswith("False negative decisions")).split()[-3:], ["0"] * 3)
        self.assertEqual(next(line for line in lines if line.startswith("Missed positive cases")).split()[-3:], ["4"] * 3)
        self.assertEqual(next(line for line in lines if line.startswith("Unresolved cases")).split()[-3:], ["8"] * 3)

    def test_unlabeled_improvement_report_does_not_claim_a_correction(self):
        baseline = [record(None, "CLEAR")]
        unresolved = [record(None)]
        async def fake_pass(**kwargs):
            rows = baseline if kwargs["is_baseline"] else unresolved
            return copy.deepcopy(rows), metrics(rows)
        dataset = [{"Log_ID": "fixture-0", "Reported_Incident": "Synthetic event.", "Location": "Fixture"}]
        with tempfile.TemporaryDirectory() as folder, fixture_services(improvement, Path(folder)), \
             patch.object(improvement, "create_challenging_dataset", return_value=dataset), \
             patch.object(improvement, "run_audit_pass", side_effect=fake_pass):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                asyncio.run(improvement.execute_experiment())
        self.assertIn("Accuracy: n/a", output.getvalue())
        self.assertNotIn("[CORRECTED CASE", output.getvalue())

    def test_unlabeled_threeway_report_does_not_claim_a_correction(self):
        baseline = [record(None, "CLEAR")]
        unresolved = [record(None)]
        async def fake_pass(**kwargs):
            return copy.deepcopy(unresolved), metrics(unresolved)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "daily_yard_log.csv").write_text("Log_ID,Reported_Incident\nfixture-0,Synthetic event.\n", encoding="utf-8")
            with fixture_services(threeway, directory), patch.object(threeway, "ROOT", directory), \
                 patch.object(threeway, "audit_ungrounded", AsyncMock(return_value=(baseline, metrics(baseline)))), \
                 patch.object(improvement, "run_audit_pass", side_effect=fake_pass):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    asyncio.run(threeway.run_3way_comparison())
        self.assertIn("F1: n/a", output.getvalue())
        self.assertIn("Verdict changes: 0 corrected by the learner, 0 regressed.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
