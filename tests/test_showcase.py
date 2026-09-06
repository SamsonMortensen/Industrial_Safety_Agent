"""Exercise the visitor workflow and its resource and evidence boundaries."""

import contextlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import showcase
from hybrid_policy import evaluate_event


class ShowcaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations = showcase.load_observations(
            ROOT / "data/showcase_observations.json"
        )

    def test_offline_tour_runs_real_features_without_network_or_writes(self):
        with patch.object(
            socket.socket, "connect", side_effect=AssertionError("network")
        ), patch("builtins.open", wraps=open):
            before = {p: showcase.file_hash(p) for p in (ROOT / "json").glob("*.json")}
            report = showcase.run_showcase(self.observations)
        after = {p: showcase.file_hash(p) for p in (ROOT / "json").glob("*.json")}
        self.assertEqual(before, after)
        live = report["live_run"]
        statuses = {item["event_id"]: item["status"] for item in live["observations"]}
        self.assertEqual(statuses["showcase-crane"], "VIOLATION")
        self.assertEqual(statuses["showcase-contained-spill"], "CLEAR")
        self.assertEqual(statuses["showcase-uncertain-condition"], "REVIEW")
        self.assertEqual(statuses["showcase-quiet-scene"], "NO_ACTION")
        self.assertEqual(live["benchmark"]["cases"], 16)
        self.assertEqual(live["benchmark"]["hazard_families"], 16)
        self.assertFalse(live["benchmark"]["labels_used_to_form_queries"])
        self.assertEqual(live["grounding_check"]["status"], "REVIEW")
        self.assertEqual(
            [item["trusted"] for item in live["learning_gate_examples"]],
            [False, True, True, False],
        )
        self.assertEqual(report["runtime"]["model_requests"], 0)
        self.assertTrue(
            all(
                item["source"] == "saved_reference_not_rerun"
                for item in report["saved_evidence"]
            )
        )
        for result in live["observations"]:
            if result["status"] == "VIOLATION":
                self.assertTrue(result["citation_in_candidates"])

    def test_uncertain_facts_are_not_treated_as_confirmed_hazards(self):
        for text in (
            "A worker is beside a cabinet; it is unknown whether parts are energized.",
            "A worker is beside energized parts; isolation is not confirmed.",
            "The image does not show whether the platform has a guardrail.",
        ):
            with self.subTest(text=text):
                result = evaluate_event({"Reported_Incident": text})
                self.assertEqual((result.status, result.citation), ("REVIEW", None))

    def test_missing_or_invalid_duty_duration_is_not_clear(self):
        for value in (None, "", "unknown", "nan", "inf", -1):
            for equipment in (
                "Freight train employee in covered switching service",
                "Passenger train employee",
            ):
                with self.subTest(value=value, equipment=equipment):
                    result = evaluate_event(
                        {"Reported_Incident": equipment, "Operator_Shift_Hours": value}
                    )
                    self.assertEqual((result.status, result.citation), ("REVIEW", None))
        observation = showcase.text_observation(
            "Freight train employee in covered switching service"
        )
        self.assertIsNone(observation.to_legacy_row()["Operator_Shift_Hours"])

    def test_benchmark_selection_and_queries_are_deterministic(self):
        self.assertEqual(showcase.benchmark_sample(16), showcase.benchmark_sample(16))
        row = showcase.benchmark_sample(1)[0]
        changed = {
            **row,
            "Expected_Section": "SENTINEL",
            "Is_Violation": "0",
            "Violation_Type": "SENTINEL",
        }
        self.assertEqual(
            showcase.benchmark_observation(row), showcase.benchmark_observation(changed)
        )
        self.assertNotIn(
            "SENTINEL", showcase.benchmark_observation(changed).evidence_text()
        )
        with self.assertRaises(ValueError):
            showcase.benchmark_sample(65)

    def test_json_output_is_machine_readable_and_does_not_persist(self):
        output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            [
                "showcase",
                "--text",
                "Unusual equipment movement",
                "--benchmark-cases",
                "0",
                "--json",
            ],
        ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(showcase.main(), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["live_run"]["benchmark"]["cases"], 0)
        self.assertFalse(report["runtime"]["learning_memory_modified"])

    def test_input_limits_reject_large_or_invalid_payloads(self):
        for text in ("", "x" * 4001):
            with self.assertRaises(ValueError):
                showcase.text_observation(text)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for payload in ([{"summary": "x"}] * 7, [1], {"summary": ""}):
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    showcase.load_observations(path)
            path.write_bytes(b" " * 65537)
            with self.assertRaises(ValueError):
                showcase.load_observations(path)

    def test_export_is_confined_to_ignored_directory(self):
        with self.assertRaises(ValueError):
            showcase.export_path(ROOT / "json/audit_results.json")
        with self.assertRaises(ValueError):
            showcase.export_path(ROOT / "benchmark_runs/showcase/../../README.md")
        with tempfile.TemporaryDirectory() as directory, patch.object(
            showcase, "ROOT", Path(directory)
        ):
            target = Path(directory) / "benchmark_runs/showcase/run.json"
            # Windows may spell temporary paths with an 8.3 directory alias.
            self.assertEqual(showcase.export_path(target), target.resolve())
            target.parent.mkdir(parents=True)
            target.write_text("original", encoding="utf-8")
            with self.assertRaises(ValueError):
                showcase.export_path(target)
            self.assertEqual(target.read_text(encoding="utf-8"), "original")

    def test_optional_model_has_one_request_budget_and_loopback_only(self):
        with patch("showcase.ollama_base_url", return_value="http://127.0.0.1:11434"):
            reasoner = showcase.OneRequestReasoner("example", 5)
        reasoner.delegate = MagicMock()
        reasoner.delegate.decide.return_value = showcase.GroundedDecision(
            "REVIEW", None, "test", "test"
        )
        observation = showcase.text_observation("Uncertain event")
        reasoner.decide(observation, [])
        self.assertEqual(reasoner.decide(observation, []).source, "showcase_budget")
        reasoner.delegate.decide.assert_called_once()
        with patch(
            "showcase.ollama_base_url", return_value="http://example.com:11434"
        ), self.assertRaises(ValueError):
            showcase.OneRequestReasoner("example", 5)

    def test_interactive_mode_stays_offline_and_can_exit(self):
        with patch(
            "builtins.input", side_effect=["A damaged sling remained in use", "quit"]
        ), patch.object(
            socket.socket, "connect", side_effect=AssertionError("network")
        ), contextlib.redirect_stdout(
            io.StringIO()
        ) as output:
            showcase.interactive_session()
        self.assertIn("visitor-incident", output.getvalue())


if __name__ == "__main__":
    unittest.main()
