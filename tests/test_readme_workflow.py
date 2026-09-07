"""Regression checks for the public first-run workflow."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import check_setup
import continuous_learner
import fetch_osha_incidents
import osha_incident_pipeline
import run_autonomous_investigation
import validate_saved_evaluation
from ollama_config import ollama_base_url


class ReadmeWorkflowTests(unittest.TestCase):
    def test_missing_source_csv_explains_download_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "pipeline",
                "--input",
                str(root / "missing.csv"),
                "--out",
                str(root / "new/observations.jsonl"),
                "--profile",
                str(root / "new/profile.json"),
            ]
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(
                io.StringIO()
            ) as error:
                with self.assertRaises(SystemExit) as raised:
                    osha_incident_pipeline.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("fetch_osha_incidents.py", error.getvalue())
            self.assertFalse((root / "new").exists())

    def test_embedding_progress_does_not_pollute_json_stdout(self):
        response = MagicMock()
        response.json.return_value = {"embeddings": [[3.0, 4.0]]}
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("requests.post", return_value=response), contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr):
            vectors = continuous_learner.embed_texts(["walkway"])
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("embedded 1/1", stderr.getvalue())
        self.assertAlmostEqual(float(vectors[0, 0]), 0.6)

    def test_ollama_addresses_are_consistent(self):
        for raw, expected in (
            ("127.0.0.1:11434", "http://127.0.0.1:11434"),
            (" http://localhost:11434/ ", "http://localhost:11434"),
            ("https://example.com/ollama/", "https://example.com/ollama"),
            ("[::1]:11434", "http://[::1]:11434"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(ollama_base_url(raw), expected)
        with patch.dict(os.environ, {"OLLAMA_HOST": "127.0.0.1:11434"}):
            self.assertEqual(ollama_base_url(), "http://127.0.0.1:11434")

    def test_invalid_ollama_addresses_fail_early(self):
        for raw in (
            "ftp://localhost",
            "http://",
            "localhost:99999",
            "http://localhost?x=y",
            "http://name:secret@localhost",
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                ollama_base_url(raw)

    def test_local_setup_does_not_need_ollama(self):
        with patch.object(sys, "argv", ["check_setup"]), patch.object(
            check_setup, "check_ollama"
        ) as remote, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_setup.main(), 0)
            remote.assert_not_called()

    def test_setup_reports_missing_checkout_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(
                any(
                    "daily_yard_log.csv" in error
                    for error in check_setup.check_local(Path(directory))
                )
            )

    def test_setup_reports_missing_models_without_loading_them(self):
        response = MagicMock()
        response.json.return_value = {"models": []}
        with patch("requests.get", return_value=response), patch(
            "requests.post"
        ) as post:
            errors = check_setup.check_ollama("http://localhost:11434", embed=True)
            self.assertEqual(len(errors), 2)
            post.assert_not_called()

    def test_setup_rejects_empty_embedding(self):
        tags = MagicMock()
        tags.json.return_value = {
            "models": [{"name": "mxbai-embed-large:latest"}, {"name": "qwen3.5:9b"}]
        }
        embedded = MagicMock()
        embedded.json.return_value = {"embeddings": []}
        with patch("requests.get", return_value=tags), patch(
            "requests.post", return_value=embedded
        ):
            self.assertTrue(
                check_setup.check_ollama("http://localhost:11434", embed=True)
            )

    def test_lexical_observation_does_not_call_embedding_service(self):
        output = io.StringIO()
        argv = [
            "investigate",
            str(ROOT / "data/example_camera_observation.json"),
            "--sample-rate",
            "0",
            "--lexical-only",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            run_autonomous_investigation, "embed_texts"
        ) as embed, contextlib.redirect_stdout(output):
            run_autonomous_investigation.main()
            embed.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(report["retrieval_mode"], "lexical")
        self.assertFalse(report["reasoner_enabled"])
        self.assertTrue(report["results"][0]["candidates"])

    def test_saved_validation_is_read_only_and_needs_no_adapter(self):
        report = ROOT / "json/finetune_evaluation_results.json"
        before = report.read_bytes()
        with patch.object(sys, "argv", ["validate"]), contextlib.redirect_stdout(
            io.StringIO()
        ):
            self.assertEqual(validate_saved_evaluation.main(), 0)
        self.assertEqual(report.read_bytes(), before)

    def test_saved_validation_rejects_incorrect_metrics_without_repairing_them(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "changed.json"
            payload = json.loads(
                (ROOT / "json/finetune_evaluation_results.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["strategies"]["base"]["metrics"]["accuracy"] = 0.5
            report.write_text(json.dumps(payload), encoding="utf-8")
            before = report.read_bytes()
            with patch.object(
                sys, "argv", ["validate", "--report", str(report)]
            ), self.assertRaisesRegex(ValueError, "metrics"):
                validate_saved_evaluation.main()
            self.assertEqual(report.read_bytes(), before)

    def test_osha_403_explains_manual_download(self):
        with tempfile.TemporaryDirectory() as directory:
            response = MagicMock()
            response.__enter__.return_value.status_code = 403
            error = io.StringIO()
            argv = ["fetch", "--directory", directory]
            with patch.object(sys, "argv", argv), patch(
                "requests.get", return_value=response
            ), contextlib.redirect_stderr(error), self.assertRaises(
                SystemExit
            ) as raised:
                fetch_osha_incidents.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("manually", error.getvalue())
            self.assertFalse(list(Path(directory).iterdir()))

    def test_csv_line_endings_are_pinned_for_fingerprints(self):
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("data/scaled_benchmark_holdout.csv text eol=crlf", attributes)
        self.assertIn("data/scaled_benchmark_train.csv text eol=crlf", attributes)
        self.assertIn(
            "*.csv text eol=lf", (ROOT / ".gitattributes").read_text(encoding="utf-8")
        )

    def test_documented_command_options_parse_without_training(self):
        for script in (
            "check_setup",
            "run_autonomous_investigation",
            "benchmark_autonomous_retrieval",
            "benchmark_authentic_incidents",
            "audit_agent",
            "audit_agent_v2",
            "apply_policy_controls",
            "fetch_osha_incidents",
            "osha_incident_pipeline",
            "train_lora_dpo",
            "evaluate_finetuned",
            "validate_saved_evaluation",
        ):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "code" / f"{script}.py"), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
