"""Protect published evidence while allowing quick local benchmark runs."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import benchmark_authentic_incidents as benchmark


class BenchmarkOutputTests(unittest.TestCase):
    def test_quick_and_full_runs_get_separate_nonreference_paths(self):
        quick = benchmark.output_path(None, 50, True)
        full = benchmark.output_path(None, 500, True)
        self.assertNotEqual(quick, full)
        self.assertEqual(quick.parent, benchmark.RUN_DIR.resolve())
        self.assertTrue(quick.name.endswith("_hybrid_50.json"))
        self.assertTrue(full.name.endswith("_hybrid_500.json"))
        self.assertNotIn(quick, benchmark.REFERENCE_OUTPUTS)

    def test_reference_rejection_happens_before_input_or_embedding_work(self):
        for reference in benchmark.REFERENCE_OUTPUTS:
            with self.subTest(reference=reference):
                argv = [
                    "benchmark",
                    "--sample",
                    "50",
                    "--dense",
                    "--input",
                    "missing-input.jsonl",
                    "--out",
                    str(reference),
                ]
                with patch.object(sys, "argv", argv), patch.object(
                    benchmark, "embed_texts"
                ) as embedder, contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as raised:
                        benchmark.main()
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("published reference", error.getvalue())
                    embedder.assert_not_called()

    def test_existing_result_survives_output_selection_and_write_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "quick.json"
            selected = benchmark.output_path(target, 50, False)
            first = {"coverage": {"cases": 50}}
            benchmark.save_report(selected, first)
            before = selected.read_bytes()
            with self.assertRaises(FileExistsError):
                benchmark.output_path(target, 500, True)
            with self.assertRaises(FileExistsError):
                benchmark.save_report(selected, {"coverage": {"cases": 500}})
            self.assertEqual(selected.read_bytes(), before)
            self.assertEqual(json.loads(before), first)

    def test_nonpositive_sample_is_rejected(self):
        for sample in (0, -1):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                benchmark.output_path(None, sample, False)

    def test_missing_or_empty_input_stops_before_embedding_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "observations.jsonl"
            target = Path(directory) / "reports" / "run.json"
            for empty in (False, True):
                if empty:
                    source.write_text("", encoding="utf-8")
                argv = [
                    "benchmark",
                    "--input",
                    str(source),
                    "--out",
                    str(target),
                    "--dense",
                ]
                with patch.object(sys, "argv", argv), patch.object(
                    benchmark, "embed_texts"
                ) as embedder, contextlib.redirect_stderr(
                    io.StringIO()
                ) as error, contextlib.redirect_stdout(
                    io.StringIO()
                ) as output:
                    with self.assertRaises(SystemExit) as raised:
                        benchmark.main()
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(
                        "empty" if empty else "osha_incident_pipeline.py",
                        error.getvalue(),
                    )
                    self.assertNotIn("Saving this run", output.getvalue())
                    embedder.assert_not_called()
                    self.assertFalse(target.parent.exists())


if __name__ == "__main__":
    unittest.main()
