"""Failed prerequisites must stop before training, evaluation, or report writes."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import evaluate_finetuned as evaluation
import train_lora_dpo as training
import validate_saved_evaluation as validation


class TrainingPreflightTests(unittest.TestCase):
    def test_label_conflict_stops_before_training(self):
        error = io.StringIO()
        with patch.object(sys, "argv", ["train"]), patch.object(
            training, "cuda_preflight", return_value={}
        ), patch.object(training, "build_sft_records", side_effect=ValueError(
            "Training row TEST conflicts with the policy"
        )), patch.object(training, "run_training") as train, contextlib.redirect_stderr(error):
            with self.assertRaises(SystemExit) as raised:
                training.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Review the training labels", error.getvalue())
        self.assertIn("No training has started", error.getvalue())
        train.assert_not_called()

    def test_cuda_failure_stops_before_reading_training_data(self):
        with patch.object(sys, "argv", ["train"]), patch.object(
            training, "cuda_preflight", side_effect=RuntimeError("CUDA is unavailable")
        ), patch.object(training, "build_sft_records") as records, patch.object(
            training, "run_training"
        ) as train, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                training.main()
        self.assertEqual(raised.exception.code, 2)
        records.assert_not_called()
        train.assert_not_called()

    def test_successful_preflight_does_not_start_training(self):
        with patch.object(sys, "argv", ["train", "--preflight-only"]), patch.object(
            training, "cuda_preflight", return_value={"device": "test"}
        ), patch.object(training, "build_sft_records", return_value=[{}]), patch.object(
            training, "run_training"
        ) as train, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(training.main(), 0)
        train.assert_not_called()

    def test_training_builder_rejects_unsupported_labels_without_changing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            path.write_text(
                "Log_ID,Equipment_ID,Equipment_Type,Location,Operator_Shift_Hours,"
                "Reported_Incident,Is_Violation,Expected_Citation\n"
                "TEST,TEST,TEST,TEST,8,No incident,1,1910.28\n",
                encoding="utf-8",
            )
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "conflicts with the policy"):
                training.build_sft_records(path)
            self.assertEqual(path.read_bytes(), before)

    def assert_evaluation_stops(self, adapter, holdout, output, message, modules=None):
        argv = ["evaluate", "--model", "unused-model", "--adapter", str(adapter),
                "--holdout", str(holdout), "--out", str(output)]
        error = io.StringIO()
        with patch.object(sys, "argv", argv), patch.object(
            evaluation, "load_rows"
        ) as rows, patch.object(evaluation, "evaluate_one") as run, patch.dict(
            sys.modules, modules or {}
        ), contextlib.redirect_stderr(error):
            with self.assertRaises(SystemExit) as raised:
                evaluation.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, error.getvalue())
        rows.assert_not_called()
        run.assert_not_called()
        self.assertFalse(output.exists())

    def test_missing_adapter_stops_before_gpu_imports_or_case_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assert_evaluation_stops(
                root / "missing", root / "missing.csv", root / "out.json",
                "Training must finish successfully", {"torch": None, "bitsandbytes": None},
            )

    def test_config_without_weights_does_not_count_as_trained_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            self.assert_evaluation_stops(
                root, root / "cases.csv", root / "out.json", "Adapter weights not found",
                {"torch": None, "bitsandbytes": None},
            )

    def test_missing_cases_stop_before_gpu_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "adapter_model.safetensors").write_bytes(b"test-only")
            self.assert_evaluation_stops(
                root, root / "missing.csv", root / "out.json", "Evaluation cases not found",
                {"torch": None, "bitsandbytes": None},
            )

    def test_missing_dependencies_and_cpu_torch_stop_before_case_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "adapter_model.safetensors").write_bytes(b"test-only")
            (root / "cases.csv").write_text("not read", encoding="utf-8")
            self.assert_evaluation_stops(
                root, root / "cases.csv", root / "out.json", "Cannot load the training dependencies",
                {"bitsandbytes": None},
            )
            fake_torch = types.SimpleNamespace(
                __version__="test+cpu", cuda=types.SimpleNamespace(is_available=lambda: False)
            )
            self.assert_evaluation_stops(
                root, root / "cases.csv", root / "out.json", "CUDA is unavailable",
                {"torch": fake_torch, "bitsandbytes": types.ModuleType("bitsandbytes"),
                 "peft": types.ModuleType("peft"), "transformers": types.ModuleType("transformers")},
            )

    def test_missing_report_explains_sequence_without_reading_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error = io.StringIO()
            with patch.object(sys, "argv", ["validate", "--report", str(root / "missing.json")]), patch.object(
                validation, "validate_case_binding"
            ) as cases, contextlib.redirect_stderr(error):
                with self.assertRaises(SystemExit) as raised:
                    validation.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("training and evaluation must finish successfully first", error.getvalue())
            self.assertIn("Omit --report", error.getvalue())
            cases.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
