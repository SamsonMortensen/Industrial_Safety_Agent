"""Keep historical evidence immutable while verifying the named policy repair."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from capture_retrieval_run import file_hash
from report_retrieval_comparison import (
    DATA, REPRODUCTION, REPRODUCTION_SCHEMA, current_capture, read, validate_capture,
)


class RetrievalReproductionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.historical = read(ROOT / DATA / "after.json")
        names = set(self.historical["input_and_code_sha256_lf"])
        names.update(f"{DATA}/{name}.json" for name in ("after", "protocol", "candidate_freeze"))
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        # Policy is not called by the lexical capture. Give this synthetic copy
        # an explicit, independently hashed policy version while retaining all
        # original retrieval-source bytes and saved case-level outcomes.
        with (self.root / "code/hybrid_policy.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# Synthetic compatibility-test policy revision.\n")
        candidate = copy.deepcopy(self.historical)
        candidate["input_and_code_sha256_lf"] = {
            name: file_hash(self.root / name) for name in self.historical["input_and_code_sha256_lf"]
        }
        self.reproduction = {
            "schema": REPRODUCTION_SCHEMA,
            "compatibility_only": True,
            "changed_source_files": ["code/hybrid_policy.py"],
            "independent_evaluation_used": False,
            "calibration_reserve_used": False,
            "model_training_performed": False,
            "model_promoted": False,
            "historical_artifacts_sha256_lf": {
                f"{DATA}/{name}.json": file_hash(self.root / DATA / f"{name}.json")
                for name in ("after", "protocol", "candidate_freeze")
            },
            "capture": candidate,
        }
        self.write_reproduction()

    def write_reproduction(self):
        path = self.root / REPRODUCTION
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.reproduction), encoding="utf-8")

    def test_named_policy_reproduction_preserves_original_evidence(self):
        hashes = dict(self.reproduction["historical_artifacts_sha256_lf"])
        result = current_capture(self.root, self.historical)
        self.assertEqual(result["records"], self.historical["records"])
        self.assertEqual(hashes, {name: file_hash(self.root / name) for name in hashes})

    def test_source_edit_after_capture_is_rejected(self):
        with (self.root / "code/hybrid_policy.py").open("a", encoding="utf-8") as stream:
            stream.write("# Changed again after the capture.\n")
        with self.assertRaisesRegex(ValueError, "differs from the named reproduction"):
            current_capture(self.root, self.historical)

    def test_retrieval_code_corpus_and_fixture_changes_are_not_exempted(self):
        for name in ("code/autonomous_investigator.py", "code/showcase.py",
                     "json/regulations.json", "json/statutes.json", "daily_yard_log.csv"):
            with self.subTest(name=name):
                original = (self.root / name).read_bytes()
                old_digest = self.reproduction["capture"]["input_and_code_sha256_lf"][name]
                try:
                    (self.root / name).write_bytes(original + b"\n")
                    self.reproduction["capture"]["input_and_code_sha256_lf"][name] = file_hash(self.root / name)
                    self.write_reproduction()
                    with self.assertRaisesRegex(ValueError, "changes frozen retrieval code or inputs"):
                        current_capture(self.root, self.historical)
                finally:
                    (self.root / name).write_bytes(original)
                    self.reproduction["capture"]["input_and_code_sha256_lf"][name] = old_digest

    def test_cannot_expand_the_reviewed_source_allowlist(self):
        self.reproduction["changed_source_files"].append("code/autonomous_investigator.py")
        self.write_reproduction()
        with self.assertRaisesRegex(ValueError, "Unsupported named"):
            current_capture(self.root, self.historical)

    def test_altered_historical_link_is_rejected(self):
        for name in ("after", "protocol", "candidate_freeze"):
            with self.subTest(name=name):
                key = f"{DATA}/{name}.json"
                original = self.reproduction["historical_artifacts_sha256_lf"][key]
                try:
                    self.reproduction["historical_artifacts_sha256_lf"][key] = "0" * 64
                    self.write_reproduction()
                    with self.assertRaisesRegex(ValueError, "unchanged historical evidence"):
                        current_capture(self.root, self.historical)
                finally:
                    self.reproduction["historical_artifacts_sha256_lf"][key] = original

    def test_case_query_changes_fail_even_when_ranks_are_identical(self):
        self.reproduction["capture"]["records"][0]["queries"][0] += " altered"
        self.write_reproduction()
        with self.assertRaisesRegex(ValueError, "historical after-run: records"):
            current_capture(self.root, self.historical)

    def test_new_case_count_is_rejected(self):
        self.reproduction["capture"]["records"].pop()
        self.write_reproduction()
        with self.assertRaisesRegex(ValueError, "cohorts changed"):
            current_capture(self.root, self.historical)

    def test_rank_and_model_request_tampering_is_rejected(self):
        changed = copy.deepcopy(self.reproduction["capture"])
        changed["records"][0]["rank"] = 99
        with self.assertRaisesRegex(ValueError, "Rank does not match"):
            validate_capture(changed)
        for value in (1, True, "0", None):
            with self.subTest(model_requests=value):
                changed = copy.deepcopy(self.reproduction["capture"])
                changed["model_requests"] = value
                with self.assertRaisesRegex(ValueError, "zero-model-request"):
                    validate_capture(changed)

    def test_protected_data_and_training_flags_are_rejected(self):
        for field in ("independent_evaluation_used", "calibration_reserve_used",
                      "model_training_performed", "model_promoted"):
            with self.subTest(field=field):
                self.reproduction[field] = True
                self.write_reproduction()
                with self.assertRaisesRegex(ValueError, "development compatibility"):
                    current_capture(self.root, self.historical)
                self.reproduction[field] = False
        for field in ("independent_evaluation_used", "calibration_reserve_used"):
            with self.subTest(capture_field=field):
                changed = copy.deepcopy(self.reproduction["capture"])
                changed[field] = True
                with self.assertRaisesRegex(ValueError, "Only development"):
                    validate_capture(changed)

    def test_source_inventory_cannot_drop_a_dependency(self):
        self.reproduction["capture"]["input_and_code_sha256_lf"].pop("code/showcase.py")
        self.write_reproduction()
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            current_capture(self.root, self.historical)


if __name__ == "__main__":
    unittest.main()
