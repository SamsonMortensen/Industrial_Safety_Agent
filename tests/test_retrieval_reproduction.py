"""Keep historical evidence immutable while verifying interactive changes."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from capture_retrieval_run import BASE_CAPTURE_SOURCES, capture_sources, file_hash
from report_retrieval_comparison import (
    DATA, PREVIOUS_REPRODUCTION, REPRODUCTION, REPRODUCTION_ADDED_SOURCES,
    REPRODUCTION_CHANGED_SOURCES, REPRODUCTION_SCHEMA, current_capture, read,
    validate_capture,
)


class RetrievalReproductionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.historical = read(ROOT / DATA / "after.json")
        names = set(self.historical["input_and_code_sha256_lf"])
        names.update(f"{DATA}/{name}.json" for name in ("after", "protocol", "candidate_freeze"))
        names.update(REPRODUCTION_ADDED_SOURCES)
        names.add(PREVIOUS_REPRODUCTION)
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        # The historical benchmark calls the frozen QueryPlanner directly.
        # Give the two permitted modules explicit synthetic versions while
        # retaining the retrieval core and all case-level outcomes.
        for name in REPRODUCTION_CHANGED_SOURCES:
            with (self.root / name).open("a", encoding="utf-8") as stream:
                stream.write("\n# Synthetic compatibility-test revision.\n")
        candidate = copy.deepcopy(self.historical)
        candidate["input_and_code_sha256_lf"] = {
            name: file_hash(self.root / name) for name in capture_sources(self.root)
        }
        self.reproduction = {
            "schema": REPRODUCTION_SCHEMA,
            "compatibility_only": True,
            "changed_source_files": list(REPRODUCTION_CHANGED_SOURCES),
            "added_source_files": list(REPRODUCTION_ADDED_SOURCES),
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
        self.reproduction["historical_artifacts_sha256_lf"][PREVIOUS_REPRODUCTION] = file_hash(self.root / PREVIOUS_REPRODUCTION)
        self.write_reproduction()

    def write_reproduction(self):
        path = self.root / REPRODUCTION
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.reproduction), encoding="utf-8")

    def test_named_interactive_reproduction_preserves_original_evidence(self):
        hashes = dict(self.reproduction["historical_artifacts_sha256_lf"])
        result = current_capture(self.root, self.historical)
        self.assertEqual(result["records"], self.historical["records"])
        self.assertEqual(hashes, {name: file_hash(self.root / name) for name in hashes})

    def test_source_edit_after_capture_is_rejected(self):
        for name in (*REPRODUCTION_CHANGED_SOURCES, *REPRODUCTION_ADDED_SOURCES):
            with self.subTest(name=name):
                original = (self.root / name).read_bytes()
                try:
                    (self.root / name).write_bytes(original + b"\n# Changed after capture.\n")
                    with self.assertRaisesRegex(ValueError, "differs from the named reproduction"):
                        current_capture(self.root, self.historical)
                finally:
                    (self.root / name).write_bytes(original)

    def test_retrieval_code_corpus_and_fixture_changes_are_not_exempted(self):
        for name in ("code/autonomous_investigator.py",
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

    def test_added_dependency_allowlist_is_exact_and_versioned(self):
        for additions in ([], ["code/other_helper.py"], [*REPRODUCTION_ADDED_SOURCES, "code/other_helper.py"]):
            with self.subTest(additions=additions):
                self.reproduction["added_source_files"] = additions
                self.write_reproduction()
                with self.assertRaisesRegex(ValueError, "Unsupported named"):
                    current_capture(self.root, self.historical)
        self.reproduction["added_source_files"] = list(REPRODUCTION_ADDED_SOURCES)
        self.reproduction["schema"] = "lexical_retrieval_compatibility_reproduction_v1"
        self.write_reproduction()
        with self.assertRaisesRegex(ValueError, "Unsupported named"):
            current_capture(self.root, self.historical)

    def test_altered_historical_link_is_rejected(self):
        for key in self.reproduction["historical_artifacts_sha256_lf"]:
            with self.subTest(name=key):
                original = self.reproduction["historical_artifacts_sha256_lf"][key]
                try:
                    self.reproduction["historical_artifacts_sha256_lf"][key] = "0" * 64
                    self.write_reproduction()
                    with self.assertRaisesRegex(ValueError, "unchanged historical evidence"):
                        current_capture(self.root, self.historical)
                finally:
                    self.reproduction["historical_artifacts_sha256_lf"][key] = original

    def test_previous_reproduction_cannot_be_rewritten_even_with_updated_link(self):
        path = self.root / PREVIOUS_REPRODUCTION
        path.write_bytes(path.read_bytes() + b"\n")
        self.reproduction["historical_artifacts_sha256_lf"][PREVIOUS_REPRODUCTION] = file_hash(path)
        self.write_reproduction()
        with self.assertRaisesRegex(ValueError, "previous named reproduction must remain unchanged"):
            current_capture(self.root, self.historical)

    def test_old_evidence_cannot_be_rewritten_even_with_updated_link(self):
        for name in ("after", "protocol", "candidate_freeze"):
            key = f"{DATA}/{name}.json"
            path = self.root / key
            original = path.read_bytes()
            old_link = self.reproduction["historical_artifacts_sha256_lf"][key]
            try:
                path.write_bytes(original + b"\n")
                self.reproduction["historical_artifacts_sha256_lf"][key] = file_hash(path)
                self.write_reproduction()
                with self.assertRaisesRegex(ValueError, "historical after-run, protocol or freeze changed"):
                    current_capture(self.root, self.historical)
            finally:
                path.write_bytes(original)
                self.reproduction["historical_artifacts_sha256_lf"][key] = old_link

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
        self.reproduction["capture"]["input_and_code_sha256_lf"].pop("code/investigation_support.py")
        self.write_reproduction()
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            current_capture(self.root, self.historical)

    def test_source_inventory_cannot_add_an_unreviewed_dependency(self):
        self.reproduction["capture"]["input_and_code_sha256_lf"]["code/other_helper.py"] = "0" * 64
        self.write_reproduction()
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            current_capture(self.root, self.historical)

    def test_capture_inventory_tracks_the_new_dependency_and_old_checkouts(self):
        self.assertEqual(capture_sources(self.root), [*BASE_CAPTURE_SOURCES, *REPRODUCTION_ADDED_SOURCES])
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(capture_sources(Path(directory)), list(BASE_CAPTURE_SOURCES))

    def test_missing_named_reproduction_cannot_fall_back_to_old_evidence(self):
        (self.root / REPRODUCTION).unlink()
        with self.assertRaisesRegex(ValueError, "requires the named retrieval reproduction"):
            current_capture(self.root, self.historical)


if __name__ == "__main__":
    unittest.main()
