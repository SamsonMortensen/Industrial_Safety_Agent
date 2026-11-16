"""Check authored relevance metrics, provenance and the prompt-only boundary."""
import copy
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import evaluate_wording_retrieval as evaluation


def case(case_id="one", group="group", kind="single_authority", targets=None):
    if targets is None:
        targets = [] if kind in {"no_safety_authority", "insufficient_facts"} else ["1910.22"]
    return {"case_id": case_id, "group_id": group, "family": "authored_example",
            "prompt": "There is oil on the walkway. Which requirement is relevant?",
            "prompt_style": "question", "fact_state": "hazard_reported",
            "expectation": {"match_type": kind, "expected_sections": targets,
                            "single_most_aligned_required": kind == "single_authority",
                            "supporting_refs": [], "violation_label": None,
                            "label_status": "authored_unreviewed_development_reference"}}


def fixture(cases=None):
    cases = cases or [case()]
    return {"schema": "grouped_wording_retrieval_cases_v1", "split": "development",
            "query_input_field": "prompt", "labels_must_not_form_queries": True,
            "existing_independent_evaluation_used": False, "calibration_reserve_used": False,
            "groups": len({c["group_id"] for c in cases}), "cases": cases}


def result(sections=None, lead=None, final_citation=None):
    sections = sections if sections is not None else ["1910.22"]
    return {"status": "REVIEW", "citation": final_citation,
            "queries": ["Only the user's prompt"],
            "candidates": [{"section": section} for section in sections],
            "citation_alignment": {"highest_ranked_research_match": {"section": lead} if lead else None,
                                   "accepted_as_correct": False}}


class WordingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_fixture(self, payload=None):
        path = self.root / "cases.json"
        path.write_text(json.dumps(payload or fixture()), encoding="utf-8")
        return path

    def test_only_prompt_enters_investigator_when_references_change(self):
        seen = []
        payload = fixture()
        rows = evaluation.evaluate_cases(payload, lambda prompt: seen.append(prompt) or result())
        changed = copy.deepcopy(payload)
        changed["cases"][0]["expectation"]["expected_sections"] = ["1910.303"]
        changed["cases"][0]["expectation"]["supporting_refs"] = ["Do not use this as a query"]
        changed["cases"][0]["family"] = "different_label"
        changed["cases"][0]["fact_state"] = "hazard_denied"
        other = evaluation.evaluate_cases(changed, lambda prompt: seen.append(prompt) or result())
        self.assertEqual(seen, [payload["cases"][0]["prompt"]] * 2)
        self.assertEqual(rows[0]["result"]["queries"], other[0]["result"]["queries"])
        self.assertNotEqual(rows[0]["ranks"], other[0]["ranks"])

    def test_unrelated_and_protected_evaluation_formats_are_rejected(self):
        for change in ({"schema": "independent_evaluation_v1"}, {"split": "independent_evaluation"},
                       {"existing_independent_evaluation_used": True}, {"calibration_reserve_used": True},
                       {"query_input_field": "expected_sections"}, {"labels_must_not_form_queries": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluation.validate_cases(dict(fixture(), **change))

    def test_training_and_violation_labels_are_not_accepted(self):
        for change in ({"violation_label": True}, {"label_status": "reviewer_approved"}):
            payload = fixture()
            payload["cases"][0]["expectation"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluation.validate_cases(payload)

    def test_duplicate_cases_and_false_group_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluation.validate_cases(fixture([case(), case()]))
        with self.assertRaises(ValueError):
            evaluation.validate_cases(dict(fixture(), groups=2))

    def test_case_text_file_and_count_budgets_are_enforced(self):
        payload = fixture()
        payload["cases"][0]["prompt"] = "a" * 4001
        with self.assertRaises(ValueError):
            evaluation.validate_cases(payload)
        with self.assertRaises(ValueError):
            evaluation.validate_cases(fixture([case(str(i)) for i in range(257)]))
        path = self.root / "large.json"
        path.write_bytes(b" " * (evaluation.MAX_CASE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "one MiB"):
            evaluation.load_cases(path)

    def test_target_counts_prevent_vacuous_multi_target_success(self):
        for value in (case(kind="single_authority", targets=[]),
                      case(kind="multiple_authorities", targets=["1910.22"]),
                      case(kind="no_safety_authority", targets=["1910.22"]),
                      case(kind="authority_not_in_corpus", targets=[])):
            with self.subTest(kind=value["expectation"]["match_type"]), self.assertRaises(ValueError):
                evaluation.validate_cases(fixture([value]))

    def test_rank_recall_and_mrr_use_every_single_target_case(self):
        rows = [evaluation.score_case(case("a"), result(["1910.22"])),
                evaluation.score_case(case("b"), result(["1910.303", "1910.22"])),
                evaluation.score_case(case("c"), result(["1910.303"]))]
        metrics = evaluation.summarize(rows)["single_authority"]
        self.assertEqual(metrics["recall_at_1"]["hits"], 1)
        self.assertEqual(metrics["recall_at_3"]["hits"], 2)
        self.assertAlmostEqual(metrics["mrr_at_16"]["mean"], .5)

    def test_group_means_give_equal_weight_to_unequal_variant_counts(self):
        rows = [evaluation.score_case(case(str(i), "many"), result()) for i in range(3)]
        rows.append(evaluation.score_case(case("miss", "single"), result([])))
        metric = evaluation.summarize(rows)["single_authority"]["recall_at_1"]
        self.assertEqual(metric["rate"], .75)
        self.assertEqual(metric["group_mean"], .5)
        self.assertEqual(metric["groups"], 2)

    def test_multi_target_scores_require_all_targets_and_show_partial_coverage(self):
        item = case(kind="multiple_authorities", targets=["1910.22", "1910.303"])
        row = evaluation.score_case(item, result(["1910.22", "1910.178", "1910.184", "1910.303"]))
        metrics = evaluation.summarize([row])["multiple_authorities"]
        self.assertEqual(metrics["all_targets_at_3"]["hits"], 0)
        self.assertEqual(metrics["all_targets_at_5"]["hits"], 1)
        self.assertEqual(metrics["mean_target_recall_at_3"]["mean"], .5)

    def test_missing_corpus_and_unrelated_cases_do_not_inflate_target_misses(self):
        rows = [evaluation.score_case(case("gap", kind="authority_not_in_corpus", targets=["1910.212"]),
                                      result(["1910.179"], lead="1910.179")),
                evaluation.score_case(case("other", kind="no_safety_authority"), result([]))]
        metrics = evaluation.summarize(rows)
        self.assertEqual(metrics["single_authority"]["recall_at_16"]["cases"], 0)
        self.assertIsNone(metrics["single_authority"]["mrr_at_16"]["mean"])
        self.assertEqual(metrics["boundary"]["false_tentative_lead"]["hits"], 1)
        self.assertEqual(metrics["boundary"]["false_tentative_lead"]["cases"], 2)

    def test_tentative_match_is_separate_from_candidate_recall_and_legal_acceptance(self):
        rows = [evaluation.score_case(case("target"), result(lead="1910.22")),
                evaluation.score_case(case("wrong"), result(["1910.303", "1910.22"], lead="1910.303")),
                evaluation.score_case(case("abstain"), result())]
        metrics = evaluation.summarize(rows)
        self.assertEqual(metrics["single_authority"]["recall_at_16"]["hits"], 3)
        for outcome in ("reference_target", "other_target", "abstained"):
            self.assertEqual(metrics["tentative_matches"][outcome]["hits"], 1)
        self.assertEqual(metrics["accepted_as_correct_claims"], 0)

    def test_match_not_present_in_candidates_is_not_counted_as_a_target_success(self):
        row = evaluation.score_case(case(), result(["1910.303"], lead="1910.22", final_citation="1910.212"))
        metrics = evaluation.summarize([row])
        self.assertEqual(metrics["tentative_matches"]["reference_target"]["hits"], 0)
        self.assertEqual(metrics["tentative_match_outside_candidates"], 1)
        self.assertEqual(metrics["final_citation_outside_candidates"], 1)

    def test_missing_alignment_contract_is_not_mistaken_for_successful_abstention(self):
        old = result()
        old.pop("citation_alignment")
        row = evaluation.score_case(case(), old)
        self.assertEqual(row["tentative_match_outcome"], "not_available")
        boundary = evaluation.score_case(case(kind="no_safety_authority"), old)
        stats = evaluation.summarize([boundary])["boundary"]
        self.assertEqual(stats["alignment_missing_cases"], 1)
        self.assertEqual(stats["false_tentative_lead"]["cases"], 0)

    def test_additional_boundary_leads_are_measured_even_when_primary_match_abstains(self):
        output = result([])
        output["additional_wording_evidence"] = [{"section": "1910.179", "accepted_as_correct": False}]
        row = evaluation.score_case(case(kind="authority_not_in_corpus", targets=["1910.212"]), output)
        metrics = evaluation.summarize([row])
        self.assertEqual(metrics["boundary"]["false_tentative_lead"]["hits"], 0)
        self.assertEqual(metrics["boundary"]["additional_wording_lead_shown"]["hits"], 1)
        self.assertEqual(metrics["boundary"]["final_citation_emitted"]["hits"], 0)

    def test_distinct_authority_budget_is_validated(self):
        for sections in (["1910.22", "1910.22"], [str(1000 + i) for i in range(17)]):
            with self.subTest(sections=sections), self.assertRaises(ValueError):
                evaluation.score_case(case(), result(sections))

    def test_default_run_is_network_free_and_binds_public_inputs_without_paths(self):
        path = self.write_fixture()
        import showcase
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), \
                patch.object(showcase, "load_corpus", return_value=[{"text": "Public source"}]), \
                patch.object(showcase, "make_investigator", return_value=object()) as make, \
                patch.object(showcase, "investigate", return_value=result(lead="1910.22")):
            report = evaluation.run(path)
        self.assertIsNone(make.call_args.kwargs["semantic_index"])
        self.assertEqual(make.call_args.kwargs["retrieval_profile"], "wording")
        self.assertEqual(report["requested_retrieval_profile"], "wording")
        self.assertEqual(report["embedding_requests"], 0)
        self.assertEqual(report["generation_requests"], 0)
        self.assertEqual(report["cases_sha256"], evaluation.file_hash(path))
        self.assertEqual(set(report["source_sha256_lf"]), set(evaluation.SOURCE_FILES))
        self.assertEqual(set(report["corpus_files_sha256"]), set(evaluation.CORPUS_FILES))
        self.assertNotIn(str(path), json.dumps(report))
        self.assertFalse(report["independent_field_performance_claim_allowed"])
        self.assertFalse(report["model_promoted"])

    def test_changed_bound_source_during_run_is_rejected(self):
        path = self.write_fixture()
        import showcase
        with patch.object(showcase, "load_corpus", return_value=[]), \
                patch.object(showcase, "make_investigator", return_value=object()), \
                patch.object(showcase, "investigate", return_value=result()), \
                patch.object(evaluation, "input_snapshot", side_effect=[{"revision": 1}, {"revision": 2}]):
            with self.assertRaisesRegex(ValueError, "changed during the run"):
                evaluation.run(path)

    def test_semantic_use_requires_explicit_index_and_digest_before_loading_inputs(self):
        for kwargs in ({"semantic_index_path": self.root / "absent.npz"}, {"model_digest": "a" * 64}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "both an explicit index"):
                evaluation.run(self.root / "absent.json", **kwargs)

    def test_semantic_index_cannot_change_the_reference_profile(self):
        with self.assertRaisesRegex(ValueError, "explicit wording experiment"):
            evaluation.run(self.root / "absent.json", retrieval_profile="reference",
                           semantic_index_path=self.root / "index.npz", model_digest="a" * 64)

    def test_saved_replays_can_be_scored_offline_without_retrieval(self):
        row = dict(case(), result=result(lead="1910.22"))
        row.pop("prompt_style")
        with patch.object(socket.socket, "connect", side_effect=AssertionError("No requests")):
            report = evaluation.aggregate([row])
        self.assertEqual(report["metrics"]["single_authority"]["recall_at_1"]["hits"], 1)
        self.assertEqual(report["records"][0]["prompt_style"], "not_recorded")
        self.assertEqual(set(report["group_metrics"]), {"group"})

    def test_public_fixture_hash_and_freeze_are_bound_without_reading_other_cases(self):
        path = self.write_fixture()
        inputs = evaluation.input_snapshot()
        protocol = {"development": {"sha256": evaluation.file_hash_lf(path)},
                    "frozen_comparison": {"file": "must-not-read.json", "sha256": "a" * 64}}
        (self.root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        freeze = {"source_sha256_lf": inputs["source_sha256_lf"]}
        (self.root / "candidate_freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
        metadata = evaluation.fixture_provenance(path, fixture(), inputs)
        self.assertTrue(metadata["fixture_matches_protocol"])
        self.assertTrue(metadata["candidate_sources_match_freeze"])
        self.assertNotIn(str(self.root), json.dumps(metadata))
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        evaluation.fixture_provenance(path, fixture(), inputs)
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "public protocol"):
            evaluation.fixture_provenance(path, fixture(), inputs)

    def test_changed_or_partially_unknown_freeze_does_not_claim_an_exact_match(self):
        path = self.write_fixture()
        inputs = evaluation.input_snapshot()
        name = evaluation.SOURCE_FILES[0]
        freeze = {"source_sha256_lf": {name: "0" * 64}}
        output = self.root / "candidate_freeze.json"
        output.write_text(json.dumps(freeze), encoding="utf-8")
        self.assertFalse(evaluation.fixture_provenance(path, fixture(), inputs)["candidate_sources_match_freeze"])
        freeze["source_sha256_lf"]["not-covered.py"] = "1" * 64
        output.write_text(json.dumps(freeze), encoding="utf-8")
        metadata = evaluation.fixture_provenance(path, fixture(), inputs)
        self.assertFalse(metadata["freeze_source_check_complete"])
        self.assertIsNone(metadata["candidate_sources_match_freeze"])

    def test_results_are_exclusive_and_invalid_json_never_creates_an_artifact(self):
        path = self.root / "result.json"
        evaluation.save_result(path, {"value": 1})
        with self.assertRaises(FileExistsError):
            evaluation.save_result(path, {"value": 2})
        self.assertEqual(json.loads(path.read_text()), {"value": 1})
        invalid = self.root / "invalid.json"
        with self.assertRaises(ValueError):
            evaluation.save_result(invalid, {"value": float("nan")})
        self.assertFalse(invalid.exists())

    def test_cli_requires_explicit_files_and_stops_before_run_for_existing_output(self):
        cases = self.write_fixture()
        output = self.root / "saved.json"
        output.write_text("old result", encoding="utf-8")
        with patch.object(evaluation, "run") as run:
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluation.main(["--cases", str(cases), "--out", str(output)])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
