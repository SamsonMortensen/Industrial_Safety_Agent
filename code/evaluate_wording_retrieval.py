"""Replay explicitly selected authored prompts and save inspectable results.

These fixtures are invented development references, including the frozen
comparison split. They are not independent field evaluation or training labels.
Only prompt text enters the investigator; relevance references score its output.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MAX_CASE_BYTES = 1024 * 1024
MAX_CASES = 256
CUTOFFS = (1, 3, 5, 16)
SOURCE_FILES = (
    "code/evaluate_wording_retrieval.py", "code/showcase.py",
    "code/autonomous_investigator.py", "code/hybrid_policy.py",
    "code/investigation_support.py", "code/wording_retrieval.py",
    "code/evidence_embeddings.py", "code/citation_alignment.py",
    "code/learning_policy.py", "code/ollama_config.py",
)
CORPUS_FILES = ("json/regulations.json", "json/statutes.json")
TARGET_TYPES = {"single_authority", "multiple_authorities"}
BOUNDARY_TYPES = {"authority_not_in_corpus", "no_safety_authority", "insufficient_facts"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hash_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_cases(payload: dict) -> None:
    """Accept this authored schema only; protected evaluation formats do not fit."""
    if not isinstance(payload, dict) or payload.get("schema") != "grouped_wording_retrieval_cases_v1":
        raise ValueError("Only the grouped authored wording fixture is supported")
    if payload.get("split") not in {"development", "comparison", "frozen_comparison"}:
        raise ValueError("Choose the authored development or frozen comparison split")
    if (payload.get("query_input_field") != "prompt" or
            payload.get("labels_must_not_form_queries") is not True or
            payload.get("existing_independent_evaluation_used") is not False or
            payload.get("calibration_reserve_used") is not False):
        raise ValueError("The fixture must keep references out of queries and protected data separate")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError("Provide between one and 256 authored cases")
    identifiers, groups = set(), set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each authored case must be an object")
        for field in ("case_id", "group_id", "family", "prompt_style", "fact_state"):
            if not isinstance(case.get(field), str) or not case[field].strip() or len(case[field]) > 100:
                raise ValueError("Case metadata is missing or outside its length budget: " + field)
        if case["case_id"] in identifiers:
            raise ValueError("Case identifiers must be unique")
        identifiers.add(case["case_id"])
        groups.add(case["group_id"])
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip() or len(case["prompt"]) > 4000:
            raise ValueError("Each authored prompt needs 1 to 4,000 characters")
        expected = case.get("expectation")
        if not isinstance(expected, dict) or expected.get("match_type") not in TARGET_TYPES | BOUNDARY_TYPES:
            raise ValueError("Unknown authored relevance expectation")
        sections = expected.get("expected_sections")
        if (not isinstance(sections, list) or len(sections) > 8 or
                any(not isinstance(section, str) or not re.fullmatch(r"\d{3,5}(?:\.\d+)?(?:\([A-Za-z0-9]+\))*", section)
                    for section in sections) or len(set(sections)) != len(sections)):
            raise ValueError("Expected sections must be a bounded list of distinct identifiers")
        kind = expected["match_type"]
        if (kind == "single_authority" and len(sections) != 1 or
                kind == "multiple_authorities" and len(sections) < 2 or
                kind == "authority_not_in_corpus" and not sections or
                kind in {"no_safety_authority", "insufficient_facts"} and sections):
            raise ValueError("Expected section count does not match the reference type")
        label_status = expected.get("label_status")
        if (expected.get("violation_label") is not None or not isinstance(label_status, str)
                or not label_status.startswith("authored_unreviewed_") or not label_status.endswith("_reference")):
            raise ValueError("Authored relevance references cannot become violation or approved training labels")
    if type(payload.get("groups")) is not int or payload["groups"] != len(groups):
        raise ValueError("The declared number of scenario groups does not match the cases")


def load_cases(path: Path) -> tuple[dict, str]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_CASE_BYTES + 1)
    if len(raw) > MAX_CASE_BYTES:
        raise ValueError("The authored fixture exceeds one MiB")
    payload = json.loads(raw.decode("utf-8-sig"))
    validate_cases(payload)
    return payload, hashlib.sha256(raw).hexdigest()


def score_case(case: dict, result: dict) -> dict:
    """Read references only after the prompt-only investigation has completed."""
    sections = [item["section"] for item in result["candidates"]]
    if len(sections) > 16 or len(sections) != len(set(sections)):
        raise ValueError("The investigator exceeded the distinct-authority budget")
    expected = case["expectation"]
    ranks = {section: sections.index(section) + 1 if section in sections else None
             for section in expected["expected_sections"]}
    alignment = result.get("citation_alignment")
    alignment_available = isinstance(alignment, dict)
    selected = alignment.get("highest_ranked_research_match") if alignment_available else None
    selected_section = selected.get("section") if isinstance(selected, dict) else None
    valid_selection = selected_section is not None and selected_section in sections
    row = {key: case[key] for key in ("case_id", "group_id", "family", "prompt", "prompt_style", "fact_state", "expectation")}
    row.update(ranks=ranks, result=result, alignment_available=alignment_available,
               tentative_match_section=selected_section,
               tentative_match_in_candidates=valid_selection,
               tentative_match_outcome=("not_available" if not alignment_available else
                                        "abstained" if selected_section is None else
                                        "reference_target" if valid_selection and selected_section in ranks else "other_target"),
               final_citation_outside_candidates=bool(result.get("citation") and result["citation"] not in sections))
    return row


def evaluate_cases(payload: dict, run_prompt) -> list[dict]:
    validate_cases(payload)
    records = []
    for case in payload["cases"]:
        # Do not pass a case object, fact-state label, group, family or expected
        # section to the planner. Questions and denials retain their exact text.
        result = run_prompt(case["prompt"])
        records.append(score_case(case, result))
    return records


def _stat(rows, values, *, count=False):
    groups = defaultdict(list)
    for row, value in zip(rows, values):
        groups[row["group_id"]].append(value)
    result = {"cases": len(rows), "groups": len(groups),
              "group_mean": sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else None}
    if count:
        result.update(hits=sum(values), rate=sum(values) / len(values) if values else None)
    else:
        result["mean"] = sum(values) / len(values) if values else None
    return result


def summarize(records):
    single = [r for r in records if r["expectation"]["match_type"] == "single_authority"]
    multiple = [r for r in records if r["expectation"]["match_type"] == "multiple_authorities"]
    targets = [r for r in records if r["expectation"]["match_type"] in TARGET_TYPES]
    boundary = [r for r in records if r["expectation"]["match_type"] in BOUNDARY_TYPES]
    metrics = {"cases": len(records), "scenario_groups": len({r["group_id"] for r in records}),
               "single_authority": {}, "multiple_authorities": {}, "tentative_matches": {},
               "boundary": {}, "boundary_by_type": {}}
    for cutoff in CUTOFFS:
        hits = [int(any(rank is not None and rank <= cutoff for rank in r["ranks"].values())) for r in single]
        metrics["single_authority"][f"recall_at_{cutoff}"] = _stat(single, hits, count=True)
        if cutoff != 1:
            all_hits = [int(all(rank is not None and rank <= cutoff for rank in r["ranks"].values())) for r in multiple]
            recall = [sum(rank is not None and rank <= cutoff for rank in r["ranks"].values()) / len(r["ranks"])
                      for r in multiple]
            metrics["multiple_authorities"][f"all_targets_at_{cutoff}"] = _stat(multiple, all_hits, count=True)
            metrics["multiple_authorities"][f"mean_target_recall_at_{cutoff}"] = _stat(multiple, recall)
    reciprocal = [max((1 / rank for rank in r["ranks"].values() if rank is not None and rank <= 16), default=0)
                  for r in single]
    metrics["single_authority"]["mrr_at_16"] = _stat(single, reciprocal)
    for outcome in ("reference_target", "other_target", "abstained", "not_available"):
        metrics["tentative_matches"][outcome] = _stat(
            targets, [int(r["tentative_match_outcome"] == outcome) for r in targets], count=True)
    for label, cohort in [("boundary", boundary), *[(kind, [r for r in boundary if r["expectation"]["match_type"] == kind])
                                                   for kind in sorted(BOUNDARY_TYPES)]]:
        aligned = [r for r in cohort if r["alignment_available"]]
        stats = {"alignment_missing_cases": len(cohort) - len(aligned),
                 "false_tentative_lead": _stat(aligned, [int(r["tentative_match_section"] is not None) for r in aligned], count=True),
                 "additional_wording_lead_shown": _stat(cohort, [int(bool(r["result"].get("additional_wording_evidence"))) for r in cohort], count=True),
                 "final_citation_emitted": _stat(cohort, [int(r["result"].get("citation") is not None) for r in cohort], count=True)}
        if label == "boundary":
            metrics[label] = stats
        else:
            metrics["boundary_by_type"][label] = stats
    metrics["final_citation_outside_candidates"] = sum(r["final_citation_outside_candidates"] for r in records)
    metrics["tentative_match_outside_candidates"] = sum(r["tentative_match_section"] is not None and
                                                         not r["tentative_match_in_candidates"] for r in records)
    metrics["accepted_as_correct_claims"] = sum(
        r["alignment_available"] and r["result"]["citation_alignment"].get("accepted_as_correct") is True
        for r in records)
    metrics["scope"] = "Counts against authored relevance references. Tentative matches are unverified research leads, not accepted legal accuracy. Group means give each invented scenario group equal weight."
    return metrics


def aggregate(records):
    """Score existing replay records offline; no retrieval or model calls."""
    scored = [score_case(dict(row, prompt_style=row.get("prompt_style", "not_recorded")), row["result"])
              for row in records]
    groups = sorted({row["group_id"] for row in scored})
    return {"metrics": summarize(scored),
            "group_metrics": {group: summarize([r for r in scored if r["group_id"] == group]) for group in groups},
            "records": scored}


def input_snapshot():
    return {"source_sha256_lf": {name: file_hash_lf(ROOT / name) for name in SOURCE_FILES},
            "corpus_files_sha256": {name: file_hash(ROOT / name) for name in CORPUS_FILES}}


def fixture_provenance(cases_path, payload, inputs):
    """Bind public sidecars when present, without inspecting the other split."""
    provenance = {"cases_sha256_lf": file_hash_lf(cases_path),
                  "protocol_sha256_lf": None, "fixture_matches_protocol": None,
                  "candidate_freeze_sha256_lf": None, "candidate_sources_match_freeze": None,
                  "freeze_sources_checked": 0, "freeze_source_check_complete": False}
    protocol_path = cases_path.parent / "protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
        protocol_key = "frozen_comparison" if payload["split"] == "comparison" else payload["split"]
        expected = protocol.get(protocol_key, {}).get("sha256")
        if expected != provenance["cases_sha256_lf"]:
            raise ValueError("The selected authored fixture does not match its public protocol")
        provenance.update(protocol_sha256_lf=file_hash_lf(protocol_path), fixture_matches_protocol=True)
    freeze_path = cases_path.parent / "candidate_freeze.json"
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
        frozen = freeze.get("source_sha256_lf", {})
        if not isinstance(frozen, dict):
            raise ValueError("The candidate freeze source inventory is invalid")
        current = dict(inputs["source_sha256_lf"], **{name: file_hash_lf(ROOT / name) for name in CORPUS_FILES})
        known = set(frozen) & set(current)
        complete = bool(frozen) and known == set(frozen)
        provenance.update(candidate_freeze_sha256_lf=file_hash_lf(freeze_path),
                          candidate_sources_match_freeze=all(current[name] == frozen[name] for name in known) if complete else None,
                          freeze_sources_checked=len(known), freeze_source_check_complete=complete)
    return provenance


def run(cases_path: Path, *, retrieval_profile="wording", semantic_index_path: Path | None = None,
        model_digest: str | None = None, base_url="http://127.0.0.1:11434") -> dict:
    if retrieval_profile not in {"reference", "wording"}:
        raise ValueError("Choose the reference or experimental wording retrieval profile")
    if semantic_index_path and retrieval_profile != "wording":
        raise ValueError("Semantic retrieval belongs to the explicit wording experiment")
    if bool(semantic_index_path) != bool(model_digest):
        raise ValueError("Semantic use requires both an explicit index and its frozen model digest")
    payload, cases_sha256 = load_cases(cases_path)
    inputs = input_snapshot()
    provenance = fixture_provenance(cases_path, payload, inputs)
    index_sha256 = file_hash(semantic_index_path) if semantic_index_path else None
    import showcase
    chunks = showcase.load_corpus(ROOT)
    semantic = None
    if semantic_index_path:
        from evidence_embeddings import FrozenSemanticIndex, OllamaEmbedder
        budget = min(4096, len(payload["cases"]) * 16)
        client = OllamaEmbedder(model_digest=model_digest, base_url=base_url, max_embed_requests=budget)
        semantic = FrozenSemanticIndex.load(semantic_index_path, chunks, client, max_queries=budget)
    investigator = showcase.make_investigator(chunks, retrieval_profile=retrieval_profile, semantic_index=semantic)
    started = time.perf_counter()
    records = evaluate_cases(payload, lambda prompt: showcase.investigate(
        showcase.text_observation(prompt), investigator, chunks))
    if (input_snapshot() != inputs or file_hash(cases_path) != cases_sha256 or
            fixture_provenance(cases_path, payload, inputs) != provenance or
            (semantic_index_path and file_hash(semantic_index_path) != index_sha256)):
        raise ValueError("A bound input changed during the run; no stable comparison can be saved")
    return {
        "schema": "authored_wording_retrieval_evaluation_v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(), "split": payload["split"],
        "requested_retrieval_profile": retrieval_profile,
        "cases_sha256": cases_sha256,
        **inputs, **provenance, "semantic_index_sha256": index_sha256,
        "semantic_profile": semantic.profile if semantic else None,
        "generation_requests": 0,
        "embedding_requests": semantic.embedder.embedding_requests if semantic else 0,
        "model_verification_requests": semantic.embedder.verification_requests if semantic else 0,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        **aggregate(records), "query_input_field": "prompt", "labels_used_to_form_queries": False,
        "independent_evaluation_used": False, "calibration_reserve_used": False,
        "training_performed": False, "model_promoted": False,
        "independent_field_performance_claim_allowed": False,
        "label_scope": "Invented, authored and unreviewed relevance references. The frozen comparison contains invented scenario groups, not independent evaluation data or reviewer-approved training labels.",
    }


def save_result(path: Path, result: dict) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("Choose an explicit .json result path")
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--retrieval-profile", choices=("reference", "wording"), default="wording",
                        help="This evaluator explicitly opts into the wording experiment by default")
    parser.add_argument("--semantic-index", type=Path)
    parser.add_argument("--model-digest")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError("Output already exists; choose a new run filename")
    if args.out.suffix.lower() != ".json" or not args.out.parent.is_dir():
        raise ValueError("Choose a .json output path in an existing directory")
    result = run(args.cases, retrieval_profile=args.retrieval_profile, semantic_index_path=args.semantic_index,
                 model_digest=args.model_digest, base_url=args.ollama_url)
    save_result(args.out, result)
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
