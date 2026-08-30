"""Stress-test grounded retrieval on deidentified authentic OSHA narratives.

This is a coverage and grounding benchmark. OSHA Severe Injury Reports do not
contain authoritative regulation labels, so this script never reports legal
accuracy, precision, or recall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code"
JSON_DIR = ROOT / "json"
sys.path.insert(0, str(CODE_DIR))

from autonomous_investigator import (
    ApplicabilityReranker,
    HybridRetriever,
    Observation,
    QueryPlanner,
    TriggerEngine,
)
from continuous_learner import embed_texts
from hybrid_policy import evaluate_event
from osha_incident_pipeline import sha256

DEFAULT_INPUT = (
    ROOT / "data" / "external" / "osha_sir" / "normalized_observations.jsonl"
)
DEFAULT_OUTPUT = JSON_DIR / "authentic_incident_retrieval.json"


def load_chunks() -> list[dict[str, Any]]:
    chunks = list(
        json.loads((JSON_DIR / "regulations.json").read_text(encoding="utf-8"))[
            "chunks"
        ]
    )
    statutes_path = JSON_DIR / "statutes.json"
    if statutes_path.exists():
        chunks.extend(
            json.loads(statutes_path.read_text(encoding="utf-8")).get("chunks", [])
        )
    return chunks


def stable_sample(
    observations: list[Observation], size: int, seed: int
) -> list[Observation]:
    ordered = sorted(
        observations,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item.event_id}".encode("utf-8")
        ).digest(),
    )
    return ordered[: min(size, len(ordered))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()

    source_path = args.input.resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        observations = [
            Observation.from_mapping(json.loads(line))
            for line in handle
            if line.strip()
        ]
    observations = stable_sample(observations, args.sample, args.seed)

    chunks = load_chunks()
    corpus_vectors = None
    embedder = None
    if args.dense:
        corpus_vectors = embed_texts(
            [
                f"{chunk.get('citation', '')} {chunk.get('heading', '')}. {chunk.get('text', '')}"
                for chunk in chunks
            ]
        )
        embedder = embed_texts

    planner = QueryPlanner()
    trigger_engine = TriggerEngine(sample_rate=0.0)
    query_sets = [planner.plan(observation) for observation in observations]
    flat_queries = [query for queries in query_sets for query in queries]
    flat_vectors = embedder(flat_queries) if embedder is not None else None
    retriever = HybridRetriever(chunks, corpus_vectors, embedder=None)
    reranker = ApplicabilityReranker()

    cursor = 0
    candidate_counts = []
    query_counts = []
    trigger_counts = Counter()
    decision_counts = Counter()
    top_authorities = Counter()
    cited_decisions = 0
    grounded_citations = 0
    records = []

    for observation, queries in zip(observations, query_sets):
        trigger = trigger_engine.evaluate(observation)
        trigger_counts["triggered" if trigger.triggered else "not_triggered"] += 1
        query_vectors = None
        if flat_vectors is not None:
            query_vectors = flat_vectors[cursor : cursor + len(queries)]
            cursor += len(queries)
        candidates = retriever.search(
            queries, top_k=args.top_k, query_vectors=query_vectors
        )
        candidates = reranker.rerank(observation, candidates)
        candidate_sections = {candidate.section for candidate in candidates}
        policy = evaluate_event(observation.to_legacy_row())
        decision_status = policy.status
        if policy.citation:
            cited_decisions += 1
            if policy.citation in candidate_sections:
                grounded_citations += 1
            else:
                decision_status = "REVIEW_UNGROUNDED_POLICY"
        decision_counts[decision_status] += 1
        query_counts.append(len(queries))
        candidate_counts.append(len(candidates))
        if candidates:
            top_authorities[candidates[0].section] += 1
        records.append(
            {
                "event_id": observation.event_id,
                "queries": len(queries),
                "candidates": len(candidates),
                "top_sections": [candidate.section for candidate in candidates[:4]],
                "policy_status_after_grounding_gate": decision_status,
            }
        )

    total = len(observations)
    report = {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "official_source": "https://www.osha.gov/severe-injury-reports",
            "normalized_input_sha256": sha256(source_path),
            "corpus_sha256": sha256(JSON_DIR / "regulations.json"),
            "statutes_sha256": sha256(JSON_DIR / "statutes.json"),
            "sample_method": "lowest SHA-256(seed:event_id)",
            "sample_seed": args.seed,
            "requested_sample": args.sample,
            "retrieval": "BM25 + dense reciprocal-rank fusion"
            if args.dense
            else "lexical BM25",
            "top_k": args.top_k,
            "oiics_codes_used_to_form_queries": False,
            "outcome_columns_used_to_form_queries": False,
            "regulation_labels_available": False,
        },
        "coverage": {
            "cases": total,
            "triggered": trigger_counts["triggered"],
            "trigger_rate": round(trigger_counts["triggered"] / total, 4)
            if total
            else 0.0,
            "nonempty_retrieval": sum(count > 0 for count in candidate_counts),
            "nonempty_retrieval_rate": round(
                sum(count > 0 for count in candidate_counts) / total, 4
            )
            if total
            else 0.0,
            "median_queries": median(query_counts) if query_counts else None,
            "median_candidates": median(candidate_counts) if candidate_counts else None,
            "unique_top_authorities": len(top_authorities),
        },
        "grounding_gate": {
            "policy_status_counts": dict(sorted(decision_counts.items())),
            "cited_policy_decisions": cited_decisions,
            "citations_present_in_retrieved_candidates": grounded_citations,
            "citation_grounding_rate": round(grounded_citations / cited_decisions, 4)
            if cited_decisions
            else None,
            "unsupported_citations_allowed": 0,
        },
        "top_retrieved_authorities": [
            {"section": section, "cases": count}
            for section, count in top_authorities.most_common(12)
        ],
        "records": records,
        "limitations": [
            "This benchmark measures ingestion, retrieval coverage, and citation grounding, not legal accuracy.",
            "OSHA Severe Injury Reports have no authoritative regulation label for each narrative.",
            "Narratives describe known injuries and may contain post-event outcome information.",
            "The source contains severe outcomes and cannot measure false alarms on ordinary camera-hours.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "coverage": report["coverage"],
                "grounding_gate": report["grounding_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
