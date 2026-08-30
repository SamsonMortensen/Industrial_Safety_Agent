"""Measure multi-query hybrid retrieval on the locked 1,000-event benchmark.

This benchmark uses only observable event fields. Labels are read after
retrieval to score whether the governing authority was surfaced; they are
never included in an observation or query.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code"
JSON_DIR = ROOT / "json"
sys.path.insert(0, str(CODE_DIR))

from autonomous_investigator import (
    ApplicabilityReranker,
    HybridRetriever,
    Observation,
    QueryPlanner,
)
from continuous_learner import embed_texts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_chunks() -> List[Dict[str, Any]]:
    regulations = json.loads(
        (JSON_DIR / "regulations.json").read_text(encoding="utf-8")
    )
    chunks = list(regulations["chunks"])
    statutes_path = JSON_DIR / "statutes.json"
    if statutes_path.exists():
        statutes = json.loads(statutes_path.read_text(encoding="utf-8"))
        chunks.extend(statutes.get("chunks", []))
    return chunks


def observation_from_row(row: Dict[str, str]) -> Observation:
    return Observation.from_mapping(
        {
            "event_id": row["Log_ID"],
            "timestamp": row["Timestamp"],
            "source_id": row["Equipment_ID"],
            "summary": row["Reported_Incident"],
            "equipment": [row["Equipment_Type"]],
            "location": row["Location"],
            "measurements": {
                "operator_shift_hours": row["Operator_Shift_Hours"],
                "payload_weight_lbs": row["Payload_Weight_lbs"],
            },
            "relations": [
                {
                    "subject": row["Equipment_Type"],
                    "predicate": "reported",
                    "object": row["Reported_Incident"],
                    "confidence": 1.0,
                }
            ],
            "perception_confidence": 1.0,
            "changed": True,
        }
    )


def summarize(ranks: List[int | None], cutoffs=(1, 4, 8, 16)) -> Dict[str, Any]:
    present = [rank for rank in ranks if rank is not None]
    result: Dict[str, Any] = {"cases": len(ranks)}
    for cutoff in cutoffs:
        result[f"recall_at_{cutoff}"] = round(
            sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks), 4
        )
    result["median_rank"] = median(present) if present else None
    result["worst_rank"] = max(present) if present else None
    result["not_retrieved"] = len(ranks) - len(present)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "daily_yard_log.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=JSON_DIR / "autonomous_retrieval_benchmark.json",
    )
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lexical-only", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    with dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["Is_Violation"]) == 1]

    chunks = load_chunks()
    corpus_vectors = None
    embedder = None
    if not args.lexical_only:
        cache_path = JSON_DIR / "reg_embeddings.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached["vectors"].shape[0] == len(chunks):
                corpus_vectors = cached["vectors"]
        if corpus_vectors is None:
            corpus_vectors = embed_texts(
                [
                    f"{chunk.get('citation', '')} {chunk.get('heading', '')}. {chunk.get('text', '')}"
                    for chunk in chunks
                ]
            )
            np.savez_compressed(cache_path, vectors=corpus_vectors)
        embedder = embed_texts

    planner = QueryPlanner()
    observations = [observation_from_row(row) for row in rows]
    query_sets = [planner.plan(observation) for observation in observations]
    flat_queries = [query for queries in query_sets for query in queries]
    flat_vectors = embedder(flat_queries) if embedder is not None else None

    retriever = HybridRetriever(chunks, corpus_vectors, embedder=None)
    reranker = ApplicabilityReranker()
    fused_ranks: List[int | None] = []
    reranked_ranks: List[int | None] = []
    records = []
    per_hazard_fused: Dict[str, List[int | None]] = defaultdict(list)
    per_hazard_reranked: Dict[str, List[int | None]] = defaultdict(list)
    cursor = 0

    for row, observation, queries in zip(rows, observations, query_sets):
        query_vectors = None
        if flat_vectors is not None:
            query_vectors = flat_vectors[cursor : cursor + len(queries)]
            cursor += len(queries)
        fused = retriever.search(queries, top_k=args.top_k, query_vectors=query_vectors)
        reranked = reranker.rerank(observation, fused)
        expected = row["Expected_Section"]

        def rank(items):
            return next(
                (
                    index
                    for index, candidate in enumerate(items, start=1)
                    if candidate.section == expected
                ),
                None,
            )

        fused_rank = rank(fused)
        reranked_rank = rank(reranked)
        fused_ranks.append(fused_rank)
        reranked_ranks.append(reranked_rank)
        hazard = row["Violation_Type"]
        per_hazard_fused[hazard].append(fused_rank)
        per_hazard_reranked[hazard].append(reranked_rank)
        records.append(
            {
                "log_id": row["Log_ID"],
                "hazard": hazard,
                "expected_section": expected,
                "queries": queries,
                "fused_rank": fused_rank,
                "reranked_rank": reranked_rank,
                "top_fused": [candidate.section for candidate in fused[:4]],
                "top_reranked": [candidate.section for candidate in reranked[:4]],
            }
        )

    report = {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset.relative_to(ROOT)),
            "dataset_sha256": sha256(dataset),
            "corpus_sha256": sha256(JSON_DIR / "regulations.json"),
            "statutes_sha256": sha256(JSON_DIR / "statutes.json"),
            "retrieval": "lexical BM25"
            if args.lexical_only
            else "BM25 + dense reciprocal-rank fusion",
            "labels_used_to_form_queries": False,
            "top_k": args.top_k,
        },
        "fused": summarize(fused_ranks),
        "applicability_reranked": summarize(reranked_ranks),
        "per_hazard": {
            hazard: {
                "fused": summarize(per_hazard_fused[hazard]),
                "applicability_reranked": summarize(per_hazard_reranked[hazard]),
            }
            for hazard in sorted(per_hazard_fused)
        },
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "fused": report["fused"],
                "applicability_reranked": report["applicability_reranked"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
