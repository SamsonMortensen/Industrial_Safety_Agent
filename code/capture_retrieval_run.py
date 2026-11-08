"""Capture a lexical-only run on the pinned public synthetic development fixture.

No calibration or independent-evaluation bundle is accepted. Labels score ranks
after queries/candidates have been constructed; only observable fields form queries.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DATASET_SHA256 = "a01f4e6b6b9d2faf7783fd0ee9a3876d073c275a3ef09387a979fd901a274301"


def file_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def summarize(records):
    if not records:
        raise ValueError("Cannot summarize an empty cohort")
    n = len(records)
    result = {"cases": n}
    for k in (1, 4, 8, 16):
        count = sum(r["rank"] is not None and r["rank"] <= k for r in records)
        result[f"hits_at_{k}"] = count
        result[f"recall_at_{k}"] = count / n
    result["mean_reciprocal_rank_at_16"] = sum(1 / r["rank"] if r["rank"] else 0 for r in records) / n
    return result


def capture(repo):
    if file_hash(repo / "daily_yard_log.csv") != DATASET_SHA256:
        raise ValueError("Only the pinned public synthetic development fixture is allowed")
    sys.path.insert(0, str(repo / "code"))
    import showcase
    from autonomous_investigator import ApplicabilityReranker, HybridRetriever, QueryPlanner

    with (repo / "daily_yard_log.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["Is_Violation"] == "1"]
    primary = {r["Log_ID"] for r in showcase.benchmark_sample(64, repo)}
    chunks = showcase.load_corpus(repo)
    retriever, planner, reranker = HybridRetriever(chunks), QueryPlanner(), ApplicabilityReranker()
    records = []
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Network is forbidden in this lexical comparison")):
        for row in rows:
            observation = showcase.benchmark_observation(row)
            queries = planner.plan(observation)
            fused = retriever.search(queries, top_k=16)
            ranked = reranker.rerank(observation, fused)
            # Labels are consumed only after query construction and retrieval.
            expected = row["Expected_Section"]
            rank = next((i for i, c in enumerate(ranked, 1) if c.section == expected), None)
            record = {"event_id": row["Log_ID"], "cohort": "primary64" if row["Log_ID"] in primary else "additional80",
                      "incident": row["Reported_Incident"], "hazard": row["Violation_Type"],
                      "expected_section": expected, "rank": rank, "queries": queries,
                      "fused_sections": [c.section for c in fused], "ranked_sections": [c.section for c in ranked]}
            if row["Log_ID"] in primary and rank is None:
                untruncated = retriever.search(queries, top_k=len(chunks))
                record["miss_diagnostics"] = {
                    "full_fused_authority_rank": next((i for i, c in enumerate(untruncated, 1) if c.section == expected), None),
                    "per_query_target_chunk_rank": [
                        next((i for i, index in enumerate(retriever._rank(retriever.lexical.scores(q), len(chunks)), 1)
                              if chunks[index]["section"] == expected), None) for q in queries],
                }
            records.append(record)
    sources = ["daily_yard_log.csv", "json/regulations.json", "json/statutes.json",
               "code/autonomous_investigator.py", "code/showcase.py", "code/hybrid_policy.py"]
    return {"schema": "lexical_retrieval_development_capture_v1", "captured_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {"python": platform.python_version(), "platform": platform.system()},
            "mode": "lexical_with_applicability_reranking", "top_k": 16,
            "input_and_code_sha256_lf": {p: file_hash(repo / p) for p in sources},
            "independent_evaluation_used": False, "calibration_reserve_used": False, "model_requests": 0,
            "selection_note": "Primary 64 cases were used to diagnose/select the change. Remaining 80 are a same-fixture development regression check, not independent data.",
            "metrics": {cohort: summarize([r for r in records if cohort == "all144" or r["cohort"] == cohort])
                        for cohort in ("primary64", "additional80", "all144")}, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT, help="Public checkout whose retrieval code should run; use a separate process per version")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
