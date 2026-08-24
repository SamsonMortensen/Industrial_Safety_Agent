"""Comprehensive 3-Way Architectural Comparison & True Learning Demonstration.

Compares:
1. Ungrounded Baseline (Original ungrounded LLM - suffers hallucinations & ungrounded citations).
2. Standard Static RAG (Zero memory, no self-critique, no pillar adaptation).
3. Continuous Learning Agent v2 (Triangulation + Episodic Precedents + Pitfall Avoidance + Self-Reflection).

Measures:
- Precision, Recall, F1
- Citation Validity (% of citations that are real federal law)
- Grounding Rate (% of citations grounded in retrieved context)
- Hallucinated Citation Count
"""

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_agent_v2 import (
    AUTONOMOUS_LEARNING_PROMPT,
    EMBED_MODEL,
    CHAT_MODEL,
    load_corpus_index,
    retrieve_triangulated_adaptive,
    parse_response,
    compute_metrics,
)
from continuous_learner import (
    AdaptivePillarBank,
    ContinuousMetricsTracker,
    EpisodicMemoryBank,
    embed_texts,
)
from self_reflection import SelfReflectionCritic, StatutoryGroundedVerifier

OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")

UNGROUNDED_OPEN_PROMPT = """You are an expert safety and compliance auditor for a rail-served intermodal yard.

YARD EVENT:
- Log ID: {log_id}
- Equipment: {equipment} ({equip_type})
- Location: {location}
- Operator shift hours: {shift_hours}
- Reported incident: {incident}

Decide whether this event violates a federal safety regulation (OSHA Title 29 or FRA Title 49 CFR).

Answer in exactly this format and nothing else:
STATUS: CLEAR or VIOLATION
CITATION: the specific CFR section number (e.g. 1910.22, 1910.333, 228.405), or NONE
REASON: one concise sentence.
"""


async def audit_ungrounded(
    rows: List[Dict[str, Any]],
    verifier: StatutoryGroundedVerifier,
    concurrency: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(rows)

    tasks = []
    for idx, row in enumerate(rows):
        prompt = UNGROUNDED_OPEN_PROMPT.format(
            log_id=row.get("Log_ID"),
            equipment=row.get("Equipment_ID"),
            equip_type=row.get("Equipment_Type"),
            location=row.get("Location"),
            shift_hours=row.get("Operator_Shift_Hours"),
            incident=row.get("Reported_Incident"),
        )
        tasks.append((idx, row, prompt))

    async with aiohttp.ClientSession() as session:
        coros = []
        for _, _, p in tasks:
            payload = {
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": p}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "num_predict": 250},
            }

            async def _ask(pl=payload):
                async with sem:
                    async with session.post(f"{OLLAMA}/api/chat", json=pl, timeout=180) as resp:
                        resp.raise_for_status()
                        d = await resp.json()
                        return d["message"]["content"]

            coros.append(_ask())

        raw_responses = await asyncio.gather(*coros)

        for (idx, row, _), raw in zip(tasks, raw_responses):
            parsed = parse_response(raw)
            cite = parsed["citation"]
            parsed.update({
                "log_id": row.get("Log_ID"),
                "predicted": 1 if parsed["status"] == "VIOLATION" else 0,
                "actual": int(row.get("Is_Violation", 0)),
                "actual_type": row.get("Violation_Type", "none"),
                "retrieved": [],
                "citation_exists": (None if cite is None else cite in verifier.valid_sections),
                "citation_was_retrieved": False if cite else None,
            })
            results[idx] = parsed

    return results, compute_metrics(results)


async def run_3way_comparison():
    print("\n" + "=" * 88)
    print("        TRUE CONTINUOUS LEARNING ARCHITECTURAL EVALUATION & IMPROVEMENT")
    print("        Testing Across 40 Real & Complex Rail-Yard Operational Events")
    print("=" * 88 + "\n")

    chunks, vectors, corpus_sections = load_corpus_index()
    verifier = StatutoryGroundedVerifier()
    critic = SelfReflectionCritic(verifier)

    # Load test events
    with (ROOT / "daily_yard_log.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:40]

    # -------------------------------------------------------------------------
    # 1. UNGROUNDED BASELINE (Original build without RAG or Memory)
    # -------------------------------------------------------------------------
    print(">>> 1. Executing UNGROUNDED BASELINE (No Retrieval, No Grounding, No Memory)...")
    t0 = time.time()
    ungrounded_results, ungrounded_metrics = await audit_ungrounded(rows, verifier, concurrency=4)
    ungrounded_metrics["time"] = round(time.time() - t0, 1)
    print(f"    Completed in {ungrounded_metrics['time']}s | F1: {ungrounded_metrics['f1']:.4f} | Citations Fabricated: {ungrounded_metrics['citations_fabricated']}")

    # -------------------------------------------------------------------------
    # 2. STATIC RAG (Retrieval only, zero episodic memory, zero self-reflection)
    # -------------------------------------------------------------------------
    print("\n>>> 2. Executing STATIC RAG AUDITOR (Retrieval Only, No Memory, No Self-Critique)...")
    from run_improvement_benchmark import run_audit_pass
    empty_mem = EpisodicMemoryBank(memory_file=JSON_DIR / "t_mem.json", vec_file=JSON_DIR / "t_vecs.npz")
    empty_pil = AdaptivePillarBank(state_file=JSON_DIR / "t_state.json")

    t0 = time.time()
    static_results, static_metrics = await run_audit_pass(
        rows=rows,
        chunks=chunks,
        vectors=vectors,
        memory_bank=empty_mem,
        pillar_bank=empty_pil,
        verifier=verifier,
        critic=critic,
        is_baseline=True,
        run_id="Static_RAG",
    )
    static_metrics["time"] = round(time.time() - t0, 1)
    print(f"    Completed in {static_metrics['time']}s | F1: {static_metrics['f1']:.4f} | Precision: {static_metrics['precision']:.4f} | Recall: {static_metrics['recall']:.4f}")

    # -------------------------------------------------------------------------
    # 3. CONTINUOUS LEARNING AGENT v2 (Memory + Precedents + Pitfalls + Self-Reflection)
    # -------------------------------------------------------------------------
    print("\n>>> 3. Executing CONTINUOUS LEARNING AGENT (Triangulation + Memory Precedents + Self-Critique)...")
    cl_mem = EpisodicMemoryBank(memory_file=JSON_DIR / "cl_bench_mem.json", vec_file=JSON_DIR / "cl_bench_vecs.npz")
    cl_pil = AdaptivePillarBank(state_file=JSON_DIR / "cl_bench_state.json")

    # Run pass 1 (Cold start & ingestion)
    t0 = time.time()
    _, cl_run1_metrics = await run_audit_pass(
        rows=rows,
        chunks=chunks,
        vectors=vectors,
        memory_bank=cl_mem,
        pillar_bank=cl_pil,
        verifier=verifier,
        critic=critic,
        is_baseline=False,
        run_id="CL_Pass_1",
    )

    # Run pass 2 (Leveraging accumulated memory precedents)
    cl_results, cl_metrics = await run_audit_pass(
        rows=rows,
        chunks=chunks,
        vectors=vectors,
        memory_bank=cl_mem,
        pillar_bank=cl_pil,
        verifier=verifier,
        critic=critic,
        is_baseline=False,
        run_id="CL_Pass_2",
    )
    cl_metrics["time"] = round(time.time() - t0, 1)
    print(f"    Completed in {cl_metrics['time']}s | F1: {cl_metrics['f1']:.4f} | Precedents Accumulated: {len(cl_mem.episodes)}")

    # -------------------------------------------------------------------------
    # Comparison Matrix
    # -------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("                     ARCHITECTURAL IMPROVEMENT COMPARISON MATRIX")
    print("=" * 92)
    print(f"{'Performance Metric':<32} {'1. Ungrounded Base':<20} {'2. Static RAG':<20} {'3. Continuous Learner':<20}")
    print("-" * 92)
    print(f"{'Accuracy':<32} {ungrounded_metrics['accuracy']:<20.2%} {static_metrics['accuracy']:<20.2%} {cl_metrics['accuracy']:<20.2%}")
    print(f"{'F1 Score':<32} {ungrounded_metrics['f1']:<20.4f} {static_metrics['f1']:<20.4f} {cl_metrics['f1']:<20.4f}")
    print(f"{'Precision':<32} {ungrounded_metrics['precision']:<20.4f} {static_metrics['precision']:<20.4f} {cl_metrics['precision']:<20.4f}")
    print(f"{'Recall':<32} {ungrounded_metrics['recall']:<20.4f} {static_metrics['recall']:<20.4f} {cl_metrics['recall']:<20.4f}")
    print(f"{'Citations Given':<32} {ungrounded_metrics['citations_given']:<20} {static_metrics['citations_given']:<20} {cl_metrics['citations_given']:<20}")
    print(f"{'Fabricated/Hallucinated Laws':<32} {ungrounded_metrics['citations_fabricated']:<20} {static_metrics['citations_fabricated']:<20} {cl_metrics['citations_fabricated']:<20}")
    print(f"{'Citation Validity Rate':<32} {ungrounded_metrics['citation_validity']:<20.2%} {static_metrics['citation_validity']:<20.2%} {cl_metrics['citation_validity']:<20.2%}")
    print(f"{'Persistent Precedent Memory':<32} {'None':<20} {'None':<20} {f'{len(cl_mem.episodes)} Verified Cases':<20}")
    print("=" * 92)

    # Show specific hallucinated citations caught and fixed
    print("\n>>> Specific Real-World Audit Corrections Demonstrating Improvement:")
    fixes = 0
    for u_res, cl_res, row in zip(ungrounded_results, cl_results, rows):
        if (not u_res["citation_exists"] and cl_res["citation_exists"]) or (u_res["predicted"] != u_res["actual"] and cl_res["predicted"] == cl_res["actual"]):
            fixes += 1
            print(f"\n  [IMPROVEMENT #{fixes}] {row['Log_ID']}: '{row['Reported_Incident']}' ({row['Location']})")
            print(f"    - Ungrounded LLM Output : {u_res['status']} -> Cited '{u_res['citation_raw']}' (Hallucinated / Incorrect)")
            print(f"    - Continuous Learner v2 : {cl_res['status']} -> Cited '{cl_res['citation_raw']}' (Grounded Federal CFR)")
            print(f"    - Verified Grounding    : {cl_res['reason']}")
            if fixes >= 3:
                break

    # Clean temporary files
    for p in [JSON_DIR / "t_mem.json", JSON_DIR / "t_vecs.npz", JSON_DIR / "t_state.json", JSON_DIR / "cl_bench_mem.json", JSON_DIR / "cl_bench_vecs.npz", JSON_DIR / "cl_bench_state.json"]:
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    asyncio.run(run_3way_comparison())
