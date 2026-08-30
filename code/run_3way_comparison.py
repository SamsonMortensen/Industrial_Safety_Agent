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
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_agent_v2 import (
    CHAT_MODEL,
    NUM_CTX,
    REQUEST_TIMEOUT,
    compute_metrics,
    load_corpus_index,
    parse_response,
)
from continuous_learner import (
    AdaptivePillarBank,
    EpisodicMemoryBank,
)
from self_reflection import SelfReflectionCritic, StatutoryGroundedVerifier

OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LIMIT = 40

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
CITATION: the specific CFR section number, or NONE
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
                "options": {"temperature": 0.0, "num_predict": 250, "num_ctx": NUM_CTX},
            }

            async def _ask(pl=payload):
                async with sem:
                    for attempt in range(1, 4):
                        try:
                            async with session.post(
                                f"{OLLAMA}/api/chat",
                                json=pl,
                                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                            ) as resp:
                                resp.raise_for_status()
                                d = await resp.json()
                                return d["message"]["content"]
                        except (asyncio.TimeoutError, aiohttp.ClientError):
                            if attempt == 3:
                                return ""
                            await asyncio.sleep(2**attempt)
                    return ""

            coros.append(_ask())

        raw_responses = await asyncio.gather(*coros, return_exceptions=True)
        raw_responses = [
            "" if isinstance(r, BaseException) else r for r in raw_responses
        ]

        for (idx, row, _), raw in zip(tasks, raw_responses):
            parsed = parse_response(raw)
            cite = parsed["citation"]
            parsed.update(
                {
                    "log_id": row.get("Log_ID"),
                    "audited_by": "model",
                    "predicted": 1 if parsed["status"] == "VIOLATION" else 0,
                    "actual": int(row.get("Is_Violation", 0)),
                    "actual_type": row.get("Violation_Type", "none"),
                    "expected_section": row.get("Expected_Section"),
                    "retrieved": [],
                    "citation_exists": (
                        None if cite is None else cite in verifier.valid_sections
                    ),
                    "citation_was_retrieved": False if cite else None,
                }
            )
            results[idx] = parsed

    return results, compute_metrics(results)


async def run_3way_comparison():
    print("\n" + "=" * 88)
    print("        TRUE CONTINUOUS LEARNING ARCHITECTURAL EVALUATION & IMPROVEMENT")
    print(f"        Comparing three architectures across {LIMIT} yard events")
    print("=" * 88 + "\n")

    chunks, vectors, corpus_sections = load_corpus_index()
    verifier = StatutoryGroundedVerifier()
    critic = SelfReflectionCritic(verifier)

    # Load test events
    with (ROOT / "daily_yard_log.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:LIMIT]

    # -------------------------------------------------------------------------
    # 1. UNGROUNDED BASELINE (Original build without RAG or Memory)
    # -------------------------------------------------------------------------
    print(
        ">>> 1. Executing UNGROUNDED BASELINE (No Retrieval, No Grounding, No Memory)..."
    )
    t0 = time.time()
    ungrounded_results, ungrounded_metrics = await audit_ungrounded(
        rows, verifier, concurrency=4
    )
    ungrounded_metrics["time"] = round(time.time() - t0, 1)
    print(
        f"    Completed in {ungrounded_metrics['time']}s | F1: {ungrounded_metrics['f1']:.4f} | Citations Fabricated: {ungrounded_metrics['citations_fabricated']}"
    )

    # -------------------------------------------------------------------------
    # 2. STATIC RAG (Retrieval only, zero episodic memory, zero self-reflection)
    # -------------------------------------------------------------------------
    print(
        "\n>>> 2. Executing STATIC RAG AUDITOR (Retrieval Only, No Memory, No Self-Critique)..."
    )
    from run_improvement_benchmark import run_audit_pass

    empty_mem = EpisodicMemoryBank(
        memory_file=JSON_DIR / "t_mem.json", vec_file=JSON_DIR / "t_vecs.npz"
    )
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
    print(
        f"    Completed in {static_metrics['time']}s | F1: {static_metrics['f1']:.4f} | Precision: {static_metrics['precision']:.4f} | Recall: {static_metrics['recall']:.4f}"
    )

    # -------------------------------------------------------------------------
    # 3. CONTINUOUS LEARNING AGENT v2 (Memory + Precedents + Pitfalls + Self-Reflection)
    # -------------------------------------------------------------------------
    print(
        "\n>>> 3. Executing CONTINUOUS LEARNING AGENT (Triangulation + Memory Precedents + Self-Critique)..."
    )
    cl_mem = EpisodicMemoryBank(
        memory_file=JSON_DIR / "cl_bench_mem.json",
        vec_file=JSON_DIR / "cl_bench_vecs.npz",
    )
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
    print(
        f"    Completed in {cl_metrics['time']}s | F1: {cl_metrics['f1']:.4f} | Precedents Accumulated: {len(cl_mem.episodes)}"
    )

    # -------------------------------------------------------------------------
    # Comparison Matrix
    # -------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("                     ARCHITECTURAL IMPROVEMENT COMPARISON MATRIX")
    print("=" * 92)
    print(
        f"{'Performance Metric':<32} {'1. Ungrounded Base':<20} {'2. Static RAG':<20} {'3. Continuous Learner':<20}"
    )
    print("-" * 92)
    print(
        f"{'Accuracy':<32} {ungrounded_metrics['accuracy']:<20.2%} {static_metrics['accuracy']:<20.2%} {cl_metrics['accuracy']:<20.2%}"
    )
    print(
        f"{'F1 Score':<32} {ungrounded_metrics['f1']:<20.4f} {static_metrics['f1']:<20.4f} {cl_metrics['f1']:<20.4f}"
    )
    print(
        f"{'Precision':<32} {ungrounded_metrics['precision']:<20.4f} {static_metrics['precision']:<20.4f} {cl_metrics['precision']:<20.4f}"
    )
    print(
        f"{'Recall':<32} {ungrounded_metrics['recall']:<20.4f} {static_metrics['recall']:<20.4f} {cl_metrics['recall']:<20.4f}"
    )
    print(
        f"{'Citations Given':<32} {ungrounded_metrics['citations_given']:<20} {static_metrics['citations_given']:<20} {cl_metrics['citations_given']:<20}"
    )
    print(
        f"{'Fabricated/Hallucinated Laws':<32} {ungrounded_metrics['citations_fabricated']:<20} {static_metrics['citations_fabricated']:<20} {cl_metrics['citations_fabricated']:<20}"
    )

    def _pct(v):
        return "n/a" if v is None else f"{v:.2%}"

    print(
        f"{'Citation Validity Rate':<32} {_pct(ungrounded_metrics['citation_validity']):<20} "
        f"{_pct(static_metrics['citation_validity']):<20} {_pct(cl_metrics['citation_validity']):<20}"
    )
    print(
        f"{'Persistent Precedent Memory':<32} {'None':<20} {'None':<20} {f'{len(cl_mem.episodes)} Verified Cases':<20}"
    )
    print("=" * 92)

    # Both directions, described from what was measured rather than asserted.
    def _describe(res):
        if not res.get("citation"):
            return "no citation"
        if res.get("citation_exists") is False:
            return "section does not exist"
        if res.get("citation_was_retrieved") is False:
            return "real, but not in retrieved context"
        return "real and retrieved"

    gains, losses = [], []
    for u_res, cl_res, row in zip(ungrounded_results, cl_results, rows):
        u_ok = u_res["predicted"] == u_res["actual"]
        c_ok = cl_res["predicted"] == cl_res["actual"]
        if c_ok and not u_ok:
            gains.append((row, u_res, cl_res))
        elif u_ok and not c_ok:
            losses.append((row, u_res, cl_res))

    print(
        f"\n>>> Verdict changes: {len(gains)} corrected by the learner, "
        f"{len(losses)} regressed."
    )
    for label, bucket in (("CORRECTED", gains), ("REGRESSED", losses)):
        for row, u_res, cl_res in bucket[:3]:
            print(f"\n  [{label}] {row['Log_ID']}: '{row['Reported_Incident'][:70]}'")
            print(
                f"    ungrounded : {u_res['status']:9} cite {str(u_res['citation_raw'])[:18]:18} ({_describe(u_res)})"
            )
            print(
                f"    learner    : {cl_res['status']:9} cite {str(cl_res['citation_raw'])[:18]:18} ({_describe(cl_res)})"
            )
            print(
                f"    truth      : {'VIOLATION' if row['Is_Violation'] == '1' else 'CLEAR':9} "
                f"governing rule {row.get('Expected_Section', 'n/a')}"
            )

    # Clean temporary files
    for p in [
        JSON_DIR / "t_mem.json",
        JSON_DIR / "t_vecs.npz",
        JSON_DIR / "t_state.json",
        JSON_DIR / "cl_bench_mem.json",
        JSON_DIR / "cl_bench_vecs.npz",
        JSON_DIR / "cl_bench_state.json",
    ]:
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Three-way architectural comparison.")
    ap.add_argument("--limit", type=int, default=40, help="events to compare across")
    LIMIT = ap.parse_args().limit
    asyncio.run(run_3way_comparison())
