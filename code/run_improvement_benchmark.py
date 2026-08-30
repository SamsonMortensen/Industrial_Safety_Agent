"""Rigorous Continuous Learning Improvement Benchmark.

Compares a Static Baseline Auditor against the Continuous Learning Agent
across 3 sequential learning runs on a challenging dataset of 40 hard industrial edge cases,
vocabulary gaps, and deceptive non-violations.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_agent_v2 import (
    AUTONOMOUS_LEARNING_PROMPT,
    CHAT_MODEL,
    compute_metrics,
    load_corpus_index,
    parse_response,
    retrieve_triangulated_adaptive,
)
from continuous_learner import (
    AdaptivePillarBank,
    EpisodicMemoryBank,
    build_query_text,
    embed_texts,
    event_signature,
)
from self_reflection import SelfReflectionCritic, StatutoryGroundedVerifier

OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
BENCHMARK_LOG = ROOT / "challenging_yard_log.csv"


def create_challenging_dataset() -> List[Dict[str, Any]]:
    """Creates a curated, challenging test dataset with subtle phrasing and deceptive negatives."""
    return [
        # --- Housekeeping & Walking-Working Surfaces (1910.22, 1910.26, 1910.28) ---
        {
            "Log_ID": "CH-101",
            "Timestamp": "2026-04-02 06:15",
            "Equipment_ID": "FL-04",
            "Equipment_Type": "Forklift",
            "Location": "Pedestrian Walkway Sector C",
            "Operator_Shift_Hours": "6.0",
            "Reported_Incident": "Slick oily sheen accumulation spread across main personnel walkway",
            "Is_Violation": "1",
            "Violation_Type": "housekeeping",
            "Expected_Citation": "1910.22",
        },
        {
            "Log_ID": "CH-102",
            "Timestamp": "2026-04-02 06:45",
            "Equipment_ID": "RS-02",
            "Equipment_Type": "Reach Stacker",
            "Location": "Maintenance Bay Floor",
            "Operator_Shift_Hours": "5.5",
            "Reported_Incident": "Lubricating grease applied inside enclosed drive axle differential",
            "Is_Violation": "0",
            "Violation_Type": "none",
            "Expected_Citation": "NONE",
        },
        {
            "Log_ID": "CH-103",
            "Timestamp": "2026-04-02 07:10",
            "Equipment_ID": "FL-08",
            "Equipment_Type": "Forklift",
            "Location": "Railcar Loading Dock 3",
            "Operator_Shift_Hours": "7.5",
            "Reported_Incident": "Portable dockboard unsecured and sliding while forklift traversing into boxcar",
            "Is_Violation": "1",
            "Violation_Type": "dockboard",
            "Expected_Citation": "1910.26",
        },
        {
            "Log_ID": "CH-104",
            "Timestamp": "2026-04-02 07:35",
            "Equipment_ID": "GC-01",
            "Equipment_Type": "Gantry Crane",
            "Location": "Track 4 Catwalk",
            "Operator_Shift_Hours": "4.0",
            "Reported_Incident": "Missing standard 42-inch safety guardrail on 8-foot elevated personnel walkway",
            "Is_Violation": "1",
            "Violation_Type": "fall_protection",
            "Expected_Citation": "1910.28",
        },
        {
            "Log_ID": "CH-105",
            "Timestamp": "2026-04-02 08:00",
            "Equipment_ID": "FL-01",
            "Equipment_Type": "Forklift",
            "Location": "Storage Yard Aisle 2",
            "Operator_Shift_Hours": "8.0",
            "Reported_Incident": "Cargo boxes stacked with 3-foot overhang obstructing designated personnel aisleway",
            "Is_Violation": "1",
            "Violation_Type": "aisle_clearance",
            "Expected_Citation": "1910.176",
        },
        {
            "Log_ID": "CH-106",
            "Timestamp": "2026-04-02 08:20",
            "Equipment_ID": "FL-03",
            "Equipment_Type": "Forklift",
            "Location": "Storage Yard Aisle 4",
            "Operator_Shift_Hours": "6.0",
            "Reported_Incident": "Cargo neatly palletized and secured within marked yellow aisle boundaries",
            "Is_Violation": "0",
            "Violation_Type": "none",
            "Expected_Citation": "NONE",
        },
        # --- Crane Rigging & Materials Handling (1910.179, 1910.184) ---
        {
            "Log_ID": "CH-107",
            "Timestamp": "2026-04-02 08:50",
            "Equipment_ID": "GC-03",
            "Equipment_Type": "Gantry Crane",
            "Location": "Intermodal Transfer Pad 1",
            "Operator_Shift_Hours": "5.0",
            "Reported_Incident": "Wire rope rigging sling with 12 broken outer wires and crushed eyelet in active lift",
            "Is_Violation": "1",
            "Violation_Type": "defective_sling",
            "Expected_Citation": "1910.184",
        },
        {
            "Log_ID": "CH-108",
            "Timestamp": "2026-04-02 09:15",
            "Equipment_ID": "GC-02",
            "Equipment_Type": "Gantry Crane",
            "Location": "Intermodal Transfer Pad 2",
            "Operator_Shift_Hours": "4.5",
            "Reported_Incident": "Synthetic web sling inspected with legible manufacturer load rating tag attached",
            "Is_Violation": "0",
            "Violation_Type": "none",
            "Expected_Citation": "NONE",
        },
        # Hours of service for freight train employees under 49 U.S.C. 21103.
        {
            "Log_ID": "CH-109",
            "Timestamp": "2026-04-02 09:40",
            "Equipment_ID": "SW-01",
            "Equipment_Type": "Switch Engine",
            "Location": "Classification Yard Spur",
            "Operator_Shift_Hours": "12.4",
            "Reported_Incident": "Freight train employee remained in covered service because the relief crew arrived late",
            "Is_Violation": "1",
            "Violation_Type": "fatigue",
            "Expected_Citation": "21103",
        },
        {
            "Log_ID": "CH-110",
            "Timestamp": "2026-04-02 10:05",
            "Equipment_ID": "SW-02",
            "Equipment_Type": "Switch Engine",
            "Location": "Classification Yard Spur",
            "Operator_Shift_Hours": "11.85",
            "Reported_Incident": "Switch crew conducting final consist tie-down before shift conclusion",
            "Is_Violation": "0",
            "Violation_Type": "none",
            "Expected_Citation": "NONE",
        },
        {
            "Log_ID": "CH-111",
            "Timestamp": "2026-04-02 10:30",
            "Equipment_ID": "RS-05",
            "Equipment_Type": "Reach Stacker",
            "Location": "Container Berth 6",
            "Operator_Shift_Hours": "13.1",
            "Reported_Incident": "Covered switch employee remained on duty for another freight train movement",
            "Is_Violation": "1",
            "Violation_Type": "fatigue",
            "Expected_Citation": "21103",
        },
        {
            "Log_ID": "CH-112",
            "Timestamp": "2026-04-02 10:55",
            "Equipment_ID": "RS-07",
            "Equipment_Type": "Reach Stacker",
            "Location": "Container Berth 4",
            "Operator_Shift_Hours": "11.95",
            "Reported_Incident": "Operator logging out at electronic timeclock at exactly shift end",
            "Is_Violation": "0",
            "Violation_Type": "none",
            "Expected_Citation": "NONE",
        },
        # --- Electrical Clearance & High Voltage Proximity (29 CFR 1910.333) ---
        {
            "Log_ID": "CH-113",
            "Timestamp": "2026-04-02 11:20",
            "Equipment_ID": "GC-04",
            "Equipment_Type": "Gantry Crane",
            "Location": "Overhead 25kV Electrified Catenary Line",
            "Operator_Shift_Hours": "6.2",
            "Reported_Incident": "Spreader boom operating at 6-foot distance from live 25,000V energized wire",
            "Is_Violation": "1",
            "Violation_Type": "electrical_clearance",
            "Expected_Citation": "1910.333",
        },
        {
            "Log_ID": "CH-114",
            "Timestamp": "2026-04-02 11:45",
            "Equipment_ID": "GC-04",
            "Equipment_Type": "Gantry Crane",
            "Location": "Track 9 Electrified Catenary Line (De-energized)",
            "Operator_Shift_Hours": "5.0",
            "Reported_Incident": "Boom operating under confirmed grounded, lock-tagged de-energized catenary section",
            "Is_Violation": "0",
            "Violation_Type": "none",
            "Expected_Citation": "NONE",
        },
        {
            "Log_ID": "CH-115",
            "Timestamp": "2026-04-02 12:10",
            "Equipment_ID": "FL-05",
            "Equipment_Type": "Forklift",
            "Location": "Main Substation Transformer Yard",
            "Operator_Shift_Hours": "4.0",
            "Reported_Incident": "Unqualified operator lifting mast within 4 feet of open energized 13.8kV busbar",
            "Is_Violation": "1",
            "Violation_Type": "electrical_clearance",
            "Expected_Citation": "1910.333",
        },
        {
            "Log_ID": "CH-116",
            "Timestamp": "2026-04-02 12:35",
            "Equipment_ID": "FL-06",
            "Equipment_Type": "Forklift",
            "Location": "Substation Perimeter Fence (Outside)",
            "Operator_Shift_Hours": "3.5",
            "Reported_Incident": "Forklift transporting wooden pallets outside 20-foot grounded substation perimeter barrier",
            "Is_Violation": "0",
            "Violation_Type": "none",
            "Expected_Citation": "NONE",
        },
        # --- Routine Telemetry & Deceptive Alarms ---
        {
            "Log_ID": "CH-117",
            "Timestamp": "2026-04-02 13:00",
            "Equipment_ID": "RS-01",
            "Equipment_Type": "Reach Stacker",
            "Location": "North Intermodal Spur",
            "Operator_Shift_Hours": "6.0",
            "Reported_Incident": "Routine Load Imbalance sensor alarm stabilized automatically after twistlock engagement",
            "Is_Violation": "0",
            "Violation_Type": "none",
            "Expected_Citation": "NONE",
        },
        {
            "Log_ID": "CH-118",
            "Timestamp": "2026-04-02 13:25",
            "Equipment_ID": "FL-07",
            "Equipment_Type": "Forklift",
            "Location": "South Gate Entrance",
            "Operator_Shift_Hours": "7.0",
            "Reported_Incident": "Routine Tire Pressure Warning indicator illuminated and tire inflated at air station",
            "Is_Violation": "0",
            "Violation_Type": "none",
            "Expected_Citation": "NONE",
        },
        {
            "Log_ID": "CH-119",
            "Timestamp": "2026-04-02 13:50",
            "Equipment_ID": "RS-09",
            "Equipment_Type": "Reach Stacker",
            "Location": "Intermodal Crosswalk Bravo",
            "Operator_Shift_Hours": "8.5",
            "Reported_Incident": "Active hydraulic fluid spraying from burst hose directly onto marked pedestrian walkway",
            "Is_Violation": "1",
            "Violation_Type": "housekeeping",
            "Expected_Citation": "1910.22",
        },
        {
            "Log_ID": "CH-120",
            "Timestamp": "2026-04-02 14:15",
            "Equipment_ID": "RS-10",
            "Equipment_Type": "Reach Stacker",
            "Location": "Equipment Wash Rack Drainage Pit",
            "Operator_Shift_Hours": "4.0",
            "Reported_Incident": "Washdown runoff contained within self-contained underground oil-water separator drain",
            "Is_Violation": "0",
            "Violation_Type": "none",
            "Expected_Citation": "NONE",
        },
    ]


async def run_audit_pass(
    rows: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    vectors: np.ndarray,
    memory_bank: EpisodicMemoryBank,
    pillar_bank: AdaptivePillarBank,
    verifier: StatutoryGroundedVerifier,
    critic: SelfReflectionCritic,
    is_baseline: bool = False,
    run_id: str = "run",
    concurrency: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs an audit pass over the benchmark records."""
    sem = asyncio.Semaphore(concurrency)
    llm_tasks = []
    results = [None] * len(rows)

    # If continuous learning is enabled, scan and adapt pillars
    if not is_baseline:
        for r in rows:
            pass  # pillar expansion is gap-driven now, applied after the pass

    for idx, row in enumerate(rows):
        if is_baseline:
            # Baseline uses flat retrieval without triangulation or continuous learning memory
            query_event = f"{row.get('Equipment_Type')} at {row.get('Location')}, shift {row.get('Operator_Shift_Hours')} hrs, incident: {row.get('Reported_Incident')}"
            q_vec = embed_texts([query_event])[0]
            scores = vectors @ q_vec
            top_k = np.argsort(-scores)[:4]
            hits = [chunks[i] for i in top_k]
        else:
            hits = retrieve_triangulated_adaptive(row, chunks, vectors, pillar_bank)

        context = "\n\n".join(
            f"[{c['citation']} -- {c['heading']}]\n{c['text'][:700]}" for c in hits
        )

        query_text = build_query_text(row)

        precedents_section = ""
        pitfalls_section = ""

        # Only inject memory in continuous learning passes (NOT in baseline)
        if not is_baseline:
            # Run 2 replays the SAME events as Run 1, so without these guards every
            # event retrieves its own Run-1 verdict and the "improvement" is just the
            # model reading back its own answer. Self- and signature-exclusion keep
            # cross-event transfer measurable while removing that shortcut.
            # No temporal bound here: the fixtures share one timeline, so bounding on
            # event time would empty memory entirely. Forward generalisation is what
            # code/temporal_eval.py measures instead.
            contrastive = memory_bank.retrieve_contrastive_precedents(
                query_text,
                exclude_log_id=str(row.get("Log_ID")),
                exclude_signature=event_signature(row),
            )
            lines = []
            if contrastive.get("positive"):
                p = contrastive["positive"]
                lines.append(
                    f"- Confirmed Violation Case: {p['incident']} at {p['location']} (Shift: {p['shift_hours']} hrs) -> "
                    f"STATUS: {p['verdict']}, CITATION: {p['citation']} ({p['reason']})"
                )
            if contrastive.get("negative"):
                n = contrastive["negative"]
                lines.append(
                    f"- Confirmed Compliant Baseline: {n['incident']} at {n['location']} (Shift: {n['shift_hours']} hrs) -> "
                    f"STATUS: {n['verdict']}, CITATION: NONE ({n['reason']})"
                )
            if lines:
                precedents_section = (
                    "VERIFIED AUDIT PRECEDENTS (Historical Case Law):\n"
                    + "\n".join(lines)
                    + "\n"
                )

            pitfalls = memory_bank.retrieve_pitfalls(
                query_text,
                k=1,
                exclude_log_id=str(row.get("Log_ID")),
                exclude_signature=event_signature(row),
            )
            if pitfalls:
                lines = ["LEARNED PITFALL WARNINGS (Avoid Past Errors):"]
                for pit in pitfalls:
                    lines.append(
                        f"- Warning: {pit['past_mistake']} -> Correct: {pit['correct_verdict']} ({pit['correct_citation']})"
                    )
                pitfalls_section = "\n".join(lines) + "\n"

        prompt = AUTONOMOUS_LEARNING_PROMPT.format(
            context=context,
            precedents_section=precedents_section,
            pitfalls_section=pitfalls_section,
            log_id=row.get("Log_ID"),
            equipment=row.get("Equipment_ID"),
            equip_type=row.get("Equipment_Type"),
            location=row.get("Location"),
            shift_hours=row.get("Operator_Shift_Hours"),
            incident=row.get("Reported_Incident"),
        )
        llm_tasks.append((idx, row, hits, prompt))

    critique_corrections_count = 0
    async with aiohttp.ClientSession() as session:
        coros = []
        for _, _, _, p in llm_tasks:
            payload = {
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": p}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "num_predict": 250},
            }

            async def _ask(pl=payload):
                async with sem:
                    async with session.post(
                        f"{OLLAMA}/api/chat", json=pl, timeout=180
                    ) as resp:
                        resp.raise_for_status()
                        d = await resp.json()
                        return d["message"]["content"]

            coros.append(_ask())

        raw_responses = await asyncio.gather(*coros)

        for (idx, row, hits, p), raw in zip(llm_tasks, raw_responses):
            parsed = parse_response(raw)

            # Self-reflection is enabled only for Continuous Learning Agent (not baseline)
            if not is_baseline:
                refined, was_corrected, _ = await critic.critique_and_reground_async(
                    session, p, parsed, row, hits, sem
                )
                if was_corrected:
                    critique_corrections_count += 1
                parsed = refined

            cite = parsed["citation"]
            retrieved_sections = {c["section"] for c in hits}

            parsed.update(
                {
                    "log_id": row.get("Log_ID"),
                    "predicted": 1 if parsed["status"] == "VIOLATION" else 0,
                    "actual": int(row.get("Is_Violation", 0)),
                    "actual_type": row.get("Violation_Type", "none"),
                    "expected_citation": row.get("Expected_Citation", "NONE"),
                    "retrieved": sorted(list(retrieved_sections)),
                    "citation_exists": (
                        None if cite is None else cite in verifier.valid_sections
                    ),
                    "citation_was_retrieved": (
                        None if cite is None else cite in retrieved_sections
                    ),
                }
            )
            results[idx] = parsed

            # Ingest experience into Memory Bank
            if not is_baseline:
                is_grounded = (
                    bool(
                        parsed.get("citation_exists")
                        and parsed.get("citation_was_retrieved")
                    )
                    if cite
                    else True
                )
                feedback_type = "verified" if is_grounded else "pitfall"
                memory_bank.add_episode(
                    log_id=str(row.get("Log_ID")),
                    row=row,
                    verdict=parsed["status"],
                    citation=parsed["citation"],
                    reason=parsed["reason"],
                    is_grounded=is_grounded,
                    confidence=parsed.get("confidence", 1.0),
                    feedback_type=feedback_type,
                    run_id=run_id,
                )

    metrics = compute_metrics(results)
    metrics["critique_corrections"] = critique_corrections_count
    return results, metrics


async def execute_experiment():
    print("\n" + "=" * 80)
    print("      CONTINUOUS LEARNING EMPIRICAL IMPROVEMENT BENCHMARK")
    print("      Evaluating 20 Challenging Industrial Edge Cases & Deceptive Scenarios")
    print("=" * 80 + "\n")

    chunks, vectors, corpus_sections = load_corpus_index()
    dataset = create_challenging_dataset()
    verifier = StatutoryGroundedVerifier()
    critic = SelfReflectionCritic(verifier)

    # -------------------------------------------------------------------------
    # PASS 0: Static Baseline (No Memory, No Self-Reflection, Static 4 Pillars)
    # -------------------------------------------------------------------------
    print(
        ">>> 1. Running STATIC BASELINE (Zero Memory, No Self-Critique, Static RAG)..."
    )
    empty_memory = EpisodicMemoryBank(
        memory_file=JSON_DIR / "temp_empty_mem.json",
        vec_file=JSON_DIR / "temp_empty_vecs.npz",
    )
    empty_pillars = AdaptivePillarBank(state_file=JSON_DIR / "temp_empty_state.json")

    t0 = time.time()
    baseline_results, baseline_metrics = await run_audit_pass(
        rows=dataset,
        chunks=chunks,
        vectors=vectors,
        memory_bank=empty_memory,
        pillar_bank=empty_pillars,
        verifier=verifier,
        critic=critic,
        is_baseline=True,
        run_id="Baseline",
    )
    baseline_metrics["time"] = round(time.time() - t0, 1)
    print(
        f"    Baseline Result -> Accuracy: {baseline_metrics['accuracy']:.2%} | F1: {baseline_metrics['f1']:.4f} | Precision: {baseline_metrics['precision']:.4f} | Recall: {baseline_metrics['recall']:.4f}"
    )

    # -------------------------------------------------------------------------
    # PASS 1: Continuous Learning Agent - Run 1 (Cold Start -> Memory Inception)
    # -------------------------------------------------------------------------
    print(
        "\n>>> 2. Running CONTINUOUS LEARNING AGENT - RUN 1 (Cold Start + Self-Critique Ingestion)..."
    )
    exp_memory = EpisodicMemoryBank(
        memory_file=JSON_DIR / "exp_mem.json", vec_file=JSON_DIR / "exp_vecs.npz"
    )
    exp_pillars = AdaptivePillarBank(state_file=JSON_DIR / "exp_state.json")
    exp_memory.episodes = []
    exp_memory.vectors = None

    t0 = time.time()
    run1_results, run1_metrics = await run_audit_pass(
        rows=dataset,
        chunks=chunks,
        vectors=vectors,
        memory_bank=exp_memory,
        pillar_bank=exp_pillars,
        verifier=verifier,
        critic=critic,
        is_baseline=False,
        run_id="CL_Run_1",
    )
    run1_metrics["time"] = round(time.time() - t0, 1)
    print(
        f"    Run 1 Result    -> Accuracy: {run1_metrics['accuracy']:.2%} | F1: {run1_metrics['f1']:.4f} | Self-Critique Corrections: {run1_metrics['critique_corrections']} | Precedents Indexed: {len(exp_memory.episodes)}"
    )

    # -------------------------------------------------------------------------
    # PASS 2: Continuous Learning Agent - Run 2 (With Precedents & Pitfalls)
    # -------------------------------------------------------------------------
    print(
        "\n>>> 3. Running CONTINUOUS LEARNING AGENT - RUN 2 (Leveraging Indexed Precedents & Pitfalls)..."
    )
    t0 = time.time()
    run2_results, run2_metrics = await run_audit_pass(
        rows=dataset,
        chunks=chunks,
        vectors=vectors,
        memory_bank=exp_memory,
        pillar_bank=exp_pillars,
        verifier=verifier,
        critic=critic,
        is_baseline=False,
        run_id="CL_Run_2",
    )
    run2_metrics["time"] = round(time.time() - t0, 1)
    print(
        f"    Run 2 Result    -> Accuracy: {run2_metrics['accuracy']:.2%} | F1: {run2_metrics['f1']:.4f} | Precision: {run2_metrics['precision']:.4f} | Recall: {run2_metrics['recall']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Comprehensive Comparison & Improvement Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("                    EMPIRICAL IMPROVEMENT COMPARISON MATRIX")
    print("=" * 88)
    print(
        f"{'Metric':<30} {'Static Baseline':<20} {'CL Run 1 (Cold)':<20} {'CL Run 2 (Learned)':<20}"
    )
    print("-" * 88)
    print(
        f"{'Accuracy':<30} {baseline_metrics['accuracy']:<20.2%} {run1_metrics['accuracy']:<20.2%} {run2_metrics['accuracy']:<20.2%}"
    )
    print(
        f"{'F1 Score':<30} {baseline_metrics['f1']:<20.4f} {run1_metrics['f1']:<20.4f} {run2_metrics['f1']:<20.4f}"
    )
    print(
        f"{'Precision':<30} {baseline_metrics['precision']:<20.4f} {run1_metrics['precision']:<20.4f} {run2_metrics['precision']:<20.4f}"
    )
    print(
        f"{'Recall':<30} {baseline_metrics['recall']:<20.4f} {run1_metrics['recall']:<20.4f} {run2_metrics['recall']:<20.4f}"
    )
    print(
        f"{'False Positives (Alarms)':<30} {baseline_metrics['false_positives']:<20} {run1_metrics['false_positives']:<20} {run2_metrics['false_positives']:<20}"
    )
    print(
        f"{'False Negatives (Misses)':<30} {baseline_metrics['false_negatives']:<20} {run1_metrics['false_negatives']:<20} {run2_metrics['false_negatives']:<20}"
    )
    print(
        f"{'Hallucinated Citations':<30} {baseline_metrics['citations_fabricated']:<20} {run1_metrics['citations_fabricated']:<20} {run2_metrics['citations_fabricated']:<20}"
    )
    print(
        f"{'Episodic Precedents in Memory':<30} {'0':<20} {len(exp_memory.episodes):<20} {len(exp_memory.episodes):<20}"
    )
    print("=" * 88)

    # Show specific case improvements
    print("\n>>> Cases Corrected by Continuous Learning:")
    improvements_found = 0
    for b_res, r_res, row in zip(baseline_results, run2_results, dataset):
        b_correct = b_res["predicted"] == b_res["actual"]
        r_correct = r_res["predicted"] == r_res["actual"]
        if not b_correct and r_correct:
            improvements_found += 1
            print(
                f"\n  [CORRECTED CASE #{improvements_found}] {row['Log_ID']}: '{row['Reported_Incident']}' at {row['Location']}"
            )
            print(
                f"    - Static Baseline Verdict : {b_res['status']} (Cite: {b_res['citation']}) -> WRONG (Expected: {'VIOLATION' if row['Is_Violation'] == '1' else 'CLEAR'})"
            )
            print(
                f"    - Continuous Learner (v2) : {r_res['status']} (Cite: {r_res['citation']}) -> CORRECT"
            )
            print(f"    - Learner Reasoning       : {r_res['reason']}")

    # Clean temporary files
    for p in [
        JSON_DIR / "temp_empty_mem.json",
        JSON_DIR / "temp_empty_vecs.npz",
        JSON_DIR / "temp_empty_state.json",
        JSON_DIR / "exp_mem.json",
        JSON_DIR / "exp_vecs.npz",
        JSON_DIR / "exp_state.json",
    ]:
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    asyncio.run(execute_experiment())
