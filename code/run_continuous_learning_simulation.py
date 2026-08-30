"""Continuous Learning Multi-Stage Simulation.

Demonstrates how the Compliance Auditor continuously learns, adapts its retrieval pillars,
accumulates episodic precedents, avoids past pitfalls, and verifies its own grounding
across successive runs.
"""

import asyncio
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_agent_v2 import (
    LOG,
    compute_metrics,
    load_corpus_index,
    run_continuous_audit,
)
from continuous_learner import (
    AdaptivePillarBank,
    ContinuousMetricsTracker,
    EpisodicMemoryBank,
)
from learning_dashboard import show_dashboard
from self_reflection import SelfReflectionCritic, StatutoryGroundedVerifier


def create_synthetic_novel_batch() -> list:
    """Creates a realistic batch containing novel hazard types and subtle edge cases."""
    return [
        {
            "Log_ID": "LOG-NOVEL-01",
            "Timestamp": "2026-04-01 08:30",
            "Equipment_ID": "FL-09",
            "Equipment_Type": "Forklift",
            "Location": "Chemical Transfer Track 4",
            "Operator_Shift_Hours": "6.5",
            "Reported_Incident": "Unsecured Hazmat Placard with Corrosive Drum Leak",
            "Is_Violation": "1",
            "Violation_Type": "hazmat",
        },
        {
            "Log_ID": "LOG-NOVEL-02",
            "Timestamp": "2026-04-01 09:15",
            "Equipment_ID": "GC-02",
            "Equipment_Type": "Gantry Crane",
            "Location": "North Intermodal Spur",
            "Operator_Shift_Hours": "11.8",
            "Reported_Incident": "Routine Gantry Spreader Sensor Calibration",
            "Is_Violation": "0",
            "Violation_Type": "none",
        },
        {
            "Log_ID": "LOG-NOVEL-03",
            "Timestamp": "2026-04-01 10:00",
            "Equipment_ID": "RS-14",
            "Equipment_Type": "Reach Stacker",
            "Location": "Pedestrian Crosswalk Bravo",
            "Operator_Shift_Hours": "7.0",
            "Reported_Incident": "Hydraulic Leak creating oil puddle",
            "Is_Violation": "1",
            "Violation_Type": "spill",
        },
        {
            "Log_ID": "LOG-NOVEL-04",
            "Timestamp": "2026-04-01 11:30",
            "Equipment_ID": "SW-01",
            "Equipment_Type": "Switch Engine",
            "Location": "Track 12 High-Voltage Overhead Line",
            "Operator_Shift_Hours": "13.2",
            "Reported_Incident": "Proximity Warning",
            "Is_Violation": "1",
            "Violation_Type": "fatigue",
        },
        {
            "Log_ID": "LOG-NOVEL-05",
            "Timestamp": "2026-04-01 12:45",
            "Equipment_ID": "FL-03",
            "Equipment_Type": "Forklift",
            "Location": "Maintenance Bay 2",
            "Operator_Shift_Hours": "4.0",
            "Reported_Incident": "Routine Tire Pressure Warning adjustment",
            "Is_Violation": "0",
            "Violation_Type": "none",
        },
    ]


async def run_simulation():
    print("\n" + "#" * 80)
    print("  COMMENCING MULTI-STAGE AUTONOMOUS CONTINUOUS LEARNING SIMULATION")
    print("#" * 80 + "\n")

    chunks, vectors, corpus_sections = load_corpus_index()
    memory_bank = EpisodicMemoryBank()
    pillar_bank = AdaptivePillarBank()
    verifier = StatutoryGroundedVerifier()
    critic = SelfReflectionCritic(verifier)
    tracker = ContinuousMetricsTracker()

    with LOG.open(encoding="utf-8") as fh:
        base_rows = list(csv.DictReader(fh))

    # =========================================================================
    # STAGE 1: Cold Start Execution (Run 1)
    # =========================================================================
    print("\n" + "=" * 80)
    print(">>> STAGE 1: COLD START AUDIT (Run 1)")
    print("    Auditing 30 baseline records with empty episodic memory.")
    print("=" * 80)
    stage1_rows = base_rows[:30]
    t0 = time.time()
    results1, extra1 = await run_continuous_audit(
        rows=stage1_rows,
        chunks=chunks,
        vectors=vectors,
        memory_bank=memory_bank,
        pillar_bank=pillar_bank,
        verifier=verifier,
        critic=critic,
        concurrency=4,
        run_id="Run-1-ColdStart",
    )
    m1 = compute_metrics(results1)
    m1.update(extra1)
    m1["seconds"] = round(time.time() - t0, 1)
    tracker.record_run(m1)
    print(
        f"Stage 1 Completed in {m1['seconds']}s. F1: {m1['f1']} | Memory Size: {len(memory_bank.episodes)} precedents stored."
    )

    # =========================================================================
    # STAGE 2: Novel Pattern Injection & Pillar Discovery (Run 2)
    # =========================================================================
    print("\n" + "=" * 80)
    print(">>> STAGE 2: NOVEL HAZARD INJECTION & PILLAR ADAPTATION (Run 2)")
    print("    Introducing novel Hazmat incidents and borderline telemetry.")
    print("=" * 80)
    stage2_rows = create_synthetic_novel_batch()
    t0 = time.time()
    results2, extra2 = await run_continuous_audit(
        rows=stage2_rows,
        chunks=chunks,
        vectors=vectors,
        memory_bank=memory_bank,
        pillar_bank=pillar_bank,
        verifier=verifier,
        critic=critic,
        concurrency=4,
        run_id="Run-2-NovelHazards",
    )
    m2 = compute_metrics(results2)
    m2.update(extra2)
    m2["seconds"] = round(time.time() - t0, 1)
    tracker.record_run(m2)
    print(
        f"Stage 2 Completed in {m2['seconds']}s. F1: {m2['f1']} | Active Pillars: {len(pillar_bank.pillars)}."
    )
    print(f"Newly Registered Pillars: {list(pillar_bank.pillars.keys())}")

    # =========================================================================
    # STAGE 3: Consolidated Continuous Learning Audit (Run 3)
    # =========================================================================
    print("\n" + "=" * 80)
    print(">>> STAGE 3: CONSOLIDATED MEMORY & ADAPTIVE PRECEDENTS RUN (Run 3)")
    print(
        "    Auditing 50 mixed records utilizing accumulated precedents & expanded pillars."
    )
    print("=" * 80)
    stage3_rows = base_rows[30:80]
    t0 = time.time()
    results3, extra3 = await run_continuous_audit(
        rows=stage3_rows,
        chunks=chunks,
        vectors=vectors,
        memory_bank=memory_bank,
        pillar_bank=pillar_bank,
        verifier=verifier,
        critic=critic,
        concurrency=4,
        run_id="Run-3-Consolidated",
    )
    m3 = compute_metrics(results3)
    m3.update(extra3)
    m3["seconds"] = round(time.time() - t0, 1)
    tracker.record_run(m3)
    print(
        f"Stage 3 Completed in {m3['seconds']}s. F1: {m3['f1']} | Total Memory: {len(memory_bank.episodes)}."
    )

    # Show final dashboard
    show_dashboard()


if __name__ == "__main__":
    asyncio.run(run_simulation())
