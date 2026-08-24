"""Autonomous Continuous Learning Compliance Auditor (v2).

Features:
1. Multi-hazard triangulation over dynamically expanding regulatory pillars.
2. Dynamic episodic memory & few-shot exemplar retrieval from past runs.
3. Negative precedent (pitfall) avoidance.
4. Self-reflection & automated statutory re-grounding verification.
5. Persistent cross-run metrics tracking and continuous self-improvement.
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
REGS = JSON_DIR / "regulations.json"
EMBED_CACHE = JSON_DIR / "reg_embeddings.npz"
LOG = ROOT / "daily_yard_log.csv"
RESULTS = JSON_DIR / "audit_results.json"
VALID_SECTIONS = JSON_DIR / "valid_sections.json"

OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen3.5:9b"
TOP_K = 4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from continuous_learner import (
    AdaptivePillarBank,
    ContinuousMetricsTracker,
    EpisodicMemoryBank,
    embed_texts,
)
from self_reflection import SelfReflectionCritic, StatutoryGroundedVerifier

AUTONOMOUS_LEARNING_PROMPT = """You are an expert federal safety and compliance auditor for a rail-served intermodal yard.

Below are excerpts from federal safety regulations (OSHA Title 29 and FRA Title 49 CFR) retrieved for this yard event.
These are the ONLY regulations you may cite. Do not cite any section number not present in these excerpts. If no violation exists, cite NONE.

RETRIEVED REGULATIONS:
{context}

{precedents_section}
{pitfalls_section}

YARD EVENT TO AUDIT:
- Log ID: {log_id}
- Equipment: {equipment} ({equip_type})
- Location: {location}
- Operator Shift Duration: {shift_hours} hours
- Reported Incident / Telemetry: {incident}

AUDITING INSTRUCTIONS:
Evaluate whether the yard event violates any federal safety standard present in the retrieved excerpts above (such as Hours of Service limits, walking-working surface housekeeping, electrical clearance distances, rigging/slings, or equipment safety).
If a condition described in the event breaches a requirement in the retrieved excerpts, mark STATUS: VIOLATION and cite the exact section number from the text.
If no violation exists or if the event represents routine operational telemetry within legal limits, mark STATUS: CLEAR and CITATION: NONE.

Answer in exactly this format and nothing else:
STATUS: CLEAR or VIOLATION
CITATION: exact section number copied from the excerpts above, or NONE
REASON: one concise sentence.
"""


def load_corpus_index() -> Tuple[List[Dict[str, Any]], np.ndarray, Set[str]]:
    corpus = json.loads(REGS.read_text(encoding="utf-8"))
    chunks = corpus["chunks"]

    if EMBED_CACHE.exists():
        try:
            cached = np.load(EMBED_CACHE)
            if cached["vectors"].shape[0] == len(chunks):
                return chunks, cached["vectors"], set(corpus.get("sections", []))
        except Exception:
            pass

    print(f"Embedding {len(chunks)} regulation chunks with {EMBED_MODEL}...")
    vectors = embed_texts([f"{c['citation']} {c['heading']}. {c['text']}" for c in chunks])
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBED_CACHE, vectors=vectors)
    return chunks, vectors, set(corpus.get("sections", []))


def retrieve_triangulated_adaptive(
    row: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    vectors: np.ndarray,
    pillar_bank: AdaptivePillarBank,
    top_dyn: int = 2,
) -> List[Dict[str, Any]]:
    """Query regulations across active adaptive pillars and event telemetry."""
    selected_indices = []

    # 1. Pillar coverage across all active hazard dimensions
    for p_name, p_vec in pillar_bank.pillar_vectors.items():
        scores = vectors @ p_vec
        best_idx = int(np.argmax(scores))
        if best_idx not in selected_indices:
            selected_indices.append(best_idx)

    # 2. Dynamic event query
    event_query = (
        f"{row.get('Equipment_Type', '')} operating at {row.get('Location', '')}, "
        f"operator on shift {row.get('Operator_Shift_Hours', '')} hours, "
        f"incident reported: {row.get('Reported_Incident', '')}"
    )
    q_vec = embed_texts([event_query])[0]
    scores = vectors @ q_vec
    order = np.argsort(-scores)
    
    max_total = len(pillar_bank.pillar_vectors) + top_dyn
    for idx in order:
        if idx not in selected_indices:
            selected_indices.append(int(idx))
        if len(selected_indices) >= max_total:
            break

    return [chunks[i] for i in selected_indices]


def parse_response(text: str) -> Dict[str, Any]:
    # Strip any thinking tags if present
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    status = re.search(r"STATUS:\s*(VIOLATION|CLEAR)", cleaned, re.I)
    citation = re.search(r"CITATION:\s*([^\n]+)", cleaned, re.I)
    reason = re.search(r"REASON:\s*([^\n]+)", cleaned, re.I)
    cite = (citation.group(1).strip() if citation else "NONE")
    m = re.search(r"\b(\d{3,4}\.\d+)", cite)
    return {
        "status": (status.group(1).upper() if status else "UNPARSED"),
        "citation_raw": cite,
        "citation": (m.group(1) if m else None),
        "reason": (reason.group(1).strip() if reason else ""),
    }


async def ask_async(session: aiohttp.ClientSession, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": 250},
        }
        async with session.post(f"{OLLAMA}/api/chat", json=payload, timeout=300) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["message"]["content"]


def is_candidate_hazard(row: Dict[str, Any]) -> bool:
    try:
        shift_hrs = float(row.get("Operator_Shift_Hours", 0.0))
    except (ValueError, TypeError):
        shift_hrs = 0.0
    loc = str(row.get("Location", ""))
    inc = str(row.get("Reported_Incident", ""))
    
    return (
        shift_hrs > 12.0
        or inc in ("Hydraulic Leak", "Proximity Warning", "Load Imbalance", "Tire Pressure Warning")
        or any(w in loc for w in ("Crosswalk", "Walkway", "High-Voltage", "Pedestrian"))
        or any(k in inc.lower() for k in ("hazard", "leak", "warning", "violation", "defect", "fail"))
    )


async def run_continuous_audit(
    rows: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    vectors: np.ndarray,
    memory_bank: EpisodicMemoryBank,
    pillar_bank: AdaptivePillarBank,
    verifier: StatutoryGroundedVerifier,
    critic: SelfReflectionCritic,
    concurrency: int = 6,
    run_id: str = "run_1",
    enable_few_shot: bool = True,
    enable_self_reflection: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    llm_tasks = []
    results = [None] * len(rows)

    # 1. Scan for novel telemetry patterns and dynamically expand pillars if needed
    for row in rows:
        pillar_bank.scan_and_adapt(row)

    # 2. Build prompts with Dynamic Triangulation + Episodic Memory Exemplars + Pitfalls
    for idx, row in enumerate(rows):
        if not is_candidate_hazard(row):
            results[idx] = {
                "log_id": row.get("Log_ID"),
                "status": "CLEAR",
                "citation_raw": "NONE",
                "citation": None,
                "reason": "Operator shift duration is within statutory limits and no equipment incident or hazard reported.",
                "predicted": 0,
                "actual": int(row.get("Is_Violation", 0)),
                "actual_type": row.get("Violation_Type", "none"),
                "retrieved": [],
                "citation_exists": None,
                "citation_was_retrieved": None,
                "confidence": 1.0,
            }
            continue

        hits = retrieve_triangulated_adaptive(row, chunks, vectors, pillar_bank)
        context = "\n\n".join(
            f"[{c['citation']} -- {c['heading']}]\n{c['text'][:800]}" for c in hits
        )

        query_text = (
            f"Equipment: {row.get('Equipment_Type')} at {row.get('Location')}, "
            f"shift: {row.get('Operator_Shift_Hours')} hrs, incident: {row.get('Reported_Incident')}"
        )

        # Episodic Contrastive Few-Shot Precedents
        precedents_section = ""
        if enable_few_shot:
            contrastive = memory_bank.retrieve_contrastive_precedents(query_text)
            lines = []
            if contrastive.get("positive"):
                p = contrastive["positive"]
                lines.append(
                    f"- Infraction Case: {p['incident']} at {p['location']} (Shift: {p['shift_hours']} hrs) -> "
                    f"STATUS: {p['verdict']}, CITATION: {p['citation']} ({p['reason']})"
                )
            if contrastive.get("negative"):
                n = contrastive["negative"]
                lines.append(
                    f"- Compliant Baseline Case: {n['incident']} at {n['location']} (Shift: {n['shift_hours']} hrs) -> "
                    f"STATUS: {n['verdict']}, CITATION: NONE ({n['reason']})"
                )
            if lines:
                precedents_section = "VERIFIED AUDIT PRECEDENTS (From Past Shifts):\n" + "\n".join(lines) + "\n"

        # Pitfall warnings from past mistakes
        pitfalls_section = ""
        if enable_few_shot:
            pitfalls = memory_bank.retrieve_pitfalls(query_text, k=1)
            if pitfalls:
                lines = ["LEARNED PITFALL WARNINGS (Avoid Past Mistakes):"]
                for pit in pitfalls:
                    lines.append(f"- Note: {pit['past_mistake']} -> Correct: {pit['correct_verdict']} ({pit['correct_citation']})")
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

    # 3. Asynchronous LLM execution
    critique_corrections_count = 0
    if llm_tasks:
        print(f"  Dispatching {len(llm_tasks)} candidate hazard audits across {concurrency} async workers...")
        async with aiohttp.ClientSession() as session:
            coros = [ask_async(session, p, sem) for _, _, _, p in llm_tasks]
            raw_responses = await asyncio.gather(*coros)

            # 4. Parsing, Verification, and Self-Reflection Critique Loop
            for (idx, row, hits, p), raw in zip(llm_tasks, raw_responses):
                parsed = parse_response(raw)

                # Self-Reflection & Critique check
                if enable_self_reflection:
                    refined, was_corrected, critique_log = await critic.critique_and_reground_async(
                        session, p, parsed, row, hits, sem
                    )
                    if was_corrected:
                        critique_corrections_count += 1
                        print(f"    [SelfReflection] Corrected {row.get('Log_ID')}: {refined.get('original_verdict')} -> {refined.get('status')}: {refined.get('citation')}")
                    parsed = refined

                cite = parsed["citation"]
                retrieved_sections = {c["section"] for c in hits}

                parsed.update({
                    "log_id": row.get("Log_ID"),
                    "predicted": 1 if parsed["status"] == "VIOLATION" else 0,
                    "actual": int(row.get("Is_Violation", 0)),
                    "actual_type": row.get("Violation_Type", "none"),
                    "retrieved": sorted(list(retrieved_sections)),
                    "citation_exists": (None if cite is None else cite in verifier.valid_sections),
                    "citation_was_retrieved": (
                        None if cite is None else cite in retrieved_sections
                    ),
                })
                results[idx] = parsed

                # 5. Ingest into Episodic Memory Bank
                is_grounded = bool(parsed.get("citation_exists") and parsed.get("citation_was_retrieved")) if cite else True
                feedback_type = "verified" if is_grounded else "pitfall"
                memory_bank.add_episode(
                    log_id=str(row.get("Log_ID")),
                    row=row,
                    verdict=parsed["status"],
                    citation=parsed["citation"],
                    reason=parsed["reason"],
                    is_grounded=is_grounded,
                    confidence=parsed.get("confidence", 1.0),
                    critique_notes=parsed.get("critique_issues", None) if not is_grounded else None,
                    feedback_type=feedback_type,
                    run_id=run_id,
                )

    metrics_extra = {
        "critique_corrections": critique_corrections_count,
        "memory_count": len(memory_bank.episodes),
        "pillar_count": len(pillar_bank.pillars),
    }
    return results, metrics_extra


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(r["predicted"] and r["actual"] for r in results)
    fp = sum(r["predicted"] and not r["actual"] for r in results)
    fn = sum(not r["predicted"] and r["actual"] for r in results)
    tn = sum(not r["predicted"] and not r["actual"] for r in results)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    cited = [r for r in results if r["citation"] is not None]
    fabricated = [r for r in cited if not r["citation_exists"]]
    ungrounded = [r for r in cited if r["citation_exists"] and not r["citation_was_retrieved"]]

    return {
        "n": len(results),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(results), 4),
        "citations_given": len(cited),
        "citations_fabricated": len(fabricated),
        "citations_outside_retrieved_set": len(ungrounded),
        "citation_validity": round(1 - len(fabricated) / len(cited), 4) if cited else 1.0,
        "unparsed": sum(r["status"] == "UNPARSED" for r in results),
    }


def main():
    ap = argparse.ArgumentParser(description="Autonomous Continuous Learning Compliance Auditor.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--disable-few-shot", action="store_true")
    ap.add_argument("--disable-self-reflection", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with LOG.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]

    chunks, vectors, corpus_sections = load_corpus_index()
    memory_bank = EpisodicMemoryBank()
    pillar_bank = AdaptivePillarBank()
    verifier = StatutoryGroundedVerifier()
    critic = SelfReflectionCritic(verifier)
    tracker = ContinuousMetricsTracker()

    run_id = args.run_id or f"run_{len(tracker.history) + 1}"

    print(f"\n========================================================")
    print(f"  AUTONOMOUS CONTINUOUS LEARNING COMPLIANCE AUDITOR (v2)")
    print(f"  Run ID: {run_id} | Events: {len(rows)} | Memory Precedents: {len(memory_bank.episodes)}")
    print(f"  Active Hazard Pillars: {list(pillar_bank.pillars.keys())}")
    print(f"========================================================\n")

    t0 = time.time()
    results, extra_info = asyncio.run(
        run_continuous_audit(
            rows=rows,
            chunks=chunks,
            vectors=vectors,
            memory_bank=memory_bank,
            pillar_bank=pillar_bank,
            verifier=verifier,
            critic=critic,
            concurrency=args.concurrency,
            run_id=run_id,
            enable_few_shot=not args.disable_few_shot,
            enable_self_reflection=not args.disable_self_reflection,
        )
    )
    elapsed = round(time.time() - t0, 1)

    metrics = compute_metrics(results)
    metrics["seconds"] = elapsed
    metrics["run_id"] = run_id
    metrics.update(extra_info)

    # Record run in persistent history
    tracker.record_run(metrics)

    out_path = Path(args.out) if args.out else RESULTS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"metrics": metrics, "results": results}, indent=2), encoding="utf-8")

    print("\n" + "=" * 56)
    print("  AUDIT METRICS & LEARNING PROGRESSION")
    print("=" * 56)
    for k, v in metrics.items():
        print(f"  {k:34} {v}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
