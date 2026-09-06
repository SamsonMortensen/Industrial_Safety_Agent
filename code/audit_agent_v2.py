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
from typing import Any, Dict, List, Set, Tuple

import aiohttp
import numpy as np
from ollama_config import ollama_base_url

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
REGS = JSON_DIR / "regulations.json"
STATUTES = JSON_DIR / "statutes.json"
EMBED_CACHE = JSON_DIR / "reg_embeddings.npz"
LOG = ROOT / "daily_yard_log.csv"
RESULTS = JSON_DIR / "audit_results.json"
VALID_SECTIONS = JSON_DIR / "valid_sections.json"

OLLAMA = ollama_base_url()

# Allow slower CPU hosts to override the inference timeout.
REQUEST_TIMEOUT = int(os.getenv("AUDIT_TIMEOUT", "900"))
MAX_ATTEMPTS = int(os.getenv("AUDIT_RETRIES", "3"))

_PROGRESS = {"done": 0, "total": 0, "failed": 0}
TEMPERATURE = 0.0
NUM_PREDICT = 250

EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen3.5:9b"

# Ollama otherwise allocates the model's full context window (262144 for
# qwen3.5:9b), which reserves roughly 15GB of KV cache and pushes the model
# onto CPU. Audit prompts run to about 1,500 tokens.
NUM_CTX = int(os.getenv("AUDIT_NUM_CTX", "8192"))
TOP_K = 4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from continuous_learner import (
    AdaptivePillarBank,
    ContinuousMetricsTracker,
    EpisodicMemoryBank,
    build_query_text,
    embed_texts,
    event_signature,
    parse_event_time,
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
Decide whether the event below violates a requirement that is actually stated in the retrieved excerpts.
Base the decision only on what those excerpts require and what the event reports. Do not apply a safety rule you happen to know but cannot locate in the excerpts, and do not cite a section number that does not appear above, even if you believe it is the governing rule. If the excerpts do not contain the rule you need, answer CLEAR.
If a requirement in the excerpts is breached, answer VIOLATION and cite that section number exactly as it appears.
Otherwise answer CLEAR with CITATION: NONE.

Answer in exactly this format and nothing else:
STATUS: CLEAR or VIOLATION
CITATION: exact section number copied from the excerpts above, or NONE
REASON: one concise sentence.
"""


def load_corpus_index() -> Tuple[List[Dict[str, Any]], np.ndarray, Set[str]]:
    corpus = json.loads(REGS.read_text(encoding="utf-8"))
    chunks = list(corpus["chunks"])
    corpus_sections = set(corpus.get("sections", []))
    if STATUTES.exists():
        statutes = json.loads(STATUTES.read_text(encoding="utf-8"))
        chunks.extend(statutes.get("chunks", []))
        corpus_sections.update(statutes.get("sections", []))

    if EMBED_CACHE.exists():
        try:
            cached = np.load(EMBED_CACHE)
            if cached["vectors"].shape[0] == len(chunks):
                return chunks, cached["vectors"], corpus_sections
        except Exception:
            pass

    print(f"Embedding {len(chunks)} regulation chunks with {EMBED_MODEL}...")
    vectors = embed_texts(
        [f"{c['citation']} {c['heading']}. {c['text']}" for c in chunks]
    )
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBED_CACHE, vectors=vectors)
    return chunks, vectors, corpus_sections


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
    cite = citation.group(1).strip() if citation else "NONE"
    m = re.search(r"\b(\d{3,4}\.\d+|21103)\b", cite)
    return {
        "status": (status.group(1).upper() if status else "UNPARSED"),
        "citation_raw": cite,
        "citation": (m.group(1) if m else None),
        "reason": (reason.group(1).strip() if reason else ""),
    }


async def ask_async(session, prompt, sem):
    """Run one audit call, with retry and backoff.

    A request that exhausts its retries returns "" and is scored UNPARSED,
    allowing the remaining batch results to complete.
    """
    async with sem:
        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT,
                "num_ctx": NUM_CTX,
            },
        }
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                async with session.post(
                    f"{OLLAMA}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                _PROGRESS["done"] += 1
                if (
                    _PROGRESS["done"] % 25 == 0
                    or _PROGRESS["done"] == _PROGRESS["total"]
                ):
                    print(
                        f"    progress: {_PROGRESS['done']}/{_PROGRESS['total']} audited"
                        f" ({_PROGRESS['failed']} failed)",
                        flush=True,
                    )
                return data["message"]["content"]
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                if attempt == MAX_ATTEMPTS:
                    _PROGRESS["failed"] += 1
                    print(
                        f"    [warn] audit call failed after {MAX_ATTEMPTS} attempts "
                        f"({type(exc).__name__}); scoring UNPARSED",
                        flush=True,
                    )
                    return ""
                await asyncio.sleep(2**attempt)
    return ""


def is_candidate_hazard(row: Dict[str, Any]) -> bool:
    """Cheap keyword gate for routine events. Off by default.

    Its measured recall is 0.27 on the current benchmark, so it remains an
    explicit cost-control option and is excluded from default measurement.
    """
    try:
        shift_hrs = float(row.get("Operator_Shift_Hours", 0.0))
    except (ValueError, TypeError):
        shift_hrs = 0.0
    loc = str(row.get("Location", ""))
    inc = str(row.get("Reported_Incident", ""))

    return (
        shift_hrs > 12.0
        or inc
        in (
            "Hydraulic Leak",
            "Proximity Warning",
            "Load Imbalance",
            "Tire Pressure Warning",
        )
        or any(w in loc for w in ("Crosswalk", "Walkway", "High-Voltage", "Pedestrian"))
        or any(
            k in inc.lower()
            for k in ("hazard", "leak", "warning", "violation", "defect", "fail")
        )
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
    ingest_memory: bool = True,
    adapt_pillars: bool = True,
    use_triage: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    llm_tasks = []
    results = [None] * len(rows)

    # Expand retrieval only from verified gaps observed in model results.

    # 2. Build prompts with Dynamic Triangulation + Episodic Memory Exemplars + Pitfalls
    for idx, row in enumerate(rows):
        if use_triage and not is_candidate_hazard(row):
            results[idx] = {
                "log_id": row.get("Log_ID"),
                "status": "CLEAR",
                "audited_by": "triage",
                "citation_raw": "NONE",
                "citation": None,
                "reason": "Operator shift duration is within statutory limits and no equipment incident or hazard reported.",
                "predicted": 0,
                "actual": int(row.get("Is_Violation", 0)),
                "actual_type": row.get("Violation_Type", "none"),
                "expected_section": row.get("Expected_Section"),
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

        query_text = build_query_text(row)

        # Episodic Contrastive Few-Shot Precedents
        precedents_section = ""
        if enable_few_shot:
            contrastive = memory_bank.retrieve_contrastive_precedents(
                query_text,
                exclude_log_id=str(row.get("Log_ID")),
                before_event_time=parse_event_time(row),
                exclude_signature=event_signature(row),
            )
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
                precedents_section = (
                    "VERIFIED AUDIT PRECEDENTS (From Past Shifts):\n"
                    + "\n".join(lines)
                    + "\n"
                )

        # Pitfall warnings from past mistakes
        pitfalls_section = ""
        if enable_few_shot:
            pitfalls = memory_bank.retrieve_pitfalls(
                query_text,
                k=1,
                exclude_log_id=str(row.get("Log_ID")),
                before_event_time=parse_event_time(row),
                exclude_signature=event_signature(row),
            )
            if pitfalls:
                lines = ["LEARNED PITFALL WARNINGS (Avoid Past Mistakes):"]
                for pit in pitfalls:
                    lines.append(
                        f"- Note: {pit['past_mistake']} -> Correct: {pit['correct_verdict']} ({pit['correct_citation']})"
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

    # 3. Asynchronous LLM execution
    critique_corrections_count = 0
    if llm_tasks:
        print(
            f"  Dispatching {len(llm_tasks)} candidate hazard audits across {concurrency} async workers..."
        )
        async with aiohttp.ClientSession() as session:
            _PROGRESS.update({"done": 0, "total": len(llm_tasks), "failed": 0})
            coros = [ask_async(session, p, sem) for _, _, _, p in llm_tasks]
            raw_responses = await asyncio.gather(*coros, return_exceptions=True)
            raw_responses = [
                "" if isinstance(r, BaseException) else r for r in raw_responses
            ]

            # 4. Parsing, Verification, and Self-Reflection Critique Loop
            for (idx, row, hits, p), raw in zip(llm_tasks, raw_responses):
                parsed = parse_response(raw)

                # Self-Reflection & Critique check
                if enable_self_reflection:
                    (
                        refined,
                        was_corrected,
                        critique_log,
                    ) = await critic.critique_and_reground_async(
                        session, p, parsed, row, hits, sem
                    )
                    if was_corrected:
                        critique_corrections_count += 1
                        print(
                            f"    [SelfReflection] Corrected {row.get('Log_ID')}: {refined.get('original_verdict')} -> {refined.get('status')}: {refined.get('citation')}"
                        )
                    parsed = refined

                cite = parsed["citation"]
                retrieved_sections = {c["section"] for c in hits}

                parsed.update(
                    {
                        "log_id": row.get("Log_ID"),
                        "audited_by": "model",
                        "predicted": 1 if parsed["status"] == "VIOLATION" else 0,
                        "actual": int(row.get("Is_Violation", 0)),
                        "actual_type": row.get("Violation_Type", "none"),
                        "expected_section": row.get("Expected_Section"),
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

                # A real section the model reached for but retrieval never showed it.
                # Label-free evidence of a blindspot, and it names the missing rule.
                if parsed.get("citation_exists") and not parsed.get(
                    "citation_was_retrieved"
                ):
                    pillar_bank.record_retrieval_gap(cite, row)

                # 5. Ingest into Episodic Memory Bank
                is_grounded = (
                    bool(
                        parsed.get("citation_exists")
                        and parsed.get("citation_was_retrieved")
                    )
                    if cite
                    else True
                )
                feedback_type = "verified" if is_grounded else "pitfall"
                if not ingest_memory:
                    continue
                memory_bank.add_episode(
                    log_id=str(row.get("Log_ID")),
                    row=row,
                    verdict=parsed["status"],
                    citation=parsed["citation"],
                    reason=parsed["reason"],
                    is_grounded=is_grounded,
                    confidence=parsed.get("confidence", 1.0),
                    critique_notes=parsed.get("critique_issues", None)
                    if not is_grounded
                    else None,
                    feedback_type=feedback_type,
                    run_id=run_id,
                )

    promoted = []
    if adapt_pillars and pillar_bank.gaps:
        promoted = pillar_bank.promote_gaps(chunks, vectors)
        if promoted:
            print(f"  Retrieval gaps promoted to pillars: {', '.join(promoted)}")

    metrics_extra = {
        "retrieval_gaps_seen": {k: v["count"] for k, v in pillar_bank.gaps.items()},
        "pillars_promoted": promoted,
        "critique_corrections": critique_corrections_count,
        "memory_count": len(memory_bank.episodes),
        "pillar_count": len(pillar_bank.pillars),
    }
    return results, metrics_extra


def _core_metrics(results):
    """Confusion matrix with unparsed outputs scored as strict errors."""
    tp = sum(r.get("predicted") == 1 and r["actual"] == 1 for r in results)
    fp = sum(r.get("predicted") != 0 and r["actual"] == 0 for r in results)
    fn = sum(r.get("predicted") != 1 and r["actual"] == 1 for r in results)
    tn = sum(r.get("predicted") == 0 and r["actual"] == 0 for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(results),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(results), 4) if results else None,
    }


def compute_metrics(results):
    """Report the model's performance separately from the triage stage's.

    A keyword pre-filter clears routine events without a model call. Those
    verdicts are constants, not predictions, so folding them into one accuracy
    figure inflates it: on the current log 432 of 1,000 events never reach the
    model, and every one is a true negative by construction. The corpus-wide
    numbers are still reported, because that is what the pipeline as a whole
    does, but "model_only" is the number that says anything about the model.
    """
    model_rows = [r for r in results if r.get("audited_by") == "model"]
    triage_rows = [r for r in results if r.get("audited_by") == "triage"]

    corpus = _core_metrics(results)
    model_only = _core_metrics(model_rows)

    cited = [r for r in results if r["citation"] is not None]
    fabricated = [r for r in cited if not r["citation_exists"]]
    ungrounded = [
        r for r in cited if r["citation_exists"] and not r["citation_was_retrieved"]
    ]

    caught = [r for r in results if r["actual"] and r["predicted"]]
    right = [
        r
        for r in caught
        if r.get("citation") and r.get("citation") == r.get("expected_section")
    ]

    out = dict(corpus)
    out.update(
        {
            "citation_correctness": round(len(right) / len(caught), 4)
            if caught
            else None,
            "citations_correct_rule": len(right),
            "model_audited": len(model_rows),
            "triage_cleared": len(triage_rows),
            "triage_missed_violations": sum(r["actual"] for r in triage_rows),
            "model_only": model_only,
            "citations_given": len(cited),
            "citations_fabricated": len(fabricated),
            "citations_outside_retrieved_set": len(ungrounded),
            "citation_validity": round(1 - len(fabricated) / len(cited), 4)
            if cited
            else None,
            "citations_grounded": len(cited) - len(fabricated) - len(ungrounded),
            "grounded_citation_rate": (
                round((len(cited) - len(fabricated) - len(ungrounded)) / len(cited), 4)
                if cited
                else None
            ),
            "unparsed": sum(r["status"] == "UNPARSED" for r in results),
        }
    )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Autonomous Continuous Learning Compliance Auditor."
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument(
        "--triage",
        action="store_true",
        help="skip routine rows with the keyword pre-filter (cheaper, and lossy: "
        "measured recall 0.27 on realistic phrasing). Off by default.",
    )
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--disable-few-shot", action="store_true")
    ap.add_argument("--disable-self-reflection", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with LOG.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    chunks, vectors, corpus_sections = load_corpus_index()
    memory_bank = EpisodicMemoryBank()
    pillar_bank = AdaptivePillarBank()
    verifier = StatutoryGroundedVerifier()
    critic = SelfReflectionCritic(verifier)
    tracker = ContinuousMetricsTracker()

    run_id = args.run_id or f"run_{len(tracker.history) + 1}"

    print("\n========================================================")
    print("  AUTONOMOUS CONTINUOUS LEARNING COMPLIANCE AUDITOR (v2)")
    print(
        f"  Run ID: {run_id} | Events: {len(rows)} | Memory Precedents: {len(memory_bank.episodes)}"
    )
    print(f"  Active Hazard Pillars: {list(pillar_bank.pillars.keys())}")
    print("========================================================\n")

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
            use_triage=args.triage,
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
    out_path.write_text(
        json.dumps({"metrics": metrics, "results": results}, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 56)
    print("  AUDIT METRICS & LEARNING PROGRESSION")
    print("=" * 56)
    for k, v in metrics.items():
        print(f"  {k:34} {v}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
