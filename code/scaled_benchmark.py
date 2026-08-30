"""Run a leakage-resistant benchmark for trusted episodic learning.

The benchmark uses disjoint synthetic training and holdout phrasing. It compares
the same triangulated RAG pipeline with no memory, trusted memory, and trusted
memory plus statutory self-reflection. Holdout labels never enter memory.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import aiohttp
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = JSON_DIR / "scaled_benchmark_results.json"
TRAIN_FILE = DATA_DIR / "scaled_benchmark_train.csv"
HOLDOUT_FILE = DATA_DIR / "scaled_benchmark_holdout.csv"
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_agent_v2 import (
    AUTONOMOUS_LEARNING_PROMPT,
    CHAT_MODEL,
    EMBED_MODEL,
    compute_metrics,
    load_corpus_index,
    parse_response,
    retrieve_triangulated_adaptive,
)
from continuous_learner import AdaptivePillarBank, EpisodicMemoryBank, embed_texts
from hybrid_policy import evaluate_event
from self_reflection import SelfReflectionCritic, StatutoryGroundedVerifier

SCENARIOS = [
    {
        "name": "housekeeping",
        "citation": "1910.22",
        "equipment": "Forklift",
        "location": "Marked Pedestrian Walkway",
        "train_positive": [
            "Hydraulic oil remained pooled across the marked walking route during operations",
            "A coolant leak left a slick film over the employee crosswalk",
            "Diesel residue spread across the personnel aisle and was not isolated",
            "Rainwater mixed with lubricant and covered the main access walkway",
            "A ruptured hose discharged fluid onto the active pedestrian path",
        ],
        "train_negative": [
            "A small leak was captured in a drip pan before reaching the walking route",
            "The crosswalk was cleaned and dried before employees were allowed to enter",
            "A sealed lubricant container was transported through the personnel aisle",
            "Wash water remained inside the curbed treatment bay",
            "Absorbent barriers contained the spill outside every walking surface",
        ],
        "test_positive": [
            "An unrepaired steering leak coated the painted footpath with slippery fluid",
            "Workers stepped around an uncontained oil puddle in the designated crosswalk",
            "A sheen of hydraulic fluid remained on the only open pedestrian passage",
            "Leaking equipment tracked grease along the active employee walkway",
        ],
        "test_negative": [
            "The affected aisle reopened only after absorbent cleanup left it dry",
            "A closed transfer hose passed above the walkway without releasing material",
            "Fluid from maintenance drained into a covered containment sump",
            "The spill area was barricaded and an alternate dry walkway was in service",
        ],
    },
    {
        "name": "fatigue",
        "citation": "21103",
        "equipment": "Switch Engine",
        "location": "Classification Yard",
        "train_positive": [
            "Covered service continued because the relief crew arrived late",
            "The crew remained on duty to finish switching the outbound consist",
            "An employee in covered service accepted an extended tour after the handoff failed",
            "The switch crew kept working through a second unscheduled movement",
            "Dispatch required the covered employee to complete additional yard moves",
        ],
        "train_negative": [
            "The employee tied down the consist and went off duty before the limit",
            "A relief crew assumed the assignment within the permitted tour",
            "The covered service ended after the scheduled final movement",
            "The employee logged out and entered the required rest period",
            "The crew transferred control before reaching the statutory ceiling",
        ],
        "test_positive": [
            "A late inbound train kept the covered switch employee on duty beyond the ceiling",
            "The yardmaster extended covered service to clear an additional departure track",
            "No relief was available, so the switch crew continued active covered duty",
            "The employee performed another switching move after exceeding the duty limit",
        ],
        "test_negative": [
            "The final covered movement ended with time remaining before the limit",
            "The employee entered rest status after a compliant tour of duty",
            "Relief personnel took over before the statutory maximum was reached",
            "The crew stopped covered service and secured equipment within the legal window",
        ],
    },
    {
        "name": "electrical_clearance",
        "citation": "1910.333",
        "equipment": "Gantry Crane",
        "location": "Energized Catenary Zone",
        "train_positive": [
            "The raised boom entered the restricted approach area around an energized conductor",
            "An unqualified operator moved the spreader within six feet of a live 25 kV line",
            "The crane mast approached exposed energized parts without protective measures",
            "Equipment crossed the minimum approach boundary beside the live busbar",
            "The load line swung into the prohibited clearance around energized wiring",
        ],
        "train_negative": [
            "The circuit was de-energized, locked out, tested, and grounded before work",
            "A spotter maintained the documented safe distance from the energized line",
            "The crane remained outside the established approach boundary",
            "Work occurred beyond the grounded substation perimeter barrier",
            "Insulating barriers prevented entry into the restricted approach area",
        ],
        "test_positive": [
            "The boom tip passed inside the posted live-line clearance boundary",
            "A forklift mast operated four feet from an exposed energized bus",
            "The suspended cable drifted toward a live overhead conductor inside minimum clearance",
            "Unqualified personnel positioned lifting equipment beside unguarded energized parts",
        ],
        "test_negative": [
            "A verified lockout and absence-of-voltage test preceded the lift",
            "The operator stopped at the marked safe approach line with a dedicated observer",
            "Grounded protective barriers separated the equipment from energized components",
            "The boom route stayed outside the engineering-approved live-line envelope",
        ],
    },
    {
        "name": "defective_sling",
        "citation": "1910.184",
        "equipment": "Gantry Crane",
        "location": "Intermodal Lift Pad",
        "train_positive": [
            "A wire-rope sling with broken outer wires remained in an active lift",
            "The synthetic sling had a melted section and no readable capacity tag",
            "A crushed eye fitting was used to hoist a loaded container",
            "The chain sling showed a stretched link but stayed in service",
            "A severely abraded web sling supported the suspended load",
        ],
        "train_negative": [
            "The inspected sling had a legible rating tag and no visible damage",
            "A damaged sling was removed from service before the lift began",
            "The qualified rigger documented the chain inspection as acceptable",
            "Protected sling eyes and rated hardware were used within capacity",
            "The crew replaced the abraded sling before attaching the load",
        ],
        "test_positive": [
            "A kinked wire rope with multiple broken strands lifted the chassis",
            "The crew used a heat-damaged web sling whose identification was missing",
            "A cracked end attachment remained connected during the hoist",
            "An elongated chain link carried a suspended container despite inspection rejection",
        ],
        "test_negative": [
            "Inspection found the rated sling undamaged before the planned lift",
            "The questionable chain was tagged out and replaced with certified rigging",
            "The sling stayed within its marked load rating and protective sleeves were fitted",
            "A qualified person approved the clean, legibly tagged rigging assembly",
        ],
    },
    {
        "name": "aisle_clearance",
        "citation": "1910.176",
        "equipment": "Reach Stacker",
        "location": "Materials Storage Aisle",
        "train_positive": [
            "Palletized cargo blocked the marked employee aisle and emergency access route",
            "Stored chassis parts projected into the designated pedestrian passage",
            "Containers were staged across the only clear materials-handling aisle",
            "Loose cargo narrowed the marked access lane below usable clearance",
            "A stack of crates obstructed the permanent aisle to the exit",
        ],
        "train_negative": [
            "Pallets remained stable and completely inside the marked storage boundary",
            "The designated aisle stayed clear during material staging",
            "Cargo was secured without projecting into any passageway",
            "The crew restored full aisle clearance before operations resumed",
            "Storage racks kept every load inside the protected footprint",
        ],
        "test_positive": [
            "Overhanging freight prevented employees from using the painted access aisle",
            "Material staging closed the permanent passage between the yard and exit",
            "Unsecured parts spilled into the route reserved for personnel travel",
            "A container queue eliminated required clearance in the handling aisle",
        ],
        "test_negative": [
            "Freight remained behind the floor line and the passage stayed unobstructed",
            "A temporary staging area preserved the full marked aisle width",
            "Secured loads occupied racks without extending into employee access",
            "The route to the exit remained open throughout container placement",
        ],
    },
    {
        "name": "dockboard",
        "citation": "1910.26",
        "equipment": "Forklift",
        "location": "Railcar Loading Dock",
        "train_positive": [
            "An unsecured portable dockboard shifted while the forklift crossed",
            "The dock plate had no anchoring device and slid away from the railcar",
            "A damaged dockboard moved under the weight of powered equipment",
            "The portable bridge plate was used without positive securing",
            "A forklift crossed a dockboard that lacked safe handholds for repositioning",
        ],
        "train_negative": [
            "The rated dockboard was anchored against movement before crossing",
            "Wheel restraints and positive dockboard securing were confirmed",
            "The bridge plate capacity exceeded the load and its anchors were engaged",
            "Employees inspected and locked the portable dockboard into position",
            "A secured dockboard spanned the gap with adequate bearing",
        ],
        "test_positive": [
            "The bridge plate crept sideways because it was never secured to the dock",
            "Powered equipment entered the railcar over a loose portable dockboard",
            "The dockboard slipped from its bearing surface during the crossing",
            "A plate with inadequate anchoring shifted beneath the loaded forklift",
        ],
        "test_negative": [
            "Positive restraints held the inspected dockboard throughout the transfer",
            "The crew verified capacity, bearing, and anchoring before forklift entry",
            "A fixed dock leveler remained locked in its operating position",
            "The portable plate was secured against displacement on both ends",
        ],
    },
    {
        "name": "fall_protection",
        "citation": "1910.28",
        "equipment": "Gantry Crane",
        "location": "Elevated Service Platform",
        "train_positive": [
            "An employee worked on the open platform edge without a guardrail or fall arrest",
            "The elevated catwalk had a missing rail and no personal fall protection",
            "A worker crossed an unprotected edge eight feet above the lower level",
            "Maintenance continued beside an open-sided platform without a protective system",
            "The access platform lacked both guardrails and an approved fall-restraint connection",
        ],
        "train_negative": [
            "A complete guardrail system protected every open side of the platform",
            "The worker remained connected to an approved fall-arrest anchorage",
            "Temporary rails and toe boards were inspected before access",
            "The elevated task stayed inside a compliant guarded work platform",
            "A travel-restraint system prevented the employee from reaching the edge",
        ],
        "test_positive": [
            "The missing midrail left an employee exposed at the elevated platform edge",
            "A technician stepped onto an open catwalk without fall protection",
            "Work proceeded beside a ten-foot drop with no guardrail or arrest system",
            "The platform gate was absent and the employee had no restraint connection",
        ],
        "test_negative": [
            "An inspected guardrail and self-closing gate enclosed the work area",
            "The employee's connected restraint prevented access to the fall edge",
            "A compliant scaffold platform provided full edge protection",
            "Work began only after temporary guardrails enclosed the elevated opening",
        ],
    },
]


def _query_text(row: Dict[str, Any]) -> str:
    return (
        f"Equipment: {row.get('Equipment_ID', '')} ({row.get('Equipment_Type', '')}) "
        f"at {row.get('Location', '')}. Shift: {row.get('Operator_Shift_Hours', '')} hrs. "
        f"Incident: {row.get('Reported_Incident', '')}."
    )


def build_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build deterministic, balanced train and holdout splits."""

    train: List[Dict[str, Any]] = []
    holdout: List[Dict[str, Any]] = []
    counters = {"train": 1, "holdout": 1}

    for scenario_index, scenario in enumerate(SCENARIOS):
        for split, destination in (("train", train), ("test", holdout)):
            for label_name, actual in (("positive", 1), ("negative", 0)):
                texts = scenario[f"{split}_{label_name}"]
                for text_index, incident in enumerate(texts):
                    split_name = "train" if split == "train" else "holdout"
                    row_number = counters[split_name]
                    counters[split_name] += 1
                    if scenario["name"] == "fatigue":
                        shift_hours = (
                            12.2 + 0.25 * text_index
                            if actual
                            else 10.5 + 0.3 * text_index
                        )
                    else:
                        shift_hours = 6.0 + 0.4 * ((scenario_index + text_index) % 8)

                    citation = scenario["citation"] if actual else "NONE"
                    destination.append(
                        {
                            "Log_ID": f"{'TR' if split_name == 'train' else 'HO'}-{row_number:04d}",
                            "Timestamp": f"2026-08-{1 + scenario_index:02d} {6 + text_index:02d}:00",
                            "Equipment_ID": f"EQ-{scenario_index + 1:02d}-{text_index + 1:02d}",
                            "Equipment_Type": scenario["equipment"],
                            "Location": scenario["location"],
                            "Operator_Shift_Hours": f"{shift_hours:.2f}",
                            "Payload_Weight_lbs": str(
                                12000 + scenario_index * 1000 + text_index * 250
                            ),
                            "Reported_Incident": incident,
                            "Is_Violation": str(actual),
                            "Violation_Type": scenario["name"] if actual else "none",
                            "Expected_Citation": citation,
                            "Reference_Reason": (
                                f"The recorded condition is governed by {scenario['citation']}."
                                if actual
                                else "The stated control removes the benchmark hazard condition."
                            ),
                        }
                    )

    train.sort(key=lambda row: row["Log_ID"])
    holdout.sort(key=lambda row: row["Log_ID"])
    return train, holdout


def dataset_fingerprint(rows: Sequence[Dict[str, Any]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_disjoint(
    train: Sequence[Dict[str, Any]], holdout: Sequence[Dict[str, Any]]
) -> None:
    train_ids = {row["Log_ID"] for row in train}
    holdout_ids = {row["Log_ID"] for row in holdout}
    if train_ids & holdout_ids:
        raise ValueError("Train and holdout Log_ID values overlap.")

    train_incidents = {row["Reported_Incident"].strip().lower() for row in train}
    holdout_incidents = {row["Reported_Incident"].strip().lower() for row in holdout}
    if train_incidents & holdout_incidents:
        raise ValueError("Train and holdout incident text overlaps.")


def balanced_subset(rows: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Select a deterministic round-robin subset across hazard locations and labels."""

    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row["Location"], row["Is_Violation"])
        buckets.setdefault(key, []).append(row)

    selected = []
    depth = 0
    ordered_keys = sorted(buckets)
    while len(selected) < min(limit, len(rows)):
        added = False
        for key in ordered_keys:
            bucket = buckets[key]
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) >= min(limit, len(rows)):
                    break
        if not added:
            break
        depth += 1
    return selected


def write_split(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def seed_trusted_memory(
    memory: EpisodicMemoryBank, rows: Sequence[Dict[str, Any]]
) -> None:
    """Load labeled training cases into memory with one batched embedding pass."""

    episodes = []
    for row in rows:
        citation = (
            None if row["Expected_Citation"] == "NONE" else row["Expected_Citation"]
        )
        verdict = "VIOLATION" if row["Is_Violation"] == "1" else "CLEAR"
        episodes.append(
            {
                "log_id": row["Log_ID"],
                "query_text": _query_text(row),
                "row": row,
                "verdict": verdict,
                "citation": citation,
                "reason": row["Reference_Reason"],
                "is_grounded": True,
                "confidence": 1.0,
                "critique_notes": None,
                "feedback_type": "verified",
                "verification_source": "ground_truth",
                "learning_reason": "Seeded from the labeled training split.",
                "correct_verdict": verdict,
                "correct_citation": citation,
                "run_id": "scaled_benchmark_train",
                "timestamp": 0.0,
            }
        )

    memory.episodes = episodes
    memory.vectors = embed_texts([episode["query_text"] for episode in episodes])


def build_prompt(
    row: Dict[str, Any],
    hits: Sequence[Dict[str, Any]],
    memory: EpisodicMemoryBank,
    use_memory: bool,
) -> str:
    context = "\n\n".join(
        f"[{chunk['citation']} -- {chunk['heading']}]\n{chunk['text'][:800]}"
        for chunk in hits
    )
    precedents_section = ""
    pitfalls_section = ""
    if use_memory:
        precedents = memory.retrieve_contrastive_precedents(_query_text(row))
        lines = []
        if precedents.get("positive"):
            item = precedents["positive"]
            lines.append(
                f"- Confirmed violation: {item['incident']} at {item['location']} -> "
                f"STATUS: {item['verdict']}, CITATION: {item['citation']} ({item['reason']})"
            )
        if precedents.get("negative"):
            item = precedents["negative"]
            lines.append(
                f"- Confirmed clear case: {item['incident']} at {item['location']} -> "
                f"STATUS: {item['verdict']}, CITATION: NONE ({item['reason']})"
            )
        if lines:
            precedents_section = (
                "TRUSTED TRAINING PRECEDENTS:\n" + "\n".join(lines) + "\n"
            )

    return AUTONOMOUS_LEARNING_PROMPT.format(
        context=context,
        precedents_section=precedents_section,
        pitfalls_section=pitfalls_section,
        log_id=row["Log_ID"],
        equipment=row["Equipment_ID"],
        equip_type=row["Equipment_Type"],
        location=row["Location"],
        shift_hours=row["Operator_Shift_Hours"],
        incident=row["Reported_Incident"],
    )


async def ask_model(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "seed": 42, "num_predict": 220},
    }
    async with semaphore:
        async with session.post(
            f"{OLLAMA}/api/chat", json=payload, timeout=600
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["message"]["content"]


async def evaluate_strategy(
    name: str,
    rows: Sequence[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    vectors: np.ndarray,
    memory: EpisodicMemoryBank,
    pillar_bank: AdaptivePillarBank,
    verifier: StatutoryGroundedVerifier,
    critic: SelfReflectionCritic,
    model: str,
    concurrency: int,
) -> Dict[str, Any]:
    if name == "hybrid":
        return evaluate_hybrid(rows, chunks, verifier)

    use_memory = name in {"trusted_memory", "trusted_memory_reflection"}
    use_reflection = name == "trusted_memory_reflection"
    semaphore = asyncio.Semaphore(concurrency)
    prepared = []

    for row in rows:
        hits = retrieve_triangulated_adaptive(row, chunks, vectors, pillar_bank)
        prompt = build_prompt(row, hits, memory, use_memory)
        prepared.append((row, hits, prompt))

    started = time.perf_counter()
    corrections = 0
    results = []
    async with aiohttp.ClientSession() as session:
        raw_outputs = await asyncio.gather(
            *(ask_model(session, prompt, model, semaphore) for _, _, prompt in prepared)
        )
        for (row, hits, prompt), raw_output in zip(prepared, raw_outputs):
            parsed = parse_response(raw_output)
            if use_reflection:
                parsed, was_corrected, _ = await critic.critique_and_reground_async(
                    session,
                    prompt,
                    parsed,
                    row,
                    hits,
                    semaphore,
                    record_preference=False,
                )
                corrections += int(was_corrected)

            citation = parsed.get("citation")
            retrieved_sections = {chunk["section"] for chunk in hits}
            expected_citation = row["Expected_Citation"]
            parsed.update(
                {
                    "log_id": row["Log_ID"],
                    "predicted": 1 if parsed.get("status") == "VIOLATION" else 0,
                    "actual": int(row["Is_Violation"]),
                    "actual_type": row["Violation_Type"],
                    "expected_citation": expected_citation,
                    "retrieved": sorted(retrieved_sections),
                    "citation_exists": (
                        None
                        if citation is None
                        else citation in verifier.valid_sections
                    ),
                    "citation_was_retrieved": (
                        None if citation is None else citation in retrieved_sections
                    ),
                    "citation_correct": (
                        citation == expected_citation
                        if expected_citation != "NONE"
                        else citation is None
                    ),
                }
            )
            results.append(parsed)

    metrics = compute_metrics(results)
    violations = [result for result in results if result["actual"] == 1]
    metrics.update(
        {
            "citation_correctness": round(
                sum(result["citation_correct"] for result in violations)
                / len(violations),
                4,
            ),
            "retrieval_recall": round(
                sum(
                    result["expected_citation"] in result["retrieved"]
                    for result in violations
                )
                / len(violations),
                4,
            ),
            "critique_corrections": corrections,
            "runtime_seconds": round(time.perf_counter() - started, 2),
        }
    )
    return {"metrics": metrics, "results": results}


def evaluate_hybrid(
    rows: Sequence[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    verifier: StatutoryGroundedVerifier,
) -> Dict[str, Any]:
    """Evaluate the scope-aware policy with exact authority selection."""

    chunks_by_section: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_section.setdefault(str(chunk.get("section", "")), []).append(chunk)

    started = time.perf_counter()
    results = []
    for row in rows:
        decision = evaluate_event(row)
        citation = decision.citation
        selected_chunks = chunks_by_section.get(citation or "", [])[:1]
        expected_citation = row["Expected_Citation"]
        result = decision.to_dict()
        result.update(
            {
                "log_id": row["Log_ID"],
                "predicted": (
                    1
                    if decision.status == "VIOLATION"
                    else 0
                    if decision.status == "CLEAR"
                    else None
                ),
                "actual": int(row["Is_Violation"]),
                "actual_type": row["Violation_Type"],
                "expected_citation": expected_citation,
                "retrieved": [chunk["section"] for chunk in selected_chunks],
                "citation_exists": None
                if citation is None
                else citation in verifier.valid_sections,
                "citation_was_retrieved": None
                if citation is None
                else bool(selected_chunks),
                "citation_correct": (
                    citation == expected_citation
                    if expected_citation != "NONE"
                    else citation is None and decision.status == "CLEAR"
                ),
            }
        )
        results.append(result)

    metric_results = []
    for result in results:
        metric_result = dict(result)
        if metric_result["predicted"] is None:
            metric_result["predicted"] = 1 - metric_result["actual"]
        metric_results.append(metric_result)
    metrics = compute_metrics(metric_results)
    violations = [result for result in results if result["actual"] == 1]
    review_count = sum(result["status"] == "REVIEW" for result in results)
    metrics.update(
        {
            "citation_correctness": round(
                sum(result["citation_correct"] for result in violations)
                / len(violations),
                4,
            ),
            "retrieval_recall": round(
                sum(
                    result["expected_citation"] in result["retrieved"]
                    for result in violations
                )
                / len(violations),
                4,
            ),
            "review_count": review_count,
            "automation_coverage": round(1 - review_count / len(results), 4),
            "critique_corrections": 0,
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }
    )
    return {"metrics": metrics, "results": results}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> List[float]:
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def exact_mcnemar(
    left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    left_only = 0
    right_only = 0
    for left_result, right_result in zip(left, right):
        left_correct = left_result["predicted"] == left_result["actual"]
        right_correct = right_result["predicted"] == right_result["actual"]
        left_only += int(left_correct and not right_correct)
        right_only += int(right_correct and not left_correct)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(left_only, right_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p": float(f"{p_value:.12g}"),
    }


def model_metadata(model: str) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{OLLAMA}/api/show",
            json={"model": model},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        details = payload.get("details", {})
        return {
            "name": model,
            "digest": payload.get("digest"),
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
        }
    except Exception as exc:
        return {"name": model, "metadata_error": str(exc)}


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    train, holdout = build_dataset()
    assert_disjoint(train, holdout)
    if args.limit:
        holdout = balanced_subset(holdout, args.limit)

    if args.write_datasets:
        write_split(TRAIN_FILE, train)
        write_split(HOLDOUT_FILE, holdout)

    needs_llm = any(strategy != "hybrid" for strategy in args.strategies)
    if needs_llm:
        chunks, vectors, corpus_sections = load_corpus_index()
    else:
        corpus_document = json.loads(
            (JSON_DIR / "regulations.json").read_text(encoding="utf-8")
        )
        chunks = list(corpus_document["chunks"])
        corpus_sections = set(corpus_document.get("sections", []))
        statutes_file = JSON_DIR / "statutes.json"
        if statutes_file.exists():
            statutes_document = json.loads(statutes_file.read_text(encoding="utf-8"))
            chunks.extend(statutes_document.get("chunks", []))
            corpus_sections.update(statutes_document.get("sections", []))
        vectors = np.empty((0, 0), dtype=np.float32)
    expected_sections = {
        row["Expected_Citation"]
        for row in train + holdout
        if row["Expected_Citation"] != "NONE"
    }
    missing_sections = sorted(expected_sections - corpus_sections)
    if missing_sections:
        raise ValueError(
            f"Benchmark citations missing from the corpus: {missing_sections}"
        )

    verifier = StatutoryGroundedVerifier()
    critic = SelfReflectionCritic(verifier, model=args.model)

    with tempfile.TemporaryDirectory(
        prefix="industrial-safety-benchmark-"
    ) as temporary:
        temporary_path = Path(temporary)
        memory = EpisodicMemoryBank(
            memory_file=temporary_path / "memory.json",
            vec_file=temporary_path / "memory.npz",
        )
        if needs_llm:
            seed_trusted_memory(memory, train)
        pillar_bank = AdaptivePillarBank(state_file=temporary_path / "pillars.json")

        strategies: Dict[str, Any] = {}
        for strategy in args.strategies:
            print(
                f"Running {strategy} on {len(holdout)} holdout cases with {args.model}...",
                flush=True,
            )
            strategies[strategy] = await evaluate_strategy(
                name=strategy,
                rows=holdout,
                chunks=chunks,
                vectors=vectors,
                memory=memory,
                pillar_bank=pillar_bank,
                verifier=verifier,
                critic=critic,
                model=args.model,
                concurrency=args.concurrency,
            )

    for strategy in strategies.values():
        correct = (
            strategy["metrics"]["true_positives"]
            + strategy["metrics"]["true_negatives"]
        )
        strategy["metrics"]["accuracy_95pct_wilson"] = wilson_interval(
            correct, len(holdout)
        )

    comparisons = {}
    if "static" in strategies and "trusted_memory" in strategies:
        comparisons["static_vs_trusted_memory"] = exact_mcnemar(
            strategies["static"]["results"],
            strategies["trusted_memory"]["results"],
        )
    if "static" in strategies and "trusted_memory_reflection" in strategies:
        comparisons["static_vs_trusted_memory_reflection"] = exact_mcnemar(
            strategies["static"]["results"],
            strategies["trusted_memory_reflection"]["results"],
        )

    corpus = json.loads((JSON_DIR / "regulations.json").read_text(encoding="utf-8"))
    report = {
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark": "scaled_disjoint_scope_aware_v2",
            "random_seed": 42,
            "temperature": 0.0,
            "train_cases": len(train),
            "holdout_cases": len(holdout),
            "train_fingerprint_sha256": dataset_fingerprint(train),
            "holdout_fingerprint_sha256": dataset_fingerprint(holdout),
            "exact_incident_overlap": 0,
            "model": (
                model_metadata(args.model)
                if needs_llm
                else {
                    "name": args.model,
                    "used": False,
                    "note": "The hybrid-only run made no model calls.",
                }
            ),
            "embedding_model": EMBED_MODEL,
            "corpus_edition": corpus.get("edition"),
            "corpus_chunks": len(chunks),
            "corpus_sections": len(corpus_sections),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "training_method": "trusted episodic in-context memory plus scope-aware policy control",
            "weight_fine_tuning_performed": False,
            "weight_fine_tuning_note": (
                "The host reported no usable CUDA device. This run evaluates trusted "
                "episodic learning and does not claim a LoRA or DPO weight update."
            ),
        },
        "strategies": strategies,
        "comparisons": comparisons,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the scaled trusted-memory benchmark."
    )
    parser.add_argument("--model", default=CHAT_MODEL)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("hybrid", "static", "trusted_memory", "trusted_memory_reflection"),
        default=("hybrid", "static", "trusted_memory"),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-datasets", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    report = asyncio.run(run_benchmark(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMeasured holdout results")
    for name, payload in report["strategies"].items():
        metrics = payload["metrics"]
        print(
            f"  {name:28} accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} citation_correctness={metrics['citation_correctness']:.4f} "
            f"seconds={metrics['runtime_seconds']:.2f}"
        )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
