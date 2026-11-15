"""Run a small, live tour of the safety backend without training or downloads."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import platform
import sys
import time
from urllib.parse import urlsplit

# A normal tour leaves the checkout, model files, and learned memory alone.
sys.dont_write_bytecode = True
try:
    from autonomous_investigator import (
        ApplicabilityReranker,
        AutonomousInvestigator,
        GroundedDecision,
        HybridRetriever,
        Observation,
        OllamaGroundedReasoner,
        QueryPlanner,
        TriggerEngine,
    )
    from hybrid_policy import evaluate_event
    from learning_policy import assess_for_learning
    from investigation_support import (
        FocusedQueryPlanner,
        build_investigation_support,
        focused_queries,
        relevant_excerpt,
    )
    from ollama_config import ollama_base_url
    from wording_retrieval import WordingRetriever, PROFILE, wording_excerpt
    from citation_alignment import InteractiveInvestigator, citation_alignment
except ModuleNotFoundError as error:
    raise SystemExit(
        f"Missing dependency: {error.name}. Run this Python with "
        "-m pip install -r requirements-showcase.txt"
    ) from error

ROOT = Path(__file__).resolve().parents[1]
MAX_OBSERVATIONS = 6
MAX_INPUT_BYTES = 65536
MAX_TEXT_CHARS = 4000
MAX_BENCHMARK_CASES = 64
TOP_K = 16


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_process_memory_mib() -> float | None:
    """Peak resident memory for this Python process, not an Ollama server."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class MemoryCounters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t)
                    for name in (
                        "peak_working_set",
                        "working_set",
                        "peak_paged_pool",
                        "paged_pool",
                        "peak_nonpaged_pool",
                        "nonpaged_pool",
                        "pagefile",
                        "peak_pagefile",
                    )
                ]

            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            query = ctypes.WinDLL("psapi").GetProcessMemoryInfo
            query.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(MemoryCounters),
                wintypes.DWORD,
            ]
            query.restype = wintypes.BOOL
            if not query(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
                return None
            return round(counters.peak_working_set / 2**20, 2)
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (2**20 if sys.platform == "darwin" else 1024), 2)
    except (ImportError, OSError, AttributeError):
        return None


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_corpus(root: Path = ROOT) -> list[dict]:
    regulations = read_json(root / "json/regulations.json")
    statutes = read_json(root / "json/statutes.json")
    return regulations["chunks"] + statutes["chunks"]


def load_observations(path: Path) -> list[Observation]:
    with path.open("rb") as handle:
        raw = handle.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Observation input is limited to 64 KiB per run")
    payload = json.loads(raw.decode("utf-8-sig"))
    values = payload if isinstance(payload, list) else [payload]
    if not 1 <= len(values) <= MAX_OBSERVATIONS:
        raise ValueError("Provide between one and six observations")
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("Each observation must be a JSON object")
    for value in values:
        if len(json.dumps(value, allow_nan=False)) > 10000:
            raise ValueError(
                "Each structured observation is limited to 10,000 characters"
            )
        if not str(value.get("summary", "")).strip():
            raise ValueError("Each observation needs a non-empty summary")
    return [Observation.from_mapping(value) for value in values]


def text_observation(text: str) -> Observation:
    if not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise ValueError("Incident text must contain 1 to 4,000 characters")
    return Observation.from_mapping(
        {
            "event_id": "visitor-incident",
            "source_id": "visitor-text",
            "summary": text.strip(),
            "human_flag": True,
        }
    )


class OneRequestReasoner:
    """Allow one optional model decision, without retries or model downloads."""

    def __init__(self, model: str, timeout: int):
        host = ollama_base_url()
        hostname = urlsplit(host).hostname
        if hostname != "localhost":
            try:
                local = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                local = False
            if not local:
                raise ValueError(
                    "The showcase only sends observations to a loopback Ollama address"
                )
        self.delegate = OllamaGroundedReasoner(
            model=model, host=host, timeout=timeout, num_ctx=4096
        )
        self.calls = 0

    def decide(self, observation, candidates):
        if self.calls:
            return GroundedDecision(
                "REVIEW",
                None,
                "The one-request model budget for this tour has been used.",
                "showcase_budget",
            )
        self.calls += 1
        return self.delegate.decide(observation, candidates)


def make_investigator(chunks: list[dict], reasoner=None, *, semantic_index=None, retrieval_profile="reference"):
    if retrieval_profile not in {"reference", "wording"}:
        raise ValueError("Choose reference or wording retrieval")
    if semantic_index is not None and retrieval_profile != "wording":
        raise ValueError("Semantic retrieval requires the explicit wording profile")
    wording = WordingRetriever(chunks, semantic_index=semantic_index)
    investigator = InteractiveInvestigator(
        wording if retrieval_profile == "wording" else HybridRetriever(chunks),
        trigger_engine=TriggerEngine(sample_rate=0.0),
        query_planner=FocusedQueryPlanner(),
        policy_fn=evaluate_event,
        reasoner=reasoner,
        candidate_limit=TOP_K,
    )
    investigator.wording_retriever = wording
    investigator.retrieval_profile = retrieval_profile
    return investigator


def attach_excerpt(candidate: dict, chunks: list[dict], query_text: str) -> None:
    """Attach a verbatim passage from this retrieved authority, with provenance."""
    choices = []
    for index, chunk in enumerate(chunks):
        if (str(chunk.get("section", "")) == candidate["section"]
                and str(chunk.get("citation", "")) == candidate["citation"]):
            excerpt = wording_excerpt(str(chunk.get("text", "")), query_text)
            choices.append((excerpt["score"], -index, excerpt, index))
    if not choices:
        index = candidate["chunk_index"]
        excerpt = wording_excerpt(str(chunks[index]["text"]), query_text)
    else:
        _, _, excerpt, index = max(choices, key=lambda item: item[:2])
    candidate.update(excerpt=excerpt["text"], excerpt_start=excerpt["start"],
                     excerpt_end=excerpt["end"], excerpt_chunk_index=index,
                     excerpt_chunk_sha256=hashlib.sha256(str(chunks[index]["text"]).encode("utf-8")).hexdigest())


def investigate(observation: Observation, investigator, chunks: list[dict]) -> dict:
    result = investigator.investigate(observation).to_dict()
    # A retrieved section is a research lead, not proof that it governs the event.
    query_text = " ".join(result["queries"])
    for candidate in result["candidates"]:
        attach_excerpt(candidate, chunks, query_text)
    # The combined ranks remain visible. Separate leads explain which existing
    # candidate best matches each focused search, without adding authorities.
    available = {item["section"]: (rank, item) for rank, item in enumerate(result["candidates"], 1)}
    leads = []
    if result["queries"]:
        for query in focused_queries(observation)[:3]:
            for item in investigator.retriever.search([query], top_k=TOP_K):
                if item.section in available:
                    rank, candidate = available[item.section]
                    lead = dict(candidate, research_query=query, combined_candidate_rank=rank)
                    attach_excerpt(lead, chunks, query)
                    if not any(old["section"] == lead["section"] for old in leads):
                        leads.append(lead)
                    break
    result["evidence_leads"] = leads
    diagnostics = investigator.wording_retriever
    result["citation_alignment"] = (citation_alignment(observation.summary, result["candidates"], diagnostics)
        if len(observation.summary) <= MAX_TEXT_CHARS else {
            "status": "research_query_limit", "highest_ranked_research_match": None,
            "accepted_as_correct": False, "governing_citation_established": False,
            "scope": "The structured report exceeds the wording research limit. Primary reference results remain available."
        })
    # Expanded search stays separate until it earns a primary-ranking release.
    # These candidates do not enter the incident decision above.
    additional = []
    research_query_fits = all(len(query) <= MAX_TEXT_CHARS for query in result["queries"])
    if investigator.retrieval_profile == "reference" and result["queries"] and research_query_fits:
        expanded = investigator.reranker.rerank(
            observation, diagnostics.search(result["queries"], top_k=TOP_K))
        primary_top = {item["section"] for item in result["candidates"][:3]}
        for candidate in expanded[:5]:
            if candidate.section not in primary_top and len(additional) < 3:
                item = candidate.to_dict()
                attach_excerpt(item, chunks, query_text)
                item["match_basis"] = diagnostics.match_diagnostics(observation.summary, item)
                item["accepted_as_correct"] = False
                additional.append(item)
    result["additional_wording_evidence"] = additional
    result["additional_wording_status"] = "within_query_limit" if research_query_fits else "skipped_query_limit"
    result["summary"] = observation.summary
    result["source_id"] = observation.source_id
    result["observation_context"] = {
        "actors": observation.actors,
        "equipment": observation.equipment,
        "location": observation.location,
        "actions": observation.actions,
        "conditions": observation.conditions,
        "controls": observation.controls,
        "measurements": observation.measurements,
        "relations": [relation.sentence() for relation in observation.relations],
    }
    result["support"] = build_investigation_support(
        observation, result["status"], result["candidates"],
        decision_source=result["decision_source"],
    )
    if result["decision_source"] == "information_request":
        result["support"]["first_assessment"] = (
            "This request asks which safety requirements may be relevant. Inspect the matching passages below; any practical precautions are conditional on an actual exposure."
        )
    result["retrieval_profile"] = (PROFILE if investigator.retrieval_profile == "wording"
                                   else "reference_primary_with_wording_research_v1")
    result["citation_in_candidates"] = (
        result["citation"] in {item["section"] for item in result["candidates"]}
        if result["citation"]
        else None
    )
    return result


def benchmark_sample(size: int, root: Path = ROOT) -> list[dict]:
    if not 0 <= size <= MAX_BENCHMARK_CASES:
        raise ValueError("The showcase benchmark accepts 0 to 64 cases")
    groups = defaultdict(list)
    with (root / "daily_yard_log.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Is_Violation"] == "1":
                groups[row["Violation_Type"]].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["Log_ID"])
    selected = []
    for offset in range(max(map(len, groups.values()), default=0)):
        for hazard in sorted(groups):
            if offset < len(groups[hazard]) and len(selected) < size:
                selected.append(groups[hazard][offset])
    return selected


def benchmark_observation(row: dict) -> Observation:
    # Labels choose a varied sample and score retrieval, but never form queries.
    return Observation.from_mapping(
        {
            "event_id": row["Log_ID"],
            "source_id": row["Equipment_ID"],
            "timestamp": row["Timestamp"],
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
            "changed": True,
        }
    )


def run_benchmark(size: int, retriever, root: Path = ROOT) -> dict:
    planner, reranker = QueryPlanner(), ApplicabilityReranker()
    records = []
    for row in benchmark_sample(size, root):
        observation = benchmark_observation(row)
        queries = planner.plan(observation)
        candidates = reranker.rerank(
            observation, retriever.search(queries, top_k=TOP_K)
        )
        rank = next(
            (
                i
                for i, item in enumerate(candidates, 1)
                if item.section == row["Expected_Section"]
            ),
            None,
        )
        records.append(
            {
                "event_id": observation.event_id,
                "hazard": row["Violation_Type"],
                "expected_section": row["Expected_Section"],
                "rank": rank,
            }
        )
    return {
        "source": "live_computation_on_saved_synthetic_inputs",
        "retrieval": "lexical_with_applicability_reranking",
        "query_planner": "published_reference_query_planner",
        "cases": len(records),
        "hazard_families": len({item["hazard"] for item in records}),
        "recall_at_16": (
            sum(item["rank"] is not None for item in records) / len(records)
            if records
            else None
        ),
        "recall_at_1": sum(item["rank"] is not None and item["rank"] <= 1 for item in records) / len(records) if records else None,
        "recall_at_5": sum(item["rank"] is not None and item["rank"] <= 5 for item in records) / len(records) if records else None,
        "labels_used_for_sample_selection": True,
        "labels_used_to_form_queries": False,
        "dataset_sha256": file_hash(root / "daily_yard_log.csv"),
        "records": records,
        "note": "A small, known synthetic sample using the published reference query planner. This separate benchmark does not test the focused interactive query expansions or establish legal accuracy or field performance.",
    }


def learning_examples() -> list[dict]:
    cases = [
        ("Unreviewed prediction", {}, "1910.184", True),
        (
            "Matching synthetic label",
            {"Is_Violation": 1, "Expected_Section": "1910.184"},
            "1910.184",
            True,
        ),
        (
            "Simulated approved review",
            {
                "Review_Status": "approved",
                "Reviewed_Verdict": "VIOLATION",
                "Reviewed_Citation": "1910.184",
            },
            "1910.184",
            True,
        ),
        (
            "Citation conflicts with the label",
            {"Is_Violation": 1, "Expected_Section": "1910.184"},
            "1910.22",
            True,
        ),
    ]
    return [
        {
            "example": name,
            **assess_for_learning(row, "VIOLATION", citation, grounded).to_dict(),
        }
        for name, row, citation, grounded in cases
    ]


def grounding_example(chunks: list[dict]) -> dict:
    restricted = [item for item in chunks if item["section"] == "1910.22"]
    observation = text_observation(
        "Crane boom moved four feet inside an energized overhead conductor approach area."
    )
    result = make_investigator(restricted).investigate(observation).to_dict()
    return {
        "source": "live_controlled_negative_test",
        "setup": "Electrical authority is deliberately withheld; only housekeeping text is available.",
        "status": result["status"],
        "citation": result["citation"],
        "reason": result["reason"],
    }


def saved_evidence(root: Path = ROOT) -> list[dict]:
    specs = [
        (
            "1,000-event synthetic audit",
            "json/full_audit_policy_controlled.json",
            "controlled_metrics",
        ),
        (
            "Hybrid authority retrieval",
            "json/autonomous_retrieval_benchmark.json",
            "applicability_reranked",
        ),
        (
            "Authentic incident-text coverage",
            "json/authentic_incident_retrieval_hybrid.json",
            "coverage",
        ),
    ]
    entries = []
    for label, name, key in specs:
        payload = read_json(root / name)
        entries.append(
            {
                "label": label,
                "source": "saved_reference_not_rerun",
                "file": name,
                "sha256": file_hash(root / name),
                "metrics": payload[key],
            }
        )
    name = "json/finetune_evaluation_results.json"
    payload = read_json(root / name)
    entries.append(
        {
            "label": "QLoRA matched synthetic holdout",
            "source": "saved_reference_not_rerun",
            "file": name,
            "sha256": file_hash(root / name),
            "metrics": {
                "cases": payload["provenance"]["holdout_cases"],
                "base": payload["strategies"]["base"]["metrics"],
                "adapter": payload["strategies"]["qlora_sft"]["metrics"],
            },
        }
    )
    return entries


def run_showcase(observations, *, benchmark_cases=16, reasoner=None, root=ROOT, semantic_index=None, retrieval_profile="reference") -> dict:
    started = time.perf_counter()
    chunks = load_corpus(root)
    investigator = make_investigator(chunks, reasoner, semantic_index=semantic_index,
                                     retrieval_profile=retrieval_profile)
    results = [investigate(item, investigator, chunks) for item in observations]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "live_run": {
            "retrieval": "wording_aware_with_applicability_reranking" if retrieval_profile == "wording" else "reference_primary_with_additional_research",
            "retrieval_profile": PROFILE if retrieval_profile == "wording" else "reference_primary_with_wording_research_v1",
            "observations": results,
            "benchmark": run_benchmark(benchmark_cases, HybridRetriever(chunks), root),
            "grounding_check": grounding_example(chunks),
            "learning_gate_examples": learning_examples(),
        },
        "saved_evidence": saved_evidence(root),
        "runtime": {
            "seconds": round(time.perf_counter() - started, 3),
            "peak_python_process_mib": peak_process_memory_mib(),
            "python": platform.python_version(),
            "os": platform.system(),
            "model_requests": reasoner.calls if reasoner else 0,
            "semantic_retrieval": semantic_index.profile if semantic_index else None,
            "training_performed": False,
            "learning_memory_modified": False,
            "source_sha256": {
                name: file_hash(root / name)
                for name in ("code/showcase.py", "code/investigation_support.py",
                             "code/wording_retrieval.py", "code/citation_alignment.py", "code/evidence_embeddings.py",
                             "code/autonomous_investigator.py", "code/hybrid_policy.py",
                             "json/regulations.json", "json/statutes.json")
            },
            "measurement_scope": "Tour work after imports; peak resident memory of this Python process only. Ollama memory is excluded.",
        },
        "limits": {
            "max_observations": MAX_OBSERVATIONS,
            "max_benchmark_cases": MAX_BENCHMARK_CASES,
            "model_request_budget": 1 if reasoner else 0,
            "focused_evidence_searches": 3,
            "model_read_timeout_seconds": (
                reasoner.delegate.timeout if reasoner else None
            ),
        },
        "scope": [
            "Included observations and review approvals are synthetic examples, not live camera detections or real reviewer approvals.",
            "CLEAR applies to the checked condition. NO_ACTION means no trigger fired, not that the environment is safe.",
            "Retrieval scores and trigger scores are not calibrated probabilities of a violation.",
            "Focused search expansions are research hypotheses, not additional observed facts. Practical precautions do not establish a regulatory violation.",
            "Saved results came from earlier runs and were not recomputed by this tour. The corpus uses the pinned 2025-01-01 eCFR edition.",
        ],
    }


def render_observation(item: dict) -> str:
    lines = [
        f"\n{item['event_id']}: {item['summary']}",
    ]
    support = item.get("support", {})
    if support:
        lines.append("  Assessment: " + support["first_assessment"])
        for text in support["possible_concerns"]:
            lines.append("  Possible concern: " + text)
        for text in support["immediate_precautions"]:
            lines.append("  Practical next step: " + text)
        for text in support["evidence_notes"]:
            lines.append("  Evidence: " + text)
        for text in support["priority_clarifications"]:
            lines.append("  Useful follow-up: " + text)
    for lead in item.get("evidence_leads", []):
        lines.append(f"  Focused evidence lead: {lead['citation']} | {lead['heading']} (combined rank {lead['combined_candidate_rank']})")
    alignment = item.get("citation_alignment", {})
    closest = alignment.get("highest_ranked_research_match")
    if closest:
        lines.append(f"  Primary search lead (verify scope): {closest['citation']} | {closest['heading']}")
        lines.append("    " + closest["excerpt"])
    elif alignment:
        lines.append("  No supported citation match identified. The corpus may need another authority, or the request may need more context.")
    if alignment:
        lines.append("  " + alignment["scope"])
    for lead in item.get("additional_wording_evidence", []):
        lines.append(f"  Additional wording evidence to review: {lead['citation']} | {lead['heading']}")
        lines.append("    " + lead["excerpt"])
    lines.extend([
        f"  Automated decision: {item['status']} | {item['decision_source']}",
        f"  Decision reason: {item['reason']}",
        "  Trigger: " + (", ".join(item["trigger"]["reasons"]) or "none"),
    ])
    context = {key: value for key, value in item.get("observation_context", {}).items() if value}
    if context:
        lines.append("  Supplied context: " + json.dumps(context, ensure_ascii=False))
    if item["citation"]:
        lines.append(
            f"  Citation: {item['citation']} | present in retrieved set: {item['citation_in_candidates']}"
        )
    if item["queries"]:
        lines.append(f"  Query plan: {len(item['queries'])} evidence-led queries")
        lines.extend("    " + query for query in item["queries"][:2])
    for candidate in item["candidates"][:3]:
        lines.append(f"  Retrieved: {candidate['citation']} | {candidate['heading']}")
    return "\n".join(lines)


def render_report(report: dict) -> str:
    live, runtime = report["live_run"], report["runtime"]
    lines = [
        "Industrial Safety Agent | Project walkthrough",
        "This run uses the actual retrieval, trigger, policy, and grounding code.",
        "No training, model downloads, or learning-memory updates.",
        "\nLIVE OBSERVATIONS",
    ]
    lines.extend(render_observation(item) for item in live["observations"])
    check = live["grounding_check"]
    lines.extend(
        [
            "\nLIVE GROUNDING CHECK",
            check["setup"],
            f"  {check['status']}: {check['reason']}",
            "\nLIVE LEARNING-GATE CHECKS",
        ]
    )
    for item in live["learning_gate_examples"]:
        action = "eligible for trusted memory" if item["trusted"] else "not trusted"
        lines.append(f"  {item['example']}: {action} ({item['feedback_type']})")
    lines.append(
        "  These are simulated approvals. No episode is saved and no weights change."
    )
    benchmark = live["benchmark"]
    lines.append("\nLIVE SMALL-SAMPLE RETRIEVAL BENCHMARK")
    if benchmark["cases"]:
        lines.append(
            f"  {benchmark['cases']} cases, {benchmark['hazard_families']} hazard families, "
            f"recall@16: {benchmark['recall_at_16']:.1%}"
        )
    else:
        lines.append("  Skipped for this run.")
    lines.extend(["  " + benchmark["note"], "\nSAVED EVIDENCE | NOT RERUN"])
    for evidence in report["saved_evidence"]:
        m = evidence["metrics"]
        if "adapter" in m:
            detail = (
                f"{m['cases']} cases; base/adapter strict accuracy "
                f"{m['base']['accuracy']:.1%}/{m['adapter']['accuracy']:.1%}; "
                f"adapter citation correctness {m['adapter']['citation_correctness']:.1%}"
            )
        elif "recall_at_16" in m:
            detail = f"{m['cases']} violations; recall@16 {m['recall_at_16']:.1%}"
        elif "nonempty_retrieval_rate" in m:
            detail = f"{m['cases']} known incidents; non-empty retrieval {m['nonempty_retrieval_rate']:.1%} (coverage, not legal accuracy)"
        else:
            detail = f"{m['n']} events; precision {m['precision']:.1%}; recall {m['recall']:.1%}; F1 {m['f1']:.4f}"
        lines.extend(
            [f"  {evidence['label']}: {detail}", f"    Source: {evidence['file']}"]
        )
    lines.append(
        f"\nFinished in {runtime['seconds']:.3f}s. Model requests: {runtime['model_requests']}."
    )
    lines.extend("  " + note for note in report["scope"])
    return "\n".join(lines)


def export_path(requested: Path) -> Path:
    path = requested.resolve()
    directory = (ROOT / "benchmark_runs/showcase").resolve()
    if not path.is_relative_to(directory) or path.suffix.lower() != ".json":
        raise ValueError(
            "Exports must be new .json files under benchmark_runs/showcase/"
        )
    if path.exists():
        raise ValueError(f"The export already exists: {path}")
    return path


def interactive_session(root=ROOT, *, retrieval_profile="reference", semantic_index=None):
    chunks = load_corpus(root)
    investigator = make_investigator(chunks, retrieval_profile=retrieval_profile, semantic_index=semantic_index)
    print(
        "\nEnter an incident to investigate. Type quit to finish. Nothing is saved. "
        + ("Local semantic requests are enabled." if semantic_index else "This session stays offline.")
    )
    while True:
        try:
            text = input("Incident> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.strip().lower() in {"quit", "exit"}:
            break
        try:
            result = investigate(text_observation(text), investigator, chunks)
            print(render_observation(result))
        except ValueError as error:
            print(error)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--text", help="Investigate your own incident text (up to 4,000 characters)"
    )
    inputs.add_argument(
        "--input", type=Path, help="One to six structured observation objects in JSON"
    )
    parser.add_argument(
        "--benchmark-cases", type=int, default=16, help="0 to 64; default 16"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report, including excerpts, as JSON",
    )
    parser.add_argument(
        "--out", type=Path, help="Save a new JSON report under benchmark_runs/showcase/"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After the tour, enter incident text offline",
    )
    parser.add_argument(
        "--reasoner",
        action="store_true",
        help="Allow one local Ollama request for an unresolved observation",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:9b",
        help="Already-installed Ollama model; no automatic pulls",
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Model read timeout, 1 to 60 seconds"
    )
    parser.add_argument("--semantic-index", type=Path,
                        help="Explicit frozen local passage index; enables local embedding requests")
    parser.add_argument("--embedding-digest",
                        help="Required with --semantic-index: SHA-256 of the installed embedding model")
    parser.add_argument("--retrieval-profile", choices=("reference", "wording"), default="reference",
                        help="reference keeps the primary ranking; wording tests the expanded ranker")
    args = parser.parse_args()
    try:
        if not 0 <= args.benchmark_cases <= MAX_BENCHMARK_CASES:
            raise ValueError("--benchmark-cases must be between 0 and 64")
        if not 1 <= args.timeout <= 60:
            raise ValueError("--timeout must be between 1 and 60 seconds")
        if args.interactive and args.json:
            raise ValueError("Use --interactive or --json, not both")
        if bool(args.semantic_index) != bool(args.embedding_digest):
            raise ValueError("Use --semantic-index and --embedding-digest together")
        if args.semantic_index and args.retrieval_profile != "wording":
            raise ValueError("--semantic-index requires --retrieval-profile wording")
        output = export_path(args.out) if args.out else None
        observations = (
            [text_observation(args.text)]
            if args.text is not None
            else load_observations(
                args.input or ROOT / "data/showcase_observations.json"
            )
        )
        reasoner = (
            OneRequestReasoner(args.model, args.timeout) if args.reasoner else None
        )
        semantic_index = None
        if args.semantic_index:
            from evidence_embeddings import FrozenSemanticIndex, OllamaEmbedder
            embedder = OllamaEmbedder(model_digest=args.embedding_digest, base_url=ollama_base_url())
            semantic_index = FrozenSemanticIndex.load(args.semantic_index, load_corpus(), embedder)
        print("Running the local walkthrough...", file=sys.stderr, flush=True)
        report = run_showcase(
            observations, benchmark_cases=args.benchmark_cases, reasoner=reasoner,
            semantic_index=semantic_index,
            retrieval_profile=args.retrieval_profile,
        )
        rendered = json.dumps(report, indent=2, allow_nan=False)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
            print(f"Saved {output}", file=sys.stderr)
        print(rendered if args.json else render_report(report))
        if args.interactive:
            interactive_session(retrieval_profile=args.retrieval_profile, semantic_index=semantic_index)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Showcase could not run: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
