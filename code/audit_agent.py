"""Retrieval-grounded compliance auditor over intermodal yard logs.

How it works
------------
Flattening an event into one search query lets equipment tokens ("Gantry Crane")
dominate the embedding and bury the rule that actually applies. So retrieval runs
across a fixed set of hazard pillars plus a query built from the event, and the
results are unioned. Every citation the model returns is then checked three ways:
does the section exist in Titles 29 or 49, was it in the retrieved context, and is
it the rule the ground truth records as governing that hazard.

Measured state (1,000 events, 144 violations across 16 hazard types)
--------------------------------------------------------------------
  Precision 0.963, recall 0.715, F1 0.821.
  107 citations, none fabricated, none ungrounded, 57.3% naming the correct rule.
  Retrieval recall@4 is 0.26 across 16 hazard types; reaching 0.72 takes k=8.

Detection is roughly independent of retrieval rank (r = +0.09) while citation
correctness tracks it (r = -0.53). Spotting an unsafe event is easy; naming the
governing rule is what retrieval decides.

The keyword pre-filter is off by default. Measured against realistic incident
phrasing its recall is 0.27, so it cannot sit in the measurement path; --triage
re-enables it as a cost control and reports its own numbers separately.

    python code/audit_agent.py                   # audit full daily_yard_log.csv
    python code/audit_agent.py --limit 50        # first 50 records
    python code/audit_agent.py --no-retrieval    # ungrounded baseline
    python code/audit_agent.py --triage          # cheap keyword gate, lossy

Requires Ollama running locally (https://ollama.com) with mxbai-embed-large and
qwen3.5:9b pulled. Nothing leaves the machine.
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

import aiohttp
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
REGS = ROOT / "json" / "regulations.json"
STATUTES = ROOT / "json" / "statutes.json"
EMBED_CACHE = ROOT / "json" / "reg_embeddings.npz"
LOG = ROOT / "daily_yard_log.csv"
RESULTS = ROOT / "json" / "audit_results.json"
VALID_SECTIONS = ROOT / "json" / "valid_sections.json"

OLLAMA = "http://localhost:11434"

# Allow slower CPU hosts to override the inference timeout.
REQUEST_TIMEOUT = int(os.getenv("AUDIT_TIMEOUT", "900"))
MAX_ATTEMPTS = int(os.getenv("AUDIT_RETRIES", "3"))

_PROGRESS = {"done": 0, "total": 0, "failed": 0}
TEMPERATURE = 0
NUM_PREDICT = 180

EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen3.5:9b"

# Ollama otherwise allocates the model's full context window (262144 for
# qwen3.5:9b), which reserves roughly 15GB of KV cache and pushes the model
# onto CPU. Audit prompts run to about 1,500 tokens.
NUM_CTX = int(os.getenv("AUDIT_NUM_CTX", "8192"))
TOP_K = 4

# Compliance dimensions used for multi-hazard triangulation
HAZARD_PILLARS = {
    "fatigue": "49 U.S.C. 21103 limitations on duty hours maximum 12 consecutive hours train employee covered service",
    "housekeeping": "29 CFR 1910.22 walking-working surfaces housekeeping floors passageways clean dry spills leaks hazards",
    "clearance": "29 CFR 1910.333 electrical safety related work practices minimum approach distance energized high-voltage line clearance",
    "equipment": "29 CFR 1910.178 1910.179 powered industrial trucks overhead gantry cranes safe operation and maintenance",
}

AUTONOMOUS_PROMPT = """You are an expert federal safety and compliance auditor for a rail-served intermodal yard.

Below are excerpts from federal safety regulations (OSHA 29 CFR and FRA 49 CFR) retrieved for this yard event.
These are the ONLY regulations you may cite. Do not cite any section number not present in these excerpts. If no violation exists, cite NONE.

RETRIEVED REGULATIONS
{context}

YARD EVENT
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

UNGROUNDED_PROMPT = """You are a safety and compliance auditor for a rail-served intermodal yard.

Cross-reference this yard event with your knowledge of OSHA and FRA regulatory standards.

YARD EVENT
- Log ID: {log_id}
- Equipment: {equipment} ({equip_type})
- Location: {location}
- Operator shift hours: {shift_hours}
- Reported incident: {incident}

Decide whether this event violates a federal safety regulation. Consider operator
hours-of-service limits, hazardous spills where people walk, and equipment
operating close to energized high-voltage equipment.

Answer in exactly this format and nothing else:
STATUS: CLEAR or VIOLATION
CITATION: the specific CFR section number, or NONE
REASON: one sentence
"""


def embed(texts, batch=32, model=None):
    model = model or EMBED_MODEL
    out = []
    for i in range(0, len(texts), batch):
        r = requests.post(
            f"{OLLAMA}/api/embed",
            json={"model": model, "input": texts[i : i + batch]},
            timeout=600,
        )
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
        print(
            f"\r  embedded {min(i + batch, len(texts))}/{len(texts)}",
            end="",
            flush=True,
        )
    print()
    v = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def real_sections():
    if not VALID_SECTIONS.exists():
        raise SystemExit("Run code/build_section_index.py first.")
    d = json.loads(VALID_SECTIONS.read_text(encoding="utf-8"))
    sections = {s for group in d["by_title"].values() for s in group}
    if STATUTES.exists():
        sections.update(
            json.loads(STATUTES.read_text(encoding="utf-8")).get("sections", [])
        )
    return sections


def load_index():
    corpus = json.loads(REGS.read_text(encoding="utf-8"))
    chunks = list(corpus["chunks"])
    sections = set(corpus["sections"])
    if STATUTES.exists():
        statutes = json.loads(STATUTES.read_text(encoding="utf-8"))
        chunks.extend(statutes.get("chunks", []))
        sections.update(statutes.get("sections", []))

    if EMBED_CACHE.exists():
        cached = np.load(EMBED_CACHE)
        if cached["vectors"].shape[0] == len(chunks):
            return chunks, cached["vectors"], sections
        print("Corpus changed since the cache was built, re-embedding.")

    print(f"Embedding {len(chunks)} regulation chunks with {EMBED_MODEL}...")
    vectors = embed([f"{c['citation']} {c['heading']}. {c['text']}" for c in chunks])
    np.savez_compressed(EMBED_CACHE, vectors=vectors)
    return chunks, vectors, sections


def get_pillar_vectors(model=None):
    keys = list(HAZARD_PILLARS.keys())
    vecs = embed([HAZARD_PILLARS[k] for k in keys], model=model)
    return dict(zip(keys, vecs))


def retrieve_triangulated(row, chunks, vectors, pillar_vecs, top_dyn=2):
    selected_indices = []

    # 1. Coverage across compliance pillars
    for key, p_vec in pillar_vecs.items():
        scores = vectors @ p_vec
        best_idx = int(np.argmax(scores))
        if best_idx not in selected_indices:
            selected_indices.append(best_idx)

    # 2. Dynamic event-specific query
    event_query = (
        f"{row['Equipment_Type']} operating at {row['Location']}, "
        f"operator on shift {row['Operator_Shift_Hours']} hours, "
        f"incident reported: {row['Reported_Incident']}"
    )
    q_vec = embed([event_query])[0]
    scores = vectors @ q_vec
    order = np.argsort(-scores)
    for idx in order:
        if idx not in selected_indices:
            selected_indices.append(int(idx))
        if len(selected_indices) >= len(pillar_vecs) + top_dyn:
            break

    return [chunks[i] for i in selected_indices]


def retrieve_flat(query, chunks, vectors, k=TOP_K):
    q = embed([query])[0]
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [chunks[i] for i in top]


def build_query(row, strategy="triangulated"):
    event = (
        f"{row['Equipment_Type']} operating at {row['Location']}, "
        f"operator on shift {row['Operator_Shift_Hours']} hours, "
        f"incident reported: {row['Reported_Incident']}"
    )
    if strategy == "raw":
        return event
    if strategy == "expanded":
        terms = ["occupational safety requirement", "required work practice"]
        loc = str(row.get("Location", "")).lower()
        if row.get("Operator_Shift_Hours"):
            terms += [
                "hours of duty",
                "maximum consecutive hours",
                "fatigue limit 12 hours",
            ]
        if any(w in loc for w in ("pedestrian", "crosswalk", "walkway")):
            terms += ["walking working surfaces", "housekeeping", "spill leak hazard"]
        if any(w in loc for w in ("voltage", "electrical", "line")):
            terms += ["energized high voltage", "approach distance", "clearance"]
        return f"{event}. {' '.join(terms)}"
    return event


def parse(text):
    status = re.search(r"STATUS:\s*(VIOLATION|CLEAR)", text, re.I)
    citation = re.search(r"CITATION:\s*([^\n]+)", text, re.I)
    reason = re.search(r"REASON:\s*([^\n]+)", text, re.I)
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


def is_candidate_hazard(row):
    """Cheap keyword gate for routine events. Off by default.

    Its measured recall is 0.27 on the current benchmark, so it remains an
    explicit cost-control option and is excluded from default measurement.
    """
    try:
        shift_hrs = float(row["Operator_Shift_Hours"])
    except (ValueError, TypeError):
        shift_hrs = 0.0
    loc = str(row.get("Location", ""))
    inc = str(row.get("Reported_Incident", ""))

    # Check if event has hazard telemetry or shift threshold breach
    return (
        shift_hrs > 12.0
        or inc
        in (
            "Hydraulic Leak",
            "Proximity Warning",
            "Load Imbalance",
            "Tire Pressure Warning",
        )
        or "Crosswalk" in loc
        or "High-Voltage" in loc
    )


async def audit_async(
    rows,
    chunks,
    vectors,
    sections,
    grounded=True,
    strategy="triangulated",
    concurrency=6,
    use_triage=False,
    top_k=TOP_K,
):
    pillar_vecs = (
        get_pillar_vectors() if (grounded and strategy == "triangulated") else None
    )
    sem = asyncio.Semaphore(concurrency)

    llm_tasks = []
    results = [None] * len(rows)

    for idx, row in enumerate(rows):
        # Opt-in keyword gate: skip routine rows without a model call.
        if use_triage and not is_candidate_hazard(row):
            results[idx] = {
                "log_id": row["Log_ID"],
                "status": "CLEAR",
                "audited_by": "triage",
                "citation_raw": "NONE",
                "citation": None,
                "reason": "Operator shift duration is within statutory limits and no equipment incident or hazard reported.",
                "predicted": 0,
                "actual": int(row["Is_Violation"]),
                "actual_type": row["Violation_Type"],
                "expected_section": row.get("Expected_Section"),
                "retrieved": [],
                "citation_exists": None,
                "citation_was_retrieved": None,
            }
            continue

        # For all hazard candidates, run full grounded LLM audit
        if grounded:
            if strategy == "triangulated":
                hits = retrieve_triangulated(row, chunks, vectors, pillar_vecs)
            else:
                q_text = build_query(row, strategy=strategy)
                hits = retrieve_flat(q_text, chunks, vectors, k=top_k)
            context = "\n\n".join(
                f"[{c['citation']} -- {c['heading']}]\n{c['text'][:800]}" for c in hits
            )
            template = AUTONOMOUS_PROMPT
        else:
            hits = []
            context = ""
            template = UNGROUNDED_PROMPT

        prompt = template.format(
            context=context,
            log_id=row["Log_ID"],
            equipment=row["Equipment_ID"],
            equip_type=row["Equipment_Type"],
            location=row["Location"],
            shift_hours=row["Operator_Shift_Hours"],
            incident=row["Reported_Incident"],
        )
        llm_tasks.append((idx, row, hits, prompt))

    if llm_tasks:
        print(
            f"  dispatching {len(llm_tasks)} candidate hazard audits across {concurrency} async workers..."
        )
        async with aiohttp.ClientSession() as session:
            _PROGRESS.update({"done": 0, "total": len(llm_tasks), "failed": 0})
            coros = [ask_async(session, p, sem) for _, _, _, p in llm_tasks]
            raw_responses = await asyncio.gather(*coros, return_exceptions=True)
            raw_responses = [
                "" if isinstance(r, BaseException) else r for r in raw_responses
            ]

        for (idx, row, hits, _), raw in zip(llm_tasks, raw_responses):
            parsed = parse(raw)
            cite = parsed["citation"]
            retrieved_sections = {c["section"] for c in hits}

            parsed.update(
                {
                    "log_id": row["Log_ID"],
                    "audited_by": "model",
                    "predicted": 1 if parsed["status"] == "VIOLATION" else 0,
                    "actual": int(row["Is_Violation"]),
                    "actual_type": row["Violation_Type"],
                    "expected_section": row.get("Expected_Section"),
                    "retrieved": sorted(retrieved_sections),
                    "citation_exists": (None if cite is None else cite in sections),
                    "citation_was_retrieved": (
                        None
                        if (cite is None or not grounded)
                        else cite in retrieved_sections
                    ),
                }
            )
            results[idx] = parsed

    return results


def _core_metrics(results):
    """Confusion matrix and derived rates over whatever subset is passed in."""
    tp = sum(r["predicted"] and r["actual"] for r in results)
    fp = sum(r["predicted"] and not r["actual"] for r in results)
    fn = sum(not r["predicted"] and r["actual"] for r in results)
    tn = sum(not r["predicted"] and not r["actual"] for r in results)
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


def score(results):
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

    # Correct rule: did the citation name the section the generator recorded as
    # governing this hazard? Read off the row, so adding a hazard type needs no
    # change here. Distinct from validity (is it real) and grounding (was it
    # retrieved): a citation can pass both and still be the wrong rule.
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
        description="Audit yard log against federal safety regulations."
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="chunks retrieved per event for raw/expanded strategies. "
        "Triangulated ignores this: it is structurally capped at "
        "len(pillars) + 2 regardless, which is why its recall plateaus.",
    )
    ap.add_argument(
        "--strategy",
        choices=["triangulated", "expanded", "raw"],
        default="triangulated",
        help="retrieval query construction strategy",
    )
    ap.add_argument(
        "--concurrency", type=int, default=6, help="concurrent Ollama requests"
    )
    ap.add_argument(
        "--triage",
        action="store_true",
        help="skip routine rows with the keyword pre-filter (cheaper, and lossy: "
        "measured recall 0.27 on realistic phrasing). Off by default.",
    )
    ap.add_argument(
        "--no-retrieval",
        action="store_true",
        help="reproduce the original ungrounded build, for comparison",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with LOG.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    chunks, vectors, corpus_sections = load_index()
    sections = real_sections()
    grounded = not args.no_retrieval
    mode = (
        f"retrieval-grounded ({args.strategy})"
        if grounded
        else "UNGROUNDED (original build)"
    )

    print(f"Auditing {len(rows)} yard events, {mode}, with {CHAT_MODEL}...")
    if grounded:
        print(
            f"  corpus: {len(chunks)} chunks across {len(corpus_sections)} sections, concurrency={args.concurrency}"
        )

    t0 = time.time()
    results = asyncio.run(
        audit_async(
            rows,
            chunks,
            vectors,
            sections,
            grounded=grounded,
            strategy=args.strategy,
            concurrency=args.concurrency,
            use_triage=args.triage,
            top_k=args.top_k,
        )
    )
    metrics = score(results)
    metrics["seconds"] = round(time.time() - t0, 1)
    metrics["chat_model"] = CHAT_MODEL
    metrics["embed_model"] = EMBED_MODEL
    metrics["strategy"] = args.strategy
    metrics["top_k"] = args.top_k if grounded else None
    metrics["grounded"] = grounded

    out_path = (
        Path(args.out)
        if args.out
        else (
            RESULTS if grounded else RESULTS.with_name("audit_results_ungrounded.json")
        )
    )
    out_path.write_text(
        json.dumps({"metrics": metrics, "results": results}, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 56)
    print("  DETECTION METRICS")
    print("=" * 56)
    for k in (
        "n",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ):
        if k in metrics:
            print(f"  {k:34} {metrics[k]}")
    print("-" * 56)
    print("  CITATION GROUNDING & CORRECTNESS")
    print("-" * 56)
    for k in (
        "citations_given",
        "citations_fabricated",
        "citations_outside_retrieved_set",
        "citation_validity",
        "citation_correctness",
        "unparsed",
    ):
        print(f"  {k:34} {metrics[k]}")
    print(f"  {'seconds':34} {metrics['seconds']}")

    missed = [r for r in results if r["actual"] and not r["predicted"]]
    if missed:
        print("\n  Missed violations:")
        for r in missed:
            print(f"    {r['log_id']} ({r['actual_type']}): {r['reason'][:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
