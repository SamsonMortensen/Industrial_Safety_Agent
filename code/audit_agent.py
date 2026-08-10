"""Retrieval-grounded compliance audit over the synthetic yard log.

What changed from the original build, and why:

The first version described itself as RAG but never retrieved anything. The
regulatory corpus was loaded, chunked, and handed off to a variable that nothing
consumed, while the audit prompt asked the model to "cross-reference this event
with your regulatory knowledge base" -- meaning its own weights. It answered from
memory and invented citations to match. Its output cited 29 CFR 1910.1030
(bloodborne pathogens) for a shift-length violation, 29 CFR 1910.107 (spray
finishing) for a hydraulic spill, and 29 CFR 1910.1035, which does not exist.

Here the model only sees regulatory text that was actually retrieved for the
event in front of it, and is told to cite from that text or cite nothing. Every
citation it returns is then checked against the corpus, so a fabricated section
number is caught and reported rather than believed.

    python code/audit_agent.py            # audit all 50 rows and score
    python code/audit_agent.py --limit 5  # quick smoke run

Requires Ollama running locally (https://ollama.com) with the models below
pulled. Nothing leaves the machine.
"""
import argparse
import json
import re
import sys
import time

import pathlib
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
REGS = ROOT / "json" / "regulations.json"
EMBED_CACHE = ROOT / "json" / "reg_embeddings.npz"
LOG = ROOT / "daily_yard_log.csv"
RESULTS = ROOT / "json" / "audit_results.json"
VALID_SECTIONS = ROOT / "json" / "valid_sections.json"

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen3.5:9b"
TOP_K = 4

PROMPT = """You are a safety and compliance auditor for a rail-served intermodal yard.

Below are excerpts from federal regulations that were retrieved for this event.
They are the ONLY regulations you may cite. Do not cite any section number that
does not appear in these excerpts. If none of them cover the event, cite NONE.

RETRIEVED REGULATIONS
{context}

YARD EVENT
- Log ID: {log_id}
- Equipment: {equipment} ({equip_type})
- Location: {location}
- Operator shift hours: {shift_hours}
- Reported incident: {incident}

Decide whether this event violates one of the retrieved regulations. Consider
operator hours-of-service limits, hazardous spills where people walk, and
equipment operating close to energized high-voltage equipment.

Answer in exactly this format and nothing else:
STATUS: CLEAR or VIOLATION
CITATION: a section number copied from the excerpts above, or NONE
REASON: one sentence
"""


def embed(texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        r = requests.post(f"{OLLAMA}/api/embed",
                          json={"model": EMBED_MODEL, "input": texts[i:i + batch]},
                          timeout=600)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
        print(f"\r  embedded {min(i + batch, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    arr = np.asarray(out, dtype=np.float32)
    #Normalise once so cosine similarity is a plain dot product
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def real_sections():
    """Every section number that exists in 29 CFR and 49 CFR, not just the ones
    this project retrieves over. Keeps the fabrication metric honest."""
    if not VALID_SECTIONS.exists():
        raise SystemExit("Run code/build_section_index.py first.")
    d = json.loads(VALID_SECTIONS.read_text(encoding="utf-8"))
    return {s for group in d["by_title"].values() for s in group}


def load_index():
    corpus = json.loads(REGS.read_text(encoding="utf-8"))
    chunks = corpus["chunks"]

    if EMBED_CACHE.exists():
        cached = np.load(EMBED_CACHE)
        if cached["vectors"].shape[0] == len(chunks):
            return chunks, cached["vectors"], set(corpus["sections"])
        print("Corpus changed since the cache was built, re-embedding.")

    print(f"Embedding {len(chunks)} regulation chunks with {EMBED_MODEL}...")
    vectors = embed([f"{c['citation']} {c['heading']}. {c['text']}" for c in chunks])
    np.savez_compressed(EMBED_CACHE, vectors=vectors)
    return chunks, vectors, set(corpus["sections"])


def retrieve(query, chunks, vectors, k=TOP_K):
    q = embed([query])[0]
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


def build_query(row):
    return (f"{row['Equipment_Type']} operating at {row['Location']}, "
            f"operator on shift {row['Operator_Shift_Hours']} hours, "
            f"incident reported: {row['Reported_Incident']}")


def ask(prompt):
    r = requests.post(f"{OLLAMA}/api/chat",
                      json={"model": CHAT_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False, "think": False,
                            "options": {"temperature": 0, "num_predict": 300}},
                      timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse(text):
    status = re.search(r"STATUS:\s*(VIOLATION|CLEAR)", text, re.I)
    citation = re.search(r"CITATION:\s*([^\n]+)", text, re.I)
    reason = re.search(r"REASON:\s*([^\n]+)", text, re.I)
    cite = (citation.group(1).strip() if citation else "NONE")
    #Keep just the section number, e.g. "29 CFR 1910.178(l)(1)" -> "1910.178"
    m = re.search(r"\b(\d{3,4}\.\d+)", cite)
    return {
        "status": (status.group(1).upper() if status else "UNPARSED"),
        "citation_raw": cite,
        "citation": (m.group(1) if m else None),
        "reason": (reason.group(1).strip() if reason else ""),
    }


#The original build's prompt: no retrieval, model answers from its own weights.
#Kept so the fabrication rate with and without grounding can be measured rather
#than asserted.
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


def audit(rows, chunks, vectors, sections, grounded=True):
    results = []
    for n, row in enumerate(rows, 1):
        if grounded:
            hits = retrieve(build_query(row), chunks, vectors)
            context = "\n\n".join(
                f"[{c['citation']} -- {c['heading']}]\n{c['text'][:900]}" for c, _ in hits)
            template = PROMPT
        else:
            hits = []
            context = ""
            template = UNGROUNDED_PROMPT

        raw = ask(template.format(
            context=context,
            log_id=row["Log_ID"], equipment=row["Equipment_ID"],
            equip_type=row["Equipment_Type"], location=row["Location"],
            shift_hours=row["Operator_Shift_Hours"], incident=row["Reported_Incident"]))

        parsed = parse(raw)
        cite = parsed["citation"]
        retrieved_sections = {c["section"] for c, _ in hits}
        #Without retrieval there is no grounded set to check against, only the
        #question of whether the cited section exists in federal law at all
        top_score = round(hits[0][1], 4) if hits else None

        parsed.update({
            "log_id": row["Log_ID"],
            "predicted": 1 if parsed["status"] == "VIOLATION" else 0,
            "actual": int(row["Is_Violation"]),
            "actual_type": row["Violation_Type"],
            "retrieved": sorted(retrieved_sections),
            "top_score": top_score,
            #Three states: no citation, a real one, or one the model invented
            "citation_exists": (None if cite is None else cite in sections),
            "citation_was_retrieved": (
                None if (cite is None or not grounded) else cite in retrieved_sections),
        })
        results.append(parsed)
        print(f"\r  audited {n}/{len(rows)}", end="", flush=True)
    print()
    return results


def score(results):
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
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(results), 4),
        "citations_given": len(cited),
        "citations_fabricated": len(fabricated),
        "citations_outside_retrieved_set": len(ungrounded),
        "citation_validity": round(1 - len(fabricated) / len(cited), 4) if cited else None,
        "unparsed": sum(r["status"] == "UNPARSED" for r in results),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-retrieval", action="store_true",
                    help="reproduce the original ungrounded build, for comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import csv
    with LOG.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]

    chunks, vectors, corpus_sections = load_index()
    sections = real_sections()
    grounded = not args.no_retrieval
    mode = "retrieval-grounded" if grounded else "UNGROUNDED (original build)"
    print(f"Auditing {len(rows)} yard events, {mode}, with {CHAT_MODEL}...")
    if grounded:
        print(f"  corpus: {len(chunks)} chunks across {len(corpus_sections)} sections, top_k={TOP_K}")

    t0 = time.time()
    results = audit(rows, chunks, vectors, sections, grounded=grounded)
    metrics = score(results)
    metrics["seconds"] = round(time.time() - t0, 1)
    metrics["chat_model"] = CHAT_MODEL
    metrics["embed_model"] = EMBED_MODEL
    metrics["top_k"] = TOP_K if grounded else None
    metrics["grounded"] = grounded

    out_path = pathlib.Path(args.out) if args.out else (
        RESULTS if grounded else RESULTS.with_name("audit_results_ungrounded.json"))
    out_path.write_text(json.dumps({"metrics": metrics, "results": results}, indent=1),
                       encoding="utf-8")

    print("\n" + "=" * 56)
    print("  DETECTION")
    print("=" * 56)
    for k in ("n", "true_positives", "false_positives", "false_negatives",
              "true_negatives", "precision", "recall", "f1", "accuracy"):
        print(f"  {k:34} {metrics[k]}")
    print("-" * 56)
    print("  CITATION GROUNDING")
    print("-" * 56)
    for k in ("citations_given", "citations_fabricated",
              "citations_outside_retrieved_set", "citation_validity", "unparsed"):
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
