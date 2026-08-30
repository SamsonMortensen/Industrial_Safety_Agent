"""Does changing the query fix retrieval?

`retrieval_diagnostic.py` established where the audit fails. This measures
whether the proposed fix works, and it measures it on RETRIEVAL RECALL rather
than on end-to-end audit recall.

The strategies under test are defined in `audit_agent`:

    raw           the baseline: restate the event operationally (vocabulary gap)
    expanded      append domain safety terms implied by the fields present
    triangulated  multi-hazard triangulation across compliance dimensions

    python code/retrieval_experiment.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_agent import (
    EMBED_MODEL,
    LOG,
    REGS,
    STATUTES,
    build_query,
    embed,
    get_pillar_vectors,
    retrieve_triangulated,
)
from retrieval_diagnostic import expected_section, section_of

DEFAULT_KS = (1, 2, 4, 8, 16, 32)
DEFAULT_STRATEGIES = ("raw", "expanded", "triangulated")


def load_violations():
    with LOG.open(encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if row["Is_Violation"] == "1"]


def corpus_vectors(model):
    corpus = json.loads(REGS.read_text(encoding="utf-8"))
    chunks = list(corpus["chunks"])
    if STATUTES.exists():
        chunks.extend(
            json.loads(STATUTES.read_text(encoding="utf-8")).get("chunks", [])
        )
    cache = REGS.parent / (
        "reg_embeddings.npz"
        if model == EMBED_MODEL
        else f"reg_embeddings_{model.replace(':', '_').replace('/', '_')}.npz"
    )

    if cache.exists():
        cached = np.load(cache)
        if cached["vectors"].shape[0] == len(chunks):
            return chunks, cached["vectors"]

    print(f"  embedding {len(chunks)} chunks with {model} (one time, cached)...")
    vectors = embed(
        [f"{c['citation']} {c['heading']}. {c['text']}" for c in chunks], model=model
    )
    np.savez_compressed(cache, vectors=vectors)
    return chunks, vectors


def ranks_for(strategy, model, chunks, vectors, violations):
    pillar_vecs = (
        get_pillar_vectors(model=model) if strategy == "triangulated" else None
    )
    out = []
    for row in violations:
        _, matches = expected_section(row)
        if matches is None:
            continue
        if strategy == "triangulated":
            hits = retrieve_triangulated(row, chunks, vectors, pillar_vecs)
            rank = None
            for pos, h in enumerate(hits, 1):
                if matches(section_of(h)):
                    rank = pos
                    break
        else:
            q_text = build_query(row, strategy=strategy)
            q_vec = embed([q_text], model=model)[0]
            order = np.argsort(-(vectors @ q_vec))
            rank = None
            for pos, idx in enumerate(order, start=1):
                if matches(section_of(chunks[idx])):
                    rank = pos
                    break
        out.append(
            {"log_id": row["Log_ID"], "type": row["Violation_Type"], "rank": rank}
        )
    return out


def recall_at(rows, ks):
    total = len(rows)
    return {
        k: sum(1 for r in rows if r["rank"] is not None and r["rank"] <= k) / total
        for k in ks
    }


def run(strategies, models, ks):
    violations = load_violations()
    results = {}
    for model in models:
        chunks, vectors = corpus_vectors(model)
        for strategy in strategies:
            rows = ranks_for(strategy, model, chunks, vectors, violations)
            results[(model, strategy)] = {
                "rows": rows,
                "recall_at": recall_at(rows, ks),
                "median_rank": sorted(r["rank"] for r in rows if r["rank"])[
                    len(rows) // 2
                ],
                "by_type": {
                    kind: sorted(
                        r["rank"] for r in rows if r["type"] == kind and r["rank"]
                    )
                    for kind in sorted({r["type"] for r in rows})
                },
            }
    return results


def render(results, strategies, models, ks):
    lines = []
    lines.append("")
    lines.append("  Does changing the query fix retrieval?")
    n_viol = next((len(e["rows"]) for e in results.values() if "rows" in e), 0)
    lines.append(f"  measured on retrieval recall across {n_viol} planted violations")
    lines.append("  " + "=" * 74)

    for model in models:
        lines.append("")
        lines.append(f"  embedding model: {model}")
        header = f"  {'strategy':<14}{'median rank':>13}"
        header += "".join(f"{'K=' + str(k):>8}" for k in ks)
        lines.append(header)
        lines.append("  " + "-" * 74)
        for strategy in strategies:
            entry = results[(model, strategy)]
            row = f"  {strategy:<14}{entry['median_rank']:>13}"
            row += "".join(f"{entry['recall_at'][k]:>7.0%} " for k in ks)
            lines.append(row)

        lines.append("")
        lines.append(f"  {'':<14}rank of the governing section, by violation type")
        lines.append("  " + "-" * 74)
        for strategy in strategies:
            by_type = results[(model, strategy)]["by_type"]
            parts = "  ".join(f"{kind}: {ranks}" for kind, ranks in by_type.items())
            lines.append(f"  {strategy:<14}{parts}")

    lines.append("  " + "=" * 74)
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare query strategies on retrieval recall."
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        choices=["raw", "expanded", "triangulated"],
    )
    parser.add_argument("--models", nargs="+", default=[EMBED_MODEL])
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--out", default="json/retrieval_experiment.json")
    args = parser.parse_args()

    ks = tuple(sorted(set(args.k)))
    results = run(args.strategies, args.models, ks)

    # Persist before rendering. The sweep costs an embedding pass over every
    # violation for every strategy; a formatting bug in the display should not
    # be able to discard it.
    payload = {
        "config": {
            "strategies": args.strategies,
            "models": args.models,
            "ks": list(ks),
        },
        "results": [
            {"model": model, "strategy": strategy, **entry}
            for (model, strategy), entry in results.items()
        ],
    }
    out = Path(__file__).resolve().parent.parent / args.out
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
