"""Is the auditor failing at retrieval, or at reasoning?

Why this exists
---------------
A recall number says something is broken. It does not say what, and the two
candidates need opposite fixes:

  retrieval failure   the governing regulation never reached the model, so no
                      amount of better prompting or a bigger model can help
  reasoning failure   the regulation was right there in the context and the model
                      failed to connect it to the event, so better embeddings
                      change nothing

Without splitting them, any improvement is a guess. This script splits them.

How
---
Every planted violation carries the section that governs it in the
Expected_Section column, written by generate_yard_log.py across 16 hazard types.
For each one this script asks whether that section appeared in the retrieved
excerpts and whether the model caught the violation. Crossing those gives the
split, and the RANK the correct section reached says whether a larger k would
have helped.

This needs embeddings but no text generation, so it runs when the audit itself is
too slow to repeat. Citation correctness is reported only when the saved verdicts
were produced against the current dataset; crossing verdicts from another dataset with current labels
yields a number that looks real and means nothing.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_agent import (
    LOG,
    RESULTS,
    build_query,
    embed,
    get_pillar_vectors,
    load_index,
    retrieve_triangulated,
)


# Read each planted violation's governing rule from the generated row.
def expected_section(row):
    """(label, predicate) for the section a given violation row should cite."""
    want = str(row.get("Expected_Section", "")).strip()
    if not want or want == "None":
        return None, None
    title = (
        "49 U.S.C."
        if want == "21103"
        else "49 CFR"
        if want.startswith("228")
        else "29 CFR"
    )
    return f"{title} {want}", (lambda sec, w=want: sec == w)


DEFAULT_KS = (1, 2, 4, 8, 16, 32, 64)


def load_violations():
    with LOG.open(encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if row["Is_Violation"] == "1"]


def section_of(chunk):
    citation = str(chunk.get("citation", ""))
    return citation.split()[-1] if citation else ""


def rank_of_correct_section(query, chunks, vectors, matches):
    query_vector = embed([query])[0]
    order = np.argsort(-(vectors @ query_vector))
    for position, index in enumerate(order, start=1):
        section = section_of(chunks[index])
        if matches(section):
            return position, section
    return None, None


def run(ks=DEFAULT_KS, strategy="triangulated"):
    chunks, vectors, _ = load_index()
    violations = load_violations()
    pillar_vecs = get_pillar_vectors() if strategy == "triangulated" else None

    # The model's own verdicts, so retrieval and reasoning can be crossed.
    predictions = {}
    if RESULTS.exists():
        for row in json.loads(RESULTS.read_text(encoding="utf-8")).get("results", []):
            predictions[row["log_id"]] = row["predicted"]

    rows = []
    for violation in violations:
        kind = violation["Violation_Type"]
        label, matches = expected_section(violation)
        if matches is None:
            continue

        if strategy == "triangulated":
            hits = retrieve_triangulated(violation, chunks, vectors, pillar_vecs)
            found = False
            for pos, h in enumerate(hits, 1):
                sec = section_of(h)
                if matches(sec):
                    rank = pos
                    section = sec
                    found = True
                    break
            if not found:
                rank, section = rank_of_correct_section(
                    build_query(violation, strategy="raw"), chunks, vectors, matches
                )
        else:
            rank, section = rank_of_correct_section(
                build_query(violation, strategy=strategy), chunks, vectors, matches
            )

        rows.append(
            {
                "log_id": violation["Log_ID"],
                "type": kind,
                "expected_section": violation.get("Expected_Section", ""),
                "expected": label,
                "rank": rank,
                "matched_section": section,
                "caught": predictions.get(violation["Log_ID"]),
            }
        )

    return rows


def citation_correctness(rows, results_path=RESULTS):
    """Does a cited section name the rule that was actually broken?

    Refuses to report if the saved verdicts were produced against a different
    dataset. Crossing verdicts from another dataset with current labels silently yields a number that
    looks real and means nothing.
    """
    if not results_path.exists():
        return None

    saved = json.loads(results_path.read_text(encoding="utf-8")).get("results", [])
    # Log IDs are positional and survive a regeneration, so they prove nothing.
    # Compare the labels themselves.
    saved_types = {r["log_id"]: r.get("actual_type") for r in saved}
    current_types = {r["log_id"]: r["type"] for r in rows}
    checked = [lid for lid in current_types if lid in saved_types]
    agree = sum(1 for lid in checked if saved_types[lid] == current_types[lid])
    match = agree / len(checked) if checked else 0.0
    if match < 0.9:
        print(
            f"  Skipping citation correctness: verdicts in {results_path.name} agree with "
            f"only {match:.0%} of the current labels, so they were produced against a "
            f"different dataset."
        )
        print("  Re-run the audit against the current log to measure it.")
        return None

    verdicts = {
        r["log_id"]: r
        for r in json.loads(results_path.read_text(encoding="utf-8")).get("results", [])
    }
    detail = []
    for row in rows:
        verdict = verdicts.get(row["log_id"])
        if not verdict or not verdict.get("predicted"):
            continue
        _, matches = expected_section(
            {"Expected_Section": row.get("expected_section", "")}
        )
        if matches is None:
            continue
        cited = str(verdict.get("citation") or "")
        detail.append(
            {
                "log_id": row["log_id"],
                "type": row["type"],
                "expected": row["expected"],
                "cited": cited or "NONE",
                "exists": verdict.get("citation_exists"),
                "correct": bool(cited) and matches(cited),
            }
        )

    return {
        "caught": len(detail),
        "cited_a_real_section": sum(1 for d in detail if d["exists"]),
        "cited_the_correct_rule": sum(1 for d in detail if d["correct"]),
        "detail": detail,
    }


def summarise(rows, ks=DEFAULT_KS):
    ranked = [r for r in rows if r["rank"] is not None]
    recall_at = {k: sum(1 for r in ranked if r["rank"] <= k) / len(rows) for k in ks}

    scored = [r for r in rows if r["caught"] is not None]
    quadrants = {
        "caught_retrieved": 0,
        "caught_not_retrieved": 0,
        "missed_retrieved": 0,
        "missed_not_retrieved": 0,
    }
    for row in scored:
        retrieved = row["rank"] is not None and row["rank"] <= 4
        caught = bool(row["caught"])
        key = ("caught" if caught else "missed") + (
            "_retrieved" if retrieved else "_not_retrieved"
        )
        quadrants[key] += 1

    return {
        "violations": len(rows),
        "in_corpus": len(ranked),
        "not_in_corpus": len(rows) - len(ranked),
        "recall_at": recall_at,
        "quadrants": quadrants,
        "median_rank": sorted(r["rank"] for r in ranked)[len(ranked) // 2]
        if ranked
        else None,
        "worst_rank": max((r["rank"] for r in ranked), default=None),
    }


def render(rows, summary, ks=DEFAULT_KS, strategy="triangulated"):
    lines = []
    lines.append("")
    lines.append(
        f"  Retrieval diagnostic ({strategy}): where does the audit actually fail?"
    )
    lines.append("  " + "=" * 74)
    lines.append("")
    lines.append(f"  {'log':<10}{'type':<12}{'expected':<18}{'rank':>6}{'caught':>9}")
    lines.append("  " + "-" * 74)
    for row in sorted(rows, key=lambda r: (r["type"], r["log_id"])):
        rank = row["rank"] if row["rank"] is not None else "absent"
        caught = "-" if row["caught"] is None else ("YES" if row["caught"] else "no")
        lines.append(
            f"  {row['log_id']:<10}{row['type']:<12}{row['expected']:<18}"
            f"{str(rank):>6}{caught:>9}"
        )

    lines.append("")
    lines.append("  Rank of the governing regulation, over the whole corpus")
    lines.append("  " + "-" * 74)
    if summary["not_in_corpus"]:
        lines.append(
            f"    NOT IN CORPUS AT ALL: {summary['not_in_corpus']} -- a coverage problem"
        )
    lines.append(
        f"    median rank {summary['median_rank']}   worst rank {summary['worst_rank']}"
    )
    lines.append("")
    lines.append("    retrieval recall at K:")
    for k in ks:
        share = summary["recall_at"][k]
        bar = "#" * int(round(share * 40))
        lines.append(f"      K={k:<4} {share:>5.0%}  {bar}")

    lines.append("")
    lines.append("  Splitting the failure (at K=4)")
    lines.append("  " + "-" * 74)
    q = summary["quadrants"]
    lines.append(f"    caught, regulation retrieved        {q['caught_retrieved']:>3}")
    lines.append(
        f"    caught, regulation NOT retrieved    {q['caught_not_retrieved']:>3}"
    )
    lines.append(
        f"    MISSED, regulation retrieved        {q['missed_retrieved']:>3}   <- reasoning failure"
    )
    lines.append(
        f"    MISSED, regulation not retrieved    {q['missed_not_retrieved']:>3}   <- retrieval failure"
    )

    citations = citation_correctness(rows)
    if citations and citations["caught"]:
        lines.append("")
        lines.append("  Citation VALIDITY is not citation CORRECTNESS")
        lines.append("  " + "-" * 74)
        lines.append(
            f"  {'log':<10}{'type':<12}{'should cite':<18}{'did cite':<14}"
            f"{'real?':>7}{'right?':>8}"
        )
        for d in citations["detail"]:
            lines.append(
                f"  {d['log_id']:<10}{d['type']:<12}{d['expected']:<18}"
                f"{d['cited']:<14}{str(bool(d['exists'])):>7}"
                f"{('YES' if d['correct'] else 'NO'):>8}"
            )
        lines.append("")
        lines.append(
            f"    of {citations['caught']} violations caught, "
            f"{citations['cited_a_real_section']} cited a real section "
            f"and {citations['cited_the_correct_rule']} cited the correct one"
        )
    lines.append("  " + "=" * 74)
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Split audit failures into retrieval vs reasoning."
    )
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument(
        "--strategy",
        choices=["triangulated", "expanded", "raw"],
        default="triangulated",
    )
    parser.add_argument("--out", default="json/retrieval_diagnostic.json")
    args = parser.parse_args()

    ks = tuple(sorted(set(args.k)))
    rows = run(ks, strategy=args.strategy)
    summary = summarise(rows, ks)
    summary["citations"] = citation_correctness(rows)
    print(render(rows, summary, ks, strategy=args.strategy))

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"strategy": args.strategy, "rows": rows, "summary": summary}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"  wrote {out.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
