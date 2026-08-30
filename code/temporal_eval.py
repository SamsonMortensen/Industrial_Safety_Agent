"""Frozen-slice temporal evaluation: does the auditor learn, or does it memorise?

The claim "continuously and autonomously improves" is only testable against events
the memory has never ingested. This harness enforces that:

  * The holdout is the FINAL chronological segment of the yard log, so every
    training shift precedes every evaluation event. That is both the honest
    experimental setup and the realistic deployment one -- you learn from past
    shifts and audit the current one.
  * The holdout is NEVER ingested into episodic memory and NEVER adapts pillars.
    It is replayed unchanged after each shift.
  * Memory starts cold. No committed state is read.

Read the resulting curve as follows:

  holdout F1 / grounded-citation rate rising over shifts
      -> learning transfers to unseen events. The claim, demonstrated.
  holdout curve flat while in-memory performance is perfect
      -> memorisation, not learning. Publishing that is worth more than a perfect
         score, because it is the experiment capable of catching you.

    python code/temporal_eval.py --dry-run          # validate the split, no model calls
    python code/temporal_eval.py --shifts 5 --holdout 200
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
LOG = ROOT / "daily_yard_log.csv"
OUT = JSON_DIR / "temporal_eval.json"

# Cold, throwaway state for the experiment -- never the shipped memory files.
EXP_MEM = JSON_DIR / "temporal_eval_mem.json"
EXP_VEC = JSON_DIR / "temporal_eval_vecs.npz"
EXP_STATE = JSON_DIR / "temporal_eval_state.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_agent_v2 import compute_metrics, load_corpus_index, run_continuous_audit
from continuous_learner import AdaptivePillarBank, EpisodicMemoryBank, parse_event_time
from self_reflection import SelfReflectionCritic, StatutoryGroundedVerifier


def split_dataset(
    rows: List[Dict[str, Any]], holdout_size: int, n_shifts: int
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Chronological split: earlier rows become shifts, the final segment is frozen."""
    ordered = sorted(
        rows, key=lambda r: (parse_event_time(r) or 0.0, r.get("Log_ID", ""))
    )
    if holdout_size >= len(ordered):
        raise SystemExit(
            f"holdout ({holdout_size}) must be smaller than the log ({len(ordered)})"
        )
    if n_shifts < 1:
        raise SystemExit("--shifts must be at least 1")

    train, holdout = ordered[:-holdout_size], ordered[-holdout_size:]
    per = len(train) // n_shifts
    if per == 0:
        raise SystemExit(
            f"{n_shifts} shifts is too many for {len(train)} training events"
        )
    shifts = [train[i * per : (i + 1) * per] for i in range(n_shifts)]
    shifts[-1].extend(train[n_shifts * per :])  # remainder joins the final shift
    return shifts, holdout


def composition(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Violation counts by type, for judging whether a slice is interpretable."""
    out: Dict[str, int] = {}
    for r in rows:
        if int(r.get("Is_Violation", 0)):
            kind = r.get("Violation_Type", "unknown")
            out[kind] = out.get(kind, 0) + 1
    return out


def grounded_citation_rate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Label-free correctness proxy.

    A citation counts as grounded only if it names a real CFR section AND that
    section appeared in the retrieved context for the event. This needs no ground
    truth, so it is computable in a real yard -- which makes it the right metric
    to optimise against.
    """
    cited = [r for r in results if r.get("citation")]
    if not cited:
        return {
            "citations": 0,
            "grounded": 0,
            "grounded_rate": None,
            "fabricated": 0,
            "outside_retrieved": 0,
        }
    grounded = [
        r for r in cited if r.get("citation_exists") and r.get("citation_was_retrieved")
    ]
    return {
        "citations": len(cited),
        "grounded": len(grounded),
        "grounded_rate": round(len(grounded) / len(cited), 4),
        "fabricated": sum(1 for r in cited if not r.get("citation_exists")),
        "outside_retrieved": sum(
            1
            for r in cited
            if r.get("citation_exists") and not r.get("citation_was_retrieved")
        ),
    }


async def evaluate_holdout(
    holdout, chunks, vectors, memory, pillars, verifier, critic, concurrency
):
    """Replay the frozen slice. Ingestion and pillar adaptation are both disabled."""
    results, _ = await run_continuous_audit(
        rows=holdout,
        chunks=chunks,
        vectors=vectors,
        memory_bank=memory,
        pillar_bank=pillars,
        verifier=verifier,
        critic=critic,
        concurrency=concurrency,
        run_id="holdout",
        enable_few_shot=True,
        enable_self_reflection=True,
        ingest_memory=False,  # the frozen slice must never enter memory
        adapt_pillars=False,  # nor shape retrieval
    )
    m = compute_metrics(results)
    m.update(grounded_citation_rate(results))
    return m


def print_split(rows, shifts, holdout, args):
    print("\n" + "=" * 74)
    print("  FROZEN-SLICE TEMPORAL EVALUATION")
    print("=" * 74)
    print(f"  Log records        : {len(rows)}")
    print(f"  Training shifts    : {args.shifts}")
    print(f"  Frozen holdout     : {len(holdout)} events (final chronological segment)")
    hcomp = composition(holdout)
    print(f"  Holdout violations : {hcomp}  (total {sum(hcomp.values())})")
    for i, s in enumerate(shifts, 1):
        print(f"    shift {i}: {len(s):4d} events, violations {composition(s)}")

    total_v = sum(hcomp.values())
    if total_v < 9:
        print(
            f"\n  WARNING: only {total_v} violations in the holdout -- recall will be very noisy."
        )
        print("           Raise --holdout for an interpretable curve.")
    for kind, n in sorted(hcomp.items()):
        if n < 3:
            print(
                f"  WARNING: holdout has only {n} '{kind}' violation(s); that per-type rate is not interpretable."
            )
    print("=" * 74)


async def run(args):
    with LOG.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    shifts, holdout = split_dataset(rows, args.holdout, args.shifts)
    if args.holdout_limit:
        holdout = holdout[: args.holdout_limit]

    print_split(rows, shifts, holdout, args)

    if args.dry_run:
        print("\n  --dry-run: split validated, no model calls made.\n")
        return 0

    for f in (EXP_MEM, EXP_VEC, EXP_STATE):
        if f.exists():
            f.unlink()
    print("\n  Cold start: experiment memory and pillar state cleared.\n")

    chunks, vectors, _ = load_corpus_index()
    memory = EpisodicMemoryBank(memory_file=EXP_MEM, vec_file=EXP_VEC)
    pillars = AdaptivePillarBank(state_file=EXP_STATE)
    verifier = StatutoryGroundedVerifier()
    critic = SelfReflectionCritic(verifier)

    curve = []
    t_all = time.time()

    print("--- shift 0: cold model, empty memory (baseline) ---")
    m0 = await evaluate_holdout(
        holdout, chunks, vectors, memory, pillars, verifier, critic, args.concurrency
    )
    m0.update({"shift": 0, "memory_size": 0, "pillars": len(pillars.pillars)})
    curve.append(m0)
    print(
        f"    holdout F1={m0['f1']:.4f}  grounded_rate={m0['grounded_rate']}  "
        f"P={m0['precision']:.3f} R={m0['recall']:.3f}\n"
    )

    for i, shift in enumerate(shifts, 1):
        print(
            f"--- shift {i}/{len(shifts)}: auditing {len(shift)} events (memory accumulating) ---"
        )
        await run_continuous_audit(
            rows=shift,
            chunks=chunks,
            vectors=vectors,
            memory_bank=memory,
            pillar_bank=pillars,
            verifier=verifier,
            critic=critic,
            concurrency=args.concurrency,
            run_id=f"shift_{i}",
            enable_few_shot=True,
            enable_self_reflection=True,
            ingest_memory=True,
            adapt_pillars=True,
        )
        print(
            f"    memory now {len(memory.episodes)} episodes, {len(pillars.pillars)} pillars"
        )

        print(f"--- re-evaluating frozen holdout after shift {i} ---")
        m = await evaluate_holdout(
            holdout,
            chunks,
            vectors,
            memory,
            pillars,
            verifier,
            critic,
            args.concurrency,
        )
        m.update(
            {
                "shift": i,
                "memory_size": len(memory.episodes),
                "pillars": len(pillars.pillars),
            }
        )
        curve.append(m)
        print(
            f"    holdout F1={m['f1']:.4f}  grounded_rate={m['grounded_rate']}  "
            f"P={m['precision']:.3f} R={m['recall']:.3f}\n"
        )

    print("=" * 74)
    print("  HOLDOUT LEARNING CURVE (frozen slice, never ingested)")
    print("=" * 74)
    print(
        f"  {'shift':<7}{'mem':<7}{'pillars':<9}{'F1':<9}{'prec':<8}{'rec':<8}{'grounded':<10}{'ungrounded'}"
    )
    for m in curve:
        gr = "n/a" if m["grounded_rate"] is None else f"{m['grounded_rate']:.3f}"
        print(
            f"  {m['shift']:<7}{m['memory_size']:<7}{m['pillars']:<9}{m['f1']:<9.4f}"
            f"{m['precision']:<8.3f}{m['recall']:<8.3f}{gr:<10}{m['outside_retrieved']}"
        )
    print("=" * 74)

    d_f1 = curve[-1]["f1"] - curve[0]["f1"]
    g0, gN = curve[0]["grounded_rate"], curve[-1]["grounded_rate"]
    d_gr = None if (g0 is None or gN is None) else round(gN - g0, 4)
    print(f"\n  Delta F1 (shift 0 -> {curve[-1]['shift']}): {d_f1:+.4f}")
    print(
        f"  Delta grounded-citation rate      : {d_gr if d_gr is not None else 'n/a'}"
    )
    print("\n  Rising  -> learning transfers to unseen events.")
    print(
        "  Flat    -> memorisation; the in-memory score was never evidence of learning.\n"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "config": {
                    "shifts": args.shifts,
                    "holdout": len(holdout),
                    "holdout_violations": composition(holdout),
                    "concurrency": args.concurrency,
                },
                "curve": curve,
                "delta_f1": round(d_f1, 4),
                "delta_grounded_rate": d_gr,
                "seconds": round(time.time() - t_all, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Wrote {OUT.relative_to(ROOT)}\n")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Frozen-slice temporal evaluation of continuous learning."
    )
    ap.add_argument(
        "--shifts", type=int, default=5, help="number of sequential training shifts"
    )
    ap.add_argument(
        "--holdout", type=int, default=200, help="size of the frozen final slice"
    )
    ap.add_argument(
        "--holdout-limit",
        type=int,
        default=None,
        help="subsample the holdout for faster iteration",
    )
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the split without any model calls",
    )
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
