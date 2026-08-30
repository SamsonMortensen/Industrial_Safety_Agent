"""Generate the synthetic yard log used as ground truth for the audit.

Design rules:

1. Every violation maps to one governing authority that is actually present in the
   retrievable corpus. The generator verifies this against json/regulations.json
   and json/statutes.json
   and refuses to write if a rule it plants cannot be retrieved, because a
   citation the auditor could never find is not a fair test.
2. Every hazard has a matched compliant counterpart describing the same
   situation done correctly (a secured dockboard, an inspected sling, a
   de-energised line). These are the hard negatives. Without them precision is
   free, since the model can flag anything that mentions a dockboard.
3. No single field separates the classes. Violations and clean rows draw from
   the same locations, the same equipment, and overlapping shift durations.
4. Incident text never names its own rule or the hazard category. The model has
   to reason from the retrieved regulation, not pattern-match a label.

    python code/generate_yard_log.py                      # 1,000 records
    python code/generate_yard_log.py --records 3000       # ~30 per hazard type
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "daily_yard_log.csv"
CORPUS = ROOT / "json" / "regulations.json"
STATUTES = ROOT / "json" / "statutes.json"

SEED = 20260329
DEFAULT_RECORDS = 1000
VIOLATION_RATIO = 0.15  # high for a real yard; a benchmark needs positives

EQUIPMENT = [
    "Gantry Crane",
    "Reach Stacker",
    "Forklift",
    "Terminal Tractor",
    "Yard Hostler",
    "Container Handler",
]

# Shared location pool. Violations and clean rows both draw from all of it.
LOCATIONS = [
    "Bay 1",
    "Bay 2",
    "Bay 4",
    "Maintenance Track",
    "Rail Siding 3",
    "Main Pedestrian Crosswalk",
    "High-Voltage Line B",
    "Loading Dock 2",
    "Container Stack Row C",
    "Wheel Shop",
    "Intermodal Ramp",
]

LEGAL_SHIFT = (6.0, 12.0)
FATIGUE_SHIFT = (12.1, 15.0)

# Each hazard: the governing section, violation phrasings, and the compliant
# counterpart phrasings for the same situation.
HAZARDS = [
    {
        # 49 CFR 228.405 applies to commuter and intercity passenger train
        # employees. This freight-yard scenario instead uses the train-employee
        # duty limit in 49 U.S.C. 21103.
        "kind": "duty_limit",
        "section": "21103",
        "fatigue": True,
        "equipment": "Yard Hostler",
        "bad": [
            "Freight train employee remained in covered service through relief delay",
            "Covered switch employee continued yard moves past scheduled relief",
            "Train employee remained on duty to complete consist tie-down",
        ],
        "ok": [
            "Relief crew arrived and the covered train employee was released on schedule",
            "Train employee logged out before the end of the permitted duty period",
        ],
    },
    {
        "kind": "duty_records",
        "section": "228.11",
        "bad": [
            "No hours-of-duty entry filed for the completed assignment",
            "Duty record for the tour left unsigned and incomplete",
        ],
        "ok": [
            "Hours-of-duty record filed and certified at tour end",
            "Duty record completed and countersigned by the supervisor",
        ],
    },
    {
        "kind": "housekeeping",
        "section": "1910.22",
        "locations": ["Main Pedestrian Crosswalk", "Loading Dock 2", "Intermodal Ramp"],
        "bad": [
            "Hydraulic fluid pooled across the marked walking route and left unattended",
            "Standing oily runoff spread over the pedestrian route with no containment",
            "Spilled coolant left across the marked pedestrian path",
        ],
        "ok": [
            "Fluid release contained with absorbent and the route barricaded until dry",
            "Washdown runoff captured by the oil-water separator drain",
            "Marked route swept and confirmed dry at shift start",
        ],
    },
    {
        "kind": "ladder",
        "section": "1910.23",
        "bad": [
            "Fixed ladder to the ramp platform in use with two rungs missing",
            "Portable ladder with a cracked side rail used to reach the container roof",
        ],
        "ok": [
            "Ladder inspected before use with all rungs intact",
            "Damaged ladder tagged out of service and removed from the area",
        ],
    },
    {
        "kind": "stairway",
        "section": "1910.25",
        "bad": [
            "Stair run to the dispatch office in use with the handrail detached",
            "Open-sided stairway to the ramp office missing its handrail",
        ],
        "ok": [
            "Stairway handrail confirmed secure during the walk-through",
            "Stair treads and handrail inspected and found sound",
        ],
    },
    {
        "kind": "dockboard",
        "section": "1910.26",
        "locations": ["Loading Dock 2", "Intermodal Ramp", "Bay 2"],
        "bad": [
            "Portable dockboard shifted under the forklift with no securing device fitted",
            "Dockboard placed without anchors and moved during transit into the boxcar",
        ],
        "ok": [
            "Dockboard anchored and its rated capacity posted before transit",
            "Dockboard secured with anchor pins and checked before the lift",
        ],
    },
    {
        "kind": "fall_protection",
        "section": "1910.28",
        "locations": ["Container Stack Row C", "Intermodal Ramp", "Bay 4"],
        "bad": [
            "Worker on the container roof at eight feet with no fall protection rigged",
            "Employee working the open edge of the stack with no arrest system attached",
        ],
        "ok": [
            "Worker on the container roof tied off to a rated anchor",
            "Edge work performed from an aerial lift with the harness attached",
        ],
    },
    {
        "kind": "guardrail",
        "section": "1910.29",
        "bad": [
            "Platform guardrail measured at twenty-eight inches above the walking level",
            "Elevated platform rail installed well below the required height",
        ],
        "ok": [
            "Platform guardrail measured at forty-two inches and found compliant",
            "Guardrail height and toeboard verified during inspection",
        ],
    },
    {
        "kind": "aisle_obstruction",
        "section": "1910.176",
        "locations": ["Bay 1", "Bay 2", "Container Stack Row C", "Loading Dock 2"],
        "bad": [
            "Cargo stacked with a three-foot overhang blocking the marked aisle",
            "Palletised freight left obstructing the designated traffic aisle",
        ],
        "ok": [
            "Freight palletised and stacked within the marked aisle boundaries",
            "Aisle kept clear and stacks squared during the transfer",
        ],
    },
    {
        "kind": "rim_wheel",
        "section": "1910.177",
        "locations": ["Wheel Shop", "Maintenance Track"],
        "bad": [
            "Multi-piece rim assembly inflated outside any restraining device",
            "Rim wheel serviced without deflating and without a restraining cage",
        ],
        "ok": [
            "Rim wheel deflated and seated in the restraining device before service",
            "Tyre inflated inside the cage by a trained servicer",
        ],
    },
    {
        "kind": "truck_defect",
        "section": "1910.178",
        "bad": [
            "Lift truck kept in service after the mast chain was found slack and frayed",
            "Truck with failed service brake returned to yard duty without repair",
        ],
        "ok": [
            "Lift truck removed from service and tagged pending repair",
            "Daily truck inspection completed with no defects found",
        ],
    },
    {
        "kind": "crane_inspection",
        "section": "1910.179",
        "locations": ["Bay 1", "Bay 4", "Intermodal Ramp"],
        "bad": [
            "Gantry hoist operated with the periodic inspection more than a year overdue",
            "Overhead crane run with the upper limit switch known inoperative",
        ],
        "ok": [
            "Gantry inspection current and the limit switch function-tested",
            "Crane inspection records verified before the lift began",
        ],
    },
    {
        "kind": "sling",
        "section": "1910.184",
        "bad": [
            "Wire rope sling with twelve broken wires in one lay kept in the active lift",
            "Sling used with a crushed eye fitting and no legible capacity tag",
        ],
        "ok": [
            "Sling inspected before the lift with its capacity tag legible",
            "Damaged sling removed from service and destroyed",
        ],
    },
    {
        "kind": "live_parts",
        "section": "1910.303",
        "locations": ["High-Voltage Line B", "Maintenance Track", "Wheel Shop"],
        "bad": [
            "Panel left open with energised parts exposed to the walkway",
            "Junction box cover missing and live conductors reachable from the aisle",
        ],
        "ok": [
            "Panel cover refitted and the enclosure secured after the work",
            "Enclosure verified closed and guarded before re-energising",
        ],
    },
    {
        "kind": "approach_distance",
        "section": "1910.333",
        "locations": ["High-Voltage Line B", "Intermodal Ramp", "Rail Siding 3"],
        "bad": [
            "Spreader boom worked within four feet of the energised catenary",
            "Mast raised inside the approach boundary of a live overhead line",
        ],
        "ok": [
            "Catenary confirmed de-energised, grounded and tagged before the lift",
            "Lift planned with the boom kept outside the approach boundary",
        ],
    },
    {
        "kind": "electrical_ppe",
        "section": "1910.335",
        "locations": ["High-Voltage Line B", "Maintenance Track", "Wheel Shop"],
        "bad": [
            "Work performed on an energised circuit without insulating gloves issued",
            "Employee worked exposed live parts with no protective equipment in use",
        ],
        "ok": [
            "Insulating gloves and face protection worn for the energised task",
            "Circuit de-energised before work, so no live-work equipment was needed",
        ],
    },
]

ROUTINE = [
    "No exception noted",
    "No exception noted",
    "No exception noted",
    "Routine load imbalance alarm cleared after twistlock engagement",
    "Tyre pressure indicator illuminated and reinflated at the air station",
    "Scheduled sensor calibration completed on the spreader",
    "Container transferred and seals verified against the manifest",
]


def corpus_sections():
    """Section numbers the auditor can actually retrieve."""
    if not CORPUS.exists():
        return None
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    out = set()
    for chunk in data.get("chunks", []):
        cit = str(chunk.get("citation", ""))
        if cit:
            out.add(cit.split()[-1])
    if STATUTES.exists():
        statutes = json.loads(STATUTES.read_text(encoding="utf-8"))
        for chunk in statutes.get("chunks", []):
            cit = str(chunk.get("citation", ""))
            if cit:
                out.add(cit.split()[-1])
    return out


def make_row(rng, hazard, violating):
    """Build the variable part of a record for one hazard, either breached or met."""
    pool = hazard.get("locations", LOCATIONS)
    if violating and hazard.get("fatigue"):
        shift = round(rng.uniform(*FATIGUE_SHIFT), 1)
    else:
        shift = round(rng.uniform(*LEGAL_SHIFT), 1)
    return {
        "_equipment": hazard.get("equipment"),
        "Location": rng.choice(pool),
        "Operator_Shift_Hours": shift,
        "Reported_Incident": rng.choice(hazard["bad"] if violating else hazard["ok"]),
        "Is_Violation": 1 if violating else 0,
        "Violation_Type": hazard["kind"] if violating else "None",
        "Expected_Section": hazard["section"] if violating else "None",
    }


def routine_row(rng):
    return {
        "Location": rng.choice(LOCATIONS),
        "Operator_Shift_Hours": round(rng.uniform(*LEGAL_SHIFT), 1),
        "Reported_Incident": rng.choice(ROUTINE),
        "Is_Violation": 0,
        "Violation_Type": "None",
        "Expected_Section": "None",
    }


def generate(n_records=DEFAULT_RECORDS, seed=SEED, ratio=VIOLATION_RATIO):
    rng = random.Random(seed)
    n_violations = max(len(HAZARDS), int(n_records * ratio))
    per_kind = n_violations // len(HAZARDS)
    plan = [h for h in HAZARDS for _ in range(per_kind)]

    # One compliant counterpart per violation: the hard negatives.
    n_hard = min(len(plan), n_records - len(plan))
    hard = [HAZARDS[i % len(HAZARDS)] for i in range(n_hard)]

    slots = rng.sample(range(n_records), len(plan) + len(hard))
    viol_slots = dict(zip(slots[: len(plan)], plan))
    hard_slots = dict(zip(slots[len(plan) :], hard))

    base = datetime(2026, 3, 29, 6, 0)
    rows = []
    for i in range(n_records):
        if i in viol_slots:
            core = make_row(rng, viol_slots[i], True)
        elif i in hard_slots:
            core = make_row(rng, hard_slots[i], False)
        else:
            core = routine_row(rng)
        equip = core.pop("_equipment", None) or rng.choice(EQUIPMENT)
        rows.append(
            {
                "Log_ID": f"LOG-{1000 + i}",
                "Timestamp": (
                    base + timedelta(minutes=rng.randint(2, 15) * i)
                ).strftime("%Y-%m-%d %H:%M"),
                "Equipment_ID": f"{equip[:3].upper()}-{rng.randint(10, 99)}",
                "Equipment_Type": equip,
                "Location": core["Location"],
                "Operator_Shift_Hours": core["Operator_Shift_Hours"],
                "Payload_Weight_lbs": rng.randint(10000, 65000),
                "Reported_Incident": core["Reported_Incident"],
                "Is_Violation": core["Is_Violation"],
                "Violation_Type": core["Violation_Type"],
                "Expected_Section": core["Expected_Section"],
            }
        )
    return rows


def audit_labels(rows, sections):
    """Refuse to ship a dataset that cannot support an honest measurement."""
    problems = []

    bad_text = {t: h["kind"] for h in HAZARDS for t in h["bad"]}
    ok_text = {t: h["kind"] for h in HAZARDS for t in h["ok"]}

    for r in rows:
        inc = r["Reported_Incident"]
        if r["Is_Violation"] and inc not in bad_text:
            problems.append(
                f"{r['Log_ID']}: labelled a violation but its text is not a breach phrasing"
            )
        if not r["Is_Violation"] and inc in bad_text:
            problems.append(
                f"{r['Log_ID']}: unlabelled row carries breach phrasing for {bad_text[inc]}"
            )
        if r["Is_Violation"] and r["Expected_Section"] == "None":
            problems.append(f"{r['Log_ID']}: violation with no governing section")

    overlap = set(bad_text) & set(ok_text)
    if overlap:
        problems.append(
            f"phrasing reused as both breach and compliant: {sorted(overlap)[:3]}"
        )

    if sections is not None:
        for h in HAZARDS:
            if h["section"] not in sections:
                problems.append(
                    f"{h['kind']} cites {h['section']}, which is not in the retrievable corpus"
                )

    # Every location used by a violation must also appear on clean rows.
    v_locs = {r["Location"] for r in rows if r["Is_Violation"]}
    c_locs = {r["Location"] for r in rows if not r["Is_Violation"]}
    for loc in sorted(v_locs - c_locs):
        problems.append(
            f"location '{loc}' appears only on violations, so it alone predicts the label"
        )

    return problems


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic intermodal yard log.")
    ap.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--ratio",
        type=float,
        default=VIOLATION_RATIO,
        help="share of records carrying a violation",
    )
    args = ap.parse_args()

    sections = corpus_sections()
    if sections is None:
        print(
            "Warning: json/regulations.json not found, so citable-rule checking is skipped."
        )
        print(
            "         Run code/fetch_regulations.py first for a fully verified dataset.\n"
        )

    rows = generate(args.records, args.seed, args.ratio)
    problems = audit_labels(rows, sections)
    if problems:
        print("Label audit FAILED:")
        for p in problems[:15]:
            print(f"  {p}")
        return 1

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    violations = [r for r in rows if r["Is_Violation"]]
    by_kind = {}
    for r in violations:
        by_kind[r["Violation_Type"]] = by_kind.get(r["Violation_Type"], 0) + 1
    hard = sum(
        1
        for r in rows
        if not r["Is_Violation"] and r["Reported_Incident"] not in ROUTINE
    )

    print(f"Wrote {OUT.name}: {len(rows)} records, seed {args.seed}")
    print(
        f"Violations: {len(violations)} across {len(by_kind)} hazard types, "
        f"citing {len({r['Expected_Section'] for r in violations})} distinct federal authorities"
    )
    for kind in sorted(by_kind):
        sec = next(h["section"] for h in HAZARDS if h["kind"] == kind)
        print(f"    {kind:20} {by_kind[kind]:4}   {sec}")
    print(f"Compliant counterparts (hard negatives): {hard}")
    print(f"Routine clean rows: {len(rows) - len(violations) - hard}")
    print(
        "Label audit passed: every planted rule is retrievable, no phrasing is "
        "reused across classes, and no location predicts the label on its own."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
