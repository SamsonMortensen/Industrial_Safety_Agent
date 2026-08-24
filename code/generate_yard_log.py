"""Generate the synthetic yard log used as ground truth for the audit.

The generator is seeded, and clean rows are drawn from pools that cannot
accidentally reproduce a violation pattern. Every row carries an explicit
Is_Violation flag and a Violation_Type, so precision and recall are computable.

    python code/generate_yard_log.py                # 1,000 records (default)
    python code/generate_yard_log.py --records 50   # 50 records (quick test)

Violation types, and the rule each one breaks:
  fatigue     operator shift over 12 hours          49 CFR 228 (hours of service)
  spill       hazardous leak in a pedestrian zone   29 CFR 1910.22 (housekeeping)
  electrical  equipment inside a high-voltage zone  29 CFR 1910.333 (clearances)
"""
import argparse
import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "daily_yard_log.csv"

SEED = 20260329
DEFAULT_RECORDS = 1000
VIOLATION_RATIO = 0.06  # ~6% violation rate across operations

EQUIPMENT = ['Gantry Crane', 'Reach Stacker', 'Forklift', 'Terminal Tractor']

ROUTINE_LOCATIONS = ['Bay 1', 'Bay 2', 'Bay 4', 'Maintenance Track', 'Rail Siding 3']
PEDESTRIAN_ZONE = 'Main Pedestrian Crosswalk'
HIGH_VOLTAGE_ZONE = 'High-Voltage Line B'

BENIGN_INCIDENTS = ['None', 'None', 'None', 'None', 'Load Imbalance', 'Tire Pressure Warning']
SPILL_INCIDENT = 'Hydraulic Leak'
PROXIMITY_INCIDENT = 'Proximity Warning'

LEGAL_SHIFT = (6.0, 11.5)
FATIGUE_SHIFT = (12.5, 15.0)


def clean_record(rng):
    return {
        'Location': rng.choice(ROUTINE_LOCATIONS),
        'Operator_Shift_Hours': round(rng.uniform(*LEGAL_SHIFT), 1),
        'Reported_Incident': rng.choice(BENIGN_INCIDENTS),
        'Is_Violation': 0,
        'Violation_Type': 'None',
    }


def violation_record(rng, kind):
    if kind == 'fatigue':
        return {
            'Location': rng.choice(ROUTINE_LOCATIONS),
            'Operator_Shift_Hours': round(rng.uniform(*FATIGUE_SHIFT), 1),
            'Reported_Incident': rng.choice(['None', 'None', 'Load Imbalance']),
            'Is_Violation': 1,
            'Violation_Type': 'fatigue',
        }
    if kind == 'spill':
        return {
            'Location': PEDESTRIAN_ZONE,
            'Operator_Shift_Hours': round(rng.uniform(*LEGAL_SHIFT), 1),
            'Reported_Incident': SPILL_INCIDENT,
            'Is_Violation': 1,
            'Violation_Type': 'spill',
        }
    if kind == 'electrical':
        return {
            'Location': HIGH_VOLTAGE_ZONE,
            'Operator_Shift_Hours': round(rng.uniform(*LEGAL_SHIFT), 1),
            'Reported_Incident': PROXIMITY_INCIDENT,
            'Is_Violation': 1,
            'Violation_Type': 'electrical',
        }
    raise ValueError(kind)


def generate(n_records=DEFAULT_RECORDS, seed=SEED):
    rng = random.Random(seed)
    
    n_violations = max(9, int(n_records * VIOLATION_RATIO))
    n_per_kind = n_violations // 3
    kinds = ['fatigue', 'spill', 'electrical'] * n_per_kind
    
    slots = rng.sample(range(n_records), len(kinds))
    planted = dict(zip(slots, kinds))

    base_time = datetime(2026, 3, 29, 6, 0)
    rows = []
    for i in range(n_records):
        core = violation_record(rng, planted[i]) if i in planted else clean_record(rng)
        equip = rng.choice(EQUIPMENT)
        record_time = base_time + timedelta(minutes=rng.randint(2, 15) * i)
        rows.append({
            'Log_ID': f"LOG-{1000 + i}",
            'Timestamp': record_time.strftime('%Y-%m-%d %H:%M'),
            'Equipment_ID': f"{equip[:3].upper()}-{rng.randint(10, 99)}",
            'Equipment_Type': equip,
            'Location': core['Location'],
            'Operator_Shift_Hours': core['Operator_Shift_Hours'],
            'Payload_Weight_lbs': rng.randint(10000, 65000),
            'Reported_Incident': core['Reported_Incident'],
            'Is_Violation': core['Is_Violation'],
            'Violation_Type': core['Violation_Type'],
        })
    return rows


def audit_labels(rows):
    problems = []
    for r in rows:
        hazards = [
            r['Operator_Shift_Hours'] > 12.0,
            r['Location'] == PEDESTRIAN_ZONE and r['Reported_Incident'] == SPILL_INCIDENT,
            r['Location'] == HIGH_VOLTAGE_ZONE and r['Reported_Incident'] == PROXIMITY_INCIDENT,
        ]
        if any(hazards) != bool(r['Is_Violation']):
            problems.append(r['Log_ID'])
        if not r['Is_Violation'] and r['Location'] in (PEDESTRIAN_ZONE, HIGH_VOLTAGE_ZONE):
            problems.append(r['Log_ID'])
    return problems


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic intermodal yard log.")
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS, help="Number of records to generate")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    args = parser.parse_args()

    rows = generate(n_records=args.records, seed=args.seed)
    problems = audit_labels(rows)
    if problems:
        print(f"Label audit FAILED on {sorted(set(problems))}")
        return 1

    with OUT.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    violations = [r for r in rows if r['Is_Violation']]
    by_kind = {}
    for r in violations:
        by_kind[r['Violation_Type']] = by_kind.get(r['Violation_Type'], 0) + 1

    print(f"Wrote {OUT.name}: {len(rows)} records, seed {args.seed}")
    print(f"Labelled violations: {len(violations)}  {by_kind}")
    print(f"Label audit passed -- no unlabelled row matches a violation pattern.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
