"""Deterministic shift-level rules for fatigue and repeat equipment defects.

What this is, and is not
------------------------
This is a rule engine, not a model. It applies two explicit thresholds to
structured fields and emits citations that are authored constants, not anything
retrieved or reasoned. Its output is tagged source="deterministic_rule" so it can
never be mistaken for, or aggregated with, an LLM audit result.

It exists because a per-row prompt cannot see hazards that accumulate over a
shift:

  1. Cumulative duty time crossing the 12.0-hour ceiling.
  2. The same asset logging repeated telemetry conditions inside one shift,
     which calls for tag-out rather than continued operation.

Both rules are stated below in code and both citations are hardcoded, which is
appropriate for a deterministic check and inappropriate anywhere it might be
reported as detection performance.

    python code/stateful_tracker.py
"""

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "daily_yard_log.csv"
OUT = ROOT / "json" / "stateful_audit.json"

# These declared terms indicate equipment conditions, not legal violations.
DEFECT_TERMS = (
    "leak",
    "imbalance",
    "pressure",
    "alarm",
    "warning",
    "defect",
    "brake",
    "frayed",
    "slack",
    "crack",
    "damaged",
    "inoperative",
)


class YardStateTracker:
    """Maintains rolling state across intermodal yard operations."""

    def __init__(self, fatigue_limit_hours=12.0, defect_escalation_count=2):
        self.fatigue_limit = fatigue_limit_hours
        self.defect_limit = defect_escalation_count
        self.equipment_history = defaultdict(list)
        self.operator_history = defaultdict(list)

    def process_event(self, row):
        """Evaluate an event in the context of preceding shift history.

        Returns a list of stateful findings (if any).
        """
        findings = []
        log_id = row["Log_ID"]
        timestamp_str = row["Timestamp"]
        t = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
        equip = row["Equipment_ID"]
        incident = row["Reported_Incident"]
        shift_hrs = float(row["Operator_Shift_Hours"])

        # 1. Operator shift duration check
        if shift_hrs > self.fatigue_limit:
            findings.append(
                {
                    "type": "fatigue_violation",
                    "source": "deterministic_rule",
                    "citation": "49 U.S.C. 21103",
                    "severity": "CRITICAL",
                    "detail": f"Operator on duty {shift_hrs:.1f} hrs exceeds {self.fatigue_limit:.1f} hr statutory limit.",
                }
            )

        # 2. Equipment defect escalation tracking
        history = self.equipment_history[equip]
        cutoff = t - timedelta(hours=8)
        recent_defects = [
            e for e in history if e["time"] >= cutoff and e["incident"] != "None"
        ]

        if any(term in incident.lower() for term in DEFECT_TERMS):
            if len(recent_defects) >= self.defect_limit:
                findings.append(
                    {
                        "type": "equipment_tagout_required",
                        "source": "deterministic_rule",
                        "citation": "29 CFR 1910.178(q)(7)",
                        "severity": "WARNING",
                        "detail": (
                            f"Equipment {equip} has logged {len(recent_defects) + 1} recurring defects "
                            f"within 8 hours without record of repair. Mandatory tag-out required."
                        ),
                    }
                )
            self.equipment_history[equip].append(
                {"time": t, "incident": incident, "log_id": log_id}
            )

        return findings


def run_stateful_audit():
    with LOG.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    tracker = YardStateTracker()
    audit_log = []

    for row in rows:
        findings = tracker.process_event(row)
        if findings:
            audit_log.append(
                {
                    "log_id": row["Log_ID"],
                    "timestamp": row["Timestamp"],
                    "equipment": row["Equipment_ID"],
                    "findings": findings,
                }
            )

    print(f"Stateful tracking processed {len(rows)} events.")
    print(f"Detected {len(audit_log)} shift-level state findings:")
    for entry in audit_log:
        for f in entry["findings"]:
            print(
                f"  [{entry['log_id']} @ {entry['timestamp']}] {f['citation']} ({f['type']}): {f['detail']}"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit_log, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.name}")
    return audit_log


if __name__ == "__main__":
    run_stateful_audit()
