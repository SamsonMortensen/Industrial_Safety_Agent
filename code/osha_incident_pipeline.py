"""Normalize OSHA reports without retaining structured establishment identifiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT / "data" / "external" / "osha_sir" / "January2015toNovember2025.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "external" / "osha_sir" / "normalized_observations.jsonl"
)
DEFAULT_PROFILE = ROOT / "json" / "osha_sir_profile.json"
SOURCE_URL = "https://www.osha.gov/severe-injury-reports"
PII_FIELDS_REMOVED = [
    "Employer",
    "Address1",
    "Address2",
    "City",
    "Zip",
    "Latitude",
    "Longitude",
    "UPA",
]
REQUIRED_COLUMNS = {
    "ID",
    "EventDate",
    "Employer",
    "Address1",
    "City",
    "State",
    "Zip",
    "Latitude",
    "Longitude",
    "Primary NAICS",
    "Hospitalized",
    "Amputation",
    "Loss of Eye",
    "Inspection",
    "Final Narrative",
    "Nature",
    "NatureTitle",
    "Part of Body",
    "Part of Body Title",
    "Event",
    "EventTitle",
    "Source",
    "SourceTitle",
    "Secondary Source",
    "Secondary Source Title",
    "FederalState",
}
TARGET_NAICS_PREFIXES = {
    "482": "rail_transportation",
    "4882": "support_activities_for_rail_transportation",
    "4883": "support_activities_for_water_transportation",
    "4931": "warehousing_and_storage",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: str) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _clean(value: str) -> str:
    return " ".join((value or "").split())


def _date_key(value: str) -> str:
    for format_string in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, format_string).date().isoformat()
        except ValueError:
            continue
    return value


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    """Create an Observation mapping while removing establishment identifiers."""
    record_id = _clean(row.get("ID", ""))
    return {
        "event_id": f"osha-sir-{record_id}",
        "timestamp": _date_key(_clean(row.get("EventDate", ""))),
        "source_id": "osha-severe-injury-report",
        "summary": _clean(row.get("Final Narrative", "")),
        "actors": [],
        "equipment": [],
        "location": _clean(row.get("State", "")),
        "actions": [],
        "conditions": [],
        "controls": [],
        "measurements": {},
        "relations": [],
        "perception_confidence": 1.0,
        "anomaly_score": 0.0,
        "novelty_score": 0.0,
        "changed": True,
        "human_flag": True,
        "evidence_refs": [f"{SOURCE_URL}#record-{record_id}"],
        "metadata": {
            "naics": _clean(row.get("Primary NAICS", "")),
            "state": _clean(row.get("State", "")),
            "hospitalized": _integer(row.get("Hospitalized", "")),
            "amputation": _integer(row.get("Amputation", "")),
            "loss_of_eye": _integer(row.get("Loss of Eye", "")),
            "inspection_present": bool(_clean(row.get("Inspection", ""))),
            "oiics": {
                "nature_code": _clean(row.get("Nature", "")),
                "nature_title": _clean(row.get("NatureTitle", "")),
                "body_part_code": _clean(row.get("Part of Body", "")),
                "body_part_title": _clean(row.get("Part of Body Title", "")),
                "event_code": _clean(row.get("Event", "")),
                "event_title": _clean(row.get("EventTitle", "")),
                "source_code": _clean(row.get("Source", "")),
                "source_title": _clean(row.get("SourceTitle", "")),
                "secondary_source_code": _clean(row.get("Secondary Source", "")),
                "secondary_source_title": _clean(row.get("Secondary Source Title", "")),
            },
        },
    }


def read_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"OSHA SIR CSV missing columns: {', '.join(sorted(missing))}"
            )
        yield from reader


def build_profile(rows: list[dict[str, str]], source_path: Path) -> dict[str, Any]:
    dates = sorted(
        _date_key(_clean(row.get("EventDate", "")))
        for row in rows
        if row.get("EventDate")
    )
    narrative_count = sum(bool(_clean(row.get("Final Narrative", ""))) for row in rows)
    event_titles = Counter(_clean(row.get("EventTitle", "")) for row in rows)
    event_titles.pop("", None)
    target_counts = Counter()
    for row in rows:
        naics = _clean(row.get("Primary NAICS", ""))
        for prefix, name in TARGET_NAICS_PREFIXES.items():
            if naics.startswith(prefix):
                target_counts[name] += 1
                break

    return {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "official_source": SOURCE_URL,
            "source_file": source_path.name,
            "source_sha256": sha256(source_path),
            "structured_identifier_fields_removed": True,
            "narrative_redaction_performed": False,
            "pii_fields_removed": PII_FIELDS_REMOVED,
            "narrative_used_for_queries": True,
            "oiics_codes_used_for_queries": False,
            "outcomes_used_for_queries": False,
        },
        "records": len(rows),
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
        "narratives_present": narrative_count,
        "narrative_coverage": round(narrative_count / len(rows), 6) if rows else 0.0,
        "states_and_territories": len(
            {_clean(row.get("State", "")) for row in rows if row.get("State")}
        ),
        "outcomes": {
            "hospitalized_workers": sum(
                _integer(row.get("Hospitalized", "")) for row in rows
            ),
            "amputation_reports": sum(
                _integer(row.get("Amputation", "")) > 0 for row in rows
            ),
            "loss_of_eye_reports": sum(
                _integer(row.get("Loss of Eye", "")) > 0 for row in rows
            ),
        },
        "inspection_linked_records": sum(
            bool(_clean(row.get("Inspection", ""))) for row in rows
        ),
        "target_domain_records": dict(sorted(target_counts.items())),
        "top_event_titles": [
            {"title": title, "records": count}
            for title, count in event_titles.most_common(10)
        ],
        "limitations": [
            "The source excludes fatalities and incidents outside federal OSHA jurisdiction.",
            "Every row is a severe reported outcome, so this is not a negative-control dataset.",
            "Narratives may reveal the injury outcome and must not be used to claim pre-incident detection accuracy.",
            "OSHA SIR records do not provide authoritative regulation labels.",
            "Free-text narratives are not redacted and remain in ignored local storage.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source_path = args.input.resolve()
    rows = list(read_rows(source_path))
    if args.limit > 0:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(normalize_row(row), ensure_ascii=False) + "\n")

    profile = build_profile(rows, source_path)
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    args.profile.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: profile[key]
                for key in (
                    "records",
                    "date_start",
                    "date_end",
                    "narrative_coverage",
                    "target_domain_records",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
