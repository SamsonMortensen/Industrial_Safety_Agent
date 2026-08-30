"""Validate and inspect the authentic incident-text source registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "data" / "incident_text_datasets.json"
RIGHTS_STATUSES = {"federal_public_data", "federal_public_reports"}
REQUIRED_FIELDS = {
    "id",
    "name",
    "agency",
    "coverage",
    "content",
    "official_url",
    "access",
    "rights_status",
    "recommended_use",
    "prohibited_inferences",
    "caveats",
}


def validate_registry(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return errors + ["datasets must be a non-empty list"]

    seen: set[str] = set()
    for index, dataset in enumerate(datasets):
        prefix = f"datasets[{index}]"
        if not isinstance(dataset, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = sorted(REQUIRED_FIELDS - set(dataset))
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(missing)}")
            continue
        if dataset["id"] in seen:
            errors.append(f"duplicate dataset id: {dataset['id']}")
        seen.add(dataset["id"])
        parsed = urlparse(dataset["official_url"])
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            errors.append(f"{prefix}.official_url must be an HTTP URL")
        if dataset["rights_status"] not in RIGHTS_STATUSES:
            errors.append(f"{prefix}.rights_status is not recognized")
        for field in ("content", "prohibited_inferences", "caveats"):
            if not isinstance(dataset[field], list) or not dataset[field]:
                errors.append(f"{prefix}.{field} must be a non-empty list")
    return errors


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_registry(payload)
    if errors:
        raise ValueError("Invalid incident dataset registry:\n- " + "\n- ".join(errors))
    return payload


def select_by_content(
    payload: dict[str, Any], term: str | None
) -> list[dict[str, Any]]:
    if not term:
        return payload["datasets"]
    normalized = term.casefold()
    return [
        dataset
        for dataset in payload["datasets"]
        if any(normalized in item.casefold() for item in dataset["content"])
        or normalized in dataset["recommended_use"].casefold()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--contains", help="Filter by a content or recommended-use term."
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = load_registry(args.registry)
    selected = select_by_content(payload, args.contains)
    if args.json:
        print(json.dumps(selected, indent=2, ensure_ascii=False))
        return
    print(
        f"Validated {len(payload['datasets'])} authentic incident sources; selected {len(selected)}."
    )
    for item in selected:
        print(f"- {item['id']}: {item['recommended_use']}")


if __name__ == "__main__":
    main()
