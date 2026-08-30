"""Validate and filter the public perception dataset registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "data" / "vision_datasets.json"
LICENSE_STATUSES = {
    "cleared_for_commercial_prototyping",
    "research_only",
    "rights_review_required",
}
REQUIRED_FIELDS = {
    "id",
    "name",
    "modality",
    "tasks",
    "scale",
    "labels",
    "official_url",
    "paper_url",
    "license",
    "license_status",
    "recommended_use",
    "caveats",
}


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_registry(payload: dict[str, Any]) -> list[str]:
    """Return every validation error instead of failing on the first one."""
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return errors + ["datasets must be a non-empty list"]

    seen: set[str] = set()
    for position, dataset in enumerate(datasets):
        prefix = f"datasets[{position}]"
        if not isinstance(dataset, dict):
            errors.append(f"{prefix} must be an object")
            continue

        missing = sorted(REQUIRED_FIELDS - set(dataset))
        if missing:
            errors.append(f"{prefix} missing fields: {', '.join(missing)}")
            continue

        dataset_id = dataset["id"]
        if not isinstance(dataset_id, str) or not dataset_id:
            errors.append(f"{prefix}.id must be a non-empty string")
        elif dataset_id in seen:
            errors.append(f"duplicate dataset id: {dataset_id}")
        else:
            seen.add(dataset_id)

        if dataset["license_status"] not in LICENSE_STATUSES:
            errors.append(f"{prefix}.license_status is not recognized")
        if not _is_http_url(dataset["official_url"]):
            errors.append(f"{prefix}.official_url must be an HTTP URL")
        if dataset["paper_url"] is not None and not _is_http_url(dataset["paper_url"]):
            errors.append(f"{prefix}.paper_url must be null or an HTTP URL")
        for list_field in ("tasks", "labels", "caveats"):
            value = dataset[list_field]
            if not isinstance(value, list) or not value:
                errors.append(f"{prefix}.{list_field} must be a non-empty list")
        if not isinstance(dataset["scale"], dict):
            errors.append(f"{prefix}.scale must be an object")

    return errors


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_registry(payload)
    if errors:
        raise ValueError("Invalid vision dataset registry:\n- " + "\n- ".join(errors))
    return payload


def eligible_datasets(payload: dict[str, Any], purpose: str) -> list[dict[str, Any]]:
    """Apply the license gate for research or potential commercial use."""
    if purpose == "research":
        allowed = {"cleared_for_commercial_prototyping", "research_only"}
    elif purpose == "commercial":
        allowed = {"cleared_for_commercial_prototyping"}
    else:
        raise ValueError("purpose must be 'research' or 'commercial'")
    return [item for item in payload["datasets"] if item["license_status"] in allowed]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--purpose", choices=("research", "commercial"), default="research"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable output."
    )
    args = parser.parse_args()

    payload = load_registry(args.registry)
    selected = eligible_datasets(payload, args.purpose)
    if args.json:
        print(json.dumps(selected, indent=2, ensure_ascii=False))
        return

    print(
        f"Validated {len(payload['datasets'])} datasets; {len(selected)} allowed for {args.purpose} use."
    )
    for item in selected:
        print(f"- {item['id']}: {item['license']} | {item['recommended_use']}")


if __name__ == "__main__":
    main()
