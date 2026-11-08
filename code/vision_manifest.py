"""Create leakage-resistant train, validation, and test splits for vision samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = {
    "sample_id",
    "source_path",
    "label",
    "group_id",
    "site_id",
    "camera_id",
    "captured_at",
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest contains no samples")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("sample_id values must be unique")
    if any(not row["group_id"] for row in rows):
        raise ValueError("Every sample needs a group_id")
    return rows


def _stable_seed(seed: int, group_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def assign_group_splits(
    rows: list[dict[str, str]],
    seed: int = 17,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> list[dict[str, str]]:
    """Greedily balance row and label counts without splitting source groups.

    Ratios are approximate: indivisible groups and label balance can conflict.
    This is a development split heuristic, not an independent evaluation design.
    """
    ratios = {
        "train": train_ratio,
        "validation": validation_ratio,
        "test": 1 - train_ratio - validation_ratio,
    }
    if (any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios.values())
            or train_ratio + validation_ratio >= 1):
        raise ValueError(
            "Ratios must be finite and leave positive train, validation, and test partitions"
        )

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required")

    totals = Counter(row["label"] for row in rows)
    targets = {name: {label: count * ratio for label, count in totals.items()}
               for name, ratio in ratios.items()}
    row_targets = {name: len(rows) * ratio for name, ratio in ratios.items()}
    split_counts = {name: Counter() for name in targets}
    split_sizes: Counter[str] = Counter()
    assignments: dict[str, str] = {}

    ordered_groups = sorted(
        groups,
        key=lambda group_id: (
            -len(groups[group_id]),
            _stable_seed(seed, group_id),
            group_id,
        ),
    )
    for group_id in ordered_groups:
        labels = Counter(row["label"] for row in groups[group_id])
        size = len(groups[group_id])
        candidates = []
        for split_name in ("train", "validation", "test"):
            # Compare the increase in sum((count - target)**2 / target),
            # for both total rows and each label. The -2*size terms are
            # identical for every split, so they do not affect the choice.
            terms = [(2 * split_sizes[split_name] * size + size * size)
                     / row_targets[split_name]]
            for label in sorted(labels):
                count = labels[label]
                current = split_counts[split_name][label]
                terms.append((2 * current * count + count * count)
                             / targets[split_name][label])
            candidates.append((math.fsum(terms), split_name))
        minimum = min(score for score, _ in candidates)
        tied = [name for score, name in candidates if abs(score - minimum) < 1e-12]
        chosen = random.Random(_stable_seed(seed, group_id)).choice(tied)
        assignments[group_id] = chosen
        split_counts[chosen].update(labels)
        split_sizes[chosen] += size

    # A small manifest can leave a split empty. Move one whole group if needed.
    group_counts = Counter(assignments.values())
    for empty_split in (name for name in targets if group_counts[name] == 0):
        donor = max(targets, key=lambda name: group_counts[name])
        movable = [
            group_id for group_id, split in assignments.items() if split == donor
        ]
        moved = min(movable, key=lambda group_id: len(groups[group_id]))
        assignments[moved] = empty_split
        group_counts[donor] -= 1
        group_counts[empty_split] += 1

    output = []
    for row in rows:
        output.append({**row, "split": assignments[row["group_id"]]})
    validate_no_group_leakage(output)
    return output


def validate_no_group_leakage(rows: list[dict[str, str]]) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
    leaked = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"Groups appear in more than one split: {', '.join(leaked)}")


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(rows: list[dict[str, str]], source_path: Path) -> dict[str, Any]:
    per_split = {}
    for split_name in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split_name]
        per_split[split_name] = {
            "samples": len(selected),
            "groups": len({row["group_id"] for row in selected}),
            "labels": dict(sorted(Counter(row["label"] for row in selected).items())),
        }
    return {
        "source_manifest": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "split_unit": "group_id",
        "per_split": per_split,
        "group_leakage": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows = assign_group_splits(read_manifest(args.manifest), seed=args.seed)
    write_manifest(rows, args.out)
    report = build_report(rows, args.manifest)
    report_path = args.report or args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
