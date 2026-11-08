"""Bounded synthetic examples for inspecting source-group allocation.

No media are read or created. Labels are arbitrary fixture categories, and the
allocation score measures split balance, not model performance or independence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import math
import random
from typing import Any, Mapping

SPLITS = ("train", "validation", "test")


def _row(group: str, frame: int, label: str, day: int = 0) -> dict[str, str]:
    return {
        "sample_id": f"{group}-f{frame}",
        "source_path": f"synthetic/{group}.mp4",
        "label": label,
        "group_id": group,
        "site_id": "synthetic-site",
        "camera_id": "synthetic-camera",
        "captured_at": (datetime(2026, 1, 1, tzinfo=timezone.utc)
                        + timedelta(days=day)).isoformat(),
    }


def _balanced_rows(groups: int) -> list[dict[str, str]]:
    return [_row(f"camera-a-day-{group}", frame, f"label-{group % 2}", group)
            for group in range(groups) for frame in range(3)]


def named_examples() -> list[dict[str, Any]]:
    """Return five illustrative cases, including uneven and mixed-label groups."""
    cases = [
        ("small_six_groups", 6, .70, .15),
        ("custom_twelve_groups", 12, .50, .25),
        ("balanced_forty_groups", 40, .70, .15),
        ("balanced_hundred_groups", 100, .70, .15),
    ]
    examples = [{"id": name, "rows": _balanced_rows(count), "seed": 21,
                 "train_ratio": train, "validation_ratio": validation}
                for name, count, train, validation in cases]
    uneven = []
    for group, size in enumerate((17, 11, 8, 5, 3, 2, 1)):
        uneven.extend(_row(f"uneven-{group}", frame, f"label-{(frame + group) % 3}", group)
                      for frame in range(size))
    examples.append({"id": "uneven_mixed_groups", "rows": uneven, "seed": 21,
                     "train_ratio": .70, "validation_ratio": .15})
    return examples


def stress_examples() -> list[dict[str, Any]]:
    """Return 100 fixed exploratory cases; these are not an evaluation sample."""
    rng = random.Random(42)
    examples = []
    for trial in range(100):
        rows = []
        for group in range(rng.randint(3, 20)):
            for frame in range(rng.randint(1, 12)):
                rows.append(_row(f"g{group}", frame, f"l{rng.randrange(3)}", group))
        examples.append({"id": f"stress_{trial:03d}", "rows": rows, "seed": 21,
                         "train_ratio": .70, "validation_ratio": .15})
    return examples


def summarize(assignments: list[dict[str, str]], ratios: Mapping[str, float]) -> dict[str, Any]:
    """Describe allocation and its combined target-weighted squared error."""
    if set(ratios) != set(SPLITS) or any(not math.isfinite(r) or r <= 0 for r in ratios.values()):
        raise ValueError("Expected finite positive train, validation, and test ratios")
    if not math.isclose(math.fsum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Split ratios must sum to one")
    if not assignments or any(row["split"] not in SPLITS for row in assignments):
        raise ValueError("Expected nonempty rows with supported split names")
    totals = Counter(row["label"] for row in assignments)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        group_splits[row["group_id"]].add(row["split"])
    per_split = {}
    terms = []
    for split in SPLITS:
        selected = [row for row in assignments if row["split"] == split]
        labels = Counter(row["label"] for row in selected)
        per_split[split] = {
            "samples": len(selected),
            "groups": len({row["group_id"] for row in selected}),
            "labels": dict(sorted(labels.items())),
        }
        target = len(assignments) * ratios[split]
        terms.append((len(selected) - target) ** 2 / target)
        for label in sorted(totals):
            target = totals[label] * ratios[split]
            terms.append((labels[label] - target) ** 2 / target)
    return {
        "per_split": per_split,
        "allocation_objective": math.fsum(terms),
        "row_count": len(assignments),
        "group_count": len(group_splits),
        "group_leakage": any(len(splits) > 1 for splits in group_splits.values()),
    }
