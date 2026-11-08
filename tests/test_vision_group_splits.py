"""Synthetic allocation regressions; never open evaluation data or train a model."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from vision_manifest import assign_group_splits, validate_no_group_leakage
from vision_split_examples import named_examples, stress_examples, summarize


def split(example):
    return assign_group_splits(example["rows"], seed=example["seed"],
                               train_ratio=example["train_ratio"],
                               validation_ratio=example["validation_ratio"])


class VisionGroupSplitTests(unittest.TestCase):
    def setUp(self):
        self.examples = {row["id"]: row for row in named_examples()}

    def counts(self, name):
        actual = Counter(row["split"] for row in split(self.examples[name]))
        return tuple(actual[name] for name in ("train", "validation", "test"))

    def test_forty_equal_groups_keep_requested_row_and_label_balance(self):
        rows = split(self.examples["balanced_forty_groups"])
        self.assertEqual(self.counts("balanced_forty_groups"), (84, 18, 18))
        report = summarize(rows, {"train": .7, "validation": .15, "test": .15})
        self.assertAlmostEqual(report["allocation_objective"], 0)

    def test_hundred_equal_groups_do_not_invert_partition_ratios(self):
        self.assertEqual(self.counts("balanced_hundred_groups"), (210, 45, 45))

    def test_six_groups_preserve_small_cohort_training_allocation(self):
        self.assertEqual(self.counts("small_six_groups"), (12, 3, 3))

    def test_custom_ratios_balance_total_rows_as_well_as_labels(self):
        self.assertEqual(self.counts("custom_twelve_groups"), (18, 9, 9))

    def test_repeated_and_reordered_rows_keep_the_same_group_assignments(self):
        for example in self.examples.values():
            with self.subTest(example=example["id"]):
                before = deepcopy(example["rows"])
                first = split(example)
                self.assertEqual(first, split(example))
                shuffled = deepcopy(example)
                random.Random(123).shuffle(shuffled["rows"])
                second = split(shuffled)
                self.assertEqual({r["sample_id"]: r["split"] for r in first},
                                 {r["sample_id"]: r["split"] for r in second})
                self.assertEqual(example["rows"], before)

    def test_uneven_and_mixed_label_groups_stay_whole(self):
        example = self.examples["uneven_mixed_groups"]
        rows = split(example)
        validate_no_group_leakage(rows)
        self.assertEqual(len(rows), len(example["rows"]))
        self.assertEqual({r["split"] for r in rows}, {"train", "validation", "test"})
        self.assertEqual(Counter(r["sample_id"] for r in rows),
                         Counter(r["sample_id"] for r in example["rows"]))
        for old, new in zip(example["rows"], rows):
            self.assertEqual({k: v for k, v in new.items() if k != "split"}, old)

    def test_three_indivisible_groups_keep_each_partition_nonempty(self):
        rows = [{"sample_id": str(i), "group_id": str(i), "label": "one"} for i in range(3)]
        result = assign_group_splits(rows, train_ratio=.98, validation_ratio=.01)
        self.assertEqual(Counter(r["split"] for r in result),
                         {"train": 1, "validation": 1, "test": 1})

    def test_nonfinite_and_nonpositive_ratios_fail_before_assignment(self):
        rows = self.examples["small_six_groups"]["rows"]
        invalid = [(float("nan"), .15), (.7, float("nan")), (float("inf"), .15),
                   (.7, float("inf")), (-float("inf"), .15), (.7, -float("inf")),
                   (0, .15), (.7, 0), (-.1, .15), (.7, -.1), (.9, .1), (.7, .3), (.8, .3)]
        for train, validation in invalid:
            with self.subTest(train=train, validation=validation):
                with self.assertRaisesRegex(ValueError, "finite.*positive"):
                    assign_group_splits(rows, train_ratio=train, validation_ratio=validation)

    def test_stress_examples_preserve_groups_rows_and_determinism(self):
        examples = stress_examples()
        self.assertEqual(examples, stress_examples())
        self.assertEqual(len(examples), 100)
        self.assertEqual(len({e["id"] for e in examples}), 100)
        for example in examples:
            with self.subTest(example=example["id"]):
                rows = split(example)
                self.assertEqual(rows, split(example))
                validate_no_group_leakage(rows)
                self.assertEqual(len(rows), len(example["rows"]))
                self.assertEqual({r["split"] for r in rows}, {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
