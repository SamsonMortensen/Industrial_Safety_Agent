import csv
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from hybrid_policy import evaluate_event


class WhitespaceInvarianceTests(unittest.TestCase):
    def test_existing_development_decisions_are_invariant_to_whitespace(self):
        with (ROOT / 'data/scaled_benchmark_holdout.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            expected = evaluate_event(row).to_dict()
            for separator in ('  ', '\t', '\n', '\r\n', '\u00a0'):
                changed = dict(row)
                for field in ('Equipment_Type', 'Location', 'Reported_Incident'):
                    changed[field] = separator.join(changed[field].split())
                with self.subTest(case=row['Log_ID'], separator=repr(separator)):
                    self.assertEqual(evaluate_event(changed).to_dict(), expected)

    def test_uncertainty_phrases_stay_reviewable_when_pasted_with_tabs(self):
        row = {'Reported_Incident': 'Cannot\tconfirm whether the forklift has a functioning horn'}
        self.assertEqual(evaluate_event(row).rule, 'uncertain_evidence')

    def test_raw_input_and_other_fields_are_preserved(self):
        row = {'Reported_Incident': 'Cannot\tconfirm\n whether brakes work', 'Operator_Shift_Hours': '8',
               'Equipment_ID': 'source  001', 'Evidence_Ref': 'raw\tvalue'}
        original = deepcopy(row)
        evaluate_event(row)
        self.assertEqual(row, original)

    def test_normalization_does_not_join_separate_words(self):
        row = {'Reported_Incident': 'Un known condition'}
        self.assertNotEqual(evaluate_event(row).rule, 'uncertain_evidence')


if __name__ == '__main__':
    unittest.main()
