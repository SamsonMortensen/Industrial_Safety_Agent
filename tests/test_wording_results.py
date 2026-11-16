"""Recompute the published comparison and check its source provenance."""
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from evaluate_wording_retrieval import aggregate, load_cases, fixture_provenance, input_snapshot
from showcase import load_corpus


class WordingResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((ROOT/'docs/results/wording_retrieval_comparison.json').read_text(encoding='utf-8'))

    def assert_metrics_match(self, actual, expected, path=()):
        self.assertIs(type(actual), type(expected), path)
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys(), path)
            for key in expected:
                self.assert_metrics_match(actual[key], expected[key], (*path, key))
        elif isinstance(expected, float) and path[-1:] in (('mean',), ('group_mean',), ('rate',)):
            # Python versions can round an average differently in its last bits.
            # Counts, keys, types, and other values still have to match exactly.
            self.assertTrue(math.isfinite(actual) and math.isfinite(expected), path)
            self.assertTrue(0.0 <= actual <= 1.0 and 0.0 <= expected <= 1.0, path)
            self.assertTrue(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12),
                            f'{path}: {actual!r} != {expected!r}')
        else:
            self.assertEqual(actual, expected, path)

    def test_metric_comparison_allows_only_small_ratio_roundoff(self):
        for key in ('mean', 'group_mean', 'rate'):
            self.assert_metrics_match({key: 0.5476190476190477, 'count': 42},
                                      {key: 0.5476190476190476, 'count': 42})

    def test_metric_comparison_rejects_changed_results_and_invalid_values(self):
        expected = {'count': 42, 'rate': 0.5, 'missing': None, 'accepted': False}
        invalid = [
            {**expected, 'count': 43},
            {**expected, 'count': 42.0},
            {**expected, 'rate': 0.50000001},
            {**expected, 'rate': float('nan')},
            {**expected, 'rate': float('inf')},
            {**expected, 'missing': 0},
            {**expected, 'accepted': 0},
            {key: value for key, value in expected.items() if key != 'count'},
            {**expected, 'extra': 1},
        ]
        for actual in invalid:
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                self.assert_metrics_match(actual, expected)
        for value in (-1e-15, 1.0 + 1e-15, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                self.assert_metrics_match({'rate': value}, {'rate': value})
        with self.assertRaises(AssertionError):
            self.assert_metrics_match({'rank': 1.0000000000000002}, {'rank': 1.0})

    def test_saved_metrics_recompute_from_every_record(self):
        self.assertEqual(len(self.payload['runs']), 6)
        for run in self.payload['runs'].values():
            self.assertEqual(len(run['records']), 48)
            rebuilt = aggregate(run['records'])
            self.assert_metrics_match(rebuilt['metrics'], run['metrics'])
            self.assert_metrics_match(rebuilt['group_metrics'], run['group_metrics'])
        for flag in ('independent_evaluation_used','calibration_reserve_used','training_performed','model_promoted'):
            self.assertIs(self.payload[flag], False)

    def test_groups_and_public_fixture_bytes_are_bound(self):
        directory = ROOT/'data/wording_retrieval'
        protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
        groups = []
        for name,key in (('development','development'),('comparison','frozen_comparison')):
            path = directory/(name+'.json')
            digest = hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            self.assertEqual(digest, protocol[key]['sha256'])
            fixture = json.loads(path.read_text(encoding='utf-8'))
            loaded, _ = load_cases(path)
            self.assertEqual(loaded, fixture)
            self.assertTrue(fixture_provenance(path, fixture, input_snapshot())['fixture_matches_protocol'])
            groups.append({r['group_id'] for r in fixture['cases']})
            self.assertEqual(len(groups[-1]),16)
        self.assertFalse(groups[0] & groups[1])

    def test_published_excerpt_offsets_point_to_real_corpus_text(self):
        chunks = load_corpus(ROOT)
        for run in self.payload['runs'].values():
            for row in run['records']:
                for item in row['result']['candidates']:
                    text = chunks[item['excerpt_chunk_index']]['text']
                    self.assertEqual(chunks[item['excerpt_chunk_index']]['section'], item['section'])
                    self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), item['excerpt_chunk_sha256'])
                    self.assertTrue(0 <= item['excerpt_start'] <= item['excerpt_end'] <= len(text))


if __name__ == '__main__':
    unittest.main()
