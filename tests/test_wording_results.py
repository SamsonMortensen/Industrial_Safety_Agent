"""Recompute the published comparison and check its source provenance."""
import hashlib
import json
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

    def test_saved_metrics_recompute_from_every_record(self):
        self.assertEqual(len(self.payload['runs']), 6)
        for run in self.payload['runs'].values():
            self.assertEqual(len(run['records']), 48)
            rebuilt = aggregate(run['records'])
            self.assertEqual(rebuilt['metrics'], run['metrics'])
            self.assertEqual(rebuilt['group_metrics'], run['group_metrics'])
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
