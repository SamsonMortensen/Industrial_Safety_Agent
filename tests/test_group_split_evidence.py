"""Recompute saved synthetic allocations and reject inconsistent evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'code'))
import check_group_split_results as evidence


class GroupSplitEvidenceTests(unittest.TestCase):
    def test_saved_examples_and_regressions_reproduce(self):
        result=evidence.check()
        self.assertEqual(result['stress_outcomes'],{'better':80,'tied':12,'worse':8})
        self.assertEqual(len(result['named_cases']),5)

    def test_baseline_counts_must_match_saved_group_assignments(self):
        data=json.loads(evidence.REPORT.read_text(encoding='utf-8'))
        data['named_cases'][0]['before']['per_split']['train']['samples']+=1
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'edited.json';path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Saved split metrics'):
                evidence.check(path)

    def test_fixture_and_candidate_source_identity_are_bound(self):
        original=json.loads(evidence.REPORT.read_text(encoding='utf-8'))
        for section in ('fixture','source'):
            data=copy.deepcopy(original)
            if section=='fixture':data['named_cases'][0]['fixture_sha256']='0'*64
            else:data['candidate_sources_sha256_lf']['code/vision_manifest.py']='0'*64
            with self.subTest(section=section),tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)/'edited.json';path.write_text(json.dumps(data),encoding='utf-8')
                with self.assertRaises(ValueError):evidence.check(path)

    def test_unrecognized_baseline_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'unrecognized.py'
            source.write_text('raise RuntimeError("This source must not run")\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'pinned published revision'):
                evidence.capture(source)


if __name__=='__main__':unittest.main()
