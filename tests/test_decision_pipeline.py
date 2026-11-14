"""Regression checks for effective decisions across parsing, review and scoring."""
import asyncio
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import audit_agent as legacy
import apply_policy_controls as controls
from audit_outcomes import parse_audit_response, prediction_from_status
from learning_policy import assess_for_learning
import self_reflection as reflection
import validate_saved_evaluation as saved


def event(number=0, actual=0):
    return {'Log_ID':f'fixture-{number}', 'Equipment_ID':'TEST', 'Equipment_Type':'Inspection',
            'Location':'Employee walkway', 'Operator_Shift_Hours':8,
            'Reported_Incident':'Oil spill on the employee walkway was not cleaned.',
            'Is_Violation':actual, 'Violation_Type':'housekeeping', 'Expected_Section':'1910.22'}


HIT = {'section':'1910.22','citation':'29 CFR 1910.22','heading':'Walking surfaces',
       'text':'Synthetic excerpt for contract testing.'}


class Reply:
    def __init__(self, text): self.text=text
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return False
    def raise_for_status(self): pass
    async def json(self): return {'message':{'content':self.text}}


class DecisionPipelineTests(unittest.TestCase):
    def legacy_run(self, rows, reply, hits=None):
        with patch.object(legacy,'get_pillar_vectors',return_value=[]), \
             patch.object(legacy,'retrieve_triangulated',return_value=[HIT] if hits is None else hits), \
             patch.object(legacy,'ask_async',AsyncMock(return_value=reply)), \
             contextlib.redirect_stdout(io.StringIO()):
            return asyncio.run(legacy.audit_async(rows, [HIT], [], {'1910.22'}))

    def test_legacy_balanced_outage_has_zero_correct_decisions(self):
        rows=[event(i, i % 2) for i in range(8)]
        results=self.legacy_run(rows,'')
        metrics=legacy.score(results)
        self.assertEqual(metrics['accuracy'],0.0)
        self.assertEqual(metrics['true_negatives'],0)
        self.assertEqual(metrics['inference_error_count'],8)
        self.assertEqual(metrics['unresolved_positive_cases'],4)
        self.assertEqual(metrics['unresolved_negative_cases'],4)
        self.assertTrue(all(r['predicted'] is None for r in results))

    def test_legacy_unlabeled_event_does_not_acquire_ground_truth(self):
        row=event(); row.pop('Is_Violation')
        result=self.legacy_run([row],'')[0]
        self.assertIsNone(result['actual'])
        self.assertIsNone(legacy.score([result])['accuracy'])

    def test_legacy_missing_authority_or_unsupported_citation_requires_review(self):
        fixtures=[('STATUS: CLEAR\nCITATION: NONE\nREASON: No authority.',[],'CLEAR'),
                  ('STATUS: VIOLATION\nCITATION: 1910.999\nREASON: Unsupported.',[HIT],'VIOLATION')]
        for text,hits,raw_status in fixtures:
            with self.subTest(raw_status=raw_status):
                result=self.legacy_run([event()],text,hits)[0]
                self.assertEqual(result['status'],'REVIEW')
                self.assertIsNone(result['predicted'])
                self.assertEqual(result['unguarded_status'],raw_status)
                self.assertEqual(result['raw_response'],text)

    def test_explicit_review_is_not_critiqued_or_admitted_as_truth(self):
        verifier=reflection.StatutoryGroundedVerifier()
        parsed=parse_audit_response('STATUS: REVIEW\nCITATION: NONE\nREASON: Control not established.')
        self.assertFalse(verifier.verify(parsed,event(),[HIT])['is_valid'])
        session=Mock()
        result,corrected,_=asyncio.run(reflection.SelfReflectionCritic(verifier).critique_and_reground_async(
            session,'test prompt',parsed,event(),[HIT],asyncio.Semaphore(1)))
        session.post.assert_not_called()
        self.assertEqual(result['status'],'REVIEW')
        self.assertFalse(corrected)
        approved=dict(event(), Review_Status='approved', Reviewed_Verdict='REVIEW', Reviewed_Citation='NONE')
        self.assertFalse(assess_for_learning(approved,'REVIEW',None,True).trusted)

    def test_failed_critique_does_not_inherit_or_record_prior_verdict(self):
        verifier=reflection.StatutoryGroundedVerifier()
        parsed=parse_audit_response('STATUS: VIOLATION\nCITATION: 1910.999\nREASON: Unsupported.')
        with tempfile.TemporaryDirectory() as folder, patch.object(reflection,'JSON_DIR',Path(folder)):
            for reply in ('','STATUS: CLEAR','STATUS: REVIEW\nCITATION: NONE\nREASON: Missing evidence.'):
                with self.subTest(reply=reply):
                    session=Mock(); session.post.return_value=Reply(reply)
                    result,corrected,_=asyncio.run(reflection.SelfReflectionCritic(verifier).critique_and_reground_async(
                        session,'test prompt',parsed,event(),[HIT],asyncio.Semaphore(1)))
                    self.assertIn(result['status'],{'UNPARSED','REVIEW'})
                    self.assertIsNone(prediction_from_status(result['status']))
                    self.assertFalse(corrected)
                    self.assertFalse(result['second_pass_valid'])
                    self.assertFalse((Path(folder)/'dpo_pairs_raw.json').exists())

    def test_policy_review_is_effective_and_raw_model_decision_is_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); raw=root/'raw.json'; data=root/'input.csv'; output=root/'new.json'
            row=event()
            with data.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
            raw.write_text(json.dumps({'results':[dict(log_id=row['Log_ID'],status='CLEAR',citation=None,
                reason='Model clearance',predicted=0,actual=0,retrieved=['1910.22'])]}))
            with patch.object(controls,'ROOT',root), patch.object(controls,'sha256',return_value='fixture'):
                report=controls.apply_controls(raw,data,output)
            result=report['results'][0]
            self.assertEqual(result['status'],'REVIEW')
            self.assertEqual(result['model_status'],'CLEAR')
            self.assertIsNone(result['predicted'])
            self.assertEqual(report['controlled_metrics']['accuracy'],0.0)
            self.assertEqual(report['controlled_metrics']['policy_review_cases'],1)

    def test_policy_score_ignores_stale_zero_on_failed_or_unlabeled_case(self):
        rows=[{'status':'UNPARSED','predicted':0,'actual':0},
              {'status':'CLEAR','predicted':0,'actual':None}]
        metrics=controls.score(rows)
        self.assertEqual(metrics['n_labeled'],1)
        self.assertEqual(metrics['n_unlabeled'],1)
        self.assertEqual(metrics['accuracy'],0.0)
        self.assertEqual(metrics['true_negatives'],0)

    def test_saved_report_case_binding_rejects_duplicate_or_stale_targets(self):
        report=json.loads((ROOT/'json/finetune_evaluation_results.json').read_text())
        saved.validate_case_binding(copy.deepcopy(report),saved.DEFAULT_HOLDOUT)
        for mutation in ('duplicate','target','citation_flag','prediction'):
            changed=copy.deepcopy(report)
            rows=changed['strategies']['base']['results']
            if mutation=='duplicate': rows[1]['log_id']=rows[0]['log_id']
            elif mutation=='target': rows[0]['actual']=1-rows[0]['actual']
            elif mutation=='citation_flag': rows[0]['citation_correct']=not rows[0]['citation_correct']
            else: rows[0]['predicted']=0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                saved.validate_case_binding(changed,saved.DEFAULT_HOLDOUT)


if __name__=='__main__': unittest.main()
