"""Check practical help and honest evidence on known development situations."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import showcase
from autonomous_investigator import Observation


class HelpfulInvestigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = showcase.load_corpus(ROOT)
        cls.investigator = showcase.make_investigator(cls.chunks)
        cls.charger = ('A turret machine charger is on a platform, six feet from the ground on an end cap aisle. '
                       'The charger is not bolted down and can be moved easily.')

    def run_text(self, text):
        return showcase.investigate(showcase.text_observation(text), self.investigator, self.chunks)

    def test_charger_gets_practical_help_without_a_model_or_final_violation(self):
        result = self.run_text(self.charger)
        self.assertEqual((result['status'], result['citation']), ('REVIEW', None))
        self.assertIn('fall', result['support']['first_assessment'])
        self.assertTrue(result['support']['immediate_precautions'])
        self.assertLessEqual(len(result['support']['priority_clarifications']), 2)
        self.assertFalse(result['support']['automatic_label'])
        self.assertFalse(result['support']['final_legal_verdict'])
        rendered = showcase.render_observation(result)
        self.assertLess(rendered.index('Assessment:'), rendered.index('Useful follow-up:'))

    def test_mounting_evidence_is_found_and_shown_as_a_separate_research_lead(self):
        result = self.run_text(self.charger)
        self.assertIn('1910.303', {c['section'] for c in result['candidates']})
        lead = result['evidence_leads'][0]
        self.assertEqual(lead['section'], '1910.303')
        self.assertIn('firmly secured', lead['excerpt'])
        self.assertEqual(result['candidates'][lead['combined_candidate_rank'] - 1]['section'], lead['section'])
        self.assertIn('mounting', lead['research_query'])

    def test_search_hypotheses_do_not_become_policy_facts(self):
        seen = []
        investigator = showcase.make_investigator(self.chunks)
        original = investigator.policy_fn
        def observe(row):
            seen.append(dict(row))
            return original(row)
        investigator.policy_fn = observe
        observation = showcase.text_observation(self.charger)
        result = showcase.investigate(observation, investigator, self.chunks)
        self.assertEqual(seen, [observation.to_legacy_row()])
        self.assertGreater(len(result['queries']), 1)
        self.assertEqual(observation.summary, self.charger)

    def test_all_new_development_cases_keep_excerpt_provenance_and_candidate_budget(self):
        data = json.loads((ROOT / 'data/investigation_development_cases.json').read_text())
        for case in data['cases']:
            with self.subTest(case=case['id']):
                observation = Observation.from_mapping(case['observation']) if 'observation' in case else showcase.text_observation(case['text'])
                result = showcase.investigate(observation, self.investigator, self.chunks)
                candidates = result['candidates']
                self.assertLessEqual(len(candidates), 16)
                self.assertLessEqual(len(result['queries']), 8)
                self.assertTrue(result['support']['first_assessment'])
                self.assertTrue(result['support']['immediate_precautions'])
                allowed = {c['section'] for c in candidates}
                for item in candidates + result['evidence_leads']:
                    self.assertIn(item['section'], allowed)
                    text = self.chunks[item['excerpt_chunk_index']]['text']
                    self.assertEqual(item['excerpt'], text[item['excerpt_start']:item['excerpt_end']])
                    self.assertEqual(item['excerpt_chunk_sha256'], hashlib.sha256(text.encode('utf-8')).hexdigest())
                if case.get('forbidden_status'):
                    self.assertNotEqual(result['status'], case['forbidden_status'])

    def test_equipment_height_does_not_become_worker_fall_exposure(self):
        for text in ('A charger is on an elevated platform without mounting bolts.',
                     'A pump stands on a high platform without mounting bolts. No employee is on the platform.'):
            result = self.run_text(text)
            self.assertNotEqual(result['status'], 'VIOLATION')
            self.assertIsNone(result['citation'])

    def test_negated_damage_is_not_repeated_as_confirmed_damage(self):
        result = self.run_text('The sling has no broken wires and remains in service.')
        self.assertNotEqual(result['status'], 'VIOLATION')
        self.assertNotIn('Reported damage to lifting equipment', result['support']['first_assessment'])

    def test_positive_sling_case_keeps_the_grounded_decision(self):
        result = self.run_text('A wire-rope sling with broken outer wires remained in an active lift.')
        self.assertEqual((result['status'], result['citation']), ('VIOLATION', '1910.184'))
        self.assertTrue(result['citation_in_candidates'])

    def test_quiet_scene_does_not_acquire_searches_or_hazard_claims(self):
        quiet = next(x for x in showcase.load_observations(ROOT / 'data/showcase_observations.json') if x.event_id == 'showcase-quiet-scene')
        result = showcase.investigate(quiet, self.investigator, self.chunks)
        self.assertEqual(result['status'], 'NO_ACTION')
        self.assertEqual(result['queries'], [])
        self.assertEqual(result['evidence_leads'], [])
        self.assertEqual(result['support']['possible_concerns'], [])

    def test_context_explains_a_short_summary(self):
        observation = Observation.from_mapping({'summary': 'No incident reported', 'equipment': ['forklift'],
            'measurements': {'operator_shift_hours': 13}, 'human_flag': True})
        result = showcase.investigate(observation, self.investigator, self.chunks)
        self.assertEqual(result['observation_context']['measurements']['operator_shift_hours'], 13)
        self.assertIn('fatigue', result['support']['first_assessment'])
        rendered = showcase.render_observation(result)
        self.assertIn('forklift', rendered)
        self.assertIn('13', rendered)

    def test_reference_benchmark_identifies_its_separate_query_planner(self):
        benchmark = showcase.run_benchmark(16, self.investigator.retriever, ROOT)
        self.assertEqual(benchmark['query_planner'], 'published_reference_query_planner')
        for cutoff in (1, 5, 16):
            expected = sum(r['rank'] is not None and r['rank'] <= cutoff for r in benchmark['records']) / 16
            self.assertEqual(benchmark[f'recall_at_{cutoff}'], expected)


if __name__ == '__main__':
    unittest.main()
