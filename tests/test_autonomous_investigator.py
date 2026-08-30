import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from autonomous_investigator import (
    ApplicabilityReranker,
    AutonomousInvestigator,
    HybridRetriever,
    Observation,
    QueryPlanner,
    Relation,
    TriggerEngine,
)
from hybrid_policy import evaluate_event

CHUNKS = [
    {
        "citation": "29 CFR 1910.333",
        "section": "1910.333",
        "heading": "Selection and use of work practices",
        "topic": "Electrical",
        "text": (
            "Unqualified persons and conductive objects shall maintain the required "
            "distance from exposed energized overhead lines."
        ),
    },
    {
        "citation": "29 CFR 1910.184",
        "section": "1910.184",
        "heading": "Slings",
        "topic": "Materials Handling",
        "text": "Damaged wire rope slings shall be immediately removed from service.",
    },
    {
        "citation": "49 CFR 228.405",
        "section": "228.405",
        "heading": "Passenger train employee duty limits",
        "topic": "Hours of Service",
        "text": "A commuter or intercity passenger train employee may not exceed 12 hours.",
    },
]


def electrical_observation(**overrides):
    values = {
        "event_id": "camera-17-0001",
        "timestamp": "2026-08-30T10:00:00-07:00",
        "source_id": "camera-17",
        "summary": "A mobile crane boom moved within four feet of an energized overhead conductor",
        "actors": ["crane operator"],
        "equipment": ["mobile crane"],
        "location": "energized overhead line zone",
        "actions": ["operating boom"],
        "conditions": ["energized conductor exposed"],
        "relations": [
            {
                "subject": "crane boom",
                "predicate": "within",
                "object": "energized overhead conductor",
                "value": 4.0,
                "unit": "feet",
                "confidence": 0.94,
                "risk_score": 0.92,
            }
        ],
        "perception_confidence": 0.94,
        "changed": True,
        "evidence_refs": ["frames/camera-17/0001.jpg"],
    }
    values.update(overrides)
    return Observation.from_mapping(values)


class AutonomousInvestigatorTests(unittest.TestCase):
    def test_observation_accepts_camera_neutral_mapping(self):
        observation = electrical_observation()
        self.assertEqual(observation.source_id, "camera-17")
        self.assertIsInstance(observation.relations[0], Relation)
        self.assertIn("4 feet", observation.evidence_text())

    def test_trigger_responds_to_relationship_without_hazard_name_dictionary(self):
        trigger = TriggerEngine(sample_rate=0.0, trigger_on_change=False).evaluate(
            electrical_observation(changed=False)
        )
        self.assertTrue(trigger.triggered)
        self.assertIn("relationship_risk", trigger.reasons)

    def test_quiet_observation_can_remain_unqueried(self):
        observation = Observation.from_mapping(
            {
                "event_id": "quiet-1",
                "timestamp": "2026-08-30T10:00:01-07:00",
                "source_id": "camera-2",
                "summary": "No scene change",
                "perception_confidence": 0.99,
            }
        )
        trigger = TriggerEngine(sample_rate=0.0).evaluate(observation)
        self.assertFalse(trigger.triggered)

    def test_query_planner_decomposes_relations_measurements_and_controls(self):
        observation = electrical_observation(
            controls=["temporary barrier"],
            measurements={"estimated_distance_ft": 4.0},
        )
        queries = QueryPlanner(max_queries=8).plan(observation)
        joined = "\n".join(queries).lower()
        self.assertGreaterEqual(len(queries), 4)
        self.assertIn("crane boom within energized overhead conductor", joined)
        self.assertIn("estimated distance ft 4.0", joined)
        self.assertIn("temporary barrier", joined)

    def test_hybrid_retrieval_finds_governing_electrical_rule(self):
        observation = electrical_observation()
        queries = QueryPlanner().plan(observation)
        candidates = HybridRetriever(CHUNKS).search(queries, top_k=3)
        self.assertEqual(candidates[0].section, "1910.333")
        self.assertGreater(len(candidates[0].matched_queries), 1)

    def test_semantic_and_lexical_ranks_are_fused(self):
        corpus_vectors = np.eye(3, dtype=np.float32)

        def embedder(queries):
            vectors = np.zeros((len(queries), 3), dtype=np.float32)
            vectors[:, 0] = 1.0
            return vectors

        retriever = HybridRetriever(CHUNKS, corpus_vectors, embedder)
        candidates = retriever.search(["equipment close to live power"], top_k=3)
        self.assertEqual(candidates[0].section, "1910.333")
        self.assertGreater(candidates[0].semantic_score, 0.0)

    def test_applicability_reranker_exposes_feature_scores(self):
        observation = electrical_observation()
        queries = QueryPlanner().plan(observation)
        candidates = HybridRetriever(CHUNKS).search(queries, top_k=3)
        reranked = ApplicabilityReranker().rerank(observation, candidates)
        self.assertEqual(reranked[0].section, "1910.333")
        self.assertIn("event_condition", reranked[0].applicability_features)

    def test_investigator_returns_grounded_policy_decision_with_evidence(self):
        investigator = AutonomousInvestigator(
            HybridRetriever(CHUNKS),
            trigger_engine=TriggerEngine(sample_rate=0.0),
            policy_fn=evaluate_event,
        )
        result = investigator.investigate(electrical_observation())
        self.assertEqual((result.status, result.citation), ("VIOLATION", "1910.333"))
        self.assertEqual(result.decision_source, "deterministic_policy")
        self.assertTrue(result.queries)
        self.assertEqual(result.evidence_refs, ["frames/camera-17/0001.jpg"])

    def test_unknown_changed_condition_is_investigated_then_sent_to_review(self):
        observation = Observation.from_mapping(
            {
                "event_id": "novel-1",
                "timestamp": "2026-08-30T10:01:00-07:00",
                "source_id": "camera-9",
                "summary": "An unfamiliar coupling oscillation began during transfer",
                "equipment": ["transfer machine"],
                "actions": ["oscillating"],
                "novelty_score": 0.9,
                "changed": True,
            }
        )
        investigator = AutonomousInvestigator(
            HybridRetriever(CHUNKS),
            trigger_engine=TriggerEngine(sample_rate=0.0),
            policy_fn=evaluate_event,
        )
        result = investigator.investigate(observation)
        self.assertEqual(result.status, "REVIEW")
        self.assertEqual(result.decision_source, "review_gate")
        self.assertTrue(result.queries)
        self.assertEqual(result.candidates, [])


if __name__ == "__main__":
    unittest.main()
