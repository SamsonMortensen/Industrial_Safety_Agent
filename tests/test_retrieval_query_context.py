"""Observable context should aid retrieval without manufacturing legal terms."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from autonomous_investigator import HybridRetriever, Observation, QueryPlanner
from showcase import benchmark_observation, benchmark_sample


class QueryContextTests(unittest.TestCase):
    def test_background_and_measurements_retain_incident(self):
        incident = "Hydraulic liquid reached the access path"
        observation = Observation.from_mapping({
            "summary": incident, "equipment": ["cargo vehicle"], "location": "Bay 9",
            "controls": ["temporary barrier"], "measurements": {"shift_hours": 6.5},
        })
        queries = QueryPlanner().plan(observation)
        self.assertGreaterEqual(len(queries), 4)
        for query in queries:
            self.assertIn(incident, query)
        self.assertTrue(any("shift hours 6.5" in q for q in queries))
        self.assertTrue(any("temporary barrier" in q for q in queries))

    def test_generated_queries_do_not_manufacture_regulatory_boilerplate(self):
        observation = Observation.from_mapping({
            "summary": "Loose packing material covers a walkway", "equipment": ["cart"],
            "measurements": {"load_mass": 12}, "novelty_score": .9,
            "relations": [{"subject": "material", "predicate": "covers", "object": "walkway"}],
        })
        queries = QueryPlanner().plan(observation)
        self.assertIn("material covers walkway", queries)
        for phrase in ("federal safety requirement", "applicable limit", "minimum requirement", "analogous regulated condition"):
            self.assertNotIn(phrase, " ".join(queries))

    def test_observed_words_are_preserved_even_when_they_resemble_boilerplate(self):
        text = "A posted federal safety requirement sign fell into the aisle"
        self.assertEqual(QueryPlanner().plan(Observation.from_mapping({"summary": text})), [text])

    def test_query_cap_empty_input_and_whitespace_deduplication(self):
        self.assertEqual(QueryPlanner().plan(Observation.from_mapping({})), [])
        observation = Observation.from_mapping({"summary": "Spill  on floor", "equipment": ["cart"],
                                               "measurements": {f"reading_{i}": i for i in range(20)}})
        queries = QueryPlanner(max_queries=3).plan(observation)
        self.assertEqual(len(queries), 3)
        self.assertTrue(all("Spill on floor" in q for q in queries))
        self.assertEqual(len(queries), len(set(queries)))

    def test_labels_and_identifiers_do_not_change_queries(self):
        row = benchmark_sample(1, ROOT)[0]
        changed = {**row, "Expected_Section": "SENTINEL", "Is_Violation": "0",
                   "Violation_Type": "SENTINEL", "Log_ID": "SENTINEL", "Equipment_ID": "SENTINEL"}
        planner = QueryPlanner()
        before = planner.plan(benchmark_observation(row))
        after = planner.plan(benchmark_observation(changed))
        self.assertEqual(before, after)
        self.assertNotIn("SENTINEL", " ".join(after))

    def test_observation_metadata_cannot_insert_an_expected_citation(self):
        obs = Observation.from_mapping({"summary": "A box blocks the walkway",
                                        "metadata": {"Expected_Section": "SENTINEL"}})
        self.assertNotIn("SENTINEL", " ".join(QueryPlanner().plan(obs)))

    def test_context_queries_do_not_retrieve_boilerplate_only_documents(self):
        chunks = [{"section": "fixture-a", "text": "Safety requirement applicable minimum federal limit"},
                  {"section": "fixture-b", "text": "Copper cable blocked the doorway"}]
        obs = Observation.from_mapping({"summary": "Copper cable blocked the doorway",
                                        "equipment": ["cart"], "measurements": {"load_kg": 4}})
        found = HybridRetriever(chunks).search(QueryPlanner().plan(obs), top_k=16)
        self.assertEqual([c.section for c in found], ["fixture-b"])


if __name__ == "__main__":
    unittest.main()
