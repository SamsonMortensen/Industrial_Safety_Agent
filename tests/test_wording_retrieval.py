"""Check wording retrieval with synthetic evidence, not reserved labels."""

import copy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from autonomous_investigator import RetrievalCandidate
from wording_retrieval import WordingRetriever, concept_terms, normalized_terms, word_form, wording_excerpt


def chunk(section, text, heading="Example evidence"):
    return {"section": section, "citation": section, "heading": heading, "topic": "", "text": text}


class FakeSemanticIndex:
    def __init__(self, values):
        self.values = values
        self.calls = []
        self.validated = False

    def validate_chunks(self, chunks):
        self.validated = True

    def query_scores(self, query):
        self.calls.append(query)
        return self.values


class WordingRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            chunk("surface", "Passageways and walking-working surfaces must be free of leaks and spills."),
            chunk("sling", "Broken wires and torn lifting slings must be removed from service."),
            chunk("electrical", "Electrical equipment must be firmly secured to its mounting surface."),
            chunk("other", "Employees may use pneumatic devices for assembly."),
        ]
        self.retriever = WordingRetriever(self.documents)

    def test_plural_and_inflected_forms_are_symmetric(self):
        for variants in (("spill", "spills", "spilled", "spilling"), ("walk", "walking", "walked"), ("battery", "batteries"), ("mount", "mounted", "mounting")):
            self.assertEqual(len({word_form(word) for word in variants}), 1)

    def test_unicode_punctuation_and_case_share_words(self):
        self.assertEqual(normalized_terms("WALKING–WORKING SURFACES"), normalized_terms("walking-working surfaces"))

    def test_polite_question_keeps_the_same_core_search(self):
        self.assertEqual(normalized_terms("Please help: which citation is most aligned? Oil spills."), normalized_terms("Oil spills."))

    def test_spill_morphology_retrieves_original_surface_text(self):
        for query in ("An oil spill remains on the employee walkway.", "Oil spills remain on employee walkways.", "Is an oil spill on the employee walkway a safety issue?"):
            result = self.retriever.search([query])[0]
            self.assertEqual(result.section, "surface")
            self.assertEqual(result.text, self.documents[0]["text"])
            self.assertIsInstance(result, RetrievalCandidate)

    def test_surface_and_liquid_words_bridge_the_search_only(self):
        query = "There is oil on the walkway where staff pass."
        self.assertEqual(self.retriever.search([query])[0].section, "surface")
        self.assertNotIn("liquid_release", concept_terms("The crude oil price rose today."))

    def test_negation_is_preserved_and_not_a_violation_decision(self):
        query = "No oil spill is present on the employee walkway."
        details = self.retriever.explain_query(query)
        self.assertIn("no", details["normalized_terms"])
        self.assertFalse(details["observation_modified"])
        match = self.retriever.match_diagnostics(query, self.retriever.search([query])[0])
        self.assertFalse(match["evidence_is_applicability_verdict"])

    def test_concept_equivalence_has_no_citation_input(self):
        result = self.retriever.search(["A hoisting strap is tearing under its load."])[0]
        self.assertEqual(result.section, "sling")
        renamed = [dict(document, section="renamed-" + str(i), citation="evidence-" + str(i)) for i, document in enumerate(self.documents)]
        self.assertEqual(WordingRetriever(renamed).search(["A hoisting strap is tearing under its load."])[0].text, result.text)

    def test_section_budget_is_applied_before_duplicate_chunks(self):
        repeated = [chunk("repeated", "spill") for _ in range(20)]
        repeated.append(chunk("distinct", "A spill was identified near a work surface."))
        results = WordingRetriever(repeated, pool_size=2).search(["spill"], top_k=2)
        self.assertEqual({item.section for item in results}, {"repeated", "distinct"})

    def test_no_query_overlap_returns_no_candidate(self):
        self.assertEqual(self.retriever.search(["How can I bake sourdough bread?"]), [])

    def test_input_evidence_is_not_mutated(self):
        before = copy.deepcopy(self.documents)
        self.retriever.search(["spill on a walkway"])
        self.assertEqual(self.documents, before)

    def test_unambiguous_typo_is_a_separate_hypothesis(self):
        retriever = WordingRetriever([chunk("example", "walkway")])
        details = retriever.explain_query("walkwy")
        self.assertEqual(details["normalized_terms"], ["walkwy"])
        self.assertEqual(details["spelling_hypotheses"], {"walkwy": "walkway"})
        self.assertEqual(retriever.search(["walkwy"])[0].section, "example")

    def test_ambiguous_typo_is_not_changed(self):
        retriever = WordingRetriever([chunk("a", "cord"), chunk("b", "card")])
        self.assertEqual(retriever.explain_query("curd")["spelling_hypotheses"], {})

    def test_known_search_words_are_not_mistaken_for_typos(self):
        retriever = WordingRetriever([chunk("a", "stiff slack")])
        self.assertEqual(retriever.explain_query("staff on a slick walkway")["spelling_hypotheses"], {})

    def test_fuzzy_work_is_limited(self):
        query = " ".join("unknown" + chr(97 + index) * 4 for index in range(25))
        details = self.retriever.explain_query(query)
        self.assertTrue(details["spelling_checks_limited"])
        self.assertLessEqual(len(details["spelling_hypotheses"]), 16)

    def test_default_has_no_embedder(self):
        self.assertIsNone(self.retriever.embedder)
        self.assertIsNone(self.retriever.semantic_index)

    def test_semantic_adapter_validates_corpus_and_scores(self):
        semantic = FakeSemanticIndex([.9, .1, .1, .1])
        retriever = WordingRetriever(self.documents, semantic_index=semantic)
        self.assertTrue(semantic.validated)
        self.assertEqual(retriever.search(["unfamiliar wording"])[0].section, "surface")
        self.assertEqual(semantic.calls, ["unfamiliar wording"])

    def test_bad_semantic_scores_are_rejected(self):
        for values in ([1.0], [float("nan")] * 4, [[1.0]] * 4):
            retriever = WordingRetriever(self.documents, semantic_index=FakeSemanticIndex(values))
            with self.assertRaisesRegex(ValueError, "Semantic scores"):
                retriever.search(["spill"])

    def test_semantic_failure_is_not_silently_reported_as_success(self):
        class FailedIndex(FakeSemanticIndex):
            def query_scores(self, query):
                raise ValueError("encoder unavailable")
        with self.assertRaisesRegex(ValueError, "encoder unavailable"):
            WordingRetriever(self.documents, semantic_index=FailedIndex([])).search(["spill"])

    def test_existing_vector_api_and_blank_query_alignment(self):
        vectors = np.eye(4)
        retriever = WordingRetriever(self.documents, corpus_vectors=vectors)
        results = retriever.search(["", "unknown"], query_vectors=np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]]))
        self.assertEqual(results[0].section, "sling")
        self.assertEqual(results[0].matched_queries, [1])

    def test_diagnostics_do_not_embed_again_and_survive_later_queries(self):
        semantic = FakeSemanticIndex([.9, .1, .1, .1])
        retriever = WordingRetriever(self.documents, semantic_index=semantic)
        candidate = retriever.search(["spill on a walkway"])[0]
        retriever.search(["sling"])
        calls = list(semantic.calls)
        details = retriever.match_diagnostics("spill on a walkway", candidate)
        self.assertEqual(semantic.calls, calls)
        self.assertTrue(details["semantic_search_used"])
        self.assertTrue(details["diagnostics_from_cached_search"])
        self.assertFalse(details["diagnostics_from_latest_search"])
        self.assertEqual(details["semantic_score"], .9)

    def test_uncached_diagnostics_have_no_semantic_claim(self):
        candidate = self.retriever.search(["spill"])[0]
        details = self.retriever.match_diagnostics("spills on walking surfaces", candidate)
        self.assertIsNotNone(details["normalized_authority_rank"])
        self.assertFalse(details["semantic_search_used"])
        self.assertIsNone(details["semantic_score"])
        self.assertFalse(details["ranking_is_confidence"])

    def test_diagnostics_accept_serialized_candidate(self):
        candidate = self.retriever.search(["spill"])[0]
        self.assertEqual(self.retriever.match_diagnostics("spill", candidate), self.retriever.match_diagnostics("spill", asdict(candidate)))

    def test_serialized_diagnostics_recover_verified_original_chunk(self):
        candidate = self.retriever.search(["spill"])[0].to_dict()
        self.assertNotIn("text", candidate)
        details = self.retriever.match_diagnostics("spill", candidate)
        self.assertEqual(details["diagnostic_text_scope"], "verified_corpus_chunk")
        self.assertIn("spill", details["meaningful_shared_terms"])

    def test_serialized_wrong_section_cannot_borrow_chunk_text(self):
        candidate = {"chunk_index": 0, "section": "wrong", "excerpt": "Other text"}
        details = self.retriever.match_diagnostics("spill", candidate)
        self.assertEqual(details["diagnostic_text_scope"], "provided_excerpt")
        self.assertEqual(details["meaningful_shared_terms"], [])

    def test_discourse_words_do_not_count_as_shared_evidence(self):
        self.assertEqual(normalized_terms("Which provision addresses this otherwise closest requirement?"), [])

    def test_excerpt_finds_plural_condition_and_preserves_source_offsets(self):
        source = "Employees use designated work areas. " * 30 + self.documents[0]["text"] + " Employees maintain normal records. " * 30
        excerpt = wording_excerpt(source, "An oil spill remains on the employee walkway.")
        self.assertIn("leaks and spills", excerpt["text"])
        self.assertEqual(excerpt["text"], source[excerpt["start"]:excerpt["end"]])
        self.assertLessEqual(len(excerpt["text"]), 650)

    def test_excerpt_finds_mounting_instead_of_generic_electrical_text(self):
        source = "Electrical installations include many components and grounded parts. " * 25 + self.documents[2]["text"] + " Electrical circuits provide power. " * 25
        excerpt = wording_excerpt(source, "A turret machine charger is on a platform six feet from the ground. It is not bolted down and can be moved easily.")
        self.assertIn("firmly secured", excerpt["text"])
        self.assertEqual(excerpt["text"], source[excerpt["start"]:excerpt["end"]])

    def test_excerpt_retains_negation_and_unicode_verbatim(self):
        source = "Before work: employees must not enter. Do not use frayed slings—even briefly. Other work continues."
        excerpt = wording_excerpt(source, "Can a frayed lifting sling be used?", max_chars=100)
        self.assertIn("Do not use frayed slings", excerpt["text"])
        self.assertEqual(excerpt["text"], source[excerpt["start"]:excerpt["end"]])

    def test_excerpt_rejects_invalid_lengths(self):
        for size in (False, 0, 39, 2001, 1.5):
            with self.assertRaises(ValueError):
                wording_excerpt("source", "query", max_chars=size)

    def test_diagnostics_exclude_generic_actor_overlap(self):
        candidate = self.retriever.search(["Employees use devices"])[0]
        details = self.retriever.match_diagnostics("Employees use devices", candidate)
        self.assertNotIn("employe", details["meaningful_shared_terms"])
        self.assertNotIn("use", details["meaningful_shared_terms"])

    def test_query_diagnostic_cache_is_bounded(self):
        for index in range(130):
            self.retriever.search(["spill " + str(index)])
        self.assertLessEqual(len(self.retriever._query_diagnostics), 128)

    def test_bad_budgets_and_inputs_are_rejected(self):
        for parameter in (0, -1, True, 1.5, None):
            with self.assertRaises(ValueError):
                WordingRetriever(self.documents, pool_size=parameter)
            with self.assertRaises(ValueError):
                WordingRetriever(self.documents, rank_constant=parameter)
        for queries in ("spill", [1], ["spill"] * 33, ["x" * 4001]):
            with self.assertRaises(ValueError):
                self.retriever.search(queries)
        for weights in ([], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                self.retriever.search(["spill"], query_weights=weights)

    def test_oversized_query_is_rejected_before_any_embedding(self):
        calls = []
        retriever = WordingRetriever(self.documents, corpus_vectors=np.eye(4), embedder=lambda queries: calls.append(queries))
        with self.assertRaises(ValueError):
            retriever.search(["x" * 4001])
        self.assertEqual(calls, [])
        semantic = FakeSemanticIndex([1., 0., 0., 0.])
        with self.assertRaises(ValueError):
            WordingRetriever(self.documents, semantic_index=semantic).search(["x" * 4001])
        self.assertEqual(semantic.calls, [])

    def test_zero_budget_and_empty_corpus_are_valid(self):
        self.assertEqual(self.retriever.search(["spill"], top_k=0), [])
        self.assertEqual(self.retriever.search([]), [])
        self.assertEqual(WordingRetriever([]).search(["spill"]), [])

    def test_ties_are_repeatable(self):
        results = self.retriever.search(["spill and sling"])
        self.assertEqual([asdict(result) for result in results], [asdict(result) for result in self.retriever.search(["spill and sling"])])


if __name__ == "__main__":
    unittest.main()
