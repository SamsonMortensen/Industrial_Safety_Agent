"""Keep research questions separate from reported incident findings."""
from __future__ import annotations

from dataclasses import replace
import re

from autonomous_investigator import AutonomousInvestigator


def _opening(text: str) -> str:
    opening = text.strip().lower()
    opening = re.sub(r"^(?:can|could|would) you (?:please )?help(?: me)?\s*[?!]\s*", "", opening)
    return re.sub(r"^(?:please\s+|hey[,!]?\s+)", "", opening)


def information_request(text: str) -> bool:
    """Recognize explicit information requests; this is not a general intent model.

    A statement followed by a question still reports a condition. Starting with
    a question or a hypothetical does not establish that its condition occurred.
    The original text continues into retrieval unchanged.
    """
    opening = _opening(text)
    return bool(re.match(
        r"(?:what|which|why|how|where|when|is|are|was|were|does|do|did|can|could|"
        r"should|would|will|may|might|explain|describe|summarize|find|list|compare|"
        r"hypothetically|suppose|imagine|assuming|if|consider\s+a\s+hypothetical)\b",
        opening,
    ))


def citation_alignment(query, candidates, retriever):
    """Expose the ranking and its uncertainty; never certify its top result.

    Direct wording overlap is a display filter, not a calibrated acceptance
    threshold. Semantic-only matches remain in the full candidate list.
    """
    matches = []
    for rank, candidate in enumerate(candidates[:3], 1):
        diagnostics = retriever.match_diagnostics(query, candidate)
        shared = diagnostics["meaningful_shared_terms"]
        concepts = diagnostics["matched_search_concepts"]
        if len(shared) >= 2 or (shared and concepts):
            matches.append(dict(candidate, candidate_rank=rank, match_basis=diagnostics))
    return {
        "status": "ranked_evidence_needs_review" if matches else "no_supported_match_identified",
        "highest_ranked_research_match": matches[0] if matches else None,
        "other_research_matches": matches[1:],
        "accepted_as_correct": False,
        "governing_citation_established": False,
        "corpus_coverage_verified": False,
        "scope": "These are ranked research leads from the pinned corpus, not verified legal applicability. The needed authority may be missing. More than one rule may apply.",
        "display_filter": "At least two shared normalized terms, or one shared term plus a search concept. This filter is not a calibrated confidence score.",
    }


class InteractiveInvestigator(AutonomousInvestigator):
    """Answer information requests with evidence, without an incident verdict."""

    def investigate(self, observation):
        explicit_hypothetical = bool(re.match(r"(?:hypothetically|suppose|imagine|assuming|if|consider\s+a\s+hypothetical)\b", _opening(observation.summary)))
        has_reported_context = bool(observation.actions or observation.conditions or observation.relations)
        if not information_request(observation.summary) or (has_reported_context and not explicit_hypothetical):
            return super().investigate(observation)
        # Use the same retrieval path without calling either decision provider.
        research = AutonomousInvestigator(
            self.retriever, trigger_engine=self.trigger_engine,
            query_planner=self.query_planner, reranker=self.reranker,
            policy_fn=None, reasoner=None, candidate_limit=self.candidate_limit,
        ).investigate(observation)
        if research.status == "NO_ACTION":
            return research
        return replace(
            research, status="REVIEW", citation=None, confidence=None,
            decision_source="information_request",
            reason="This is a question or hypothetical. Retrieved evidence can help answer it, but it does not establish an actual violation or a clear condition.",
        )
