"""Find candidate evidence across everyday wording without changing the facts.

The raw lexical channel stays in the comparison. Additional channels share
word-form normalization between queries and documents, recover unambiguous
one-edit misspellings, and bridge a small, visible vocabulary gap. Vocabulary
groups contain no citations and do not decide whether a condition is present.
The historical reference retriever remains unchanged.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np

from autonomous_investigator import HybridRetriever, RetrievalCandidate, tokenize

PROFILE = "wording_aware_authority_retrieval_v1"
_WRAPPERS = frozenset("please help tell explain give show find identify correct most aligned relevant applicable citation citations regulation regulations rule rules requirement requirements provision provisions addresses otherwise closest situation issue question answer safety unsafe what which why how where when does do did could would should can may might will need want our my me us their there here i we you your they them he she it its remain remains present".split())
_GENERIC_ACTORS = frozenset({"employee", "worker", "person", "people", "staff", "employer", "operator"})
_LOW_INFORMATION = frozenset("no not must shall would should may might will are was were be been being has have had using used use keep kept make made get got one two first next good bad new old today yesterday tomorrow through across under over from into with without".split())

# These groups describe search language, not legal scope or observed hazards.
# They deliberately contain both conditions and control vocabulary: a denied
# spill can still need the housekeeping provision without being a violation.
SEARCH_CONCEPTS = {
    "walking_surface": r"\b(?:walking[- ]working\s+surfaces?|walking\s+(?:surfaces?|routes?|paths?)|walkways?|walk\s+ways?|passageways?|footpaths?|pedestrian\s+(?:paths?|routes?|aisles?)|(?:path|route)\s+(?:people|workers?|employees?|staff)\s+use)\b",
    "liquid_release": r"\b(?:spills?|spilled|spilling|spillage|leaks?|leaked|leaking|puddles?|pooled|oily|greasy|slick|slippery|wet)\b",
    "lifting_attachment": r"\b(?:slings?|(?:lifting|hoisting|rigging)\s+(?:straps?|webbing))\b",
    "material_damage": r"\b(?:broken|breaks?|frayed|fraying|torn|tearing|tears?|cracked|damaged|damage|defects?|defective)\b",
    "equipment_mounting": r"\b(?:mounting|mounted|fastening|fastened|anchored|anchoring|bolts?|bolted|unbolted|retaining\s+brackets?|firmly\s+secured|securely\s+(?:mounted|fastened))\b",
    "electrical_equipment": r"\b(?:electric(?:al)?|chargers?|charging|batter(?:y|ies))\b",
    "exposed_electrical_energy": r"\b(?:energized|energised|live\s+(?:parts?|conductors?|wires?|lines?)|exposed\s+(?:conductors?|wires?|electrical\s+parts?))\b",
    "edge_protection": r"\b(?:guardrails?|fall[- ]arrest|fall[- ]restraint|edge\s+protection|unprotected\s+edges?|open[- ]sided\s+platforms?)\b",
}
_CONCEPT_PATTERNS = {key: re.compile(pattern, re.I) for key, pattern in SEARCH_CONCEPTS.items()}


def _text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().translate(str.maketrans({
        "\u2019": "'", "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-",
    }))


def word_form(word: str) -> str:
    """A symmetric search stem, not a language interpretation or corrected fact."""
    if word in {"people", "persons"}:
        word = "person"
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 5 and word.endswith("sses"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    if len(word) > 3 and word[-1:] == word[-2:-1] and word[-1:] in "bdgmnprt":
        word = word[:-1]
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


_GENERIC_STEMS = frozenset(word_form(word) for word in _GENERIC_ACTORS)
_LOW_INFORMATION_STEMS = frozenset(word_form(word) for word in _LOW_INFORMATION)


def normalized_terms(text: str) -> list[str]:
    return [word_form(word) for word in tokenize(_text(text)) if word not in _WRAPPERS]


def concept_terms(text: str) -> list[str]:
    normalized = _text(text)
    found = [key for key, pattern in _CONCEPT_PATTERNS.items() if pattern.search(normalized)]
    # A surface and a liquid mentioned together are useful search context.
    # This also applies to a contained or denied spill; it asserts no hazard.
    if "walking_surface" in found and "liquid_release" not in found and re.search(r"\b(?:oil|water|liquid|fluid)\b", normalized):
        found.append("liquid_release")
    return found


def wording_excerpt(text: str, query: str, max_chars: int = 650) -> dict[str, Any]:
    """Select an original-text window using the same search vocabulary.

    Offsets are Python character indices in the untouched source. The score
    ranks passage relevance only; it does not establish a legal conclusion.
    """
    if not isinstance(text, str) or not isinstance(query, str):
        raise TypeError("Excerpt text and query must be strings")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 40 <= max_chars <= 2000:
        raise ValueError("Excerpt length must be between 40 and 2,000 characters")
    terms = set(normalized_terms(query)) - _GENERIC_STEMS - _LOW_INFORMATION_STEMS
    concepts = set(concept_terms(query))
    hits = [match.start() for match in re.finditer(r"[^\W_]+", text, re.UNICODE)
            if word_form(_text(match.group())) in terms]
    for concept in concepts:
        hits.extend(match.start() for match in _CONCEPT_PATTERNS[concept].finditer(text))
    starts = {0}
    for hit in hits:
        proposed = max(0, hit - 100)
        boundary = max(text.rfind(". ", max(0, proposed - 120), hit),
                       text.rfind("\n", max(0, proposed - 120), hit))
        if boundary >= 0:
            proposed = boundary + (2 if text[boundary:boundary + 2] == ". " else 1)
        starts.add(proposed)
    best = (0.0, 0, min(len(text), max_chars))
    for start in sorted(starts):
        end = min(len(text), start + max_chars)
        passage = text[start:end]
        shared_terms = terms.intersection(normalized_terms(passage))
        shared_concepts = concepts.intersection(concept_terms(passage))
        score = float(len(shared_terms) + 4 * len(shared_concepts))
        if score > best[0]:
            best = (score, start, end)
    score, start, end = best
    return {"text": text[start:end], "start": start, "end": end, "score": score}


def _one_edit(left: str, right: str) -> bool:
    """Insertion, deletion, replacement or adjacent transposition, exactly one."""
    if left == right or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        positions = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
        return len(positions) == 1 or (len(positions) == 2 and positions[1] == positions[0] + 1
            and left[positions[0]] == right[positions[1]] and left[positions[1]] == right[positions[0]])
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    index = next((i for i, (a, b) in enumerate(zip(shorter, longer)) if a != b), len(shorter))
    return shorter[index:] == longer[index + 1:]


class _TermIndex:
    def __init__(self, documents: Sequence[Sequence[str]]):
        self.counts = [Counter(words) for words in documents]
        self.lengths = np.array([sum(counts.values()) for counts in self.counts], dtype=float)
        self.average = float(self.lengths.mean()) if len(self.lengths) else 1.0
        self.average = max(self.average, 1.0)
        frequency = Counter(term for counts in self.counts for term in counts)
        total = max(len(documents), 1)
        self.idf = {term: math.log(1.0 + (total - count + .5) / (count + .5)) for term, count in frequency.items()}

    def scores(self, terms: Sequence[str], *, generic_weight: float = 1.0, coverage: bool = False) -> np.ndarray:
        output = np.zeros(len(self.counts), dtype=float)
        selected = set(terms)
        for term in selected:
            if term not in self.idf:
                continue
            weight = generic_weight if term in _GENERIC_STEMS else 1.0
            for index, counts in enumerate(self.counts):
                frequency = counts.get(term, 0)
                if frequency:
                    denominator = frequency + 1.5 * (.25 + .75 * self.lengths[index] / self.average)
                    output[index] += weight * self.idf[term] * frequency * 2.5 / denominator
        if coverage and selected:
            for index, counts in enumerate(self.counts):
                output[index] *= (len(selected.intersection(counts)) / len(selected)) ** 2
        return output


class WordingRetriever(HybridRetriever):
    """Fuse raw, normalized and vocabulary evidence at the authority level.

    A positive retrieval score is only a ranking signal. Every returned section
    still needs an applicability check; arbitrary input need not have a valid
    citation in this corpus. No model is loaded or contacted by default.
    """
    profile = PROFILE

    def __init__(self, chunks: Sequence[Mapping[str, Any]], corpus_vectors=None, embedder=None,
                 rank_constant: int = 60, pool_size: int = 50, *, semantic_index=None):
        if isinstance(rank_constant, bool) or not isinstance(rank_constant, int) or rank_constant <= 0:
            raise ValueError("rank_constant must be a positive integer")
        if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size <= 0:
            raise ValueError("pool_size must be a positive integer")
        super().__init__(chunks, corpus_vectors=corpus_vectors, embedder=embedder,
                         rank_constant=rank_constant, pool_size=pool_size)
        self.authority_pool_size = pool_size
        self.semantic_index = semantic_index
        if semantic_index is not None:
            semantic_index.validate_chunks(self.chunks)
        documents = [f"{c.get('heading', '')} {c.get('topic', '')} {c.get('text', '')}" for c in self.chunks]
        self.normalized = _TermIndex([normalized_terms(document) for document in documents])
        self.concepts = _TermIndex([concept_terms(document) for document in documents])
        self._vocabulary = defaultdict(list)
        for term in sorted(self.normalized.idf):
            if term.isalpha() and 3 <= len(term) <= 25:
                self._vocabulary[len(term)].append(term)
        self._typo_cache: dict[str, str | None] = {}
        self._query_diagnostics: dict[str, dict[str, Any]] = {}
        self._search_generation = 0

    def _correct(self, term: str) -> str | None:
        if term in self._typo_cache:
            return self._typo_cache[term]
        matches = []
        for size in (len(term) - 1, len(term), len(term) + 1):
            for candidate in self._vocabulary[size]:
                if _one_edit(term, candidate):
                    matches.append(candidate)
                    if len(matches) > 1:
                        break
            if len(matches) > 1:
                break
        result = matches[0] if len(matches) == 1 else None
        if len(self._typo_cache) < 2048:
            self._typo_cache[term] = result
        return result

    def explain_query(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or len(query) > 4000:
            raise ValueError("Queries must be text with at most 4000 characters")
        terms = normalized_terms(query)
        # Corpus absence alone does not make ordinary vocabulary misspelled.
        # Keep known concept words and generic actors out of the fuzzy channel.
        protected = set(_GENERIC_STEMS)
        for pattern in _CONCEPT_PATTERNS.values():
            for match in pattern.finditer(_text(query)):
                protected.update(normalized_terms(match.group()))
        unknown = list(dict.fromkeys(t for t in terms if t not in self.normalized.idf and t not in protected and t.isalpha() and 4 <= len(t) <= 24))
        corrected = {term: value for term in unknown[:16] if (value := self._correct(term)) is not None}
        return {"normalized_terms": terms, "search_concepts": concept_terms(query),
                "spelling_hypotheses": corrected, "spelling_checks_limited": len(unknown) > 16,
                "observation_modified": False, "ranking_is_confidence": False}

    def _section_ranks(self, scores: np.ndarray) -> list[tuple[str, int]]:
        """Apply the candidate budget after distinct authorities are selected."""
        selected = []
        seen = set()
        for index in self._rank(scores, len(scores)):
            chunk = self.chunks[index]
            section = str(chunk.get("section") or chunk.get("citation") or index)
            if section not in seen:
                seen.add(section)
                selected.append((section, index))
                if len(selected) == self.authority_pool_size:
                    break
        return selected

    def match_diagnostics(self, query: str, candidate: RetrievalCandidate | Mapping[str, Any]) -> dict[str, Any]:
        """Describe retrieval overlap without another semantic request or verdict.

        Semantic scores come only from a cached search for that exact query.
        An uncached query gets fresh lexical ranks and no semantic score. Later
        focused searches do not discard the original query's cached evidence.
        """
        analysis = self.explain_query(query)
        value = candidate.get if isinstance(candidate, Mapping) else lambda key, default=None: getattr(candidate, key, default)
        section = str(value("section", value("citation", "")))
        source_text = str(value("text", ""))
        text_scope = "candidate_text"
        if not source_text:
            index = value("chunk_index")
            if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(self.chunks) and str(self.chunks[index].get("section", "")) == section:
                source_text = str(self.chunks[index].get("text", ""))
                text_scope = "verified_corpus_chunk"
            else:
                source_text = str(value("excerpt", ""))
                text_scope = "provided_excerpt"
        meaningful = set(analysis["normalized_terms"]) - _GENERIC_STEMS - _LOW_INFORMATION_STEMS
        shared = meaningful.intersection(normalized_terms(source_text))
        concepts = set(analysis["search_concepts"])
        source_concepts = set(concept_terms(source_text))
        cached = self._query_diagnostics.get(query, {})
        ranks = cached or {
            "raw_ranks": {section: rank for rank, (section, _) in enumerate(self._section_ranks(self.lexical.scores(query)), 1)},
            "normalized_ranks": {section: rank for rank, (section, _) in enumerate(self._section_ranks(self.normalized.scores(analysis["normalized_terms"], generic_weight=.12)), 1)},
            "concept_ranks": {section: rank for rank, (section, _) in enumerate(self._section_ranks(self.concepts.scores(analysis["search_concepts"], coverage=True)), 1)},
        }
        return {
            "profile": self.profile,
            "diagnostic_text_scope": text_scope,
            "meaningful_query_terms": sorted(meaningful),
            "meaningful_shared_terms": sorted(shared),
            "query_search_concepts": sorted(concepts),
            "matched_search_concepts": sorted(concepts.intersection(source_concepts)),
            "spelling_hypotheses": analysis["spelling_hypotheses"],
            "raw_authority_rank": ranks.get("raw_ranks", {}).get(section),
            "normalized_authority_rank": ranks.get("normalized_ranks", {}).get(section),
            "concept_authority_rank": ranks.get("concept_ranks", {}).get(section),
            "semantic_score": cached.get("semantic_scores", {}).get(section),
            "semantic_search_used": cached.get("semantic_search_used", False),
            "diagnostics_from_cached_search": bool(cached),
            "diagnostics_from_latest_search": cached.get("generation") == self._search_generation,
            "ranking_is_confidence": False,
            "evidence_is_applicability_verdict": False,
        }

    def search(self, queries: Sequence[str], top_k: int = 12, query_vectors=None,
               query_weights: Sequence[float] | None = None) -> list[RetrievalCandidate]:
        self._search_generation += 1
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not queries or top_k == 0:
            return []
        if isinstance(queries, (str, bytes)) or len(queries) > 32 or any(not isinstance(query, str) for query in queries):
            raise ValueError("Use at most 32 text queries")
        if any(len(query) > 4000 for query in queries):
            raise ValueError("Queries must be text with at most 4000 characters")
        if query_weights is None:
            query_weights = [1.0 / (1.0 + .25 * index) for index in range(len(queries))]
        if len(query_weights) != len(queries) or any(not math.isfinite(float(w)) or float(w) < 0 for w in query_weights):
            raise ValueError("Query weights must be finite, nonnegative and aligned")
        if query_vectors is None and self.embedder is not None and self.corpus_vectors is not None:
            query_vectors = self.embedder(list(queries))
        if query_vectors is not None and len(query_vectors) != len(queries):
            raise ValueError("Query vectors must align with queries")
        fusion = defaultdict(float); peak = defaultdict(float); chunk_fusion = defaultdict(float)
        matched = defaultdict(set); lexical = defaultdict(float); semantic = defaultdict(float)
        section_indices = defaultdict(set)
        for query_index, query in enumerate(queries):
            if not query.strip() or not query_weights[query_index]:
                continue
            analysis = self.explain_query(query)
            raw = self.lexical.scores(query)
            normal = self.normalized.scores(analysis["normalized_terms"], generic_weight=.12)
            concepts = self.concepts.scores(analysis["search_concepts"], coverage=True)
            channels = [(raw, .3, "literal"), (normal, 1.0, "literal"), (concepts, 1.0, "concept")]
            literal_terms = set(analysis["normalized_terms"])
            if analysis["spelling_hypotheses"]:
                corrected = [analysis["spelling_hypotheses"].get(t, t) for t in analysis["normalized_terms"]]
                literal_terms.update(corrected)
                channels.append((self.normalized.scores(corrected, generic_weight=.12), .55, "literal"))
            dense = None
            if self.semantic_index is not None:
                dense = np.asarray(self.semantic_index.query_scores(query), dtype=float)
            elif query_vectors is not None and self.corpus_vectors is not None:
                dense = np.asarray(self.corpus_vectors @ query_vectors[query_index], dtype=float)
            if dense is not None:
                if dense.shape != (len(self.chunks),) or not np.isfinite(dense).all():
                    raise ValueError("Semantic scores must be finite and aligned to corpus chunks")
                channels.append((dense, 1.0, "semantic"))
            semantic_scores: dict[str, float] = {}
            if dense is not None:
                for index, chunk in enumerate(self.chunks):
                    section = str(chunk.get("section") or chunk.get("citation") or index)
                    semantic_scores[section] = max(semantic_scores.get(section, -math.inf), float(dense[index]))
            if len(self._query_diagnostics) >= 128 and query not in self._query_diagnostics:
                self._query_diagnostics.pop(next(iter(self._query_diagnostics)))
            self._query_diagnostics[query] = {
                "generation": self._search_generation,
                "raw_ranks": {section: rank for rank, (section, _) in enumerate(self._section_ranks(raw), 1)},
                "normalized_ranks": {section: rank for rank, (section, _) in enumerate(self._section_ranks(normal), 1)},
                "concept_ranks": {section: rank for rank, (section, _) in enumerate(self._section_ranks(concepts), 1)},
                "semantic_scores": semantic_scores,
                "semantic_search_used": dense is not None,
            }
            for scores, channel_weight, channel_kind in channels:
                for rank, (section, index) in enumerate(self._section_ranks(scores), 1):
                    contribution = float(query_weights[query_index]) * channel_weight / (self.rank_constant + rank)
                    if channel_kind == "literal":
                        weights = {term: .12 if term in _GENERIC_STEMS else 1.0 for term in literal_terms}
                        denominator = sum(weights.values())
                        coverage = sum(weight for term, weight in weights.items() if term in self.normalized.counts[index]) / denominator if denominator else 0.0
                        contribution *= coverage
                    elif channel_kind == "concept":
                        # Rank alone compresses a complete concept match and a
                        # partial one into almost identical RRF contributions.
                        requested_concepts = analysis["search_concepts"]
                        coverage = len(set(requested_concepts).intersection(self.concepts.counts[index])) / len(set(requested_concepts))
                        contribution *= coverage ** 2
                    if contribution <= 0:
                        continue
                    fusion[section] += contribution
                    peak[section] = max(peak[section], contribution)
                    chunk_fusion[index] += contribution
                    section_indices[section].add(index)
                    matched[section].add(query_index)
                    lexical[section] = max(lexical[section], float(raw[index]))
                    if dense is not None:
                        semantic[section] = max(semantic[section], float(dense[index]))
        order = sorted(fusion, key=lambda s: (-fusion[s] - 2 * peak[s], s))[:top_k]
        results = []
        for section in order:
            indices = sorted(section_indices[section], key=lambda i: (-chunk_fusion[i], i))
            index = indices[0]; chunk = self.chunks[index]
            results.append(RetrievalCandidate(chunk_index=index, citation=str(chunk.get("citation", section)),
                section=str(chunk.get("section", section)), heading=str(chunk.get("heading", "")),
                topic=str(chunk.get("topic", "")), text=" ".join(str(self.chunks[i].get("text", "")) for i in indices[:3]),
                lexical_score=round(lexical[section], 6), semantic_score=round(semantic[section], 6),
                fusion_score=round(fusion[section] + 2 * peak[section], 8), matched_queries=sorted(matched[section])))
        return results
