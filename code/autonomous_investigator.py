"""Autonomous observation-to-regulation investigation pipeline.

The existing auditor starts with a completed text log. This module defines the
boundary immediately before that point: a camera, sensor fusion model, or human
can submit a structured observation without naming a hazard or regulation.
The pipeline decides whether the observation deserves investigation, builds
several evidence-led legal queries, retrieves with lexical and semantic search,
reranks candidates for factual applicability, and returns a grounded decision
or REVIEW.

Perception models remain replaceable. They only need to produce Observation;
they never select the governing law or write directly to trusted memory.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

import numpy as np
import requests
from ollama_config import ollama_base_url

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
SECTION_RE = re.compile(r"\b(?:\d{3,4}\.\d+|21103)\b")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def tokenize(text: str) -> List[str]:
    """Tokenize operational and legal text without requiring an NLP service."""

    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def _strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


@dataclass(frozen=True)
class Relation:
    """One relationship inferred from a frame or temporal observation window."""

    subject: str
    predicate: str
    object: str
    value: Optional[float] = None
    unit: Optional[str] = None
    confidence: float = 1.0
    risk_score: Optional[float] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Relation":
        raw_value = value.get("value")
        try:
            numeric_value = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            numeric_value = None
        risk = value.get("risk_score")
        try:
            risk_score = float(risk) if risk is not None else None
        except (TypeError, ValueError):
            risk_score = None
        return cls(
            subject=str(value.get("subject", "object")),
            predicate=str(value.get("predicate", "related to")),
            object=str(value.get("object", "unknown")),
            value=numeric_value,
            unit=str(value.get("unit")) if value.get("unit") is not None else None,
            confidence=float(value.get("confidence", 1.0)),
            risk_score=risk_score,
        )

    def sentence(self) -> str:
        measurement = ""
        if self.value is not None:
            measurement = f" at {self.value:g}{(' ' + self.unit) if self.unit else ''}"
        return f"{self.subject} {self.predicate} {self.object}{measurement}".strip()


@dataclass(frozen=True)
class Observation:
    """Model-independent evidence emitted by camera, sensor, log, or reviewer."""

    event_id: str
    timestamp: str
    source_id: str
    summary: str = ""
    actors: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    location: str = ""
    actions: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)
    measurements: Dict[str, Any] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    perception_confidence: float = 1.0
    anomaly_score: float = 0.0
    novelty_score: float = 0.0
    changed: bool = False
    human_flag: bool = False
    evidence_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Observation":
        return cls(
            event_id=str(
                value.get("event_id") or value.get("Log_ID") or "event-unknown"
            ),
            timestamp=str(value.get("timestamp") or value.get("Timestamp") or ""),
            source_id=str(
                value.get("source_id") or value.get("Equipment_ID") or "unknown-source"
            ),
            summary=str(value.get("summary") or value.get("Reported_Incident") or ""),
            actors=_strings(value.get("actors")),
            equipment=_strings(value.get("equipment") or value.get("Equipment_Type")),
            location=str(value.get("location") or value.get("Location") or ""),
            actions=_strings(value.get("actions")),
            conditions=_strings(value.get("conditions")),
            controls=_strings(value.get("controls")),
            measurements=dict(value.get("measurements") or {}),
            relations=[
                item if isinstance(item, Relation) else Relation.from_mapping(item)
                for item in value.get("relations", [])
            ],
            perception_confidence=float(value.get("perception_confidence", 1.0)),
            anomaly_score=float(value.get("anomaly_score", 0.0)),
            novelty_score=float(value.get("novelty_score", 0.0)),
            changed=bool(value.get("changed", False)),
            human_flag=bool(value.get("human_flag", False)),
            evidence_refs=_strings(value.get("evidence_refs")),
            metadata=dict(value.get("metadata") or {}),
        )

    def evidence_text(self) -> str:
        parts = [
            self.summary,
            " ".join(self.actors),
            " ".join(self.equipment),
            self.location,
            " ".join(self.actions),
            " ".join(self.conditions),
            " ".join(self.controls),
            " ".join(relation.sentence() for relation in self.relations),
            " ".join(f"{key} {value}" for key, value in self.measurements.items()),
        ]
        return " ".join(part for part in parts if part).strip()

    def to_legacy_row(self) -> Dict[str, Any]:
        """Map camera-neutral evidence into the existing deterministic policy."""

        shift = self.measurements.get(
            "operator_shift_hours", self.measurements.get("shift_hours")
        )
        incident_parts = [
            self.summary,
            *self.actions,
            *self.conditions,
            *(relation.sentence() for relation in self.relations),
        ]
        if self.controls:
            incident_parts.append("controls observed: " + ", ".join(self.controls))
        return {
            "Log_ID": self.event_id,
            "Timestamp": self.timestamp,
            "Equipment_ID": self.source_id,
            "Equipment_Type": ", ".join(self.equipment),
            "Location": self.location,
            "Operator_Shift_Hours": shift,
            "Reported_Incident": "; ".join(part for part in incident_parts if part),
        }


@dataclass(frozen=True)
class TriggerDecision:
    triggered: bool
    score: float
    reasons: List[str]


class TriggerEngine:
    """Decide whether an observation deserves regulatory investigation.

    The trigger is deliberately hazard-name agnostic. It responds to changes,
    anomaly or novelty signals, uncertain relationships, risk scores emitted by
    perception, human flags, and deterministic sampling of quiet observations.
    """

    def __init__(
        self,
        anomaly_threshold: float = 0.65,
        novelty_threshold: float = 0.75,
        relation_risk_threshold: float = 0.55,
        uncertainty_low: float = 0.35,
        uncertainty_high: float = 0.80,
        sample_rate: float = 0.02,
        trigger_on_change: bool = True,
    ):
        self.anomaly_threshold = anomaly_threshold
        self.novelty_threshold = novelty_threshold
        self.relation_risk_threshold = relation_risk_threshold
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.trigger_on_change = trigger_on_change

    def _sampled(self, observation: Observation) -> bool:
        if self.sample_rate <= 0:
            return False
        digest = hashlib.sha256(observation.event_id.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        return value < self.sample_rate

    def evaluate(self, observation: Observation) -> TriggerDecision:
        reasons: List[str] = []
        scores: List[float] = []
        if observation.human_flag:
            reasons.append("human_flag")
            scores.append(1.0)
        if observation.anomaly_score >= self.anomaly_threshold:
            reasons.append("perception_anomaly")
            scores.append(observation.anomaly_score)
        if observation.novelty_score >= self.novelty_threshold:
            reasons.append("novel_observation")
            scores.append(observation.novelty_score)
        relation_risks = [
            relation.risk_score
            for relation in observation.relations
            if relation.risk_score is not None
        ]
        if relation_risks and max(relation_risks) >= self.relation_risk_threshold:
            reasons.append("relationship_risk")
            scores.append(max(relation_risks))
        if self.trigger_on_change and observation.changed:
            reasons.append("scene_or_state_change")
            scores.append(0.6)
        uncertain = (
            self.uncertainty_low
            <= observation.perception_confidence
            <= self.uncertainty_high
        )
        if uncertain and (observation.relations or observation.changed):
            reasons.append("questionable_observation")
            scores.append(1.0 - observation.perception_confidence)
        if self._sampled(observation):
            reasons.append("background_safety_sample")
            scores.append(0.25)
        return TriggerDecision(
            bool(reasons), round(max(scores, default=0.0), 4), reasons
        )


class QueryPlanner:
    """Translate observable facts into several legal-information queries."""

    def __init__(self, max_queries: int = 8):
        self.max_queries = max_queries

    @staticmethod
    def _deduplicate(queries: Iterable[str]) -> List[str]:
        seen = set()
        output = []
        for query in queries:
            normalized = " ".join(query.split()).strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                output.append(normalized)
        return output

    def plan(self, observation: Observation) -> List[str]:
        queries: List[str] = []
        if observation.summary:
            queries.append(observation.summary)
        for relation in sorted(
            observation.relations,
            key=lambda item: (item.risk_score or 0.0, item.confidence),
            reverse=True,
        ):
            queries.append(relation.sentence() + " applicable safety requirement")

        entities = " ".join(
            [*observation.actors, *observation.equipment, observation.location]
        ).strip()
        event_state = " ".join([*observation.actions, *observation.conditions]).strip()
        if entities or event_state:
            queries.append(f"{entities} {event_state} federal safety requirement")
        if observation.controls:
            queries.append(
                f"{entities} {' '.join(observation.controls)} required protective control"
            )
        for name, value in observation.measurements.items():
            queries.append(
                f"{entities} {event_state} {name.replace('_', ' ')} {value} "
                "applicable limit or minimum requirement"
            )
        if observation.novelty_score > 0:
            queries.append(
                f"{observation.evidence_text()} analogous regulated condition"
            )
        return self._deduplicate(queries)[: self.max_queries]


class BM25Index:
    """Small in-memory lexical index used alongside dense embeddings."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(document) for document in documents]
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        self.lengths = np.array(
            [len(tokens) for tokens in self.tokens], dtype=np.float32
        )
        self.avg_length = float(self.lengths.mean()) if len(self.lengths) else 1.0
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = max(len(self.tokens), 1)
        self.idf = {
            term: math.log(1.0 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        output = np.zeros(len(self.tokens), dtype=np.float32)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, counts in enumerate(self.term_counts):
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[index] / self.avg_length
                )
                output[index] += idf * frequency * (self.k1 + 1.0) / denominator
        return output


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk_index: int
    citation: str
    section: str
    heading: str
    topic: str
    text: str
    lexical_score: float
    semantic_score: float
    fusion_score: float
    matched_queries: List[int]
    applicability_score: float = 0.0
    applicability_features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_text: bool = False) -> Dict[str, Any]:
        value = asdict(self)
        if not include_text:
            value.pop("text", None)
        return value


class HybridRetriever:
    """Fuse lexical and semantic ranks across all generated queries."""

    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        corpus_vectors: Optional[np.ndarray] = None,
        embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
        rank_constant: int = 60,
        pool_size: int = 50,
    ):
        self.chunks = [dict(chunk) for chunk in chunks]
        self.corpus_vectors = corpus_vectors
        self.embedder = embedder
        self.rank_constant = rank_constant
        self.pool_size = min(pool_size, len(self.chunks))
        documents = [
            f"{chunk.get('citation', '')} {chunk.get('heading', '')} "
            f"{chunk.get('topic', '')} {chunk.get('text', '')}"
            for chunk in self.chunks
        ]
        self.lexical = BM25Index(documents)
        if corpus_vectors is not None and len(corpus_vectors) != len(self.chunks):
            raise ValueError("corpus_vectors must align one-to-one with chunks")

    @staticmethod
    def _rank(scores: np.ndarray, limit: int) -> List[int]:
        if not len(scores):
            return []
        return [
            int(index) for index in np.argsort(-scores)[:limit] if scores[index] > 0
        ]

    def search(
        self,
        queries: Sequence[str],
        top_k: int = 12,
        query_vectors: Optional[np.ndarray] = None,
        query_weights: Optional[Sequence[float]] = None,
    ) -> List[RetrievalCandidate]:
        queries = [query for query in queries if query.strip()]
        if not queries:
            return []
        if (
            query_vectors is None
            and self.embedder is not None
            and self.corpus_vectors is not None
        ):
            query_vectors = self.embedder(list(queries))
        if query_vectors is not None and len(query_vectors) != len(queries):
            raise ValueError("query_vectors must align one-to-one with queries")
        if query_weights is None:
            query_weights = [
                1.0 / (1.0 + 0.25 * index) for index in range(len(queries))
            ]
        if len(query_weights) != len(queries):
            raise ValueError("query_weights must align one-to-one with queries")

        fusion: Dict[int, float] = defaultdict(float)
        peak_contribution: Dict[int, float] = defaultdict(float)
        best_lexical: Dict[int, float] = defaultdict(float)
        best_semantic: Dict[int, float] = defaultdict(float)
        matched: Dict[int, set[int]] = defaultdict(set)

        for query_index, query in enumerate(queries):
            query_weight = float(query_weights[query_index])
            lexical_scores = self.lexical.scores(query)
            for rank, index in enumerate(
                self._rank(lexical_scores, self.pool_size), start=1
            ):
                contribution = query_weight / (self.rank_constant + rank)
                fusion[index] += contribution
                peak_contribution[index] = max(peak_contribution[index], contribution)
                best_lexical[index] = max(
                    best_lexical[index], float(lexical_scores[index])
                )
                matched[index].add(query_index)

            if query_vectors is not None and self.corpus_vectors is not None:
                semantic_scores = self.corpus_vectors @ query_vectors[query_index]
                for rank, index in enumerate(
                    self._rank(semantic_scores, self.pool_size), start=1
                ):
                    contribution = query_weight / (self.rank_constant + rank)
                    fusion[index] += contribution
                    peak_contribution[index] = max(
                        peak_contribution[index], contribution
                    )
                    best_semantic[index] = max(
                        best_semantic[index], float(semantic_scores[index])
                    )
                    matched[index].add(query_index)

        # Rank authorities, not chunks. A long section may have several chunks;
        # allowing all of them into top-k silently crowds out other governing
        # sections and makes recall depend on document length.
        by_section: Dict[str, List[int]] = defaultdict(list)
        for index in fusion:
            chunk = self.chunks[index]
            key = str(chunk.get("section") or chunk.get("citation") or index)
            by_section[key].append(index)

        section_order = sorted(
            by_section,
            key=lambda key: max(
                fusion[index] + 2.0 * peak_contribution[index]
                for index in by_section[key]
            ),
            reverse=True,
        )[:top_k]
        output = []
        for section in section_order:
            indices = sorted(
                by_section[section],
                key=lambda index: fusion[index] + 2.0 * peak_contribution[index],
                reverse=True,
            )
            best_index = indices[0]
            chunk = self.chunks[best_index]
            section_matches = sorted(
                {query_index for index in indices for query_index in matched[index]}
            )
            section_text = " ".join(
                str(self.chunks[index].get("text", "")) for index in indices[:3]
            )
            output.append(
                RetrievalCandidate(
                    chunk_index=best_index,
                    citation=str(chunk.get("citation", chunk.get("section", ""))),
                    section=str(chunk.get("section", "")),
                    heading=str(chunk.get("heading", "")),
                    topic=str(chunk.get("topic", "")),
                    text=section_text,
                    lexical_score=round(
                        max(best_lexical[index] for index in indices), 6
                    ),
                    semantic_score=round(
                        max(best_semantic[index] for index in indices), 6
                    ),
                    fusion_score=round(
                        max(
                            fusion[index] + 2.0 * peak_contribution[index]
                            for index in indices
                        ),
                        8,
                    ),
                    matched_queries=section_matches,
                )
            )
        return output


class ApplicabilityReranker:
    """Score whether a retrieved rule fits the observable actors and facts."""

    @staticmethod
    def _coverage(terms: Iterable[str], candidate_tokens: set[str]) -> float:
        selected = set(terms)
        if not selected:
            return 0.0
        return len(selected & candidate_tokens) / len(selected)

    @staticmethod
    def _measurement_match(observation: Observation, text: str) -> float:
        if not observation.measurements:
            return 0.0
        candidate_numbers = [float(value) for value in NUMBER_RE.findall(text)]
        if not candidate_numbers:
            return 0.0
        matches = 0
        measurable = 0
        for value in observation.measurements.values():
            try:
                observed = float(value)
            except (TypeError, ValueError):
                continue
            measurable += 1
            if any(
                abs(observed - candidate) <= max(0.1, abs(candidate) * 0.15)
                for candidate in candidate_numbers
            ):
                matches += 1
        return matches / measurable if measurable else 0.0

    def rerank(
        self,
        observation: Observation,
        candidates: Sequence[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        if not candidates:
            return []
        max_fusion = max(candidate.fusion_score for candidate in candidates) or 1.0
        entity_terms = tokenize(
            " ".join(
                [*observation.actors, *observation.equipment, observation.location]
            )
        )
        event_terms = tokenize(
            " ".join(
                [
                    *observation.actions,
                    *observation.conditions,
                    *(relation.sentence() for relation in observation.relations),
                ]
            )
        )
        control_terms = tokenize(" ".join(observation.controls))
        max_query_hits = (
            max(len(candidate.matched_queries) for candidate in candidates) or 1
        )
        output = []
        for candidate in candidates:
            candidate_tokens = set(
                tokenize(f"{candidate.heading} {candidate.topic} {candidate.text}")
            )
            features = {
                "retrieval": candidate.fusion_score / max_fusion,
                "entity_scope": self._coverage(entity_terms, candidate_tokens),
                "event_condition": self._coverage(event_terms, candidate_tokens),
                "control_match": self._coverage(control_terms, candidate_tokens),
                "measurement_match": self._measurement_match(
                    observation, candidate.text
                ),
                "query_coverage": len(candidate.matched_queries) / max_query_hits,
            }
            score = (
                0.55 * features["retrieval"]
                + 0.15 * features["entity_scope"]
                + 0.18 * features["event_condition"]
                + 0.05 * features["control_match"]
                + 0.05 * features["measurement_match"]
                + 0.02 * min(1.0, features["query_coverage"])
            )
            output.append(
                replace(
                    candidate,
                    applicability_score=round(score, 6),
                    applicability_features={
                        key: round(value, 6) for key, value in features.items()
                    },
                )
            )
        return sorted(output, key=lambda item: item.applicability_score, reverse=True)


@dataclass(frozen=True)
class GroundedDecision:
    status: str
    citation: Optional[str]
    reason: str
    source: str
    confidence: Optional[float] = None


class Reasoner(Protocol):
    def decide(
        self, observation: Observation, candidates: Sequence[RetrievalCandidate]
    ) -> GroundedDecision: ...


class OllamaGroundedReasoner:
    """Optional local reasoner constrained to the retrieved candidate set."""

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        host: str = "http://localhost:11434",
        timeout: int = 900,
        num_ctx: int = 8192,
    ):
        self.model = model
        self.host = ollama_base_url(host)
        self.timeout = timeout
        self.num_ctx = num_ctx

    def decide(
        self, observation: Observation, candidates: Sequence[RetrievalCandidate]
    ) -> GroundedDecision:
        context = "\n\n".join(
            f"[{candidate.citation} - {candidate.heading}]\n{candidate.text[:1200]}"
            for candidate in candidates[:8]
        )
        prompt = f"""You are reviewing structured safety evidence against retrieved federal law.
Use only the retrieved authorities. Do not infer an unobserved fact. If legal scope, a threshold,
or a required fact is missing, return REVIEW. A control observed in the evidence may make the event
CLEAR even when the surrounding activity is hazardous.

OBSERVATION:
{observation.evidence_text()}

RETRIEVED AUTHORITIES:
{context}

Return exactly:
STATUS: CLEAR, VIOLATION, or REVIEW
CITATION: one retrieved section number, or NONE
REASON: one evidence-led sentence.
"""
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 220,
                        "num_ctx": self.num_ctx,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            text = response.json()["message"]["content"]
        except Exception as exc:
            return GroundedDecision(
                "REVIEW",
                None,
                f"Local reasoner unavailable: {type(exc).__name__}.",
                "reasoner_error",
            )

        status_match = re.search(r"STATUS:\s*(CLEAR|VIOLATION|REVIEW)", text, re.I)
        citation_match = re.search(r"CITATION:\s*([^\n]+)", text, re.I)
        reason_match = re.search(r"REASON:\s*([^\n]+)", text, re.I)
        status = status_match.group(1).upper() if status_match else "REVIEW"
        raw_citation = citation_match.group(1) if citation_match else "NONE"
        section_match = SECTION_RE.search(raw_citation)
        citation = section_match.group(0) if section_match else None
        allowed = {candidate.section for candidate in candidates}
        reason = (
            reason_match.group(1).strip()
            if reason_match
            else "Response did not provide a reason."
        )
        if status == "VIOLATION" and citation not in allowed:
            return GroundedDecision(
                "REVIEW",
                None,
                "The proposed citation was not in the retrieved candidate set.",
                "grounding_guard",
                0.0,
            )
        if status != "VIOLATION":
            citation = None
        return GroundedDecision(status, citation, reason, "local_grounded_reasoner")


@dataclass(frozen=True)
class InvestigationResult:
    event_id: str
    status: str
    citation: Optional[str]
    reason: str
    decision_source: str
    trigger: TriggerDecision
    queries: List[str]
    candidates: List[Dict[str, Any]]
    evidence_refs: List[str]
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutonomousInvestigator:
    """Orchestrate trigger, planning, retrieval, applicability, and decision."""

    def __init__(
        self,
        retriever: HybridRetriever,
        trigger_engine: Optional[TriggerEngine] = None,
        query_planner: Optional[QueryPlanner] = None,
        reranker: Optional[ApplicabilityReranker] = None,
        policy_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        reasoner: Optional[Reasoner] = None,
        candidate_limit: int = 12,
    ):
        self.retriever = retriever
        self.trigger_engine = trigger_engine or TriggerEngine()
        self.query_planner = query_planner or QueryPlanner()
        self.reranker = reranker or ApplicabilityReranker()
        self.policy_fn = policy_fn
        self.reasoner = reasoner
        self.candidate_limit = candidate_limit

    def investigate(self, observation: Observation) -> InvestigationResult:
        trigger = self.trigger_engine.evaluate(observation)
        if not trigger.triggered:
            return InvestigationResult(
                observation.event_id,
                "NO_ACTION",
                None,
                "No change, anomaly, uncertainty, relationship risk, human flag, or sample trigger.",
                "trigger_engine",
                trigger,
                [],
                [],
                observation.evidence_refs,
                trigger.score,
            )

        queries = self.query_planner.plan(observation)
        candidates = self.retriever.search(queries, top_k=self.candidate_limit)
        candidates = self.reranker.rerank(observation, candidates)

        decision: Optional[GroundedDecision] = None
        policy_rejection_reason: Optional[str] = None
        policy_review_reason: Optional[str] = None
        if self.policy_fn is not None:
            policy = self.policy_fn(observation.to_legacy_row())
            if getattr(policy, "status", "REVIEW") == "REVIEW":
                policy_review_reason = getattr(policy, "reason", None)
            if getattr(policy, "status", "REVIEW") != "REVIEW":
                candidate_sections = {candidate.section for candidate in candidates}
                citation_is_grounded = (
                    policy.citation is None or policy.citation in candidate_sections
                )
                if citation_is_grounded:
                    decision = GroundedDecision(
                        policy.status,
                        policy.citation,
                        policy.reason,
                        "deterministic_policy",
                        1.0,
                    )
                else:
                    policy_rejection_reason = (
                        f"The deterministic policy proposed {policy.citation}, but that authority "
                        "was not present in the retrieved candidate set."
                    )
        if decision is None and self.reasoner is not None:
            decision = self.reasoner.decide(observation, candidates)
        if decision is None:
            decision = GroundedDecision(
                "REVIEW",
                None,
                policy_rejection_reason
                or policy_review_reason
                or "The observation triggered investigation, but no authoritative decision provider resolved it.",
                "review_gate",
            )

        return InvestigationResult(
            event_id=observation.event_id,
            status=decision.status,
            citation=decision.citation,
            reason=decision.reason,
            decision_source=decision.source,
            trigger=trigger,
            queries=queries,
            candidates=[candidate.to_dict() for candidate in candidates],
            evidence_refs=observation.evidence_refs,
            confidence=decision.confidence,
        )
