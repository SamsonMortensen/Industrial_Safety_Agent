"""Continuous Learning & Episodic Memory Engine for Industrial Compliance Auditor.

This module provides persistent memory, few-shot precedent retrieval, pitfall avoidance,
and dynamic hazard pillar discovery that evolves with every audit execution.
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from learning_policy import assess_for_learning, episode_is_trusted
from ollama_config import ollama_base_url

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
MEMORY_FILE = JSON_DIR / "learning_memory.json"
MEMORY_VEC_FILE = JSON_DIR / "learning_memory_vecs.npz"
STATE_FILE = JSON_DIR / "learning_state.json"
RUN_HISTORY_FILE = JSON_DIR / "run_history.json"
REGS_FILE = JSON_DIR / "regulations.json"

OLLAMA = ollama_base_url()
EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen3.5:9b"
ECFR_EDITION = "2025-01-01"

# Leakage guards for episodic precedent retrieval.
# An episode at or above this cosine similarity is the same operational event
# restated, not a precedent -- surfacing it would hand the model its own answer.
MAX_PRECEDENT_SIMILARITY = 0.98

# Initial base compliance pillars
DEFAULT_PILLARS = {
    "fatigue": "49 U.S.C. 21103 limitations on duty hours maximum 12 consecutive hours train employee covered service",
    "housekeeping": "29 CFR 1910.22 walking-working surfaces housekeeping floors passageways clean dry spills leaks hazards",
    "clearance": "29 CFR 1910.333 electrical safety related work practices minimum approach distance energized high-voltage line clearance",
    "equipment": "29 CFR 1910.178 1910.179 powered industrial trucks overhead gantry cranes safe operation and maintenance",
}


def build_query_text(row: Dict[str, Any]) -> str:
    """Canonical text form of a yard event.

    Ingestion and retrieval MUST use this same template. When they diverge, an
    event no longer lands near its own stored episode in vector space, which
    silently disables any similarity-based duplicate guard.
    """
    return (
        f"Equipment: {row.get('Equipment_Type', '')} at {row.get('Location', '')}, "
        f"shift: {row.get('Operator_Shift_Hours', '')} hrs, "
        f"incident: {row.get('Reported_Incident', '')}"
    )


def event_signature(row: Dict[str, Any]) -> str:
    """Observable-feature fingerprint of an event.

    Two events sharing a signature are indistinguishable to the auditor, so one
    cannot serve as a precedent for the other without simply restating its
    verdict. This check is deterministic and does not depend on how well the
    embedding model happens to be calibrated.
    """
    parts = [
        str(row.get("Equipment_Type", "")).strip().lower(),
        str(row.get("Location", "")).strip().lower(),
        str(row.get("Operator_Shift_Hours", "")).strip(),
        str(row.get("Reported_Incident", "")).strip().lower(),
    ]
    return "|".join(parts)


def parse_event_time(row: Dict[str, Any]) -> Optional[float]:
    """Epoch seconds for a yard event's own Timestamp column.

    This is deliberately NOT wall-clock audit time: temporal isolation has to be
    judged on when the event happened in the yard, not when we happened to audit it.
    Returns None when the row carries no parseable timestamp.
    """
    raw = str(row.get("Timestamp", "")).strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt).timestamp()
        except ValueError:
            continue
    return None


def embed_texts(
    texts: List[str], model: str = EMBED_MODEL, batch: int = 32
) -> np.ndarray:
    """Compute normalized vector embeddings via Ollama."""
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)
    print(f"Embedding {len(texts)} texts with {model}...", file=sys.stderr, flush=True)
    out = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i : i + batch]
        r = requests.post(
            f"{OLLAMA}/api/embed",
            json={"model": model, "input": batch_texts},
            timeout=180,
        )
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
        print(
            f"  embedded {min(i + batch, len(texts))}/{len(texts)}",
            file=sys.stderr,
            flush=True,
        )
    v = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class EpisodicMemoryBank:
    """Stores and retrieves historical audit episodes, verified precedents, and pitfall counterexamples."""

    def __init__(
        self, memory_file: Path = MEMORY_FILE, vec_file: Path = MEMORY_VEC_FILE
    ):
        self.memory_file = memory_file
        self.vec_file = vec_file
        self.episodes: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self.last_guard_stats: Dict[str, int] = {}
        self._load()

    def _load(self):
        """Load stored episodes and embeddings from disk."""
        if self.memory_file.exists():
            try:
                self.episodes = json.loads(self.memory_file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[MemoryBank] Warning loading {self.memory_file}: {e}")
                self.episodes = []

        if self.vec_file.exists() and len(self.episodes) > 0:
            try:
                data = np.load(self.vec_file)
                if data["vectors"].shape[0] == len(self.episodes):
                    self.vectors = data["vectors"]
                else:
                    self._reindex_vectors()
            except Exception:
                self._reindex_vectors()
        elif len(self.episodes) > 0:
            self._reindex_vectors()

    def _reindex_vectors(self):
        """Recompute and save normalized vectors for all episodes."""
        if not self.episodes:
            self.vectors = None
            return
        texts = [ep["query_text"] for ep in self.episodes]
        self.vectors = embed_texts(texts)
        self.vec_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.vec_file, vectors=self.vectors)

    def save(self):
        """Persist memory records and vector embeddings."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(
            json.dumps(self.episodes, indent=2), encoding="utf-8"
        )
        if self.vectors is not None:
            np.savez_compressed(self.vec_file, vectors=self.vectors)

    def add_episode(
        self,
        log_id: str,
        row: Dict[str, Any],
        verdict: str,
        citation: Optional[str],
        reason: str,
        is_grounded: bool = True,
        confidence: float = 1.0,
        critique_notes: Optional[str] = None,
        feedback_type: str = "verified",
        run_id: str = "run_1",
    ) -> None:
        """Add an audit episode after independently classifying its trust level."""
        query_text = build_query_text(row)
        learning = assess_for_learning(row, verdict, citation, is_grounded)
        feedback_type = learning.feedback_type

        # Avoid exact duplicate log IDs
        for ep in self.episodes:
            if ep.get("log_id") == log_id:
                ep.update(
                    {
                        "verdict": verdict,
                        "citation": citation,
                        "reason": reason,
                        "is_grounded": is_grounded,
                        "confidence": confidence,
                        "critique_notes": critique_notes,
                        "feedback_type": feedback_type,
                        "verification_source": learning.verification_source,
                        "learning_reason": learning.reason,
                        "correct_verdict": learning.correct_verdict,
                        "correct_citation": learning.correct_citation,
                        "timestamp": time.time(),
                    }
                )
                self.save()
                return

        new_vec = embed_texts([query_text])
        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = new_vec
        else:
            self.vectors = np.vstack([self.vectors, new_vec])

        episode_record = {
            "log_id": log_id,
            "query_text": query_text,
            "event_time": parse_event_time(row),
            "signature": event_signature(row),
            "row": row,
            "verdict": verdict,
            "citation": citation,
            "reason": reason,
            "is_grounded": is_grounded,
            "confidence": confidence,
            "critique_notes": critique_notes,
            "feedback_type": feedback_type,
            "verification_source": learning.verification_source,
            "learning_reason": learning.reason,
            "correct_verdict": learning.correct_verdict,
            "correct_citation": learning.correct_citation,
            "run_id": run_id,
            "timestamp": time.time(),
        }
        self.episodes.append(episode_record)
        self.save()

    def _ranked_eligible(
        self,
        query_text: str,
        exclude_log_id: Optional[str],
        before_event_time: Optional[float],
        max_similarity: float,
        exclude_signature: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        """Rank episodes by similarity to the query, with leakage guards applied.

        Three guards, all active by default:
          1. self-exclusion    an episode may never serve as its own precedent
          2. near-duplicate    sim >= max_similarity is the same event restated
          3. temporal order    only strictly earlier events are visible

        Guard 3 is conservative: an episode with no recorded event_time cannot be
        shown to precede the query, so it is dropped whenever a temporal bound is
        supplied. Legacy memory files written before event_time existed will
        therefore contribute nothing under temporal isolation, which is the
        intended behaviour -- unprovable ordering is treated as contamination.
        """
        if not self.episodes or self.vectors is None:
            self.last_guard_stats = {}
            return []

        q_vec = embed_texts([query_text])[0]
        sims = self.vectors @ q_vec

        eligible: List[Tuple[int, float]] = []
        skipped = {"self": 0, "duplicate": 0, "not_earlier": 0}

        for raw_idx in np.argsort(-sims):
            idx = int(raw_idx)
            ep = self.episodes[idx]
            sim = float(sims[idx])

            if exclude_log_id is not None and ep.get("log_id") == exclude_log_id:
                skipped["self"] += 1
                continue
            if sim >= max_similarity:
                skipped["duplicate"] += 1
                continue
            if (
                exclude_signature is not None
                and ep.get("signature") == exclude_signature
            ):
                skipped["duplicate"] += 1
                continue
            if before_event_time is not None:
                ep_time = ep.get("event_time")
                if ep_time is None or float(ep_time) >= before_event_time:
                    skipped["not_earlier"] += 1
                    continue

            eligible.append((idx, sim))

        self.last_guard_stats = skipped
        return eligible

    @staticmethod
    def _as_item(ep: Dict[str, Any], sim: float) -> Dict[str, Any]:
        return {
            "log_id": ep["log_id"],
            "similarity": round(sim, 3),
            "verdict": ep["verdict"],
            "citation": ep["citation"],
            "reason": ep["reason"],
            "incident": ep["row"].get("Reported_Incident", ""),
            "location": ep["row"].get("Location", ""),
            "shift_hours": ep["row"].get("Operator_Shift_Hours", ""),
        }

    def retrieve_precedents(
        self,
        query_text: str,
        k: int = 2,
        min_similarity: float = 0.55,
        only_grounded: bool = True,
        exclude_log_id: Optional[str] = None,
        before_event_time: Optional[float] = None,
        max_similarity: float = MAX_PRECEDENT_SIMILARITY,
        exclude_signature: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant past verified precedents as few-shot exemplars."""
        results = []
        for idx, sim in self._ranked_eligible(
            query_text,
            exclude_log_id,
            before_event_time,
            max_similarity,
            exclude_signature,
        ):
            ep = self.episodes[idx]
            if sim < min_similarity:
                break
            if only_grounded and not episode_is_trusted(ep):
                continue
            if ep.get("feedback_type") != "verified":
                continue
            results.append(self._as_item(ep, sim))
            if len(results) >= k:
                break
        return results

    def retrieve_contrastive_precedents(
        self,
        query_text: str,
        min_similarity: float = 0.50,
        exclude_log_id: Optional[str] = None,
        before_event_time: Optional[float] = None,
        max_similarity: float = MAX_PRECEDENT_SIMILARITY,
        exclude_signature: Optional[str] = None,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Retrieve balanced contrastive precedents: 1 violation case + 1 compliant baseline."""
        pos_match = None
        neg_match = None

        for idx, sim in self._ranked_eligible(
            query_text,
            exclude_log_id,
            before_event_time,
            max_similarity,
            exclude_signature,
        ):
            ep = self.episodes[idx]
            if sim < min_similarity:
                break
            if not episode_is_trusted(ep) or ep.get("feedback_type") != "verified":
                continue

            if ep["verdict"] == "VIOLATION" and pos_match is None:
                pos_match = self._as_item(ep, sim)
            elif ep["verdict"] == "CLEAR" and neg_match is None:
                neg_match = self._as_item(ep, sim)

            if pos_match is not None and neg_match is not None:
                break

        return {"positive": pos_match, "negative": neg_match}

    def retrieve_pitfalls(
        self,
        query_text: str,
        k: int = 1,
        min_similarity: float = 0.60,
        exclude_log_id: Optional[str] = None,
        before_event_time: Optional[float] = None,
        max_similarity: float = MAX_PRECEDENT_SIMILARITY,
        exclude_signature: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve known pitfalls (past false alarms / ungrounded citations) as warnings."""
        pitfalls = []
        for idx, sim in self._ranked_eligible(
            query_text,
            exclude_log_id,
            before_event_time,
            max_similarity,
            exclude_signature,
        ):
            ep = self.episodes[idx]
            if sim < min_similarity:
                break
            if (
                ep.get("feedback_type") in ("pitfall", "corrected")
                and ep.get("verification_source") in {"ground_truth", "human_review"}
                and ep.get("correct_verdict")
            ):
                pitfalls.append(
                    {
                        "log_id": ep["log_id"],
                        "similarity": round(sim, 3),
                        "past_mistake": ep.get("critique_notes")
                        or f"Incorrectly cited {ep.get('citation')}",
                        "correct_verdict": ep["correct_verdict"],
                        "correct_citation": ep.get("correct_citation"),
                    }
                )
                if len(pitfalls) >= k:
                    break
        return pitfalls


class AdaptivePillarBank:
    """Manages and dynamically expands hazard triangulation dimensions as new telemetry is encountered."""

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.pillars: Dict[str, str] = dict(DEFAULT_PILLARS)
        self.gaps: Dict[str, Dict[str, Any]] = {}
        self.pillar_vectors: Optional[Dict[str, np.ndarray]] = None
        self._load()

    def _load(self):
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
                if "pillars" in state and isinstance(state["pillars"], dict):
                    self.pillars.update(state["pillars"])
            except Exception as e:
                print(f"[PillarBank] Warning loading state: {e}")
        self._recompute_vectors()

    def _recompute_vectors(self):
        keys = list(self.pillars.keys())
        texts = [self.pillars[k] for k in keys]
        vecs = embed_texts(texts)
        self.pillar_vectors = {k: vecs[i] for i, k in enumerate(keys)}

    def save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        current_state = {}
        if self.state_file.exists():
            try:
                current_state = json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                current_state = {}
        current_state["pillars"] = self.pillars
        current_state["last_updated"] = time.time()
        self.state_file.write_text(
            json.dumps(current_state, indent=2), encoding="utf-8"
        )

    def register_new_hazard_pillar(
        self, hazard_name: str, query_definition: str
    ) -> bool:
        """Dynamically add a new compliance pillar discovered during audit operations."""
        if (
            hazard_name in self.pillars
            and self.pillars[hazard_name] == query_definition
        ):
            return False

        print(
            f"[PillarBank] Discovered new regulatory hazard pillar: [{hazard_name}] -> '{query_definition}'"
        )
        self.pillars[hazard_name] = query_definition
        self._recompute_vectors()
        self.save()
        return True

    # Treat a valid but unretrieved citation as evidence of a retrieval gap.
    # Build candidates from regulatory text and register only candidates that
    # retrieve their source section.

    def record_retrieval_gap(
        self, section: str, row: Optional[Dict[str, Any]] = None
    ) -> None:
        """Note that a real section was cited but never retrieved."""
        if not section:
            return
        entry = self.gaps.setdefault(section, {"count": 0, "examples": []})
        entry["count"] += 1
        if row is not None and len(entry["examples"]) < 3:
            entry["examples"].append(str(row.get("Log_ID", "")))

    def _section_text(
        self, section: str, chunks: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Heading and body for a section, taken from the retrievable corpus."""
        parts = []
        for c in chunks:
            cit = str(c.get("citation", ""))
            if cit and cit.split()[-1] == section:
                parts.append(f"{c.get('heading', '')} {c.get('text', '')}")
        return " ".join(parts).strip() or None

    def _fetch_section_text(self, section: str) -> Optional[str]:
        """Pull a section from eCFR when the local corpus does not carry it."""
        title = (
            "49"
            if section.split(".")[0].isdigit() and int(section.split(".")[0]) < 1000
            else "29"
        )
        part = section.split(".")[0]
        url = (
            f"https://www.ecfr.gov/api/versioner/v1/full/{ECFR_EDITION}/"
            f"title-{title}.xml?part={part}&section={section}"
        )
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            text = re.sub(r"<[^>]+>", " ", r.text)
            return re.sub(r"\s+", " ", text).strip() or None
        except Exception as exc:
            print(
                f"[PillarBank] Could not fetch {section} from eCFR: {type(exc).__name__}"
            )
            return None

    @staticmethod
    def _derive_query(section: str, text: str, max_words: int = 60) -> str:
        """Build a retrieval query from the regulation's own words."""
        words = re.sub(r"\s+", " ", text).split()
        return f"{section} " + " ".join(words[:max_words])

    def promote_gaps(
        self,
        chunks: List[Dict[str, Any]],
        vectors: Optional[np.ndarray] = None,
        min_occurrences: int = 2,
    ) -> List[str]:
        """Turn recurring retrieval gaps into pillars, if they verify.

        Keep a candidate only when its query ranks the source section first.
        This prevents unverified pillars from expanding the search space.
        """
        promoted = []
        for section, entry in sorted(self.gaps.items(), key=lambda kv: -kv[1]["count"]):
            if entry["count"] < min_occurrences:
                continue
            name = f"gap_{section}"
            if name in self.pillars:
                continue

            text = self._section_text(section, chunks) or self._fetch_section_text(
                section
            )
            if not text:
                continue
            query = self._derive_query(section, text)

            if vectors is not None and len(chunks) == len(vectors):
                q_vec = embed_texts([query])[0]
                best = int(np.argmax(vectors @ q_vec))
                got = str(chunks[best].get("citation", "")).split()[-1]
                if got != section:
                    print(
                        f"[PillarBank] Rejected candidate for {section}: "
                        f"its own text retrieves {got} first."
                    )
                    continue

            self.register_new_hazard_pillar(name, query)
            promoted.append(section)
        return promoted


class ContinuousMetricsTracker:
    """Logs and tracks performance trajectories, hallucination rates, and learning progression across runs."""

    def __init__(self, history_file: Path = RUN_HISTORY_FILE):
        self.history_file = history_file
        self.history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.history_file.exists():
            try:
                self.history = json.loads(self.history_file.read_text(encoding="utf-8"))
            except Exception:
                self.history = []

    def record_run(self, run_metrics: Dict[str, Any]):
        run_record = {
            "run_id": run_metrics.get("run_id", f"run_{len(self.history) + 1}"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "events_audited": run_metrics.get("n", 0),
            "precision": run_metrics.get("precision", 0.0),
            "recall": run_metrics.get("recall", 0.0),
            "f1": run_metrics.get("f1", 0.0),
            "accuracy": run_metrics.get("accuracy", 0.0),
            "citations_given": run_metrics.get("citations_given", 0),
            "citations_fabricated": run_metrics.get("citations_fabricated", 0),
            "citations_outside_retrieved_set": run_metrics.get(
                "citations_outside_retrieved_set", 0
            ),
            "citation_validity": run_metrics.get("citation_validity", 1.0),
            "memory_precedents_stored": run_metrics.get("memory_count", 0),
            "active_pillars_count": run_metrics.get("pillar_count", 4),
            "critique_corrections": run_metrics.get("critique_corrections", 0),
            "runtime_seconds": run_metrics.get("seconds", 0.0),
        }
        self.history.append(run_record)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file.write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        return run_record
