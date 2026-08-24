"""Continuous Learning & Episodic Memory Engine for Industrial Compliance Auditor.

This module provides persistent memory, few-shot precedent retrieval, pitfall avoidance,
and dynamic hazard pillar discovery that evolves with every audit execution.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"
MEMORY_FILE = JSON_DIR / "learning_memory.json"
MEMORY_VEC_FILE = JSON_DIR / "learning_memory_vecs.npz"
STATE_FILE = JSON_DIR / "learning_state.json"
RUN_HISTORY_FILE = JSON_DIR / "run_history.json"
REGS_FILE = JSON_DIR / "regulations.json"

OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen3.5:9b"

# Initial base compliance pillars
DEFAULT_PILLARS = {
    "fatigue": "49 CFR 228 limitations on duty hours maximum 12 consecutive hours operator fatigue time on duty",
    "housekeeping": "29 CFR 1910.22 walking-working surfaces housekeeping floors passageways clean dry spills leaks hazards",
    "clearance": "29 CFR 1910.333 electrical safety related work practices minimum approach distance energized high-voltage line clearance",
    "equipment": "29 CFR 1910.178 1910.179 powered industrial trucks overhead gantry cranes safe operation and maintenance",
}


def embed_texts(texts: List[str], model: str = EMBED_MODEL, batch: int = 32) -> np.ndarray:
    """Compute normalized vector embeddings via Ollama."""
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)
    out = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i + batch]
        r = requests.post(
            f"{OLLAMA}/api/embed",
            json={"model": model, "input": batch_texts},
            timeout=180,
        )
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    v = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class EpisodicMemoryBank:
    """Stores and retrieves historical audit episodes, verified precedents, and pitfall counterexamples."""

    def __init__(self, memory_file: Path = MEMORY_FILE, vec_file: Path = MEMORY_VEC_FILE):
        self.memory_file = memory_file
        self.vec_file = vec_file
        self.episodes: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
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
        self.memory_file.write_text(json.dumps(self.episodes, indent=2), encoding="utf-8")
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
        """Add an audit episode into persistent episodic memory."""
        query_text = (
            f"Equipment: {row.get('Equipment_ID', '')} ({row.get('Equipment_Type', '')}) at {row.get('Location', '')}. "
            f"Shift: {row.get('Operator_Shift_Hours', '')} hrs. Incident: {row.get('Reported_Incident', '')}."
        )

        # Avoid exact duplicate log IDs
        for ep in self.episodes:
            if ep.get("log_id") == log_id:
                ep.update({
                    "verdict": verdict,
                    "citation": citation,
                    "reason": reason,
                    "is_grounded": is_grounded,
                    "confidence": confidence,
                    "critique_notes": critique_notes,
                    "feedback_type": feedback_type,
                    "timestamp": time.time(),
                })
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
            "row": row,
            "verdict": verdict,
            "citation": citation,
            "reason": reason,
            "is_grounded": is_grounded,
            "confidence": confidence,
            "critique_notes": critique_notes,
            "feedback_type": feedback_type,
            "run_id": run_id,
            "timestamp": time.time(),
        }
        self.episodes.append(episode_record)
        self.save()

    def retrieve_precedents(
        self,
        query_text: str,
        k: int = 2,
        min_similarity: float = 0.55,
        only_grounded: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant past verified precedents as few-shot exemplars."""
        if not self.episodes or self.vectors is None:
            return []

        q_vec = embed_texts([query_text])[0]
        sims = self.vectors @ q_vec
        ranked_indices = np.argsort(-sims)

        results = []
        for idx in ranked_indices:
            ep = self.episodes[idx]
            sim = float(sims[idx])
            if sim < min_similarity:
                continue
            if only_grounded and not ep.get("is_grounded", True):
                continue
            if ep.get("feedback_type") == "pitfall":
                continue

            results.append({
                "log_id": ep["log_id"],
                "similarity": round(sim, 3),
                "verdict": ep["verdict"],
                "citation": ep["citation"],
                "reason": ep["reason"],
                "incident": ep["row"].get("Reported_Incident", ""),
                "location": ep["row"].get("Location", ""),
                "shift_hours": ep["row"].get("Operator_Shift_Hours", ""),
            })
            if len(results) >= k:
                break
        return results

    def retrieve_contrastive_precedents(
        self,
        query_text: str,
        min_similarity: float = 0.50,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Retrieve balanced contrastive precedents: 1 positive violation precedent + 1 negative compliant precedent."""
        if not self.episodes or self.vectors is None:
            return {"positive": None, "negative": None}

        q_vec = embed_texts([query_text])[0]
        sims = self.vectors @ q_vec
        ranked_indices = np.argsort(-sims)

        pos_match = None
        neg_match = None

        for idx in ranked_indices:
            ep = self.episodes[idx]
            sim = float(sims[idx])
            if sim < min_similarity:
                continue
            if not ep.get("is_grounded", True) or ep.get("feedback_type") == "pitfall":
                continue

            item = {
                "log_id": ep["log_id"],
                "similarity": round(sim, 3),
                "verdict": ep["verdict"],
                "citation": ep["citation"],
                "reason": ep["reason"],
                "incident": ep["row"].get("Reported_Incident", ""),
                "location": ep["row"].get("Location", ""),
                "shift_hours": ep["row"].get("Operator_Shift_Hours", ""),
            }

            if ep["verdict"] == "VIOLATION" and pos_match is None:
                pos_match = item
            elif ep["verdict"] == "CLEAR" and neg_match is None:
                neg_match = item

            if pos_match is not None and neg_match is not None:
                break

        return {"positive": pos_match, "negative": neg_match}

    def retrieve_pitfalls(
        self,
        query_text: str,
        k: int = 1,
        min_similarity: float = 0.60,
    ) -> List[Dict[str, Any]]:
        """Retrieve known negative precedents (pitfalls / past false alarms) to warn against misapplication."""
        if not self.episodes or self.vectors is None:
            return []

        q_vec = embed_texts([query_text])[0]
        sims = self.vectors @ q_vec
        ranked_indices = np.argsort(-sims)

        pitfalls = []
        for idx in ranked_indices:
            ep = self.episodes[idx]
            sim = float(sims[idx])
            if sim < min_similarity:
                continue
            if ep.get("feedback_type") in ("pitfall", "corrected") or not ep.get("is_grounded", True):
                pitfalls.append({
                    "log_id": ep["log_id"],
                    "similarity": round(sim, 3),
                    "past_mistake": ep.get("critique_notes") or f"Incorrectly cited {ep.get('citation')}",
                    "correct_verdict": ep["verdict"],
                    "correct_citation": ep["citation"],
                })
                if len(pitfalls) >= k:
                    break
        return pitfalls


class AdaptivePillarBank:
    """Manages and dynamically expands hazard triangulation dimensions as new telemetry is encountered."""

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.pillars: Dict[str, str] = dict(DEFAULT_PILLARS)
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
        self.state_file.write_text(json.dumps(current_state, indent=2), encoding="utf-8")

    def register_new_hazard_pillar(self, hazard_name: str, query_definition: str) -> bool:
        """Dynamically add a new compliance pillar discovered during audit operations."""
        if hazard_name in self.pillars and self.pillars[hazard_name] == query_definition:
            return False

        print(f"[PillarBank] Discovered new regulatory hazard pillar: [{hazard_name}] -> '{query_definition}'")
        self.pillars[hazard_name] = query_definition
        self._recompute_vectors()
        self.save()
        return True

    def scan_and_adapt(self, row: Dict[str, Any]) -> Optional[str]:
        """Scan event telemetry for novel hazard patterns and expand pillars if needed."""
        incident = str(row.get("Reported_Incident", "")).lower()
        location = str(row.get("Location", "")).lower()

        # Novel pattern detection
        if "hazmat" in incident or "placard" in incident or "chemical" in incident:
            if "hazmat" not in self.pillars:
                self.register_new_hazard_pillar(
                    "hazmat",
                    "49 CFR 172 hazardous materials regulations placards shipping papers uncontained dangerous goods",
                )
                return "hazmat"

        if "fall" in incident or "guardrail" in incident or "scaffold" in incident or "harness" in incident:
            if "fall_protection" not in self.pillars:
                self.register_new_hazard_pillar(
                    "fall_protection",
                    "29 CFR 1910.28 1910.140 duty to have fall protection guardrail systems personal fall arrest",
                )
                return "fall_protection"

        if "lockout" in incident or "tagout" in incident or "stored energy" in incident:
            if "lockout_tagout" not in self.pillars:
                self.register_new_hazard_pillar(
                    "lockout_tagout",
                    "29 CFR 1910.147 control of hazardous energy lockout tagout servicing and maintenance",
                )
                return "lockout_tagout"

        if "brake" in incident or "air hose" in incident or "coupler" in incident:
            if "rail_braking" not in self.pillars:
                self.register_new_hazard_pillar(
                    "rail_braking",
                    "49 CFR 232 railroad power brakes brake system safety standards inspections tests",
                )
                return "rail_braking"

        return None


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
            "citations_outside_retrieved_set": run_metrics.get("citations_outside_retrieved_set", 0),
            "citation_validity": run_metrics.get("citation_validity", 1.0),
            "memory_precedents_stored": run_metrics.get("memory_count", 0),
            "active_pillars_count": run_metrics.get("pillar_count", 4),
            "critique_corrections": run_metrics.get("critique_corrections", 0),
            "runtime_seconds": run_metrics.get("seconds", 0.0),
        }
        self.history.append(run_record)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return run_record
