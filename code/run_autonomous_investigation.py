"""Run autonomous investigation on camera-neutral observation JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "code"
JSON_DIR = ROOT / "json"
sys.path.insert(0, str(CODE_DIR))

from autonomous_investigator import (
    AutonomousInvestigator,
    HybridRetriever,
    Observation,
    OllamaGroundedReasoner,
    TriggerEngine,
)
from continuous_learner import embed_texts
from hybrid_policy import evaluate_event


def load_corpus(
    *, lexical_only: bool = False,
) -> tuple[List[Dict[str, Any]], np.ndarray | None]:
    regulations = json.loads(
        (JSON_DIR / "regulations.json").read_text(encoding="utf-8")
    )
    chunks = list(regulations["chunks"])
    statutes_path = JSON_DIR / "statutes.json"
    if statutes_path.exists():
        chunks.extend(
            json.loads(statutes_path.read_text(encoding="utf-8")).get("chunks", [])
        )

    if lexical_only:
        return chunks, None

    cache_path = JSON_DIR / "reg_embeddings.npz"
    vectors = None
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached["vectors"].shape[0] == len(chunks):
            vectors = cached["vectors"]
    if vectors is None:
        vectors = embed_texts(
            [
                f"{chunk.get('citation', '')} {chunk.get('heading', '')}. {chunk.get('text', '')}"
                for chunk in chunks
            ]
        )
        np.savez_compressed(cache_path, vectors=vectors)
    return chunks, vectors


def load_observations(path: Path) -> List[Observation]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        values = payload if isinstance(payload, list) else [payload]
    return [Observation.from_mapping(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigate structured camera or sensor observations against federal law."
    )
    parser.add_argument("observation", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--reasoner",
        action="store_true",
        help="Use the local grounded language model when deterministic policy returns REVIEW.",
    )
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--sample-rate", type=float, default=0.02)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument(
        "--lexical-only", action="store_true",
        help="Use lexical retrieval without embedding requests; --reasoner still needs Ollama.",
    )
    args = parser.parse_args()

    chunks, vectors = load_corpus(lexical_only=args.lexical_only)
    retriever = HybridRetriever(
        chunks, vectors, embedder=None if args.lexical_only else embed_texts
    )
    reasoner = None
    if args.reasoner:
        reasoner = OllamaGroundedReasoner(
            model=args.model,
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            timeout=int(os.getenv("AUDIT_TIMEOUT", "900")),
            num_ctx=int(os.getenv("AUDIT_NUM_CTX", "8192")),
        )
    investigator = AutonomousInvestigator(
        retriever,
        trigger_engine=TriggerEngine(sample_rate=args.sample_rate),
        policy_fn=evaluate_event,
        reasoner=reasoner,
        candidate_limit=args.candidate_limit,
    )
    report = {
        "source": str(args.observation.resolve()),
        "reasoner_enabled": args.reasoner,
        "retrieval_mode": "lexical" if args.lexical_only else "hybrid",
        "results": [
            investigator.investigate(observation).to_dict()
            for observation in load_observations(args.observation.resolve())
        ],
    }
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
