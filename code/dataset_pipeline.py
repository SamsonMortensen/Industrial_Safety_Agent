"""Automated Continuous Training Dataset Pipeline (DPO & SFT).

Extracts high-reward training pairs from episodic memory and self-reflection critique logs,
synthesizes Direct Preference Optimization (DPO) and Supervised Fine-Tuning (SFT) datasets,
and formats them for training with Hugging Face TRL, Unsloth, and PyTorch.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "json"

MEMORY_FILE = JSON_DIR / "learning_memory.json"
CRITIQUE_PAIRS_FILE = JSON_DIR / "dpo_pairs_raw.json"
REGS_FILE = JSON_DIR / "regulations.json"
LOG_FILE = ROOT / "daily_yard_log.csv"

OUT_DPO = JSON_DIR / "dpo_training_dataset.jsonl"
OUT_SFT = JSON_DIR / "sft_training_dataset.jsonl"
OUT_METRICS = JSON_DIR / "dataset_metrics.json"


class ComplianceDatasetPipeline:
    """Extracts, verifies, and packages training data from autonomous audit memories."""

    def __init__(self):
        self.memory_records: List[Dict[str, Any]] = []
        self.critique_pairs: List[Dict[str, Any]] = []
        self.regulations: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if MEMORY_FILE.exists():
            try:
                self.memory_records = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[DatasetPipeline] Warning loading memory: {e}")

        if CRITIQUE_PAIRS_FILE.exists():
            try:
                self.critique_pairs = json.loads(CRITIQUE_PAIRS_FILE.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[DatasetPipeline] Warning loading critique pairs: {e}")

        if REGS_FILE.exists():
            try:
                self.regulations = json.loads(REGS_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

    def build_sft_dataset(self) -> List[Dict[str, Any]]:
        """Constructs Supervised Fine-Tuning (SFT) instruction dataset."""
        sft_records = []
        for ep in self.memory_records:
            if not ep.get("is_grounded", True) or ep.get("feedback_type") == "pitfall":
                continue

            row = ep.get("row", {})
            input_text = (
                f"Log ID: {ep.get('log_id')}\n"
                f"Equipment: {row.get('Equipment_ID')} ({row.get('Equipment_Type')})\n"
                f"Location: {row.get('Location')}\n"
                f"Operator Shift Duration: {row.get('Operator_Shift_Hours')} hours\n"
                f"Reported Incident / Telemetry: {row.get('Reported_Incident')}"
            )
            output_text = (
                f"STATUS: {ep.get('verdict')}\n"
                f"CITATION: {ep.get('citation') or 'NONE'}\n"
                f"REASON: {ep.get('reason')}"
            )

            sft_records.append({
                "instruction": "You are a federal safety and compliance auditor for a rail-served intermodal yard. Audit this yard event against federal OSHA (Title 29) and FRA (Title 49) regulations. Cite exact section numbers if a violation exists, or NONE if compliant.",
                "input": input_text,
                "output": output_text,
                "log_id": ep.get("log_id"),
                "category": ep.get("verdict"),
            })
        return sft_records

    def build_dpo_dataset(self) -> List[Dict[str, Any]]:
        """Constructs Direct Preference Optimization (DPO) pairwise training dataset."""
        dpo_records = []

        # 1. High-value empirical self-reflection critique pairs
        for pair in self.critique_pairs:
            dpo_records.append({
                "log_id": pair.get("log_id"),
                "source": "self_reflection_critique",
                "prompt": pair.get("prompt"),
                "chosen": pair.get("chosen"),
                "rejected": pair.get("rejected"),
                "critique_reason": pair.get("critique_issues", []),
            })

        # 2. Synthetic hard-negative & anti-hallucination contrastive pairs from memory
        for ep in self.memory_records:
            row = ep.get("row", {})
            verdict = ep.get("verdict")
            citation = ep.get("citation") or "NONE"
            reason = ep.get("reason")
            log_id = ep.get("log_id")

            # Check if pair already exists
            if any(p.get("log_id") == log_id for p in dpo_records):
                continue

            prompt_text = (
                f"You are a federal compliance auditor. Audit the following intermodal yard event:\n"
                f"- Log ID: {log_id}\n"
                f"- Equipment: {row.get('Equipment_ID')} ({row.get('Equipment_Type')})\n"
                f"- Location: {row.get('Location')}\n"
                f"- Shift: {row.get('Operator_Shift_Hours')} hrs\n"
                f"- Incident: {row.get('Reported_Incident')}"
            )

            chosen_text = f"STATUS: {verdict}\nCITATION: {citation}\nREASON: {reason}"

            # Generate synthetic rejected counterexample
            if verdict == "CLEAR":
                # Rejected attempt: Over-eager false alarm citing non-applicable rule
                rejected_text = (
                    f"STATUS: VIOLATION\n"
                    f"CITATION: 1910.178\n"
                    f"REASON: Unnecessary citation issued on normal operational telemetry."
                )
            else:
                # Rejected attempt: Hallucinating a non-existent or ungrounded general rule
                rejected_text = (
                    f"STATUS: VIOLATION\n"
                    f"CITATION: 1910.999\n"
                    f"REASON: General safety violation citing ungrounded regulation."
                )

            dpo_records.append({
                "log_id": log_id,
                "source": "contrastive_memory_synthesis",
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
            })

        return dpo_records

    def export(self) -> Dict[str, Any]:
        """Processes and writes SFT and DPO training sets to JSONL."""
        sft_data = self.build_sft_dataset()
        dpo_data = self.build_dpo_dataset()

        # Shuffle deterministically
        random.seed(42)
        random.shuffle(sft_data)
        random.shuffle(dpo_data)

        # Write SFT JSONL
        OUT_SFT.parent.mkdir(parents=True, exist_ok=True)
        with OUT_SFT.open("w", encoding="utf-8") as f:
            for item in sft_data:
                f.write(json.dumps(item) + "\n")

        # Write DPO JSONL
        with OUT_DPO.open("w", encoding="utf-8") as f:
            for item in dpo_data:
                f.write(json.dumps(item) + "\n")

        # Compute dataset distribution metrics
        metrics = {
            "total_sft_examples": len(sft_data),
            "total_dpo_pairs": len(dpo_data),
            "critique_derived_pairs": sum(1 for p in dpo_data if p.get("source") == "self_reflection_critique"),
            "contrastive_pairs": sum(1 for p in dpo_data if p.get("source") == "contrastive_memory_synthesis"),
            "violations_count": sum(1 for s in sft_data if "VIOLATION" in s.get("output", "")),
            "clears_count": sum(1 for s in sft_data if "CLEAR" in s.get("output", "")),
            "dpo_file_path": "json/dpo_training_dataset.jsonl",
            "sft_file_path": "json/sft_training_dataset.jsonl",
        }
        OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics


def main():
    ap = argparse.ArgumentParser(description="Continuous Training Dataset Generator (DPO / SFT).")
    ap.add_argument("--export", action="store_true", default=True, help="Generate and export training datasets")
    ap.add_argument("--report", action="store_true", default=True, help="Print summary report")
    args = ap.parse_args()

    pipeline = ComplianceDatasetPipeline()
    metrics = pipeline.export()

    if args.report:
        print("\n" + "=" * 80)
        print("          AUTOMATED CONTINUOUS TRAINING DATASET PIPELINE")
        print("=" * 80)
        print(f"\n[Generated Datasets for Model Alignment & Fine-Tuning]:")
        print(f"  - DPO Preference Pairs File : {metrics['dpo_file_path']}")
        print(f"  - SFT Instruction File      : {metrics['sft_file_path']}")
        print(f"\n[Dataset Composition & Statistics]:")
        print(f"  - Total DPO Preference Pairs: {metrics['total_dpo_pairs']}")
        print(f"    * Live Critique Corrections: {metrics['critique_derived_pairs']}")
        print(f"    * Contrastive Precedents   : {metrics['contrastive_pairs']}")
        print(f"  - Total SFT Instructions    : {metrics['total_sft_examples']}")
        print(f"    * Verified Violations      : {metrics['violations_count']}")
        print(f"    * Verified Compliant Clears: {metrics['clears_count']}")
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
