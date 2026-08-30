"""Run reproducible QLoRA supervised fine-tuning on trusted training rows.

The historical file name is retained for compatibility. The implemented first
stage is SFT because the existing preference artifacts predate the legal-scope
correction and are not eligible for DPO. The holdout file is never opened here.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_TRAIN = DATA_DIR / "scaled_benchmark_train.csv"
DEFAULT_OUTPUT = ROOT / "models" / "compliance-auditor-qwen25-3b-lora"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_policy import evaluate_event

SYSTEM_PROMPT = """You classify federal safety events for a rail-served industrial yard.
Use only these authorities when the recorded scope and facts support them: 49 U.S.C. 21103,
29 CFR 1910.22, 1910.26, 1910.28, 1910.176, 1910.184, and 1910.333.
An ordinary equipment-operator shift is not automatically covered by railroad hours-of-service law.
Return exactly three lines: STATUS, CITATION, and REASON. Use CITATION: NONE for a clear event."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_prompt(row: Dict[str, Any]) -> str:
    return (
        f"Log ID: {row.get('Log_ID', '')}\n"
        f"Equipment: {row.get('Equipment_ID', '')} ({row.get('Equipment_Type', '')})\n"
        f"Location: {row.get('Location', '')}\n"
        f"Shift duration: {row.get('Operator_Shift_Hours', '')} hours\n"
        f"Incident: {row.get('Reported_Incident', '')}"
    )


def build_sft_records(train_file: Path = DEFAULT_TRAIN) -> List[Dict[str, Any]]:
    """Build SFT messages from the trusted split without reading the holdout."""

    with train_file.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The training split is empty.")

    records = []
    for row in rows:
        decision = evaluate_event(row)
        expected_status = "VIOLATION" if str(row["Is_Violation"]) == "1" else "CLEAR"
        expected_citation = row.get("Expected_Citation", "NONE")
        normalized_citation = decision.citation or "NONE"
        if (
            decision.status != expected_status
            or normalized_citation != expected_citation
        ):
            raise ValueError(
                f"Training row {row.get('Log_ID')} conflicts with the policy: "
                f"expected {expected_status}/{expected_citation}, got "
                f"{decision.status}/{normalized_citation}."
            )
        answer = (
            f"STATUS: {decision.status}\n"
            f"CITATION: {normalized_citation}\n"
            f"REASON: {decision.reason}"
        )
        records.append(
            {
                "log_id": row["Log_ID"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": event_prompt(row)},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    return records


def cuda_preflight(min_free_gib: float) -> Dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Use the repository's CUDA training environment."
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / 2**30
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.2f} GiB of {total_bytes / 2**30:.2f} GiB is free. "
            f"At least {min_free_gib:.2f} GiB is required; stop competing GPU workloads."
        )
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "free_gib": round(free_gib, 3),
        "total_gib": round(total_bytes / 2**30, 3),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def tokenize_records(records, tokenizer, max_length: int):
    from torch.utils.data import Dataset

    def extract_ids(value):
        if hasattr(value, "keys") and "input_ids" in value:
            value = value["input_ids"]
        elif hasattr(value, "ids"):
            value = value.ids
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        return list(value)

    def tokenize(record):
        prompt_messages = record["messages"][:-1]
        full_messages = record["messages"]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_ids = extract_ids(prompt_ids)
        full_ids = extract_ids(full_ids)
        full_ids = full_ids[:max_length]
        prompt_length = min(len(prompt_ids), len(full_ids))
        attention_mask = [1] * len(full_ids)
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        pad_length = max_length - len(full_ids)
        input_ids = full_ids + [tokenizer.pad_token_id] * pad_length
        attention_mask += [0] * pad_length
        labels += [-100] * pad_length
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = [tokenize(record) for record in records]

    class TokenizedRecords(Dataset):
        def __len__(self):
            return len(tokenized)

        def __getitem__(self, index):
            return tokenized[index]

    return TokenizedRecords()


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)

    import accelerate
    import bitsandbytes
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )

    hardware = cuda_preflight(args.min_free_gib)
    records = build_sft_records(args.train_file)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    dataset = tokenize_records(records, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch" if args.max_steps < 0 else "no",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        use_cache=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    started = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - started
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "QLoRA supervised fine-tuning",
        "base_model": args.model,
        "base_model_commit": getattr(model.config, "_commit_hash", None),
        "adapter_path": str(args.output_dir.resolve()),
        "train_file": str(args.train_file.resolve()),
        "train_sha256": file_sha256(args.train_file),
        "train_examples": len(records),
        "holdout_opened_by_trainer": False,
        "seed": args.seed,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "hardware": hardware,
        "runtime_seconds": round(elapsed, 3),
        "train_metrics": train_result.metrics,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT for the compliance classifier."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.gradient_checkpointing = not args.no_gradient_checkpointing

    hardware = cuda_preflight(args.min_free_gib)
    records = build_sft_records(args.train_file)
    print(json.dumps({"hardware": hardware, "train_examples": len(records)}, indent=2))
    if args.preflight_only:
        return 0

    report = run_training(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
