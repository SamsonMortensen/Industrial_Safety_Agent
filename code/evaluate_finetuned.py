"""Evaluate the base model and saved LoRA adapter on the locked holdout."""

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOLDOUT = ROOT / "data" / "scaled_benchmark_holdout.csv"
DEFAULT_ADAPTER = ROOT / "models" / "compliance-auditor-qwen25-3b-lora"
DEFAULT_OUTPUT = ROOT / "json" / "finetune_evaluation_results.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_agent_v2 import compute_metrics, parse_response
from scaled_benchmark import dataset_fingerprint, exact_mcnemar, wilson_interval
from self_reflection import StatutoryGroundedVerifier
from train_lora_dpo import SYSTEM_PROMPT, event_prompt, file_sha256


def portable_path(path: Path) -> str:
    """Prefer a repository-relative artifact path in saved provenance."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The holdout is empty.")
    return rows


def load_quantized_model(model_path: str, adapter_path: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=compute_dtype,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model


def build_chat_text(tokenizer, row: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": event_prompt(row)},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError(
            f"Expected rendered chat text, received {type(rendered).__name__}."
        )
    return rendered


def evaluate_one(
    label: str,
    model_path: str,
    rows: Sequence[Dict[str, Any]],
    verifier: StatutoryGroundedVerifier,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    adapter_path: Path | None = None,
) -> Dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_quantized_model(model_path, adapter_path)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results = []

    for start in range(0, len(rows), batch_size):
        batch_rows = list(rows[start : start + batch_size])
        prompts = [build_chat_text(tokenizer, row) for row in batch_rows]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to("cuda")
        input_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        outputs = tokenizer.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )

        for row, raw_output in zip(batch_rows, outputs):
            parsed = parse_response(raw_output)
            citation = parsed.get("citation")
            actual = int(row["Is_Violation"])
            expected_citation = row["Expected_Citation"]
            parsed.update(
                {
                    "log_id": row["Log_ID"],
                    "audited_by": "model",
                    "raw_output": raw_output,
                    "predicted": (
                        1
                        if parsed.get("status") == "VIOLATION"
                        else 0
                        if parsed.get("status") == "CLEAR"
                        else None
                    ),
                    "actual": actual,
                    "actual_type": row["Violation_Type"],
                    "expected_citation": expected_citation,
                    "expected_section": (
                        expected_citation if expected_citation != "NONE" else None
                    ),
                    "citation_exists": None
                    if citation is None
                    else citation in verifier.valid_sections,
                    "citation_was_retrieved": None,
                    "citation_correct": (
                        citation == expected_citation
                        if expected_citation != "NONE"
                        else citation is None and parsed.get("status") == "CLEAR"
                    ),
                }
            )
            results.append(parsed)

    elapsed = time.perf_counter() - started
    metrics = compute_metrics(results)
    violations = [result for result in results if result["actual"] == 1]
    correct = sum(result["predicted"] == result["actual"] for result in results)
    metrics.update(
        {
            "citation_correctness": round(
                sum(result["citation_correct"] for result in violations)
                / len(violations),
                4,
            ),
            "all_case_verdict_and_citation_accuracy": round(
                sum(
                    result["predicted"] == result["actual"]
                    and result["citation_correct"]
                    for result in results
                )
                / len(results),
                4,
            ),
            "accuracy_95pct_wilson": wilson_interval(correct, len(results)),
            "runtime_seconds": round(elapsed, 3),
            "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        }
    )
    del model
    torch.cuda.empty_cache()
    return {"label": label, "metrics": metrics, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate base and QLoRA models on holdout."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    import bitsandbytes
    import peft
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched QLoRA evaluation.")
    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")

    rows = load_rows(args.holdout)
    verifier = StatutoryGroundedVerifier()
    base = evaluate_one(
        "base",
        args.model,
        rows,
        verifier,
        args.batch_size,
        args.max_input_tokens,
        args.max_new_tokens,
    )
    adapter = evaluate_one(
        "qlora_sft",
        args.model,
        rows,
        verifier,
        args.batch_size,
        args.max_input_tokens,
        args.max_new_tokens,
        adapter_path=args.adapter,
    )
    report = {
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": (
                "Qwen/Qwen2.5-3B-Instruct" if Path(args.model).exists() else args.model
            ),
            "base_model_commit": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "adapter": portable_path(args.adapter),
            "adapter_config_sha256": file_sha256(args.adapter / "adapter_config.json"),
            "holdout": portable_path(args.holdout),
            "holdout_sha256": file_sha256(args.holdout),
            "holdout_fingerprint_sha256": dataset_fingerprint(rows),
            "holdout_cases": len(rows),
            "temperature": 0.0,
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "device": torch.cuda.get_device_name(0),
        },
        "strategies": {"base": base, "qlora_sft": adapter},
        "comparison": exact_mcnemar(base["results"], adapter["results"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for key in ("base", "qlora_sft"):
        metrics = report["strategies"][key]["metrics"]
        print(
            f"{key:12} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"citation_correctness={metrics['citation_correctness']:.4f} "
            f"joint={metrics['all_case_verdict_and_citation_accuracy']:.4f}"
        )
    print(json.dumps(report["comparison"], indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
