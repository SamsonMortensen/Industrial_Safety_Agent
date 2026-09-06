"""Check the base installation before starting model or dataset work."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from ollama_config import ollama_base_url

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = (
    "daily_yard_log.csv",
    "json/regulations.json",
    "json/statutes.json",
    "json/valid_sections.json",
    "data/example_camera_observation.json",
)


def check_local(root: Path = ROOT) -> list[str]:
    errors = []
    for name in ("requests", "numpy", "pandas", "aiohttp"):
        try:
            importlib.import_module(name)
        except ImportError:
            errors.append(
                f"Cannot import {name}; run this Python with "
                "-m pip install -r requirements.txt"
            )
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            errors.append(f"Missing checkout file: {name}")
    return errors


def check_ollama(host: str, *, embed: bool = False) -> list[str]:
    import requests

    response = requests.get(f"{host}/api/tags", timeout=10)
    response.raise_for_status()
    installed = {item["name"] for item in response.json().get("models", [])}
    errors = []
    for model in ("mxbai-embed-large", "qwen3.5:9b"):
        if not {model, model + ":latest"}.intersection(installed):
            errors.append(f"Missing model: run ollama pull {model}")
    if embed and not errors:
        response = requests.post(
            f"{host}/api/embed",
            json={
                "model": "mxbai-embed-large",
                "input": ["A worker observes a yard walkway."],
            },
            timeout=60,
        )
        response.raise_for_status()
        import numpy as np

        vectors = np.asarray(response.json().get("embeddings", []), dtype=float)
        if (
            vectors.ndim != 2
            or vectors.shape[0] != 1
            or not vectors.size
            or not np.isfinite(vectors).all()
        ):
            errors.append("Ollama did not return one finite embedding vector")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Check the server and required model names",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Also send one short embedding request (implies --ollama)",
    )
    args = parser.parse_args()
    errors = check_local()
    host = None
    if (args.ollama or args.embed) and not errors:
        import requests

        try:
            host = ollama_base_url()
            errors.extend(check_ollama(host, embed=args.embed))
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"Invalid Ollama configuration or response: {error}")
        except requests.RequestException as error:
            errors.append(
                "Ollama check failed. Start the Ollama app or ollama serve, "
                "check OLLAMA_HOST, and confirm the models are installed. " + str(error)
            )
    warnings = []
    if os.getenv("OLLAMA_LLM_LIBRARY", "").lower() == "cpu":
        warnings.append(
            "OLLAMA_LLM_LIBRARY=cpu is set. Ollama may run on CPU; "
            "use ollama ps during inference to check. "
            "This is separate from PyTorch CUDA."
        )
    print(
        json.dumps(
            {
                "python": sys.executable,
                "ollama_host": host,
                "ok": not errors,
                "errors": errors,
                "warnings": warnings,
            },
            indent=2,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
