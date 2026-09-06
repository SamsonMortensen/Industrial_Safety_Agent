"""Use the same Ollama HTTP address throughout the local pipeline."""

import os
from urllib.parse import urlsplit


def ollama_base_url(value: str | None = None) -> str:
    raw = (
        os.getenv("OLLAMA_HOST", "http://localhost:11434") if value is None else value
    ).strip()
    if not raw:
        raw = "http://localhost:11434"
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "OLLAMA_HOST must be an HTTP(S) address, " "such as http://127.0.0.1:11434"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "OLLAMA_HOST must not contain credentials, a query, or a fragment"
        )
    # Accessing port also validates malformed and out-of-range port numbers.
    _ = parsed.port
    return raw.rstrip("/")
