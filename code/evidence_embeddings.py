"""Optional local passage embeddings with a frozen corpus and model identity.

The default showcase does not import a model or send a request. Call this module
explicitly to build or use an index. Similarity finds evidence to inspect; it is
not a probability or a legal decision.

API: https://docs.ollama.com/api/embed (truncate=False rejects oversized input).
Query prefix: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence
from urllib.parse import urlsplit
import zipfile

import numpy as np
import requests

SCHEMA = "frozen_evidence_passages_v1"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL = "mxbai-embed-large:latest"
MAX_QUERY_CHARS = 4000
MAX_PASSAGES = 20000
MAX_INDEX_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_NPY_HEADER_BYTES = 10000
# Leave room for the manifest, NPY/ZIP headers and compression overhead.
MAX_MATRIX_BYTES = MAX_INDEX_BYTES - 1024 * 1024
MAX_DIMENSIONS = 8192
REPO_ROOT = Path(__file__).resolve().parents[1]


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def corpus_fingerprint(chunks: Sequence[Mapping]) -> str:
    """Bind source order, metadata and exact text, not just the number of rows."""
    return _hash([dict(chunk) for chunk in chunks])


def _windows(text: str, max_chars: int, overlap: int):
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Prefer a complete word, while guaranteeing progress and full coverage.
        if end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        yield start, end, text[start:end]
        if end == len(text):
            break
        start = end - overlap


def make_passages(chunks: Sequence[Mapping], *, max_chars: int = 1000,
                  overlap: int = 120) -> list[dict]:
    """Keep bounded verbatim slices and offsets into each original chunk."""
    if type(max_chars) is not int or not 256 <= max_chars <= 1000:
        raise ValueError("Passages must use between 256 and 1000 characters")
    if type(overlap) is not int or not 0 <= overlap < max_chars // 4:
        raise ValueError("Passage overlap must be smaller than one quarter of the window")
    if not 1 <= len(chunks) <= 10000:
        raise ValueError("Provide between one and 10,000 source chunks")
    passages = []
    for index, chunk in enumerate(chunks):
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every source chunk needs non-empty text")
        source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        for start, end, excerpt in _windows(text, max_chars, overlap):
            passages.append({"chunk_index": index, "start": start, "end": end,
                             "text": excerpt, "source_text_sha256": source_sha256})
            if len(passages) > MAX_PASSAGES:
                raise ValueError("The passage limit was exceeded")
    return passages


def _vectors(value, rows: int, dimensions: int | None = None) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu":
            raise ValueError("Embedding values must be real numbers")
    except (TypeError, OverflowError) as exc:
        raise ValueError("Invalid embedding matrix") from exc
    if (raw.ndim != 2 or raw.shape[0] != rows or
            not 1 <= raw.shape[1] <= MAX_DIMENSIONS or
            (dimensions is not None and raw.shape[1] != dimensions)):
        raise ValueError("Embedding row count or dimensions do not match")
    _check_matrix_budget(rows, raw.shape[1])
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain non-finite values")
    norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Embeddings must have finite non-zero norms")
    return array


def _check_matrix_budget(rows: int, dimensions: int) -> None:
    if rows * dimensions * np.dtype(np.float64).itemsize > MAX_MATRIX_BYTES:
        raise ValueError("The embedding matrix exceeds the index byte budget")


def _read_cached_array(archive, name: str, *, rows=None, dimensions=None):
    """Validate a member's declared allocation before reading its array data."""
    info = archive.getinfo(name)
    with archive.open(info) as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            reader = np.lib.format.read_array_header_1_0
            length_bytes = 2
        elif version == (2, 0):
            reader = np.lib.format.read_array_header_2_0
            length_bytes = 4
        else:
            raise ValueError("The semantic index uses an unsupported array header")
        header_position = member.tell()
        encoded_length = member.read(length_bytes)
        if (len(encoded_length) != length_bytes or
                not 0 < int.from_bytes(encoded_length, "little") <= MAX_NPY_HEADER_BYTES):
            raise ValueError("The semantic index array header is invalid or oversized")
        member.seek(header_position)
        shape, fortran_order, dtype = reader(member, max_header_size=MAX_NPY_HEADER_BYTES)
        if name == "manifest.npy":
            if shape != () or dtype.kind != "U" or not 0 < dtype.itemsize <= MAX_MANIFEST_BYTES:
                raise ValueError("The semantic index manifest is invalid or oversized")
            count = 1
        else:
            if (len(shape) != 2 or shape != (rows, dimensions) or
                    dtype.kind not in "fiu" or not 1 <= dtype.itemsize <= 8):
                raise ValueError("Cached embedding shape or dtype does not match")
            _check_matrix_budget(rows, dimensions)
            count = rows * dimensions
        data_bytes = count * dtype.itemsize
        if member.tell() + data_bytes != info.file_size:
            raise ValueError("Cached array header does not match its payload length")
        data = member.read(data_bytes + 1)
        if len(data) != data_bytes:
            raise ValueError("Cached array data is incomplete or has trailing bytes")
    # No untrusted shape reaches NumPy's allocating NPY reader.
    return np.frombuffer(data, dtype=dtype).reshape(shape, order="F" if fortran_order else "C")


def _normalize(value, rows: int, dimensions: int | None = None) -> np.ndarray:
    array = _vectors(value, rows, dimensions)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


class OllamaEmbedder:
    """A bounded, explicit loopback client. It cannot pull or choose a model."""

    def __init__(self, *, model_digest: str, base_url: str = "http://127.0.0.1:11434",
                 model: str = MODEL, batch_size: int = 16, timeout: int = 120,
                 max_embed_requests: int = 256):
        parsed = urlsplit(base_url)
        host = parsed.hostname
        try:
            loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            loopback = False
        if (parsed.scheme not in {"http", "https"} or not loopback or
                parsed.username or parsed.password or parsed.query or parsed.fragment or
                parsed.path not in {"", "/"}):
            raise ValueError("Embeddings require an explicit loopback HTTP(S) address")
        _ = parsed.port
        if model not in {MODEL, "mxbai-embed-large"}:
            raise ValueError("This version uses only the configured mxbai-embed-large model")
        digest = model_digest.removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Provide the frozen model SHA-256 digest")
        for value, lower, upper, name in ((batch_size, 1, 32, "batch size"),
                                           (timeout, 1, 180, "timeout"),
                                           (max_embed_requests, 1, 4096, "request budget")):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"Invalid embedding {name}")
        # Do not let proxy settings send local observations to a remote service.
        self.session = requests.Session()
        self.session.trust_env = False
        self.base_url = base_url.rstrip("/")
        self.model = MODEL
        self.model_digest = digest
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_embed_requests = max_embed_requests
        self.embedding_requests = 0
        self.verification_requests = 0
        self.dimensions = None

    def _response(self, method: str, path: str, **kwargs) -> dict:
        response = self.session.request(method, self.base_url + path,
                                        timeout=(5, self.timeout), allow_redirects=False,
                                        **kwargs)
        if response.status_code != 200:
            # Server error bodies can contain user text. Keep the error bounded.
            raise ValueError(f"Local embedding service returned HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("The embedding service returned an invalid response")
        return payload

    def verify_model(self) -> None:
        self.verification_requests += 1
        payload = self._response("GET", "/api/tags")
        models = payload.get("models")
        if not isinstance(models, list):
            raise ValueError("The local model inventory is invalid")
        matching = [m for m in models if isinstance(m, dict) and
                    m.get("name", m.get("model")) == self.model]
        if len(matching) != 1 or matching[0].get("digest", "").removeprefix("sha256:") != self.model_digest:
            raise ValueError("The installed embedding model does not match the frozen digest")

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        if not texts or len(texts) > MAX_PASSAGES:
            raise ValueError("The embedding input count is outside the bounded budget")
        if any(not isinstance(text, str) or not text.strip() or len(text) > 1000 for text in texts):
            raise ValueError("Each embedding input needs 1 to 1000 characters")
        needed = math.ceil(len(texts) / self.batch_size)
        if self.embedding_requests + needed > self.max_embed_requests:
            raise ValueError("The embedding request budget has been used")
        self.verify_model()
        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = [(QUERY_PREFIX + text) if query else text for text in batch]
            self.embedding_requests += 1
            payload = self._response("POST", "/api/embed", json={
                "model": self.model, "input": inputs, "truncate": False})
            if payload.get("model") not in {self.model, "mxbai-embed-large"}:
                raise ValueError("The embedding response names a different model")
            vectors = _normalize(payload.get("embeddings"), len(batch), self.dimensions)
            if self.dimensions is None:
                self.dimensions = vectors.shape[1]
            # Check the complete operation before requesting the remaining batches.
            _check_matrix_budget(len(texts), self.dimensions)
            batches.append(vectors)
        self.verify_model()
        return np.vstack(batches)

    @property
    def request_counts(self) -> dict:
        return {"embedding_requests": self.embedding_requests,
                "model_verification_requests": self.verification_requests}


def _binding(chunks, embedder, passages, max_chars, overlap):
    return {"schema": SCHEMA, "corpus_sha256": corpus_fingerprint(chunks),
            "chunk_count": len(chunks), "passage_count": len(passages),
            "passages_sha256": _hash(passages), "model": embedder.model,
            "model_digest": embedder.model_digest, "query_prefix": QUERY_PREFIX,
            "document_prefix": "", "normalization": "l2_float64_v1",
            "passage_window_chars": max_chars, "passage_overlap_chars": overlap,
            "query_window_chars": 1000, "query_overlap_chars": 120,
            "query_pooling": "mean_normalized_windows_then_l2_v1",
            "chunk_pooling": "maximum_passage_cosine_v1", "truncate": False}


class FrozenSemanticIndex:
    """Optional query scores aligned to the original corpus; never a verdict."""

    def __init__(self, chunks, embedder, passages, vectors, *, max_chars=1000,
                 overlap=120, max_queries=512):
        if type(max_queries) is not int or not 1 <= max_queries <= 4096:
            raise ValueError("Query budget must be between one and 4096")
        expected = make_passages(chunks, max_chars=max_chars, overlap=overlap)
        if passages != expected:
            raise ValueError("Passage text or offsets do not match the source corpus")
        self.vectors = _normalize(vectors, len(passages), embedder.dimensions)
        self.vectors.setflags(write=False)
        embedder.dimensions = self.vectors.shape[1]
        self._binding = _binding(chunks, embedder, passages, max_chars, overlap)
        self.corpus_sha256 = self._binding["corpus_sha256"]
        self.passages = tuple(dict(p) for p in passages)
        self.embedder = embedder
        self.max_queries = max_queries
        self._queries = {}

    @classmethod
    def build(cls, chunks, embedder: OllamaEmbedder, *, max_chars=1000,
              overlap=120, max_queries=512):
        passages = make_passages(chunks, max_chars=max_chars, overlap=overlap)
        vectors = embedder.embed([p["text"] for p in passages])
        return cls(chunks, embedder, passages, vectors, max_chars=max_chars,
                   overlap=overlap, max_queries=max_queries)

    def validate_chunks(self, chunks) -> None:
        if corpus_fingerprint(chunks) != self.corpus_sha256:
            raise ValueError("The semantic index belongs to a different ordered corpus")

    @property
    def profile(self) -> dict:
        return dict(self._binding, dimensions=self.vectors.shape[1],
                    query_count=len(self._queries), **self.embedder.request_counts)

    def query_scores(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            raise ValueError("Semantic queries need 1 to 4,000 characters")
        if (self.embedder.model != self._binding["model"] or
                self.embedder.model_digest != self._binding["model_digest"] or
                self.embedder.dimensions != self.vectors.shape[1] or
                QUERY_PREFIX != self._binding["query_prefix"]):
            raise ValueError("The embedding client identity changed after the index was frozen")
        if query in self._queries:
            return list(self._queries[query])
        if len(self._queries) >= self.max_queries:
            raise ValueError("The semantic query budget has been used")
        windows = [text for _, _, text in _windows(query, 1000, 120)]
        vectors = self.embedder.embed(windows, query=True)
        pooled = _normalize(vectors.mean(axis=0, keepdims=True), 1, self.vectors.shape[1])[0]
        scores = np.clip(self.vectors @ pooled, -1.0, 1.0)
        combined = np.full(self._binding["chunk_count"], -1.0)
        for passage, score in zip(self.passages, scores):
            index = passage["chunk_index"]
            combined[index] = max(combined[index], score)
        result = combined.tolist()
        self._queries[query] = result
        return list(result)

    def save(self, path: Path | str) -> Path:
        """Write a new, explicit artifact outside the source checkout."""
        path = Path(path).resolve()
        if path == REPO_ROOT or REPO_ROOT in path.parents:
            raise ValueError("Save semantic indexes outside the repository")
        if path.suffix != ".npz":
            raise ValueError("A semantic index needs an explicit .npz output path")
        _check_matrix_budget(*self.vectors.shape)
        manifest = dict(self._binding, dimensions=self.vectors.shape[1],
                        vectors_sha256=hashlib.sha256(self.vectors.astype("<f8").tobytes()).hexdigest())
        # Exclusive creation prevents an experiment from replacing earlier evidence.
        with path.open("xb") as handle:
            np.savez_compressed(handle, vectors=self.vectors, manifest=np.array(_json(manifest)))
        return path

    @classmethod
    def load(cls, path: Path | str, chunks, embedder: OllamaEmbedder, *, max_chars=1000,
             overlap=120, max_queries=512):
        path = Path(path)
        if path.stat().st_size > MAX_INDEX_BYTES:
            raise ValueError("The semantic index exceeds the file budget")
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if ([entry.filename for entry in entries] != ["vectors.npy", "manifest.npy"] or
                    sum(entry.file_size for entry in entries) > MAX_INDEX_BYTES):
                raise ValueError("The semantic index has unexpected or oversized entries")
            passages = make_passages(chunks, max_chars=max_chars, overlap=overlap)
            expected = _binding(chunks, embedder, passages, max_chars, overlap)
            raw_manifest = _read_cached_array(archive, "manifest.npy")
            manifest = json.loads(str(raw_manifest))
            if not isinstance(manifest, dict) or set(manifest) != set(expected) | {"dimensions", "vectors_sha256"}:
                raise ValueError("The semantic index manifest has unexpected fields")
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError("The semantic index is stale or belongs to different inputs")
            dimensions = manifest["dimensions"]
            if type(dimensions) is not int or not 1 <= dimensions <= MAX_DIMENSIONS:
                raise ValueError("The semantic index dimensions are invalid")
            vectors = _vectors(_read_cached_array(archive, "vectors.npy",
                                                  rows=len(passages), dimensions=dimensions),
                               len(passages), dimensions)
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=0, atol=1e-10):
                raise ValueError("Cached embeddings are not normalized")
            digest = hashlib.sha256(vectors.astype("<f8").tobytes()).hexdigest()
            if digest != manifest["vectors_sha256"]:
                raise ValueError("Cached embedding content does not match its manifest")
        return cls(chunks, embedder, passages, vectors, max_chars=max_chars,
                   overlap=overlap, max_queries=max_queries)
