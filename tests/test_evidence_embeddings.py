"""Check optional embeddings without contacting a service or loading a model."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import evidence_embeddings as embeddings

DIGEST = "a" * 64
CHUNKS = [
    {"section": "1910.303", "citation": "29 CFR 1910.303", "text": "Electrical equipment must be secured."},
    {"section": "1910.22", "citation": "29 CFR 1910.22", "text": "Floors must remain clean."},
]


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []
        self.trust_env = True
        self.digest = DIGEST
        self.vector_override = None
        self.response_model = embeddings.MODEL
        self.status = 200
        self.change_digest_after_post = False

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/api/tags"):
            return Response({"models": [{"name": embeddings.MODEL, "digest": self.digest}]}, self.status)
        if not url.endswith("/api/embed") or method != "POST":
            raise AssertionError("Unexpected network operation")
        inputs = kwargs["json"]["input"]
        vectors = [[3.0, 4.0] if "electrical" in text.lower() else [0.0, 5.0] for text in inputs]
        if self.vector_override is not None:
            vectors = self.vector_override(inputs)
        if self.change_digest_after_post:
            self.digest = "b" * 64
        return Response({"model": self.response_model, "embeddings": vectors}, self.status)


class EvidenceEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.patch = patch.object(embeddings.requests, "Session", return_value=self.session)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def client(self, **kwargs):
        return embeddings.OllamaEmbedder(model_digest=DIGEST, **kwargs)

    def index(self, **kwargs):
        return embeddings.FrozenSemanticIndex.build(CHUNKS, self.client(), **kwargs)

    def cache(self):
        index = self.index()
        path = Path(self.temp.name) / "index.npz"
        index.save(path)
        return path

    def rewrite(self, path, edit_manifest=None, edit_vectors=None):
        with np.load(path, allow_pickle=False) as saved:
            manifest = json.loads(str(saved["manifest"]))
            vectors = saved["vectors"].copy()
        if edit_manifest:
            edit_manifest(manifest)
        if edit_vectors:
            edit_vectors(vectors)
        with path.open("wb") as handle:
            np.savez_compressed(handle, vectors=vectors, manifest=np.array(json.dumps(manifest)))

    def test_constructor_is_network_free_and_disables_proxy_environment(self):
        self.client()
        self.assertEqual(self.session.calls, [])
        self.assertFalse(self.session.trust_env)

    def test_only_explicit_loopback_addresses_are_accepted(self):
        for url in ("https://example.com", "http://127.0.0.1.evil.test:11434",
                    "http://example:example@localhost", "http://127.0.0.1/api",
                    "http://localhost?server=remote", "file:///tmp/model", "http://0.0.0.0"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.client(base_url=url)
        for url in ("http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"):
            self.client(base_url=url)
        self.assertEqual(self.session.calls, [])

    def test_config_requires_frozen_model_and_bounded_requests(self):
        invalid = ({"model": "another-model"}, {"batch_size": 0}, {"batch_size": 33},
                   {"timeout": 181}, {"max_embed_requests": 0}, {"max_embed_requests": 4097})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.client(**kwargs)
        with self.assertRaises(ValueError):
            embeddings.OllamaEmbedder(model_digest="latest")

    def test_model_calls_have_correct_prefix_truncation_timeout_and_no_redirects(self):
        client = self.client()
        client.embed(["Electrical equipment"])
        client.embed(["How should electrical equipment be secured?"], query=True)
        posts = [c for c in self.session.calls if c[0] == "POST"]
        self.assertEqual(posts[0][2]["json"]["input"], ["Electrical equipment"])
        self.assertTrue(posts[1][2]["json"]["input"][0].startswith(embeddings.QUERY_PREFIX))
        for _, _, kwargs in self.session.calls:
            self.assertEqual(kwargs["timeout"], (5, 120))
            self.assertFalse(kwargs["allow_redirects"])
        self.assertFalse(posts[0][2]["json"]["truncate"])
        self.assertEqual(client.request_counts, {"embedding_requests": 2, "model_verification_requests": 4})

    def test_no_post_occurs_for_an_absent_or_different_model(self):
        self.session.digest = "b" * 64
        with self.assertRaises(ValueError):
            self.client().embed(["Electrical equipment"])
        self.assertFalse(any(c[0] == "POST" for c in self.session.calls))

    def test_model_change_during_embedding_is_rejected(self):
        self.session.change_digest_after_post = True
        with self.assertRaises(ValueError):
            self.client().embed(["Electrical equipment"])

    def test_different_response_model_and_redirects_are_rejected(self):
        self.session.response_model = "other"
        with self.assertRaises(ValueError):
            self.client().embed(["Electrical equipment"])
        self.session.status = 302
        with self.assertRaisesRegex(ValueError, "HTTP 302"):
            self.client().embed(["Electrical equipment"])

    def test_invalid_vector_values_shapes_and_zero_norms_are_rejected(self):
        for vector in ([], [[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]],
                       [["1", "2"]], [[True, False]], [[1, 0], [0, 1]]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                self.session.vector_override = lambda texts, v=vector: v
                self.client().embed(["Electrical equipment"])

    def test_dimensions_cannot_change_between_batches_or_queries(self):
        client = self.client()
        client.embed(["Electrical equipment"])
        self.session.vector_override = lambda texts: [[1, 2, 3] for _ in texts]
        with self.assertRaises(ValueError):
            client.embed(["Electrical equipment"])

    def test_request_budget_is_checked_before_any_part_of_a_batch(self):
        client = self.client(batch_size=1, max_embed_requests=1)
        with self.assertRaises(ValueError):
            client.embed(["First", "Second"])
        self.assertEqual(self.session.calls, [])
        client.embed(["First"])
        before = len(self.session.calls)
        with self.assertRaises(ValueError):
            client.embed(["Second"])
        self.assertEqual(len(self.session.calls), before)

    def test_passages_are_bounded_verbatim_and_cover_the_whole_chunk(self):
        text = "Electrical equipment must stay secure. " * 110
        chunks = [dict(CHUNKS[0], text=text)]
        passages = embeddings.make_passages(chunks)
        covered = set()
        for passage in passages:
            self.assertLessEqual(len(passage["text"]), 1000)
            self.assertEqual(passage["text"], text[passage["start"]:passage["end"]])
            self.assertEqual(passage["source_text_sha256"], hashlib.sha256(text.encode()).hexdigest())
            covered.update(range(passage["start"], passage["end"]))
        self.assertEqual(len(covered), len(text))

    def test_invalid_passage_configuration_is_rejected(self):
        for kwargs in ({"max_chars": 1001}, {"max_chars": 200}, {"overlap": 250}, {"overlap": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                embeddings.make_passages(CHUNKS, **kwargs)
        with self.assertRaises(ValueError):
            embeddings.make_passages([{"text": " "}])

    def test_scores_are_normalized_cosines_aligned_to_source_chunks(self):
        index = self.index()
        scores = index.query_scores("Electrical equipment")
        self.assertEqual(len(scores), len(CHUNKS))
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertAlmostEqual(scores[1], .8)
        self.assertTrue(np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0))
        self.assertFalse(index.vectors.flags.writeable)

    def test_long_queries_cover_all_text_and_use_normalized_pooling(self):
        index = self.index()
        query = ("Electrical equipment is unsecured. " * 130)[:4000]
        index.query_scores(query)
        post = [c for c in self.session.calls if c[0] == "POST"][-1]
        inputs = post[2]["json"]["input"]
        self.assertGreater(len(inputs), 1)
        windows = list(embeddings._windows(query, 1000, 120))
        self.assertEqual(inputs, [embeddings.QUERY_PREFIX + text for _, _, text in windows])
        covered = {i for start, end, _ in windows for i in range(start, end)}
        self.assertEqual(len(covered), len(query))
        self.assertLessEqual(len(inputs), 11)

    def test_oversized_empty_and_budgeted_queries_fail_without_silent_truncation(self):
        index = self.index(max_queries=1)
        before = len(self.session.calls)
        for text in ("", " ", "x" * 4001):
            with self.subTest(length=len(text)), self.assertRaises(ValueError):
                index.query_scores(text)
        self.assertEqual(len(self.session.calls), before)
        index.query_scores("Electrical equipment")
        with self.assertRaises(ValueError):
            index.query_scores("Floors")

    def test_repeated_queries_are_cached_and_return_independent_lists(self):
        index = self.index()
        scores = index.query_scores("Electrical equipment")
        before = len(self.session.calls)
        scores[0] = -99
        self.assertAlmostEqual(index.query_scores("Electrical equipment")[0], 1.0)
        self.assertEqual(len(self.session.calls), before)

    def test_changed_client_identity_is_rejected_before_cached_or_new_queries(self):
        for field, changed in (("model", "other"), ("model_digest", "b" * 64),
                               ("dimensions", 3)):
            with self.subTest(field=field):
                index = self.index()
                index.query_scores("Electrical equipment")
                before = len(self.session.calls)
                setattr(index.embedder, field, changed)
                for query in ("Electrical equipment", "New query"):
                    with self.assertRaisesRegex(ValueError, "identity changed"):
                        index.query_scores(query)
                self.assertEqual(len(self.session.calls), before)
        index = self.index()
        index.query_scores("Electrical equipment")
        before = len(self.session.calls)
        with patch.object(embeddings, "QUERY_PREFIX", "changed"):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                index.query_scores("Electrical equipment")
        self.assertEqual(len(self.session.calls), before)

    def test_complete_matrix_budget_stops_before_remaining_embedding_batches(self):
        client = self.client(batch_size=1)
        with patch.object(embeddings, "MAX_MATRIX_BYTES", 31):
            with self.assertRaisesRegex(ValueError, "byte budget"):
                client.embed(["First", "Second"])
        self.assertEqual(sum(call[0] == "POST" for call in self.session.calls), 1)

    def test_same_matrix_byte_budget_applies_to_build_save_and_load(self):
        index = self.index()
        path = Path(self.temp.name) / "budget.npz"
        index.save(path)
        rejected = Path(self.temp.name) / "too-large.npz"
        with patch.object(embeddings, "MAX_MATRIX_BYTES", 31):
            with self.assertRaisesRegex(ValueError, "byte budget"):
                self.index()
            with self.assertRaisesRegex(ValueError, "byte budget"):
                index.save(rejected)
            with self.assertRaisesRegex(ValueError, "byte budget"):
                embeddings.FrozenSemanticIndex.load(path, CHUNKS, self.client())
        self.assertFalse(rejected.exists())

    def test_index_checks_ordered_corpus_identity(self):
        index = self.index()
        index.validate_chunks(CHUNKS)
        for chunks in (list(reversed(CHUNKS)), [dict(CHUNKS[0], text="changed"), CHUNKS[1]],
                       [dict(CHUNKS[0], citation="wrong"), CHUNKS[1]]):
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                index.validate_chunks(chunks)

    def test_cache_round_trip_is_network_free_and_keeps_bound_metadata(self):
        path = self.cache()
        before = len(self.session.calls)
        loaded = embeddings.FrozenSemanticIndex.load(path, CHUNKS, self.client())
        self.assertEqual(len(self.session.calls), before)
        self.assertEqual(loaded.profile["model_digest"], DIGEST)
        self.assertEqual(loaded.profile["query_prefix"], embeddings.QUERY_PREFIX)
        self.assertEqual(loaded.profile["normalization"], "l2_float64_v1")
        self.assertAlmostEqual(loaded.query_scores("Electrical equipment")[0], 1.0)

    def test_cache_rejects_same_length_changed_corpus_and_wrong_model_digest(self):
        path = self.cache()
        for chunks in (list(reversed(CHUNKS)), [dict(CHUNKS[0], text="changed"), CHUNKS[1]]):
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                embeddings.FrozenSemanticIndex.load(path, chunks, self.client())
        client = embeddings.OllamaEmbedder(model_digest="b" * 64)
        with self.assertRaises(ValueError):
            embeddings.FrozenSemanticIndex.load(path, CHUNKS, client)

    def test_cache_rejects_stale_schema_prefix_normalization_offsets_or_dimensions(self):
        changes = {"schema": "old", "query_prefix": "wrong", "document_prefix": "wrong",
                   "normalization": "none", "passages_sha256": "0" * 64,
                   "passage_overlap_chars": 0, "dimensions": 3, "truncate": True}
        for key, value in changes.items():
            with self.subTest(key=key):
                path = Path(self.temp.name) / f"{key}.npz"
                self.index().save(path)
                self.rewrite(path, edit_manifest=lambda m, k=key, v=value: m.update({k: v}))
                with self.assertRaises(ValueError):
                    embeddings.FrozenSemanticIndex.load(path, CHUNKS, self.client())

    def test_cache_rejects_nonfinite_nonunit_and_tampered_vectors(self):
        for n, replacement in enumerate(([float("nan"), 1], [0, 2], [1, 0])):
            with self.subTest(replacement=replacement):
                path = Path(self.temp.name) / f"vector{n}.npz"
                self.index().save(path)
                self.rewrite(path, edit_vectors=lambda v, r=replacement: v.__setitem__(0, r))
                with self.assertRaises(ValueError):
                    embeddings.FrozenSemanticIndex.load(path, CHUNKS, self.client())

    def test_cache_refuses_pickle_or_extra_archive_members(self):
        path = Path(self.temp.name) / "unsafe.npz"
        np.savez(path, vectors=np.array([[{}]], dtype=object), manifest=np.array("{}"), extra=np.array(1))
        with self.assertRaises(ValueError):
            embeddings.FrozenSemanticIndex.load(path, CHUNKS, self.client())

    def test_cache_headers_are_checked_before_any_numpy_array_allocation(self):
        path = self.cache()
        with zipfile.ZipFile(path) as archive:
            original = {name: archive.read(name) for name in archive.namelist()}
        headers = [
            ("vectors.npy", {"shape": (1000000000, 2), "descr": "<f8", "fortran_order": False}),
            ("vectors.npy", {"shape": (2, 2), "descr": "|O", "fortran_order": False}),
            ("manifest.npy", {"shape": (), "descr": "<U100000000", "fortran_order": False}),
            ("manifest.npy", {"shape": (1000000000,), "descr": "<U1", "fortran_order": False}),
        ]
        for name, header in headers:
            with self.subTest(name=name, header=header):
                buffer = io.BytesIO()
                np.lib.format.write_array_header_1_0(buffer, header)
                with zipfile.ZipFile(path, "w") as archive:
                    for member, data in original.items():
                        archive.writestr(member, buffer.getvalue() if member == name else data)
                # The old np.load path would call this allocating reader before
                # checking the supplied shape. Keep the regression safe there too.
                with patch.object(np.lib.format, "read_array", side_effect=AssertionError("unvalidated allocation")):
                    with self.assertRaises(ValueError):
                        embeddings.FrozenSemanticIndex.load(path, CHUNKS, self.client())

    def test_cache_requires_exact_array_payload_lengths(self):
        path = self.cache()
        with zipfile.ZipFile(path) as archive:
            original = {name: archive.read(name) for name in archive.namelist()}
        for member in ("vectors.npy", "manifest.npy"):
            for change in (lambda data: data[:-1], lambda data: data + b"extra"):
                with self.subTest(member=member, change=change):
                    with zipfile.ZipFile(path, "w") as archive:
                        for name, data in original.items():
                            archive.writestr(name, change(data) if name == member else data)
                    with self.assertRaisesRegex(ValueError, "payload length"):
                        embeddings.FrozenSemanticIndex.load(path, CHUNKS, self.client())

    def test_cache_rejects_oversized_manifest_header(self):
        path = self.cache()
        with zipfile.ZipFile(path) as archive:
            original = {name: archive.read(name) for name in archive.namelist()}
        header = {"shape": (), "descr": "<U1", "fortran_order": False}
        buffer = io.BytesIO()
        np.lib.format.write_array_header_1_0(buffer, header)
        encoded = buffer.getvalue()
        header_size = int.from_bytes(encoded[8:10], "little")
        padding = b" " * (embeddings.MAX_NPY_HEADER_BYTES + 1)
        oversized = encoded[:8] + (header_size + len(padding)).to_bytes(2, "little") + padding + encoded[10:]
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in original.items():
                archive.writestr(name, oversized if name == "manifest.npy" else data)
        with patch.object(np.lib.format, "read_array_header_1_0", side_effect=AssertionError("oversized header read")):
            with self.assertRaises(ValueError):
                embeddings.FrozenSemanticIndex.load(path, CHUNKS, self.client())

    def test_explicit_artifact_save_never_overwrites_or_writes_into_repository(self):
        index = self.index()
        path = Path(self.temp.name) / "index.npz"
        index.save(path)
        with self.assertRaises(FileExistsError):
            index.save(path)
        with self.assertRaises(ValueError):
            index.save(ROOT / "must-not-write.npz")
        self.assertFalse((ROOT / "must-not-write.npz").exists())
        with self.assertRaises(ValueError):
            index.save(Path(self.temp.name) / "index.json")


if __name__ == "__main__":
    unittest.main()
