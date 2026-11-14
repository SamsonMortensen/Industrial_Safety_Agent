"""The benchmark's critic options must work without writing evaluation cases."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from audit_outcomes import parse_audit_response
import self_reflection as reflection
import scaled_benchmark as scaled


class Reply:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return {"message": {"content": "STATUS: CLEAR\nCITATION: NONE\nREASON: Synthetic corrected reply."}}


class CriticRuntimeTests(unittest.TestCase):
    def critic(self, model=None):
        verifier = reflection.StatutoryGroundedVerifier.__new__(reflection.StatutoryGroundedVerifier)
        verifier.valid_sections = {"1910.22"}
        return reflection.SelfReflectionCritic(verifier, model=model) if model else reflection.SelfReflectionCritic(verifier)

    def run_critique(self, critic, **options):
        session = Mock()
        session.post.return_value = Reply()
        parsed = parse_audit_response("STATUS: VIOLATION\nCITATION: 1910.999\nREASON: Synthetic unsupported reply.")
        result = asyncio.run(critic.critique_and_reground_async(
            session, "Synthetic prompt", parsed, {"Log_ID": "fixture-critic"},
            [{"section": "1910.22"}], asyncio.Semaphore(1), **options))
        return session, result

    def test_selected_model_reaches_request_and_recording_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(reflection, "JSON_DIR", Path(folder)):
            session, (result, corrected, _) = self.run_critique(
                self.critic("fixture-selected-model"), record_preference=False)
            self.assertEqual(session.post.call_args.kwargs["json"]["model"], "fixture-selected-model")
            self.assertTrue(corrected)
            self.assertEqual(result["status"], "CLEAR")
            self.assertFalse((Path(folder) / "dpo_pairs_raw.json").exists())

    def test_disabled_recording_preserves_existing_candidate_file(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(reflection, "JSON_DIR", Path(folder)):
            path = Path(folder) / "dpo_pairs_raw.json"
            before = b'[{"log_id":"existing-fixture"}]\n'
            path.write_bytes(before)
            self.run_critique(self.critic(), record_preference=False)
            self.assertEqual(path.read_bytes(), before)

    def test_existing_default_model_and_recording_behavior_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(reflection, "JSON_DIR", Path(folder)):
            session, (_, corrected, _) = self.run_critique(self.critic())
            self.assertEqual(session.post.call_args.kwargs["json"]["model"], reflection.CHAT_MODEL)
            self.assertTrue(corrected)
            self.assertTrue((Path(folder) / "dpo_pairs_raw.json").exists())

    def test_actual_hybrid_cli_completes_without_model_or_preference_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "hybrid.json"
            preference = Path(folder) / "dpo_pairs_raw.json"
            with patch.object(sys, "argv", ["scaled_benchmark", "--strategies", "hybrid", "--limit", "4",
                                             "--model", "fixture-unused-model", "--out", str(output)]), \
                 patch.object(reflection, "JSON_DIR", Path(folder)), \
                 patch("requests.sessions.Session.request", side_effect=AssertionError("HTTP request during policy-only run")) as http, \
                 patch.object(scaled.aiohttp, "ClientSession", side_effect=AssertionError("Async HTTP during policy-only run")) as async_http, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(scaled.main(), 0)
            http.assert_not_called()
            async_http.assert_not_called()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["strategies"]["hybrid"]["results"]), 4)
            self.assertFalse(report["provenance"]["model"]["used"])
            self.assertFalse(report["provenance"]["weight_fine_tuning_performed"])
            self.assertNotIn("CUDA", report["provenance"]["weight_fine_tuning_note"])
            self.assertFalse(preference.exists())


if __name__ == "__main__":
    unittest.main()
