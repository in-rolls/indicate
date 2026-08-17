"""Tests for batch-mode LLM transliteration (indicate.batch).

All provider calls are mocked with a stateful fake of LiteLLM's batch API, so these
run with no network access and no API spend.

The batching tests pass ``engine=("llm",)`` on purpose. Their fixtures are real
Punjabi words, so with the table on nothing would ever be submitted and the tests
would pass while exercising none of the batch machinery they exist to cover.
:class:`TestLookupSeeding` covers the table path itself.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import indicate.batch as batch_mod
from indicate.batch import (
    collect_transliteration_batches,
    submit_transliteration_batches,
    transliterate_tokens_batched,
)

_NUMBERED = re.compile(r"^\s*\d+\.\s+(.+)$", re.MULTILINE)


def _tokens_in_request(request: dict) -> list[str]:
    user = request["body"]["messages"][-1]["content"]
    return _NUMBERED.findall(user)


class FakeBatchAPI:
    """Minimal stand-in for the litellm batch functions used by indicate.batch.

    Echoes ``xlit-<token>`` for each token. Set ``mismatch_multi=True`` to make any
    group with >1 token emit one too few output lines (exercises the requeue path).
    Set ``pending_first=True`` to report the first ``retrieve_batch`` as in-progress.
    """

    def __init__(self, *, mismatch_multi: bool = False):
        self.mismatch_multi = mismatch_multi
        self.pending_first = False
        self._n = 0
        self.input_files: dict[str, list[dict]] = {}
        self.batches: dict[str, str] = {}  # batch_id -> input_file_id
        self.retrieve_calls: list[str] = []

    def create_file(self, *, file, purpose, custom_llm_provider):
        self._n += 1
        file_id = f"file-{self._n}"
        lines = file.read().decode("utf-8").splitlines()
        self.input_files[file_id] = [json.loads(line) for line in lines if line.strip()]
        return SimpleNamespace(id=file_id)

    def create_batch(
        self, *, completion_window, endpoint, input_file_id, custom_llm_provider
    ):
        self._n += 1
        batch_id = f"batch-{self._n}"
        self.batches[batch_id] = input_file_id
        return SimpleNamespace(id=batch_id)

    def retrieve_batch(self, *, batch_id, custom_llm_provider):
        self.retrieve_calls.append(batch_id)
        if self.pending_first and len(self.retrieve_calls) == 1:
            return SimpleNamespace(status="in_progress", output_file_id=None)
        return SimpleNamespace(status="completed", output_file_id=f"out-{batch_id}")

    def file_content(self, *, file_id, custom_llm_provider):
        batch_id = file_id[len("out-") :]
        requests = self.input_files[self.batches[batch_id]]
        out_lines = []
        for request in requests:
            tokens = _tokens_in_request(request)
            emit = tokens
            if self.mismatch_multi and len(tokens) > 1:
                emit = tokens[:-1]  # one too few -> count mismatch
            body_text = "\n".join(f"{i + 1}. xlit-{tok}" for i, tok in enumerate(emit))
            out_lines.append(
                json.dumps(
                    {
                        "custom_id": request["custom_id"],
                        "response": {
                            "status_code": 200,
                            "body": {"choices": [{"message": {"content": body_text}}]},
                        },
                        "error": None,
                    }
                )
            )
        return "\n".join(out_lines)


class BatchTestBase(unittest.TestCase):
    def setUp(self):
        os.environ["OPENAI_API_KEY"] = "test-key"
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt = Path(self._tmp.name) / "tokens.jsonl"

    def tearDown(self):
        os.environ.pop("OPENAI_API_KEY", None)
        self._tmp.cleanup()


class TestSubmit(BatchTestBase):
    def test_submit_writes_state_and_groups(self):
        fake = FakeBatchAPI()
        with patch.object(batch_mod, "litellm", fake):
            state = submit_transliteration_batches(
                ["ਰਾਜ", "ਪੰਜਾਬ", "ਸਿੰਘ"],
                "punjabi",
                "english",
                checkpoint_path=self.ckpt,
                provider="openai",
                group_size=10,
                use_few_shot=False,
                engine=("llm",),
            )
        # One batch, one group (group_size >= n), all three tokens mapped.
        self.assertTrue(batch_mod._state_path(self.ckpt).exists())
        self.assertEqual(len(state.jobs), 1)
        (group,) = state.jobs[0].custom_id_to_tokens.values()
        self.assertEqual(group, ["ਰਾਜ", "ਪੰਜਾਬ", "ਸਿੰਘ"])
        # The submitted JSONL request carries those tokens.
        (file_requests,) = fake.input_files.values()
        self.assertEqual(_tokens_in_request(file_requests[0]), ["ਰਾਜ", "ਪੰਜਾਬ", "ਸਿੰਘ"])
        body = file_requests[0]["body"]
        self.assertIn("max_completion_tokens", body)
        self.assertNotIn("max_tokens", body)

    def test_submit_skips_already_resolved(self):
        # Pre-populate the resolved checkpoint with one token.
        self.ckpt.write_text(
            json.dumps({"token": "ਰਾਜ", "translit": "raj"}) + "\n", encoding="utf-8"
        )
        fake = FakeBatchAPI()
        with patch.object(batch_mod, "litellm", fake):
            state = submit_transliteration_batches(
                ["ਰਾਜ", "ਪੰਜਾਬ"],
                "punjabi",
                "english",
                checkpoint_path=self.ckpt,
                provider="openai",
                group_size=10,
                use_few_shot=False,
                engine=("llm",),
            )
        (group,) = state.jobs[0].custom_id_to_tokens.values()
        self.assertEqual(group, ["ਪੰਜਾਬ"])  # resolved token dropped


@pytest.mark.needs_lookup
class TestLookupSeeding(BatchTestBase):
    """The table answers before anything is submitted."""

    def test_known_tokens_are_never_submitted(self):
        fake = FakeBatchAPI()
        with patch.object(batch_mod, "litellm", fake):
            state = submit_transliteration_batches(
                ["ਸਿੰਘ", "ਕੌਰ", "ZZZQQ"],
                "punjabi",
                "english",
                checkpoint_path=self.ckpt,
                provider="openai",
                group_size=10,
                use_few_shot=False,
            )
        if not batch_mod._resolve_locally(
            ["ਸਿੰਘ"], "punjabi", "english", ("lookup", "llm")
        ):
            self.skipTest("punjabi lookup table not built")
        (group,) = state.jobs[0].custom_id_to_tokens.values()
        self.assertEqual(group, ["ZZZQQ"])

    def test_table_answers_land_in_the_checkpoint(self):
        # They must be written, not merely skipped, or the caller loses them.
        fake = FakeBatchAPI()
        with patch.object(batch_mod, "litellm", fake):
            submit_transliteration_batches(
                ["ਸਿੰਘ", "ZZZQQ"],
                "punjabi",
                "english",
                checkpoint_path=self.ckpt,
                provider="openai",
                use_few_shot=False,
            )
        resolved = batch_mod._load_resolved(self.ckpt)
        if not resolved:
            self.skipTest("punjabi lookup table not built")
        self.assertEqual(resolved, {"ਸਿੰਘ": "singh"})

    def test_a_language_with_no_table_reaches_no_network(self):
        # Tamil ships no table; asking for one must not attempt a download.
        with patch("indicate.resources.resolve_data") as resolve:
            self.assertEqual(
                batch_mod._resolve_locally(
                    ["வணக்கம்"], "tamil", "english", ("lookup", "llm")
                ),
                {},
            )
        resolve.assert_not_called()

    def test_a_non_english_target_uses_no_table(self):
        self.assertEqual(
            batch_mod._resolve_locally(["ਸਿੰਘ"], "punjabi", "tamil", ("lookup", "llm")),
            {},
        )


class TestCollect(BatchTestBase):
    def test_collect_pending_then_completed(self):
        fake = FakeBatchAPI()
        fake.pending_first = True
        with patch.object(batch_mod, "litellm", fake):
            submit_transliteration_batches(
                ["ਰਾਜ", "ਪੰਜਾਬ"],
                "punjabi",
                "english",
                checkpoint_path=self.ckpt,
                provider="openai",
                group_size=10,
                use_few_shot=False,
                engine=("llm",),
            )
            done, resolved = collect_transliteration_batches(self.ckpt)
            self.assertFalse(done)
            self.assertEqual(resolved, {})

            done, resolved = collect_transliteration_batches(self.ckpt)
            self.assertTrue(done)
            self.assertEqual(resolved, {"ਰਾਜ": "xlit-ਰਾਜ", "ਪੰਜਾਬ": "xlit-ਪੰਜਾਬ"})
        # Resolved pairs are durably written to the JSONL checkpoint.
        self.assertEqual(
            batch_mod._load_resolved(self.ckpt),
            {"ਰਾਜ": "xlit-ਰਾਜ", "ਪੰਜਾਬ": "xlit-ਪੰਜਾਬ"},
        )


class TestDriver(BatchTestBase):
    def test_driver_end_to_end(self):
        fake = FakeBatchAPI()
        with patch.object(batch_mod, "litellm", fake):
            resolved = transliterate_tokens_batched(
                ["ਰਾਜ", "ਪੰਜਾਬ", "ਸਿੰਘ"],
                "punjabi",
                "english",
                checkpoint_path=self.ckpt,
                provider="openai",
                group_size=10,
                use_few_shot=False,
                engine=("llm",),
                poll_interval=0,
            )
        self.assertEqual(
            resolved,
            {"ਰਾਜ": "xlit-ਰਾਜ", "ਪੰਜਾਬ": "xlit-ਪੰਜਾਬ", "ਸਿੰਘ": "xlit-ਸਿੰਘ"},
        )
        # State file cleaned up once everything resolved.
        self.assertFalse(batch_mod._state_path(self.ckpt).exists())

    def test_driver_requeues_count_mismatch(self):
        # mismatch_multi drops a line for the initial >1-token group, forcing the
        # driver to requeue those tokens one-per-request (which then resolve).
        fake = FakeBatchAPI(mismatch_multi=True)
        with patch.object(batch_mod, "litellm", fake):
            resolved = transliterate_tokens_batched(
                ["ਰਾਜ", "ਪੰਜਾਬ"],
                "punjabi",
                "english",
                checkpoint_path=self.ckpt,
                provider="openai",
                group_size=2,
                use_few_shot=False,
                engine=("llm",),
                poll_interval=0,
            )
        self.assertEqual(resolved, {"ਰਾਜ": "xlit-ਰਾਜ", "ਪੰਜਾਬ": "xlit-ਪੰਜਾਬ"})
        # >1 batch created: the initial group plus per-token requeues.
        self.assertGreater(len(fake.batches), 1)

    def test_driver_resumes_existing_batch(self):
        fake = FakeBatchAPI()
        with patch.object(batch_mod, "litellm", fake):
            # First, only submit (leave a batch in flight, no collection).
            submit_transliteration_batches(
                ["ਰਾਜ", "ਪੰਜਾਬ"],
                "punjabi",
                "english",
                checkpoint_path=self.ckpt,
                provider="openai",
                group_size=10,
                use_few_shot=False,
                engine=("llm",),
            )
            batches_after_submit = len(fake.batches)
            # Driver should resume the in-flight batch, not submit a new one.
            resolved = transliterate_tokens_batched(
                ["ਰਾਜ", "ਪੰਜਾਬ"],
                "punjabi",
                "english",
                checkpoint_path=self.ckpt,
                provider="openai",
                group_size=10,
                use_few_shot=False,
                engine=("llm",),
                poll_interval=0,
            )
        self.assertEqual(len(fake.batches), batches_after_submit)  # no new submit
        self.assertEqual(resolved, {"ਰਾਜ": "xlit-ਰਾਜ", "ਪੰਜਾਬ": "xlit-ਪੰਜਾਬ"})


if __name__ == "__main__":
    unittest.main()
