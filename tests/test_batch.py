"""Tests for batch-mode LLM transliteration (indicate.batch).

Provider HTTP calls are mocked while the real batchlane adapter and journal
run end to end, with no API spend.

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
from unittest.mock import patch

import httpx
import pytest
import respx

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
    """Stateful fake of OpenAI HTTP endpoints used by the real batchlane adapter.

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

    def __enter__(self):
        self.router = respx.mock(assert_all_called=False)
        self.router.__enter__()
        api_root = "https://" + "api.openai.com/v1"
        self.router.post(f"{api_root}/files").mock(side_effect=self.create_file)
        self.router.post(f"{api_root}/batches").mock(side_effect=self.create_batch)
        self.router.get(url__regex=rf"{api_root}/batches/batch-\d+$").mock(
            side_effect=self.retrieve_batch
        )
        self.router.get(url__regex=rf"{api_root}/files/out-batch-\d+/content$").mock(
            side_effect=lambda request: httpx.Response(
                200, text=self.file_content(file_id=request.url.path.split("/")[-2])
            )
        )
        return self

    def __exit__(self, *args):
        return self.router.__exit__(*args)

    def create_file(self, request):
        self._n += 1
        file_id = f"file-{self._n}"
        lines = request.content.decode("utf-8").splitlines()
        self.input_files[file_id] = [
            json.loads(line) for line in lines if line.startswith("{")
        ]
        return httpx.Response(200, json={"id": file_id})

    def create_batch(self, request):
        self._n += 1
        batch_id = f"batch-{self._n}"
        self.batches[batch_id] = json.loads(request.content)["input_file_id"]
        return httpx.Response(200, json={"id": batch_id})

    def retrieve_batch(self, request):
        batch_id = request.url.path.split("/")[-1]
        self.retrieve_calls.append(batch_id)
        if self.pending_first and len(self.retrieve_calls) == 1:
            return httpx.Response(200, json={"status": "in_progress"})
        return httpx.Response(
            200, json={"status": "completed", "output_file_id": f"out-{batch_id}"}
        )

    def file_content(self, *, file_id):
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
        with fake:
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
        with fake:
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
        with fake:
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
        with fake:
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
        with fake:
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
        with fake:
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
        with fake:
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
        with fake:
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


def test_resume_completes_interrupted_submission_without_regenerating_prompts(
    tmp_path, monkeypatch
):
    from batchlane.adapters.base import KEY_FIELD

    from indicate.llm_indic import IndicLLMTransliterator

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    examples = []

    def generate(self):
        examples.append(True)
        return [{"source": "ਰਾਜ", "target": "raj"}]

    monkeypatch.setattr(IndicLLMTransliterator, "generate_few_shot_examples", generate)
    checkpoint = tmp_path / "run.jsonl"
    fake = FakeBatchAPI()
    with fake:
        accepted = []
        attempts = []

        def submit(request):
            attempts.append(True)
            if len(attempts) == 2:
                raise RuntimeError("process stopped")
            reply = fake.create_batch(request)
            accepted.append(
                {
                    "id": reply.json()["id"],
                    "metadata": json.loads(request.content)["metadata"],
                }
            )
            return reply

        fake.router.post("https://api.openai.com/v1/batches").mock(side_effect=submit)
        fake.router.get("https://api.openai.com/v1/batches").mock(
            side_effect=lambda request: httpx.Response(200, json={"data": accepted})
        )
        with pytest.raises(RuntimeError, match="process stopped"):
            submit_transliteration_batches(
                ["ਰਾਜ", "ਪੰਜਾਬ"],
                "punjabi",
                "english",
                checkpoint_path=checkpoint,
                provider="openai",
                group_size=1,
                max_requests_per_batch=1,
                engine=("llm",),
            )
        assert len(fake.batches) == 1
        assert accepted[0]["metadata"][KEY_FIELD]
        done, pairs = collect_transliteration_batches(checkpoint)
        assert done
        assert pairs == {"ਰਾਜ": "xlit-ਰਾਜ", "ਪੰਜਾਬ": "xlit-ਪੰਜਾਬ"}
        assert len(fake.batches) == 2
        assert len(examples) == 1


def test_explicit_key_survives_submission_and_collection_without_environment(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake = FakeBatchAPI()
    with fake:
        result = transliterate_tokens_batched(
            ["ਰਾਜ"],
            "punjabi",
            "english",
            checkpoint_path=tmp_path / "run.jsonl",
            provider="openai",
            api_key="explicit-key",
            use_few_shot=False,
            engine=("llm",),
            poll_interval=0,
        )
        assert result == {"ਰਾਜ": "xlit-ਰਾਜ"}
        assert all(
            call.request.headers["authorization"] == "Bearer explicit-key"
            for call in fake.router.calls
        )


def test_resume_rejects_changed_language_before_provider_io(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    checkpoint = tmp_path / "run.jsonl"
    with FakeBatchAPI() as fake:
        submit_transliteration_batches(
            ["ਰਾਜ"],
            "punjabi",
            "english",
            checkpoint_path=checkpoint,
            provider="openai",
            use_few_shot=False,
            engine=("llm",),
        )
        calls = len(fake.router.calls)
        with pytest.raises(ValueError, match="checkpoint"):
            transliterate_tokens_batched(
                ["ਰਾਜ"],
                "hindi",
                "english",
                checkpoint_path=checkpoint,
                provider="openai",
                use_few_shot=False,
                engine=("llm",),
            )
        assert len(fake.router.calls) == calls


def test_gemini_uses_batchlane_and_disables_thinking(tmp_path, monkeypatch):
    from batchlane.adapters.gemini import BASE_URL

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with respx.mock as router:
        submit = router.post(
            f"{BASE_URL}/models/gemini-2.5-flash:batchGenerateContent"
        ).mock(return_value=httpx.Response(200, json={"name": "batches/one"}))
        state = submit_transliteration_batches(
            ["ਰਾਜ"],
            "punjabi",
            "english",
            checkpoint_path=tmp_path / "run.jsonl",
            provider="gemini",
            model="gemini-2.5-flash",
            use_few_shot=False,
            engine=("llm",),
        )
        request = json.loads(submit.calls[0].request.content)["batch"]["input_config"][
            "requests"
        ]["requests"][0]["request"]
        assert request["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
        assert state.jobs[0].batch_id == "batches/one"


def test_unsupported_lane_is_rejected_before_generating_paid_examples(tmp_path):
    import batchlane as bl

    with (
        patch(
            "indicate.llm_indic.IndicLLMTransliterator.generate_few_shot_examples"
        ) as examples,
        pytest.raises(bl.AdapterNotShippedError),
    ):
        submit_transliteration_batches(
            ["ਰਾਜ"],
            "punjabi",
            "english",
            checkpoint_path=tmp_path / "run.jsonl",
            provider="bedrock",
            model="anthropic.claude-sonnet-4-6",
            engine=("llm",),
        )
    examples.assert_not_called()
