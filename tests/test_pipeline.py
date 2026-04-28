from __future__ import annotations

import unittest

from lecture_notes.pipeline import (
    ProcessedDocument,
    StageConfig,
    run_pipeline,
    run_pipeline_with_progress,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeCompletion:
        self.calls.append(kwargs)
        content = f"stage-{len(self.calls)}"
        return _FakeCompletion(content)


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


class PipelineTests(unittest.TestCase):
    def test_run_pipeline_preserves_four_stage_order(self) -> None:
        client = _FakeClient()

        result = run_pipeline("raw text", client=client, model="gpt-test")

        self.assertIsInstance(result, ProcessedDocument)
        self.assertEqual(result.corrected_text, "stage-1")
        self.assertEqual(result.formatted_transcript, "stage-2")
        self.assertEqual(result.summary_text, "stage-3")
        self.assertEqual(result.cornell_notes_text, "stage-4")

        calls = client.chat.completions.calls
        self.assertEqual(calls[0]["messages"][1]["content"], "raw text")
        self.assertEqual(calls[1]["messages"][1]["content"], "stage-1")
        self.assertEqual(calls[2]["messages"][1]["content"], "stage-2")
        self.assertEqual(calls[3]["messages"][1]["content"], "stage-2")

    def test_run_pipeline_with_progress_reports_all_stages(self) -> None:
        client = _FakeClient()
        stages: list[tuple[int, str]] = []

        run_pipeline_with_progress(
            "raw text",
            client=client,
            model="gpt-test",
            on_stage=lambda stage_number, stage_name: stages.append(
                (stage_number, stage_name)
            ),
        )

        self.assertEqual(
            stages,
            [
                (1, "correcting transcript"),
                (2, "formatting transcript"),
                (3, "summarizing transcript"),
                (4, "creating Cornell notes"),
            ],
        )

    def test_stage_configs_apply_models_and_request_options(self) -> None:
        correction_client = _FakeClient()
        openai_client = _FakeClient()
        stage_configs = {
            "correction": StageConfig(
                name="correction",
                client=correction_client,
                model="local-model",
            ),
            "formatting": StageConfig(
                name="formatting",
                client=correction_client,
                model="local-model",
            ),
            "summary": StageConfig(
                name="summary",
                client=openai_client,
                model="gpt-test",
                request_options={"reasoning_effort": "minimal"},
            ),
            "cornell": StageConfig(
                name="cornell",
                client=openai_client,
                model="gpt-test",
                request_options={"reasoning_effort": "medium"},
            ),
        }

        result = run_pipeline_with_progress(
            "raw text",
            stage_configs=stage_configs,
        )

        self.assertEqual(result.summary_text, "stage-1")
        self.assertEqual(correction_client.chat.completions.calls[0]["model"], "local-model")
        self.assertEqual(openai_client.chat.completions.calls[0]["model"], "gpt-test")
        self.assertEqual(
            openai_client.chat.completions.calls[0]["reasoning_effort"],
            "minimal",
        )
        self.assertEqual(
            openai_client.chat.completions.calls[1]["reasoning_effort"],
            "medium",
        )
