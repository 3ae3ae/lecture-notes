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


_DEFAULT_CHOICES = object()


class _FakeCompletion:
    def __init__(self, content: str, *, choices: object = _DEFAULT_CHOICES) -> None:
        self.choices = [_FakeChoice(content)] if choices is _DEFAULT_CHOICES else choices


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeCompletion:
        self.calls.append(kwargs)
        content = f"stage-{len(self.calls)}"
        return _FakeCompletion(content)


class _FakeResponse:
    def __init__(
        self,
        content: str,
        *,
        status: str | None = None,
        incomplete_details: object | None = None,
    ) -> None:
        self.output_text = content
        self.status = status
        self.incomplete_details = incomplete_details


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        content = f"response-stage-{len(self.calls)}"
        return _FakeResponse(content)


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()
        self.responses = _FakeResponses()


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
                request_options={"temperature": 0.1},
            ),
            "cornell": StageConfig(
                name="cornell",
                client=openai_client,
                model="gpt-test",
                request_options={"temperature": 0.2},
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
            openai_client.chat.completions.calls[0]["temperature"],
            0.1,
        )
        self.assertEqual(
            openai_client.chat.completions.calls[1]["temperature"],
            0.2,
        )

    def test_chat_completions_stage_rejects_missing_choices(self) -> None:
        client = _FakeClient()
        client.chat.completions.create = lambda **kwargs: _FakeCompletion(
            "",
            choices=None,
        )
        stage_configs = {
            stage_name: StageConfig(
                name=stage_name,
                client=client,
                model="local-test",
                api="chat_completions",
            )
            for stage_name in ("correction", "formatting", "summary", "cornell")
        }

        with self.assertRaisesRegex(RuntimeError, "no chat completion choices"):
            run_pipeline_with_progress("raw text", stage_configs=stage_configs)

    def test_chat_completions_stage_rejects_empty_content(self) -> None:
        client = _FakeClient()
        client.chat.completions.create = lambda **kwargs: _FakeCompletion("")
        stage_configs = {
            stage_name: StageConfig(
                name=stage_name,
                client=client,
                model="local-test",
                api="chat_completions",
            )
            for stage_name in ("correction", "formatting", "summary", "cornell")
        }

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            run_pipeline_with_progress("raw text", stage_configs=stage_configs)

    def test_responses_stage_uses_responses_api_shape(self) -> None:
        client = _FakeClient()
        stage_configs = {
            "correction": StageConfig(
                name="correction",
                client=client,
                model="gpt-test",
                api="responses",
                request_options={
                    "max_output_tokens": 100,
                    "reasoning": {"effort": "medium"},
                    "store": False,
                },
            ),
            "formatting": StageConfig(
                name="formatting",
                client=client,
                model="gpt-test",
                api="responses",
            ),
            "summary": StageConfig(
                name="summary",
                client=client,
                model="gpt-test",
                api="responses",
            ),
            "cornell": StageConfig(
                name="cornell",
                client=client,
                model="gpt-test",
                api="responses",
            ),
        }

        result = run_pipeline_with_progress("raw text", stage_configs=stage_configs)

        self.assertEqual(result.corrected_text, "response-stage-1")
        self.assertEqual(client.chat.completions.calls, [])
        first_call = client.responses.calls[0]
        self.assertEqual(first_call["model"], "gpt-test")
        self.assertIn("instructions", first_call)
        self.assertEqual(first_call["input"], "raw text")
        self.assertEqual(first_call["max_output_tokens"], 100)
        self.assertEqual(first_call["reasoning"], {"effort": "medium"})
        self.assertFalse(first_call["store"])

    def test_responses_stage_rejects_empty_output_text(self) -> None:
        client = _FakeClient()
        client.responses.create = lambda **kwargs: _FakeResponse("")
        stage_configs = {
            stage_name: StageConfig(
                name=stage_name,
                client=client,
                model="gpt-test",
                api="responses",
            )
            for stage_name in ("correction", "formatting", "summary", "cornell")
        }

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            run_pipeline_with_progress("raw text", stage_configs=stage_configs)

    def test_responses_stage_rejects_incomplete_status(self) -> None:
        client = _FakeClient()
        client.responses.create = lambda **kwargs: _FakeResponse(
            "",
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
        stage_configs = {
            stage_name: StageConfig(
                name=stage_name,
                client=client,
                model="gpt-test",
                api="responses",
            )
            for stage_name in ("correction", "formatting", "summary", "cornell")
        }

        with self.assertRaisesRegex(RuntimeError, "max_output_tokens"):
            run_pipeline_with_progress("raw text", stage_configs=stage_configs)

    def test_mixed_pipeline_can_use_responses_and_chat_completions(self) -> None:
        openai_client = _FakeClient()
        local_client = _FakeClient()
        stage_configs = {
            "correction": StageConfig(
                name="correction",
                client=openai_client,
                model="gpt-test",
                api="responses",
            ),
            "formatting": StageConfig(
                name="formatting",
                client=openai_client,
                model="gpt-test",
                api="responses",
            ),
            "summary": StageConfig(
                name="summary",
                client=openai_client,
                model="gpt-test",
                api="responses",
            ),
            "cornell": StageConfig(
                name="cornell",
                client=local_client,
                model="local-test",
                api="chat_completions",
            ),
        }

        result = run_pipeline_with_progress("raw text", stage_configs=stage_configs)

        self.assertEqual(len(openai_client.responses.calls), 3)
        self.assertEqual(len(local_client.chat.completions.calls), 1)
        self.assertEqual(result.cornell_notes_text, "stage-1")


class TruncationTests(unittest.TestCase):
    def test_chat_truncation_stops_pipeline(self):
        from unittest import mock
        client = _FakeClient()
        completion = _FakeCompletion("partial")
        completion.choices[0].finish_reason = "length"
        with mock.patch.object(client.chat.completions, "create", return_value=completion) as create:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                run_pipeline("raw", client, "test")
            self.assertEqual(create.call_count, 1)


class RetryTests(unittest.TestCase):
    def test_transient_failure_retries_only_failed_stage(self):
        from unittest import mock
        from lecture_notes.pipeline import RetryConfig
        client = _FakeClient()
        error = RuntimeError("unavailable")
        error.status_code = 503
        responses = [error, *(_FakeCompletion(f"stage-{i}") for i in range(1, 5))]
        with mock.patch.object(client.chat.completions, "create", side_effect=responses) as create, \
             mock.patch("lecture_notes.pipeline.time.sleep") as sleep:
            result = run_pipeline_with_progress("raw", client, "test", retry_config=RetryConfig(2, 0.5))
        self.assertEqual(create.call_count, 5)
        self.assertEqual(result.cornell_notes_text, "stage-4")
        sleep.assert_called_once_with(0.5)

    def test_unrelated_error_is_not_retried(self):
        from unittest import mock
        from lecture_notes.pipeline import RetryConfig
        class GenerateError(Exception):
            pass
        client = _FakeClient()
        with mock.patch.object(client.chat.completions, "create", side_effect=GenerateError) as create:
            with self.assertRaises(GenerateError):
                run_pipeline_with_progress("raw", client, "test", retry_config=RetryConfig(2, 0))
        self.assertEqual(create.call_count, 1)

    def test_retry_status_codes_and_builtin_connection_errors(self):
        from lecture_notes.pipeline import _is_retryable_error
        for code in (400, 401, 403, 408, 429, 500, 503):
            error = RuntimeError("http error")
            error.status_code = code
            self.assertEqual(_is_retryable_error(error), code in (408, 429, 500, 503))
        self.assertTrue(_is_retryable_error(ConnectionResetError()))
        self.assertTrue(_is_retryable_error(TimeoutError()))


class SummaryTitleTests(unittest.TestCase):
    def test_summary_title_is_extracted_without_extra_api_call(self):
        from unittest import mock
        client = _FakeClient()
        outputs = ["corrected", "formatted", "# 상관관계와 인과관계\n\n### 핵심 요약\n- 내용", "# 중복 제목\n\n| 단서 / 질문 | 필기 |"]
        with mock.patch.object(client.chat.completions, "create", side_effect=[_FakeCompletion(x) for x in outputs]) as create:
            result = run_pipeline("raw", client, "test")
        self.assertEqual(result.title, "상관관계와 인과관계")
        self.assertTrue(result.cornell_notes_text.startswith("| 단서 / 질문"))
        self.assertTrue(result.summary_text.startswith("### 핵심 요약"))
        self.assertEqual(create.call_count, 4)


if __name__ == "__main__":
    unittest.main()
