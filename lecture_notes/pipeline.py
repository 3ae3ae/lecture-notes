"""OpenAI pipeline for transcript correction, formatting, and summarization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

from lecture_notes.prompts import (
    CORRECTION_SYSTEM_PROMPT,
    CORNELL_NOTES_SYSTEM_PROMPT,
    FORMATTING_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from openai import OpenAI


@dataclass(slots=True)
class ProcessedDocument:
    corrected_text: str
    formatted_transcript: str
    summary_text: str
    cornell_notes_text: str


@dataclass(slots=True)
class StageConfig:
    name: str
    client: Any
    model: str
    request_options: dict[str, Any] = field(default_factory=dict)
    api: str = "chat_completions"


@dataclass(slots=True)
class RetryConfig:
    retries: int = 0
    backoff_seconds: float = 1.0


STAGE_ORDER = ("correction", "formatting", "summary", "cornell")

_STAGE_DETAILS = {
    "correction": (1, "correcting transcript", CORRECTION_SYSTEM_PROMPT),
    "formatting": (2, "formatting transcript", FORMATTING_SYSTEM_PROMPT),
    "summary": (3, "summarizing transcript", SUMMARY_SYSTEM_PROMPT),
    "cornell": (4, "creating Cornell notes", CORNELL_NOTES_SYSTEM_PROMPT),
}


class ChatCompletionsModelClient:
    def __init__(self, client: Any) -> None:
        self.client = client

    def create_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        request_options: Mapping[str, Any],
    ) -> str:
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            **request_options,
        )
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned an empty response.")
        return content.strip()


class ResponsesModelClient:
    def __init__(self, client: Any) -> None:
        self.client = client

    def create_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_text: str,
        request_options: Mapping[str, Any],
    ) -> str:
        response = self.client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_text,
            **request_options,
        )
        if getattr(response, "status", None) == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None)
            if reason is None and isinstance(details, Mapping):
                reason = details.get("reason")
            message = "OpenAI returned an incomplete response."
            if reason:
                message = f"{message} reason={reason}"
            raise RuntimeError(message)

        content = getattr(response, "output_text", None)
        if content is None or not content.strip():
            raise RuntimeError("OpenAI returned an empty response.")
        return content.strip()


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True

    name = type(exc).__name__.lower()
    return "timeout" in name or "rate" in name or "connection" in name


def _model_client_for_stage(
    stage_config: StageConfig,
) -> ChatCompletionsModelClient | ResponsesModelClient:
    if stage_config.api == "chat_completions":
        return ChatCompletionsModelClient(stage_config.client)
    if stage_config.api == "responses":
        return ResponsesModelClient(stage_config.client)
    raise ValueError(f"unsupported stage API: {stage_config.api}")


def _call_model(
    *,
    stage_config: StageConfig,
    system_prompt: str,
    user_text: str,
    retry_config: RetryConfig | None = None,
) -> str:
    attempts = 1
    backoff_seconds = 1.0
    if retry_config is not None:
        attempts += max(0, retry_config.retries)
        backoff_seconds = retry_config.backoff_seconds

    model_client = _model_client_for_stage(stage_config)
    for attempt_number in range(1, attempts + 1):
        try:
            return model_client.create_text(
                model=stage_config.model,
                system_prompt=system_prompt,
                user_text=user_text,
                request_options=stage_config.request_options,
            )
        except Exception as exc:
            if attempt_number >= attempts or not _is_retryable_error(exc):
                raise
            time.sleep(backoff_seconds * (2 ** (attempt_number - 1)))

    raise RuntimeError("OpenAI returned an empty response.")


def run_pipeline(raw_text: str, client: "OpenAI", model: str) -> ProcessedDocument:
    return run_pipeline_with_progress(raw_text, client=client, model=model)


def _default_stage_configs(client: Any, model: str) -> dict[str, StageConfig]:
    return {
        stage_name: StageConfig(name=stage_name, client=client, model=model)
        for stage_name in STAGE_ORDER
    }


def run_pipeline_with_progress(
    raw_text: str,
    client: "OpenAI" | None = None,
    model: str | None = None,
    on_stage: Callable[[int, str], None] | None = None,
    stage_configs: Mapping[str, StageConfig] | None = None,
    retry_config: RetryConfig | None = None,
) -> ProcessedDocument:
    if stage_configs is None:
        if client is None or model is None:
            raise ValueError("client and model are required without stage_configs.")
        stage_configs = _default_stage_configs(client, model)

    missing_stages = [name for name in STAGE_ORDER if name not in stage_configs]
    if missing_stages:
        raise ValueError(f"missing stage config(s): {', '.join(missing_stages)}")

    correction_config = stage_configs["correction"]
    stage_number, stage_name, system_prompt = _STAGE_DETAILS["correction"]
    if on_stage is not None:
        on_stage(stage_number, stage_name)
    corrected_text = _call_model(
        stage_config=correction_config,
        system_prompt=system_prompt,
        user_text=raw_text,
        retry_config=retry_config,
    )

    formatting_config = stage_configs["formatting"]
    stage_number, stage_name, system_prompt = _STAGE_DETAILS["formatting"]
    if on_stage is not None:
        on_stage(stage_number, stage_name)
    formatted_transcript = _call_model(
        stage_config=formatting_config,
        system_prompt=system_prompt,
        user_text=corrected_text,
        retry_config=retry_config,
    )

    summary_config = stage_configs["summary"]
    stage_number, stage_name, system_prompt = _STAGE_DETAILS["summary"]
    if on_stage is not None:
        on_stage(stage_number, stage_name)
    summary_text = _call_model(
        stage_config=summary_config,
        system_prompt=system_prompt,
        user_text=formatted_transcript,
        retry_config=retry_config,
    )

    cornell_config = stage_configs["cornell"]
    stage_number, stage_name, system_prompt = _STAGE_DETAILS["cornell"]
    if on_stage is not None:
        on_stage(stage_number, stage_name)
    cornell_notes_text = _call_model(
        stage_config=cornell_config,
        system_prompt=system_prompt,
        user_text=formatted_transcript,
        retry_config=retry_config,
    )
    return ProcessedDocument(
        corrected_text=corrected_text,
        formatted_transcript=formatted_transcript,
        summary_text=summary_text,
        cornell_notes_text=cornell_notes_text,
    )
