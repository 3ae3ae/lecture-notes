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


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True

    name = type(exc).__name__.lower()
    return "timeout" in name or "rate" in name or "connection" in name


def _call_chat_completion(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    request_options: Mapping[str, Any] | None = None,
    retry_config: RetryConfig | None = None,
) -> str:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }
    if request_options:
        kwargs.update(request_options)

    attempts = 1
    backoff_seconds = 1.0
    if retry_config is not None:
        attempts += max(0, retry_config.retries)
        backoff_seconds = retry_config.backoff_seconds

    for attempt_number in range(1, attempts + 1):
        try:
            completion = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            if attempt_number >= attempts or not _is_retryable_error(exc):
                raise
            time.sleep(backoff_seconds * (2 ** (attempt_number - 1)))

    content = completion.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI returned an empty response.")
    return content.strip()


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
    corrected_text = _call_chat_completion(
        client=correction_config.client,
        model=correction_config.model,
        system_prompt=system_prompt,
        user_text=raw_text,
        request_options=correction_config.request_options,
        retry_config=retry_config,
    )

    formatting_config = stage_configs["formatting"]
    stage_number, stage_name, system_prompt = _STAGE_DETAILS["formatting"]
    if on_stage is not None:
        on_stage(stage_number, stage_name)
    formatted_transcript = _call_chat_completion(
        client=formatting_config.client,
        model=formatting_config.model,
        system_prompt=system_prompt,
        user_text=corrected_text,
        request_options=formatting_config.request_options,
        retry_config=retry_config,
    )

    summary_config = stage_configs["summary"]
    stage_number, stage_name, system_prompt = _STAGE_DETAILS["summary"]
    if on_stage is not None:
        on_stage(stage_number, stage_name)
    summary_text = _call_chat_completion(
        client=summary_config.client,
        model=summary_config.model,
        system_prompt=system_prompt,
        user_text=formatted_transcript,
        request_options=summary_config.request_options,
        retry_config=retry_config,
    )

    cornell_config = stage_configs["cornell"]
    stage_number, stage_name, system_prompt = _STAGE_DETAILS["cornell"]
    if on_stage is not None:
        on_stage(stage_number, stage_name)
    cornell_notes_text = _call_chat_completion(
        client=cornell_config.client,
        model=cornell_config.model,
        system_prompt=system_prompt,
        user_text=formatted_transcript,
        request_options=cornell_config.request_options,
        retry_config=retry_config,
    )
    return ProcessedDocument(
        corrected_text=corrected_text,
        formatted_transcript=formatted_transcript,
        summary_text=summary_text,
        cornell_notes_text=cornell_notes_text,
    )
