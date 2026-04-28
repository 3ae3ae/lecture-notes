"""OpenAI pipeline for transcript correction, formatting, and summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

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


def _call_chat_completion(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    content = completion.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI returned an empty response.")
    return content.strip()


def run_pipeline(raw_text: str, client: "OpenAI", model: str) -> ProcessedDocument:
    return run_pipeline_with_progress(raw_text, client=client, model=model)


def run_pipeline_with_progress(
    raw_text: str,
    client: "OpenAI",
    model: str,
    on_stage: Callable[[int, str], None] | None = None,
) -> ProcessedDocument:
    if on_stage is not None:
        on_stage(1, "correcting transcript")
    corrected_text = _call_chat_completion(
        client=client,
        model=model,
        system_prompt=CORRECTION_SYSTEM_PROMPT,
        user_text=raw_text,
    )
    if on_stage is not None:
        on_stage(2, "formatting transcript")
    formatted_transcript = _call_chat_completion(
        client=client,
        model=model,
        system_prompt=FORMATTING_SYSTEM_PROMPT,
        user_text=corrected_text,
    )
    if on_stage is not None:
        on_stage(3, "summarizing transcript")
    summary_text = _call_chat_completion(
        client=client,
        model=model,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_text=formatted_transcript,
    )
    if on_stage is not None:
        on_stage(4, "creating Cornell notes")
    cornell_notes_text = _call_chat_completion(
        client=client,
        model=model,
        system_prompt=CORNELL_NOTES_SYSTEM_PROMPT,
        user_text=formatted_transcript,
    )
    return ProcessedDocument(
        corrected_text=corrected_text,
        formatted_transcript=formatted_transcript,
        summary_text=summary_text,
        cornell_notes_text=cornell_notes_text,
    )
