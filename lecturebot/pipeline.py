"""OpenAI pipeline for transcript correction, formatting, and summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lecturebot.prompts import (
    CORRECTION_SYSTEM_PROMPT,
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
    corrected_text = _call_chat_completion(
        client=client,
        model=model,
        system_prompt=CORRECTION_SYSTEM_PROMPT,
        user_text=raw_text,
    )
    formatted_transcript = _call_chat_completion(
        client=client,
        model=model,
        system_prompt=FORMATTING_SYSTEM_PROMPT,
        user_text=corrected_text,
    )
    summary_text = _call_chat_completion(
        client=client,
        model=model,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_text=formatted_transcript,
    )
    return ProcessedDocument(
        corrected_text=corrected_text,
        formatted_transcript=formatted_transcript,
        summary_text=summary_text,
    )
