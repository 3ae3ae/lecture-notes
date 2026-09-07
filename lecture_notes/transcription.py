"""Local, word-aligned speaker transcription with whispermlx."""

from __future__ import annotations

import os
import platform
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm", ".mkv"}


def validate_audio_environment() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("whispermlx requires Apple Silicon macOS; txt input works on other platforms.")
    if sys.version_info >= (3, 14):
        raise RuntimeError("whispermlx requires Python 3.11–3.13 for this project.")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required: brew install ffmpeg")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "Set HF_TOKEN and accept the pyannote/speaker-diarization-community-1 model agreement."
        )


def render_transcript(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for segment in result["segments"]:
        # Word labels preserve speaker changes inside a single ASR segment.
        words = segment.get("words") or []
        if not words:
            words = [{"word": segment["text"], "start": segment["start"],
                      "speaker": segment.get("speaker", "UNKNOWN")}]
        speaker = None
        parts: list[str] = []
        start = segment["start"]
        for word in words:
            current = word.get("speaker", "UNKNOWN")
            if parts and current != speaker:
                lines.append(f"[{start:.2f}s] [{speaker}] {' '.join(parts)}")
                parts = []
            if not parts:
                start = word.get("start", segment["start"])
                speaker = current
            text = word["word"].strip()
            if text:
                parts.append(text)
        if parts:
            lines.append(f"[{start:.2f}s] [{speaker}] {' '.join(parts)}")
    return "\n".join(lines)


@contextmanager
def _progress(stage: str, on_stage: Callable[[str], None]):
    started = time.monotonic()
    stopped = threading.Event()
    percent = 0

    def report(message: str) -> None:
        on_stage(f"{stage}: {message} (elapsed {time.monotonic() - started:.0f}s)")

    def update(value: float) -> None:
        nonlocal percent
        current = min(100, max(0, int(value)))
        if current > percent:
            percent = current
            report(f"{current}%")

    def heartbeat() -> None:
        while not stopped.wait(30):
            report(f"waiting for next update; last reported {percent}%")

    report("0%")
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield update
    finally:
        stopped.set()
        worker.join()
    report("completed")


def transcribe_audio(
    path: Path,
    *,
    model: str = "large-v3",
    language: str | None = "ko",
    on_stage: Callable[[str], None] = print,
) -> str:
    validate_audio_environment()
    try:
        import whispermlx
        from whispermlx.diarize import DiarizationPipeline
    except ImportError as exc:
        raise RuntimeError('Install audio support: uv tool install ".[audio]" --python 3.12') from exc

    on_stage("loading audio and transcribing with whispermlx")
    audio = whispermlx.load_audio(str(path))
    asr = whispermlx.load_model(model, device="cpu", language=language, vad_method="silero")
    result = asr.transcribe(audio)
    del asr
    if not result["segments"]:
        return ""
    on_stage("loading word alignment model")
    aligner, metadata = whispermlx.load_align_model(language_code=result["language"], device="cpu")
    with _progress("aligning words", on_stage) as progress:
        result = whispermlx.align(
            result["segments"], aligner, metadata, audio, device="cpu",
            progress_callback=progress,
        )
    del aligner
    on_stage("loading speaker diarization model")
    diarizer = DiarizationPipeline(token=os.environ["HF_TOKEN"], device="cpu")
    with _progress("identifying speakers", on_stage) as progress:
        speakers = diarizer(audio, progress_callback=progress)
    return render_transcript(whispermlx.assign_word_speakers(speakers, result))
