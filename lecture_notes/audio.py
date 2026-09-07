"""Audio preparation and resumable local transcription."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from lecture_notes.transcription import transcribe_audio


def transcript_cache_path(path: Path) -> Path:
    return path.with_name(path.name + ".transcript.json")


def cached_transcription(path: Path, *, model: str, language: str | None,
                         on_stage: Callable[[str], None]) -> str:
    stat = path.stat()
    signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "model": model, "language": language, "version": 1}
    cache = transcript_cache_path(path)
    try:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if (isinstance(saved, dict) and saved.get("source") == signature
                and isinstance(saved.get("text"), str) and saved["text"].strip()):
            on_stage("reusing saved speaker transcript")
            return saved["text"]
    except (OSError, ValueError):
        pass
    text = transcribe_audio(path, model=model, language=language, on_stage=on_stage)
    if text.strip():
        if path.stat().st_mtime_ns != stat.st_mtime_ns or path.stat().st_size != stat.st_size:
            raise RuntimeError(f"audio changed during transcription: {path}; retry")
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as temp:
                temp_path = Path(temp.name)
                json.dump({"source": signature, "text": text}, temp, ensure_ascii=False)
                temp.write("\n")
            temp_path.replace(cache)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    return text


def _probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_type:format=bit_rate,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration", 0))
    if not data.get("streams") or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"no valid audio stream: {path}")
    return data["format"]


def compress_audio(path: Path) -> str:
    """Replace M4A only after a successful, smaller mono AAC conversion."""
    original = path.stat()
    before = _probe(path)
    bitrate = str(before.get("bit_rate", ""))
    if bitrate.isdigit() and int(bitrate) <= 69000:
        return "skipped (already <= 69 kbps)"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".m4a", dir=path.parent)
    os.close(descriptor)
    temp = Path(name)
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(path),
             "-map", "0:a:0", "-vn", "-c:a", "aac", "-ac", "1", "-b:a", "64k", str(temp)],
            check=True, capture_output=True, text=True,
        )
        if temp.stat().st_size == 0:
            raise RuntimeError(f"empty compressed audio: {path}")
        after = _probe(temp)
        if abs(float(after["duration"]) - float(before["duration"])) > 1:
            raise RuntimeError(f"compressed audio duration changed: {path}")
        if temp.stat().st_size >= path.stat().st_size:
            return "skipped (compressed file is not smaller)"
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != (original.st_size, original.st_mtime_ns):
            raise RuntimeError(f"audio changed during compression: {path}")
        temp.replace(path)
        return "compressed to mono AAC 64 kbps"
    finally:
        temp.unlink(missing_ok=True)
