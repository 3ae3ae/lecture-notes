"""CLI entrypoint for lecturebot."""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from lecturebot.pipeline import run_pipeline

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_EXCLUDE_DIRS = {".git", ".venv", "node_modules", "__pycache__"}
READ_ENCODINGS = ("utf-8", "utf-8-sig", "cp949")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lecturebot",
        description="Process lecture transcript txt files into markdown summaries.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Root directory to recursively search for transcript txt files.",
    )
    parser.add_argument(
        "--model",
        help="Model name. Falls back to LECTUREBOT_MODEL.",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key. Falls back to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--include-glob",
        action="append",
        default=None,
        help="Glob pattern for files to include. Repeatable. Default: *.txt",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=None,
        help="Directory name to exclude from recursion. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed or skipped without calling the API.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file progress logs.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first processing error.",
    )
    return parser.parse_args(argv)


def discover_txt_files(
    root: Path,
    include_globs: Iterable[str],
    exclude_dirs: Iterable[str],
) -> list[Path]:
    include_patterns = tuple(include_globs)
    excluded = set(exclude_dirs)
    found: list[Path] = []

    for current_root, dirs, files in os.walk(root):
        dirs[:] = sorted(directory for directory in dirs if directory not in excluded)
        for filename in sorted(files):
            if any(fnmatch.fnmatch(filename, pattern) for pattern in include_patterns):
                found.append(Path(current_root, filename))
    return found


def should_skip(txt_path: Path) -> bool:
    return txt_path.with_suffix(".md").exists()


def read_text_file(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in READ_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read {path}")


def write_markdown(output_path: Path, summary_text: str, transcript_text: str) -> None:
    content = f"[핵심 요약]\n{summary_text.strip()}\n\n[전체 전사문]\n{transcript_text.strip()}\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    temp_path.replace(output_path)


def _log(message: str, *, verbose: bool = True, stream: object = sys.stdout) -> None:
    if verbose:
        print(message, file=stream)


def _resolve_model(args: argparse.Namespace) -> str | None:
    return args.model or os.environ.get("LECTUREBOT_MODEL")


def _build_client(args: argparse.Namespace) -> "OpenAI":
    from openai import OpenAI

    client_kwargs: dict[str, str] = {}
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if api_key:
        client_kwargs["api_key"] = api_key
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    return OpenAI(**client_kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.path).resolve()
    include_globs = args.include_glob or ["*.txt"]
    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir or [])

    if not root.exists():
        print(f"error: path does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"error: path is not a directory: {root}", file=sys.stderr)
        return 2

    model = _resolve_model(args)
    if not args.dry_run and not model:
        print(
            "error: model is required. Pass --model or set LECTUREBOT_MODEL.",
            file=sys.stderr,
        )
        return 2

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        print(
            "error: API key is required. Pass --api-key or set OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 2

    txt_files = discover_txt_files(root, include_globs, exclude_dirs)
    client: Any = None if args.dry_run else _build_client(args)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    if args.verbose:
        _log(f"Searching under {root}", verbose=True)
        _log(f"Found {len(txt_files)} matching txt file(s)", verbose=True)

    for txt_path in txt_files:
        output_path = txt_path.with_suffix(".md")

        if should_skip(txt_path):
            skipped_count += 1
            print(f"skip {txt_path} -> {output_path}")
            continue

        try:
            raw_text = read_text_file(txt_path)
            if not raw_text.strip():
                skipped_count += 1
                print(f"skip empty {txt_path}")
                continue

            if args.dry_run:
                processed_count += 1
                print(f"would-process {txt_path} -> {output_path}")
                continue

            _log(f"processing {txt_path}", verbose=args.verbose)
            result = run_pipeline(raw_text, client=client, model=model)
            write_markdown(
                output_path,
                summary_text=result.summary_text,
                transcript_text=result.formatted_transcript,
            )
            processed_count += 1
            print(f"processed {txt_path} -> {output_path}")
        except Exception as exc:  # pragma: no cover - exercised by CLI tests
            error_count += 1
            print(f"error {txt_path}: {exc}", file=sys.stderr)
            if args.fail_fast:
                return 1

    print(
        f"done processed={processed_count} skipped={skipped_count} errors={error_count}"
    )
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
