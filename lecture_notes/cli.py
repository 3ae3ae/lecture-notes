"""CLI entrypoint for lecture_notes."""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import os
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from lecture_notes.pipeline import RetryConfig, StageConfig, run_pipeline_with_progress

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_EXCLUDE_DIRS = {".git", ".venv", "node_modules", "__pycache__"}
READ_ENCODINGS = ("utf-8", "utf-8-sig", "cp949")
CONFIG_FILENAME = "lecture-notes.toml"
STAGE_NAMES = ("correction", "formatting", "summary", "cornell")
COMMON_REQUEST_OPTIONS = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "timeout",
}
OPENAI_ONLY_REQUEST_OPTIONS = {
    "reasoning_effort",
    "service_tier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "store",
    "metadata",
    "safety_identifier",
}
REQUEST_OPTIONS = COMMON_REQUEST_OPTIONS | OPENAI_ONLY_REQUEST_OPTIONS


class ConfigError(ValueError):
    """Raised when CLI or TOML configuration is invalid."""


@dataclass(slots=True)
class ProviderConfig:
    name: str
    type: str
    base_url: str | None
    api_key: str | None
    request_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageSettings:
    name: str
    provider_name: str
    model: str
    request_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineSettings:
    providers: dict[str, ProviderConfig]
    stages: dict[str, StageSettings]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lecture-notes",
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
        help="Model name. Falls back to LECTURE_NOTES_MODEL.",
    )
    parser.add_argument(
        "--api-key",
        help=(
            "API key for OpenAI or an OpenAI-compatible server. "
            "Falls back to LECTURE_NOTES_API_KEY or OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--base-url",
        help=(
            "Optional OpenAI-compatible base URL. "
            "Falls back to LECTURE_NOTES_BASE_URL."
        ),
    )
    parser.add_argument(
        "--config",
        help=f"Path to a TOML config file. Default: ./{CONFIG_FILENAME} if present.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Profile name from the TOML config. Default: default.",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess txt files even when matching md files already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most N matching txt files after discovery.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of files to process concurrently. Default: 1.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count for transient API errors. Default: 2.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="Initial retry backoff in seconds. Default: 1.0.",
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


def normalize_summary_text(summary_text: str) -> str:
    normalized_lines: list[str] = []
    replacements = {
        "[핵심 요약]": "### 핵심 요약",
        "[교수님 강조 포인트]": "### 교수님 강조 포인트",
        "## 핵심 요약": "### 핵심 요약",
        "## 교수님 강조 포인트": "### 교수님 강조 포인트",
    }

    for line in summary_text.strip().splitlines():
        stripped = line.strip()
        normalized_lines.append(replacements.get(stripped, line))

    normalized = "\n".join(normalized_lines).strip()
    if not normalized:
        return "### 핵심 요약"
    if "### 핵심 요약" not in normalized:
        normalized = f"### 핵심 요약\n{normalized}"
    return normalized


def write_markdown(
    output_path: Path,
    summary_text: str,
    cornell_notes_text: str,
    transcript_text: str,
) -> None:
    normalized_summary = normalize_summary_text(summary_text)
    content = (
        f"## 요약\n\n{normalized_summary}\n\n"
        f"## 코넬 노트\n\n{cornell_notes_text.strip()}\n\n"
        f"## 전체 전사문\n\n{transcript_text.strip()}\n"
    )
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


def _load_config_file(args: argparse.Namespace) -> tuple[Path | None, dict[str, Any]]:
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise ConfigError(f"config file does not exist: {config_path}")
    else:
        config_path = Path.cwd() / CONFIG_FILENAME
        if not config_path.exists():
            return None, {}

    try:
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"failed to parse {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a TOML table: {config_path}")
    return config_path, data


def _expect_table(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a table.")
    return value


def _request_options_from_table(
    table: Mapping[str, Any],
    *,
    known_keys: set[str],
    context: str,
) -> dict[str, Any]:
    unknown_keys = set(table) - known_keys - REQUEST_OPTIONS
    if unknown_keys:
        raise ConfigError(
            f"{context} has unknown option(s): {', '.join(sorted(unknown_keys))}"
        )
    options = {key: table[key] for key in REQUEST_OPTIONS if key in table}
    if "max_tokens" in options and "max_completion_tokens" in options:
        raise ConfigError(
            f"{context} cannot set both max_tokens and max_completion_tokens."
        )
    return options


def _validate_request_options(
    *,
    provider_type: str,
    options: Mapping[str, Any],
    context: str,
) -> None:
    if "max_tokens" in options and "max_completion_tokens" in options:
        raise ConfigError(
            f"{context} cannot set both max_tokens and max_completion_tokens."
        )
    if provider_type == "compatible":
        openai_only = sorted(set(options) & OPENAI_ONLY_REQUEST_OPTIONS)
        if openai_only:
            raise ConfigError(
                f"{context} uses OpenAI-only option(s) with compatible provider: "
                f"{', '.join(openai_only)}"
            )


def _resolve_api_key(*, explicit: str | None, env_name: str | None) -> str | None:
    if explicit:
        return explicit
    if env_name:
        return os.environ.get(env_name)
    return None


def _resolve_model(args: argparse.Namespace) -> str | None:
    return args.model or os.environ.get("LECTURE_NOTES_MODEL")


def _build_client(args: argparse.Namespace) -> "OpenAI":
    from openai import OpenAI

    client_kwargs: dict[str, str] = {}
    api_key = (
        args.api_key
        or os.environ.get("LECTURE_NOTES_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if api_key:
        client_kwargs["api_key"] = api_key
    base_url = (
        args.base_url
        or os.environ.get("LECTURE_NOTES_BASE_URL")
    )
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def _build_client_from_provider(provider: ProviderConfig) -> "OpenAI":
    from openai import OpenAI

    client_kwargs: dict[str, str] = {}
    if provider.api_key:
        client_kwargs["api_key"] = provider.api_key
    if provider.base_url:
        client_kwargs["base_url"] = provider.base_url
    return OpenAI(**client_kwargs)


def _implicit_pipeline_settings(args: argparse.Namespace) -> PipelineSettings:
    model = _resolve_model(args)
    if not model:
        raise ConfigError(
            "model is required. Pass --model or set LECTURE_NOTES_MODEL."
        )

    base_url = args.base_url or os.environ.get("LECTURE_NOTES_BASE_URL")
    provider_type = "compatible" if base_url else "openai"
    api_key = (
        args.api_key
        or os.environ.get("LECTURE_NOTES_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise ConfigError(
            "API key is required. Pass --api-key or set "
            "LECTURE_NOTES_API_KEY or OPENAI_API_KEY."
        )

    provider = ProviderConfig(
        name="default",
        type=provider_type,
        base_url=base_url,
        api_key=api_key,
    )
    stages = {
        stage_name: StageSettings(
            name=stage_name,
            provider_name="default",
            model=model,
        )
        for stage_name in STAGE_NAMES
    }
    return PipelineSettings(providers={"default": provider}, stages=stages)


def _parse_provider_configs(
    config_data: Mapping[str, Any],
) -> dict[str, ProviderConfig]:
    providers_table = _expect_table(config_data.get("providers", {}), "providers")
    if not providers_table:
        raise ConfigError("config file must define at least one provider.")

    providers: dict[str, ProviderConfig] = {}
    provider_keys = {"type", "base_url", "api_key_env"}
    for provider_name, raw_provider in providers_table.items():
        provider_table = _expect_table(raw_provider, f"providers.{provider_name}")
        provider_type = provider_table.get("type", "compatible")
        if provider_type not in {"openai", "compatible"}:
            raise ConfigError(
                f"providers.{provider_name}.type must be 'openai' or 'compatible'."
            )
        options = _request_options_from_table(
            provider_table,
            known_keys=provider_keys,
            context=f"providers.{provider_name}",
        )
        _validate_request_options(
            provider_type=provider_type,
            options=options,
            context=f"providers.{provider_name}",
        )
        api_key_env = provider_table.get("api_key_env")
        if api_key_env is not None and not isinstance(api_key_env, str):
            raise ConfigError(f"providers.{provider_name}.api_key_env must be a string.")
        base_url = provider_table.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ConfigError(f"providers.{provider_name}.base_url must be a string.")
        providers[provider_name] = ProviderConfig(
            name=provider_name,
            type=provider_type,
            base_url=base_url,
            api_key=_resolve_api_key(explicit=None, env_name=api_key_env),
            request_options=options,
        )
    return providers


def _select_stages_table(
    config_data: Mapping[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    profiles_table = _expect_table(config_data.get("profiles", {}), "profiles")
    if profile_name != "default" or profile_name in profiles_table:
        if profile_name not in profiles_table:
            raise ConfigError(f"profile not found: {profile_name}")
        profile_table = _expect_table(
            profiles_table[profile_name],
            f"profiles.{profile_name}",
        )
        base_stages = dict(_expect_table(config_data.get("stages", {}), "stages"))
        if "stages" in profile_table:
            profile_stages = _expect_table(
                profile_table["stages"],
                f"profiles.{profile_name}.stages",
            )
        else:
            profile_stages = profile_table
        return {**base_stages, **profile_stages}
    return _expect_table(config_data.get("stages", {}), "stages")


def _parse_stage_settings(
    config_data: Mapping[str, Any],
    providers: Mapping[str, ProviderConfig],
    args: argparse.Namespace,
) -> dict[str, StageSettings]:
    stages_table = _select_stages_table(config_data, args.profile)
    if not stages_table:
        raise ConfigError("config file must define stages.")

    stages: dict[str, StageSettings] = {}
    stage_keys = {"provider", "model"}
    for stage_name in STAGE_NAMES:
        if stage_name not in stages_table:
            raise ConfigError(f"missing stage config: {stage_name}")
        stage_table = _expect_table(stages_table[stage_name], f"stages.{stage_name}")
        provider_name = stage_table.get("provider")
        if not isinstance(provider_name, str):
            raise ConfigError(f"stages.{stage_name}.provider must be a string.")
        if provider_name not in providers:
            raise ConfigError(
                f"stages.{stage_name}.provider references unknown provider: "
                f"{provider_name}"
            )
        model = args.model or stage_table.get("model")
        if not isinstance(model, str) or not model:
            raise ConfigError(f"stages.{stage_name}.model must be a string.")

        provider = providers[provider_name]
        stage_options = _request_options_from_table(
            stage_table,
            known_keys=stage_keys,
            context=f"stages.{stage_name}",
        )
        request_options = {**provider.request_options, **stage_options}
        _validate_request_options(
            provider_type=provider.type,
            options=request_options,
            context=f"stages.{stage_name}",
        )
        stages[stage_name] = StageSettings(
            name=stage_name,
            provider_name=provider_name,
            model=model,
            request_options=request_options,
        )

    unknown_stages = set(stages_table) - set(STAGE_NAMES)
    if unknown_stages:
        raise ConfigError(
            f"stages has unknown stage(s): {', '.join(sorted(unknown_stages))}"
        )
    return stages


def _pipeline_settings_from_config(
    config_data: Mapping[str, Any],
    args: argparse.Namespace,
) -> PipelineSettings:
    root_keys = {"providers", "stages", "profiles"}
    unknown_root_keys = set(config_data) - root_keys
    if unknown_root_keys:
        raise ConfigError(
            f"config file has unknown top-level table(s): "
            f"{', '.join(sorted(unknown_root_keys))}"
        )
    providers = _parse_provider_configs(config_data)
    stages = _parse_stage_settings(config_data, providers, args)
    return PipelineSettings(providers=providers, stages=stages)


def _resolve_pipeline_settings(args: argparse.Namespace) -> tuple[Path | None, PipelineSettings]:
    config_path, config_data = _load_config_file(args)
    if not config_data:
        if args.profile != "default":
            raise ConfigError("--profile requires a config file.")
        return config_path, _implicit_pipeline_settings(args)
    return config_path, _pipeline_settings_from_config(config_data, args)


def _build_stage_configs(settings: PipelineSettings) -> dict[str, StageConfig]:
    client_cache: dict[str, Any] = {}
    stage_configs: dict[str, StageConfig] = {}
    for stage_name in STAGE_NAMES:
        stage = settings.stages[stage_name]
        if stage.provider_name not in client_cache:
            provider = settings.providers[stage.provider_name]
            if not provider.api_key:
                raise ConfigError(
                    f"API key is required for provider '{provider.name}'. "
                    "Set its api_key_env environment variable."
                )
            client_cache[stage.provider_name] = _build_client_from_provider(provider)
        stage_configs[stage_name] = StageConfig(
            name=stage_name,
            client=client_cache[stage.provider_name],
            model=stage.model,
            request_options=dict(stage.request_options),
        )
    return stage_configs


def _format_progress(index: int, total: int, txt_path: Path) -> str:
    return f"[{index}/{total}] {txt_path}"


def _process_file(
    *,
    index: int,
    total_files: int,
    txt_path: Path,
    args: argparse.Namespace,
    stage_configs: Mapping[str, StageConfig] | None,
    retry_config: RetryConfig,
) -> tuple[str, str, str | None]:
    output_path = txt_path.with_suffix(".md")
    progress_prefix = _format_progress(index, total_files, txt_path)

    if not args.overwrite and should_skip(txt_path):
        return "skipped", f"{progress_prefix} skip existing -> {output_path}", None

    try:
        print(f"{progress_prefix} reading")
        raw_text = read_text_file(txt_path)
        if not raw_text.strip():
            return "skipped", f"{progress_prefix} skip empty", None

        if args.dry_run:
            return "processed", f"{progress_prefix} would-process -> {output_path}", None

        result = run_pipeline_with_progress(
            raw_text,
            stage_configs=stage_configs,
            retry_config=retry_config,
            on_stage=(
                lambda stage_number, stage_name: _log(
                    f"{progress_prefix} stage {stage_number}/4 {stage_name}",
                    verbose=args.verbose,
                )
            ),
        )
        _log(f"{progress_prefix} writing markdown", verbose=args.verbose)
        write_markdown(
            output_path,
            summary_text=result.summary_text,
            cornell_notes_text=result.cornell_notes_text,
            transcript_text=result.formatted_transcript,
        )
        return "processed", f"{progress_prefix} processed -> {output_path}", None
    except Exception as exc:  # pragma: no cover - exercised by CLI tests
        return "error", "", f"{progress_prefix} error: {exc}"


def _count_result(
    result: tuple[str, str, str | None],
    counts: dict[str, int],
) -> None:
    status, message, error_message = result
    if status == "processed":
        counts["processed"] += 1
    elif status == "skipped":
        counts["skipped"] += 1
    else:
        counts["errors"] += 1
    if message:
        print(message)
    if error_message:
        print(error_message, file=sys.stderr)


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

    if args.limit is not None and args.limit < 0:
        print("error: --limit must be >= 0.", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print("error: --jobs must be >= 1.", file=sys.stderr)
        return 2
    if args.retries < 0:
        print("error: --retries must be >= 0.", file=sys.stderr)
        return 2
    if args.retry_backoff < 0:
        print("error: --retry-backoff must be >= 0.", file=sys.stderr)
        return 2

    stage_configs: Mapping[str, StageConfig] | None = None
    if not args.dry_run:
        try:
            config_path, pipeline_settings = _resolve_pipeline_settings(args)
            stage_configs = _build_stage_configs(pipeline_settings)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if config_path is not None:
            _log(f"using config {config_path}", verbose=args.verbose)

    txt_files = discover_txt_files(root, include_globs, exclude_dirs)
    if args.limit is not None:
        txt_files = txt_files[: args.limit]

    counts = {"processed": 0, "skipped": 0, "errors": 0}
    retry_config = RetryConfig(
        retries=args.retries,
        backoff_seconds=args.retry_backoff,
    )

    print(f"searching {root}")
    print(f"found {len(txt_files)} matching txt file(s)")

    total_files = len(txt_files)
    file_jobs = [
        (index, txt_path)
        for index, txt_path in enumerate(txt_files, start=1)
    ]

    if args.jobs == 1 or args.dry_run:
        for index, txt_path in file_jobs:
            result = _process_file(
                index=index,
                total_files=total_files,
                txt_path=txt_path,
                args=args,
                stage_configs=stage_configs,
                retry_config=retry_config,
            )
            _count_result(result, counts)
            if result[0] == "error" and args.fail_fast:
                return 1
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(
                    _process_file,
                    index=index,
                    total_files=total_files,
                    txt_path=txt_path,
                    args=args,
                    stage_configs=stage_configs,
                    retry_config=retry_config,
                )
                for index, txt_path in file_jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                _count_result(result, counts)
                if result[0] == "error" and args.fail_fast:
                    for pending in futures:
                        pending.cancel()
                    return 1

    print(
        "done "
        f"processed={counts['processed']} "
        f"skipped={counts['skipped']} "
        f"errors={counts['errors']}"
    )
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
