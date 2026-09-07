"""CLI entrypoint for lecture_notes."""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import math
import json
import re
import unicodedata
import os
import sys
import tempfile
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from lecture_notes.pipeline import RetryConfig, StageConfig, run_pipeline_with_progress
from lecture_notes.transcription import AUDIO_SUFFIXES
from lecture_notes.audio import cached_transcription, compress_audio, transcript_cache_path

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_EXCLUDE_DIRS = {".git", ".venv", "node_modules", "__pycache__"}
AUDIO_LOCK = threading.Lock()
READ_ENCODINGS = ("utf-8", "utf-8-sig", "cp949")
CONFIG_FILENAME = "lecture-notes.toml"
GLOBAL_CONFIG_PATH = Path("~/.config/lecture-notes/config.toml")
DEFAULT_CONFIG_TEMPLATE = """[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[providers.local]
type = "compatible"
base_url = "http://localhost:1234/v1"
api_key_env = "LECTURE_NOTES_API_KEY"

[stages.correction]
provider = "openai"
model = "gpt-5.6-luna"
max_output_tokens = 20000

[stages.correction.request.reasoning]
effort = "medium"

[stages.formatting]
provider = "openai"
model = "gpt-5.6-luna"
max_output_tokens = 20000

[stages.formatting.request.reasoning]
effort = "low"

[stages.summary]
provider = "openai"
model = "gpt-5.6-terra"
max_output_tokens = 8000
service_tier = "flex"

[stages.summary.request.reasoning]
effort = "medium"

[stages.cornell]
provider = "openai"
model = "gpt-5.6-terra"
max_output_tokens = 12000
"""
STAGE_NAMES = ("correction", "formatting", "summary", "cornell")
COMMON_REQUEST_OPTIONS = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "timeout",
}
OPENAI_ONLY_REQUEST_OPTIONS = {
    "reasoning",
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
    api: str
    base_url: str | None
    api_key: str | None
    request_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageSettings:
    name: str
    provider_name: str
    model: str
    api: str
    request_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineSettings:
    providers: dict[str, ProviderConfig]
    stages: dict[str, StageSettings]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lecture-notes",
        description="Turn lecture recordings and transcripts into speaker-aware Markdown notes.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Recording, transcript, or directory to search recursively.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Temporary model override. Must be used with --api-key and "
            "--base-url."
        ),
    )
    parser.add_argument(
        "--api-key",
        help=(
            "Temporary API key override. Must be used with --model and "
            "--base-url."
        ),
    )
    parser.add_argument(
        "--base-url",
        help=(
            "Temporary OpenAI-compatible base URL override. Must be used "
            "with --model and --api-key."
        ),
    )
    parser.add_argument(
        "--config",
        help=(
            "Path to a TOML config file. Defaults to local or global config "
            "when present."
        ),
    )
    parser.add_argument(
        "--print-config-paths",
        action="store_true",
        help="Print checked config paths and exit.",
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
        help="Glob pattern for files to include. Repeatable. Default: txt and supported audio/video files.",
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
        help="Reprocess files even when matching md files already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most N matching files after discovery.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of files to process concurrently. Default: 4. Each file can make two concurrent LLM requests.",
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
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto",
                        help="Alignment/diarization device. Default: MPS when available, otherwise CPU.")
    parser.add_argument("--cpu-threads", type=int,
                        help="Torch CPU threads for alignment/diarization. Default: preserve Torch's initial setting.")
    parser.add_argument("--asr-model", default="large-v3", help="whispermlx model. Default: large-v3.")
    parser.add_argument("--language", default="ko", help="Audio language: ko (default), en, or auto.")
    parser.add_argument("--name-from-content", action="store_true",
                        help="Name new notes using the source filename and generated lecture title.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--transcribe-only", action="store_true", help="Save speaker transcripts without generating notes.")
    modes.add_argument("--compress-audio", action="store_true", help="Only compress M4A files in place to mono AAC 64 kbps.")
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
            if any(fnmatch.fnmatch(filename.lower(), pattern.lower()) for pattern in include_patterns):
                found.append(Path(current_root, filename))
    return found


def select_inputs(paths: list[Path]) -> list[Path]:
    """Prefer audio over a legacy txt with the same output basename."""
    audio_stems = {path.with_suffix("") for path in paths if path.suffix.lower() in AUDIO_SUFFIXES}
    return [path for path in paths
            if path.suffix.lower() != ".txt" or path.with_suffix("") not in audio_stems]


def _compress_files(paths: list[Path], args: argparse.Namespace) -> int:
    errors = 0
    for index, path in enumerate(paths, 1):
        try:
            status = "would-compress to mono AAC 64 kbps" if args.dry_run else compress_audio(path)
            print(f"[{index}/{len(paths)}] {path}: {status}")
        except Exception as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            errors += 1
            if args.fail_fast:
                break
    return 1 if errors else 0


def _source_marker(path: Path) -> str:
    return "<!-- lecture-notes-source: " + json.dumps(path.name, ensure_ascii=True) + " -->"


def existing_output(path: Path) -> Path | None:
    original = path.with_suffix(".md")
    if original.exists():
        return original
    # ponytail: scan note headers per source; index them once if large folders become slow.
    matches = []
    for candidate in sorted(path.parent.glob("*.md")):
        try:
            with candidate.open(encoding="utf-8") as file:
                markers = {_source_marker(path)}
                if path.suffix.lower() in AUDIO_SUFFIXES:
                    markers.add(_source_marker(path.with_suffix(".txt")))
                if file.readline().rstrip() in markers:
                    matches.append(candidate)
        except (OSError, UnicodeError):
            continue
    if len(matches) > 1:
        raise ValueError(f"multiple notes for {path}; keep one output before rerunning")
    return matches[0] if matches else None


def should_skip(txt_path: Path) -> bool:
    return existing_output(txt_path) is not None


def content_output_path(source: Path, title: str) -> Path:
    def clean(value: str, byte_limit: int) -> str:
        value = unicodedata.normalize("NFC", value)
        value = re.sub(r'[<>:"/\\|?*#\[\]\x00-\x1f\x7f]', " ", value)
        value = " ".join(value.split()).strip(" .")
        return value.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore").rstrip(" .")
    topic = clean(title, 120)
    if not topic:
        return source.with_suffix(".md")
    stem = clean(source.stem, 100) or "lecture"
    return source.with_name(f"{stem} - {topic}.md")


def read_text_file(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in READ_ENCODINGS:
        try:
            return path.read_text(encoding=encoding).removeprefix("\ufeff")
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
    *,
    title: str = "",
    source: Path | None = None,
    overwrite: bool = True,
) -> None:
    normalized_summary = normalize_summary_text(summary_text)
    content = (
        f"## 요약\n\n{normalized_summary}\n\n"
        f"## 코넬 노트\n\n{cornell_notes_text.strip()}\n\n"
        f"## 전체 전사문\n\n{transcript_text.strip()}\n"
    )
    if title:
        content = f"# {title}\n\n" + content
    if source is not None:
        content = _source_marker(source) + "\n\n" + content
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
        if overwrite:
            temp_path.replace(output_path)
        else:
            os.link(temp_path, output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _log(message: str, *, verbose: bool = True, stream: object = sys.stdout) -> None:
    if verbose:
        print(message, file=stream)


def _local_config_path() -> Path:
    return Path.cwd() / CONFIG_FILENAME


def _global_config_path() -> Path:
    return GLOBAL_CONFIG_PATH.expanduser()


def _format_config_path(path: Path) -> str:
    home = Path.home()
    try:
        return f"~/{path.relative_to(home)}"
    except ValueError:
        return str(path)


def _create_default_global_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")


def _read_config_file(config_path: Path) -> dict[str, Any]:
    resolved_path = config_path.expanduser().resolve()

    try:
        with resolved_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"failed to read or parse {resolved_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a TOML table: {resolved_path}")
    return data


def _load_config_file(
    args: argparse.Namespace,
) -> tuple[Path | None, dict[str, Any], bool]:
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise ConfigError(f"config file does not exist: {config_path}")
        return config_path, _read_config_file(config_path), False

    local_path = _local_config_path()
    if local_path.exists():
        return local_path, _read_config_file(local_path), False

    global_path = _global_config_path()
    if global_path.exists():
        return global_path, _read_config_file(global_path), False

    try:
        _create_default_global_config(global_path)
    except OSError as exc:
        raise ConfigError(f"failed to create default config: {global_path}") from exc
    return global_path, _read_config_file(global_path), True


def _expect_table(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a table.")
    return value


def _copy_request_options(options: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in options.items():
        if isinstance(value, dict):
            copied[key] = _copy_request_options(value)
        else:
            copied[key] = value
    return copied


def _merge_request_options(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = _copy_request_options(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_request_options(merged[key], value)
        elif isinstance(value, dict):
            merged[key] = _copy_request_options(value)
        else:
            merged[key] = value
    return merged


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
    raw_request = table.get("request", {})
    if raw_request is None:
        raw_request = {}
    request_options = _expect_table(raw_request, f"{context}.request")
    options = _copy_request_options(request_options)
    flat_options = {key: table[key] for key in REQUEST_OPTIONS if key in table}
    options = _merge_request_options(options, flat_options)
    if "max_tokens" in options and "max_completion_tokens" in options:
        raise ConfigError(
            f"{context} cannot set both max_tokens and max_completion_tokens."
        )
    return options


def _openai_only_request_options(options: Mapping[str, Any]) -> list[str]:
    openai_only = set(options) & OPENAI_ONLY_REQUEST_OPTIONS
    return sorted(openai_only)


def _validate_request_options(
    *,
    provider_type: str,
    api: str,
    options: Mapping[str, Any],
    context: str,
) -> None:
    if "max_tokens" in options and "max_completion_tokens" in options:
        raise ConfigError(
            f"{context} cannot set both max_tokens and max_completion_tokens."
        )
    if "max_tokens" in options and "max_output_tokens" in options:
        raise ConfigError(
            f"{context} cannot set both max_tokens and max_output_tokens."
        )
    if "max_completion_tokens" in options and "max_output_tokens" in options:
        raise ConfigError(
            f"{context} cannot set both max_completion_tokens and max_output_tokens."
        )
    if api == "chat_completions" and "max_output_tokens" in options:
        raise ConfigError(
            f"{context} cannot use max_output_tokens with chat_completions API."
        )
    if api == "responses" and "max_tokens" in options:
        raise ConfigError(f"{context} cannot use max_tokens with responses API.")
    if api == "responses" and "max_completion_tokens" in options:
        raise ConfigError(
            f"{context} cannot use max_completion_tokens with responses API."
        )
    if provider_type == "compatible":
        openai_only = _openai_only_request_options(options)
        if openai_only:
            raise ConfigError(
                f"{context} uses OpenAI-only option(s) with compatible provider: "
                f"{', '.join(openai_only)}"
            )


def _normalize_provider_type(provider_type: object, context: str) -> str:
    if provider_type == "local":
        return "compatible"
    if provider_type in {"openai", "compatible"}:
        return str(provider_type)
    raise ConfigError(f"{context}.type must be 'openai', 'compatible', or 'local'.")


def _default_api_for_provider_type(provider_type: str) -> str:
    if provider_type == "openai":
        return "responses"
    return "chat_completions"


def _normalize_provider_api(
    api: object,
    *,
    provider_type: str,
    context: str,
) -> str:
    if api is None:
        return _default_api_for_provider_type(provider_type)
    if api in {"responses", "chat_completions"}:
        normalized_api = str(api)
        if provider_type == "compatible" and normalized_api == "responses":
            raise ConfigError(f"{context}.api='responses' requires type='openai'.")
        return normalized_api
    raise ConfigError(f"{context}.api must be 'responses' or 'chat_completions'.")


def _normalize_request_options_for_api(
    options: Mapping[str, Any],
    *,
    api: str,
    context: str,
) -> dict[str, Any]:
    normalized = _copy_request_options(options)
    if api == "responses":
        normalized.setdefault("store", False)
    return normalized


def _resolve_api_key(*, env_name: str | None) -> str | None:
    if env_name:
        return os.environ.get(env_name)
    return None


def _build_client_from_provider(provider: ProviderConfig) -> "OpenAI":
    from openai import OpenAI

    client_kwargs: dict[str, str] = {}
    if provider.api_key:
        client_kwargs["api_key"] = provider.api_key
    if provider.base_url:
        client_kwargs["base_url"] = provider.base_url
    return OpenAI(**client_kwargs)


def _parse_provider_configs(
    config_data: Mapping[str, Any],
) -> dict[str, ProviderConfig]:
    providers_table = _expect_table(config_data.get("providers", {}), "providers")
    if not providers_table:
        raise ConfigError("config file must define at least one provider.")

    providers: dict[str, ProviderConfig] = {}
    provider_keys = {"type", "api", "base_url", "api_key_env", "request"}
    for provider_name, raw_provider in providers_table.items():
        provider_table = _expect_table(raw_provider, f"providers.{provider_name}")
        provider_type = _normalize_provider_type(
            provider_table.get("type", "compatible"),
            f"providers.{provider_name}",
        )
        provider_api = _normalize_provider_api(
            provider_table.get("api"),
            provider_type=provider_type,
            context=f"providers.{provider_name}",
        )
        options = _request_options_from_table(
            provider_table,
            known_keys=provider_keys,
            context=f"providers.{provider_name}",
        )
        _validate_request_options(
            provider_type=provider_type,
            api=provider_api,
            options=options,
            context=f"providers.{provider_name}",
        )
        options = _normalize_request_options_for_api(
            options,
            api=provider_api,
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
            api=provider_api,
            base_url=base_url,
            api_key=_resolve_api_key(env_name=api_key_env),
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
    stage_keys = {"provider", "model", "request"}
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
        model = stage_table.get("model")
        if not isinstance(model, str) or not model:
            raise ConfigError(f"stages.{stage_name}.model must be a string.")

        provider = providers[provider_name]
        stage_options = _request_options_from_table(
            stage_table,
            known_keys=stage_keys,
            context=f"stages.{stage_name}",
        )
        request_options = _merge_request_options(
            provider.request_options,
            stage_options,
        )
        _validate_request_options(
            provider_type=provider.type,
            api=provider.api,
            options=request_options,
            context=f"stages.{stage_name}",
        )
        request_options = _normalize_request_options_for_api(
            request_options,
            api=provider.api,
            context=f"stages.{stage_name}",
        )
        stages[stage_name] = StageSettings(
            name=stage_name,
            provider_name=provider_name,
            model=model,
            api=provider.api,
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


def _has_full_cli_override(args: argparse.Namespace) -> bool:
    override_values = (args.model, args.api_key, args.base_url)
    provided = [value is not None for value in override_values]
    if any(provided) and not all(provided):
        raise ConfigError("--model, --api-key, and --base-url must be used together.")
    if all(provided) and not all(isinstance(value, str) and value for value in override_values):
        raise ConfigError("--model, --api-key, and --base-url cannot be empty.")
    return all(provided)


def _apply_cli_override(
    settings: PipelineSettings,
    args: argparse.Namespace,
) -> PipelineSettings:
    if not _has_full_cli_override(args):
        return settings

    provider = ProviderConfig(
        name="cli",
        type="compatible",
        api="chat_completions",
        base_url=args.base_url,
        api_key=args.api_key,
    )
    stages: dict[str, StageSettings] = {}
    for stage_name in STAGE_NAMES:
        stage = settings.stages[stage_name]
        request_options = _copy_request_options(stage.request_options)
        if request_options.get("store") is False:
            request_options.pop("store")
        _validate_request_options(
            provider_type=provider.type,
            api=provider.api,
            options=request_options,
            context=f"stages.{stage_name}",
        )
        stages[stage_name] = StageSettings(
            name=stage_name,
            provider_name=provider.name,
            model=args.model,
            api=provider.api,
            request_options=request_options,
        )
    return PipelineSettings(providers={provider.name: provider}, stages=stages)


def _resolve_pipeline_settings(
    args: argparse.Namespace,
) -> tuple[Path | None, PipelineSettings, bool]:
    config_path, config_data, created_config = _load_config_file(args)
    settings = _pipeline_settings_from_config(config_data, args)
    return config_path, _apply_cli_override(settings, args), created_config


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
            api=stage.api,
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

    try:
        previous_output = None if args.transcribe_only else existing_output(txt_path)
        if previous_output is not None:
            output_path = previous_output
            if not args.overwrite:
                return "skipped", f"{progress_prefix} skip existing -> {output_path}", None
        print(f"{progress_prefix} reading")
        if txt_path.suffix.lower() in AUDIO_SUFFIXES:
            if args.dry_run:
                target = transcript_cache_path(txt_path) if args.transcribe_only else output_path
                return "processed", f"{progress_prefix} would-transcribe -> {target}", None
            print(f"{progress_prefix} waiting for local transcription")
            with AUDIO_LOCK:
                raw_text = cached_transcription(
                    txt_path, model=args.asr_model,
                    language=None if args.language == "auto" else args.language,
                    cpu_threads=args.cpu_threads,
                    device=args.device,
                    on_stage=lambda stage: print(f"{progress_prefix} {stage}", flush=True),
                )
        else:
            raw_text = read_text_file(txt_path)
        if not raw_text.strip():
            return "skipped", f"{progress_prefix} skip empty", None

        if args.transcribe_only:
            return "processed", f"{progress_prefix} transcribed -> {transcript_cache_path(txt_path)}", None
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
        title = getattr(result, "title", "")
        if args.name_from_content and previous_output is None:
            output_path = content_output_path(txt_path, title)
        _log(f"{progress_prefix} writing markdown", verbose=args.verbose)
        write_markdown(
            output_path,
            summary_text=result.summary_text,
            cornell_notes_text=result.cornell_notes_text,
            transcript_text=result.formatted_transcript,
            title=title,
            source=txt_path,
            overwrite=args.overwrite and previous_output == output_path,
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


def _print_config_paths(args: argparse.Namespace) -> None:
    explicit_path = Path(args.config).expanduser().resolve() if args.config else None
    if explicit_path is not None:
        print(f"explicit: {explicit_path}")
    print(f"local: {_local_config_path()}")
    print(f"global: {_global_config_path()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_config_paths:
        _print_config_paths(args)
        return 0

    root = Path(args.path).expanduser().resolve()
    include_globs = args.include_glob or ["*.txt", *(f"*{suffix}" for suffix in sorted(AUDIO_SUFFIXES))]
    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir or [])

    if not root.exists():
        print(f"error: path does not exist: {root}", file=sys.stderr)
        return 2

    if args.limit is not None and args.limit < 0:
        print("error: --limit must be >= 0.", file=sys.stderr)
        return 2
    if args.cpu_threads is not None and args.cpu_threads < 1:
        print("error: --cpu-threads must be >= 1.", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print("error: --jobs must be >= 1.", file=sys.stderr)
        return 2
    if args.retries < 0:
        print("error: --retries must be >= 0.", file=sys.stderr)
        return 2
    if not math.isfinite(args.retry_backoff) or args.retry_backoff < 0:
        print("error: --retry-backoff must be finite and >= 0.", file=sys.stderr)
        return 2

    txt_files = discover_txt_files(root, include_globs, exclude_dirs) if root.is_dir() else [root]
    if args.compress_audio or args.transcribe_only:
        allowed = {".m4a"} if args.compress_audio else AUDIO_SUFFIXES
        if root.is_file() and root.suffix.lower() not in allowed:
            print(f"error: unsupported input for this mode: {root}", file=sys.stderr)
            return 2
        txt_files = [path for path in txt_files if path.suffix.lower() in allowed]
    unsupported = [path for path in txt_files if path.suffix.lower() not in AUDIO_SUFFIXES | {".txt"}]
    if unsupported:
        print(f"error: unsupported input: {unsupported[0]}", file=sys.stderr)
        return 2
    txt_files = select_inputs(txt_files)
    if args.limit is not None:
        txt_files = txt_files[: args.limit]
    if args.compress_audio:
        return _compress_files(txt_files, args)

    stage_configs: Mapping[str, StageConfig] | None = None
    if not args.transcribe_only:
        try:
            config_path, pipeline_settings, created_config = _resolve_pipeline_settings(args)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if created_config and config_path is not None:
            print(f"created default config -> {_format_config_path(config_path)}")
        if config_path is not None:
            _log(f"using config {config_path}", verbose=args.verbose)

    outputs: dict[Path, Path] = {}
    try:
        for path in txt_files:
            if not args.transcribe_only and not args.overwrite and should_skip(path):
                continue
            output = transcript_cache_path(path) if args.transcribe_only else path.with_suffix(".md")
            if output in outputs:
                print(f"error: output collision: {outputs[output]} and {path}; select one with --include-glob.", file=sys.stderr)
                return 2
            outputs[output] = path
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.dry_run and not args.transcribe_only and outputs:
        try:
            stage_configs = _build_stage_configs(pipeline_settings)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    counts = {"processed": 0, "skipped": 0, "errors": 0}
    retry_config = RetryConfig(
        retries=args.retries,
        backoff_seconds=args.retry_backoff,
    )

    print(f"searching {root}")
    print(f"found {len(txt_files)} matching file(s)")

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
