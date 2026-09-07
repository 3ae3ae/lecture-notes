# lecture-notes

[한국어 README](README.md)

`lecture-notes` is a Python CLI that recursively finds recordings and lecture transcript `*.txt` files, runs them through a 4-step AI workflow, and writes polished Markdown notes next to the source files.

It is designed for cases where you already have raw transcript text and want:

- corrected transcript text
- readable paragraph formatting
- a compact summary for review
- detailed Markdown table-based Cornell note-taking style notes
- Obsidian-friendly Markdown output

The CLI uses OpenAI's Python SDK. Official OpenAI providers use the Responses API by default, while OpenAI-compatible providers use the Chat Completions API.

## CPU processing speed

After Silero changes Torch to one thread, the original thread count is restored immediately after ASR. The CPU thread count used for alignment/diarization is logged; override it with, for example, `--cpu-threads 8`. `--jobs` controls concurrent files/API work, while `--cpu-threads` controls local model CPU parallelism. Speedup is not necessarily proportional to thread count; the ASR model remains `large-v3`.

## Unified workflow

Use `lecture-notes` instead of the previous `transcribe.sh` and `compress_audio.sh` scripts.

```bash
# Install/update from this repository on Apple Silicon
uv tool install --force --python 3.12 ".[audio]"

# Recordings → speaker transcripts → notes
lecture-notes /Users/3ae/Github/obsidian --name-from-content

# Preview without transcription, API calls, or compression
lecture-notes /Users/3ae/Github/obsidian --dry-run

# Transcription only, without LLM configuration or credentials
lecture-notes /Users/3ae/Github/obsidian --transcribe-only

# Compression only: replace M4A originals with mono AAC at 64 kbps
lecture-notes /Users/3ae/Github/obsidian --compress-audio
```

Defaults are `large-v3` and Korean (`ko`). Select Turbo with `--asr-model large-v3-turbo`, or language detection with `--language auto`. Professor/student roles are inferred during note generation; transcription-only output contains acoustic speaker IDs.

When a recording and legacy TXT share a basename, audio takes priority and the TXT remains untouched. Use `--include-glob '*.txt'` to process existing text instead. Multiple recordings sharing a basename require selecting one for note generation. Existing Markdown is skipped unless `--overwrite` is set.

Speaker transcripts are cached in `recording.m4a.transcript.json`, keyed by source size/mtime, ASR model, and language. Reruns reuse the transcript after note-generation failures. Source/config changes invalidate it. `--overwrite` regenerates notes; delete the cache to force retranscription. Existing TXT files are never automatically overwritten.

Compression is a separate, M4A-only operation. Inputs at or below 69 kbps are skipped. Originals are replaced only after successful conversion, duration validation, and a size reduction. This is lossy compression; transcribe first when needed. Neither standalone mode reads LLM configuration.

Audio dependencies constrain TorchCodec to the newest compatible series, 0.7, for WhisperMLX 3.13.1 and its required Torch 2.8. uv cannot infer undeclared binary compatibility. Run the install/update command above after active jobs finish.

## Install

Install directly from GitHub with `uv tool`:

```bash
uv tool install git+https://github.com/3ae3ae/lecture-notes.git
```

Update an existing install:

```bash
uv tool install --refresh git+https://github.com/3ae3ae/lecture-notes.git
```

Local development install:

```bash
uv tool install .
```

## Speaker-aware notes from recordings

[whispermlx](https://github.com/KalebJS/whispermlx) performs local transcription, word alignment, and speaker diarization. Audio support requires Apple Silicon macOS and Python 3.11–3.13. Text-only usage does not require audio dependencies.

```bash
brew install ffmpeg
uv tool install ".[audio]" --python 3.12
export HF_TOKEN="your-huggingface-token"
lecture-notes "./lecture recording.m4a" --language en
lecture-notes ./lectures --dry-run
```

Accept the [pyannote model agreement](https://huggingface.co/pyannote/speaker-diarization-community-1) and provide a token with access. Models download on first use. Configure the LLM provider as described below as well; transcript text is sent to that provider for note generation.

- Discovery includes `.txt`, `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.mp4`, `.mkv`, and `.webm`, case-insensitively. Single file paths are supported.
- `--asr-model large-v3` selects the transcription model; `--model` remains an LLM option.
- `--language en` selects English; default is Korean. Use `--language auto` for automatic detection.
- Speaker count is inferred, not fixed at two. Word-level speaker changes are retained.
- The correction LLM infers professor/student roles from classroom context, preserving speaker IDs and marking roles as inferred (e.g. `[교수 추정 · SPEAKER_00]`). Insufficient evidence yields `역할 미상` (unknown role). Diarization and role inference can be wrong. Plain txt without speaker information cannot recover actual speaker identities.
- Prompts preserve timestamps, speaker IDs, and question/answer boundaries, and distinguish student guesses from instructor explanations.
- Audio takes priority over matching TXT. Multiple recordings sharing an output basename cause a note collision error. Select one using, for example, `--include-glob '*.m4a'`.
- Only local audio transcription is serialized; completed transcripts proceed through LLM stages while the next recording is transcribed. Dry runs do not load/download models or transcribe audio.
- Output is a sibling `.md`; existing results are skipped unless `--overwrite` is supplied. Token-truncated LLM responses fail without saving partial notes.

Real transcription and role accuracy depend on recording quality and model behavior. Split recordings that exceed the LLM context window; automatic chunking is not implemented.

## Quick Start

On first use, the CLI creates a global config file when none exists. The default config references the `OPENAI_API_KEY` environment variable.

```bash
lecture-notes ./lectures --dry-run
export OPENAI_API_KEY="your-api-key"
```

Run in the current directory:

```bash
lecture-notes
```

Run on a specific folder:

```bash
lecture-notes ./lectures
```

Preview targets without calling the API:

```bash
lecture-notes ./lectures --dry-run
```

Show step-by-step progress:

```bash
lecture-notes ./lectures --verbose
```

## OpenAI-Compatible Servers

You can use OpenAI-compatible providers by setting a provider base URL and stage model in the config:

```toml
[providers.local]
type = "compatible"
base_url = "https://your-openai-compatible-server/v1"
api_key_env = "LECTURE_NOTES_API_KEY"

[stages.correction]
provider = "local"
model = "your-model-name"
```

Then set the environment variable referenced by `api_key_env` and run the CLI:

```bash
export LECTURE_NOTES_API_KEY="your-api-key"
lecture-notes ./lectures
```

## Config File

Use `lecture-notes.toml` when different stages should use different models or API URLs. Config files are checked in this order:

1. The file passed with `--config`
2. `lecture-notes.toml` in the current working directory
3. The user-level config at `~/.config/lecture-notes/config.toml`

If neither local nor global config exists, the CLI creates the global default config on first use. This makes `uv tool install` usage work without manually copying a config file.

Inspect paths:

```bash
lecture-notes --print-config-paths
```

```toml
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[providers.local]
type = "compatible"
base_url = "http://localhost:1234/v1"
api_key_env = "LECTURE_NOTES_API_KEY"

[stages.correction]
provider = "local"
model = "qwen-transcriber"

[stages.formatting]
provider = "local"
model = "qwen-transcriber"

[stages.summary]
provider = "openai"
model = "gpt-5.6-luna"
max_output_tokens = 2000

[stages.summary.request.reasoning]
effort = "low"

[stages.cornell]
provider = "openai"
model = "gpt-5.6-terra"

[stages.cornell.request.reasoning]
effort = "medium"
```

Provider `type` can be:

- `openai`: the official OpenAI API. Uses the Responses API by default.
- `compatible`: an OpenAI-compatible Chat Completions server. OpenAI-only request options are rejected before any API call.
- `local`: an alias for `compatible`.

You can explicitly set `api = "responses"` or `api = "chat_completions"` on a provider. `responses` is only valid for `type = "openai"` providers. When the Responses API is used, `store = false` is applied by default.

Providers and stages can define a nested `[...request]` table. These values are passed to the OpenAI Python SDK as `responses.create(..., **request)` or `chat.completions.create(..., **request)`. Provider request values act as defaults, and stage request values override matching keys.

```toml
[providers.openai.request.metadata]
app = "lecture-notes"

[stages.summary]
provider = "openai"
model = "gpt-5.6-terra"
max_output_tokens = 8000
service_tier = "flex"

[stages.summary.request.reasoning]
effort = "medium"
```

Common request options:

- `temperature`
- `top_p`
- `max_tokens`
- `max_completion_tokens`
- `max_output_tokens`
- `presence_penalty`
- `frequency_penalty`
- `seed`
- `timeout`

Only set model-specific options such as `temperature` when the selected model or compatible server supports them.

Common OpenAI Responses provider options:

```toml
[stages.summary]
provider = "openai"
model = "gpt-5.6-terra"
max_output_tokens = 8000
service_tier = "flex"
store = false

[stages.summary.request.reasoning]
effort = "medium"
```

Common OpenAI-compatible Chat Completions provider options:

```toml
[stages.cornell]
provider = "local"
model = "your-model-name"
max_completion_tokens = 12000
temperature = 0.2
top_p = 0.9
presence_penalty = 0.1
frequency_penalty = 0.1
seed = 42
timeout = 120
```

Token limit rules:

- Chat Completions uses `max_tokens` or `max_completion_tokens`.
- Responses uses `max_output_tokens`.
- Responses providers reject `max_tokens` and `max_completion_tokens`.
- Mixing `max_tokens`, `max_completion_tokens`, and `max_output_tokens` in the same stage/provider is rejected.

OpenAI-only request options:

- `reasoning`
- `service_tier`
- `prompt_cache_key`
- `prompt_cache_retention`
- `store`
- `metadata`
- `safety_identifier`

Profiles can override stage settings under `[profiles.<name>.stages]`.

```toml
[profiles.fast.stages.summary]
provider = "openai"
model = "gpt-5.6-luna"

[profiles.fast.stages.summary.request.reasoning]
effort = "low"
```

Run with:

```bash
lecture-notes ./lectures --profile fast
lecture-notes ./lectures --config ./my-lecture-notes.toml
```

## Content-based titles, filenames, and study quality

The summary stage also generates a descriptive lecture title, shown at the top of the note, without an additional API call.

```bash
lecture-notes "./Recording 01.m4a" --name-from-content --language en
# Example: Recording 01 - Correlation and causation.md
```

`--name-from-content` names new notes `source name - topic.md`. Source files remain unchanged. Without the flag, the existing `source name.md` convention remains. Missing titles fall back to the source name. Unsafe filename and Obsidian link characters are removed, and UTF-8 filename length is bounded.

A hidden source identifier on the first line lets reruns find and skip generated notes. `--overwrite` updates the existing file while retaining its name to preserve links. Removing that comment or renaming the source breaks the association. Collisions with unrelated notes fail without overwriting them.

Summaries connect concepts, reasons, conditions, and exceptions; preserve figures, units, exam inclusions/exclusions, and instructor corrections; and include assignment/schedule and actual Q&A sections when relevant. Cornell notes use recall questions and adapt row count to the material. Evidence timestamps are included only when present in the source.

Defaults use `gpt-5.6-luna` for correction/formatting and `gpt-5.6-terra` for summaries/Cornell notes. Local `lecture-notes.toml` takes precedence over global configuration.

The synthetic Korean lecture at `tests/fixtures/lecture_quality.txt` exercises exam exclusions, an assignment correction (2→3 examples), an unanswered group-submission question, and late-lecture absolute/relative risk and percentage-point distinctions. Run it with a real provider to inspect these criteria; success on this example does not establish accuracy on all lectures.

## CLI Options

- `lecture-notes [PATH]`
- `--config <path>`
- `--print-config-paths`
- `--profile <name>`
- `--asr-model <name>`
- `--cpu-threads <n>`
- `--name-from-content`
- `--transcribe-only`
- `--compress-audio`
- `--language <code>`
- `--model <name>`
- `--api-key <key>`
- `--base-url <url>`
- `--include-glob <pattern>` repeatable
- `--exclude-dir <name>` repeatable
- `--dry-run`
- `--verbose`
- `--fail-fast`
- `--overwrite`
- `--limit <n>`
- `--jobs <n>`
- `--retries <n>`
- `--retry-backoff <seconds>`

`--model`, `--api-key`, and `--base-url` only work when all three are provided together. In that mode, every stage temporarily uses one OpenAI-compatible provider. Supplying only some of the three options is rejected.

## How It Works

Recordings are transcribed and diarized with whispermlx, then the resulting transcripts and existing `*.txt` files follow these stages:

1. Correct transcription mistakes while preserving meaning.
2. Reformat the transcript into readable paragraphs.
3. Generate a compact summary focused on review-worthy points.
4. Generate detailed Markdown table-based Cornell note-taking style notes that can stand in for reading the full transcript.

The tool then writes a sibling Markdown file with the same basename:

- `lecture.txt` -> `lecture.md`
- if `lecture.md` already exists, that `txt` file is skipped

Default excluded directories:

- `.git`
- `.venv`
- `node_modules`
- `__pycache__`

Text decoding fallback order:

- `utf-8`
- `utf-8-sig`
- `cp949`

Additional behavior:

- Korean filenames and filenames with spaces are supported.
- Per-file progress, word-alignment/diarization percentages, and elapsed time are printed without `--verbose`. A heartbeat every 30 seconds reports the last known percentage while processing; it does not imply additional progress. Model loading is logged separately.
- `--verbose` adds per-stage pipeline logs.
- Output is written through a temporary file and renamed into place.
- `--jobs` controls concurrent files (default: 4). Within each file, correction precedes formatting, then summary and Cornell generation run concurrently. At most two LLM requests per file can run at once (up to 8 by default). Even `--jobs 1` permits parallel summary/Cornell requests.
- Transient timeout, rate limit, and 5xx errors are retried according to `--retries` and `--retry-backoff`.

## Output Format

Generated Markdown is Obsidian-friendly and uses headings instead of bracketed labels:

```md
## 요약

### 핵심 요약
- ...

### 교수님 강조 포인트
- ...

## 코넬 노트

### 코넬 노트

| 단서 / 질문 | 필기 |
|---|---|
| ... | ... |

| 요약 |
|---|
| ... |

## 전체 전사문

...
```

## Environment Variables

Environment variables are not automatic fallbacks. The CLI reads an environment variable only when a config provider explicitly references it with `api_key_env = "OPENAI_API_KEY"`.

Example:

```toml
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"
```

```bash
export OPENAI_API_KEY="your-api-key"
```

Config files are selected in this order: `--config`, `lecture-notes.toml` in the current working directory, then the global config.

When installed with `uv tool`, the global config still lives at `~/.config/lecture-notes/config.toml`. Put `lecture-notes.toml` in a lecture folder when that folder needs settings that override the global config.

## Failure handling and reruns

- Empty batches, batches with all outputs already present, and `--limit 0` finish without requiring API credentials or creating API clients.
- Output collisions are checked before API initialization. Already completed inputs that will be skipped do not cause collisions.
- Failed Markdown saves preserve existing output and remove temporary files.
- Unreadable or invalid config files produce an error identifying the path.
- Only connection errors, timeouts, and HTTP 408, 429, and 5xx responses are retried. Authentication and ordinary processing errors are not retried.
- `--retry-backoff` must be finite and nonnegative.
- Exit codes: success `0`, file processing failure `1`, input/config error `2`. By default, remaining files continue after a file fails.

## Development

Run tests:

```bash
python -m unittest discover -s tests
```

## License

MIT. See [LICENSE](LICENSE).
