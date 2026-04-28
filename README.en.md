# lecture-notes

[한국어 README](README.md)

`lecture-notes` is a Python CLI that recursively finds lecture transcript `*.txt` files, runs them through a 4-step AI workflow, and writes polished Markdown notes next to the source files.

It is designed for cases where you already have raw transcript text and want:

- corrected transcript text
- readable paragraph formatting
- a compact summary for review
- detailed Cornell note-taking style notes
- Obsidian-friendly Markdown output

The CLI uses OpenAI's Python SDK. Official OpenAI providers use the Responses API by default, while OpenAI-compatible providers use the Chat Completions API.

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

## Quick Start

Set your model and API key:

```bash
export LECTURE_NOTES_MODEL="gpt-4o-mini"
export LECTURE_NOTES_API_KEY="your-api-key"
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

You can use OpenAI-compatible providers by setting a base URL and model name:

```bash
export LECTURE_NOTES_BASE_URL="https://your-openai-compatible-server/v1"
export LECTURE_NOTES_API_KEY="your-api-key"
export LECTURE_NOTES_MODEL="your-model-name"

lecture-notes ./lectures
```

`OPENAI_API_KEY` is also accepted as a fallback for OpenAI usage.

## Config File

Use `lecture-notes.toml` when different stages should use different models or API URLs. By default, the CLI reads `lecture-notes.toml` from the current working directory. Use `--config` to point at another file.

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
model = "gpt-5.1-mini"
temperature = 0.2
max_completion_tokens = 2000
reasoning_effort = "minimal"

[stages.cornell]
provider = "openai"
model = "gpt-5.1"
reasoning_effort = "medium"
```

Provider `type` can be:

- `openai`: the official OpenAI API. Uses the Responses API by default.
- `compatible`: an OpenAI-compatible Chat Completions server. OpenAI-only request options are rejected before any API call.
- `local`: an alias for `compatible`.

You can explicitly set `api = "responses"` or `api = "chat_completions"` on a provider. `responses` is only valid for `type = "openai"` providers. When the Responses API is used, `store = false` is applied by default.

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

Token limit rules:

- Chat Completions uses `max_tokens` or `max_completion_tokens`.
- Responses uses `max_output_tokens`.
- For OpenAI Responses providers, `max_completion_tokens` in the config is automatically converted to `max_output_tokens`.
- Mixing `max_tokens`, `max_completion_tokens`, and `max_output_tokens` in the same stage/provider is rejected.

OpenAI-only request options:

- `reasoning_effort`
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
model = "gpt-5.1-mini"
reasoning_effort = "minimal"
```

Run with:

```bash
lecture-notes ./lectures --profile fast
lecture-notes ./lectures --config ./my-lecture-notes.toml
```

## CLI Options

- `lecture-notes [PATH]`
- `--config <path>`
- `--profile <name>`
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

## How It Works

For each `*.txt` file under the target directory:

1. Correct transcription mistakes while preserving meaning.
2. Reformat the transcript into readable paragraphs.
3. Generate a compact summary focused on review-worthy points.
4. Generate detailed Cornell note-taking style notes that can stand in for reading the full transcript.

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
- Progress is printed per file even without `--verbose`.
- `--verbose` adds per-stage pipeline logs.
- Output is written through a temporary file and renamed into place.
- `--jobs` processes multiple files concurrently while keeping the 4 stages inside each file sequential.
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

### 단서 / 질문
- ...

### 필기
- ...

### 요약
- ...

## 전체 전사문

...
```

## Environment Variables

- `LECTURE_NOTES_MODEL`: default model name
- `LECTURE_NOTES_API_KEY`: API key for OpenAI or an OpenAI-compatible server
- `LECTURE_NOTES_BASE_URL`: base URL for an OpenAI-compatible server
- `OPENAI_API_KEY`: fallback API key for OpenAI

Config-file providers use `api_key_env` to name the environment variable that contains the API key.

For simple runs without a config file, CLI arguments take precedence over environment variables.

## Development

Run tests:

```bash
python -m unittest discover -s tests
```

## License

MIT. See [LICENSE](LICENSE).
