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
model = "gpt-5.4-mini"
temperature = 0.2
max_output_tokens = 2000

[stages.summary.request.reasoning]
effort = "minimal"

[stages.cornell]
provider = "openai"
model = "gpt-5.4"

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
model = "gpt-5.4"
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
model = "gpt-5.4-mini"

[profiles.fast.stages.summary.request.reasoning]
effort = "minimal"
```

Run with:

```bash
lecture-notes ./lectures --profile fast
lecture-notes ./lectures --config ./my-lecture-notes.toml
```

## CLI Options

- `lecture-notes [PATH]`
- `--config <path>`
- `--print-config-paths`
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

`--model`, `--api-key`, and `--base-url` only work when all three are provided together. In that mode, every stage temporarily uses one OpenAI-compatible provider. Supplying only some of the three options is rejected.

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

## Development

Run tests:

```bash
python -m unittest discover -s tests
```

## License

MIT. See [LICENSE](LICENSE).
