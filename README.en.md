# lecture-notes

[한국어 README](README.md)

`lecture-notes` is a Python CLI that recursively finds lecture transcript `*.txt` files, runs them through a 3-step AI workflow, and writes polished Markdown notes next to the source files.

It is designed for cases where you already have raw transcript text and want:

- corrected transcript text
- readable paragraph formatting
- a compact summary for review
- Obsidian-friendly Markdown output

The CLI uses OpenAI's Python SDK and works with both OpenAI and OpenAI-compatible Chat Completions APIs.

## Install

Install directly from GitHub with `uv tool`:

```bash
uv tool install git+https://github.com/3ae3ae/lecture_notes.git
```

Update an existing install:

```bash
uv tool install --refresh git+https://github.com/3ae3ae/lecture_notes.git
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

## CLI Options

- `lecture-notes [PATH]`
- `--model <name>`
- `--api-key <key>`
- `--base-url <url>`
- `--include-glob <pattern>` repeatable
- `--exclude-dir <name>` repeatable
- `--dry-run`
- `--verbose`
- `--fail-fast`

## How It Works

For each `*.txt` file under the target directory:

1. Correct transcription mistakes while preserving meaning.
2. Reformat the transcript into readable paragraphs.
3. Generate a compact summary focused on review-worthy points.

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

## Output Format

Generated Markdown is Obsidian-friendly and uses headings instead of bracketed labels:

```md
## 요약

### 핵심 요약
- ...

### 교수님 강조 포인트
- ...

## 전체 전사문

...
```

## Environment Variables

- `LECTURE_NOTES_MODEL`: default model name
- `LECTURE_NOTES_API_KEY`: API key for OpenAI or an OpenAI-compatible server
- `LECTURE_NOTES_BASE_URL`: base URL for an OpenAI-compatible server
- `OPENAI_API_KEY`: fallback API key for OpenAI

CLI arguments take precedence over environment variables.

## Development

Run tests:

```bash
python -m unittest discover -s tests
```

## License

MIT. See [LICENSE](LICENSE).
