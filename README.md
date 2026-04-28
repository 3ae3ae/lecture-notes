# lecture-notes

[English README](README.en.md)

`lecture-notes`는 강의 전사 `*.txt` 파일을 재귀적으로 찾아 4단계 AI 워크플로우로 처리한 뒤, 원본 파일 옆에 정리된 Markdown 노트를 생성하는 Python CLI입니다.

다음과 같은 용도에 맞춰져 있습니다.

- 전사 오류 교정
- 읽기 쉬운 문단 구조로 정리
- 복습용 핵심 요약 생성
- 코넬 노트테이킹법 기반 상세 필기본 생성
- Obsidian 친화적인 Markdown 출력

내부적으로 OpenAI Python SDK를 사용하며, OpenAI와 OpenAI 호환 Chat Completions API 서버를 지원합니다.

## 설치

GitHub에서 바로 설치:

```bash
uv tool install git+https://github.com/3ae3ae/lecture-notes.git
```

기존 설치 갱신:

```bash
uv tool install --refresh git+https://github.com/3ae3ae/lecture-notes.git
```

로컬 개발 환경에서 설치:

```bash
uv tool install .
```

## 빠른 시작

모델과 API 키를 먼저 설정합니다.

```bash
export LECTURE_NOTES_MODEL="gpt-4o-mini"
export LECTURE_NOTES_API_KEY="your-api-key"
```

현재 디렉터리 기준으로 실행:

```bash
lecture-notes
```

특정 폴더를 지정해 실행:

```bash
lecture-notes ./lectures
```

API 호출 없이 처리 대상만 확인:

```bash
lecture-notes ./lectures --dry-run
```

단계별 진행 로그까지 보기:

```bash
lecture-notes ./lectures --verbose
```

## OpenAI 호환 서버 사용

OpenAI 호환 제공자를 사용할 때는 base URL과 모델명을 함께 설정하면 됩니다.

```bash
export LECTURE_NOTES_BASE_URL="https://your-openai-compatible-server/v1"
export LECTURE_NOTES_API_KEY="your-api-key"
export LECTURE_NOTES_MODEL="your-model-name"

lecture-notes ./lectures
```

OpenAI를 사용할 때는 `OPENAI_API_KEY`도 fallback으로 사용할 수 있습니다.

## 설정파일 사용

복수 모델이나 복수 API URL을 함께 쓰려면 `lecture-notes.toml`을 사용합니다. 기본 실행 시 현재 작업 디렉터리의 `lecture-notes.toml`을 자동으로 읽고, 다른 파일은 `--config`로 지정할 수 있습니다.

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

Provider `type`은 다음 둘 중 하나입니다.

- `openai`: OpenAI 공식 API. OpenAI 전용 요청 인자를 사용할 수 있습니다.
- `compatible`: OpenAI 호환 Chat Completions 서버. OpenAI 전용 요청 인자를 사용하면 API 호출 전에 오류로 중단합니다.

공통 요청 인자:

- `temperature`
- `top_p`
- `max_tokens`
- `max_completion_tokens`
- `presence_penalty`
- `frequency_penalty`
- `seed`
- `timeout`

OpenAI 전용 요청 인자:

- `reasoning_effort`
- `service_tier`
- `prompt_cache_key`
- `prompt_cache_retention`
- `store`
- `metadata`
- `safety_identifier`

Profile이 필요하면 `[profiles.<name>.stages]` 아래에 단계 설정을 둘 수 있습니다.

```toml
[profiles.fast.stages.summary]
provider = "openai"
model = "gpt-5.1-mini"
reasoning_effort = "minimal"
```

실행:

```bash
lecture-notes ./lectures --profile fast
lecture-notes ./lectures --config ./my-lecture-notes.toml
```

## CLI 옵션

- `lecture-notes [PATH]`
- `--config <path>`
- `--profile <name>`
- `--model <name>`
- `--api-key <key>`
- `--base-url <url>`
- `--include-glob <pattern>` 반복 가능
- `--exclude-dir <name>` 반복 가능
- `--dry-run`
- `--verbose`
- `--fail-fast`
- `--overwrite`
- `--limit <n>`
- `--jobs <n>`
- `--retries <n>`
- `--retry-backoff <seconds>`

## 동작 방식

대상 디렉터리 아래의 각 `*.txt` 파일에 대해 다음 순서로 처리합니다.

1. 전사 오류를 교정하되 의미를 최대한 보존합니다.
2. 전사문을 읽기 쉬운 문단 구조로 재정리합니다.
3. 복습에 유용한 핵심 요약을 생성합니다.
4. 전체 전사문을 대체해 복습할 수 있는 코넬 노트 형식의 필기본을 생성합니다.

그 다음 같은 basename의 Markdown 파일을 원본 옆에 저장합니다.

- `lecture.txt` -> `lecture.md`
- `lecture.md`가 이미 있으면 해당 `txt`는 건너뜁니다.

기본 제외 디렉터리:

- `.git`
- `.venv`
- `node_modules`
- `__pycache__`

텍스트 디코딩 fallback 순서:

- `utf-8`
- `utf-8-sig`
- `cp949`

추가 동작:

- 한글 파일명과 공백이 포함된 파일명을 지원합니다.
- `--verbose` 없이도 파일 단위 진행 상황을 출력합니다.
- `--verbose`를 주면 파이프라인 단계별 로그를 추가로 출력합니다.
- 출력 파일은 임시 파일에 먼저 쓴 뒤 원자적으로 교체합니다.
- `--jobs`를 주면 여러 파일을 병렬 처리하되, 한 파일 내부의 4단계 순서는 유지합니다.
- 일시적인 timeout, rate limit, 5xx 오류는 `--retries`와 `--retry-backoff` 설정에 따라 재시도합니다.

## 출력 형식

생성되는 Markdown은 Obsidian에서 보기 좋도록 대괄호 라벨 대신 헤딩을 사용합니다.

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

## 환경 변수

- `LECTURE_NOTES_MODEL`: 기본 모델명
- `LECTURE_NOTES_API_KEY`: OpenAI 또는 OpenAI 호환 서버용 API 키
- `LECTURE_NOTES_BASE_URL`: OpenAI 호환 서버 base URL
- `OPENAI_API_KEY`: OpenAI 사용 시 fallback API 키

설정파일 provider에서는 `api_key_env` 값으로 사용할 환경변수 이름을 직접 지정합니다.

설정파일을 쓰지 않는 단순 실행에서는 CLI 인자가 환경변수보다 우선합니다.

## 개발

테스트 실행:

```bash
python -m unittest discover -s tests
```

## 라이선스

MIT. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
