# lecture-notes

[English README](README.en.md)

`lecture-notes`는 강의 전사 녹음·영상과 전사 `*.txt` 파일을 재귀적으로 찾아 4단계 AI 워크플로우로 처리한 뒤, 원본 파일 옆에 정리된 Markdown 노트를 생성하는 Python CLI입니다.

다음과 같은 용도에 맞춰져 있습니다.

- 전사 오류 교정
- 읽기 쉬운 문단 구조로 정리
- 복습용 핵심 요약 생성
- 코넬 노트테이킹법 기반 상세 필기본 생성
- Obsidian 친화적인 Markdown 출력

내부적으로 OpenAI Python SDK를 사용합니다. OpenAI 공식 API는 기본적으로 Responses API를 사용하고, OpenAI 호환 서버는 Chat Completions API를 사용합니다.

## Apple GPU 사용

기본 `--device auto`는 MPS를 사용할 수 있으면 단어 정렬 모델과 화자 구분 모델을 Apple GPU에 올립니다. MPS가 없으면 CPU를 선택합니다. 실제 선택된 장치는 로그에 표시합니다. Whisper 전사는 이 옵션과 무관하게 MLX GPU를 사용하고, Silero 음성 탐지는 CPU를 사용합니다.

MPS에서 지원하지 않는 연산이나 메모리 오류가 생기면 `--device cpu`로 재실행하세요. 실행 중 오류에 대한 자동 CPU 재시도는 하지 않습니다. `--device mps`는 MPS 사용을 명시적으로 요구합니다. 정렬 후처리·군집화 등 CPU 작업은 남으며, 속도와 정확도의 비교 테스트는 수행하지 않았습니다. 장치 변경만으로 기존 전사 캐시는 무효화하지 않습니다.

## CPU 처리 속도

Silero 음성 탐지가 PyTorch를 1스레드로 바꿔도 전사 직후 원래 스레드 수를 복원합니다. 단어 정렬·화자 구분에 사용할 CPU 스레드 수는 로그에 표시되며, `--cpu-threads 8`처럼 지정할 수도 있습니다. `--jobs`는 API/파일 병렬도, `--cpu-threads`는 로컬 모델의 CPU 연산 병렬도입니다. 스레드 수에 비례하는 속도 향상을 보장하지는 않으며, 전사 모델은 `large-v3`를 유지합니다.

## 개인용 통합 사용법

이제 `transcribe.sh`와 `compress_audio.sh` 대신 `lecture-notes` 하나를 사용합니다.

```bash
# 저장소에서 설치/갱신 (Apple Silicon Mac)
uv tool install --force --python 3.12 ".[audio]"

# Obsidian 폴더의 녹음 → 화자 전사 → 요약·코넬 노트
lecture-notes /Users/3ae/Github/obsidian --name-from-content

# 처리 대상만 확인 (전사·API 호출·압축 없음)
lecture-notes /Users/3ae/Github/obsidian --dry-run

# 전사 결과만 저장 (LLM 설정/API 키 불필요)
lecture-notes /Users/3ae/Github/obsidian --transcribe-only

# 용량 정리만 실행: M4A 원본을 모노 AAC 64 kbps로 교체
lecture-notes /Users/3ae/Github/obsidian --compress-audio
```

기본 모델은 `large-v3`, 기본 언어는 한국어(`ko`)입니다. Turbo는 `--asr-model large-v3-turbo`, 언어 자동 감지는 `--language auto`로 선택합니다. 교수·학생 역할 추론은 노트 생성 단계에서 수행하며, 전사 전용 결과에는 음성 화자 ID가 남습니다.

녹음과 같은 basename의 기존 TXT가 함께 있으면 녹음을 우선하고 TXT는 보존합니다. 기존 TXT만 쓰려면 `--include-glob '*.txt'`를 지정하세요. 같은 basename의 녹음이 여러 형식으로 있으면 노트 생성 시 하나를 선택해야 합니다. 기존 Markdown은 계속 건너뛰며, 다시 만들 때만 `--overwrite`를 사용합니다.

전사 결과는 `녹음.m4a.transcript.json`에 원본의 크기·수정시각, 전사 모델·언어와 함께 저장합니다. 노트 생성이 실패해도 재실행 시 동일한 전사를 재사용합니다. 원본이나 설정이 바뀌면 새로 전사합니다. `--overwrite`는 노트만 다시 만들며, 전사 자체를 강제로 다시 하려면 해당 캐시 파일을 삭제하세요. 기존 TXT는 자동으로 덮어쓰지 않습니다.

M4A는 기본으로 전사 전에 모노 AAC 64 kbps로 압축합니다. 이미 69 kbps 이하면 건너뛰며, 변환 성공·오디오 길이·파일 크기를 확인한 후에만 원본을 교체합니다. 압축을 건너뛰려면 `--no-compress-audio`를 사용하세요. `--compress-audio`는 전사 없이 M4A만 정리하는 전용 모드입니다. 손실 압축이므로 원본 보관이 필요하면 이 기본값을 끄세요. 두 전용 모드 모두 LLM 설정을 읽지 않습니다.

오디오 의존성은 WhisperMLX 3.13.1의 Torch 2.8 요구에 맞춰 TorchCodec을 최신 호환 계열인 0.7로 제한합니다. uv도 패키지에 명시되지 않은 바이너리 호환성까지 자동으로 판별하지는 않습니다. 실행 중인 작업이 끝난 뒤 위 설치/갱신 명령을 실행하세요.

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

## 녹음에서 교수·학생을 구분해 노트 만들기

[whispermlx](https://github.com/KalebJS/whispermlx)를 사용해 로컬에서 전사, 단어 시간 정렬, 화자 구분을 수행합니다. 오디오 기능은 Apple Silicon Mac과 Python 3.11–3.13이 필요합니다. 기존 txt 처리에는 오디오 의존성이 필요하지 않습니다.

```bash
brew install ffmpeg
uv tool install ".[audio]" --python 3.12
export HF_TOKEN="your-huggingface-token"
lecture-notes "./강의 녹음.m4a" --language ko
lecture-notes ./lectures --dry-run
```

Hugging Face에서 [pyannote 모델 이용 조건](https://huggingface.co/pyannote/speaker-diarization-community-1)에 동의하고 접근 가능한 토큰을 설정하세요. 최초 실행에는 모델 다운로드가 필요합니다. 노트 생성에는 아래의 LLM provider 설정도 필요하며, 전사 텍스트는 설정한 provider로 전달됩니다.

- 기본 탐색: `.txt`, `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.mp4`, `.mkv`, `.webm` (대소문자 무관). 단일 파일 경로도 지원합니다.
- `--asr-model large-v3`: 전사 모델 선택. `--model`은 기존 LLM 옵션입니다.
- `--language ko`: 기본값은 한국어. `--language auto`로 자동 감지를 선택합니다.
- 화자 수를 2명으로 고정하지 않습니다. 단어 단위로 화자 전환을 보존합니다.
- 교정 LLM이 수업 문맥으로 `[교수 추정 · SPEAKER_00]`, `[학생 추정 · SPEAKER_01]` 역할을 자동 표시합니다. 음성 화자 구분과 역할 추론은 오류가 가능하며, 불확실하면 `역할 미상`으로 남깁니다. 화자 정보가 없는 txt에서 실제 화자를 복원하지는 않습니다.
- 시간과 화자 ID를 전사문에 보존하고 학생 질문과 교수 답변을 구분하도록 요청합니다. 학생의 추측을 교수의 확정된 설명으로 취급하지 않도록 합니다.
- 녹음과 TXT는 녹음을 우선합니다. 같은 basename의 녹음이 여러 개면 노트 출력 충돌 오류를 냅니다. `--include-glob '*.m4a'` 등으로 하나를 선택하세요.
- 로컬 음성 전사만 한 번에 하나씩 실행하며, 전사가 끝난 파일의 LLM 작업은 다음 전사와 병렬 진행합니다. `--dry-run`은 모델 로드·다운로드·전사를 하지 않습니다.
- 결과는 원본 옆 `.md`에 저장하며, 기존 결과는 `--overwrite` 없이는 건너뜁니다. LLM 응답이 토큰 제한으로 잘리면 저장하지 않고 오류로 처리합니다.

실제 녹음의 전사·역할 판별 품질은 녹음 상태와 모델에 따라 달라집니다. 긴 강의가 LLM 문맥 한도를 넘으면 녹음을 나누어 처리하세요. 자동 청킹은 제공하지 않습니다.

## 빠른 시작

처음 실행하면 전역 설정파일이 없을 때 자동으로 생성합니다. 기본 설정은 `OPENAI_API_KEY` 환경변수를 참조합니다.

```bash
lecture-notes ./lectures --dry-run
export OPENAI_API_KEY="your-api-key"
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

```toml
[providers.local]
type = "compatible"
base_url = "https://your-openai-compatible-server/v1"
api_key_env = "LECTURE_NOTES_API_KEY"

[stages.correction]
provider = "local"
model = "your-model-name"
```

그 다음 `api_key_env`가 참조하는 환경변수에 API 키를 넣고 실행합니다.

```bash
export LECTURE_NOTES_API_KEY="your-api-key"
lecture-notes ./lectures
```

## 설정파일 사용

복수 모델이나 복수 API URL을 함께 쓰려면 `lecture-notes.toml`을 사용합니다. 설정파일 탐색 순서는 다음과 같습니다.

1. `--config`로 지정한 파일
2. 현재 작업 디렉터리의 `lecture-notes.toml`
3. 사용자 전역 설정 `~/.config/lecture-notes/config.toml`

로컬/전역 설정이 모두 없으면 첫 실행 시 전역 기본 설정파일을 자동 생성합니다. 그래서 `uv tool install`로 설치한 뒤에도 별도 복사 없이 바로 사용할 수 있습니다.

확인:

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

Provider `type`은 다음 값을 지원합니다.

- `openai`: OpenAI 공식 API. 기본적으로 Responses API를 사용합니다.
- `compatible`: OpenAI 호환 Chat Completions 서버. OpenAI 전용 요청 인자를 사용하면 API 호출 전에 오류로 중단합니다.
- `local`: `compatible`의 별칭입니다.

필요하면 provider에 `api = "responses"` 또는 `api = "chat_completions"`를 명시할 수 있습니다. 단, `responses`는 `type = "openai"` provider에서만 사용할 수 있습니다. OpenAI Responses API를 사용할 때는 기본적으로 `store = false`가 적용됩니다.

provider나 stage에는 `[...request]` 하위 테이블을 둘 수 있습니다. 이 값은 OpenAI Python SDK의 `responses.create(..., **request)` 또는 `chat.completions.create(..., **request)`로 전달됩니다. provider의 request는 기본값처럼 쓰이고, stage의 request가 같은 키를 덮어씁니다.

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

공통 요청 인자:

- `temperature`
- `top_p`
- `max_tokens`
- `max_completion_tokens`
- `max_output_tokens`
- `presence_penalty`
- `frequency_penalty`
- `seed`
- `timeout`

`temperature`처럼 모델별로 지원 여부가 다른 인자는 사용하는 모델이나 호환 서버가 지원할 때만 설정합니다.

OpenAI Responses provider에서 자주 쓰는 옵션 예시:

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

OpenAI 호환 Chat Completions provider에서 자주 쓰는 옵션 예시:

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

토큰 제한 인자 규칙:

- Chat Completions는 `max_tokens` 또는 `max_completion_tokens`를 사용합니다.
- Responses는 `max_output_tokens`를 사용합니다.
- Responses provider에서 `max_tokens`나 `max_completion_tokens`를 쓰면 오류로 중단합니다.
- 같은 stage/provider에서 `max_tokens`, `max_completion_tokens`, `max_output_tokens`를 섞어 쓰면 오류로 중단합니다.

OpenAI 전용 요청 인자:

- `reasoning`
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
model = "gpt-5.6-luna"

[profiles.fast.stages.summary.request.reasoning]
effort = "low"
```

실행:

```bash
lecture-notes ./lectures --profile fast
lecture-notes ./lectures --config ./my-lecture-notes.toml
```

## 내용 기반 제목·파일명과 복습 품질

요약 단계에서 강의 핵심 개념을 담은 제목을 함께 생성해 노트 첫 줄에 표시합니다. 별도의 제목 생성 API 호출은 추가하지 않습니다.

```bash
lecture-notes "./녹음 01.m4a" --name-from-content --language ko
# 예: 녹음 01 - 상관관계와 인과관계.md
```

`--name-from-content`는 새 노트를 `원본명 - 핵심 주제.md`로 저장합니다. 원본 파일은 바꾸지 않습니다. 생략하면 기존 `원본명.md` 규칙을 유지합니다. 제목이 없으면 원본명을 사용합니다. 제목의 경로 구분자·파일명 금지 문자·Obsidian 링크 특수문자를 정리하고 UTF-8 파일명 길이를 제한합니다.

노트 첫 줄의 숨겨진 원본 파일 식별 주석으로 재실행 시 기존 노트를 찾아 건너뜁니다. `--overwrite`는 찾은 파일의 내용을 갱신하며 파일명은 유지하므로 기존 링크가 깨지지 않습니다. 식별 주석을 지우거나 원본 파일명을 바꾸면 이 연결은 유지되지 않습니다. 다른 노트와 이름이 충돌하면 덮어쓰지 않고 오류로 중단합니다.

요약은 개념·이유·적용 조건·예외를 연결하고, 수치와 단위, 시험 포함/제외 범위, 교수의 정정 발언을 보존하도록 작성합니다. 과제·일정과 실제 질문·답변은 내용이 있을 때 별도 섹션으로 표시합니다. 코넬 노트는 회상 질문을 사용하며 고정 행 수 때문에 주요 내용을 버리지 않도록 했습니다. 전사 캐시에는 시간이 남지만 최종 Markdown에서는 초 표시를 제거합니다.

기본 모델은 교정·서식화 `gpt-5.6-luna`, 요약·코넬 노트 `gpt-5.6-terra`입니다. 로컬 `lecture-notes.toml`은 전역 설정보다 우선합니다.

품질 확인용 합성 전사는 `tests/fixtures/lecture_quality.txt`입니다. 실제 provider로 실행해 다음을 확인할 수 있습니다: 시험 제외인 회귀분석 유도, 과제 사례 수 정정(2→3), 답변 없는 조별 제출 질문, 후반부 절대/상대 위험과 %포인트 구별. 이 예제 통과가 모든 강의의 정확성을 보장하지는 않습니다.

## CLI 옵션

- `lecture-notes [PATH]`
- `--config <path>`
- `--print-config-paths`
- `--profile <name>`
- `--asr-model <name>`
- `--cpu-threads <n>`
- `--device <auto|mps|cpu>`
- `--name-from-content`
- `--transcribe-only`
- `--compress-audio`
- `--no-compress-audio`
- `--language <code>`
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

`--model`, `--api-key`, `--base-url`은 세 옵션을 모두 함께 제공할 때만 전체 stage를 임시 OpenAI 호환 provider로 실행합니다. 일부만 제공하면 오류로 중단합니다.

## 동작 방식

대상 녹음 파일을 whispermlx로 전사·화자 구분한 뒤, 전사문과 기존 `*.txt` 파일에 대해 다음 순서로 처리합니다.

1. 전사 오류를 교정하되 의미를 최대한 보존합니다.
2. 전사문을 읽기 쉬운 문단 구조로 재정리합니다.
3. 복습에 유용한 핵심 요약을 생성합니다.
4. 전체 전사문을 대체해 복습할 수 있는 Markdown 표 기반 코넬 노트 형식의 필기본을 생성합니다.

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
- `--verbose` 없이도 파일 단위 진행 상황과 단어 정렬·화자 구분의 진행률/경과 시간을 출력합니다. 처리 중에는 30초마다 마지막 보고 진행률도 표시합니다. 이 대기 메시지는 진행이 늘었다는 뜻은 아니며, 모델 로딩 로그와는 별개입니다.
- 교정·서식화·요약·코넬 노트 단계는 기본으로 출력하며, `--verbose`는 설정 경로와 저장 등 추가 로그를 표시합니다. 로컬 추론 라이브러리 초기 로딩도 명시적으로 안내합니다.
- 오디오는 FFmpeg CLI로 디코딩해 메모리로 전달합니다. 이 경로에서 사용하지 않는 pyannote 내장 TorchCodec 디코더의 로딩 경고만 숨기며, 실제 전사·정렬·화자 구분 오류는 그대로 보고합니다.
- 출력 파일은 임시 파일에 먼저 쓴 뒤 원자적으로 교체합니다.
- `--jobs`는 동시 처리할 파일 수이며 기본값은 4입니다. 각 파일은 교정 → 서식화 후 요약과 코넬 노트를 동시에 생성합니다. 최대 LLM 동시 요청 수는 파일 수의 2배(기본 최대 8개)입니다. `--jobs 1`이어도 요약과 코넬 노트는 병렬입니다.
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

## 환경 변수

환경변수는 자동 fallback 설정으로 쓰지 않습니다. 설정파일의 provider가 `api_key_env = "OPENAI_API_KEY"`처럼 명시적으로 참조한 경우에만 해당 환경변수를 읽습니다.

예:

```toml
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"
```

```bash
export OPENAI_API_KEY="your-api-key"
```

전체 설정파일 탐색 순서는 `--config`, 현재 작업 디렉터리의 `lecture-notes.toml`, 전역 설정 순서입니다.

`uv tool`로 설치한 경우에도 전역 설정은 `~/.config/lecture-notes/config.toml`에 저장됩니다. 폴더별 설정을 쓰고 싶으면 해당 폴더에 `lecture-notes.toml`을 두면 전역 설정보다 우선 적용됩니다.

## 실패 처리와 재실행

- 처리 대상이 없거나 모든 결과가 이미 있으면 API 키 없이 정상 종료합니다. `--limit 0`도 API 클라이언트를 만들지 않습니다.
- 입력 파일의 출력 경로 충돌은 API 연결 전에 검사합니다. 이미 완료되어 건너뛸 파일들은 충돌 검사에서 제외합니다.
- Markdown 저장 실패 시 기존 결과를 보존하고 임시 파일을 정리합니다.
- 읽을 수 없거나 잘못된 설정파일은 경로가 포함된 오류 메시지로 안내합니다.
- 네트워크 연결 오류·시간 초과, HTTP 408·429·5xx만 재시도합니다. 인증 오류와 일반 처리 오류는 재시도하지 않습니다.
- `--retry-backoff`는 유한한 0 이상의 값이어야 합니다.
- 종료 코드: 성공 `0`, 파일 처리 실패 `1`, 입력·설정 오류 `2`. 일부 파일이 실패해도 기본적으로 나머지 파일을 계속 처리합니다.

## 개발

테스트 실행:

```bash
python -m unittest discover -s tests
```

## 라이선스

MIT. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
