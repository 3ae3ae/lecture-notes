# AGENTS.md

이 문서는 `lecture-notes` 저장소에서 작업하는 코딩 에이전트를 위한 간단한 작업 가이드입니다.

## 프로젝트 개요

`lecture-notes`는 녹음·영상과 강의 전사 `*.txt`를 찾아 처리하는 Python CLI입니다. 녹음은 whispermlx로 화자 전사한 뒤 다음 4단계 AI 파이프라인으로 처리합니다. 같은 basename의 녹음과 TXT는 녹음을 우선합니다.

1. 전사 오류 교정
2. 전사문 서식화
3. 핵심 요약 생성
4. 코넬 노트테이킹법 기반 상세 필기본 생성

교정 → 서식화 순서를 지키고 요약·코넬 노트는 병렬 실행합니다. 기본 `--jobs 4`로 파일별 LLM 처리를 병렬화하며, 로컬 음성 전사만 잠금으로 직렬화합니다.

최종 결과물은 원본 옆의 `md` 파일로 저장됩니다. `--name-from-content`는 내용 기반 파일명을 사용합니다. M4A는 전사 전에 기본으로 압축하며 `--no-compress-audio`로 건너뛸 수 있습니다. `--transcribe-only`는 전사 JSON 캐시만 생성하고, `--compress-audio`는 M4A 압축만 실행합니다. 두 전용 모드는 LLM 설정이 필요 없습니다.

## 기본 명령

설치:

```bash
uv tool install .
```

CLI 실행:

```bash
lecture-notes
lecture-notes ./lectures --dry-run
lecture-notes ./lectures --verbose
```

테스트:

```bash
python -m unittest discover -s tests
```

## 주요 파일

- `lecture_notes/cli.py`: CLI 인자 처리, 파일 탐색, 출력 저장, 진행 로그
- `lecture_notes/audio.py`: 전사 캐시 및 M4A 압축
- `lecture_notes/transcription.py`: whispermlx 전사·시간 정렬·화자 구분
- `lecture_notes/pipeline.py`: 4단계 LLM 호출 파이프라인
- `lecture_notes/prompts.py`: 각 단계 시스템 프롬프트
- `tests/test_cli.py`: CLI/파일 처리 테스트
- `tests/test_pipeline.py`: 파이프라인 호출 순서 테스트
- `README.md`: 기본 한국어 문서
- `README.en.md`: 영어 문서

## 작업 규칙

- 패키지명은 `lecture_notes`, CLI 명령은 `lecture-notes`를 유지합니다.
- 출력 Markdown은 Obsidian 친화적으로 헤딩 기반 형식을 유지합니다.
- 기존 동작을 바꾸는 수정이면 `README.md`와 필요한 경우 `README.en.md`도 함께 업데이트합니다.
- 작업을 마친 뒤 변경 성격에 맞게 프로젝트 버전을 올립니다. 버전은 `pyproject.toml`과 `lecture_notes/__init__.py`를 함께 갱신합니다.
- 파일명 한글/공백 지원을 깨지 않도록 `pathlib` 기반 처리를 유지합니다.
- OpenAI 호환 서버 지원은 `LECTURE_NOTES_BASE_URL`과 `chat.completions.create(...)` 호출 기준을 유지합니다.

## 기대 동작

- 기본 실행 시 현재 디렉터리부터 재귀 탐색합니다.
- 같은 basename의 `.md`가 이미 있으면 해당 `txt`는 건너뜁니다.
- 기본 제외 디렉터리는 `.git`, `.venv`, `node_modules`, `__pycache__`입니다.
- 텍스트는 `utf-8`, `utf-8-sig`, `cp949` 순으로 읽기를 시도합니다.
- `--verbose` 없이도 파일 단위 진행 상황이 출력되어야 합니다.

## 수정 후 확인

코드 수정 후 최소한 아래는 확인합니다.

```bash
python -m unittest discover -s tests
python -m lecture_notes.cli --help
```
