# lecturebot

`lecturebot`은 강의 전사 `txt` 파일을 찾아 3단계 AI 워크플로우로 처리하고, 요약과 전사문이 함께 들어 있는 `md` 파일을 생성하는 Python CLI입니다.

## 설치

로컬 프로젝트에서:

```bash
uv tool install .
```

원격 저장소를 배포한 뒤에는 일반적인 `uv tool install <repo-or-package>` 형태로 설치할 수 있습니다.

## 환경 변수

- `OPENAI_API_KEY`: OpenAI API 키
- `LECTUREBOT_MODEL`: 기본 모델명

CLI 인자가 환경변수보다 우선합니다.

## 사용법

현재 디렉터리 재귀 탐색:

```bash
lecturebot --model gpt-4o-mini
```

특정 루트 디렉터리 지정:

```bash
lecturebot ./lectures --model gpt-4o-mini
```

실제 처리 없이 대상만 확인:

```bash
lecturebot ./lectures --dry-run
```

상세 단계 로그까지 보기:

```bash
lecturebot ./lectures --model gpt-4o-mini --verbose
```

추가 옵션:

- `--api-key <key>`
- `--base-url <url>`
- `--include-glob <pattern>` 반복 가능
- `--exclude-dir <name>` 반복 가능
- `--verbose`
- `--fail-fast`

## 동작 방식

`lecturebot`은 실행 위치 또는 지정한 경로 아래에서 `*.txt`를 재귀 탐색합니다.

- 기본 제외 디렉터리: `.git`, `.venv`, `node_modules`, `__pycache__`
- `foo.txt` 옆에 `foo.md`가 이미 있으면 해당 파일은 건너뜁니다.
- 텍스트는 `utf-8`, `utf-8-sig`, `cp949` 순으로 읽기를 시도합니다.
- 출력 파일은 항상 원본과 같은 폴더의 같은 이름 `md`로 저장됩니다.
- 기본 실행만으로도 파일별 진행 상황이 `[현재/전체]` 형식으로 출력됩니다.
- `--verbose`를 주면 단계별 진행 로그를 추가로 출력합니다.
- 한글 파일명과 공백이 포함된 파일명도 `pathlib` 기반으로 그대로 처리합니다.

최종 Markdown 형식:

```md
[핵심 요약]
{summary}

[전체 전사문]
{formatted_transcript}
```

## 개발 테스트

```bash
python -m unittest discover -s tests
```
