from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from lecture_notes import cli


class DiscoverTxtFilesTests(unittest.TestCase):
    def test_discover_txt_files_recurses_and_excludes_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "keep").mkdir()
            (root / ".git").mkdir()
            (root / "node_modules").mkdir()
            (root / "keep" / "a.txt").write_text("a", encoding="utf-8")
            (root / ".git" / "ignored.txt").write_text("x", encoding="utf-8")
            (root / "node_modules" / "ignored.txt").write_text("y", encoding="utf-8")
            (root / "keep" / "b.md").write_text("b", encoding="utf-8")

            found = cli.discover_txt_files(
                root,
                include_globs=["*.txt"],
                exclude_dirs=cli.DEFAULT_EXCLUDE_DIRS,
            )

            self.assertEqual(found, [root / "keep" / "a.txt"])

    def test_should_skip_checks_matching_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir, "lecture.txt")
            txt_path.write_text("hello", encoding="utf-8")
            self.assertFalse(cli.should_skip(txt_path))
            txt_path.with_suffix(".md").write_text("done", encoding="utf-8")
            self.assertTrue(cli.should_skip(txt_path))


class ReadWriteTests(unittest.TestCase):
    def test_read_text_file_falls_back_to_cp949(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "lecture.txt")
            path.write_bytes("안녕하세요".encode("cp949"))

            self.assertEqual(cli.read_text_file(path), "안녕하세요")

    def test_write_markdown_uses_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir, "lecture.md")
            cli.write_markdown(output_path, "### 핵심 요약\n- 요약", "전사문")

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "## 요약\n\n### 핵심 요약\n- 요약\n\n## 전체 전사문\n\n전사문\n",
            )

    def test_normalize_summary_text_converts_bracket_sections(self) -> None:
        summary_text = "[핵심 요약]\n- A\n\n[교수님 강조 포인트]\n- B"

        self.assertEqual(
            cli.normalize_summary_text(summary_text),
            "### 핵심 요약\n- A\n\n### 교수님 강조 포인트\n- B",
        )


class MainTests(unittest.TestCase):
    def test_main_requires_model_when_not_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli.main([tmpdir])

            self.assertEqual(exit_code, 2)
            self.assertIn("model is required", stderr.getvalue())

    def test_main_requires_api_key_with_updated_env_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"LECTURE_NOTES_MODEL": "gpt-test"},
            clear=True,
        ):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli.main([tmpdir])

            self.assertEqual(exit_code, 2)
            self.assertIn("LECTURE_NOTES_API_KEY", stderr.getvalue())

    def test_main_dry_run_lists_process_and_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            todo = root / "todo.txt"
            done = root / "done.txt"
            todo.write_text("todo", encoding="utf-8")
            done.write_text("done", encoding="utf-8")
            done.with_suffix(".md").write_text("existing", encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli.main([tmpdir, "--dry-run"])

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("searching", output)
            self.assertIn("found 2 matching txt file(s)", output)
            self.assertIn("[1/2]", output)
            self.assertIn("would-process", output)
            self.assertIn("skip", output)
            self.assertIn("processed=1 skipped=1 errors=0", output)
            self.assertEqual("", stderr.getvalue())

    def test_main_continues_after_processing_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"LECTURE_NOTES_MODEL": "gpt-test", "OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            root = Path(tmpdir)
            good = root / "good.txt"
            bad = root / "bad.txt"
            good.write_text("good", encoding="utf-8")
            bad.write_text("bad", encoding="utf-8")

            def fake_pipeline(
                raw_text: str,
                client: object,
                model: str,
                on_stage: object | None = None,
            ):
                if raw_text == "bad":
                    raise RuntimeError("boom")

                class Result:
                    formatted_transcript = "formatted"
                    summary_text = "summary"

                return Result()

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("lecture_notes.cli._build_client", return_value=object()),
                mock.patch("lecture_notes.cli.run_pipeline_with_progress", side_effect=fake_pipeline),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main([tmpdir])

            self.assertEqual(exit_code, 1)
            self.assertTrue((root / "good.md").exists())
            self.assertFalse((root / "bad.md").exists())
            self.assertIn("errors=1", stdout.getvalue())
            self.assertIn("boom", stderr.getvalue())

    def test_main_handles_korean_and_space_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"LECTURE_NOTES_MODEL": "gpt-test", "OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            root = Path(tmpdir)
            txt_path = root / "강의 노트 1.txt"
            txt_path.write_text("원본 전사", encoding="utf-8")

            class Result:
                formatted_transcript = "정리된 전사문"
                summary_text = "핵심 요약"

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("lecture_notes.cli._build_client", return_value=object()),
                mock.patch("lecture_notes.cli.run_pipeline_with_progress", return_value=Result()),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main([tmpdir])

            output_path = root / "강의 노트 1.md"
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertIn("강의 노트 1.txt", stdout.getvalue())
            self.assertEqual("", stderr.getvalue())

    def test_build_client_uses_openai_compatible_env_fallbacks(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LECTURE_NOTES_API_KEY": "compat-key",
                "LECTURE_NOTES_BASE_URL": "https://compat.example/v1",
            },
            clear=True,
        ):
            args = cli.parse_args([])
            mock_openai = mock.Mock()
            fake_module = types.SimpleNamespace(OpenAI=mock_openai)

            with mock.patch.dict(sys.modules, {"openai": fake_module}):
                cli._build_client(args)

            mock_openai.assert_called_once_with(
                api_key="compat-key",
                base_url="https://compat.example/v1",
            )
