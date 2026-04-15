from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from lecturebot import cli


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
            cli.write_markdown(output_path, "요약", "전사문")

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "[핵심 요약]\n요약\n\n[전체 전사문]\n전사문\n",
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
            self.assertIn("would-process", output)
            self.assertIn("skip", output)
            self.assertIn("processed=1 skipped=1 errors=0", output)
            self.assertEqual("", stderr.getvalue())

    def test_main_continues_after_processing_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"LECTUREBOT_MODEL": "gpt-test", "OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            root = Path(tmpdir)
            good = root / "good.txt"
            bad = root / "bad.txt"
            good.write_text("good", encoding="utf-8")
            bad.write_text("bad", encoding="utf-8")

            def fake_pipeline(raw_text: str, client: object, model: str):
                if raw_text == "bad":
                    raise RuntimeError("boom")

                class Result:
                    formatted_transcript = "formatted"
                    summary_text = "summary"

                return Result()

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("lecturebot.cli._build_client", return_value=object()),
                mock.patch("lecturebot.cli.run_pipeline", side_effect=fake_pipeline),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main([tmpdir])

            self.assertEqual(exit_code, 1)
            self.assertTrue((root / "good.md").exists())
            self.assertFalse((root / "bad.md").exists())
            self.assertIn("errors=1", stdout.getvalue())
            self.assertIn("boom", stderr.getvalue())
