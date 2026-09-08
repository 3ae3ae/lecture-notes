from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from lecture_notes import cli

DEFAULT_TEST_CONFIG = """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[stages.correction]
provider = "openai"
model = "global-model"

[stages.formatting]
provider = "openai"
model = "global-model"

[stages.summary]
provider = "openai"
model = "global-model"

[stages.cornell]
provider = "openai"
model = "global-model"
"""


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

    def test_remove_timestamps_preserves_speakers_and_content(self) -> None:
        transcript = (
            "[0.00s] [SPEAKER_00] 설명\n"
            "  [123.45s] [SPEAKER_01] 질문\n"
            "시간 10초는 강의 내용"
        )

        self.assertEqual(
            cli.remove_timestamps(transcript),
            "[SPEAKER_00] 설명\n[SPEAKER_01] 질문\n시간 10초는 강의 내용",
        )

    def test_write_markdown_uses_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir, "lecture.md")
            cli.write_markdown(
                output_path,
                "### 핵심 요약\n- 요약",
                "### 코넬 노트\n\n"
                "| 단서 / 질문 | 필기 |\n"
                "|---|---|\n"
                "| 질문 | 필기 |\n\n"
                "| 요약 |\n"
                "|---|\n"
                "| 정리 |",
                "전사문",
            )

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "## 요약\n\n"
                "### 핵심 요약\n- 요약\n\n"
                "## 코넬 노트\n\n"
                "### 코넬 노트\n\n"
                "| 단서 / 질문 | 필기 |\n"
                "|---|---|\n"
                "| 질문 | 필기 |\n\n"
                "| 요약 |\n"
                "|---|\n"
                "| 정리 |\n\n"
                "## 전체 전사문\n\n"
                "전사문\n",
            )

    def test_normalize_summary_text_converts_bracket_sections(self) -> None:
        summary_text = "[핵심 요약]\n- A\n\n[교수님 강조 포인트]\n- B"

        self.assertEqual(
            cli.normalize_summary_text(summary_text),
            "### 핵심 요약\n- A\n\n### 교수님 강조 포인트\n- B",
        )


class MainTests(unittest.TestCase):
    def test_main_dry_run_creates_global_config_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            root = Path(tmpdir)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=root),
                mock.patch.dict(os.environ, {"HOME": tmpdir}, clear=False),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main([tmpdir, "--dry-run"])

            self.assertEqual(exit_code, 0)
            self.assertEqual("", stderr.getvalue())
            self.assertTrue((root / ".config" / "lecture-notes" / "config.toml").exists())
            self.assertIn("created default config ->", stdout.getvalue())

    def test_main_reports_missing_explicit_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"LECTURE_NOTES_MODEL": "gpt-test"},
            clear=True,
        ):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=Path(tmpdir)),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main([tmpdir, "--config", str(Path(tmpdir, "missing.toml"))])

            self.assertEqual(exit_code, 2)
            self.assertIn("config file does not exist", stderr.getvalue())

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
            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=root),
                mock.patch.dict(os.environ, {"HOME": tmpdir}, clear=False),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main([tmpdir, "--dry-run"])

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("searching", output)
            self.assertIn("found 2 matching file(s)", output)
            self.assertIn("[1/2]", output)
            self.assertIn("would-process", output)
            self.assertIn("skip", output)
            self.assertIn("processed=1 skipped=1 errors=0", output)
            self.assertEqual("", stderr.getvalue())

    def test_local_config_wins_over_global_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            root = Path(tmpdir)
            local_config = root / "lecture-notes.toml"
            global_config = root / ".config" / "lecture-notes" / "config.toml"
            global_config.parent.mkdir(parents=True)
            global_config.write_text(DEFAULT_TEST_CONFIG.replace("global-model", "global-only"), encoding="utf-8")
            local_config.write_text(DEFAULT_TEST_CONFIG.replace("global-model", "local-only"), encoding="utf-8")

            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=root),
                mock.patch.dict(os.environ, {"HOME": tmpdir}, clear=False),
            ):
                config_path, settings, created = cli._resolve_pipeline_settings(cli.parse_args([tmpdir]))

            self.assertEqual(config_path, local_config)
            self.assertFalse(created)
            self.assertEqual(settings.stages["summary"].model, "local-only")

    def test_explicit_config_wins_over_local_and_global_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            root = Path(tmpdir)
            explicit_config = root / "explicit.toml"
            local_config = root / "lecture-notes.toml"
            global_config = root / ".config" / "lecture-notes" / "config.toml"
            global_config.parent.mkdir(parents=True)
            explicit_config.write_text(DEFAULT_TEST_CONFIG.replace("global-model", "explicit-only"), encoding="utf-8")
            local_config.write_text(DEFAULT_TEST_CONFIG.replace("global-model", "local-only"), encoding="utf-8")
            global_config.write_text(DEFAULT_TEST_CONFIG.replace("global-model", "global-only"), encoding="utf-8")

            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=root),
                mock.patch.dict(os.environ, {"HOME": tmpdir}, clear=False),
            ):
                config_path, settings, created = cli._resolve_pipeline_settings(
                    cli.parse_args([tmpdir, "--config", str(explicit_config)])
                )

            self.assertEqual(config_path, explicit_config.resolve())
            self.assertFalse(created)
            self.assertEqual(settings.stages["summary"].model, "explicit-only")

    def test_global_config_is_loaded_when_local_config_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            root = Path(tmpdir)
            global_config = root / ".config" / "lecture-notes" / "config.toml"
            global_config.parent.mkdir(parents=True)
            global_config.write_text(DEFAULT_TEST_CONFIG, encoding="utf-8")

            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=root),
                mock.patch.dict(os.environ, {"HOME": tmpdir}, clear=False),
            ):
                config_path, settings, created = cli._resolve_pipeline_settings(cli.parse_args([tmpdir]))

            self.assertEqual(config_path, global_config)
            self.assertFalse(created)
            self.assertEqual(settings.stages["summary"].model, "global-model")

    def test_global_config_is_auto_created_for_first_non_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            root = Path(tmpdir)
            txt_path = root / "lecture.txt"
            txt_path.write_text("raw", encoding="utf-8")
            global_config = root / ".config" / "lecture-notes" / "config.toml"

            class Result:
                formatted_transcript = "formatted"
                summary_text = "summary"
                cornell_notes_text = "cornell notes"

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=root),
                mock.patch.dict(os.environ, {"HOME": tmpdir}, clear=False),
                mock.patch("lecture_notes.cli._build_client_from_provider", return_value=object()),
                mock.patch("lecture_notes.cli.run_pipeline_with_progress", return_value=Result()),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main([tmpdir])

            self.assertEqual(exit_code, 0)
            self.assertTrue(global_config.exists())
            self.assertIn("created default config ->", stdout.getvalue())
            self.assertEqual("", stderr.getvalue())

    def test_existing_global_config_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key"},
            clear=True,
        ):
            root = Path(tmpdir)
            global_config = root / ".config" / "lecture-notes" / "config.toml"
            global_config.parent.mkdir(parents=True)
            original_content = DEFAULT_TEST_CONFIG.replace("global-model", "kept-model")
            global_config.write_text(original_content, encoding="utf-8")

            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=root),
                mock.patch.dict(os.environ, {"HOME": tmpdir}, clear=False),
            ):
                _, settings, created = cli._resolve_pipeline_settings(cli.parse_args([tmpdir]))

            self.assertFalse(created)
            self.assertEqual(global_config.read_text(encoding="utf-8"), original_content)
            self.assertEqual(settings.stages["summary"].model, "kept-model")

    def test_print_config_paths_exits_without_discovering_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=Path(tmpdir)),
                mock.patch.dict(os.environ, {"HOME": tmpdir}, clear=False),
                mock.patch("lecture_notes.cli.discover_txt_files") as discover,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main(["--print-config-paths"])

            self.assertEqual(exit_code, 0)
            self.assertIn("local:", stdout.getvalue())
            self.assertIn("global:", stdout.getvalue())
            self.assertEqual("", stderr.getvalue())
            self.assertFalse((Path(tmpdir) / ".config" / "lecture-notes" / "config.toml").exists())
            discover.assert_not_called()

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
                **kwargs: object,
            ):
                if raw_text == "bad":
                    raise RuntimeError("boom")

                class Result:
                    formatted_transcript = "formatted"
                    summary_text = "summary"
                    cornell_notes_text = "cornell notes"

                return Result()

            stdout = io.StringIO()
            stderr = io.StringIO()
            global_config = root / ".config" / "lecture-notes" / "config.toml"
            global_config.parent.mkdir(parents=True)
            global_config.write_text(
                """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[stages.correction]
provider = "openai"
model = "gpt-test"

[stages.formatting]
provider = "openai"
model = "gpt-test"

[stages.summary]
provider = "openai"
model = "gpt-test"

[stages.cornell]
provider = "openai"
model = "gpt-test"
""",
                encoding="utf-8",
            )
            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=Path(tmpdir)),
                mock.patch.dict(os.environ, {"HOME": tmpdir}, clear=False),
                mock.patch("lecture_notes.cli._build_client_from_provider", return_value=object()),
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
                cornell_notes_text = "코넬 노트"

            stdout = io.StringIO()
            stderr = io.StringIO()
            global_config = root / ".config" / "lecture-notes" / "config.toml"
            global_config.parent.mkdir(parents=True)
            global_config.write_text(
                """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[stages.correction]
provider = "openai"
model = "gpt-test"

[stages.formatting]
provider = "openai"
model = "gpt-test"

[stages.summary]
provider = "openai"
model = "gpt-test"

[stages.cornell]
provider = "openai"
model = "gpt-test"
""",
                encoding="utf-8",
            )
            with (
                mock.patch("lecture_notes.cli.Path.cwd", return_value=Path(tmpdir)),
                mock.patch.dict(os.environ, {"HOME": tmpdir}, clear=False),
                mock.patch("lecture_notes.cli._build_client_from_provider", return_value=object()),
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

    def test_main_rejects_openai_only_option_for_compatible_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"LECTURE_NOTES_API_KEY": "test-key"},
            clear=True,
        ):
            root = Path(tmpdir)
            (root / "lecture.txt").write_text("raw", encoding="utf-8")
            config_path = root / "lecture-notes.toml"
            config_path.write_text(
                """
[providers.local]
type = "compatible"
base_url = "http://localhost:1234/v1"
api_key_env = "LECTURE_NOTES_API_KEY"

[stages.correction]
provider = "local"
model = "local-model"

[stages.formatting]
provider = "local"
model = "local-model"

[stages.summary]
provider = "local"
model = "local-model"

[stages.summary.request.reasoning]
effort = "minimal"

[stages.cornell]
provider = "local"
model = "local-model"
""",
                encoding="utf-8",
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch("lecture_notes.cli._build_client_from_provider") as build_client,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = cli.main([tmpdir, "--config", str(config_path)])

            self.assertEqual(exit_code, 2)
            self.assertIn("OpenAI-only option", stderr.getvalue())
            build_client.assert_not_called()

    def test_config_builds_distinct_provider_clients(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "openai-key", "LOCAL_API_KEY": "local-key"},
            clear=True,
        ):
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(
                """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[providers.local]
type = "compatible"
base_url = "http://localhost:1234/v1"
api_key_env = "LOCAL_API_KEY"

[stages.correction]
provider = "local"
model = "local-model"

[stages.formatting]
provider = "local"
model = "local-model"

[stages.summary]
provider = "openai"
model = "gpt-test"

[stages.summary.request.reasoning]
effort = "minimal"

[stages.cornell]
provider = "openai"
model = "gpt-test"
""",
                encoding="utf-8",
            )

            args = cli.parse_args([tmpdir, "--config", str(config_path)])
            _, settings, _ = cli._resolve_pipeline_settings(args)
            clients = {"openai": object(), "local": object()}

            def fake_build(provider: cli.ProviderConfig) -> object:
                return clients[provider.name]

            with mock.patch("lecture_notes.cli._build_client_from_provider", side_effect=fake_build):
                stage_configs = cli._build_stage_configs(settings)

            self.assertIs(stage_configs["correction"].client, clients["local"])
            self.assertIs(stage_configs["formatting"].client, clients["local"])
            self.assertIs(stage_configs["summary"].client, clients["openai"])
            self.assertEqual(
                stage_configs["summary"].request_options["reasoning"],
                {"effort": "minimal"},
            )
            self.assertEqual(stage_configs["summary"].api, "responses")
            self.assertFalse(stage_configs["summary"].request_options["store"])

    def test_full_cli_override_replaces_all_stages_with_compatible_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "env-openai", "LOCAL_API_KEY": "env-local"},
            clear=True,
        ):
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(
                """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[providers.local]
type = "compatible"
base_url = "http://localhost:1234/v1"
api_key_env = "LOCAL_API_KEY"

[stages.correction]
provider = "local"
model = "local-model"

[stages.formatting]
provider = "local"
model = "local-model"

[stages.summary]
provider = "openai"
model = "gpt-test"

[stages.cornell]
provider = "openai"
model = "gpt-test"
""",
                encoding="utf-8",
            )

            args = cli.parse_args(
                [
                    tmpdir,
                    "--config",
                    str(config_path),
                    "--model",
                    "override-model",
                    "--api-key",
                    "cli-key",
                    "--base-url",
                    "https://cli.example/v1",
                ]
            )
            _, settings, _ = cli._resolve_pipeline_settings(args)

            self.assertEqual(set(settings.providers), {"cli"})
            provider = settings.providers["cli"]
            self.assertEqual(provider.type, "compatible")
            self.assertEqual(provider.api, "chat_completions")
            self.assertEqual(provider.api_key, "cli-key")
            self.assertEqual(provider.base_url, "https://cli.example/v1")
            for stage in settings.stages.values():
                self.assertEqual(stage.provider_name, "cli")
                self.assertEqual(stage.api, "chat_completions")
                self.assertEqual(stage.model, "override-model")

    def test_partial_cli_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(DEFAULT_TEST_CONFIG, encoding="utf-8")

            args = cli.parse_args(
                [
                    tmpdir,
                    "--config",
                    str(config_path),
                    "--model",
                    "override-model",
                ]
            )

            with self.assertRaises(cli.ConfigError) as error:
                cli._resolve_pipeline_settings(args)

            self.assertIn("must be used together", str(error.exception))

    def test_cli_override_rejects_openai_only_stage_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(
                """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[stages.correction]
provider = "openai"
model = "gpt-test"

[stages.correction.request.reasoning]
effort = "minimal"

[stages.formatting]
provider = "openai"
model = "gpt-test"

[stages.summary]
provider = "openai"
model = "gpt-test"

[stages.cornell]
provider = "openai"
model = "gpt-test"
""",
                encoding="utf-8",
            )

            args = cli.parse_args(
                [
                    tmpdir,
                    "--config",
                    str(config_path),
                    "--model",
                    "override-model",
                    "--api-key",
                    "cli-key",
                    "--base-url",
                    "https://cli.example/v1",
                ]
            )

            with self.assertRaises(cli.ConfigError) as error:
                cli._resolve_pipeline_settings(args)

            self.assertIn("OpenAI-only option", str(error.exception))

    def test_env_base_url_does_not_override_config_provider_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "LECTURE_NOTES_BASE_URL": "https://env.example/v1",
            },
            clear=True,
        ):
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(DEFAULT_TEST_CONFIG, encoding="utf-8")

            args = cli.parse_args([tmpdir, "--config", str(config_path)])
            _, settings, _ = cli._resolve_pipeline_settings(args)

            self.assertIsNone(settings.providers["openai"].base_url)

    def test_env_model_does_not_override_config_stage_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "openai-key", "LECTURE_NOTES_MODEL": "env-model"},
            clear=True,
        ):
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(DEFAULT_TEST_CONFIG, encoding="utf-8")

            args = cli.parse_args([tmpdir, "--config", str(config_path)])
            _, settings, _ = cli._resolve_pipeline_settings(args)

            self.assertEqual(settings.stages["summary"].model, "global-model")

    def test_profile_overrides_top_level_stage_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "openai-key"},
            clear=True,
        ):
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(
                """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[stages.correction]
provider = "openai"
model = "full-model"

[stages.formatting]
provider = "openai"
model = "full-model"

[stages.summary]
provider = "openai"
model = "full-model"

[stages.cornell]
provider = "openai"
model = "full-model"

[profiles.fast.stages.summary]
provider = "openai"
model = "fast-model"

[profiles.fast.stages.summary.request.reasoning]
effort = "minimal"
""",
                encoding="utf-8",
            )

            args = cli.parse_args([tmpdir, "--config", str(config_path), "--profile", "fast"])
            _, settings, _ = cli._resolve_pipeline_settings(args)

            self.assertEqual(settings.stages["correction"].model, "full-model")
            self.assertEqual(settings.stages["summary"].model, "fast-model")
            self.assertEqual(
                settings.stages["summary"].request_options["reasoning"],
                {"effort": "minimal"},
            )

    def test_provider_api_defaults_and_local_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "openai-key", "LOCAL_API_KEY": "local-key"},
            clear=True,
        ):
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(
                """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[providers.local]
type = "local"
base_url = "http://localhost:1234/v1"
api_key_env = "LOCAL_API_KEY"

[stages.correction]
provider = "openai"
model = "gpt-test"
max_output_tokens = 100

[stages.formatting]
provider = "openai"
model = "gpt-test"

[stages.summary]
provider = "openai"
model = "gpt-test"

[stages.cornell]
provider = "local"
model = "local-test"
max_completion_tokens = 50
""",
                encoding="utf-8",
            )

            args = cli.parse_args([tmpdir, "--config", str(config_path)])
            _, settings, _ = cli._resolve_pipeline_settings(args)

            self.assertEqual(settings.providers["openai"].api, "responses")
            self.assertEqual(settings.providers["local"].type, "compatible")
            self.assertEqual(settings.providers["local"].api, "chat_completions")
            self.assertEqual(
                settings.stages["correction"].request_options["max_output_tokens"],
                100,
            )
            self.assertNotIn(
                "max_completion_tokens",
                settings.stages["correction"].request_options,
            )
            self.assertFalse(settings.stages["correction"].request_options["store"])
            self.assertEqual(
                settings.stages["cornell"].request_options["max_completion_tokens"],
                50,
            )

    def test_invalid_provider_api_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(
                """
[providers.openai]
type = "openai"
api = "assistants"
api_key_env = "OPENAI_API_KEY"

[stages.correction]
provider = "openai"
model = "gpt-test"

[stages.formatting]
provider = "openai"
model = "gpt-test"

[stages.summary]
provider = "openai"
model = "gpt-test"

[stages.cornell]
provider = "openai"
model = "gpt-test"
""",
                encoding="utf-8",
            )

            args = cli.parse_args([tmpdir, "--config", str(config_path)])

            with self.assertRaises(cli.ConfigError) as error:
                cli._resolve_pipeline_settings(args)

            self.assertIn("api must be", str(error.exception))

    def test_responses_rejects_max_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(
                """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[stages.correction]
provider = "openai"
model = "gpt-test"
max_tokens = 100

[stages.formatting]
provider = "openai"
model = "gpt-test"

[stages.summary]
provider = "openai"
model = "gpt-test"

[stages.cornell]
provider = "openai"
model = "gpt-test"
""",
                encoding="utf-8",
            )

            args = cli.parse_args([tmpdir, "--config", str(config_path)])

            with self.assertRaises(cli.ConfigError) as error:
                cli._resolve_pipeline_settings(args)

            self.assertIn("max_tokens", str(error.exception))

    def test_nested_request_tables_are_merged_for_provider_and_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "openai-key"},
            clear=True,
        ):
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(
                """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"
max_output_tokens = 1000

[providers.openai.request.metadata]
source = "lecture-notes"

[providers.openai.request.reasoning]
effort = "low"
summary = "auto"

[stages.correction]
provider = "openai"
model = "gpt-test"

[stages.formatting]
provider = "openai"
model = "gpt-test"

[stages.summary]
provider = "openai"
model = "gpt-test"
max_output_tokens = 2000

[stages.summary.request.reasoning]
effort = "medium"

[stages.cornell]
provider = "openai"
model = "gpt-test"
""",
                encoding="utf-8",
            )

            args = cli.parse_args([tmpdir, "--config", str(config_path)])
            _, settings, _ = cli._resolve_pipeline_settings(args)

            summary_options = settings.stages["summary"].request_options
            self.assertEqual(summary_options["max_output_tokens"], 2000)
            self.assertEqual(
                summary_options["reasoning"],
                {"effort": "medium", "summary": "auto"},
            )
            self.assertEqual(
                summary_options["metadata"],
                {"source": "lecture-notes"},
            )
            self.assertFalse(summary_options["store"])

    def test_responses_rejects_max_completion_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir, "lecture-notes.toml")
            config_path.write_text(
                """
[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[stages.correction]
provider = "openai"
model = "gpt-test"
max_completion_tokens = 100

[stages.formatting]
provider = "openai"
model = "gpt-test"

[stages.summary]
provider = "openai"
model = "gpt-test"

[stages.cornell]
provider = "openai"
model = "gpt-test"
""",
                encoding="utf-8",
            )

            args = cli.parse_args([tmpdir, "--config", str(config_path)])

            with self.assertRaises(cli.ConfigError) as error:
                cli._resolve_pipeline_settings(args)

            self.assertIn("max_completion_tokens", str(error.exception))


class AudioInputTests(unittest.TestCase):
    def test_audio_dry_run_does_not_load_models_and_accepts_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir, "한글 강의.M4A")
            audio.touch()
            with mock.patch.object(cli, "_resolve_pipeline_settings", return_value=(None, None, False)), \
                 mock.patch.object(cli, "cached_transcription") as transcribe, redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main([str(audio), "--dry-run"]), 0)
                transcribe.assert_not_called()

    def test_colliding_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "강의.wav").touch()
            Path(tmpdir, "강의.mp3").touch()
            with mock.patch.object(cli, "_resolve_pipeline_settings", return_value=(None, None, False)), \
                 redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(cli.main([tmpdir, "--dry-run"]), 2)
                self.assertIn("output collision", stderr.getvalue())

    def test_audio_flows_into_notes(self):
        from lecture_notes.pipeline import ProcessedDocument, RetryConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "한글 강의.wav")
            path.touch()
            with mock.patch.object(cli, "cached_transcription", return_value="[0.00s] [SPEAKER_00] 수업") as asr, \
                 mock.patch.object(cli, "run_pipeline_with_progress", return_value=ProcessedDocument("교정", "전사", "요약", "필기")) as pipeline, \
                 redirect_stdout(io.StringIO()):
                result = cli._process_file(index=1, total_files=1, txt_path=path,
                    args=cli.parse_args([]), stage_configs={}, retry_config=RetryConfig())
            self.assertEqual(result[0], "processed")
            self.assertEqual(pipeline.call_args.args[0], "[SPEAKER_00] 수업")
            self.assertIn("전사", path.with_suffix(".md").read_text())
            self.assertEqual(asr.call_count, 1)

    def test_utf8_bom_is_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "강의.txt")
            path.write_text("강의", encoding="utf-8-sig")
            self.assertEqual(cli.read_text_file(path), "강의")


class ReliabilityTests(unittest.TestCase):
    def test_save_failure_preserves_original_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir, "강의.md")
            output.write_text("original", encoding="utf-8")
            with mock.patch.object(Path, "replace", side_effect=OSError("disk error")):
                with self.assertRaisesRegex(OSError, "disk error"):
                    cli.write_markdown(output, "summary", "notes", "transcript")
            self.assertEqual(output.read_text(), "original")
            self.assertEqual(list(Path(tmpdir).iterdir()), [output])

    def test_unreadable_config_reports_cli_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(cli.main([tmpdir, "--config", tmpdir]), 2)
            self.assertIn("failed to read or parse", stderr.getvalue())

    def test_nonfinite_backoff_rejected_before_config_or_api(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), \
                 mock.patch.object(cli, "_resolve_pipeline_settings") as config, \
                 redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main([f"--retry-backoff={value}"]), 2)
                config.assert_not_called()

    def test_empty_completed_and_zero_limit_batches_need_no_api_key(self):
        for scenario in ("empty", "completed", "zero-limit"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                if scenario != "empty":
                    (root / "lecture.txt").write_text("text")
                if scenario == "completed":
                    (root / "lecture.md").write_text("existing notes")
                    (root / "lecture.mp3").touch()
                config = root / "settings.toml"
                config.write_text(DEFAULT_TEST_CONFIG)
                args = [tmpdir, "--config", str(config)]
                if scenario == "zero-limit":
                    args.extend(["--limit", "0"])
                with mock.patch.dict(os.environ, {}, clear=True), \
                     mock.patch.object(cli, "_build_stage_configs") as build, \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(args), 0)
                    build.assert_not_called()

    def test_collision_is_reported_before_api_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "lecture.mp3").touch()
            (root / "lecture.wav").touch()
            with mock.patch.object(cli, "_resolve_pipeline_settings", return_value=(None, None, False)), \
                 mock.patch.object(cli, "_build_stage_configs") as build, \
                 redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(cli.main([tmpdir]), 2)
                self.assertIn("output collision", stderr.getvalue())
                build.assert_not_called()


class ContentNamingTests(unittest.TestCase):
    def test_title_is_safe_and_fits_utf8_filename_limit(self):
        source = Path("/tmp/" + "원본" * 50 + ".txt")
        path = cli.content_output_path(source, '../위험: [개념] / 비교?' + '한글' * 100)
        self.assertEqual(path.parent, source.parent)
        self.assertLessEqual(len(path.name.encode("utf-8")), 255)
        for char in '<>:"/\\|?*#[]':
            self.assertNotIn(char, path.name)
        self.assertEqual(cli.content_output_path(source, '...'), source.with_suffix('.md'))

    def test_generated_name_is_found_on_rerun_and_kept_on_overwrite(self):
        from lecture_notes.pipeline import ProcessedDocument, RetryConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir, "녹음 01.txt")
            source.write_text("text")
            result = ProcessedDocument("corrected", "formatted", "summary", "notes", "상관관계와 인과관계")
            args = cli.parse_args(["--name-from-content"])
            with mock.patch.object(cli, "run_pipeline_with_progress", return_value=result) as pipeline, \
                 redirect_stdout(io.StringIO()):
                kwargs = dict(index=1, total_files=1, txt_path=source, args=args,
                              stage_configs={}, retry_config=RetryConfig())
                self.assertEqual(cli._process_file(**kwargs)[0], "processed")
                output = cli.existing_output(source)
                self.assertEqual(output.name, "상관관계와 인과관계.md")
                self.assertEqual(cli._process_file(**kwargs)[0], "skipped")
                self.assertEqual(pipeline.call_count, 1)
                args.overwrite = True
                result.title = "새 제목"
                self.assertEqual(cli._process_file(**kwargs)[0], "processed")
                self.assertEqual(list(Path(tmpdir).glob("*.md")), [output])
                self.assertIn("# 새 제목", output.read_text())

    def test_generated_name_never_overwrites_unrelated_note(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir, "existing.md")
            output.write_text("keep me")
            with self.assertRaises(FileExistsError):
                cli.write_markdown(output, "summary", "notes", "text", overwrite=False)
            self.assertEqual(output.read_text(), "keep me")
            self.assertEqual(list(Path(tmpdir).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
