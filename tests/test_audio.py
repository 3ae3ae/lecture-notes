import io
import json
import subprocess
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import TestCase, mock

from lecture_notes import audio, cli


class AudioWorkflowTests(TestCase):
    def test_cached_transcript_is_reused_and_invalidated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "강의.m4a")
            path.write_bytes(b"audio")
            kwargs = dict(model="large-v3", language="ko", on_stage=lambda _: None)
            with mock.patch.object(audio, "transcribe_audio", return_value="[SPEAKER_00] 설명") as asr:
                audio.cached_transcription(path, **kwargs)
                audio.cached_transcription(path, **kwargs)
                self.assertEqual(asr.call_count, 1)
                path.write_bytes(b"changed audio")
                audio.cached_transcription(path, **kwargs)
                self.assertEqual(asr.call_count, 2)
                audio.cached_transcription(path, **{**kwargs, "language": "en"})
                self.assertEqual(asr.call_count, 3)
            self.assertEqual(json.loads(audio.transcript_cache_path(path).read_text())["text"], "[SPEAKER_00] 설명")

    def test_failed_note_generation_keeps_cached_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "강의.m4a")
            path.touch()
            with mock.patch.object(audio, "transcribe_audio", return_value="전사") as asr, \
                 mock.patch.object(cli, "run_pipeline_with_progress", side_effect=RuntimeError("failed")), \
                 redirect_stdout(io.StringIO()):
                kwargs = dict(index=1, total_files=1, txt_path=path, args=cli.parse_args([]),
                              stage_configs={}, retry_config=cli.RetryConfig())
                self.assertEqual(cli._process_file(**kwargs)[0], "error")
                self.assertEqual(cli._process_file(**kwargs)[0], "error")
                self.assertEqual(asr.call_count, 1)
            self.assertTrue(audio.transcript_cache_path(path).exists())
            self.assertFalse(path.with_suffix('.md').exists())

    def test_audio_preferred_over_legacy_text_before_limit(self):
        paths = [Path("강의.txt"), Path("강의.M4A"), Path("기타.txt")]
        self.assertEqual(cli.select_inputs(paths), paths[1:])

    def test_standalone_modes_do_not_load_llm_config_or_run_on_dry_run(self):
        for mode in ("--compress-audio", "--transcribe-only"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                Path(directory, "강의.m4a").touch()
                with mock.patch.object(cli, "_resolve_pipeline_settings") as config, \
                     mock.patch.object(cli, "compress_audio") as compress, \
                     mock.patch.object(cli, "cached_transcription") as asr, redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main([directory, mode, "--dry-run"]), 0)
                    config.assert_not_called()
                    compress.assert_not_called()
                    asr.assert_not_called()

    def test_compression_failure_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "강의.m4a")
            path.write_bytes(b"original audio")
            with mock.patch.object(audio, "_probe", return_value={"duration": "10", "bit_rate": "128000"}), \
                 mock.patch.object(audio.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
                with self.assertRaises(subprocess.CalledProcessError):
                    audio.compress_audio(path)
            self.assertEqual(path.read_bytes(), b"original audio")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_compression_replaces_only_valid_smaller_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "강의.m4a")
            path.write_bytes(b"long original audio")
            def convert(command, **kwargs):
                Path(command[-1]).write_bytes(b"smaller")
            with mock.patch.object(audio, "_probe", return_value={"duration": "10", "bit_rate": "128000"}), \
                 mock.patch.object(audio.subprocess, "run", side_effect=convert):
                self.assertIn("compressed", audio.compress_audio(path))
            self.assertEqual(path.read_bytes(), b"smaller")

    def test_compression_skips_low_bitrate_and_rejects_shortened_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "강의.m4a")
            path.write_bytes(b"original audio")
            with mock.patch.object(audio, "_probe", return_value={"duration": "10", "bit_rate": "64000"}), \
                 mock.patch.object(audio.subprocess, "run") as run:
                self.assertIn("skipped", audio.compress_audio(path))
                run.assert_not_called()
            def convert(command, **kwargs):
                Path(command[-1]).write_bytes(b"short")
            with mock.patch.object(audio, "_probe", side_effect=[{"duration": "10"}, {"duration": "2"}]), \
                 mock.patch.object(audio.subprocess, "run", side_effect=convert):
                with self.assertRaisesRegex(RuntimeError, "duration"):
                    audio.compress_audio(path)
            self.assertEqual(path.read_bytes(), b"original audio")
