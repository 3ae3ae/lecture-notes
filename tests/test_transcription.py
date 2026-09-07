from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

from lecture_notes.transcription import render_transcript, transcribe_audio, validate_audio_environment


class TranscriptionTests(TestCase):
    def test_word_speaker_changes_and_unknown_words(self):
        result = {"segments": [{"start": 0, "text": "설명 질문 미상", "words": [
            {"word": "설명", "start": 0, "speaker": "SPEAKER_00"},
            {"word": "질문", "start": 1, "speaker": "SPEAKER_01"},
            {"word": "미상"},
        ]}, {"start": 3, "text": "정렬 불가", "speaker": "SPEAKER_00"}]}
        self.assertEqual(render_transcript(result),
                         "[0.00s] [SPEAKER_00] 설명\n[1.00s] [SPEAKER_01] 질문\n"
                         "[0.00s] [UNKNOWN] 미상\n[3.00s] [SPEAKER_00] 정렬 불가")

    def test_audio_api_contract(self):
        api = mock.Mock()
        result = {"language": "ko", "segments": [{"start": 0, "text": "안녕하세요"}]}
        api.load_model.return_value.transcribe.return_value = result
        api.load_align_model.return_value = ("aligner", "metadata")
        def align(*args, progress_callback, **kwargs):
            progress_callback(10)
            progress_callback(50)
            return result
        api.align.side_effect = align
        api.assign_word_speakers.return_value = result
        diarizer = mock.Mock()
        speakers = object()
        def diarize(*args, progress_callback):
            progress_callback(25)
            progress_callback(100)
            return speakers
        diarizer.return_value.side_effect = diarize
        logs = []
        with mock.patch.dict("sys.modules", {"whispermlx": api,
                             "whispermlx.diarize": SimpleNamespace(DiarizationPipeline=diarizer)}), \
             mock.patch("lecture_notes.transcription.validate_audio_environment"), \
             mock.patch.dict("os.environ", {"HF_TOKEN": "test"}):
            text = transcribe_audio(Path("한글 강의.m4a"), language="ko", on_stage=logs.append)
        api.load_audio.assert_called_once_with("한글 강의.m4a")
        api.load_model.assert_called_once_with("large-v3", device="cpu", language="ko", vad_method="silero")
        api.align.assert_called_once_with(result["segments"], "aligner", "metadata", api.load_audio.return_value, device="cpu", progress_callback=mock.ANY)
        diarizer.assert_called_once_with(token="test", device="cpu")
        api.assign_word_speakers.assert_called_once_with(speakers, result)
        self.assertTrue(any("aligning words: 50%" in line for line in logs))
        self.assertTrue(any("identifying speakers: 25%" in line for line in logs))
        self.assertTrue(any("identifying speakers: completed" in line for line in logs))
        self.assertIn("안녕하세요", text)

    def test_unsupported_platform_has_actionable_error(self):
        with mock.patch("platform.system", return_value="Linux"):
            with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
                validate_audio_environment()

    def test_heartbeat_and_failure_do_not_report_false_completion(self):
        from lecture_notes.transcription import _progress
        logs = []
        event = mock.Mock()
        event.wait.side_effect = [False, True]
        def worker(*, target, daemon):
            return SimpleNamespace(start=target, join=lambda: None)
        with mock.patch("lecture_notes.transcription.threading.Event", return_value=event), \
             mock.patch("lecture_notes.transcription.threading.Thread", side_effect=worker):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with _progress("aligning words", logs.append) as update:
                    update(20)
                    update(20)
                    raise RuntimeError("failed")
        self.assertTrue(any("waiting for next update; last reported 0%" in line for line in logs))
        self.assertEqual(sum("aligning words: 20%" in line for line in logs), 1)
        self.assertFalse(any("completed" in line for line in logs))
        event.set.assert_called_once()
