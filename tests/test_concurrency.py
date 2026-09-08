import io
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from lecture_notes import cli, pipeline


class ConcurrencyTests(unittest.TestCase):
    def test_summary_and_cornell_overlap_after_formatting(self):
        barrier = threading.Barrier(2, timeout=3)
        configs = {name: pipeline.StageConfig(name, None, 'test') for name in pipeline.STAGE_ORDER}
        calls = []

        def model_call(*, stage_config, user_text, **kwargs):
            name = stage_config.name
            calls.append(name)
            if name == 'correction':
                self.assertEqual(user_text, 'raw')
                return 'corrected'
            if name == 'formatting':
                self.assertEqual(user_text, 'corrected')
                return 'formatted'
            self.assertEqual(user_text, 'formatted')
            barrier.wait()
            return name

        with mock.patch.object(pipeline, '_call_model', side_effect=model_call):
            result = pipeline.run_pipeline_with_progress('raw', stage_configs=configs)
        self.assertEqual(calls[:2], ['correction', 'formatting'])
        self.assertCountEqual(calls[2:], ['summary', 'cornell'])
        self.assertEqual(result.summary_text, 'summary')
        self.assertEqual(result.cornell_notes_text, 'cornell')

    def test_next_audio_overlaps_previous_llm_work(self):
        llm_started = threading.Event()
        next_audio_started = threading.Event()
        asr_calls = []

        def transcribe(path, **kwargs):
            asr_calls.append(path)
            if len(asr_calls) == 2:
                self.assertTrue(llm_started.wait(3))
                next_audio_started.set()
            return str(path)

        def notes(raw_text, **kwargs):
            llm_started.set()
            self.assertTrue(next_audio_started.wait(3))
            return pipeline.ProcessedDocument(raw_text, raw_text, 'summary', 'cornell')

        with tempfile.TemporaryDirectory() as directory:
            for name in ('a.m4a', 'b.m4a'):
                Path(directory, name).touch()
            with mock.patch.object(cli, '_resolve_pipeline_settings', return_value=(None, None, False)), \
                 mock.patch.object(cli, '_build_stage_configs', return_value={}), \
                 mock.patch.object(cli, 'compress_audio', return_value='skipped'), \
                 mock.patch.object(cli, 'cached_transcription', side_effect=transcribe), \
                 mock.patch.object(cli, 'run_pipeline_with_progress', side_effect=notes), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main([directory, '--jobs', '2']), 0)
            self.assertEqual(len(list(Path(directory).glob('*.md'))), 2)


if __name__ == '__main__':
    unittest.main()
