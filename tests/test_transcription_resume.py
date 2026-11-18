import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_translator.core.models import TranscriptSegment, WordTimestamp
from video_translator.transcription.service import TranscriptionService


class FakeChunker:
    def __init__(self, chunk_seconds: int = 240) -> None:
        self.chunk_seconds = chunk_seconds

    def ensure_chunks(self, audio_path, chunk_dir, cancel_event=None):
        chunk_dir.mkdir(parents=True, exist_ok=True)
        first = chunk_dir / "chunk_0000.wav"
        second = chunk_dir / "chunk_0001.wav"
        if not first.exists():
            first.write_bytes(b"RIFF" + b"a" * 128)
        if not second.exists():
            second.write_bytes(b"RIFF" + b"b" * 128)
        return [first, second], 120.0


class FirstRunWhisper:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def transcribe_chunks(
        self,
        chunks,
        model_size,
        language,
        cancel_event,
        chunk_callback,
    ):
        self.requested_indexes = [item[0] for item in chunks]
        chunk_callback(
            0,
            [
                TranscriptSegment(
                    id=1,
                    start=1.0,
                    end=2.0,
                    text="Hello",
                    words=[WordTimestamp(1.0, 1.5, "Hello", 0.99)],
                )
            ],
            "en",
            0.98,
        )
        raise RuntimeError("simulated process/API interruption after chunk 1")


class SecondRunWhisper:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requested_indexes = []

    def transcribe_chunks(
        self,
        chunks,
        model_size,
        language,
        cancel_event,
        chunk_callback,
    ):
        self.requested_indexes = [item[0] for item in chunks]
        self.assert_only_pending_chunk = self.requested_indexes == [1]
        chunk_callback(
            1,
            [
                TranscriptSegment(
                    id=1,
                    start=61.0,
                    end=62.0,
                    text="Welcome back",
                    words=[
                        WordTimestamp(61.0, 61.4, "Welcome", 0.97),
                        WordTimestamp(61.5, 62.0, " back", 0.96),
                    ],
                )
            ],
            "en",
            0.98,
        )
        return "en", 0.98


class TranscriptionResumeTests(unittest.TestCase):
    def test_completed_whisper_chunk_is_not_reprocessed_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "film.mp4"
            video.write_bytes(b"fake video input")
            output = root / "output"
            output.mkdir()

            def fake_extract(source, target, cancel_event=None):
                target.write_bytes(b"RIFF" + b"x" * 256)
                return target

            first_engine = FirstRunWhisper()
            second_engine = SecondRunWhisper()

            with (
                patch(
                    "video_translator.transcription.service.AudioExtractor"
                ) as extractor,
                patch(
                    "video_translator.transcription.service.AudioChunker",
                    FakeChunker,
                ),
                patch(
                    "video_translator.transcription.service.WhisperEngine",
                    side_effect=[first_engine, second_engine],
                ),
            ):
                extractor.return_value.extract.side_effect = fake_extract
                first_service = TranscriptionService(chunk_seconds=60)
                with self.assertRaisesRegex(RuntimeError, "simulated process"):
                    first_service.run(video, output, model_size="small", language="en")

                partial_path = output / "film.transcript.json"
                partial = json.loads(partial_path.read_text(encoding="utf-8"))
                self.assertFalse(partial["is_complete"])
                self.assertEqual(
                    [segment["text"] for segment in partial["segments"]],
                    ["Hello"],
                )

                second_service = TranscriptionService(chunk_seconds=60)
                json_path, srt_path, language, segment_count = second_service.run(
                    video,
                    output,
                    model_size="small",
                    language="en",
                )

            self.assertEqual(first_engine.requested_indexes, [0, 1])
            self.assertEqual(second_engine.requested_indexes, [1])
            self.assertEqual(extractor.return_value.extract.call_count, 1)
            self.assertEqual(language, "en")
            self.assertEqual(segment_count, 2)

            completed = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(completed["is_complete"])
            self.assertEqual(
                [segment["text"] for segment in completed["segments"]],
                ["Hello", "Welcome back"],
            )
            self.assertEqual(
                [segment["id"] for segment in completed["segments"]],
                [1, 2],
            )
            self.assertTrue(srt_path.is_file())


if __name__ == "__main__":
    unittest.main()
