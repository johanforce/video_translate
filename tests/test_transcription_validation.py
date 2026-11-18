import tempfile
import unittest
from pathlib import Path

from video_translator.transcription.audio_extractor import (
    AudioExtractionError,
    AudioExtractor,
)
from video_translator.transcription.service import (
    TranscriptionService,
    TranscriptionServiceError,
)
from video_translator.transcription.whisper_engine import (
    WhisperEngine,
    WhisperTranscriptionError,
)


class TranscriptionValidationTests(unittest.TestCase):
    def test_audio_extractor_rejects_missing_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "missing.mp4"
            target = Path(temp_dir) / "audio.wav"
            with self.assertRaises(AudioExtractionError):
                AudioExtractor().extract(source, target)

    def test_whisper_engine_rejects_missing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "missing.wav"
            with self.assertRaises(WhisperTranscriptionError):
                WhisperEngine().transcribe(audio)

    def test_service_rejects_missing_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "missing.mp4"
            output = Path(temp_dir) / "output"
            with self.assertRaises(TranscriptionServiceError):
                TranscriptionService().run(source, output)


if __name__ == "__main__":
    unittest.main()
