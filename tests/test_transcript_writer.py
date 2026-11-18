import json
import tempfile
import unittest
from pathlib import Path

from video_translator.core.models import (
    TranscriptDocument,
    TranscriptSegment,
    WordTimestamp,
)
from video_translator.transcription.transcript_writer import (
    format_srt_timestamp,
    render_srt,
    save_transcript,
)


class TranscriptWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = TranscriptDocument(
            source_video="",
            language="en",
            language_probability=0.98,
            duration_seconds=8.0,
            model_name="small",
            segments=[
                TranscriptSegment(
                    id=1,
                    start=0.5,
                    end=2.25,
                    text="Hello everyone.",
                    words=[
                        WordTimestamp(0.5, 0.9, "Hello", 0.99),
                        WordTimestamp(1.0, 2.0, " everyone.", 0.98),
                    ],
                ),
                TranscriptSegment(
                    id=2,
                    start=3.0,
                    end=5.0,
                    text="Chúng ta bắt đầu nhé.",
                ),
            ],
        )

    def test_formats_srt_timestamp(self) -> None:
        self.assertEqual(format_srt_timestamp(65.125), "00:01:05,125")
        self.assertEqual(format_srt_timestamp(3661.007), "01:01:01,007")

    def test_renders_srt_with_real_line_breaks(self) -> None:
        actual = render_srt(self.document)
        self.assertIn("1\n00:00:00,500 --> 00:00:02,250\nHello everyone.", actual)
        self.assertIn("\n\n2\n00:00:03,000 --> 00:00:05,000\n", actual)
        self.assertTrue(actual.endswith("\n"))
        self.assertNotIn("\\n", actual)

    def test_saves_json_and_srt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "sample movie.mp4"
            video.write_bytes(b"video")
            output = root / "output"

            json_path, srt_path = save_transcript(self.document, video, output)

            self.assertTrue(json_path.is_file())
            self.assertTrue(srt_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["language"], "en")
            self.assertEqual(payload["segments"][0]["id"], 1)
            self.assertEqual(payload["segments"][0]["words"][0]["text"], "Hello")
            self.assertIn("Chúng ta bắt đầu nhé.", srt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
