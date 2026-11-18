import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from video_translator.transcription.audio_chunker import AudioChunker


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
class AudioChunkerTests(unittest.TestCase):
    def test_chunks_audio_and_reuses_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi",
                    "-i", "sine=frequency=440:sample_rate=16000:duration=65",
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    str(source),
                ],
                check=True,
                capture_output=True,
            )

            chunker = AudioChunker(chunk_seconds=60)
            chunk_dir = root / "chunks"
            chunks, duration = chunker.ensure_chunks(source, chunk_dir)
            self.assertGreaterEqual(duration, 64.9)
            self.assertEqual(len(chunks), 2)
            self.assertTrue(all(path.stat().st_size > 44 for path in chunks))
            timestamps = [path.stat().st_mtime_ns for path in chunks]

            reused, reused_duration = chunker.ensure_chunks(source, chunk_dir)
            self.assertEqual(reused, chunks)
            self.assertEqual(reused_duration, duration)
            self.assertEqual([path.stat().st_mtime_ns for path in reused], timestamps)


if __name__ == "__main__":
    unittest.main()
