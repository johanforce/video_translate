import tempfile
import unittest
from pathlib import Path

from video_translator.download.downloader import VideoDownloader


class DownloaderTests(unittest.TestCase):
    def test_finds_mp4_by_video_id_in_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            expected = output_dir / "A Film Title [abc123].mp4"
            expected.write_bytes(b"test")

            downloader = VideoDownloader(output_dir=output_dir)
            actual = downloader._find_output_file({"id": "abc123"}, "abc123")

            self.assertEqual(actual, expected)

    def test_respects_1080p_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = VideoDownloader(output_dir=Path(temp_dir), max_height=2160)
            self.assertEqual(downloader.max_height, 1080)

    def test_supports_lower_configured_height(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            downloader = VideoDownloader(output_dir=Path(temp_dir), max_height=720)
            self.assertEqual(downloader.max_height, 720)


if __name__ == "__main__":
    unittest.main()
