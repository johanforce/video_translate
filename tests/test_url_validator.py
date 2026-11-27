import unittest

from video_translator.core.errors import InvalidVideoUrlError
from video_translator.download.url_validator import validate_video_url


class URLValidatorTests(unittest.TestCase):
    def test_accepts_youtube_watch_url(self) -> None:
        url = "https://www.youtube.com/watch?v=video123"
        self.assertEqual(validate_video_url(url), url)

    def test_accepts_short_youtube_url(self) -> None:
        url = "https://youtu.be/video123"
        self.assertEqual(validate_video_url(url), url)

    def test_accepts_bilibili_video_url(self) -> None:
        url = "https://www.bilibili.com/video/BV1234567890"
        self.assertEqual(validate_video_url(url), url)

    def test_accepts_dailymotion_short_url(self) -> None:
        url = "https://dai.ly/x60kz57"
        self.assertEqual(validate_video_url(url), url)

    def test_accepts_dailymotion_video_url(self) -> None:
        url = "https://www.dailymotion.com/video/x60kz57"
        self.assertEqual(validate_video_url(url), url)

    def test_rejects_unsupported_host(self) -> None:
        with self.assertRaises(InvalidVideoUrlError):
            validate_video_url("https://example.com/video")

    def test_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(InvalidVideoUrlError):
            validate_video_url("file:///tmp/video.mp4")

    def test_rejects_empty_url(self) -> None:
        with self.assertRaises(InvalidVideoUrlError):
            validate_video_url("")


if __name__ == "__main__":
    unittest.main()
