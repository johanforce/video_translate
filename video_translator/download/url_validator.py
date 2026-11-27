from __future__ import annotations

from urllib.parse import urlparse

from video_translator.core.errors import InvalidVideoUrlError


_SUPPORTED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "bilibili.com",
    "www.bilibili.com",
    "m.bilibili.com",
    "b23.tv",
    "bilibili.tv",
    "www.bilibili.tv",
    # Dailymotion full and short video URLs.
    "dailymotion.com",
    "www.dailymotion.com",
    "m.dailymotion.com",
    "dai.ly",
}


def validate_video_url(url: str) -> str:
    """Validate and normalize a supported YouTube, Bilibili or Dailymotion URL."""
    candidate = (url or "").strip()
    if not candidate:
        raise InvalidVideoUrlError(
            "Vui lòng nhập link video YouTube, Bilibili hoặc Dailymotion."
        )

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise InvalidVideoUrlError("Link video phải bắt đầu bằng http:// hoặc https://.")

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname not in _SUPPORTED_HOSTS:
        raise InvalidVideoUrlError(
            "Nền tảng chưa được hỗ trợ. Hiện tại ứng dụng nhận link "
            "YouTube, Bilibili và Dailymotion."
        )

    if not parsed.path or parsed.path == "/":
        raise InvalidVideoUrlError("Link không chứa mã hoặc đường dẫn video hợp lệ.")

    return candidate
