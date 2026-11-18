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
}


def validate_video_url(url: str) -> str:
    """Validate and normalize a supported YouTube or Bilibili URL."""
    candidate = (url or "").strip()
    if not candidate:
        raise InvalidVideoUrlError("Vui lòng nhập link video YouTube hoặc Bilibili.")

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise InvalidVideoUrlError("Link video phải bắt đầu bằng http:// hoặc https://.")

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname not in _SUPPORTED_HOSTS:
        raise InvalidVideoUrlError(
            "Nền tảng chưa được hỗ trợ. Hiện tại chỉ nhận link YouTube và Bilibili."
        )

    if not parsed.path or parsed.path == "/":
        raise InvalidVideoUrlError("Link không chứa mã hoặc đường dẫn video hợp lệ.")

    return candidate
