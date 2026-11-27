from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Callable, Any

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from video_translator.core.errors import (
    DownloadCancelledError,
    VideoDownloadError,
)
from video_translator.core.models import DownloadProgress, DownloadResult
from video_translator.download.url_validator import validate_video_url


ProgressCallback = Callable[[DownloadProgress], None]
LogCallback = Callable[[str], None]


class _YtDlpLogger:
    def __init__(self, callback: LogCallback | None) -> None:
        self._callback = callback

    def debug(self, message: str) -> None:
        if self._callback and message.strip():
            self._callback(message.strip())

    def info(self, message: str) -> None:
        if self._callback and message.strip():
            self._callback(message.strip())

    def warning(self, message: str) -> None:
        if self._callback and message.strip():
            self._callback(f"Cảnh báo: {message.strip()}")

    def error(self, message: str) -> None:
        if self._callback and message.strip():
            self._callback(f"Lỗi: {message.strip()}")


class VideoDownloader:
    """Download one YouTube/Bilibili video and produce an MP4 file."""

    def __init__(
        self,
        output_dir: Path,
        max_height: int = 1080,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.max_height = max(144, min(int(max_height), 1080))
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.cancel_event = cancel_event or threading.Event()

    def download(self, url: str) -> DownloadResult:
        normalized_url = validate_video_url(url)
        if shutil.which("ffmpeg") is None:
            raise VideoDownloadError(
                "Không tìm thấy FFmpeg. Hãy cài FFmpeg và thêm thư mục bin vào PATH."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        options: dict[str, Any] = {
            "format": (
                f"bv*[height<={self.max_height}][ext=mp4]+ba[ext=m4a]/"
                f"b[height<={self.max_height}][ext=mp4]/"
                f"bv*[height<={self.max_height}]+ba/"
                f"b[height<={self.max_height}]"
            ),
            "outtmpl": str(self.output_dir / "%(title).160B [%(id)s].%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": False,
            "logger": _YtDlpLogger(self.log_callback),
            "merge_output_format": "mp4",
            "postprocessors": [
                {
                    "key": "FFmpegVideoRemuxer",
                    "preferedformat": "mp4",
                }
            ],
            "progress_hooks": [self._progress_hook],
            "continuedl": True,
            "overwrites": False,
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
        }

        self._log("Đang kiểm tra thông tin video...")
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(normalized_url, download=True)
        except DownloadCancelledError:
            raise
        except DownloadError as exc:
            raise VideoDownloadError(self._friendly_download_error(str(exc))) from exc
        except Exception as exc:
            raise VideoDownloadError(f"Không thể tải video: {exc}") from exc

        if not isinstance(info, dict):
            raise VideoDownloadError("yt-dlp không trả về thông tin video hợp lệ.")

        video_id = str(info.get("id") or "unknown")
        output_path = self._find_output_file(info, video_id)
        if output_path is None:
            raise VideoDownloadError(
                "Tải xuống đã kết thúc nhưng không tìm thấy file MP4 đầu ra. "
                "Hãy kiểm tra log FFmpeg và thư mục lưu."
            )

        source_height = info.get("height")
        if not isinstance(source_height, int):
            heights = [
                item.get("height")
                for item in (info.get("requested_formats") or [])
                if isinstance(item, dict) and isinstance(item.get("height"), int)
            ]
            source_height = max(heights) if heights else None

        self._log(f"Hoàn tất: {output_path.name}")
        self._emit_progress(
            DownloadProgress(percent=100.0, message="Tải video hoàn tất.")
        )
        return DownloadResult(
            video_id=video_id,
            title=str(info.get("title") or output_path.stem),
            path=output_path,
            duration_seconds=(
                float(info["duration"])
                if isinstance(info.get("duration"), (int, float))
                else None
            ),
            source_height=source_height,
        )

    def _progress_hook(self, data: dict[str, Any]) -> None:
        if self.cancel_event.is_set():
            raise DownloadCancelledError("Người dùng đã hủy tải xuống.")

        status = data.get("status")
        if status == "downloading":
            downloaded = data.get("downloaded_bytes")
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            downloaded = downloaded if isinstance(downloaded, int) else None
            total = total if isinstance(total, int) else None
            percent = (downloaded / total * 100.0) if downloaded and total else None
            speed = data.get(" speed") or data.get("speed")
            eta = data.get("eta")
            message = "Đang tải video..."
            if percent is not None:
                message = f"Đang tải video... {percent:.1f}%"
            self._emit_progress(
                DownloadProgress(
                    percent=percent,
                    message=message,
                    downloaded_bytes=downloaded,
                    total_bytes=total,
                    speed_bytes_per_second=(
                        float(speed) if isinstance(speed, (int, float)) else None
                    ),
                    eta_seconds=eta if isinstance(eta, int) else None,
                )
            )
        elif status == "finished":
            self._emit_progress(
                DownloadProgress(
                    percent=100.0,
                    message="Đang ghép audio/video và chuyển container MP4...",
                )
            )

    def _find_output_file(self, info: dict[str, Any], video_id: str) -> Path | None:
        filepath = info.get("filepath")
        if isinstance(filepath, str):
            candidate = Path(filepath)
            if candidate.is_file() and candidate.suffix.lower() == ".mp4":
                return candidate

        requested_downloads = info.get("requested_downloads")
        if isinstance(requested_downloads, list):
            for item in requested_downloads:
                if not isinstance(item, dict):
                    continue
                requested_path = item.get("filepath")
                if isinstance(requested_path, str):
                    candidate = Path(requested_path).with_suffix(".mp4")
                    if candidate.is_file():
                        return candidate

        prepared = info.get("_filename")
        if isinstance(prepared, str):
            candidate = Path(prepared).with_suffix(".mp4")
            if candidate.is_file():
                return candidate

        # The title template embeds the video ID in square brackets. Match
        # the ID as a substring; square brackets have special meaning in glob().
        candidates = sorted(
            self.output_dir.glob(f"*{video_id}*.mp4"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _emit_progress(self, progress: DownloadProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    @staticmethod
    def _friendly_download_error(message: str) -> str:
        lowered = message.lower()
        if "impersonation" in lowered or "impersonate targets are available" in lowered:
            return (
                "Dailymotion yêu cầu thư viện giả lập trình duyệt nhưng môi trường "
                "chưa có dependency phù hợp (curl-cffi). Hãy chạy trong đúng "
                "Python environment của ứng dụng: "
                'python -m pip install -U "yt-dlp[default,curl-cffi]" '
                "sau đó khởi động lại ứng dụng."
            )
        if "private video" in lowered or "private" in lowered:
            return "Video riêng tư hoặc không có quyền truy cập."
        if "unsupported url" in lowered:
            return "yt-dlp không nhận diện được URL này. Hãy kiểm tra lại link."
        if "ffmpeg" in lowered:
            return "FFmpeg gặp lỗi khi ghép hoặc chuyển video sang MP4."
        return message.strip() or "Không xác định được nguyên nhân tải xuống thất bại."
