from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

from video_translator.core.errors import TranscriptionCancelledError, VideoTranslatorError


LogCallback = Callable[[str], None]


class AudioExtractionError(VideoTranslatorError):
    """Raised when FFmpeg cannot extract audio from the source video."""


class AudioExtractor:
    """Extract mono 16 kHz PCM audio for speech recognition."""

    def __init__(self, log_callback: LogCallback | None = None) -> None:
        self.log_callback = log_callback

    def extract(
        self,
        video_path: Path,
        audio_path: Path,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        source = Path(video_path).expanduser()
        target = Path(audio_path).expanduser()
        cancel_event = cancel_event or threading.Event()

        if not source.is_file():
            raise AudioExtractionError(f"Không tìm thấy video: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(target),
        ]
        self._log("FFmpeg: đang trích xuất audio mono 16 kHz...")

        # A temporary file prevents a full stderr pipe from blocking FFmpeg.
        with tempfile.TemporaryFile(mode="w+b") as stderr_file:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                )
            except FileNotFoundError as exc:
                raise AudioExtractionError(
                    "Không tìm thấy FFmpeg. Hãy cài FFmpeg và thêm thư mục bin vào PATH."
                ) from exc
            except OSError as exc:
                raise AudioExtractionError(f"Không thể khởi chạy FFmpeg: {exc}") from exc

            while process.poll() is None:
                if cancel_event.wait(0.2):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise TranscriptionCancelledError("Đã hủy trích xuất audio.")

            stderr_file.seek(0)
            stderr = stderr_file.read().decode("utf-8", errors="replace")

        if process.returncode != 0:
            detail = stderr.strip()[-2000:]
            raise AudioExtractionError(
                "FFmpeg không trích xuất được audio. "
                + (detail if detail else "Video có thể không có track audio.")
            )
        if not target.is_file() or target.stat().st_size <= 44:
            raise AudioExtractionError(
                "Không tìm thấy dữ liệu audio sau khi trích xuất. "
                "Video có thể không có track âm thanh."
            )

        self._log(f"Đã trích xuất audio: {target.stat().st_size / (1024 * 1024):.1f} MB")
        return target

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
