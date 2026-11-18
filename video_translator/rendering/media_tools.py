from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, Any

from video_translator.core.errors import VideoTranslatorError


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, str], None]


class MediaToolError(VideoTranslatorError):
    """Raised when FFmpeg or ffprobe cannot process a media file."""


def require_media_tools() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    missing = []
    if not ffmpeg:
        missing.append("ffmpeg")
    if not ffprobe:
        missing.append("ffprobe")
    if missing:
        raise MediaToolError(
            "Không tìm thấy " + ", ".join(missing) +
            ". Hãy cài FFmpeg và thêm thư mục bin vào PATH."
        )
    return ffmpeg, ffprobe


def probe_media(path: Path) -> dict[str, Any]:
    _, ffprobe = require_media_tools()
    source = Path(path)
    if not source.is_file():
        raise MediaToolError(f"Không tìm thấy media: {source}")
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries",
                "format=duration,format_name:stream=codec_type,codec_name,width,height,sample_rate,channels",
                "-of", "json",
                str(source),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaToolError(f"ffprobe quá thời gian chờ: {source}") from exc
    if result.returncode != 0:
        raise MediaToolError(
            f"Không đọc được thông tin media: {result.stderr.strip()[-2000:]}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaToolError("ffprobe trả dữ liệu JSON không hợp lệ.") from exc
    if not isinstance(payload, dict):
        raise MediaToolError("ffprobe trả cấu trúc không hợp lệ.")
    return payload


def media_duration(path: Path) -> float:
    payload = probe_media(path)
    try:
        duration = float(payload.get("format", {}).get("duration"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise MediaToolError(f"Không xác định được thời lượng: {path}") from exc
    if duration <= 0:
        raise MediaToolError(f"Thời lượng media không hợp lệ: {path}")
    return duration


def run_ffmpeg(
    command: list[str],
    duration_seconds: float | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> None:
    """Run FFmpeg using its progress pipe; stdout and stderr share one drained pipe."""
    cancel_event = cancel_event or threading.Event()
    ffmpeg, _ = require_media_tools()
    if command and Path(command[0]).name.lower().startswith("ffmpeg"):
        command = [ffmpeg, *command[1:]]
    else:
        command = [ffmpeg, *command]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise MediaToolError(f"Không thể chạy FFmpeg: {exc}") from exc

    output_time = 0.0
    error_lines: list[str] = []
    assert process.stdout is not None
    try:
        while True:
            if cancel_event.is_set() and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise MediaToolError("Đã hủy tác vụ FFmpeg.")
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            stripped = line.strip()
            if not stripped:
                continue
            if "=" in stripped:
                key, value = stripped.split("=", 1)
                if key in {"out_time_us", "out_time_ms"}:
                    try:
                        number = float(value)
                        # FFmpeg's progress fields are microseconds despite the legacy ms name.
                        output_time = number / 1_000_000.0
                        if duration_seconds and duration_seconds > 0 and progress_callback:
                            percent = min(99.0, output_time / duration_seconds * 100.0)
                            progress_callback(percent, f"FFmpeg đang render... {percent:.1f}%")
                    except ValueError:
                        pass
            elif stripped not in {"progress=continue", "progress=end"}:
                error_lines.append(stripped)
                if len(error_lines) > 80:
                    error_lines.pop(0)
                if log_callback and ("error" in stripped.lower() or "warning" in stripped.lower()):
                    log_callback(stripped)
        return_code = process.wait()
    finally:
        # Close the progress pipe on success, cancellation and Python exceptions.
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()

    if return_code != 0:
        detail = "\n".join(error_lines[-25:])
        raise MediaToolError(
            f"FFmpeg thất bại (exit {return_code}).\n{detail or 'Không có log chi tiết.'}"
        )
    if progress_callback:
        progress_callback(100.0, "FFmpeg hoàn tất.")
