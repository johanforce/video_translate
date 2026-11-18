from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from video_translator.core.errors import TranscriptionCancelledError, VideoTranslatorError


class AudioChunkError(VideoTranslatorError):
    """Raised when persistent audio chunks cannot be prepared."""


class AudioChunker:
    """Split extracted WAV into deterministic, reusable Whisper-sized chunks."""

    def __init__(self, chunk_seconds: int = 240) -> None:
        self.chunk_seconds = max(30, min(int(chunk_seconds), 900))

    @staticmethod
    def duration(audio_path: Path) -> float:
        if shutil.which("ffprobe") is None:
            raise AudioChunkError("Không tìm thấy ffprobe trong PATH.")
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise AudioChunkError(
                f"Không đọc được thời lượng audio: {result.stderr.strip()[-1000:]}"
            )
        try:
            duration = float(result.stdout.strip())
        except (TypeError, ValueError) as exc:
            raise AudioChunkError("ffprobe không trả thời lượng audio hợp lệ.") from exc
        if duration <= 0:
            raise AudioChunkError("Audio có thời lượng bằng 0.")
        return duration

    def ensure_chunks(
        self,
        audio_path: Path,
        chunk_dir: Path,
        cancel_event: threading.Event | None = None,
    ) -> tuple[list[Path], float]:
        cancel_event = cancel_event or threading.Event()
        source = Path(audio_path)
        destination = Path(chunk_dir)
        if not source.is_file():
            raise AudioChunkError(f"Không tìm thấy audio nguồn: {source}")
        if shutil.which("ffmpeg") is None:
            raise AudioChunkError("Không tìm thấy FFmpeg trong PATH.")
        duration = self.duration(source)
        expected_count = max(1, int(math.ceil(duration / self.chunk_seconds)))
        destination.mkdir(parents=True, exist_ok=True)
        chunks = sorted(destination.glob("chunk_*.wav"))

        if len(chunks) == expected_count and all(
            path.is_file() and path.stat().st_size > 44 for path in chunks
        ):
            return chunks, duration

        # A partial split is never trusted: rebuild all chunks from the cached WAV.
        for path in destination.glob("chunk_*"):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

        if cancel_event.is_set():
            raise TranscriptionCancelledError("Đã hủy chia audio.")
        temporary_dir = destination / "split_tmp"
        shutil.rmtree(temporary_dir, ignore_errors=True)
        temporary_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = temporary_dir / "chunk_%04d.wav"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(source),
            "-map", "0:a:0", "-vn",
            "-f", "segment",
            "-segment_time", str(self.chunk_seconds),
            "-reset_timestamps", "1",
            "-segment_format", "wav",
            "-c:a", "pcm_s16le",
            str(output_pattern),
        ]

        with tempfile.TemporaryFile(mode="w+b") as stderr_file:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                )
            except OSError as exc:
                raise AudioChunkError(f"Không thể chạy FFmpeg segment: {exc}") from exc

            while process.poll() is None:
                if cancel_event.wait(0.2):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    shutil.rmtree(temporary_dir, ignore_errors=True)
                    raise TranscriptionCancelledError("Đã hủy chia audio.")
            stderr_file.seek(0)
            stderr = stderr_file.read().decode("utf-8", errors="replace")

        if process.returncode != 0:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise AudioChunkError(
                f"FFmpeg chia audio thất bại: {stderr.strip()[-2000:]}"
            )

        generated = sorted(temporary_dir.glob("chunk_*.wav"))
        generated = [
            path for path in generated
            if path.is_file() and path.stat().st_size > 44
        ]
        if len(generated) != expected_count:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise AudioChunkError(
                f"FFmpeg tạo {len(generated)} chunk, dự kiến {expected_count}. "
                "Hãy kiểm tra audio nguồn."
            )

        for index, path in enumerate(generated):
            path.replace(destination / f"chunk_{index:04d}.wav")
        shutil.rmtree(temporary_dir, ignore_errors=True)
        return sorted(destination.glob("chunk_*.wav")), duration
