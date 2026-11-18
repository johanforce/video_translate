from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable

from video_translator.core.errors import TranscriptionCancelledError, VideoTranslatorError
from video_translator.core.models import TranscriptionProgress
from video_translator.transcription.audio_extractor import AudioExtractor
from video_translator.transcription.transcript_writer import save_transcript
from video_translator.transcription.whisper_engine import WhisperEngine


ProgressCallback = Callable[[TranscriptionProgress], None]
LogCallback = Callable[[str], None]


class TranscriptionServiceError(VideoTranslatorError):
    """Raised when the transcription pipeline cannot be completed."""


class TranscriptionService:
    """Extract audio, transcribe it, then persist timestamped JSON and SRT files."""

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.log_callback = log_callback

    def run(
        self,
        video_path: Path,
        output_dir: Path,
        model_size: str = "small",
        language: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, Path, str, int]:
        source = Path(video_path).expanduser()
        destination = Path(output_dir).expanduser()
        cancel_event = cancel_event or threading.Event()

        if not source.is_file():
            raise TranscriptionServiceError(f"Không tìm thấy video: {source}")
        if shutil.which("ffmpeg") is None:
            raise TranscriptionServiceError(
                "Không tìm thấy FFmpeg. Hãy cài FFmpeg và thêm thư mục bin vào PATH."
            )
        if cancel_event.is_set():
            raise TranscriptionCancelledError("Đã hủy nhận diện lời thoại.")

        destination.mkdir(parents=True, exist_ok=True)
        self._log(f"Video nguồn: {source}")
        self._emit(TranscriptionProgress(2, "Chuẩn bị trích xuất audio..."))

        with tempfile.TemporaryDirectory(prefix="video_translator_audio_") as temp_dir:
            audio_path = Path(temp_dir) / "source_16k_mono.wav"
            AudioExtractor(self.log_callback).extract(
                source,
                audio_path,
                cancel_event,
            )
            if cancel_event.is_set():
                raise TranscriptionCancelledError("Đã hủy nhận diện lời thoại.")

            self._emit(TranscriptionProgress(15, "Audio đã sẵn sàng. Đang tải/khởi tạo Whisper..."))
            document = WhisperEngine(
                progress_callback=self.progress_callback,
                log_callback=self.log_callback,
            ).transcribe(
                audio_path=audio_path,
                model_size=model_size,
                language=language,
                cancel_event=cancel_event,
            )

        if cancel_event.is_set():
            raise TranscriptionCancelledError("Đã hủy nhận diện lời thoại.")

        json_path, srt_path = save_transcript(document, source, destination)
        self._emit(
            TranscriptionProgress(
                100,
                f"Hoàn tất: {len(document.segments)} đoạn lời thoại.",
                len(document.segments),
            )
        )
        self._log(f"Transcript JSON: {json_path}")
        self._log(f"Phụ đề SRT: {srt_path}")
        return json_path, srt_path, document.language, len(document.segments)

    def _emit(self, progress: TranscriptionProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
