from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from video_translator.core.models import TranscriptionProgress
from video_translator.gemini.film_translation_service import FilmTranslationService
from video_translator.gemini.movie_context_service import MovieContextService


class GeminiWorker(QThread):
    progress_changed = Signal(object)
    log_message = Signal(str)
    context_succeeded = Signal(str)
    translation_succeeded = Signal(str, str, str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        mode: str,
        api_key: str,
        transcript_path: Path,
        context_path: Path | None,
        model: str,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.api_key = api_key
        self.transcript_path = transcript_path
        self.context_path = context_path
        self.model = model
        self.batch_size = batch_size
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            if self.mode == "context":
                service = MovieContextService(
                    api_key=self.api_key,
                    model=self.model,
                    progress_callback=self.progress_changed.emit,
                    log_callback=self.log_message.emit,
                )
                path = service.run(
                    self.transcript_path,
                    batch_segments=max(40, self.batch_size * 6),
                    cancel_event=self._cancel_event,
                )
                self.context_succeeded.emit(str(path))
            elif self.mode == "translate":
                if not self.context_path or not self.context_path.is_file():
                    raise ValueError("Chưa có Context Profile. Hãy phân tích ngữ cảnh phim trước.")
                service = FilmTranslationService(
                    api_key=self.api_key,
                    model=self.model,
                    progress_callback=self.progress_changed.emit,
                    log_callback=self.log_message.emit,
                )
                translated, subtitle, quality = service.run(
                    self.transcript_path,
                    self.context_path,
                    batch_size=self.batch_size,
                    cancel_event=self._cancel_event,
                )
                self.translation_succeeded.emit(
                    str(translated), str(subtitle), str(quality)
                )
            else:
                raise ValueError(f"Gemini worker mode không hợp lệ: {self.mode}")
        except Exception as exc:
            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
