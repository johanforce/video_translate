from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from video_translator.core.errors import TranscriptionCancelledError
from video_translator.core.models import TranscriptionProgress
from video_translator.transcription.service import TranscriptionService


class TranscriptionWorker(QThread):
    progress_changed = Signal(object)
    log_message = Signal(str)
    succeeded = Signal(str, str, str, int)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        model_size: str = "small",
        language: str | None = None,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.model_size = model_size
        self.language = language
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        service = TranscriptionService(
            progress_callback=self.progress_changed.emit,
            log_callback=self.log_message.emit,
        )
        try:
            json_path, srt_path, language, count = service.run(
                video_path=self.video_path,
                output_dir=self.output_dir,
                model_size=self.model_size,
                language=self.language,
                cancel_event=self._cancel_event,
            )
            self.succeeded.emit(str(json_path), str(srt_path), language, count)
        except TranscriptionCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
