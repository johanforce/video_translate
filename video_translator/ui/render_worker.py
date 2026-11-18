from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from video_translator.core.models import TranscriptionProgress
from video_translator.rendering.render_service import RenderService


class RenderWorker(QThread):
    progress_changed = Signal(object)
    log_message = Signal(str)
    succeeded = Signal(str, str, bool)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        translated_json: Path,
        subtitle_mode: str,
        original_audio_mode: str,
        output_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.translated_json = translated_json
        self.subtitle_mode = subtitle_mode
        self.original_audio_mode = original_audio_mode
        self.output_path = output_path
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            service = RenderService(
                progress_callback=self.progress_changed.emit,
                log_callback=self.log_message.emit,
            )
            output, metadata, cached = service.run(
                translated_json=self.translated_json,
                subtitle_mode=self.subtitle_mode,
                original_audio_mode=self.original_audio_mode,
                output_path=self.output_path,
                cancel_event=self._cancel_event,
            )
            self.succeeded.emit(str(output), str(metadata), cached)
        except Exception as exc:
            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
