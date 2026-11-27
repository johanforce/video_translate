from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from video_translator.core.batch_pipeline_service import (
    BatchPipelineCancelledError,
    BatchPipelineService,
)


class BatchPipelineWorker(QThread):
    """Run the multi-URL pipeline queue outside the Qt UI thread."""

    progress_changed = Signal(object)
    log_message = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        urls: list[str],
        api_keys: list[str],
        output_dir: Path,
        whisper_model: str,
        source_language: str | None,
        gemini_model: str,
        translation_batch_size: int,
        voice_id: str,
        voice_speed: float,
        subtitle_mode: str,
        original_audio_mode: str,
    ) -> None:
        super().__init__()
        self.arguments: dict[str, Any] = {
            "urls": urls,
            "api_keys": api_keys,
            "output_dir": output_dir,
            "whisper_model": whisper_model,
            "source_language": source_language,
            "gemini_model": gemini_model,
            "translation_batch_size": translation_batch_size,
            "voice_id": voice_id,
            "voice_speed": voice_speed,
            "subtitle_mode": subtitle_mode,
            "original_audio_mode": original_audio_mode,
        }
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            service = BatchPipelineService(
                progress_callback=self.progress_changed.emit,
                log_callback=self.log_message.emit,
            )
            results = service.run(
                **self.arguments,
                retry_delay_seconds=10.0,
                failures_per_key=5,
                cancel_event=self._cancel_event,
            )
            normalized = [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in result.items()
                }
                for result in results
            ]
            self.succeeded.emit(normalized)
        except Exception as exc:
            message = str(exc)
            if (
                self._cancel_event.is_set()
                or isinstance(exc, BatchPipelineCancelledError)
                or "đã hủy" in message.lower()
                or "đã huỷ" in message.lower()
            ):
                self.cancelled.emit()
            else:
                self.failed.emit(message)
