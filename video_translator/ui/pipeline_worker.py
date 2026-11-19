from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from video_translator.core.models import TranscriptionProgress
from video_translator.core.pipeline_service import ProjectPipelineService


class PipelineWorker(QThread):
    """Run the complete project or resume it from the first incomplete stage."""

    progress_changed = Signal(object)
    log_message = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        output_dir: Path,
        video_path: Path | None,
        video_url: str,
        transcript_path: Path | None,
        whisper_model: str,
        source_language: str | None,
        gemini_api_key: str,
        gemini_model: str,
        translation_batch_size: int,
        voice_id: str,
        voice_speed: float,
        subtitle_mode: str,
        original_audio_mode: str,
    ) -> None:
        super().__init__()
        self.arguments: dict[str, Any] = {
            "output_dir": output_dir,
            "video_path": video_path,
            "video_url": video_url,
            "transcript_path": transcript_path,
            "whisper_model": whisper_model,
            "source_language": source_language,
            "gemini_api_key": gemini_api_key,
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
            service = ProjectPipelineService(
                progress_callback=self.progress_changed.emit,
                log_callback=self.log_message.emit,
            )
            result = service.run(
                **self.arguments,
                cancel_event=self._cancel_event,
            )
            self.succeeded.emit({key: str(value) for key, value in result.items()})
        except Exception as exc:
            message = str(exc)
            if self._cancel_event.is_set() or "đã hủy" in message.lower() or "đã huỷ" in message.lower():
                self.cancelled.emit()
            else:
                self.failed.emit(message)
