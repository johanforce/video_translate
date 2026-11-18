from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from video_translator.core.models import TranscriptionProgress
from video_translator.tts.tts_service import VieNeuTTSService
from video_translator.tts.vieneu_client import VieNeuClient


class VoiceListWorker(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, api_key: str = "") -> None:
        super().__init__()
        self.api_key = api_key

    def run(self) -> None:
        try:
            voices = VieNeuClient.list_voices(self.api_key)
            self.loaded.emit(voices)
        except Exception as exc:
            self.failed.emit(str(exc))


class TTSWorker(QThread):
    progress_changed = Signal(object)
    log_message = Signal(str)
    succeeded = Signal(str, int, int)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        api_key: str,
        translated_json: Path,
        voice_id: str,
        speed: float,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.translated_json = translated_json
        self.voice_id = voice_id
        self.speed = speed
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            service = VieNeuTTSService(
                api_key=self.api_key,
                progress_callback=self.progress_changed.emit,
                log_callback=self.log_message.emit,
            )
            manifest, completed, total = service.run(
                translated_json=self.translated_json,
                voice_id=self.voice_id,
                speed=self.speed,
                cancel_event=self._cancel_event,
            )
            self.succeeded.emit(str(manifest), completed, total)
        except Exception as exc:
            if self._cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))
