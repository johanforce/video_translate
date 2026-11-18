from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from video_translator.core.errors import DownloadCancelledError
from video_translator.core.models import DownloadProgress
from video_translator.download.downloader import VideoDownloader


class DownloadWorker(QThread):
    progress_changed = Signal(object)
    log_message = Signal(str)
    succeeded = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, url: str, output_dir: Path, max_height: int = 1080) -> None:
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        self.max_height = max_height
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        downloader = VideoDownloader(
            output_dir=self.output_dir,
            max_height=self.max_height,
            progress_callback=self.progress_changed.emit,
            log_callback=self.log_message.emit,
            cancel_event=self._cancel_event,
        )
        try:
            result = downloader.download(self.url)
            self.succeeded.emit(str(result.path))
        except DownloadCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
