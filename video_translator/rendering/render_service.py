from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from video_translator.core.models import TranscriptionProgress
from video_translator.rendering.video_renderer import VideoRenderer


ProgressCallback = Callable[[TranscriptionProgress], None]
LogCallback = Callable[[str], None]


class RenderService:
    """Public orchestration API for building narration and final video."""

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.log_callback = log_callback

    def run(
        self,
        translated_json: Path,
        subtitle_mode: str = "both",
        original_audio_mode: str = "duck",
        output_path: Path | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, Path, bool]:
        renderer = VideoRenderer(
            progress_callback=self.progress_callback,
            log_callback=self.log_callback,
        )
        return renderer.render(
            translated_json=translated_json,
            subtitle_mode=subtitle_mode,
            original_audio_mode=original_audio_mode,
            output_path=output_path,
            cancel_event=cancel_event,
        )
