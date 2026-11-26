from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from video_translator.core.models import TranscriptionProgress
from video_translator.core.pipeline_service import ProjectPipelineService
from video_translator.download.url_validator import validate_video_url


ProgressCallback = Callable[[TranscriptionProgress], None]
LogCallback = Callable[[str], None]
PipelineFactory = Callable[..., ProjectPipelineService]


class BatchPipelineError(RuntimeError):
    """Raised when an automated batch cannot continue safely."""


class BatchApiKeysExhaustedError(BatchPipelineError):
    """Raised when every supplied Gemini key has exhausted its retry allowance."""


class BatchPipelineCancelledError(BatchPipelineError):
    """Raised when the user cancels a batch run."""


class BatchPipelineService:
    """Process a URL queue sequentially, resuming project checkpoints after failures."""

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.pipeline_factory = pipeline_factory or ProjectPipelineService

    def run(
        self,
        urls: list[str],
        api_keys: list[str],
        output_dir: Path,
        whisper_model: str = "small",
        source_language: str | None = None,
        gemini_model: str = "gemini-3.8-flash",
        translation_batch_size: int = 30,
        voice_id: str = "",
        voice_speed: float = 1.0,
        subtitle_mode: str = "both",
        original_audio_mode: str = "duck",
        retry_delay_seconds: float = 10.0,
        failures_per_key: int = 5,
        cancel_event: threading.Event | None = None,
    ) -> list[dict[str, Any]]:
        cancel_event = cancel_event or threading.Event()
        clean_urls = [validate_video_url(url) for url in urls if str(url).strip()]
        clean_keys = list(dict.fromkeys(str(key).strip() for key in api_keys if str(key).strip()))

        if not clean_urls:
            raise BatchPipelineError("Danh sách URL không có video hợp lệ.")
        if not clean_keys:
            raise BatchPipelineError("Cần ít nhất một Gemini API key.")
        if failures_per_key < 1:
            raise BatchPipelineError("Số lần lỗi tối đa cho mỗi key phải lớn hơn 0.")
        if retry_delay_seconds < 0:
            raise BatchPipelineError("Thời gian chờ retry không được âm.")

        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        active_key_index = 0
        total_urls = len(clean_urls)

        for url_index, url in enumerate(clean_urls):
            self._check_cancel(cancel_event)
            failures_with_key = 0
            progress_callback = self._url_progress(url_index, total_urls)

            while True:
                self._check_cancel(cancel_event)
                key_label = f"{active_key_index + 1}/{len(clean_keys)}"
                attempt_label = failures_with_key + 1
                self._log(
                    f"[BATCH {url_index + 1}/{total_urls}] Bắt đầu URL: {url} "
                    f"(Gemini key {key_label}, lần thử {attempt_label}/{failures_per_key})."
                )
                try:
                    pipeline = self.pipeline_factory(
                        progress_callback=progress_callback,
                        log_callback=self.log_callback,
                    )
                    result = pipeline.run(
                        output_dir=destination,
                        video_url=url,
                        whisper_model=whisper_model,
                        source_language=source_language,
                        gemini_api_key=clean_keys[active_key_index],
                        gemini_model=gemini_model,
                        translation_batch_size=translation_batch_size,
                        voice_id=voice_id,
                        voice_speed=voice_speed,
                        subtitle_mode=subtitle_mode,
                        original_audio_mode=original_audio_mode,
                        cancel_event=cancel_event,
                    )
                    results.append(result)
                    self._log(
                        f"[BATCH {url_index + 1}/{total_urls}] Hoàn tất: "
                        f"{result.get('video', 'đã hoàn tất project')}."
                    )
                    self._emit_progress(
                        TranscriptionProgress(
                            percent=int((url_index + 1) * 100 / total_urls),
                            message=f"Đã hoàn tất video {url_index + 1}/{total_urls}.",
                            segments_done=0,
                        )
                    )
                    break
                except Exception as exc:
                    self._check_cancel(cancel_event)
                    failures_with_key += 1
                    if failures_with_key >= failures_per_key:
                        if active_key_index + 1 >= len(clean_keys):
                            safe_error = self._redact_secrets(str(exc), clean_keys)
                            message = (
                                f"Hết Gemini API key sau {failures_per_key} lần lỗi "
                                f"với URL: {url}. Lỗi cuối: {safe_error}"
                            )
                            self._log("[BATCH DỪNG] " + message)
                            raise BatchApiKeysExhaustedError(message) from exc
                        active_key_index += 1
                        failures_with_key = 0
                        self._log(
                            f"[BATCH] Key hiện tại đã lỗi {failures_per_key} lần. "
                            f"Chuyển sang Gemini key {active_key_index + 1}/{len(clean_keys)}; "
                            "project sẽ tiếp tục từ checkpoint."
                        )
                    else:
                        safe_error = self._redact_secrets(str(exc), clean_keys)
                        self._log(
                            f"[BATCH] Lỗi lần {failures_with_key}/{failures_per_key}: "
                            f"{safe_error}"
                        )

                    self._log(
                        f"[BATCH] Chờ {retry_delay_seconds:g} giây rồi tự động thử lại URL..."
                    )
                    if cancel_event.wait(retry_delay_seconds):
                        raise BatchPipelineCancelledError(
                            "Đã hủy hàng đợi trong thời gian chờ retry."
                        ) from exc

        self._log(
            f"[BATCH HOÀN TẤT] Đã xử lý thành công {len(results)}/{total_urls} video."
        )
        return results

    def _url_progress(
        self,
        url_index: int,
        total_urls: int,
    ) -> ProgressCallback:
        def callback(progress: TranscriptionProgress) -> None:
            if progress.percent is None:
                overall = None
            else:
                fraction = max(0.0, min(100.0, float(progress.percent))) / 100.0
                overall = int((url_index + fraction) * 100 / total_urls)
            self._emit_progress(
                TranscriptionProgress(
                    percent=overall,
                    message=f"[Video {url_index + 1}/{total_urls}] {progress.message}",
                    segments_done=progress.segments_done,
                )
            )
        return callback

    def _emit_progress(self, progress: TranscriptionProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    @staticmethod
    def _redact_secrets(message: str, api_keys: list[str]) -> str:
        redacted = message
        for api_key in api_keys:
            if api_key:
                redacted = redacted.replace(api_key, "[REDACTED]")
        return redacted

    @staticmethod
    def _check_cancel(cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            raise BatchPipelineCancelledError(
                "Đã hủy hàng đợi. Checkpoint của video đang xử lý được giữ lại."
            )
