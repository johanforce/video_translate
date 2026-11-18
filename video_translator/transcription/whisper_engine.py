from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from video_translator.core.errors import TranscriptionCancelledError, VideoTranslatorError
from video_translator.core.models import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptionProgress,
    WordTimestamp,
)


ProgressCallback = Callable[[TranscriptionProgress], None]
LogCallback = Callable[[str], None]


class WhisperTranscriptionError(VideoTranslatorError):
    """Raised when faster-whisper cannot transcribe the extracted audio."""


class WhisperEngine:
    """Run multilingual faster-whisper transcription with segment and word timing."""

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.log_callback = log_callback

    def transcribe(
        self,
        audio_path: Path,
        model_size: str = "small",
        language: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> TranscriptDocument:
        cancel_event = cancel_event or threading.Event()
        audio_file = Path(audio_path)
        if not audio_file.is_file():
            raise WhisperTranscriptionError(f"Không tìm thấy audio: {audio_file}")

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise WhisperTranscriptionError(
                "Chưa cài faster-whisper. Hãy chạy: pip install -r requirements.txt"
            ) from exc

        if cancel_event.is_set():
            raise TranscriptionCancelledError("Đã hủy nhận diện lời thoại.")

        self._emit(TranscriptionProgress(15, "Đang khởi tạo Whisper..."))
        self._log(
            f"Whisper model: {model_size} | device=auto | compute_type=int8_float16/auto"
        )
        try:
            model = WhisperModel(model_size, device="auto", compute_type="default")
            if cancel_event.is_set():
                raise TranscriptionCancelledError("Đã hủy nhận diện lời thoại.")

            self._emit(TranscriptionProgress(20, "Đang nhận diện lời thoại..."))
            segments_iter, info = model.transcribe(
                str(audio_file),
                language=language,
                beam_size=5,
                best_of=5,
                vad_filter=True,
                word_timestamps=True,
                condition_on_previous_text=True,
            )

            duration = float(info.duration) if info.duration else None
            detected_language = str(info.language or "unknown")
            language_probability = (
                float(info.language_probability)
                if info.language_probability is not None
                else None
            )
            segments: list[TranscriptSegment] = []

            for raw_segment in segments_iter:
                if cancel_event.is_set():
                    raise TranscriptionCancelledError("Đã hủy nhận diện lời thoại.")

                words: list[WordTimestamp] = []
                for raw_word in raw_segment.words or []:
                    if raw_word.start is None or raw_word.end is None:
                        continue
                    words.append(
                        WordTimestamp(
                            start=float(raw_word.start),
                            end=float(raw_word.end),
                            text=str(raw_word.word),
                            probability=(
                                float(raw_word.probability)
                                if raw_word.probability is not None
                                else None
                            ),
                        )
                    )

                text = str(raw_segment.text or "").strip()
                if not text:
                    continue
                segments.append(
                    TranscriptSegment(
                        id=len(segments) + 1,
                        start=max(0.0, float(raw_segment.start)),
                        end=max(float(raw_segment.start), float(raw_segment.end)),
                        text=text,
                        speaker_id=None,
                        words=words,
                    )
                )

                percent = 20
                if duration and duration > 0:
                    percent = min(
                        95,
                        max(20, 20 + int(75 * float(raw_segment.end) / duration)),
                    )
                self._emit(
                    TranscriptionProgress(
                        percent,
                        f"Đã nhận diện {len(segments)} đoạn lời thoại...",
                        len(segments),
                    )
                )

        except TranscriptionCancelledError:
            raise
        except Exception as exc:
            raise WhisperTranscriptionError(f"Whisper nhận diện thất bại: {exc}") from exc

        self._log(
            f"Hoàn tất nhận diện: {len(segments)} đoạn | "
            f"ngôn ngữ={detected_language}"
            + (
                f" ({language_probability:.0%})"
                if language_probability is not None
                else ""
            )
        )
        self._emit(
            TranscriptionProgress(
                95,
                f"Đã nhận diện {len(segments)} đoạn. Đang lưu transcript...",
                len(segments),
            )
        )
        return TranscriptDocument(
            source_video="",
            language=detected_language,
            language_probability=language_probability,
            duration_seconds=duration,
            model_name=model_size,
            segments=segments,
        )

    def _emit(self, progress: TranscriptionProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
