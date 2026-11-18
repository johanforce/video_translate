from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from video_translator.core.errors import TranscriptionCancelledError, VideoTranslatorError
from video_translator.core.models import (
    TranscriptDocument,
    TranscriptSegment,
    TranscriptionProgress,
    WordTimestamp,
)
from video_translator.gemini.json_utils import (
    read_json,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from video_translator.transcription.audio_chunker import AudioChunker
from video_translator.transcription.audio_extractor import AudioExtractor
from video_translator.transcription.transcript_writer import save_transcript
from video_translator.transcription.whisper_engine import WhisperEngine


ProgressCallback = Callable[[TranscriptionProgress], None]
LogCallback = Callable[[str], None]


class TranscriptionServiceError(VideoTranslatorError):
    """Raised when the transcription pipeline cannot be completed."""


class TranscriptionService:
    """Persistent, chunk-based Whisper pipeline with crash-safe checkpoints."""

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
        chunk_seconds: int = 240,
    ) -> None:
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.chunk_seconds = max(30, min(int(chunk_seconds), 900))

    def run(
        self,
        video_path: Path,
        output_dir: Path,
        model_size: str = "small",
        language: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, Path, str, int]:
        source = Path(video_path).expanduser().resolve()
        destination = Path(output_dir).expanduser().resolve()
        cancel_event = cancel_event or threading.Event()

        if not source.is_file():
            raise TranscriptionServiceError(f"Không tìm thấy video: {source}")
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise TranscriptionServiceError(
                "Không tìm thấy FFmpeg/ffprobe. Hãy cài FFmpeg và thêm thư mục bin vào PATH."
            )
        if cancel_event.is_set():
            raise TranscriptionCancelledError("Đã hủy nhận diện lời thoại.")

        destination.mkdir(parents=True, exist_ok=True)
        source_fingerprint = sha256_file(source)
        language_key = (language or "auto").strip().lower()
        config_signature = sha256_text(
            f"{source_fingerprint}|{model_size}|{language_key}|{self.chunk_seconds}|"
            "mono16k-pcm|word-timestamps-v1"
        )

        cache_dir = destination / ".video_translator_cache" / source.stem
        chunk_dir = cache_dir / "chunks"
        audio_path = cache_dir / "source_16k_mono.wav"
        audio_meta_path = cache_dir / "audio_meta.json"
        checkpoint_path = cache_dir / "transcription_state.json"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"Video nguồn: {source}")
        self._log(f"Transcription cache: {cache_dir}")
        self._emit(TranscriptionProgress(1, "Kiểm tra cache audio và Whisper..."))

        audio_valid = self._is_audio_cache_valid(
            audio_path,
            audio_meta_path,
            source_fingerprint,
        )
        if not audio_valid:
            if cancel_event.is_set():
                raise TranscriptionCancelledError("Đã hủy trích xuất audio.")
            self._log("Không có audio cache hợp lệ; đang trích xuất lại bằng FFmpeg.")
            temporary_audio = audio_path.with_name(audio_path.name + ".partial")
            temporary_audio.unlink(missing_ok=True)
            AudioExtractor(self.log_callback).extract(
                source,
                temporary_audio,
                cancel_event,
            )
            if cancel_event.is_set():
                temporary_audio.unlink(missing_ok=True)
                raise TranscriptionCancelledError("Đã hủy trích xuất audio.")
            temporary_audio.replace(audio_path)
            write_json_atomic(
                audio_meta_path,
                {
                    "schema_version": 1,
                    "source_video": str(source),
                    "source_fingerprint": source_fingerprint,
                    "audio_path": str(audio_path),
                    "audio_fingerprint": sha256_file(audio_path),
                },
            )
            shutil.rmtree(chunk_dir, ignore_errors=True)
        else:
            self._log("Đã tìm thấy audio cache hợp lệ; bỏ qua trích xuất.")

        chunker = AudioChunker(self.chunk_seconds)
        chunk_paths, duration = chunker.ensure_chunks(
            audio_path,
            chunk_dir,
            cancel_event,
        )
        chunk_fingerprints = {
            index: sha256_file(path)
            for index, path in enumerate(chunk_paths)
        }
        self._log(
            f"Audio dài {duration:.1f}s, chia thành {len(chunk_paths)} chunk "
            f"({self.chunk_seconds}s/chunk)."
        )

        state = self._load_checkpoint(
            checkpoint_path,
            config_signature,
            source_fingerprint,
            model_size,
            language_key,
            len(chunk_paths),
        )
        completed_chunks = state.setdefault("chunks", {})

        pending: list[tuple[int, Path, float]] = []
        for index, path in enumerate(chunk_paths):
            saved = completed_chunks.get(str(index))
            if (
                isinstance(saved, dict)
                and saved.get("chunk_fingerprint") == chunk_fingerprints[index]
                and isinstance(saved.get("segments"), list)
            ):
                continue
            pending.append((index, path, index * self.chunk_seconds))

        completed_count = len(chunk_paths) - len(pending)
        self._log(
            f"Whisper checkpoint: {completed_count}/{len(chunk_paths)} chunk đã hoàn tất; "
            f"còn {len(pending)} chunk cần nhận diện."
        )

        language_result = str(state.get("detected_language") or language or "unknown")
        language_probability = state.get("language_probability")
        language_probability = (
            float(language_probability)
            if isinstance(language_probability, (int, float))
            else None
        )

        def save_chunk(
            chunk_index: int,
            chunk_segments: list[TranscriptSegment],
            detected_language: str,
            probability: float | None,
        ) -> None:
            nonlocal completed_count, language_result, language_probability
            chunk_payload = [
                self._segment_to_dict(segment)
                for segment in chunk_segments
            ]
            completed_chunks[str(chunk_index)] = {
                "chunk_fingerprint": chunk_fingerprints[chunk_index],
                "segments": chunk_payload,
                "segment_count": len(chunk_payload),
                "detected_language": detected_language,
                "language_probability": probability,
            }
            completed_count = len(
                [
                    key for key, value in completed_chunks.items()
                    if key.isdigit()
                    and isinstance(value, dict)
                    and value.get("chunk_fingerprint") == chunk_fingerprints.get(int(key))
                ]
            )
            if detected_language and detected_language != "unknown":
                language_result = detected_language
            if language_probability is None and probability is not None:
                language_probability = probability

            state.update(
                {
                    "schema_version": 1,
                    "config_signature": config_signature,
                    "source_video": str(source),
                    "source_fingerprint": source_fingerprint,
                    "audio_fingerprint": sha256_file(audio_path),
                    "model": model_size,
                    "requested_language": language_key,
                    "detected_language": language_result,
                    "language_probability": language_probability,
                    "chunk_seconds": self.chunk_seconds,
                    "total_chunks": len(chunk_paths),
                    "completed_chunks": completed_count,
                    "is_complete": False,
                    "chunks": completed_chunks,
                }
            )
            write_json_atomic(checkpoint_path, state)

            partial_segments = self._collect_segments(
                completed_chunks,
                chunk_fingerprints,
            )
            partial_document = TranscriptDocument(
                source_video=str(source),
                language=language_result,
                language_probability=language_probability,
                duration_seconds=duration,
                model_name=model_size,
                segments=partial_segments,
            )
            save_transcript(
                partial_document,
                source,
                destination,
                is_complete=False,
                transcription_signature=config_signature,
                checkpoint_path=checkpoint_path,
            )
            percent = int(completed_count / len(chunk_paths) * 100)
            self._emit(
                TranscriptionProgress(
                    min(98, max(5, percent)),
                    f"Đã lưu checkpoint Whisper {completed_count}/{len(chunk_paths)} chunk.",
                    len(partial_segments),
                )
            )
            self._log(
                f"Đã checkpoint chunk {chunk_index + 1}/{len(chunk_paths)}: "
                f"{len(chunk_segments)} đoạn thoại."
            )

        if pending:
            engine = WhisperEngine(
                progress_callback=lambda progress: self._emit(
                    TranscriptionProgress(
                        min(94, max(5, progress.percent or 5)),
                        progress.message,
                        progress.segments_done,
                    )
                ),
                log_callback=self.log_callback,
            )
            language_result, language_probability = engine.transcribe_chunks(
                chunks=pending,
                model_size=model_size,
                language=language,
                cancel_event=cancel_event,
                chunk_callback=save_chunk,
            )
            state["detected_language"] = language_result
            state["language_probability"] = language_probability
        else:
            self._log("Tất cả chunk đã có cache Whisper; không chạy model lần nữa.")

        if cancel_event.is_set():
            raise TranscriptionCancelledError(
                "Đã hủy nhận diện lời thoại. Các chunk đã hoàn tất vẫn được giữ trong cache."
            )

        segments = self._collect_segments(completed_chunks, chunk_fingerprints)
        if completed_count != len(chunk_paths):
            raise TranscriptionServiceError(
                "Transcript chưa đủ chunk. Cache được giữ; chạy lại để tiếp tục."
            )

        document = TranscriptDocument(
            source_video=str(source),
            language=language_result,
            language_probability=language_probability,
            duration_seconds=duration,
            model_name=model_size,
            segments=segments,
        )
        state.update(
            {
                "schema_version": 1,
                "config_signature": config_signature,
                "source_video": str(source),
                "source_fingerprint": source_fingerprint,
                "audio_fingerprint": sha256_file(audio_path),
                "model": model_size,
                "requested_language": language_key,
                "detected_language": language_result,
                "language_probability": language_probability,
                "chunk_seconds": self.chunk_seconds,
                "total_chunks": len(chunk_paths),
                "completed_chunks": len(chunk_paths),
                "total_segments": len(segments),
                "is_complete": True,
                "chunks": completed_chunks,
            }
        )
        write_json_atomic(checkpoint_path, state)
        json_path, srt_path = save_transcript(
            document,
            source,
            destination,
            is_complete=True,
            transcription_signature=config_signature,
            checkpoint_path=checkpoint_path,
        )
        self._emit(
            TranscriptionProgress(
                100,
                f"Hoàn tất Whisper: {len(segments)} đoạn lời thoại.",
                len(segments),
            )
        )
        self._log(f"Transcript JSON: {json_path}")
        self._log(f"Phụ đề SRT: {srt_path}")
        return json_path, srt_path, document.language, len(document.segments)

    @staticmethod
    def _is_audio_cache_valid(
        audio_path: Path,
        metadata_path: Path,
        source_fingerprint: str,
    ) -> bool:
        if not audio_path.is_file() or audio_path.stat().st_size <= 44:
            return False
        try:
            metadata = read_json(metadata_path)
            return (
                metadata.get("source_fingerprint") == source_fingerprint
                and metadata.get("audio_fingerprint") == sha256_file(audio_path)
            )
        except (OSError, ValueError):
            return False

    @staticmethod
    def _load_checkpoint(
        path: Path,
        config_signature: str,
        source_fingerprint: str,
        model_size: str,
        language_key: str,
        total_chunks: int,
    ) -> dict[str, Any]:
        if path.is_file():
            try:
                existing = read_json(path)
            except (OSError, ValueError):
                existing = {}
            if (
                existing.get("config_signature") == config_signature
                and existing.get("source_fingerprint") == source_fingerprint
                and existing.get("model") == model_size
                and existing.get("requested_language") == language_key
                and existing.get("total_chunks") == total_chunks
            ):
                return existing
        return {
            "schema_version": 1,
            "config_signature": config_signature,
            "source_fingerprint": source_fingerprint,
            "model": model_size,
            "requested_language": language_key,
            "total_chunks": total_chunks,
            "completed_chunks": 0,
            "chunks": {},
            "is_complete": False,
        }

    @classmethod
    def _collect_segments(
        cls,
        chunk_data: dict[str, Any],
        fingerprints: dict[int, str],
    ) -> list[TranscriptSegment]:
        collected: list[dict[str, Any]] = []
        for key, value in chunk_data.items():
            if not key.isdigit() or not isinstance(value, dict):
                continue
            index = int(key)
            if value.get("chunk_fingerprint") != fingerprints.get(index):
                continue
            segments = value.get("segments")
            if isinstance(segments, list):
                collected.extend(item for item in segments if isinstance(item, dict))
        collected.sort(
            key=lambda item: (
                float(item.get("start", 0.0)),
                float(item.get("end", 0.0)),
            )
        )
        result: list[TranscriptSegment] = []
        for index, item in enumerate(collected, start=1):
            words = [
                WordTimestamp(
                    start=float(word["start"]),
                    end=float(word["end"]),
                    text=str(word.get("text", "")),
                    probability=(
                        float(word["probability"])
                        if isinstance(word.get("probability"), (int, float))
                        else None
                    ),
                )
                for word in item.get("words", [])
                if isinstance(word, dict) and "start" in word and "end" in word
            ]
            result.append(
                TranscriptSegment(
                    id=index,
                    start=float(item.get("start", 0.0)),
                    end=float(item.get("end", 0.0)),
                    text=str(item.get("text", "")),
                    speaker_id=item.get("speaker_id"),
                    words=words,
                )
            )
        return result

    @staticmethod
    def _segment_to_dict(segment: TranscriptSegment) -> dict[str, Any]:
        return {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "speaker_id": segment.speaker_id,
            "words": [
                {
                    "start": word.start,
                    "end": word.end,
                    "text": word.text,
                    "probability": word.probability,
                }
                for word in segment.words
            ],
        }

    def _emit(self, progress: TranscriptionProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
