from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable

from video_translator.core.errors import VideoTranslatorError
from video_translator.core.models import TranscriptionProgress
from video_translator.gemini.json_utils import read_json, sha256_text, write_json_atomic
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.tts.vieneu_client import LOCAL_ENGINE, LogCallback, VieNeuClient


ProgressCallback = Callable[[TranscriptionProgress], None]


class TTSServiceError(VideoTranslatorError):
    """Raised when local Vietnamese speech generation cannot complete."""


class VieNeuTTSService:
    """Generate and cache each dialogue with the on-device VieNeu SDK."""

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.client = VieNeuClient()
        self.progress_callback = progress_callback
        self.log_callback = log_callback

    def run(
        self,
        translated_json: Path,
        voice_id: str,
        speed: float = 1.0,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, int, int]:
        cancel_event = cancel_event or threading.Event()
        paths = MovieProjectPaths.from_transcript(
            Path(read_json(translated_json).get("source_transcript") or translated_json)
        )
        payload = read_json(translated_json)
        segments = payload.get("segments")
        if not isinstance(segments, list) or not segments:
            raise TTSServiceError("File bản dịch không có segments.")

        # If the original transcript is unavailable, keep TTS files beside the translation.
        if not paths.transcript_json.exists():
            base = Path(translated_json).parent
            paths = MovieProjectPaths(
                transcript_json=Path(translated_json),
                context_json=base / "context.json",
                translated_json=Path(translated_json),
                translated_srt=base / "translated.srt",
                quality_json=base / "quality.json",
                tts_dir=base / f"{Path(translated_json).stem}.tts",
                tts_manifest=base / f"{Path(translated_json).stem}.tts" / "manifest.json",
            )

        voice_id = (voice_id or "").strip()
        if not voice_id:
            raise TTSServiceError("Hãy chọn một giọng VieNeu SDK.")
        speed = max(0.5, min(3.0, float(speed)))
        translation_text = Path(translated_json).read_text(encoding="utf-8")
        translation_fingerprint = sha256_text(translation_text)
        config_signature = sha256_text(
            f"{voice_id}|{speed:.2f}|{LOCAL_ENGINE}|{translation_fingerprint}"
        )
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "translated_json": str(Path(translated_json).resolve()),
            "translation_fingerprint": translation_fingerprint,
            "voice_id": voice_id,
            "speed": speed,
            "engine": LOCAL_ENGINE,
            "config_signature": config_signature,
            "segments": {},
        }
        if paths.tts_manifest.is_file():
            try:
                existing = read_json(paths.tts_manifest)
            except ValueError:
                existing = {}
            if existing.get("config_signature") == config_signature:
                manifest = existing
                manifest["translated_json"] = str(Path(translated_json).resolve())
                manifest["translation_fingerprint"] = translation_fingerprint

        manifest["engine"] = LOCAL_ENGINE
        manifest.setdefault("segments", {})
        paths.tts_dir.mkdir(parents=True, exist_ok=True)
        completed = 0
        total = len(segments)

        try:
            for index, segment in enumerate(segments, start=1):
                if cancel_event.is_set():
                    raise TTSServiceError(
                        "Đã hủy tạo giọng đọc. Audio đã hoàn thành được lưu để tiếp tục."
                    )
                if not isinstance(segment, dict):
                    raise TTSServiceError(f"Segment #{index} trong file bản dịch không hợp lệ.")
                segment_id = int(segment.get("id", index))
                text = str(segment.get("translated_text") or "").strip()
                if not text:
                    completed += 1
                    continue
                spoken_text = self._apply_v3_cues(text, segment)

                request_signature = sha256_text(
                    f"{segment_id}|{spoken_text}|{voice_id}|{speed:.2f}|{LOCAL_ENGINE}"
                )
                request_key = hashlib.sha256(
                    request_signature.encode("utf-8")
                ).hexdigest()
                entries = manifest["segments"]
                entry = entries.get(str(segment_id), {})
                audio_path_text = entry.get("audio_path") if isinstance(entry, dict) else None
                audio_path = Path(audio_path_text) if audio_path_text else None
                expected_path = paths.tts_dir / (
                    f"segment_{segment_id:06d}_{request_key[:10]}.wav"
                )

                if (
                    isinstance(entry, dict)
                    and entry.get("request_signature") == request_signature
                    and audio_path is not None
                    and audio_path.is_file()
                    and audio_path.stat().st_size > 64
                ):
                    completed += 1
                    self._emit_progress(completed, total)
                    continue

                # Recover audio written just before a crash but not yet committed to the manifest.
                if (
                    isinstance(entry, dict)
                    and entry.get("request_signature") == request_signature
                    and expected_path.is_file()
                    and expected_path.stat().st_size > 64
                ):
                    entries[str(segment_id)] = {
                        **entry,
                        "status": "completed",
                        "audio_path": str(expected_path),
                        "request_key": request_key,
                        "voice_id": voice_id,
                        "speed": speed,
                        "engine": LOCAL_ENGINE,
                    }
                    write_json_atomic(paths.tts_manifest, manifest)
                    completed += 1
                    self._emit_progress(completed, total)
                    continue

                attempt = int(entry.get("attempt", 0)) if isinstance(entry, dict) else 0
                previous_status = str(entry.get("status", "")) if isinstance(entry, dict) else ""
                if (
                    not isinstance(entry, dict)
                    or entry.get("request_signature") != request_signature
                    or previous_status == "failed"
                ):
                    attempt += 1

                entry = {
                    "request_signature": request_signature,
                    "request_key": request_key,
                    "attempt": max(1, attempt),
                    "status": "generating",
                    "audio_path": "",
                    "voice_id": voice_id,
                    "speed": speed,
                    "engine": LOCAL_ENGINE,
                }
                entries[str(segment_id)] = entry
                write_json_atomic(paths.tts_manifest, manifest)

                try:
                    audio_path, metadata = self.client.synthesize_segment(
                        text=spoken_text,
                        voice_id=voice_id,
                        speed=speed,
                        segment_id=segment_id,
                        request_key=request_key,
                        output_dir=paths.tts_dir,
                        cancel_event=cancel_event,
                        log_callback=self.log_callback,
                    )
                except Exception as exc:
                    entry["status"] = "failed"
                    entry["last_error"] = str(exc)
                    write_json_atomic(paths.tts_manifest, manifest)
                    raise

                entries[str(segment_id)] = {
                    **entry,
                    **metadata,
                    "request_signature": request_signature,
                    "status": "completed",
                    "audio_path": str(audio_path),
                }
                write_json_atomic(paths.tts_manifest, manifest)
                completed += 1
                self._emit_progress(completed, total)

            manifest["completed_segments"] = completed
            manifest["total_segments"] = total
            manifest["is_complete"] = completed == total
            write_json_atomic(paths.tts_manifest, manifest)
        finally:
            self.client.close()

        self._log(f"VieNeu local audio manifest: {paths.tts_manifest}")
        self._emit(
            TranscriptionProgress(100, f"VieNeu SDK local hoàn tất {completed}/{total} đoạn.", completed)
        )
        return paths.tts_manifest, completed, total

    @staticmethod
    def _apply_v3_cues(text: str, segment: dict[str, Any]) -> str:
        """Map the available laughter label to the SDK's experimental inline cue."""
        delivery = str(segment.get("delivery") or "").lower()
        cue = "[cười]" if delivery == "laughing" else ""
        if cue and not text.lstrip().startswith(cue):
            return f"{cue} {text}"
        return text

    def _emit_progress(self, completed: int, total: int) -> None:
        self._emit(
            TranscriptionProgress(
                int(completed / total * 100),
                f"Đã tạo audio local {completed}/{total} đoạn...",
                completed,
            )
        )

    def _emit(self, progress: TranscriptionProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
