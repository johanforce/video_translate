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
from video_translator.tts.vieneu_client import (
    LogCallback,
    VieNeuClient,
    VieNeuJobFailedError,
)


ProgressCallback = Callable[[TranscriptionProgress], None]


class TTSServiceError(VideoTranslatorError):
    """Raised when Vietnamese speech generation cannot complete."""


class VieNeuTTSService:
    """Generate and cache separate audio files for each translated dialogue segment."""

    def __init__(
        self,
        api_key: str,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.client = VieNeuClient(api_key)
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
        payload = read_json(translated_json)
        segments = payload.get("segments")
        if not isinstance(segments, list) or not segments:
            raise TTSServiceError("File bản dịch không có segments.")

        paths = MovieProjectPaths.from_transcript(
            Path(payload.get("source_transcript") or translated_json)
        )
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
            raise TTSServiceError("Hãy chọn giọng VieNeu.")
        speed = max(0.5, min(3.0, float(speed)))
        translation_text = Path(translated_json).read_text(encoding="utf-8")
        translation_fingerprint = sha256_text(translation_text)
        config_signature = sha256_text(
            f"{voice_id}|{speed:.2f}|v4|{translation_fingerprint}"
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "translated_json": str(Path(translated_json).resolve()),
            "translation_fingerprint": translation_fingerprint,
            "voice_id": voice_id,
            "speed": speed,
            "engine": "v4",
            "config_signature": config_signature,
            "segments": {},
        }
        if paths.tts_manifest.is_file():
            try:
                existing = read_json(paths.tts_manifest)
            except ValueError:
                existing = {}
            if (
                existing.get("config_signature") == config_signature
                and existing.get("translation_fingerprint") == translation_fingerprint
            ):
                manifest = existing
        manifest.setdefault("segments", {})
        paths.tts_dir.mkdir(parents=True, exist_ok=True)
        completed = 0
        total = len(segments)

        for index, segment in enumerate(segments, start=1):
            if cancel_event.is_set():
                raise TTSServiceError("Đã hủy tạo giọng đọc. Manifest đã lưu để tiếp tục.")
            if not isinstance(segment, dict):
                raise TTSServiceError(f"Segment #{index} trong file bản dịch không hợp lệ.")
            segment_id = int(segment.get("id", index))
            text = str(segment.get("translated_text") or "").strip()
            if not text:
                completed += 1
                continue

            request_signature = sha256_text(
                f"{segment_id}|{text}|{voice_id}|{speed:.2f}|v4"
            )
            entries = manifest["segments"]
            entry = entries.get(str(segment_id), {})
            audio_path_text = entry.get("audio_path") if isinstance(entry, dict) else None
            audio_path = Path(audio_path_text) if audio_path_text else None

            if (
                isinstance(entry, dict)
                and entry.get("request_signature") == request_signature
                and audio_path is not None
                and audio_path.is_file()
                and audio_path.stat().st_size > 64
            ):
                completed += 1
                self._emit(
                    TranscriptionProgress(
                        int(completed / total * 100),
                        f"Đã có audio {completed}/{total} đoạn...",
                        completed,
                    )
                )
                continue

            attempt = int(entry.get("attempt", 0)) if isinstance(entry, dict) else 0
            previous_status = str(entry.get("status", "")) if isinstance(entry, dict) else ""
            job_id = str(entry.get("job_id", "")) if isinstance(entry, dict) else ""
            request_key = str(entry.get("request_key", "")) if isinstance(entry, dict) else ""

            if (
                not isinstance(entry, dict)
                or entry.get("request_signature") != request_signature
                or not request_key
                or previous_status in {"failed", "cancelled"}
            ):
                attempt += 1
                request_key = hashlib.sha256(
                    f"{request_signature}|attempt:{attempt}".encode("utf-8")
                ).hexdigest()
                job_id = ""
                entry = {
                    "request_signature": request_signature,
                    "request_key": request_key,
                    "attempt": attempt,
                    "status": "submitting",
                    "job_id": "",
                    "audio_path": "",
                }
                entries[str(segment_id)] = entry
                write_json_atomic(paths.tts_manifest, manifest)

            try:
                if job_id:
                    audio_path, metadata = self.client.resume_segment(
                        job_id=job_id,
                        segment_id=segment_id,
                        request_key=request_key,
                        voice_id=voice_id,
                        speed=speed,
                        output_dir=paths.tts_dir,
                        cancel_event=cancel_event,
                        log_callback=self.log_callback,
                    )
                else:
                    audio_path, metadata = self.client.synthesize_segment(
                        text=text,
                        voice_id=voice_id,
                        speed=speed,
                        segment_id=segment_id,
                        request_key=request_key,
                        output_dir=paths.tts_dir,
                        cancel_event=cancel_event,
                        log_callback=self.log_callback,
                        job_callback=lambda job_id, response, sid=segment_id, ent=entry: self._save_job_checkpoint(
                            paths.tts_manifest, manifest, sid, ent, job_id, response
                        ),
                    )
            except VieNeuJobFailedError as exc:
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
            self._emit(
                TranscriptionProgress(
                    int(completed / total * 100),
                    f"Đã tạo audio {completed}/{total} đoạn...",
                    completed,
                )
            )

        manifest["completed_segments"] = completed
        manifest["total_segments"] = total
        manifest["is_complete"] = completed == total
        write_json_atomic(paths.tts_manifest, manifest)
        self._log(f"VieNeu audio manifest: {paths.tts_manifest}")
        self._emit(
            TranscriptionProgress(100, f"VieNeu hoàn tất {completed}/{total} đoạn.", completed)
        )
        return paths.tts_manifest, completed, total

    @staticmethod
    def _save_job_checkpoint(
        manifest_path: Path,
        manifest: dict[str, Any],
        segment_id: int,
        entry: dict[str, Any],
        job_id: str,
        response: dict[str, Any],
    ) -> None:
        entry["job_id"] = job_id
        entry["status"] = str(response.get("status") or "queued")
        entry["job_response"] = {
            key: response.get(key)
            for key in ("status", "duration", "audioUrlExpiresIn")
            if response.get(key) is not None
        }
        manifest.setdefault("segments", {})[str(segment_id)] = entry
        write_json_atomic(manifest_path, manifest)

    def _emit(self, progress: TranscriptionProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
