from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from video_translator.core.errors import VideoTranslatorError
from video_translator.core.models import TranscriptionProgress
from video_translator.gemini.json_utils import (
    read_json,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.rendering.media_tools import (
    media_duration,
    run_ffmpeg,
)


ProgressCallback = Callable[[TranscriptionProgress], None]
LogCallback = Callable[[str], None]


class AudioTimelineError(VideoTranslatorError):
    """Raised when narration clips cannot be assembled into a timeline."""


class AudioTimelineBuilder:
    """Prepare and place cached TTS clips at source timestamps.

    Prepared clip files are cached individually. If the process is killed while mixing,
    the inexpensive PCM mix is rebuilt from those prepared clips without calling VieNeu.
    """

    SAMPLE_RATE = 24_000
    MAX_ATEMPO = 1.30
    VOICE_GAIN = 0.88

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.log_callback = log_callback

    def build(
        self,
        translated_json: Path,
        tts_manifest: Path | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, Path]:
        cancel_event = cancel_event or threading.Event()
        translation_path = Path(translated_json).expanduser().resolve()
        translation = read_json(translation_path)
        segments = translation.get("segments")
        if not isinstance(segments, list) or not segments:
            raise AudioTimelineError("Bản dịch không có danh sách segment.")

        transcript_text = translation.get("source_transcript")
        transcript_path = Path(transcript_text).expanduser() if transcript_text else None
        if transcript_path is None or not transcript_path.is_file():
            raise AudioTimelineError(
                "Không tìm thấy transcript gốc trong trường source_transcript của file dịch."
            )
        paths = MovieProjectPaths.from_transcript(transcript_path)
        manifest_path = (
            Path(tts_manifest).expanduser().resolve()
            if tts_manifest
            else paths.tts_manifest
        )
        if not manifest_path.is_file():
            raise AudioTimelineError(f"Không tìm thấy VieNeu manifest: {manifest_path}")
        tts_state = read_json(manifest_path)
        if tts_state.get("is_complete") is not True:
            raise AudioTimelineError(
                "VieNeu chưa tạo xong toàn bộ audio. Hãy chạy lại TTS để tiếp tục từ manifest."
            )

        entries = tts_state.get("segments")
        if not isinstance(entries, dict):
            raise AudioTimelineError("VieNeu manifest không có danh sách audio segment.")

        source_video = Path(transcript.get("source_video", "")).expanduser()
        if not source_video.is_file():
            raise AudioTimelineError(
                "Không tìm thấy video nguồn được ghi trong transcript JSON."
            )
        duration = self._video_duration(transcript, source_video)
        sample_rate = self.SAMPLE_RATE
        total_samples = max(1, int(math.ceil(duration * sample_rate)))
        source_hash = sha256_file(translation_path)
        manifest_hash = sha256_file(manifest_path)
        signature = sha256_text(
            f"{source_hash}|{manifest_hash}|{duration:.6f}|{sample_rate}|"
            f"{self.MAX_ATEMPO:.3f}|{self.VOICE_GAIN:.3f}"
        )

        output_dir = manifest_path.parent
        prepared_dir = output_dir / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared_cache_path = prepared_dir / "prepared_manifest.json"
        prepared_state = self._load_prepared_cache(
            prepared_cache_path,
            signature,
        )
        prepared_entries: dict[str, Any] = prepared_state["segments"]

        ordered_segments = sorted(
            [item for item in segments if isinstance(item, dict)],
            key=lambda item: (float(item.get("start", 0.0)), int(item.get("id", 0))),
        )
        speech_segments = [
            item for item in ordered_segments
            if str(item.get("translated_text", "")).strip()
        ]
        missing = []
        for item in speech_segments:
            key = str(int(item.get("id", 0)))
            entry = entries.get(key)
            audio_path = Path(str(entry.get("audio_path", ""))).expanduser() if isinstance(entry, dict) else None
            if (
                not isinstance(entry, dict)
                or entry.get("status") != "completed"
                or audio_path is None
                or not audio_path.is_file()
            ):
                missing.append(key)
        if missing:
            raise AudioTimelineError(
                f"Thiếu audio VieNeu cho {len(missing)} segment (ví dụ: {', '.join(missing[:8])}). "
                "Hãy chạy lại VieNeu; các segment đã hoàn thành sẽ dùng cache."
            )
        if not speech_segments:
            raise AudioTimelineError("Không có câu thoại tiếng Việt để lồng tiếng.")

        self._log(
            f"Chuẩn bị {len(speech_segments)} audio segment, timeline {duration:.1f} giây."
        )
        prepared_paths: dict[str, Path] = {}
        warnings: list[str] = []

        for index, segment in enumerate(speech_segments, start=1):
            if cancel_event.is_set():
                raise AudioTimelineError("Đã hủy chuẩn bị timeline audio.")
            segment_id = int(segment["id"])
            key = str(segment_id)
            entry = entries[key]
            source_audio = Path(str(entry["audio_path"])).expanduser()
            start = max(0.0, float(segment.get("start", 0.0)))
            next_start = self._next_start(ordered_segments, segment_id, start, duration)
            available = max(0.35, min(duration - start, next_start - start - 0.06))
            request_signature = sha256_text(
                f"{entry.get('request_signature', '')}|{source_audio.stat().st_size}|"
                f"{source_audio.stat().st_mtime_ns}|{start:.4f}|{available:.4f}|"
                f"{sample_rate}|{self.MAX_ATEMPO}"
            )
            cache_entry = prepared_entries.get(key, {})
            cached_path = (
                Path(str(cache_entry.get("path"))).expanduser()
                if isinstance(cache_entry, dict) and cache_entry.get("path")
                else None
            )
            if (
                isinstance(cache_entry, dict)
                and cache_entry.get("signature") == request_signature
                and cached_path is not None
                and cached_path.is_file()
                and cached_path.stat().st_size >= 4
            ):
                prepared_paths[key] = cached_path
                self._emit(
                    int(index / len(speech_segments) * 45),
                    f"Đã có cache audio chuẩn bị {index}/{len(speech_segments)}...",
                    index,
                )
                continue

            source_duration = self._safe_duration(source_audio)
            atempo = 1.0
            if source_duration > available * 1.03:
                atempo = min(self.MAX_ATEMPO, source_duration / available)
                if source_duration / atempo > available * 1.05:
                    warnings.append(
                        f"Segment {segment_id}: lời đọc dài hơn khoảng thoại, có thể chồng câu kế tiếp."
                    )

            prepared_path = prepared_dir / f"segment_{segment_id:06d}_{request_signature[:12]}.f32"
            temporary_path = prepared_path.with_suffix(".f32.tmp")
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source_audio), "-vn",
            ]
            if atempo > 1.001:
                command.extend(["-af", f"atempo={atempo:.4f}"])
            command.extend(
                [
                    "-ac", "1",
                    "-ar", str(sample_rate),
                    "-f", "f32le",
                    "-progress", "pipe:1",
                    "-nostats",
                    str(temporary_path),
                ]
            )
            self._log(
                f"Chuẩn bị audio {index}/{len(speech_segments)}: segment {segment_id}"
                + (f" (tăng tốc {atempo:.2f}x)" if atempo > 1.001 else "")
            )
            run_ffmpeg(
                command,
                duration_seconds=source_duration / atempo,
                cancel_event=cancel_event,
                log_callback=self.log_callback,
            )
            if not temporary_path.is_file() or temporary_path.stat().st_size < 4:
                temporary_path.unlink(missing_ok=True)
                raise AudioTimelineError(f"Audio chuẩn bị rỗng ở segment {segment_id}.")
            temporary_path.replace(prepared_path)
            prepared_paths[key] = prepared_path
            prepared_entries[key] = {
                "path": str(prepared_path),
                "signature": request_signature,
                "start": start,
                "available_seconds": available,
                "atempo": atempo,
            }
            write_json_atomic(
                prepared_cache_path,
                {
                    "schema_version": 1,
                    "signature": signature,
                    "sample_rate": sample_rate,
                    "segments": prepared_entries,
                },
            )
            self._emit(
                int(index / len(speech_segments) * 45),
                f"Đã chuẩn bị {index}/{len(speech_segments)} audio...",
                index,
            )

        if cancel_event.is_set():
            raise AudioTimelineError("Đã hủy ghép timeline audio.")

        mix_raw = output_dir / "narration_mix.f32.tmp"
        voice_wav = output_dir / "narration.wav"
        voice_wav_tmp = output_dir / "narration.wav.tmp"
        mix = np.memmap(mix_raw, dtype=np.float32, mode="w+", shape=(total_samples,))
        mix[:] = 0.0
        mix.flush()

        for index, segment in enumerate(speech_segments, start=1):
            if cancel_event.is_set():
                del mix
                mix_raw.unlink(missing_ok=True)
                raise AudioTimelineError("Đã hủy ghép timeline audio.")
            key = str(int(segment["id"]))
            raw = prepared_paths[key]
            samples = np.fromfile(raw, dtype=np.float32)
            if samples.size == 0:
                continue
            start_sample = max(0, int(round(float(segment.get("start", 0.0)) * sample_rate)))
            if start_sample >= total_samples:
                continue
            end_sample = min(total_samples, start_sample + int(samples.size))
            amount = end_sample - start_sample
            mix[start_sample:end_sample] += samples[:amount] * self.VOICE_GAIN
            if index % 25 == 0:
                mix.flush()
            self._emit(
                45 + int(index / len(speech_segments) * 35),
                f"Đang xếp timeline {index}/{len(speech_segments)}...",
                index,
            )

        mix.flush()
        peak = 0.0
        chunk_size = sample_rate * 30
        for offset in range(0, total_samples, chunk_size):
            peak = max(
                peak,
                float(np.max(np.abs(mix[offset:min(total_samples, offset + chunk_size)]))),
            )
        if peak > 0.96:
            gain = 0.96 / peak
            self._log(f"Chuẩn hóa peak narration: gain {gain:.3f}.")
            for offset in range(0, total_samples, chunk_size):
                end = min(total_samples, offset + chunk_size)
                mix[offset:end] *= gain
        mix.flush()
        del mix

        run_ffmpeg(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "1",
                "-i", str(mix_raw),
                "-c:a", "pcm_s16le",
                "-progress", "pipe:1", "-nostats",
                str(voice_wav_tmp),
            ],
            duration_seconds=duration,
            cancel_event=cancel_event,
            progress_callback=lambda percent, message: self._emit(
                80 + int(percent * 0.18), message, len(speech_segments)
            ),
            log_callback=self.log_callback,
        )
        if not voice_wav_tmp.is_file() or voice_wav_tmp.stat().st_size <= 44:
            voice_wav_tmp.unlink(missing_ok=True)
            mix_raw.unlink(missing_ok=True)
            raise AudioTimelineError("Không tạo được narration.wav hợp lệ.")
        voice_wav_tmp.replace(voice_wav)
        mix_raw.unlink(missing_ok=True)

        timeline_path = output_dir / "timeline.json"
        timeline = {
            "schema_version": 1,
            "signature": signature,
            "source_video": str(source_video),
            "translated_json": str(translation_path),
            "tts_manifest": str(manifest_path),
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "voice_track": str(voice_wav),
            "segments": [
                {
                    "id": int(segment["id"]),
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                    "audio": str(prepared_paths[str(int(segment["id"]))]),
                }
                for segment in speech_segments
            ],
            "warnings": warnings,
        }
        write_json_atomic(timeline_path, timeline)
        self._emit(100, "Timeline thuyết minh đã sẵn sàng.", len(speech_segments))
        self._log(f"Voice track: {voice_wav}")
        for warning in warnings:
            self._log("Cảnh báo đồng bộ: " + warning)
        return voice_wav, timeline_path

    @staticmethod
    def _load_prepared_cache(path: Path, signature: str) -> dict[str, Any]:
        if path.is_file():
            try:
                value = read_json(path)
                if value.get("signature") == signature:
                    return value
            except (ValueError, OSError):
                pass
        return {
            "schema_version": 1,
            "signature": signature,
            "sample_rate": AudioTimelineBuilder.SAMPLE_RATE,
            "segments": {},
        }

    @staticmethod
    def _video_duration(transcript: dict[str, Any], video_path: Path) -> float:
        value = transcript.get("duration_seconds")
        try:
            duration = float(value)
            if duration > 0:
                return duration
        except (TypeError, ValueError):
            pass
        return media_duration(video_path)

    @staticmethod
    def _safe_duration(path: Path) -> float:
        try:
            return media_duration(path)
        except Exception:
            return path.stat().st_size / (AudioTimelineBuilder.SAMPLE_RATE * 4)

    @staticmethod
    def _next_start(
        ordered_segments: list[dict[str, Any]],
        segment_id: int,
        start: float,
        total_duration: float,
    ) -> float:
        found = False
        for item in ordered_segments:
            current_id = int(item.get("id", -1))
            current_start = float(item.get("start", 0.0))
            if found and current_start > start + 0.001 and str(item.get("translated_text", "")).strip():
                return min(total_duration, current_start)
            if current_id == segment_id:
                found = True
        return total_duration

    def _emit(self, percent: int, message: str, segments_done: int = 0) -> None:
        if self.progress_callback:
            self.progress_callback(
                TranscriptionProgress(
                    max(0, min(100, percent)),
                    message,
                    segments_done,
                )
            )

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
