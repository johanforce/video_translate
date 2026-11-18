from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from video_translator.core.errors import VideoTranslatorError
from video_translator.core.models import TranscriptionProgress
from video_translator.gemini.client import GeminiJsonClient
from video_translator.gemini.json_utils import read_json, sha256_file, sha256_text, write_json_atomic
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.gemini.prompt_loader import load_prompt, render_prompt
from video_translator.transcription.transcript_writer import format_srt_timestamp


ProgressCallback = Callable[[TranscriptionProgress], None]
LogCallback = Callable[[str], None]


class FilmTranslationError(VideoTranslatorError):
    """Raised when movie dialogue translation or validation fails."""


_ALLOWED_EMOTIONS = {
    "neutral", "happy", "sad", "angry", "fearful", "surprised",
    "sarcastic", "humorous", "romantic", "disgusted", "curious",
    "reassuring", "threatening", "uncertain",
}
_ALLOWED_DELIVERIES = {
    "normal", "shouting", "whispering", "crying", "laughing",
    "breathless", "trembling", "singing", "robotic", "soft", "emphatic",
}


class FilmTranslationService:
    """Translate transcript batches, QC each batch, and checkpoint after every batch."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.client = GeminiJsonClient(api_key, model)
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.model = model

    def run(
        self,
        transcript_path: Path,
        context_path: Path,
        batch_size: int = 30,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, Path, Path]:
        cancel_event = cancel_event or threading.Event()
        paths = MovieProjectPaths.from_transcript(transcript_path)
        transcript = read_json(paths.transcript_json)
        context_envelope = read_json(context_path)
        context = context_envelope.get("profile")
        if not isinstance(context, dict):
            raise FilmTranslationError("Context Profile không hợp lệ.")
        source_segments = transcript.get("segments")
        if not isinstance(source_segments, list) or not source_segments:
            raise FilmTranslationError("Transcript không có segments để dịch.")

        source_fingerprint = sha256_file(paths.transcript_json)
        context_fingerprint = sha256_file(context_path)
        batch_size = max(5, min(int(batch_size), 100))
        config_signature = sha256_text(
            source_fingerprint
            + context_fingerprint
            + self.model
            + str(batch_size)
            + load_prompt("system_prompt.txt")
            + load_prompt("translation.txt")
            + load_prompt("quality_control.txt")
        )

        state = self._load_or_create_state(paths, config_signature, source_fingerprint, context_fingerprint)
        translated_by_id = {
            int(item["id"]): item
            for item in state.get("segments", [])
            if isinstance(item, dict) and str(item.get("id", "")).isdigit()
        }
        quality_issues = list(state.get("issues", []))
        total_batches = (len(source_segments) + batch_size - 1) // batch_size

        system_prompt = load_prompt("system_prompt.txt")
        translation_template = load_prompt("translation.txt")
        quality_template = load_prompt("quality_control.txt")

        for batch_index, start_index in enumerate(range(0, len(source_segments), batch_size)):
            current = source_segments[start_index:start_index + batch_size]
            ids = [int(item["id"]) for item in current]
            if all(segment_id in translated_by_id for segment_id in ids):
                continue
            if cancel_event.is_set():
                raise FilmTranslationError("Đã hủy dịch phim.")

            previous_context = source_segments[max(0, start_index - 6):start_index]
            next_context = source_segments[start_index + len(current):start_index + len(current) + 6]
            prompt = render_prompt(
                translation_template,
                {
                    "CONTEXT_PROFILE_JSON": json.dumps(context, ensure_ascii=False),
                    "PREVIOUS_CONTEXT_JSON": json.dumps(previous_context, ensure_ascii=False),
                    "NEXT_CONTEXT_JSON": json.dumps(next_context, ensure_ascii=False),
                    "SEGMENTS_JSON": json.dumps(current, ensure_ascii=False),
                },
            )
            self._log(f"Dịch batch {batch_index + 1}/{total_batches} ({len(current)} đoạn)...")
            translated_response = self.client.generate_json(
                prompt,
                system_prompt,
                cancel_event=cancel_event,
            )
            translated_batch = self._validate_segments(
                current,
                translated_response.get("segments"),
                "dịch",
            )

            qc_prompt = render_prompt(
                quality_template,
                {
                    "CONTEXT_PROFILE_JSON": json.dumps(context, ensure_ascii=False),
                    "SEGMENTS_JSON": json.dumps(current, ensure_ascii=False),
                    "TRANSLATION_JSON": json.dumps(
                        {"segments": translated_batch},
                        ensure_ascii=False,
                    ),
                },
            )
            self._log(f"Kiểm tra chất lượng batch {batch_index + 1}/{total_batches}...")
            qc_response = self.client.generate_json(
                qc_prompt,
                system_prompt,
                cancel_event=cancel_event,
            )
            checked_batch = self._validate_segments(
                current,
                qc_response.get("segments"),
                "kiểm soát chất lượng",
            )

            issues = qc_response.get("issues", [])
            if isinstance(issues, list):
                quality_issues.extend(
                    issue for issue in issues if isinstance(issue, dict)
                )
            for original, translated in zip(current, checked_batch):
                translated_by_id[int(original["id"])] = self._merge_source_fields(
                    original,
                    translated,
                )

            ordered = [
                translated_by_id[int(item["id"])]
                for item in source_segments
                if int(item["id"]) in translated_by_id
            ]
            state.update(
                {
                    "schema_version": 1,
                    "source_transcript": str(paths.transcript_json),
                    "context_profile": str(context_path),
                    "source_fingerprint": source_fingerprint,
                    "context_fingerprint": context_fingerprint,
                    "config_signature": config_signature,
                    "model": self.model,
                    "batch_size": batch_size,
                    "completed_segments": len(ordered),
                    "total_segments": len(source_segments),
                    "is_complete": len(ordered) == len(source_segments),
                    "segments": ordered,
                    "issues": quality_issues,
                }
            )
            write_json_atomic(paths.translated_json, state)
            write_json_atomic(
                paths.quality_json,
                {
                    "schema_version": 1,
                    "model": self.model,
                    "issues": quality_issues,
                    "summary": {
                        "reviewed_segments": len(ordered),
                        "issue_count": len(quality_issues),
                    },
                },
            )
            percent = int(len(ordered) / len(source_segments) * 95)
            self._emit(
                TranscriptionProgress(
                    percent,
                    f"Đã dịch và kiểm tra {len(ordered)}/{len(source_segments)} đoạn...",
                    len(ordered),
                )
            )

        if len(translated_by_id) != len(source_segments):
            raise FilmTranslationError(
                "Bản dịch chưa đủ số đoạn. Dữ liệu đã lưu, hãy chạy lại để tiếp tục."
            )

        state["is_complete"] = True
        ordered_segments = [
            translated_by_id[int(item["id"])] for item in source_segments
        ]
        state["segments"] = ordered_segments
        write_json_atomic(paths.translated_json, state)
        self._write_translated_srt(paths.translated_srt, ordered_segments)
        write_json_atomic(
            paths.quality_json,
            {
                "schema_version": 1,
                "model": self.model,
                "issues": quality_issues,
                "summary": {
                    "reviewed_segments": len(ordered_segments),
                    "issue_count": len(quality_issues),
                },
            },
        )
        self._emit(
            TranscriptionProgress(
                100,
                "Dịch phim và kiểm soát chất lượng hoàn tất.",
                len(ordered_segments),
            )
        )
        self._log(f"Vietnamese translation: {paths.translated_json}")
        self._log(f"Vietnamese SRT: {paths.translated_srt}")
        self._log(f"Quality report: {paths.quality_json}")
        return paths.translated_json, paths.translated_srt, paths.quality_json

    @staticmethod
    def _load_or_create_state(
        paths: MovieProjectPaths,
        signature: str,
        source_fingerprint: str,
        context_fingerprint: str,
    ) -> dict[str, Any]:
        if paths.translated_json.is_file():
            try:
                state = read_json(paths.translated_json)
            except ValueError:
                state = {}
            if state.get("config_signature") == signature:
                return state
        return {
            "schema_version": 1,
            "source_transcript": str(paths.transcript_json),
            "context_profile": str(paths.context_json),
            "source_fingerprint": source_fingerprint,
            "context_fingerprint": context_fingerprint,
            "config_signature": signature,
            "segments": [],
            "issues": [],
            "is_complete": False,
        }

    @staticmethod
    def _validate_segments(
        source: list[dict[str, Any]],
        response_segments: Any,
        operation: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(response_segments, list):
            raise FilmTranslationError(f"Gemini không trả danh sách đoạn khi {operation}.")
        if len(response_segments) != len(source):
            raise FilmTranslationError(
                f"Sai số lượng segment khi {operation}: "
                f"nhận {len(response_segments)}, cần {len(source)}."
            )

        result: list[dict[str, Any]] = []
        for original, candidate in zip(source, response_segments):
            if not isinstance(candidate, dict):
                raise FilmTranslationError(f"Segment trả về không hợp lệ khi {operation}.")
            try:
                source_id = int(original["id"])
                returned_id = int(candidate["id"])
                source_start = float(original["start"])
                source_end = float(original["end"])
                returned_start = float(candidate["start"])
                returned_end = float(candidate["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FilmTranslationError(f"Segment thiếu ID/timestamp khi {operation}.") from exc

            if returned_id != source_id:
                raise FilmTranslationError(
                    f"Gemini thay đổi thứ tự hoặc ID segment {source_id} khi {operation}."
                )
            if returned_start != source_start or returned_end != source_end:
                raise FilmTranslationError(
                    f"Gemini thay đổi timestamp segment {source_id}; đã từ chối batch."
                )
            translated_text = candidate.get("translated_text")
            if not isinstance(translated_text, str):
                raise FilmTranslationError(
                    f"Segment {source_id} thiếu translated_text khi {operation}."
                )
            if str(original.get("text", "")).strip() and not translated_text.strip():
                raise FilmTranslationError(
                    f"Segment {source_id} bị bỏ trống bản dịch khi {operation}."
                )

            emotion = candidate.get("emotion")
            delivery = candidate.get("delivery")
            result.append(
                {
                    "id": source_id,
                    "start": source_start,
                    "end": source_end,
                    "speaker_id": candidate.get("speaker_id"),
                    "translated_text": translated_text.strip(),
                    "emotion": emotion if emotion in _ALLOWED_EMOTIONS else "neutral",
                    "delivery": delivery if delivery in _ALLOWED_DELIVERIES else "normal",
                }
            )
        return result

    @staticmethod
    def _merge_source_fields(
        source: dict[str, Any],
        translated: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            **translated,
            "id": int(source["id"]),
            "start": float(source["start"]),
            "end": float(source["end"]),
            "speaker_id": source.get("speaker_id") or translated.get("speaker_id"),
            "original_text": str(source.get("text", "")),
        }

    @staticmethod
    def _write_translated_srt(path: Path, segments: list[dict[str, Any]]) -> None:
        blocks: list[str] = []
        cue = 0
        for segment in segments:
            text = str(segment.get("translated_text", "")).strip()
            if not text:
                continue
            cue += 1
            blocks.append(
                f"{cue}\n"
                f"{format_srt_timestamp(float(segment['start']))} --> "
                f"{format_srt_timestamp(float(segment['end']))}\n"
                f"{text}"
            )
        content = "\n\n".join(blocks) + ("\n" if blocks else "")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)

    def _emit(self, progress: TranscriptionProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
