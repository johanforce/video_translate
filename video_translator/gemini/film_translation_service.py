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
        if transcript.get("is_complete") is False:
            raise FilmTranslationError(
                "Whisper transcript còn dang dở. Hãy hoàn tất nhận diện trước khi dịch."
            )
        context_envelope = read_json(context_path)
        context = context_envelope.get("profile")
        if not isinstance(context, dict):
            raise FilmTranslationError("Context Profile không hợp lệ.")
        if context_envelope.get("is_complete") is not True:
            raise FilmTranslationError(
                "Context Profile chưa phân tích toàn bộ transcript; hãy hoàn tất bước phân tích phim trước."
            )
        source_segments = transcript.get("segments")
        if not isinstance(source_segments, list) or not source_segments:
            raise FilmTranslationError("Transcript không có segments để dịch.")

        source_fingerprint = sha256_file(paths.transcript_json)
        context_fingerprint = sha256_file(context_path)
        batch_size = max(5, min(int(batch_size), 100))
        prompt_signature = (
            load_prompt("system_prompt.txt")
            + load_prompt("translation.txt")
            + load_prompt("quality_control.txt")
        )
        # Keep accepted translation/QC batches when the selected Gemini model changes.
        config_signature = sha256_text(
            source_fingerprint + context_fingerprint + str(batch_size) + prompt_signature
        )

        state = self._load_or_create_state(
            paths,
            config_signature,
            source_fingerprint,
            context_fingerprint,
            batch_size,
            prompt_signature,
            self.model,
        )
        state["model"] = self.model
        translated_by_id = {
            int(item["id"]): item
            for item in state.get("segments", [])
            if isinstance(item, dict) and str(item.get("id", "")).isdigit()
        }
        quality_issues = list(state.get("issues", []))
        pending_batches = state.setdefault("pending_batches", {})
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

            previous_context = self._prompt_segments(
                source_segments[max(0, start_index - 6):start_index]
            )
            current_prompt_segments = self._prompt_segments(current)
            next_context = self._prompt_segments(
                source_segments[start_index + len(current):start_index + len(current) + 6]
            )
            prompt = render_prompt(
                translation_template,
                {
                    "CONTEXT_PROFILE_JSON": json.dumps(context, ensure_ascii=False),
                    "PREVIOUS_CONTEXT_JSON": json.dumps(previous_context, ensure_ascii=False),
                    "NEXT_CONTEXT_JSON": json.dumps(next_context, ensure_ascii=False),
                    "SEGMENTS_JSON": json.dumps(current_prompt_segments, ensure_ascii=False),
                },
            )
            batch_key = f"{ids[0]}-{ids[-1]}"
            batch_signature = sha256_text(
                config_signature + json.dumps(current, ensure_ascii=False, sort_keys=True)
            )
            pending = pending_batches.get(batch_key)
            if not isinstance(pending, dict) or pending.get("signature") != batch_signature:
                pending = {"signature": batch_signature}
                pending_batches[batch_key] = pending
                write_json_atomic(paths.translated_json, state)

            cached_translation = pending.get("translation")
            if isinstance(cached_translation, list):
                self._log(
                    f"Dùng cache bản dịch batch {batch_index + 1}/{total_batches}."
                )
                translated_batch = self._validate_segments(
                    current,
                    cached_translation,
                    "khôi phục cache dịch",
                )
            else:
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
                # Persist immediately after Gemini returns, before the QC request.
                pending["translation"] = translated_batch
                pending["translation_saved"] = True
                write_json_atomic(paths.translated_json, state)

            cached_qc = pending.get("quality_control")
            if isinstance(cached_qc, list):
                self._log(
                    f"Dùng cache QC batch {batch_index + 1}/{total_batches}."
                )
                checked_batch = self._validate_segments(
                    current,
                    cached_qc,
                    "khôi phục cache QC",
                )
                issues = pending.get("issues", [])
            else:
                qc_prompt = render_prompt(
                    quality_template,
                    {
                        "CONTEXT_PROFILE_JSON": json.dumps(context, ensure_ascii=False),
                        "SEGMENTS_JSON": json.dumps(current_prompt_segments, ensure_ascii=False),
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
                if not isinstance(issues, list):
                    issues = []
                issues = [issue for issue in issues if isinstance(issue, dict)]
                # Persist the QC result before marking the batch complete.
                pending["quality_control"] = checked_batch
                pending["issues"] = issues
                pending["quality_control_saved"] = True
                write_json_atomic(paths.translated_json, state)

            skipped_ids = [
                int(original["id"])
                for original, translated in zip(current, checked_batch)
                if not str(translated.get("translated_text", "")).strip()
            ]
            if skipped_ids:
                self._log(
                    "Bỏ qua segment không có bản dịch: "
                    + ", ".join(str(item) for item in skipped_ids)
                    + ". Các đoạn còn lại vẫn tiếp tục."
                )

            if not isinstance(issues, list):
                issues = []
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
                    "pending_batches": pending_batches,
                }
            )
            pending_batches.pop(batch_key, None)
            state["pending_batches"] = pending_batches
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
        state["pending_batches"] = {}
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
    def _prompt_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Send only dialogue identity/content; timestamps stay exclusively in source data."""
        fields = ("id", "text", "speaker_id")
        return [
            {key: item[key] for key in fields if key in item}
            for item in segments
            if isinstance(item, dict)
        ]

    @staticmethod
    def _load_or_create_state(
        paths: MovieProjectPaths,
        signature: str,
        source_fingerprint: str,
        context_fingerprint: str,
        batch_size: int,
        prompt_signature: str,
        current_model: str,
    ) -> dict[str, Any]:
        if paths.translated_json.is_file():
            try:
                state = read_json(paths.translated_json)
            except ValueError:
                state = {}
            if state.get("config_signature") == signature:
                return state

            # Migrate projects created before model-independent cache signatures.
            # Prefer the model recorded in the project; known UI models cover older
            # checkpoints that were saved before their first completed batch.
            legacy_models = {
                str(state.get("model") or ""),
                current_model,
                "gemini-3.8-flash",
                "gemini-3.6-flash",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
            }
            legacy_models.discard("")
            legacy_signatures = {
                # Previous v0.7 cache signature already excluded the model but
                # still included batch size.
                sha256_text(
                    source_fingerprint
                    + context_fingerprint
                    + str(batch_size)
                    + prompt_signature
                )
            }
            legacy_signatures.update(
                sha256_text(
                    source_fingerprint
                    + context_fingerprint
                    + model
                    + str(batch_size)
                    + prompt_signature
                )
                for model in legacy_models
            )
            if state.get("config_signature") in legacy_signatures:
                state["config_signature"] = signature
                state["model"] = current_model
                return state

            # Prompt revisions can change the response schema (for example, no
            # longer asking Gemini to echo timestamps). Keep already accepted,
            # QC-completed segments for the same transcript/context/batch size.
            # Pending responses are discarded because they may follow the old schema.
            try:
                saved_batch_size = int(state.get("batch_size", 0) or 0)
            except (TypeError, ValueError):
                saved_batch_size = 0
            if (
                state.get("source_fingerprint") == source_fingerprint
                and state.get("context_fingerprint") == context_fingerprint
                and saved_batch_size == batch_size
                and isinstance(state.get("segments"), list)
            ):
                state["config_signature"] = signature
                state["model"] = current_model
                state["pending_batches"] = {}
                state["is_complete"] = False
                return state

        return {
            "schema_version": 1,
            "source_transcript": str(paths.transcript_json),
            "context_profile": str(paths.context_json),
            "source_fingerprint": source_fingerprint,
            "context_fingerprint": context_fingerprint,
            "config_signature": signature,
            "model": current_model,
            "segments": [],
            "issues": [],
            "pending_batches": {},
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

        source_by_id: dict[int, dict[str, Any]] = {}
        for original in source:
            try:
                source_id = int(original["id"])
                float(original["start"])
                float(original["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FilmTranslationError(
                    f"Transcript nguồn thiếu ID/timestamp hợp lệ khi {operation}."
                ) from exc
            if source_id in source_by_id:
                raise FilmTranslationError(
                    f"Transcript nguồn có ID segment trùng lặp {source_id}."
                )
            source_by_id[source_id] = original

        # Match returned dialogue by ID, not by array position. Gemini may reorder
        # objects or alter/omit echoed timing values; neither is authoritative.
        candidates_by_id: dict[int, dict[str, Any]] = {}
        for candidate in response_segments:
            if not isinstance(candidate, dict):
                raise FilmTranslationError(
                    f"Segment trả về không hợp lệ khi {operation}."
                )
            try:
                returned_id = int(candidate["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FilmTranslationError(
                    f"Segment thiếu ID khi {operation}."
                ) from exc

            if returned_id not in source_by_id:
                raise FilmTranslationError(
                    f"Gemini trả về ID segment không tồn tại {returned_id} khi {operation}."
                )
            if returned_id in candidates_by_id:
                raise FilmTranslationError(
                    f"Gemini trả lặp ID segment {returned_id} khi {operation}."
                )
            candidates_by_id[returned_id] = candidate

        missing_ids = set(source_by_id) - set(candidates_by_id)
        if missing_ids:
            missing_text = ", ".join(str(item) for item in sorted(missing_ids))
            raise FilmTranslationError(
                f"Gemini bỏ thiếu segment khi {operation}: {missing_text}."
            )

        result: list[dict[str, Any]] = []
        for original in source:
            source_id = int(original["id"])
            candidate = candidates_by_id[source_id]
            translated_text = candidate.get("translated_text")
            if not isinstance(translated_text, str):
                raise FilmTranslationError(
                    f"Segment {source_id} thiếu translated_text khi {operation}."
                )
            # An empty translation is treated as an intentionally skipped subtitle.
            # Keep the segment ID/timing in JSON; the SRT writer omits empty cues.
            emotion = candidate.get("emotion")
            delivery = candidate.get("delivery")
            result.append(
                {
                    "id": source_id,
                    # Source transcript is the sole authority for timestamps.
                    # Never reject useful translation text because Gemini rounded
                    # or hallucinated these redundant fields.
                    "start": float(original["start"]),
                    "end": float(original["end"]),
                    "speaker_id": candidate.get("speaker_id"),
                    "translated_text": translated_text.strip(),
                    "emotion": (
                        emotion
                        if isinstance(emotion, str) and emotion in _ALLOWED_EMOTIONS
                        else "neutral"
                    ),
                    "delivery": (
                        delivery
                        if isinstance(delivery, str) and delivery in _ALLOWED_DELIVERIES
                        else "normal"
                    ),
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
