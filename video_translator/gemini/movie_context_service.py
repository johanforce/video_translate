from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from video_translator.core.errors import VideoTranslatorError
from video_translator.core.models import TranscriptionProgress
from video_translator.gemini.client import GeminiJsonClient
from video_translator.gemini.json_utils import (
    read_json,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.gemini.prompt_loader import load_prompt, render_prompt


ProgressCallback = Callable[[TranscriptionProgress], None]
LogCallback = Callable[[str], None]


class MovieContextError(VideoTranslatorError):
    """Raised when whole-film context analysis cannot complete."""


_PROFILE_DEFAULTS: dict[str, Any] = {
    "film_profile": {
        "is_partial": True,
        "genre": "",
        "setting": "",
        "premise": "",
        "narrative_tone": "",
        "localization_style": "",
        "continuity_facts": [],
    },
    "characters": [],
    "relationships": [],
    "terminology": [],
    "style_guide": {
        "pronoun_policy": "",
        "dialogue_style": "",
        "comedy_policy": "",
        "profanity_policy": "",
        "special_instructions": [],
    },
    "uncertainties": [],
}


class MovieContextService:
    """Analyze transcript chapters and checkpoint a reusable character/context profile."""

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
        batch_segments: int = 220,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        cancel_event = cancel_event or threading.Event()
        paths = MovieProjectPaths.from_transcript(transcript_path)
        if not paths.transcript_json.is_file():
            raise MovieContextError(f"Không tìm thấy transcript: {paths.transcript_json}")

        transcript = read_json(paths.transcript_json)
        if transcript.get("is_complete") is False:
            raise MovieContextError(
                "Whisper transcript còn dang dở. Hãy tiếp tục nhận diện các chunk còn thiếu trước."
            )
        segments = transcript.get("segments")
        if not isinstance(segments, list) or not segments:
            raise MovieContextError("Transcript không có danh sách segments hợp lệ.")

        fingerprint = sha256_file(paths.transcript_json)
        batch_size = max(40, min(int(batch_segments), 500))
        batches = [
            segments[index:index + batch_size]
            for index in range(0, len(segments), batch_size)
        ]

        state: dict[str, Any]
        if paths.context_json.is_file():
            try:
                candidate = read_json(paths.context_json)
            except ValueError:
                candidate = {}
            if candidate.get("transcript_fingerprint") == fingerprint:
                state = candidate
            else:
                state = self._new_state(fingerprint)
        else:
            state = self._new_state(fingerprint)

        if (
            state.get("is_complete") is not True
            and state.get("batch_segments") not in (None, batch_size)
        ):
            self._log("Kích thước chapter đã đổi; tạo checkpoint phân tích mới để tránh bỏ sót đoạn.")
            state = self._new_state(fingerprint)

        system_prompt = load_prompt("system_prompt.txt")
        analysis_template = load_prompt("context_analysis.txt")
        finalize_template = load_prompt("context_finalize.txt")
        config_signature = sha256_text(
            fingerprint
            + self.model
            + str(batch_size)
            + system_prompt
            + analysis_template
            + finalize_template
        )
        if state.get("config_signature") != config_signature:
            state = self._new_state(fingerprint)
        state["config_signature"] = config_signature

        profile = self._normalize_profile(state.get("profile"))
        processed = int(state.get("processed_segments", 0) or 0)
        processed = min(processed, len(segments))
        if (
            state.get("config_signature") == config_signature
            and state.get("is_complete") is True
            and processed == len(segments)
            and isinstance(state.get("profile"), dict)
        ):
            self._emit(
                TranscriptionProgress(
                    100,
                    "Context Profile đã hoàn tất; không cần gọi Gemini lại.",
                    len(segments),
                )
            )
            self._log(f"Đã dùng lại Context Profile: {paths.context_json}")
            return paths.context_json

        self._log(f"Phân tích {len(segments)} segment thành {len(batches)} chapter.")
        self._log(
            "Resume context: "
            + (f"{processed}/{len(segments)} segments đã phân tích." if processed else "bắt đầu mới.")
        )

        for batch_index, chunk in enumerate(batches):
            chunk_end = (batch_index + 1) * batch_size
            if chunk_end <= processed:
                continue
            if cancel_event.is_set():
                raise MovieContextError("Đã hủy phân tích ngữ cảnh phim.")

            chunk_start = batch_index * batch_size
            chapter_payload = {
                "chapter_index": batch_index + 1,
                "chapter_count": len(batches),
                "first_segment_id": chunk[0].get("id"),
                "last_segment_id": chunk[-1].get("id"),
                "transcript_is_complete_input": len(batches) == 1,
                "segments": self._prompt_segments(chunk),
            }
            prompt = render_prompt(
                analysis_template,
                {
                    "TRANSCRIPT_JSON": json.dumps(chapter_payload, ensure_ascii=False),
                    "CONTEXT_PROFILE_JSON": json.dumps(profile, ensure_ascii=False),
                },
            )
            self._log(
                f"Gemini context chapter {batch_index + 1}/{len(batches)} "
                f"(segments {chunk_start + 1}-{min(chunk_end, len(segments))})..."
            )
            response = self.client.generate_json(
                prompt,
                system_prompt,
                cancel_event=cancel_event,
            )
            profile = self._merge_profiles(profile, response)
            processed = min(chunk_end, len(segments))
            state.update(
                {
                    "schema_version": 1,
                    "transcript_fingerprint": fingerprint,
                    "source_transcript": str(paths.transcript_json),
                    "model": self.model,
                    "config_signature": config_signature,
                    "batch_segments": batch_size,
                    "processed_segments": processed,
                    "total_segments": len(segments),
                    "is_complete": False,
                    "profile": profile,
                }
            )
            write_json_atomic(paths.context_json, state)
            percent = min(90, int(processed / len(segments) * 90))
            self._emit(
                TranscriptionProgress(
                    percent,
                    f"Đã phân tích ngữ cảnh {processed}/{len(segments)} đoạn...",
                    processed,
                )
            )

        if cancel_event.is_set():
            raise MovieContextError("Đã hủy phân tích ngữ cảnh phim.")

        finalize_prompt = render_prompt(
            finalize_template,
            {
                "CONTEXT_PROFILE_JSON": json.dumps(profile, ensure_ascii=False),
                "SEGMENT_COUNT": str(len(segments)),
            },
        )
        cached_finalize = state.get("finalize_response")
        if (
            state.get("finalize_signature") == config_signature
            and isinstance(cached_finalize, dict)
        ):
            self._log("Dùng cache phản hồi hợp nhất Context Profile.")
            finalized = cached_finalize
        else:
            self._log("Gemini đang hợp nhất nhân vật, quan hệ và thuật ngữ...")
            finalized = self.client.generate_json(
                finalize_prompt,
                system_prompt,
                cancel_event=cancel_event,
            )
            # Save the final API response before normalizing/writing the completed profile.
            state["finalize_signature"] = config_signature
            state["finalize_response"] = finalized
            write_json_atomic(paths.context_json, state)

        profile = self._merge_profiles(profile, finalized)
        profile["film_profile"]["is_partial"] = False
        state.update(
            {
                "config_signature": config_signature,
                "model": self.model,
                "batch_segments": batch_size,
                "processed_segments": len(segments),
                "total_segments": len(segments),
                "is_complete": True,
                "profile": profile,
            }
        )
        write_json_atomic(paths.context_json, state)
        self._emit(
            TranscriptionProgress(
                100,
                "Phân tích ngữ cảnh phim hoàn tất.",
                len(segments),
            )
        )
        self._log(f"Context Profile: {paths.context_json}")
        return paths.context_json

    @staticmethod
    def _prompt_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Exclude verbose word timings; Gemini needs dialogue-level context only."""
        fields = ("id", "start", "end", "text", "speaker_id")
        return [
            {key: item[key] for key in fields if key in item}
            for item in segments
            if isinstance(item, dict)
        ]

    @classmethod
    def _merge_profiles(cls, previous: dict[str, Any], incoming: Any) -> dict[str, Any]:
        """Merge chapter observations instead of dropping facts from earlier chapters."""
        old = cls._normalize_profile(previous)
        new = cls._normalize_profile(incoming)

        film = dict(old["film_profile"])
        for key, value in new["film_profile"].items():
            if key == "continuity_facts" and isinstance(value, list):
                film[key] = cls._unique_items(
                    list(film.get(key, [])) + value
                )
            elif value not in ("", None, [], {}):
                film[key] = value
        old["film_profile"] = film

        old["characters"] = cls._merge_objects(
            old.get("characters", []), new.get("characters", []), ("character_id", "name")
        )
        old["relationships"] = cls._unique_items(
            list(old.get("relationships", [])) + list(new.get("relationships", []))
        )
        old["terminology"] = cls._merge_objects(
            old.get("terminology", []), new.get("terminology", []), ("source_term",)
        )
        old["uncertainties"] = cls._unique_items(
            list(old.get("uncertainties", [])) + list(new.get("uncertainties", []))
        )

        style = dict(old["style_guide"])
        for key, value in new["style_guide"].items():
            if isinstance(value, list):
                style[key] = cls._unique_items(list(style.get(key, [])) + value)
            elif value not in ("", None, [], {}):
                style[key] = value
        old["style_guide"] = style
        return cls._normalize_profile(old)

    @staticmethod
    def _item_key(item: Any, fields: tuple[str, ...]) -> str:
        if isinstance(item, dict):
            for field in fields:
                value = item.get(field)
                if isinstance(value, str) and value.strip():
                    return f"{field}:{value.strip().casefold()}"
        return json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)

    @classmethod
    def _merge_objects(
        cls,
        previous: Any,
        incoming: Any,
        key_fields: tuple[str, ...],
    ) -> list[Any]:
        result: list[Any] = []
        positions: dict[str, int] = {}
        for item in list(previous or []) + list(incoming or []):
            key = cls._item_key(item, key_fields)
            if key in positions and isinstance(item, dict) and isinstance(result[positions[key]], dict):
                merged = dict(result[positions[key]])
                for field, value in item.items():
                    if isinstance(value, list):
                        merged[field] = cls._unique_items(list(merged.get(field, [])) + value)
                    elif value not in ("", None, [], {}):
                        merged[field] = value
                result[positions[key]] = merged
            else:
                positions[key] = len(result)
                result.append(item)
        return result

    @staticmethod
    def _unique_items(items: list[Any]) -> list[Any]:
        unique: list[Any] = []
        seen: set[str] = set()
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    @staticmethod
    def _new_state(fingerprint: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "transcript_fingerprint": fingerprint,
            "processed_segments": 0,
            "total_segments": 0,
            "is_complete": False,
            "profile": _PROFILE_DEFAULTS,
        }

    @staticmethod
    def _normalize_profile(value: Any) -> dict[str, Any]:
        profile = dict(value) if isinstance(value, dict) else {}
        for key, default in _PROFILE_DEFAULTS.items():
            current = profile.get(key)
            if isinstance(default, dict):
                merged = dict(default)
                if isinstance(current, dict):
                    merged.update(current)
                profile[key] = merged
            elif not isinstance(current, list):
                profile[key] = list(default)
        return profile

    def _emit(self, progress: TranscriptionProgress) -> None:
        if self.progress_callback:
            self.progress_callback(progress)

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
