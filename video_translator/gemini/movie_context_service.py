from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from video_translator.core.errors import VideoTranslatorError
from video_translator.core.models import TranscriptionProgress
from video_translator.gemini.client import GeminiJsonClient, GeminiOutputLimitError
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
        batch_segments: int = 120,
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

        system_prompt = load_prompt("system_prompt.txt")
        analysis_template = load_prompt("context_analysis.txt")
        finalize_template = load_prompt("context_finalize.txt")
        prompt_signature = system_prompt + analysis_template + finalize_template
        # Model and chapter size are execution settings, not profile identity.
        # A model switch or smaller chapter size must preserve accepted chapters.
        config_signature = sha256_text(fingerprint + prompt_signature)
        saved_signature = str(state.get("config_signature") or "")
        saved_model = str(state.get("model") or "")
        try:
            saved_batch_size = int(state.get("batch_segments") or batch_size)
        except (TypeError, ValueError):
            saved_batch_size = batch_size

        legacy_signatures = {
            sha256_text(
                fingerprint + str(saved_batch_size) + prompt_signature
            ),
        }
        if saved_model:
            legacy_signatures.add(
                sha256_text(
                    fingerprint
                    + saved_model
                    + str(saved_batch_size)
                    + prompt_signature
                )
            )

        signature_is_compatible = (
            saved_signature == config_signature
            or saved_signature in legacy_signatures
        )
        if not signature_is_compatible:
            # Transcript or analysis prompt changed: old observations are no longer
            # guaranteed to be valid, so start a new context checkpoint.
            state = self._new_state(fingerprint)
            saved_signature = config_signature
            saved_batch_size = batch_size

        profile = self._normalize_profile(state.get("profile"))
        processed = int(state.get("processed_segments", 0) or 0)
        processed = min(processed, len(segments))

        if (
            state.get("is_complete") is True
            and processed == len(segments)
            and isinstance(state.get("profile"), dict)
            and signature_is_compatible
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

        if (
            saved_signature != config_signature
            or saved_batch_size != batch_size
            or saved_model != self.model
        ):
            self._log(
                f"Giữ checkpoint {processed}/{len(segments)} segment; "
                f"chapter size {saved_batch_size} → {batch_size}, "
                f"Gemini model {saved_model or 'chưa ghi nhận'} → {self.model}."
            )
            # Old chapter cache keys/signatures were tied to the previous chunk
            # boundaries or model. The merged profile and processed count remain safe.
            state["chapter_responses"] = {}
            state.pop("finalize_response", None)
            state.pop("finalize_signature", None)
            state["config_signature"] = config_signature
            state["batch_segments"] = batch_size
            state["model"] = self.model
            state["transcript_fingerprint"] = fingerprint
            state["processed_segments"] = processed
            state["total_segments"] = len(segments)
            state["profile"] = profile
            state["is_complete"] = False
            write_json_atomic(paths.context_json, state)

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

        def analyze_chunk(
            chunk: list[dict[str, Any]],
            chunk_start: int,
            batch_index: int,
            base_chunk_start: int,
            base_chunk_end: int,
            current_profile: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal processed
            if cancel_event.is_set():
                raise MovieContextError("Đã hủy phân tích ngữ cảnh phim.")

            chunk_end = chunk_start + len(chunk)
            if not chunk:
                return current_profile

            chapter_payload = {
                "chapter_index": batch_index + 1,
                "chapter_count": len(batches),
                "chapter_segment_range": [chunk_start + 1, chunk_end],
                "first_segment_id": chunk[0].get("id"),
                "last_segment_id": chunk[-1].get("id"),
                "transcript_is_complete_input": len(batches) == 1,
                "segments": self._prompt_segments(chunk),
            }
            chapter_signature = sha256_text(
                config_signature
                + json.dumps(chapter_payload, ensure_ascii=False, sort_keys=True)
                + json.dumps(current_profile, ensure_ascii=False, sort_keys=True)
            )
            is_original_chapter = (
                chunk_start == base_chunk_start and chunk_end == base_chunk_end
            )
            cache_key = (
                str(batch_index)
                if is_original_chapter
                else f"{batch_index}:{chunk_start + 1}-{chunk_end}"
            )
            chapter_responses = state.setdefault("chapter_responses", {})
            cached_chapter = chapter_responses.get(cache_key)

            if (
                isinstance(cached_chapter, dict)
                and cached_chapter.get("signature") == chapter_signature
                and isinstance(cached_chapter.get("response"), dict)
            ):
                self._log(
                    f"Dùng cache phản hồi chapter {batch_index + 1}, "
                    f"segments {chunk_start + 1}-{chunk_end}."
                )
                response = cached_chapter["response"]
            else:
                prompt = render_prompt(
                    analysis_template,
                    {
                        "TRANSCRIPT_JSON": json.dumps(chapter_payload, ensure_ascii=False),
                        "CONTEXT_PROFILE_JSON": json.dumps(
                            current_profile, ensure_ascii=False
                        ),
                    },
                )
                self._log(
                    f"Gemini context chapter {batch_index + 1}/{len(batches)} "
                    f"(segments {chunk_start + 1}-{chunk_end})..."
                )
                if is_original_chapter:
                    debug_name = f"{paths.context_json.stem}.chapter-{batch_index + 1:02d}.raw.txt"
                else:
                    debug_name = (
                        f"{paths.context_json.stem}.chapter-{batch_index + 1:02d}."
                        f"seg-{chunk_start + 1:04d}-{chunk_end:04d}.raw.txt"
                    )
                debug_path = paths.context_json.with_name(debug_name)
                try:
                    response = self.client.generate_json(
                        prompt,
                        system_prompt,
                        cancel_event=cancel_event,
                        debug_path=debug_path,
                    )
                except GeminiOutputLimitError as exc:
                    if len(chunk) <= 1:
                        raise MovieContextError(
                            "Gemini vẫn chạm MAX_TOKENS khi chỉ phân tích 1 segment. "
                            f"Phản hồi lỗi: {exc}"
                        ) from exc

                    middle = len(chunk) // 2
                    left = chunk[:middle]
                    right = chunk[middle:]
                    self._log(
                        f"[MAX_TOKENS] Chapter {batch_index + 1}: "
                        f"{chunk_start + 1}-{chunk_end} quá dài; tự chia thành "
                        f"{chunk_start + 1}-{chunk_start + middle} và "
                        f"{chunk_start + middle + 1}-{chunk_end}."
                    )
                    current_profile = analyze_chunk(
                        left,
                        chunk_start,
                        batch_index,
                        base_chunk_start,
                        base_chunk_end,
                        current_profile,
                    )
                    return analyze_chunk(
                        right,
                        chunk_start + middle,
                        batch_index,
                        base_chunk_start,
                        base_chunk_end,
                        current_profile,
                    )

                # Checkpoint the valid JSON response before merging it into the profile.
                chapter_responses[cache_key] = {
                    "signature": chapter_signature,
                    "response": response,
                }
                state["chapter_responses"] = chapter_responses
                write_json_atomic(paths.context_json, state)

            current_profile = self._merge_profiles(current_profile, response)
            processed = max(processed, chunk_end)
            chapter_responses.pop(cache_key, None)
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
                    "profile": current_profile,
                    "chapter_responses": chapter_responses,
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
            return current_profile

        for batch_index in range(len(batches)):
            base_chunk_start = batch_index * batch_size
            base_chunk_end = min((batch_index + 1) * batch_size, len(segments))
            if base_chunk_end <= processed:
                continue
            chunk_start = max(base_chunk_start, processed)
            chunk = segments[chunk_start:base_chunk_end]
            if not chunk:
                continue
            profile = analyze_chunk(
                chunk,
                chunk_start,
                batch_index,
                base_chunk_start,
                base_chunk_end,
                profile,
            )

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
            chapter_signature = sha256_text(
                config_signature
                + json.dumps(chapter_payload, ensure_ascii=False, sort_keys=True)
                + json.dumps(profile, ensure_ascii=False, sort_keys=True)
            )
            chapter_responses = state.setdefault("chapter_responses", {})
            cached_chapter = chapter_responses.get(str(batch_index))
            if (
                isinstance(cached_chapter, dict)
                and cached_chapter.get("signature") == chapter_signature
                and isinstance(cached_chapter.get("response"), dict)
            ):
                self._log(
                    f"Dùng cache phản hồi chapter {batch_index + 1}/{len(batches)}."
                )
                response = cached_chapter["response"]
            else:
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
                debug_path = paths.context_json.with_name(
                    f"{paths.context_json.stem}.chapter-{batch_index + 1:02d}.raw.txt"
                )
                response = self.client.generate_json(
                    prompt,
                    system_prompt,
                    cancel_event=cancel_event,
                    debug_path=debug_path,
                )
                # Store the parsed chapter response before merging it into the accumulated profile.
                chapter_responses[str(batch_index)] = {
                    "signature": chapter_signature,
                    "response": response,
                }
                state["chapter_responses"] = chapter_responses
                write_json_atomic(paths.context_json, state)

            profile = self._merge_profiles(profile, response)
            processed = min(chunk_end, len(segments))
            chapter_responses.pop(str(batch_index), None)
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
                    "chapter_responses": chapter_responses,
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
            debug_path = paths.context_json.with_name(
                f"{paths.context_json.stem}.finalize.raw.txt"
            )
            try:
                finalized = self.client.generate_json(
                    finalize_prompt,
                    system_prompt,
                    cancel_event=cancel_event,
                    debug_path=debug_path,
                )
            except GeminiOutputLimitError as exc:
                self._log(
                    "Gemini chạm MAX_TOKENS khi hợp nhất Context Profile; "
                    "dùng kết quả hợp nhất cục bộ đã chống trùng lặp. "
                    f"Chi tiết: {exc}"
                )
                finalized = {}
            else:
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
            "chapter_responses": {},
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
