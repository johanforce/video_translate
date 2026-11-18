from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from video_translator.core.errors import VideoTranslatorError
from video_translator.core.models import TranscriptionProgress
from video_translator.gemini.client import GeminiJsonClient
from video_translator.gemini.json_utils import read_json, sha256_file, write_json_atomic
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

        profile = self._normalize_profile(state.get("profile"))
        processed = int(state.get("processed_segments", 0) or 0)
        processed = min(processed, len(segments))
        system_prompt = load_prompt("system_prompt.txt")
        analysis_template = load_prompt("context_analysis.txt")

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
                "transcript_is_complete_input": True,
                "segments": chunk,
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
            profile = self._normalize_profile(response)
            processed = min(chunk_end, len(segments))
            state.update(
                {
                    "schema_version": 1,
                    "transcript_fingerprint": fingerprint,
                    "source_transcript": str(paths.transcript_json),
                    "model": self.model,
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

        finalize_template = load_prompt("context_finalize.txt")
        finalize_prompt = render_prompt(
            finalize_template,
            {
                "CONTEXT_PROFILE_JSON": json.dumps(profile, ensure_ascii=False),
                "SEGMENT_COUNT": str(len(segments)),
            },
        )
        self._log("Gemini đang hợp nhất nhân vật, quan hệ và thuật ngữ...")
        finalized = self.client.generate_json(
            finalize_prompt,
            system_prompt,
            cancel_event=cancel_event,
        )
        profile = self._normalize_profile(finalized)
        profile["film_profile"]["is_partial"] = False
        state.update(
            {
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
