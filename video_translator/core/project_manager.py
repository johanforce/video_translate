from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from video_translator.gemini.json_utils import (
    read_json,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from video_translator.gemini.project_paths import MovieProjectPaths


class ProjectStateError(RuntimeError):
    """Raised when project state cannot be loaded or reconciled."""


class ProjectManager:
    """Persistent project manifest inferred from durable pipeline artifacts."""

    STAGES = ("transcription", "context", "translation", "tts", "render")

    def __init__(self, transcript_path: Path) -> None:
        self.paths = MovieProjectPaths.from_transcript(transcript_path)
        self.project_file = self.paths.transcript_json.parent / (
            self.paths.transcript_json.name.removesuffix(".transcript.json") + ".project.json"
        )
        self.state = self._load()
        self.refresh()

    def _load(self) -> dict[str, Any]:
        if self.project_file.is_file():
            try:
                payload = read_json(self.project_file)
                if payload.get("schema_version") == 1:
                    return payload
            except (ValueError, OSError):
                pass
        now = self._now()
        return {
            "schema_version": 1,
            "project_name": self.paths.transcript_json.name.removesuffix(".transcript.json"),
            "created_at": now,
            "updated_at": now,
            "source_transcript": str(self.paths.transcript_json),
            "artifacts": {},
            "stages": {
                stage: {
                    "status": "not_started",
                    "progress": 0,
                    "updated_at": now,
                }
                for stage in self.STAGES
            },
            "history": [],
        }

    def refresh(self) -> dict[str, Any]:
        """Reconcile interrupted 'running' stages against on-disk checkpoints."""
        artifacts = {
            "transcript": self._artifact(self.paths.transcript_json),
            "context": self._artifact(self.paths.context_json),
            "translation": self._artifact(self.paths.translated_json),
            "subtitle": self._artifact(self.paths.translated_srt),
            "quality_report": self._artifact(self.paths.quality_json),
            "tts_manifest": self._artifact(self.paths.tts_manifest),
        }
        self.state["artifacts"] = artifacts
        self.state["source_transcript"] = str(self.paths.transcript_json)

        stages = self.state.setdefault("stages", {})
        self._set_reconciled(stages, "transcription", bool(artifacts["transcript"]))
        self._set_reconciled(
            stages,
            "context",
            self._context_complete(),
            self._context_progress(),
        )
        self._set_reconciled(
            stages,
            "translation",
            self._translation_complete(),
            self._translation_progress(),
        )
        self._set_reconciled(
            stages,
            "tts",
            self._tts_complete(),
            self._tts_progress(),
        )
        self._set_reconciled(stages, "render", self._render_complete())
        self._save()
        return self.state

    def start_stage(self, stage: str) -> None:
        self._validate_stage(stage)
        entry = self.state["stages"].setdefault(stage, {})
        entry.update(
            {
                "status": "running",
                "progress": int(entry.get("progress", 0)),
                "started_at": self._now(),
                "updated_at": self._now(),
                "last_error": "",
            }
        )
        self._append_history(stage, "running")
        self._save()

    def update_progress(self, stage: str, progress: int, message: str = "") -> None:
        self._validate_stage(stage)
        entry = self.state["stages"].setdefault(stage, {})
        entry.update(
            {
                "status": "running",
                "progress": max(0, min(100, int(progress))),
                "message": message,
                "updated_at": self._now(),
            }
        )
        self._save()

    def complete_stage(self, stage: str, **details: Any) -> None:
        self._validate_stage(stage)
        entry = self.state["stages"].setdefault(stage, {})
        entry.update(
            {
                "status": "completed",
                "progress": 100,
                "updated_at": self._now(),
                "last_error": "",
                **details,
            }
        )
        self._append_history(stage, "completed")
        self._save()
        self.refresh()

    def fail_stage(self, stage: str, error: str) -> None:
        self._validate_stage(stage)
        entry = self.state["stages"].setdefault(stage, {})
        entry.update(
            {
                "status": "interrupted" if "hủy" in error.lower() else "failed",
                "last_error": error,
                "updated_at": self._now(),
            }
        )
        self._append_history(stage, entry["status"], error)
        self._save()
        self.refresh()

    def next_incomplete_stage(self) -> str | None:
        self.refresh()
        for stage in self.STAGES:
            if self.state["stages"].get(stage, {}).get("status") != "completed":
                return stage
        return None

    def _context_complete(self) -> bool:
        if not self.paths.context_json.is_file():
            return False
        try:
            value = read_json(self.paths.context_json)
            return (
                value.get("is_complete") is True
                and value.get("transcript_fingerprint") == sha256_file(self.paths.transcript_json)
            )
        except (ValueError, OSError):
            return False

    def _context_progress(self) -> int:
        if not self.paths.context_json.is_file():
            return 0
        try:
            value = read_json(self.paths.context_json)
            total = int(value.get("total_segments", 0))
            processed = int(value.get("processed_segments", 0))
            return int(processed / total * 100) if total else 0
        except (ValueError, OSError, TypeError):
            return 0

    def _translation_complete(self) -> bool:
        if not self.paths.translated_json.is_file() or not self.paths.translated_srt.is_file():
            return False
        try:
            value = read_json(self.paths.translated_json)
            if not (
                value.get("is_complete") is True
                and isinstance(value.get("segments"), list)
                and bool(value["segments"])
                and value.get("source_fingerprint") == sha256_file(self.paths.transcript_json)
            ):
                return False
            context = read_json(self.paths.context_json)
            return value.get("context_fingerprint") == sha256_file(self.paths.context_json) and context.get("is_complete") is True
        except (ValueError, OSError):
            return False

    def _translation_progress(self) -> int:
        if not self.paths.translated_json.is_file():
            return 0
        try:
            value = read_json(self.paths.translated_json)
            total = int(value.get("total_segments", 0))
            done = int(value.get("completed_segments", 0))
            return int(done / total * 100) if total else 0
        except (ValueError, OSError, TypeError):
            return 0

    def _tts_complete(self) -> bool:
        if not self.paths.tts_manifest.is_file():
            return False
        try:
            value = read_json(self.paths.tts_manifest)
            if value.get("is_complete") is not True or not self.paths.translated_json.is_file():
                return False
            translation_fingerprint = sha256_text(
                self.paths.translated_json.read_text(encoding="utf-8")
            )
            if value.get("translation_fingerprint") != translation_fingerprint:
                return False
            entries = value.get("segments", {})
            if not isinstance(entries, dict):
                return False
            for entry in entries.values():
                if not isinstance(entry, dict) or entry.get("status") != "completed":
                    return False
                audio = Path(str(entry.get("audio_path", "")))
                if not audio.is_file() or audio.stat().st_size <= 64:
                    return False
            return True
        except (ValueError, OSError):
            return False

    def _tts_progress(self) -> int:
        if not self.paths.tts_manifest.is_file():
            return 0
        try:
            value = read_json(self.paths.tts_manifest)
            total = int(value.get("total_segments", 0))
            done = int(value.get("completed_segments", 0))
            return int(done / total * 100) if total else 0
        except (ValueError, OSError, TypeError):
            return 0

    def _render_complete(self) -> bool:
        output = self.paths.transcript_json.parent / (
            self.paths.transcript_json.name.removesuffix(".transcript.json") + ".vi.dubbed.mp4"
        )
        metadata = output.with_suffix(".render.json")
        if not output.is_file() or output.stat().st_size < 1024 or not metadata.is_file():
            return False
        try:
            info = read_json(metadata)
            if info.get("output_signature") != sha256_file(output):
                return False
            if not self.paths.translated_json.is_file() or not self.paths.tts_manifest.is_file():
                return False
            if info.get("translation_fingerprint") != sha256_file(self.paths.translated_json):
                return False
            transcript = read_json(self.paths.transcript_json)
            source_video = Path(str(transcript.get("source_video", "")))
            if not source_video.is_file():
                return False
            if info.get("source_video_fingerprint") != sha256_file(source_video):
                return False
            if info.get("tts_manifest_fingerprint") != sha256_file(self.paths.tts_manifest):
                return False
            return True
        except (ValueError, OSError):
            return False

    @staticmethod
    def _artifact(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        stat = path.stat()
        return {
            "path": str(path),
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
        }

    @staticmethod
    def _set_reconciled(
        stages: dict[str, Any],
        stage: str,
        complete: bool,
        progress: int | None = None,
    ) -> None:
        entry = stages.setdefault(stage, {})
        # Do not erase failed/interrupted labels if there is unfinished checkpoint data.
        if complete:
            entry.update({"status": "completed", "progress": 100})
        elif entry.get("status") == "running":
            entry["status"] = "interrupted"
            if progress is not None:
                entry["progress"] = progress
        elif entry.get("status") == "completed":
            entry["status"] = "not_started"
            if progress is not None:
                entry["progress"] = progress
        elif progress is not None:
            entry["progress"] = progress
        entry["updated_at"] = ProjectManager._now()

    def _append_history(self, stage: str, status: str, error: str = "") -> None:
        history = self.state.setdefault("history", [])
        history.append(
            {
                "stage": stage,
                "status": status,
                "error": error[:1000],
                "at": self._now(),
            }
        )
        if len(history) > 200:
            del history[:-200]

    def _save(self) -> None:
        self.state["updated_at"] = self._now()
        write_json_atomic(self.project_file, self.state)

    def _validate_stage(self, stage: str) -> None:
        if stage not in self.STAGES:
            raise ValueError(f"Stage không hợp lệ: {stage}")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
