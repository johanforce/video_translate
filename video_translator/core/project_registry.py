from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from video_translator.gemini.json_utils import read_json, write_json_atomic


class ProjectRegistry:
    """Maps source videos to their transcript-backed project files across app restarts."""

    def __init__(self, registry_path: Path | None = None) -> None:
        self.path = (
            Path(registry_path).expanduser()
            if registry_path is not None
            else Path.home() / ".video_translator" / "projects.json"
        )

    @staticmethod
    def _key(path: Path) -> str:
        return os.path.normcase(str(Path(path).expanduser().resolve()))

    def register(self, transcript_path: Path) -> None:
        transcript = Path(transcript_path).expanduser().resolve()
        if not transcript.is_file():
            return
        try:
            payload = read_json(transcript)
        except (OSError, ValueError):
            return
        source_text = str(payload.get("source_video") or "").strip()
        if not source_text:
            return
        source = Path(source_text).expanduser()
        projects = self._load()
        projects[self._key(source)] = str(transcript)
        write_json_atomic(self.path, {"schema_version": 1, "projects": projects})

    def find_for_video(
        self,
        video_path: Path,
        search_directories: tuple[Path, ...] = (),
    ) -> Path | None:
        video = Path(video_path).expanduser().resolve()
        candidates: list[Path] = []

        registered = self._load().get(self._key(video))
        if isinstance(registered, str) and registered.strip():
            candidates.append(Path(registered).expanduser())

        for directory in search_directories:
            folder = Path(directory).expanduser()
            if not folder.is_dir():
                continue
            direct = folder / f"{video.stem}.transcript.json"
            if direct.is_file():
                candidates.append(direct)
            try:
                candidates.extend(folder.glob("*.transcript.json"))
            except OSError:
                pass

        seen: set[str] = set()
        for candidate in candidates:
            try:
                candidate = candidate.resolve()
                key = self._key(candidate)
                if key in seen or not candidate.is_file():
                    continue
                seen.add(key)
                payload = read_json(candidate)
                source_text = str(payload.get("source_video") or "").strip()
                if not source_text:
                    continue
                recorded_source = Path(source_text).expanduser()
                try:
                    same_source = self._key(recorded_source) == self._key(video)
                except OSError:
                    same_source = False
                # If the original video was moved, the transcript remains discoverable
                # by its filename only when the old source path no longer exists.
                moved_same_name = (
                    not recorded_source.is_file()
                    and recorded_source.name.casefold() == video.name.casefold()
                    and candidate.name == f"{video.stem}.transcript.json"
                )
                if same_source or moved_same_name:
                    return candidate
            except (OSError, ValueError):
                continue
        return None

    def _load(self) -> dict[str, Any]:
        try:
            payload = read_json(self.path)
        except (OSError, ValueError):
            return {}
        projects = payload.get("projects")
        return projects if isinstance(projects, dict) else {}
