from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DownloadResult:
    video_id: str
    title: str
    path: Path
    duration_seconds: float | None
    source_height: int | None


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    percent: float | None
    message: str
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    speed_bytes_per_second: float | None = None
    eta_seconds: int | None = None
