from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    speaker_id: str | None = None
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptDocument:
    source_video: str
    language: str
    language_probability: float | None
    duration_seconds: float | None
    model_name: str
    segments: list[TranscriptSegment] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TranscriptionProgress:
    percent: int | None
    message: str
    segments_done: int = 0
