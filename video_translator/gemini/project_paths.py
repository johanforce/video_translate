from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MovieProjectPaths:
    transcript_json: Path
    context_json: Path
    translated_json: Path
    translated_srt: Path
    quality_json: Path
    tts_dir: Path
    tts_manifest: Path

    @classmethod
    def from_transcript(cls, transcript_path: Path) -> "MovieProjectPaths":
        transcript = Path(transcript_path).expanduser().resolve()
        stem = transcript.name.removesuffix(".transcript.json")
        if stem == transcript.name:
            stem = transcript.stem
        parent = transcript.parent
        return cls(
            transcript_json=transcript,
            context_json=parent / f"{stem}.context.json",
            translated_json=parent / f"{stem}.vi.translation.json",
            translated_srt=parent / f"{stem}.vi.srt",
            quality_json=parent / f"{stem}.vi.quality.json",
            tts_dir=parent / f"{stem}.tts",
            tts_manifest=parent / f"{stem}.tts" / "manifest.json",
        )
