from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from video_translator.core.models import TranscriptDocument


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp (HH:MM:SS,mmm)."""
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{millis:03d}"


def render_srt(document: TranscriptDocument) -> str:
    blocks: list[str] = []
    cue_number = 0
    for segment in document.segments:
        text = segment.text.strip().replace("\r\n", "\n").replace("\r", "\n")
        if not text:
            continue
        cue_number += 1
        blocks.append(
            f"{cue_number}\n"
            f"{format_srt_timestamp(segment.start)} --> "
            f"{format_srt_timestamp(segment.end)}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def save_transcript(
    document: TranscriptDocument,
    video_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Atomically save transcript JSON and source-language SRT."""
    source = Path(video_path)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    document.source_video = str(source.resolve())
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": document.source_video,
        "language": document.language,
        "language_probability": document.language_probability,
        "duration_seconds": document.duration_seconds,
        "model": document.model_name,
        "segments": [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker_id": segment.speaker_id,
                "words": [
                    {
                        "start": word.start,
                        "end": word.end,
                        "text": word.text,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ],
            }
            for segment in document.segments
        ],
    }

    json_path = destination / f"{source.stem}.transcript.json"
    srt_path = destination / f"{source.stem}.source.srt"
    _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2))
    _atomic_write(srt_path, render_srt(document))
    return json_path, srt_path


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
