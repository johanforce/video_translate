from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Runtime settings. Secrets are read from environment, not committed."""

    output_dir: Path
    max_height: int = 1080
    allow_playlists: bool = False
    gemini_model: str = "gemini-3.6-flash"
    vieneu_voice: str = "Ngọc Lan"

    @classmethod
    def load(cls) -> "AppSettings":
        configured_dir = os.getenv("VIDEO_TRANSLATOR_OUTPUT_DIR", "").strip()
        output_dir = (
            Path(configured_dir).expanduser()
            if configured_dir
            else Path.home() / "Videos" / "VideoTranslator"
        )
        return cls(
            output_dir=output_dir,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip(),
            vieneu_voice=os.getenv("VIENEU_DEFAULT_VOICE", "Ngọc Lan").strip(),
        )
