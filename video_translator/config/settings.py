from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Runtime settings for the desktop application."""

    output_dir: Path
    max_height: int = 1080
    allow_playlists: bool = False

    @classmethod
    def load(cls) -> "AppSettings":
        configured_dir = os.getenv("VIDEO_TRANSLATOR_OUTPUT_DIR", "").strip()
        if configured_dir:
            output_dir = Path(configured_dir).expanduser()
        else:
            output_dir = Path.home() / "Videos" / "VideoTranslator"

        return cls(output_dir=output_dir)
