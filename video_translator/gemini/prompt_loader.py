from __future__ import annotations

from pathlib import Path


_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    path = _PROMPT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy prompt: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, values: dict[str, str]) -> str:
    """Replace only declared {{TOKEN}} markers; leave all other braces intact."""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered
