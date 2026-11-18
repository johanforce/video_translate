from __future__ import annotations

from typing import Any


_META_FIELDS = {"evidence_segment_ids", "confidence", "start", "end"}


def _clean(value: Any, max_text: int = 320) -> Any:
    if isinstance(value, str):
        compact = " ".join(value.split())
        return compact if len(compact) <= max_text else compact[: max_text - 1].rstrip() + "…"
    if isinstance(value, list):
        return [_clean(item, max_text) for item in value[:100] if item not in (None, "", [], {})]
    if isinstance(value, dict):
        return {
            str(key): _clean(item, max_text)
            for key, item in value.items()
            if key not in _META_FIELDS and item not in (None, "", [], {})
        }
    return value


def compact_context_profile(value: Any, *, include_uncertainties: bool = False) -> dict[str, Any]:
    """Remove evidence metadata and verbose fields not needed in repeated Gemini prompts."""
    source = value if isinstance(value, dict) else {}
    film = source.get("film_profile")
    film = film if isinstance(film, dict) else {}
    characters = source.get("characters")
    relationships = source.get("relationships")
    terminology = source.get("terminology")
    style = source.get("style_guide")
    compact: dict[str, Any] = {
        "film_profile": {
            key: _clean(film.get(key))
            for key in (
                "genre", "setting", "premise", "narrative_tone",
                "localization_style", "continuity_facts",
            )
            if film.get(key) not in (None, "", [], {})
        },
        "characters": [
            _clean({
                key: item.get(key)
                for key in (
                    "character_id", "name", "aliases", "role", "personality",
                    "speech_style", "relationships", "addressing_rules",
                )
                if item.get(key) not in (None, "", [], {})
            })
            for item in (characters if isinstance(characters, list) else [])
            if isinstance(item, dict)
        ][:80],
        "relationships": _clean(relationships[:100] if isinstance(relationships, list) else []),
        "terminology": [
            _clean({
                key: item.get(key)
                for key in ("source_term", "vietnamese_term", "keep_original", "meaning_in_context")
                if item.get(key) not in (None, "", [], {})
            })
            for item in (terminology if isinstance(terminology, list) else [])
            if isinstance(item, dict)
        ][:100],
        "style_guide": _clean({
            key: style.get(key)
            for key in (
                "pronoun_policy", "dialogue_style", "comedy_policy",
                "profanity_policy", "special_instructions",
            )
            if isinstance(style, dict) and style.get(key) not in (None, "", [], {})
        }),
    }
    facts = compact["film_profile"].get("continuity_facts")
    if isinstance(facts, list):
        compact["film_profile"]["continuity_facts"] = facts[-20:]
    if include_uncertainties:
        uncertainties = source.get("uncertainties")
        compact["uncertainties"] = _clean(
            uncertainties[-12:] if isinstance(uncertainties, list) else []
        )
    return compact
