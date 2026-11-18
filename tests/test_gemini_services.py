import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_translator.gemini.film_translation_service import FilmTranslationError, FilmTranslationService
from video_translator.gemini.movie_context_service import MovieContextService
from video_translator.gemini.project_paths import MovieProjectPaths


class FakeContextClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.calls = 0

    def generate_json(self, prompt: str, system_instruction: str, cancel_event=None):
        self.calls += 1
        return {
            "film_profile": {
                "is_partial": True,
                "genre": "Drama",
                "setting": "Modern city",
                "premise": "A family conflict",
                "narrative_tone": "Serious",
                "localization_style": "Natural spoken Vietnamese",
                "continuity_facts": ["A family is in conflict"],
            },
            "characters": [
                {
                    "character_id": "char_1",
                    "name": "Anna",
                    "aliases": [],
                    "role": "Daughter",
                    "personality": "Determined",
                    "speech_style": "Direct",
                    "relationships": [],
                    "addressing_rules": [],
                    "evidence_segment_ids": [1],
                    "confidence": "high",
                }
            ],
            "relationships": [],
            "terminology": [],
            "style_guide": {
                "pronoun_policy": "Use context-based Vietnamese pronouns",
                "dialogue_style": "Natural",
                "comedy_policy": "",
                "profanity_policy": "",
                "special_instructions": [],
            },
            "uncertainties": [],
        }


class FakeTranslationClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.calls = 0

    def generate_json(self, prompt: str, system_instruction: str, cancel_event=None):
        self.calls += 1
        translated = [
            {
                "id": 1,
                "start": 0.25,
                "end": 1.75,
                "speaker_id": "char_1",
                "translated_text": "Đừng đi!",
                "emotion": "fearful",
                "delivery": "shouting",
            },
            {
                "id": 2,
                "start": 2.0,
                "end": 4.5,
                "speaker_id": "char_1",
                "translated_text": "Con vẫn chưa nói xong.",
                "emotion": "sad",
                "delivery": "soft",
            },
        ]
        if self.calls % 2 == 1:
            return {"segments": translated}
        return {
            "segments": translated,
            "issues": [],
            "summary": {
                "reviewed_segments": 2,
                "corrected_segments": 0,
                "critical_issues": 0,
                "notes": [],
            },
        }


class GeminiServiceTests(unittest.TestCase):
    def _write_transcript(self, root: Path) -> Path:
        transcript_path = root / "movie.transcript.json"
        transcript_path.write_text(
            json.dumps(
                {
                    "language": "en",
                    "duration_seconds": 5.0,
                    "segments": [
                        {"id": 1, "start": 0.25, "end": 1.75, "text": "Don't go!"},
                        {"id": 2, "start": 2.0, "end": 4.5, "text": "I am not finished."},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return transcript_path

    def test_context_profile_is_saved_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript_path = self._write_transcript(root)
            fake_client = FakeContextClient("test", "test-model")
            with patch(
                "video_translator.gemini.movie_context_service.GeminiJsonClient",
                return_value=fake_client,
            ):
                service = MovieContextService("test", "test-model")
                context_path = service.run(transcript_path, batch_segments=40)
                self.assertTrue(context_path.is_file())
                state = json.loads(context_path.read_text(encoding="utf-8"))
                self.assertTrue(state["is_complete"])
                self.assertFalse(state["profile"]["film_profile"]["is_partial"])
                self.assertEqual(state["profile"]["characters"][0]["character_id"], "char_1")
                first_call_count = fake_client.calls

                service.run(transcript_path, batch_segments=40)
                self.assertEqual(fake_client.calls, first_call_count)

    def test_translation_preserves_ids_and_source_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript_path = self._write_transcript(root)
            paths = MovieProjectPaths.from_transcript(transcript_path)
            paths.context_json.write_text(
                json.dumps(
                    {
                        "is_complete": True,
                        "profile": {
                            "film_profile": {"genre": "Drama"},
                            "characters": [],
                            "relationships": [],
                            "terminology": [],
                            "style_guide": {},
                            "uncertainties": [],
                        }
                    }
                ),
                encoding="utf-8",
            )
            fake_client = FakeTranslationClient("test", "test-model")
            with patch(
                "video_translator.gemini.film_translation_service.GeminiJsonClient",
                return_value=fake_client,
            ):
                service = FilmTranslationService("test", "test-model")
                json_path, srt_path, qc_path = service.run(
                    transcript_path,
                    paths.context_json,
                    batch_size=10,
                )

            translated = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual([item["id"] for item in translated["segments"]], [1, 2])
            self.assertEqual(
                [(item["start"], item["end"]) for item in translated["segments"]],
                [(0.25, 1.75), (2.0, 4.5)],
            )
            self.assertEqual(translated["segments"][0]["translated_text"], "Đừng đi!")
            self.assertTrue(srt_path.is_file())
            self.assertTrue(qc_path.is_file())
            srt = srt_path.read_text(encoding="utf-8")
            self.assertIn("00:00:00,250 --> 00:00:01,750", srt)
            self.assertIn("Đừng đi!", srt)

    def test_translation_rejects_timestamp_changes(self) -> None:
        source = [{"id": 1, "start": 1.0, "end": 2.0, "text": "Hello"}]
        result = [
            {
                "id": 1,
                "start": 1.1,
                "end": 2.0,
                "translated_text": "Xin chào",
            }
        ]
        with self.assertRaisesRegex(FilmTranslationError, "timestamp"):
            FilmTranslationService._validate_segments(source, result, "test")


if __name__ == "__main__":
    unittest.main()
