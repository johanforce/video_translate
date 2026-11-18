import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_translator.gemini.film_translation_service import FilmTranslationError, FilmTranslationService
from video_translator.gemini.movie_context_service import MovieContextService
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.gemini.json_utils import write_json_atomic


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


class FakeRecoverableTranslationClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.translation_calls = 0
        self.qc_calls = 0
        self.fail_qc_once = True
        self.translated = [
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

    def generate_json(self, prompt: str, system_instruction: str, cancel_event=None):
        if "TASK: TRANSLATE A TIMESTAMPED FILM DIALOGUE BATCH" in prompt:
            self.translation_calls += 1
            return {"segments": self.translated}
        self.qc_calls += 1
        if self.fail_qc_once:
            self.fail_qc_once = False
            raise RuntimeError("503 temporary Gemini outage")
        return {
            "segments": self.translated,
            "issues": [],
            "summary": {"reviewed_segments": 2, "corrected_segments": 0, "critical_issues": 0, "notes": []},
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

    def test_context_finalize_response_is_cached_after_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript_path = self._write_transcript(root)
            fake_client = FakeContextClient("test", "test-model")
            real_write = write_json_atomic
            writes = {"count": 0}

            def write_then_simulate_crash(path, payload):
                writes["count"] += 1
                real_write(path, payload)
                if writes["count"] == 3:
                    raise RuntimeError("simulated power loss after final response checkpoint")

            with (
                patch(
                    "video_translator.gemini.movie_context_service.GeminiJsonClient",
                    return_value=fake_client,
                ),
                patch(
                    "video_translator.gemini.movie_context_service.write_json_atomic",
                    side_effect=write_then_simulate_crash,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated power loss"):
                    MovieContextService("test", "test-model").run(
                        transcript_path,
                        batch_segments=40,
                    )

            context_path = MovieProjectPaths.from_transcript(transcript_path).context_json
            checkpoint = json.loads(context_path.read_text(encoding="utf-8"))
            self.assertIn("finalize_response", checkpoint)
            calls_after_crash = fake_client.calls
            self.assertEqual(calls_after_crash, 2)

            with patch(
                "video_translator.gemini.movie_context_service.GeminiJsonClient",
                return_value=fake_client,
            ):
                MovieContextService("test", "test-model").run(
                    transcript_path,
                    batch_segments=40,
                )

            self.assertEqual(fake_client.calls, calls_after_crash)
            completed = json.loads(context_path.read_text(encoding="utf-8"))
            self.assertTrue(completed["is_complete"])

    def test_context_chapter_response_is_cached_before_profile_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript_path = self._write_transcript(root)
            fake_client = FakeContextClient("test", "test-model")
            real_write = write_json_atomic
            writes = {"count": 0}

            def write_then_simulate_crash(path, payload):
                writes["count"] += 1
                real_write(path, payload)
                if writes["count"] == 1:
                    raise RuntimeError("simulated power loss after chapter response checkpoint")

            with (
                patch(
                    "video_translator.gemini.movie_context_service.GeminiJsonClient",
                    return_value=fake_client,
                ),
                patch(
                    "video_translator.gemini.movie_context_service.write_json_atomic",
                    side_effect=write_then_simulate_crash,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "after chapter response"):
                    MovieContextService("test", "test-model").run(
                        transcript_path,
                        batch_segments=40,
                    )

            context_path = MovieProjectPaths.from_transcript(transcript_path).context_json
            checkpoint = json.loads(context_path.read_text(encoding="utf-8"))
            self.assertIn("0", checkpoint["chapter_responses"])
            self.assertEqual(fake_client.calls, 1)

            with patch(
                "video_translator.gemini.movie_context_service.GeminiJsonClient",
                return_value=fake_client,
            ):
                MovieContextService("test", "test-model").run(
                    transcript_path,
                    batch_segments=40,
                )

            # The chapter response was read from disk; only the finalizer is called.
            self.assertEqual(fake_client.calls, 2)
            completed = json.loads(context_path.read_text(encoding="utf-8"))
            self.assertTrue(completed["is_complete"])
            self.assertEqual(completed["chapter_responses"], {})

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

    def test_translation_reuses_translation_cache_after_qc_api_error(self) -> None:
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
                        },
                    }
                ),
                encoding="utf-8",
            )
            fake_client = FakeRecoverableTranslationClient("test", "test-model")
            with patch(
                "video_translator.gemini.film_translation_service.GeminiJsonClient",
                return_value=fake_client,
            ):
                service = FilmTranslationService("test", "test-model")
                with self.assertRaisesRegex(RuntimeError, "503"):
                    service.run(transcript_path, paths.context_json, batch_size=10)

                checkpoint = json.loads(paths.translated_json.read_text(encoding="utf-8"))
                pending = checkpoint["pending_batches"]["1-2"]
                self.assertEqual(len(pending["translation"]), 2)
                self.assertNotIn("quality_control", pending)
                self.assertEqual(fake_client.translation_calls, 1)

                service.run(transcript_path, paths.context_json, batch_size=10)

            self.assertEqual(fake_client.translation_calls, 1)
            self.assertEqual(fake_client.qc_calls, 2)
            completed = json.loads(paths.translated_json.read_text(encoding="utf-8"))
            self.assertTrue(completed["is_complete"])
            self.assertEqual(completed["pending_batches"], {})

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
