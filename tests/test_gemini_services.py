import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_translator.gemini.client import GeminiOutputLimitError
from video_translator.gemini.film_translation_service import FilmTranslationError, FilmTranslationService
from video_translator.gemini.movie_context_service import MovieContextService
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.gemini.json_utils import sha256_file, sha256_text, write_json_atomic
from video_translator.gemini.prompt_loader import load_prompt


class FakeContextClient:
    def __init__(self, api_key: str, model: str) -> None:
        self.calls = 0

    def generate_json(self, prompt: str, system_instruction: str, cancel_event=None, debug_path=None):
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
        if "TASK: TRANSLATE A FILM DIALOGUE BATCH" in prompt:
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

    def test_context_model_switch_preserves_completed_chapters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript_path = root / "movie.transcript.json"
            segments = [
                {
                    "id": index,
                    "start": float(index),
                    "end": float(index) + 0.8,
                    "text": f"Dialogue {index}",
                }
                for index in range(1, 81)
            ]
            transcript_path.write_text(
                json.dumps({"is_complete": True, "segments": segments}),
                encoding="utf-8",
            )
            paths = MovieProjectPaths.from_transcript(transcript_path)
            old_model = "gemini-3.6-flash"
            batch_size = 40
            system_prompt = load_prompt("system_prompt.txt")
            analysis_prompt = load_prompt("context_analysis.txt")
            finalize_prompt = load_prompt("context_finalize.txt")
            legacy_signature = sha256_text(
                sha256_file(transcript_path)
                + old_model
                + str(batch_size)
                + system_prompt
                + analysis_prompt
                + finalize_prompt
            )
            paths.context_json.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "transcript_fingerprint": sha256_file(transcript_path),
                        "config_signature": legacy_signature,
                        "model": old_model,
                        "batch_segments": batch_size,
                        "processed_segments": 40,
                        "total_segments": 80,
                        "is_complete": False,
                        "profile": {
                            "film_profile": {
                                "is_partial": True,
                                "genre": "Drama",
                                "continuity_facts": ["Fact from chapter 1"],
                            },
                            "characters": [],
                            "relationships": [],
                            "terminology": [],
                            "style_guide": {},
                            "uncertainties": [],
                        },
                        "chapter_responses": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            fake_client = FakeContextClient("test", "gemini-3.8-flash")
            with patch(
                "video_translator.gemini.movie_context_service.GeminiJsonClient",
                return_value=fake_client,
            ):
                MovieContextService("test", "gemini-3.8-flash").run(
                    transcript_path,
                    batch_segments=batch_size,
                )

            completed = json.loads(paths.context_json.read_text(encoding="utf-8"))
            # One remaining chapter plus the final profile consolidation.
            self.assertEqual(fake_client.calls, 2)
            self.assertTrue(completed["is_complete"])
            self.assertEqual(completed["processed_segments"], 80)
            self.assertEqual(
                completed["profile"]["film_profile"]["continuity_facts"][0],
                "Fact from chapter 1",
            )
            self.assertEqual(completed["model"], "gemini-3.8-flash")

    def test_translation_model_switch_preserves_completed_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript_path = self._write_transcript(root)
            paths = MovieProjectPaths.from_transcript(transcript_path)
            source_fingerprint = "source-fingerprint"
            context_fingerprint = "context-fingerprint"
            batch_size = 5
            prompt_signature = (
                load_prompt("system_prompt.txt")
                + load_prompt("translation.txt")
                + load_prompt("quality_control.txt")
            )
            old_model = "gemini-3.6-flash"
            legacy_signature = sha256_text(
                source_fingerprint
                + context_fingerprint
                + old_model
                + str(batch_size)
                + prompt_signature
            )
            new_signature = sha256_text(
                source_fingerprint
                + context_fingerprint
                + str(batch_size)
                + prompt_signature
            )
            preserved = [
                {
                    "id": 1,
                    "start": 0.25,
                    "end": 1.75,
                    "translated_text": "Đã dịch trước đó",
                }
            ]
            paths.translated_json.write_text(
                json.dumps(
                    {
                        "config_signature": legacy_signature,
                        "model": old_model,
                        "source_fingerprint": source_fingerprint,
                        "context_fingerprint": context_fingerprint,
                        "batch_size": batch_size,
                        "segments": preserved,
                        "pending_batches": {},
                        "issues": [],
                        "is_complete": False,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            resumed = FilmTranslationService._load_or_create_state(
                paths,
                new_signature,
                source_fingerprint,
                context_fingerprint,
                batch_size,
                prompt_signature,
                "gemini-3.8-flash",
            )

            self.assertEqual(resumed["segments"], preserved)
            self.assertEqual(resumed["config_signature"], new_signature)
            self.assertEqual(resumed["model"], "gemini-3.8-flash")

    def test_context_splits_chapter_automatically_after_max_tokens(self) -> None:
        class AutoSplitContextClient:
            def __init__(self, api_key: str, model: str) -> None:
                self.calls = 0
                self.analyzed_ranges = []

            def generate_json(self, prompt: str, system_instruction: str, cancel_event=None, debug_path=None):
                self.calls += 1
                if "INPUT TRANSCRIPT:\n" not in prompt:
                    return {}
                raw = prompt.split("INPUT TRANSCRIPT:\n", 1)[1].split(
                    "\n\nEXISTING PROJECT CONTEXT", 1
                )[0]
                payload = json.loads(raw)
                if len(payload["segments"]) > 4:
                    raise GeminiOutputLimitError("finish_reason=MAX_TOKENS")
                segment_range = payload["chapter_segment_range"]
                self.analyzed_ranges.append(segment_range)
                return {
                    "film_profile": {
                        "is_partial": True,
                        "continuity_facts": [
                            f"Analyzed {segment_range[0]}-{segment_range[1]}"
                        ],
                    },
                    "characters": [],
                    "relationships": [],
                    "terminology": [],
                    "style_guide": {},
                    "uncertainties": [],
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript_path = root / "split.transcript.json"
            transcript_path.write_text(
                json.dumps({
                    "is_complete": True,
                    "segments": [
                        {
                            "id": index,
                            "start": float(index),
                            "end": float(index) + 0.5,
                            "text": f"Dialogue {index}",
                        }
                        for index in range(1, 9)
                    ],
                }),
                encoding="utf-8",
            )
            fake_client = AutoSplitContextClient("test", "gemini-3.8-flash")
            logs = []
            with patch(
                "video_translator.gemini.movie_context_service.GeminiJsonClient",
                return_value=fake_client,
            ):
                context_path = MovieContextService(
                    "test",
                    "gemini-3.8-flash",
                    log_callback=logs.append,
                ).run(transcript_path, batch_segments=40)

            state = json.loads(context_path.read_text(encoding="utf-8"))
            self.assertTrue(state["is_complete"])
            self.assertEqual(state["processed_segments"], 8)
            self.assertEqual(fake_client.calls, 4)
            self.assertEqual(fake_client.analyzed_ranges, [[1, 4], [5, 8]])
            self.assertTrue(any("[MAX_TOKENS]" in line for line in logs))

    def test_context_smaller_chapters_resume_from_existing_processed_count(self) -> None:
        class RecordingContextClient:
            def __init__(self, api_key: str, model: str) -> None:
                self.calls = 0
                self.analyzed_ranges = []

            def generate_json(self, prompt: str, system_instruction: str, cancel_event=None, debug_path=None):
                self.calls += 1
                if "INPUT TRANSCRIPT:\n" not in prompt:
                    return {}
                raw = prompt.split("INPUT TRANSCRIPT:\n", 1)[1].split(
                    "\n\nEXISTING PROJECT CONTEXT", 1
                )[0]
                payload = json.loads(raw)
                self.analyzed_ranges.append(payload["chapter_segment_range"])
                return {
                    "film_profile": {
                        "is_partial": True,
                        "continuity_facts": [
                            f"Analyzed {payload['chapter_segment_range'][0]}-"
                            f"{payload['chapter_segment_range'][1]}"
                        ],
                    },
                    "characters": [],
                    "relationships": [],
                    "terminology": [],
                    "style_guide": {},
                    "uncertainties": [],
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript_path = root / "resume.transcript.json"
            segments = [
                {
                    "id": index,
                    "start": float(index),
                    "end": float(index) + 0.5,
                    "text": f"Dialogue {index}",
                }
                for index in range(1, 528)
            ]
            transcript_path.write_text(
                json.dumps({"is_complete": True, "segments": segments}),
                encoding="utf-8",
            )
            paths = MovieProjectPaths.from_transcript(transcript_path)
            old_model = "gemini-3.6-flash"
            old_batch_size = 180
            prompt_signature = (
                load_prompt("system_prompt.txt")
                + load_prompt("context_analysis.txt")
                + load_prompt("context_finalize.txt")
            )
            # Simulate a saved project created with the previous longer prompt.
            legacy_signature = sha256_text(
                sha256_file(transcript_path)
                + old_model
                + str(old_batch_size)
                + "previous context prompt revision"
            )
            paths.context_json.write_text(
                json.dumps({
                    "schema_version": 1,
                    "transcript_fingerprint": sha256_file(transcript_path),
                    "config_signature": legacy_signature,
                    "model": old_model,
                    "batch_segments": old_batch_size,
                    "processed_segments": 360,
                    "total_segments": 527,
                    "is_complete": False,
                    "profile": {
                        "film_profile": {
                            "is_partial": True,
                            "continuity_facts": ["Fact from first 360 segments"],
                        },
                        "characters": [],
                        "relationships": [],
                        "terminology": [],
                        "style_guide": {},
                        "uncertainties": [],
                    },
                    "chapter_responses": {},
                }),
                encoding="utf-8",
            )
            fake_client = RecordingContextClient("test", "gemini-3.8-flash")
            logs = []
            with patch(
                "video_translator.gemini.movie_context_service.GeminiJsonClient",
                return_value=fake_client,
            ):
                MovieContextService(
                    "test",
                    "gemini-3.8-flash",
                    log_callback=logs.append,
                ).run(transcript_path, batch_segments=120)

            completed = json.loads(paths.context_json.read_text(encoding="utf-8"))
            self.assertEqual(
                fake_client.analyzed_ranges,
                [[361, 480], [481, 527]],
            )
            self.assertEqual(fake_client.calls, 3)
            self.assertTrue(completed["is_complete"])
            self.assertEqual(completed["processed_segments"], 527)
            self.assertIn(
                "Fact from first 360 segments",
                completed["profile"]["film_profile"]["continuity_facts"],
            )
            self.assertTrue(any("Giữ checkpoint 360/527" in line for line in logs))

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

    def test_translation_keeps_completed_batches_after_prompt_schema_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript_path = self._write_transcript(root)
            paths = MovieProjectPaths.from_transcript(transcript_path)
            paths.context_json.write_text(
                json.dumps({
                    "is_complete": True,
                    "profile": {"film_profile": {"genre": "Drama"}},
                }),
                encoding="utf-8",
            )
            source_fingerprint = sha256_file(transcript_path)
            context_fingerprint = sha256_file(paths.context_json)
            accepted_segment = {
                "id": 1,
                "start": 0.25,
                "end": 1.75,
                "speaker_id": "char_1",
                "translated_text": "Đừng đi!",
            }
            paths.translated_json.write_text(
                json.dumps({
                    "config_signature": "old-prompt-signature",
                    "source_fingerprint": source_fingerprint,
                    "context_fingerprint": context_fingerprint,
                    "batch_size": 10,
                    "model": "gemini-3.6-flash",
                    "segments": [accepted_segment],
                    "issues": [{"segment_id": 1, "severity": "low"}],
                    "pending_batches": {
                        "2-2": {"translation": [{"id": 2, "translated_text": "stale"}]}
                    },
                    "is_complete": False,
                }),
                encoding="utf-8",
            )

            migrated = FilmTranslationService._load_or_create_state(
                paths,
                signature="new-prompt-signature",
                source_fingerprint=source_fingerprint,
                context_fingerprint=context_fingerprint,
                batch_size=10,
                prompt_signature="new prompt schema",
                current_model="gemini-3.8-flash",
            )

            self.assertEqual(migrated["segments"], [accepted_segment])
            self.assertEqual(migrated["issues"], [{"segment_id": 1, "severity": "low"}])
            self.assertEqual(migrated["pending_batches"], {})
            self.assertEqual(migrated["config_signature"], "new-prompt-signature")
            self.assertEqual(migrated["model"], "gemini-3.8-flash")

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


    def test_translation_restores_source_timestamps_when_gemini_changes_them(self) -> None:
        source = [{"id": 51, "start": 12.345, "end": 13.678, "text": "Hello"}]
        result = [
            {
                "id": 51,
                "start": 12.35,
                "end": 13.7,
                "translated_text": "Xin chào",
            }
        ]

        validated = FilmTranslationService._validate_segments(source, result, "test")

        self.assertEqual(validated[0]["id"], 51)
        self.assertEqual(validated[0]["start"], 12.345)
        self.assertEqual(validated[0]["end"], 13.678)
        self.assertEqual(validated[0]["translated_text"], "Xin chào")

    def test_translation_does_not_require_gemini_to_return_timestamps(self) -> None:
        source = [{"id": 51, "start": 12.345, "end": 13.678, "text": "Hello"}]
        result = [{"id": 51, "translated_text": "Xin chào"}]

        validated = FilmTranslationService._validate_segments(source, result, "test")

        self.assertEqual(
            (validated[0]["start"], validated[0]["end"]),
            (12.345, 13.678),
        )

    def test_translation_matches_results_by_id_and_restores_source_order(self) -> None:
        source = [
            {"id": 51, "start": 1.0, "end": 2.0, "text": "Hello"},
            {"id": 52, "start": 2.0, "end": 3.0, "text": "Goodbye"},
        ]
        result = [
            {"id": 52, "translated_text": "Tạm biệt"},
            {"id": 51, "translated_text": "Xin chào"},
        ]

        validated = FilmTranslationService._validate_segments(source, result, "test")

        self.assertEqual([item["id"] for item in validated], [51, 52])
        self.assertEqual(
            [item["translated_text"] for item in validated],
            ["Xin chào", "Tạm biệt"],
        )

    def test_translation_allows_empty_text_and_skips_it_in_srt(self) -> None:
        source = [
            {"id": 247, "start": 10.0, "end": 11.0, "text": "Hello"},
            {"id": 248, "start": 11.0, "end": 12.0, "text": "Unclear audio"},
            {"id": 249, "start": 12.0, "end": 13.0, "text": "Goodbye"},
        ]
        response = [
            {"id": 247, "translated_text": "Xin chào"},
            {"id": 248, "translated_text": ""},
            {"id": 249, "translated_text": "Tạm biệt"},
        ]

        validated = FilmTranslationService._validate_segments(
            source, response, "test"
        )
        self.assertEqual([item["id"] for item in validated], [247, 248, 249])
        self.assertEqual(validated[1]["translated_text"], "")

        with tempfile.TemporaryDirectory() as temp_dir:
            srt_path = Path(temp_dir) / "skip.srt"
            FilmTranslationService._write_translated_srt(srt_path, validated)
            srt = srt_path.read_text(encoding="utf-8")
            self.assertIn("Xin chào", srt)
            self.assertIn("Tạm biệt", srt)
            self.assertNotIn("Unclear audio", srt)
            self.assertNotIn("00:00:11,000", srt)

    def test_translation_still_rejects_missing_or_duplicate_segment_ids(self) -> None:
        source = [
            {"id": 51, "start": 1.0, "end": 2.0, "text": "Hello"},
            {"id": 52, "start": 2.0, "end": 3.0, "text": "Goodbye"},
        ]
        duplicate = [
            {"id": 51, "translated_text": "Xin chào"},
            {"id": 51, "translated_text": "Chào bạn"},
        ]
        with self.assertRaisesRegex(FilmTranslationError, "lặp ID"):
            FilmTranslationService._validate_segments(source, duplicate, "test")


if __name__ == "__main__":
    unittest.main()
