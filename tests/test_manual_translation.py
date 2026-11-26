import json
import tempfile
import unittest
from pathlib import Path

from video_translator.core.pipeline_service import ProjectPipelineService
from video_translator.core.project_manager import ProjectManager
from video_translator.gemini.json_utils import read_json, sha256_file
from video_translator.gemini.manual_translation import (
    ManualTranslationError,
    create_manual_translation_template,
    import_manual_translation,
)
from video_translator.gemini.project_paths import MovieProjectPaths


class ManualTranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.transcript_path = self.root / "film.transcript.json"
        self.transcript_path.write_text(
            json.dumps(
                {
                    "is_complete": True,
                    "segments": [
                        {"id": 1, "start": 0.0, "end": 1.2, "speaker_id": "S1", "text": "Hello there."},
                        {"id": 2, "start": 1.4, "end": 2.1, "speaker_id": "S2", "text": "Are you ready?"},
                        {"id": 3, "start": 2.3, "end": 2.5, "speaker_id": None, "text": ""},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.paths = MovieProjectPaths.from_transcript(self.transcript_path)
        self.template_path = self.root / "film.manual-translation.json"
        create_manual_translation_template(self.transcript_path, self.template_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _complete_template(self) -> dict:
        payload = read_json(self.template_path)
        translations = {
            1: ("Chào anh.", "neutral", "normal"),
            2: ("Anh sẵn sàng chưa?", "curious", "soft"),
            3: ("", "neutral", "normal"),
        }
        for row in payload["segments"]:
            translated, emotion, delivery = translations[row["id"]]
            row["translated_text"] = translated
            row["emotion"] = emotion
            row["delivery"] = delivery
            row["reviewed"] = True
        payload["quality_control"]["completed"] = True
        payload["quality_control"]["issues"] = []
        return payload

    def test_template_exports_source_identity_and_empty_translation_fields(self) -> None:
        payload = read_json(self.template_path)
        self.assertEqual(payload["source_fingerprint"], sha256_file(self.transcript_path))
        self.assertEqual([row["id"] for row in payload["segments"]], [1, 2, 3])
        self.assertTrue(all(row["translated_text"] == "" for row in payload["segments"]))
        self.assertTrue(all(row["reviewed"] is False for row in payload["segments"]))

    def test_import_creates_native_artifacts_without_gemini_context(self) -> None:
        payload = self._complete_template()
        self.template_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        json_path, srt_path, quality_path, count = import_manual_translation(
            self.transcript_path,
            self.template_path,
        )
        self.assertEqual(count, 3)
        self.assertEqual(json_path, self.paths.translated_json)
        self.assertTrue(srt_path.is_file())
        self.assertTrue(quality_path.is_file())
        self.assertFalse(self.paths.context_json.exists())

        translation = read_json(json_path)
        self.assertEqual(translation["translation_source"], "manual")
        self.assertTrue(translation["is_complete"])
        self.assertEqual(translation["segments"][1]["speaker_id"], "S2")
        self.assertEqual(translation["segments"][1]["start"], 1.4)
        self.assertEqual(translation["segments"][1]["emotion"], "curious")
        self.assertEqual(
            srt_path.read_text(encoding="utf-8"),
            "1\n00:00:00,000 --> 00:00:01,200\nChào anh.\n\n"
            "2\n00:00:01,400 --> 00:00:02,100\nAnh sẵn sàng chưa?\n",
        )

        manager = ProjectManager(self.transcript_path)
        self.assertEqual(manager.state["stages"]["context"]["status"], "not_started")
        self.assertEqual(manager.state["stages"]["translation"]["status"], "completed")
        self.assertEqual(manager.next_incomplete_stage(), "tts")

        self.assertTrue(
            ProjectPipelineService._is_stage_completed(
                manager, "context", "gemini-test", "small", None, "", 1.0,
                30, "both", "duck",
            )
        )
        self.assertTrue(
            ProjectPipelineService._is_stage_completed(
                manager, "translation", "gemini-test", "small", None, "", 1.0,
                30, "both", "duck",
            )
        )

    def test_rejects_template_from_a_different_transcript(self) -> None:
        payload = self._complete_template()
        payload["source_fingerprint"] = "0" * 64
        self.template_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ManualTranslationError, "transcript khác"):
            import_manual_translation(self.transcript_path, self.template_path)
        self.assertFalse(self.paths.translated_json.exists())

    def test_rejects_missing_translation_before_writing_artifacts(self) -> None:
        payload = self._complete_template()
        payload["segments"][0]["translated_text"] = ""
        self.template_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ManualTranslationError, "chưa có bản dịch"):
            import_manual_translation(self.transcript_path, self.template_path)
        self.assertFalse(self.paths.translated_json.exists())

    def test_rejects_unreviewed_segment(self) -> None:
        payload = self._complete_template()
        payload["segments"][1]["reviewed"] = False
        self.template_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ManualTranslationError, "chưa được đánh dấu"):
            import_manual_translation(self.transcript_path, self.template_path)
        self.assertFalse(self.paths.translated_json.exists())

    def test_rejects_incomplete_final_quality_review(self) -> None:
        payload = self._complete_template()
        payload["quality_control"]["completed"] = False
        self.template_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ManualTranslationError, "quality_control.completed"):
            import_manual_translation(self.transcript_path, self.template_path)
        self.assertFalse(self.paths.translated_json.exists())

    def test_rejects_duplicate_or_modified_source_segment(self) -> None:
        payload = self._complete_template()
        payload["segments"][1]["id"] = payload["segments"][0]["id"]
        self.template_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ManualTranslationError, "trùng segment ID"):
            import_manual_translation(self.transcript_path, self.template_path)

        payload = self._complete_template()
        payload["segments"][0]["source_text"] = "Changed source"
        self.template_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ManualTranslationError, "source_text"):
            import_manual_translation(self.transcript_path, self.template_path)


if __name__ == "__main__":
    unittest.main()
