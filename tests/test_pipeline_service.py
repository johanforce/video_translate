import json
import tempfile
import unittest
from pathlib import Path

from video_translator.core.pipeline_service import ProjectPipelineService
from video_translator.core.project_manager import ProjectManager
from video_translator.gemini.json_utils import sha256_file
from video_translator.gemini.project_paths import MovieProjectPaths


class PipelineStageCacheTests(unittest.TestCase):
    def _prepare_project(self, root: Path) -> tuple[MovieProjectPaths, ProjectManager]:
        video = root / "film.mp4"
        video.write_bytes(b"source-video")
        transcript_path = root / "film.transcript.json"
        transcript_path.write_text(
            json.dumps(
                {
                    "source_video": str(video),
                    "is_complete": True,
                    "model": "small",
                    "requested_language": "auto",
                    "segments": [
                        {"id": 1, "start": 0.0, "end": 1.0, "text": "Hello"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths = MovieProjectPaths.from_transcript(transcript_path)
        context = {
            "is_complete": True,
            "transcript_fingerprint": sha256_file(transcript_path),
            "model": "gemini-test",
            "batch_segments": 180,
            "profile": {"film_profile": {"genre": "Drama"}},
        }
        paths.context_json.write_text(json.dumps(context), encoding="utf-8")
        translation = {
            "is_complete": True,
            "source_transcript": str(transcript_path),
            "source_fingerprint": sha256_file(transcript_path),
            "context_fingerprint": sha256_file(paths.context_json),
            "model": "gemini-test",
            "batch_size": 30,
            "segments": [
                {
                    "id": 1,
                    "start": 0.0,
                    "end": 1.0,
                    "translated_text": "Xin chào",
                }
            ],
        }
        paths.translated_json.write_text(
            json.dumps(translation, ensure_ascii=False),
            encoding="utf-8",
        )
        paths.translated_srt.write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nXin chào\n",
            encoding="utf-8",
        )
        return paths, ProjectManager(transcript_path)

    def test_matching_completed_gemini_stages_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths, manager = self._prepare_project(Path(temp_dir))

            self.assertTrue(
                ProjectPipelineService._is_stage_completed(
                    manager,
                    "context",
                    "gemini-test",
                    "small",
                    None,
                    "",
                    1.0,
                    30,
                    "both",
                    "duck",
                )
            )
            self.assertTrue(
                ProjectPipelineService._is_stage_completed(
                    manager,
                    "translation",
                    "gemini-test",
                    "small",
                    None,
                    "",
                    1.0,
                    30,
                    "both",
                    "duck",
                )
            )

    def test_changed_batch_configuration_invalidates_only_dependent_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths, manager = self._prepare_project(Path(temp_dir))

            self.assertFalse(
                ProjectPipelineService._is_stage_completed(
                    manager,
                    "context",
                    "gemini-test",
                    "small",
                    None,
                    "",
                    1.0,
                    40,
                    "both",
                    "duck",
                )
            )
            self.assertFalse(
                ProjectPipelineService._is_stage_completed(
                    manager,
                    "translation",
                    "gemini-test",
                    "small",
                    None,
                    "",
                    1.0,
                    40,
                    "both",
                    "duck",
                )
            )

    def test_completed_artifacts_reconcile_after_project_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _, manager = self._prepare_project(Path(temp_dir))
            manager.start_stage("translation")
            reopened = ProjectManager(manager.paths.transcript_json)

            self.assertEqual(
                reopened.state["stages"]["translation"]["status"],
                "completed",
            )
            # Durable translation artifacts reconcile the stage to completed, but
            # the cache signature still gates reuse at the pipeline layer.
            self.assertTrue(
                ProjectPipelineService._is_stage_completed(
                    reopened,
                    "translation",
                    "gemini-test",
                    "small",
                    None,
                    "",
                    1.0,
                    30,
                    "both",
                    "duck",
                )
            )


if __name__ == "__main__":
    unittest.main()
