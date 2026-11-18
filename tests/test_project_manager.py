import json
import tempfile
import unittest
from pathlib import Path

from video_translator.core.project_manager import ProjectManager
from video_translator.gemini.json_utils import sha256_file, sha256_text
from video_translator.gemini.project_paths import MovieProjectPaths


class ProjectManagerTests(unittest.TestCase):
    def test_project_reconciles_completed_stages_and_detects_stale_tts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "film.mp4"
            video.write_bytes(b"fake-video")
            transcript_path = root / "film.transcript.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "source_video": str(video),
                        "duration_seconds": 10.0,
                        "segments": [{"id": 1, "start": 0.0, "end": 1.0, "text": "Hello"}],
                    }
                ),
                encoding="utf-8",
            )
            paths = MovieProjectPaths.from_transcript(transcript_path)
            paths.context_json.write_text(
                json.dumps(
                    {
                        "is_complete": True,
                        "transcript_fingerprint": sha256_file(transcript_path),
                        "profile": {},
                    }
                ),
                encoding="utf-8",
            )
            translated = {
                "is_complete": True,
                "source_fingerprint": sha256_file(transcript_path),
                "context_fingerprint": sha256_file(paths.context_json),
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
                json.dumps(translated, ensure_ascii=False),
                encoding="utf-8",
            )
            paths.translated_srt.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nXin chào\n",
                encoding="utf-8",
            )
            paths.tts_dir.mkdir(parents=True, exist_ok=True)
            audio_path = paths.tts_dir / "segment_000001.wav"
            audio_path.write_bytes(b"RIFF" + b"x" * 256)
            paths.tts_manifest.write_text(
                json.dumps(
                    {
                        "is_complete": True,
                        "translation_fingerprint": sha256_text(
                            paths.translated_json.read_text(encoding="utf-8")
                        ),
                        "segments": {
                            "1": {"status": "completed", "audio_path": str(audio_path)}
                        },
                    }
                ),
                encoding="utf-8",
            )

            manager = ProjectManager(transcript_path)
            self.assertEqual(manager.state["stages"]["context"]["status"], "completed")
            self.assertEqual(manager.state["stages"]["translation"]["status"], "completed")
            self.assertEqual(manager.state["stages"]["tts"]["status"], "completed")
            self.assertEqual(manager.next_incomplete_stage(), "render")
            self.assertTrue(manager.project_file.is_file())

            translated["segments"][0]["translated_text"] = "Xin chào mọi người"
            paths.translated_json.write_text(
                json.dumps(translated, ensure_ascii=False),
                encoding="utf-8",
            )
            manager.refresh()
            self.assertEqual(manager.state["stages"]["tts"]["status"], "not_started")
            self.assertEqual(manager.next_incomplete_stage(), "tts")

    def test_running_stage_becomes_interrupted_after_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript = root / "film.transcript.json"
            transcript.write_text(json.dumps({"segments": []}), encoding="utf-8")
            manager = ProjectManager(transcript)
            manager.start_stage("context")

            reopened = ProjectManager(transcript)
            self.assertEqual(
                reopened.state["stages"]["context"]["status"],
                "interrupted",
            )


if __name__ == "__main__":
    unittest.main()
