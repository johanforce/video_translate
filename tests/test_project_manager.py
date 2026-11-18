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

    def test_partial_transcript_is_resumable_not_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_video = root / "film.mp4"
            source_video.write_bytes(b"video")
            transcript = root / "film.transcript.json"
            checkpoint = root / ".video_translator_cache" / "film" / "transcription_state.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(
                json.dumps(
                    {
                        "total_chunks": 4,
                        "completed_chunks": 2,
                        "is_complete": False,
                    }
                ),
                encoding="utf-8",
            )
            transcript.write_text(
                json.dumps(
                    {
                        "source_video": str(source_video),
                        "is_complete": False,
                        "checkpoint_path": str(checkpoint),
                        "segments": [
                            {"id": 1, "start": 0.0, "end": 1.0, "text": "Hello"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manager = ProjectManager(transcript)
            self.assertEqual(
                manager.state["stages"]["transcription"]["status"],
                "not_started",
            )
            self.assertEqual(
                manager.state["stages"]["transcription"]["progress"],
                50,
            )
            self.assertEqual(manager.next_incomplete_stage(), "transcription")

    def test_stage_settings_are_persisted_before_work_starts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript = root / "film.transcript.json"
            transcript.write_text(json.dumps({"segments": []}), encoding="utf-8")
            manager = ProjectManager(transcript)
            manager.start_stage(
                "tts",
                voice_id="voice-123",
                speed=1.2,
            )

            reopened = ProjectManager(transcript)
            stage = reopened.state["stages"]["tts"]
            self.assertEqual(stage["status"], "interrupted")
            self.assertEqual(stage["voice_id"], "voice-123")
            self.assertEqual(stage["speed"], 1.2)

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
