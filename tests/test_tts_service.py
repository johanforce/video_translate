import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.tts.tts_service import VieNeuTTSService


class FakeVieNeuClient:
    calls = 0

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def synthesize_segment(
        self,
        text,
        voice_id,
        speed,
        segment_id,
        request_key,
        output_dir,
        cancel_event,
        log_callback=None,
        job_callback=None,
    ):
        type(self).calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / f"segment_{segment_id:06d}.wav"
        audio_path.write_bytes(b"RIFF" + b"x" * 200)
        if job_callback:
            job_callback(f"job-{segment_id}", {"status": "queued"})
        return audio_path, {
            "job_id": f"job-{segment_id}",
            "status": "completed",
            "audio_path": str(audio_path),
            "duration_seconds": 1.0,
            "voice_id": voice_id,
            "speed": speed,
            "request_key": request_key,
        }


class TTSServiceTests(unittest.TestCase):
    def test_completed_audio_is_cached_on_rerun(self) -> None:
        FakeVieNeuClient.calls = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcript_path = root / "movie.transcript.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "segments": [
                            {"id": 1, "start": 0, "end": 1, "text": "Hello"},
                            {"id": 2, "start": 1, "end": 2, "text": "Goodbye"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            paths = MovieProjectPaths.from_transcript(transcript_path)
            paths.translated_json.write_text(
                json.dumps(
                    {
                        "source_transcript": str(transcript_path),
                        "segments": [
                            {
                                "id": 1,
                                "start": 0,
                                "end": 1,
                                "translated_text": "Xin chào",
                            },
                            {
                                "id": 2,
                                "start": 1,
                                "end": 2,
                                "translated_text": "Tạm biệt",
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch(
                "video_translator.tts.tts_service.VieNeuClient",
                FakeVieNeuClient,
            ):
                service = VieNeuTTSService("test-key")
                manifest_path, completed, total = service.run(
                    paths.translated_json,
                    "Ngọc Lan",
                    speed=1.0,
                )
                self.assertEqual((completed, total), (2, 2))
                self.assertTrue(manifest_path.is_file())
                self.assertEqual(FakeVieNeuClient.calls, 2)

                service.run(paths.translated_json, "Ngọc Lan", speed=1.0)
                self.assertEqual(FakeVieNeuClient.calls, 2)

                # Migrate a legacy v0.5 manifest that predates translation_fingerprint.
                legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                legacy_manifest.pop("translation_fingerprint", None)
                manifest_path.write_text(
                    json.dumps(legacy_manifest),
                    encoding="utf-8",
                )
                service.run(paths.translated_json, "Ngọc Lan", speed=1.0)
                self.assertEqual(FakeVieNeuClient.calls, 2)

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertTrue(manifest["is_complete"])
                self.assertTrue(manifest["translation_fingerprint"])
                self.assertEqual(manifest["segments"]["1"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
