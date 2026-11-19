import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.tts.tts_service import VieNeuTTSService


class FakeVieNeuClient:
    calls = 0

    def __init__(self) -> None:
        pass

    def close(self) -> None:
        pass

    @staticmethod
    def resolve_voice_id(requested_voice="", fallback_voice="Hải Đăng") -> str:
        return requested_voice or fallback_voice

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
    ):
        type(self).calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / f"segment_{segment_id:06d}.wav"
        audio_path.write_bytes(b"RIFF" + b"x" * 200)
        return audio_path, {
            "status": "completed",
            "engine": "v3turbo",
            "audio_path": str(audio_path),
            "duration_seconds": 1.0,
            "voice_id": voice_id,
            "speed": speed,
            "request_key": request_key,
        }


class TTSServiceTests(unittest.TestCase):
    def test_vieneu_v3_maps_laughter_to_sdk_inline_cue(self) -> None:
        self.assertEqual(
            VieNeuTTSService._apply_v3_cues(
                "Tôi không tin nổi!",
                {"delivery": "laughing", "emotion": "humorous"},
            ),
            "[cười] Tôi không tin nổi!",
        )
        self.assertEqual(
            VieNeuTTSService._apply_v3_cues(
                "Tôi không tin nổi!",
                {"delivery": "shouting", "emotion": "angry"},
            ),
            "Tôi không tin nổi!",
        )

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
                service = VieNeuTTSService()
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
