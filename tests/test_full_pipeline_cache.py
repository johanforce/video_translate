import json
import tempfile
import unittest
from pathlib import Path

from video_translator.core.pipeline_service import ProjectPipelineService
from video_translator.gemini.json_utils import sha256_file, sha256_text
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.gemini.prompt_loader import load_prompt


class FullPipelineCacheTests(unittest.TestCase):
    def test_pipeline_skips_all_matching_completed_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "film.mp4"
            video.write_bytes(b"source video bytes")
            transcript = root / "film.transcript.json"
            transcript.write_text(
                json.dumps({
                    "source_video": str(video),
                    "duration_seconds": 12.0,
                    "language": "en",
                    "model": "small",
                    "requested_language": "auto",
                    "segments": [{"id": 1, "start": 0.0, "end": 1.2, "text": "Hello"}],
                }),
                encoding="utf-8",
            )
            paths = MovieProjectPaths.from_transcript(transcript)
            context_signature = sha256_text(
                sha256_file(transcript)
                + "gemini-test"
                + "180"
                + load_prompt("system_prompt.txt")
                + load_prompt("context_analysis.txt")
                + load_prompt("context_finalize.txt")
            )
            paths.context_json.write_text(
                json.dumps({
                    "is_complete": True,
                    "transcript_fingerprint": sha256_file(transcript),
                    "config_signature": context_signature,
                    "model": "gemini-test",
                    "batch_segments": 180,
                    "processed_segments": 1,
                    "total_segments": 1,
                    "profile": {"film_profile": {"genre": "Drama"}},
                }),
                encoding="utf-8",
            )
            translation_signature = sha256_text(
                sha256_file(transcript)
                + sha256_file(paths.context_json)
                + "gemini-test"
                + "30"
                + load_prompt("system_prompt.txt")
                + load_prompt("translation.txt")
                + load_prompt("quality_control.txt")
            )
            paths.translated_json.write_text(
                json.dumps({
                    "is_complete": True,
                    "source_transcript": str(transcript),
                    "source_fingerprint": sha256_file(transcript),
                    "context_fingerprint": sha256_file(paths.context_json),
                    "config_signature": translation_signature,
                    "model": "gemini-test",
                    "batch_size": 30,
                    "completed_segments": 1,
                    "total_segments": 1,
                    "segments": [{
                        "id": 1, "start": 0.0, "end": 1.2,
                        "original_text": "Hello", "translated_text": "Xin chào"
                    }],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            paths.translated_srt.write_text(
                "1\n00:00:00,000 --> 00:00:01,200\nXin chào\n",
                encoding="utf-8",
            )
            paths.tts_dir.mkdir(parents=True, exist_ok=True)
            audio = paths.tts_dir / "segment_000001.wav"
            audio.write_bytes(b"RIFF" + b"a" * 256)
            paths.tts_manifest.write_text(
                json.dumps({
                    "is_complete": True,
                    "translation_fingerprint": sha256_text(
                        paths.translated_json.read_text(encoding="utf-8")
                    ),
                    "voice_id": "voice-test",
                    "speed": 1.0,
                    "segments": {
                        "1": {"status": "completed", "audio_path": str(audio)}
                    },
                }),
                encoding="utf-8",
            )
            output = root / "film.vi.dubbed.mp4"
            output.write_bytes(b"final video" * 256)
            output.with_suffix(".render.json").write_text(
                json.dumps({
                    "output_signature": sha256_file(output),
                    "translation_fingerprint": sha256_file(paths.translated_json),
                    "source_video_fingerprint": sha256_file(video),
                    "tts_manifest_fingerprint": sha256_file(paths.tts_manifest),
                    "subtitle_fingerprint": sha256_file(paths.translated_srt),
                    "subtitle_mode": "both",
                    "original_audio_mode": "duck",
                }),
                encoding="utf-8",
            )

            result = ProjectPipelineService().run(
                output_dir=root,
                video_path=video,
                transcript_path=transcript,
                whisper_model="small",
                source_language=None,
                gemini_api_key="",
                gemini_model="gemini-test",
                translation_batch_size=30,
                vieneu_api_key="",
                voice_id="voice-test",
                voice_speed=1.0,
                subtitle_mode="both",
                original_audio_mode="duck",
            )

            self.assertEqual(result["video"], output)
            self.assertEqual(result["transcript"], transcript)
            self.assertEqual(result["translation"], paths.translated_json)
            self.assertTrue(result["project"].is_file())


if __name__ == "__main__":
    unittest.main()
