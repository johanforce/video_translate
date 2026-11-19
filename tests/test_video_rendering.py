import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from video_translator.gemini.json_utils import sha256_text
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.rendering.audio_timeline import AudioTimelineBuilder
from video_translator.rendering.media_tools import probe_media
from video_translator.rendering.video_renderer import VideoRenderer


class SubtitleStyleTests(unittest.TestCase):
    def test_hard_subtitles_use_white_text_on_solid_black_background(self) -> None:
        subtitle_filter = VideoRenderer._subtitle_filter(Path("subtitle.srt"))

        self.assertIn("PrimaryColour=&H00FFFFFF", subtitle_filter)
        self.assertIn("BorderStyle=3", subtitle_filter)
        self.assertIn("BackColour=&H00000000", subtitle_filter)
        self.assertNotIn("BorderStyle=1", subtitle_filter)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
class VideoRenderingTests(unittest.TestCase):
    def test_render_mp4_with_soft_subtitle_and_reuse_render_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "film.mp4"
            audio_path = root / "tts.wav"

            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                    "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=24000:duration=0.6",
                    "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
                    str(audio_path),
                ],
                check=True,
                capture_output=True,
            )

            transcript_path = root / "film.transcript.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "source_video": str(video_path),
                        "duration_seconds": 2.0,
                        "language": "en",
                        "segments": [
                            {"id": 1, "start": 0.2, "end": 1.0, "text": "Hello."}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            paths = MovieProjectPaths.from_transcript(transcript_path)
            translated_payload = {
                "is_complete": True,
                "source_transcript": str(transcript_path),
                "segments": [
                    {
                        "id": 1,
                        "start": 0.2,
                        "end": 1.0,
                        "original_text": "Hello.",
                        "translated_text": "Xin chào.",
                        "emotion": "happy",
                        "delivery": "normal",
                    }
                ],
            }
            paths.translated_json.write_text(
                json.dumps(translated_payload, ensure_ascii=False),
                encoding="utf-8",
            )
            paths.translated_srt.write_text(
                "1\n00:00:00,200 --> 00:00:01,000\nXin chào.\n",
                encoding="utf-8",
            )
            paths.tts_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "is_complete": True,
                "translation_fingerprint": sha256_text(
                    paths.translated_json.read_text(encoding="utf-8")
                ),
                "voice_id": "test-voice",
                "speed": 1.0,
                "engine": "v4",
                "segments": {
                    "1": {
                        "status": "completed",
                        "audio_path": str(audio_path),
                        "request_signature": "test-signature",
                    }
                },
            }
            paths.tts_manifest.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            renderer = VideoRenderer()
            output_path, metadata_path, cached = renderer.render(
                paths.translated_json,
                subtitle_mode="soft",
                original_audio_mode="duck",
            )
            self.assertFalse(cached)
            self.assertTrue(output_path.is_file())
            self.assertTrue(metadata_path.is_file())
            self.assertGreater(output_path.stat().st_size, 1024)

            output_info = probe_media(output_path)
            stream_types = {
                item.get("codec_type")
                for item in output_info.get("streams", [])
                if isinstance(item, dict)
            }
            self.assertIn("video", stream_types)
            self.assertIn("audio", stream_types)
            self.assertIn("subtitle", stream_types)

            second_output, second_metadata, second_cached = renderer.render(
                paths.translated_json,
                subtitle_mode="soft",
                original_audio_mode="duck",
            )
            self.assertEqual(second_output, output_path)
            self.assertEqual(second_metadata, metadata_path)
            self.assertTrue(second_cached)

            # Timeline cache is independent from final MP4 cache and must be safe
            # to reuse when a new render configuration is selected.
            cached_voice, cached_timeline = AudioTimelineBuilder().build(
                paths.translated_json
            )
            self.assertTrue(cached_voice.is_file())
            self.assertTrue(cached_timeline.is_file())


if __name__ == "__main__":
    unittest.main()
