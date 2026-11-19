import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from video_translator.tts.vieneu_client import LOCAL_ENGINE, VieNeuClient, VieNeuError


class VieNeuClientTests(unittest.TestCase):
    def test_lists_local_sdk_preset_voices_without_loading_model(self) -> None:
        payload = {
            "default_voice": "Hải Đăng",
            "presets": {
                "Trúc Ly": {"description": "Female voice", "featured": 2, "gender": "female"},
                "Hải Đăng": {"description": "Male voice", "featured": 1, "gender": "male"},
            },
        }

        class Resource:
            def open(self, *_args, **_kwargs):
                return io.StringIO(json.dumps(payload))

        class PackageResources:
            def joinpath(self, *_args):
                return Resource()

        with patch(
            "video_translator.tts.vieneu_client.importlib.resources.files",
            return_value=PackageResources(),
        ):
            voices = VieNeuClient.list_voices()

        self.assertEqual([voice["id"] for voice in voices], ["Hải Đăng", "Trúc Ly"])
        self.assertEqual(voices[0]["engine"], LOCAL_ENGINE)
        self.assertEqual(voices[0]["kind"], "preset")

    def test_missing_sdk_shows_install_command(self) -> None:
        with patch(
            "video_translator.tts.vieneu_client.importlib.resources.files",
            side_effect=ModuleNotFoundError("vieneu"),
        ):
            with self.assertRaisesRegex(VieNeuError, "pip install vieneu"):
                VieNeuClient.list_voices()

    def test_synthesis_uses_local_sdk_and_never_needs_api_key(self) -> None:
        class FakeEngine:
            sample_rate = 48_000

            def __init__(self):
                self.infer_args = None
                self.closed = False

            def infer(self, text, voice):
                self.infer_args = (text, voice)
                return [0.0, 0.1]

            def save(self, audio, path):
                Path(path).write_bytes(b"RIFF" + b"x" * 256)

            def close(self):
                self.closed = True

        engine = FakeEngine()
        module = types.ModuleType("vieneu")
        module.Vieneu = lambda **kwargs: self._make_engine(engine, kwargs)

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules, {"vieneu": module}
        ):
            client = VieNeuClient()
            path, metadata = client.synthesize_segment(
                text="Xin chào!",
                voice_id="Hải Đăng",
                speed=1.0,
                segment_id=7,
                request_key="abcdef123456",
                output_dir=Path(temp_dir),
                cancel_event=__import__("threading").Event(),
            )

            self.assertTrue(path.is_file())
            self.assertEqual(engine.infer_args, ("Xin chào!", "Hải Đăng"))
            self.assertEqual(metadata["engine"], "v3turbo")
            self.assertNotIn("job_id", metadata)
            client.close()
            self.assertTrue(engine.closed)

    @staticmethod
    def _make_engine(engine, kwargs):
        if kwargs != {"mode": "v3turbo"}:
            raise AssertionError(f"Expected local v3 Turbo mode, got {kwargs}")
        return engine

    def test_time_stretch_uses_pitch_preserving_atempo_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.wav"
            destination = Path(temp_dir) / "destination.wav"
            source.write_bytes(b"RIFF" + b"x" * 256)

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"RIFF" + b"y" * 256)
                return types.SimpleNamespace(returncode=0, stderr="")

            with (
                patch("video_translator.tts.vieneu_client.shutil.which", return_value="ffmpeg"),
                patch("video_translator.tts.vieneu_client.subprocess.run", side_effect=fake_run) as run,
            ):
                VieNeuClient._time_stretch(source, destination, 3.0)

            self.assertTrue(destination.is_file())
            command = run.call_args.args[0]
            self.assertIn("atempo=2.0,atempo=1.500000", command)
            self.assertIn("pcm_s16le", command)


if __name__ == "__main__":
    unittest.main()
