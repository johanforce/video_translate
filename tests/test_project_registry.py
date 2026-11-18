import json
import tempfile
import unittest
from pathlib import Path

from video_translator.core.project_registry import ProjectRegistry


class ProjectRegistryTests(unittest.TestCase):
    def test_registered_video_reopens_project_after_registry_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "source.mp4"
            video.write_bytes(b"video")
            transcript = root / "project" / "source.transcript.json"
            transcript.parent.mkdir()
            transcript.write_text(
                json.dumps({"source_video": str(video.resolve()), "segments": []}),
                encoding="utf-8",
            )
            registry_path = root / "user-data" / "projects.json"

            ProjectRegistry(registry_path).register(transcript)
            reopened = ProjectRegistry(registry_path)

            self.assertEqual(
                reopened.find_for_video(video, (root / "empty",)),
                transcript.resolve(),
            )

    def test_finds_existing_project_in_output_folder_without_registry_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "source.mp4"
            video.write_bytes(b"video")
            project_dir = root / "output"
            project_dir.mkdir()
            transcript = project_dir / "source.transcript.json"
            transcript.write_text(
                json.dumps({"source_video": str(video.resolve()), "segments": []}),
                encoding="utf-8",
            )

            found = ProjectRegistry(root / "missing-registry.json").find_for_video(
                video,
                (project_dir,),
            )

            self.assertEqual(found, transcript.resolve())

    def test_finds_legacy_transcript_without_source_video_using_matching_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "source.mp4"
            video.write_bytes(b"video")
            transcript = root / "source.transcript.json"
            transcript.write_text(json.dumps({"segments": []}), encoding="utf-8")

            found = ProjectRegistry(root / "missing-registry.json").find_for_video(
                video,
                (root,),
            )

            self.assertEqual(found, transcript.resolve())

    def test_does_not_match_project_for_another_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "first.mp4"
            other_video = root / "second.mp4"
            video.write_bytes(b"first")
            other_video.write_bytes(b"second")
            transcript = root / "second.transcript.json"
            transcript.write_text(
                json.dumps({"source_video": str(other_video.resolve()), "segments": []}),
                encoding="utf-8",
            )

            found = ProjectRegistry(root / "missing-registry.json").find_for_video(
                video,
                (root,),
            )

            self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
