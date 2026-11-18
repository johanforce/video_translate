from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from video_translator.core.errors import VideoTranslatorError
from video_translator.core.models import TranscriptionProgress
from video_translator.gemini.json_utils import (
    read_json,
    sha256_file,
    sha256_text,
    write_json_atomic,
)
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.rendering.audio_timeline import AudioTimelineBuilder
from video_translator.rendering.media_tools import (
    media_duration,
    probe_media,
    run_ffmpeg,
)


ProgressCallback = Callable[[TranscriptionProgress], None]
LogCallback = Callable[[str], None]


class VideoRenderError(VideoTranslatorError):
    """Raised when the final Vietnamese MP4 cannot be rendered."""


class VideoRenderer:
    """Mix source ambience with narration and mux/burn Vietnamese subtitles."""

    SUBTITLE_MODES = {"hard", "soft", "both", "none"}
    AUDIO_MODES = {"duck", "mix", "mute"}

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.log_callback = log_callback

    def render(
        self,
        translated_json: Path,
        subtitle_mode: str = "both",
        original_audio_mode: str = "duck",
        output_path: Path | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[Path, Path, bool]:
        cancel_event = cancel_event or threading.Event()
        translation_path = Path(translated_json).expanduser().resolve()
        if subtitle_mode not in self.SUBTITLE_MODES:
            raise VideoRenderError(f"Chế độ phụ đề không hợp lệ: {subtitle_mode}")
        if original_audio_mode not in self.AUDIO_MODES:
            raise VideoRenderError(f"Chế độ audio gốc không hợp lệ: {original_audio_mode}")

        translation = read_json(translation_path)
        if translation.get("is_complete") is not True:
            raise VideoRenderError("Bản dịch chưa hoàn tất; hãy hoàn thành Gemini trước.")
        transcript_text = translation.get("source_transcript")
        if not transcript_text:
            raise VideoRenderError("Bản dịch thiếu đường dẫn transcript gốc.")
        paths = MovieProjectPaths.from_transcript(Path(transcript_text))
        transcript = read_json(paths.transcript_json)
        source_video = Path(str(transcript.get("source_video", ""))).expanduser()
        if not source_video.is_file():
            raise VideoRenderError(f"Không tìm thấy video nguồn: {source_video}")
        subtitle_path = paths.translated_srt
        if subtitle_mode != "none" and not subtitle_path.is_file():
            raise VideoRenderError(f"Không tìm thấy phụ đề tiếng Việt: {subtitle_path}")
        manifest_path = paths.tts_manifest
        if not manifest_path.is_file():
            raise VideoRenderError(f"Không tìm thấy VieNeu manifest: {manifest_path}")
        tts_state = read_json(manifest_path)
        if tts_state.get("is_complete") is not True:
            raise VideoRenderError(
                "VieNeu chưa hoàn thành. Hãy chạy lại TTS; audio đã có sẽ được lấy từ cache."
            )

        output = (
            Path(output_path).expanduser().resolve()
            if output_path
            else source_video.parent / f"{source_video.stem}.vi.dubbed.mp4"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = output.with_suffix(".render.json")
        source_info = probe_media(source_video)
        duration = media_duration(source_video)
        has_source_audio = any(
            stream.get("codec_type") == "audio"
            for stream in source_info.get("streams", [])
            if isinstance(stream, dict)
        )

        source_video_fingerprint = sha256_file(source_video)
        translation_fingerprint = sha256_file(translation_path)
        tts_manifest_fingerprint = sha256_file(manifest_path)
        subtitle_fingerprint = sha256_file(subtitle_path) if subtitle_path.is_file() else ""
        signature = sha256_text(
            "|".join(
                (
                    source_video_fingerprint,
                    translation_fingerprint,
                    tts_manifest_fingerprint,
                    subtitle_fingerprint,
                    subtitle_mode,
                    original_audio_mode,
                    "h264-crf18-aac192k",
                )
            )
        )
        if output.is_file() and output.stat().st_size > 1024 and metadata_path.is_file():
            try:
                saved = read_json(metadata_path)
                if (
                    saved.get("render_signature") == signature
                    and saved.get("output_signature") == sha256_file(output)
                ):
                    self._emit(100, "Video MP4 đã render; dùng lại cache.")
                    self._log(f"Đã dùng lại video render: {output}")
                    return output, metadata_path, True
            except (ValueError, OSError):
                pass

        self._log("Đang xây dựng timeline giọng thuyết minh...")
        voice_track, timeline_path = AudioTimelineBuilder(
            progress_callback=lambda item: self._emit(
                min(75, int(item.percent * 0.75)), item.message, item.segments_done
            ),
            log_callback=self.log_callback,
        ).build(
            translated_json=translation_path,
            tts_manifest=manifest_path,
            cancel_event=cancel_event,
        )
        if cancel_event.is_set():
            raise VideoRenderError("Đã hủy render video.")

        temporary = output.with_name(output.stem + ".partial.mp4")
        temporary.unlink(missing_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(source_video),
            "-i", str(voice_track),
        ]

        subtitle_input_index: int | None = None
        if subtitle_mode in {"soft", "both"}:
            command.extend(["-i", str(subtitle_path)])
            subtitle_input_index = 2

        if has_source_audio and original_audio_mode != "mute":
            background_volume = 0.18 if original_audio_mode == "duck" else 1.0
            filter_complex = (
                f"[0:a:0]volume={background_volume:.3f}[background];"
                "[1:a:0]volume=1.0[narration];"
                "[background][narration]"
                "amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
        else:
            filter_complex = "[1:a:0]anull[aout]"

        command.extend(["-filter_complex", filter_complex])
        command.extend(["-map", "0:v:0", "-map", "[aout]"])

        if subtitle_input_index is not None:
            command.extend(["-map", f"{subtitle_input_index}:0"])

        if subtitle_mode in {"hard", "both"}:
            subtitle_filter = self._subtitle_filter(subtitle_path)
            command.extend(["-vf", subtitle_filter])
            command.extend(
                [
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                ]
            )
        else:
            command.extend(["-c:v", "copy"])

        command.extend(["-c:a", "aac", "-b:a", "192k"])
        if subtitle_input_index is not None:
            command.extend(["-c:s", "mov_text"])
        command.extend(
            [
                "-map_metadata", "0",
                "-movflags", "+faststart",
                "-t", f"{duration:.3f}",
                "-progress", "pipe:1",
                "-nostats",
                str(temporary),
            ]
        )

        self._log(
            f"FFmpeg render: subtitle={subtitle_mode}, original_audio={original_audio_mode}."
        )
        try:
            run_ffmpeg(
                command,
                duration_seconds=duration,
                cancel_event=cancel_event,
                progress_callback=lambda percent, message: self._emit(
                    75 + int(percent * 0.24), message
                ),
                log_callback=self.log_callback,
            )
            if not temporary.is_file() or temporary.stat().st_size < 1024:
                raise VideoRenderError("FFmpeg không tạo được MP4 đầu ra hợp lệ.")

            rendered_info = probe_media(temporary)
            rendered_streams = rendered_info.get("streams", [])
            if not any(
                isinstance(stream, dict) and stream.get("codec_type") == "video"
                for stream in rendered_streams
            ):
                raise VideoRenderError("Video MP4 đầu ra không có video stream.")
            if not any(
                isinstance(stream, dict) and stream.get("codec_type") == "audio"
                for stream in rendered_streams
            ):
                raise VideoRenderError("Video MP4 đầu ra không có audio thuyết minh.")
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        rendered_duration = media_duration(output)
        metadata = {
            "schema_version": 1,
            "render_signature": signature,
            "output_signature": sha256_file(output),
            "source_video_fingerprint": source_video_fingerprint,
            "translation_fingerprint": translation_fingerprint,
            "tts_manifest_fingerprint": tts_manifest_fingerprint,
            "subtitle_fingerprint": subtitle_fingerprint,
            "source_video": str(source_video),
            "translated_json": str(translation_path),
            "translated_srt": str(subtitle_path),
            "tts_manifest": str(manifest_path),
            "timeline": str(timeline_path),
            "output": str(output),
            "duration_seconds": rendered_duration,
            "subtitle_mode": subtitle_mode,
            "original_audio_mode": original_audio_mode,
            "video_codec": "h264" if subtitle_mode in {"hard", "both"} else "copied",
            "audio_codec": "aac",
        }
        write_json_atomic(metadata_path, metadata)
        self._emit(100, "Video thuyết minh tiếng Việt hoàn tất.")
        self._log(f"MP4: {output}")
        self._log(f"Render metadata: {metadata_path}")
        return output, metadata_path, False

    @staticmethod
    def _subtitle_filter(path: Path) -> str:
        escaped = path.resolve().as_posix()
        escaped = escaped.replace("\\", "\\\\")
        for character in (":", "'", ",", "[", "]"):
            escaped = escaped.replace(character, "\\" + character)
        style = (
            "FontName=Arial,FontSize=20,Outline=2,Shadow=1,"
            "MarginV=28,Alignment=2,BorderStyle=1"
        )
        return f"subtitles=filename='{escaped}':force_style='{style}'"

    def _emit(self, percent: int, message: str, segments_done: int = 0) -> None:
        if self.progress_callback:
            self.progress_callback(
                TranscriptionProgress(
                    max(0, min(100, percent)),
                    message,
                    segments_done,
                )
            )

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)
