from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from video_translator.config.settings import AppSettings
from video_translator.core.models import DownloadProgress, TranscriptionProgress
from video_translator.core.project_manager import ProjectManager
from video_translator.download.downloader import VideoDownloader
from video_translator.gemini.film_translation_service import FilmTranslationService
from video_translator.gemini.json_utils import read_json, sha256_file, sha256_text
from video_translator.gemini.movie_context_service import MovieContextService
from video_translator.gemini.prompt_loader import load_prompt
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.rendering.render_service import RenderService
from video_translator.transcription.service import TranscriptionService
from video_translator.tts.tts_service import VieNeuTTSService
from video_translator.tts.vieneu_client import VieNeuClient


ProgressCallback = Callable[[TranscriptionProgress], None]
LogCallback = Callable[[str], None]


class PipelineError(RuntimeError):
    """Raised when the end-to-end video translation pipeline cannot complete."""


class ProjectPipelineService:
    """Run or resume every project stage, relying on each stage's durable artifacts."""

    _WEIGHTS = {
        "download": (0, 8),
        "transcription": (8, 25),
        "context": (33, 17),
        "translation": (50, 25),
        "tts": (75, 17),
        "render": (92, 8),
    }

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.settings = AppSettings.load()

    def run(
        self,
        output_dir: Path,
        video_path: Path | None = None,
        video_url: str = "",
        transcript_path: Path | None = None,
        whisper_model: str = "small",
        source_language: str | None = None,
        gemini_api_key: str = "",
        gemini_model: str = "gemini-3.8-flash",
        translation_batch_size: int = 30,
        voice_id: str = "",
        voice_speed: float = 1.0,
        subtitle_mode: str = "both",
        original_audio_mode: str = "duck",
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Path]:
        cancel_event = cancel_event or threading.Event()
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

        current_video = Path(video_path).expanduser().resolve() if video_path else None
        current_transcript = (
            Path(transcript_path).expanduser().resolve() if transcript_path else None
        )

        if current_transcript and current_transcript.is_file():
            try:
                existing_transcript = read_json(current_transcript)
                recorded_video = Path(
                    str(existing_transcript.get("source_video", ""))
                ).expanduser()
                if current_video is None and recorded_video.is_file():
                    current_video = recorded_video.resolve()
            except (OSError, ValueError):
                pass

        if current_video is None or not current_video.is_file():
            if not video_url.strip():
                raise PipelineError(
                    "Chưa có video nguồn. Hãy chọn video, nhập URL hoặc mở transcript "
                    "có đường dẫn source_video còn tồn tại."
                )
            self._check_cancel(cancel_event)
            self._log("Bắt đầu / tiếp tục download video...")
            result = VideoDownloader(
                output_dir=destination,
                max_height=self.settings.max_height,
                progress_callback=self._download_progress,
                log_callback=self.log_callback,
                cancel_event=cancel_event,
            ).download(video_url)
            current_video = result.path.resolve()
            self._emit_stage(
                "download",
                100,
                "Download hoàn tất.",
                0,
                cancel_event,
            )

        if current_transcript is None:
            current_transcript = destination / f"{current_video.stem}.transcript.json"

        # If an old transcript points at a moved/deleted source, update the pointer.
        # Transcription's source fingerprint then decides whether existing chunk cache
        # is still valid; completed chunks are reused whenever the bytes are identical.
        if current_transcript.is_file():
            try:
                transcript_data = read_json(current_transcript)
            except (OSError, ValueError):
                transcript_data = {}
            recorded_video = Path(
                str(transcript_data.get("source_video", ""))
            ).expanduser()
            if (
                current_video.is_file()
                and (
                    not recorded_video.is_file()
                    or recorded_video.resolve() != current_video.resolve()
                )
            ):
                # Point the transcript at the selected source. The transcription
                # checkpoint compares content hashes, so the same moved file is reused
                # while a genuinely different source invalidates dependent stages.
                transcript_data["source_video"] = str(current_video)
                self._write_existing_transcript(current_transcript, transcript_data)

        paths = MovieProjectPaths.from_transcript(current_transcript)
        manager = ProjectManager(paths.transcript_json)
        self._log(f"Project cache: {manager.project_file}")

        def stage_progress(stage: str, weight_offset: int, weight: int):
            def callback(progress: TranscriptionProgress) -> None:
                percent = progress.percent
                if percent is not None:
                    manager.update_progress(stage, int(percent), progress.message)
                    global_percent = weight_offset + int(weight * percent / 100)
                else:
                    global_percent = weight_offset
                self._emit_global(
                    global_percent,
                    f"{stage}: {progress.message}",
                    progress.segments_done,
                )
            return callback

        def run_stage(stage: str, action: Callable[[], Any], **details: Any) -> Any:
            self._check_cancel(cancel_event)
            manager.refresh()
            if self._is_stage_completed(
                manager,
                stage,
                gemini_model,
                whisper_model,
                source_language,
                voice_id,
                voice_speed,
                translation_batch_size,
                subtitle_mode,
                original_audio_mode,
            ):
                self._log(f"[CACHE] Bỏ qua {stage}: kết quả đã hoàn tất và hợp lệ.")
                return None

            manager.start_stage(stage, **details)
            self._log(f"[{stage.upper()}] Bắt đầu hoặc tiếp tục từ checkpoint...")
            try:
                value = action()
                self._check_cancel(cancel_event)
                manager.complete_stage(stage, **details)
                self._log(f"[{stage.upper()}] Hoàn tất.")
                return value
            except Exception as exc:
                manager.fail_stage(stage, str(exc))
                raise

        # 1. Whisper transcription — its own chunk cache preserves completed chunks.
        transcription_result = run_stage(
            "transcription",
            lambda: TranscriptionService(
                progress_callback=stage_progress("transcription", 8, 25),
                log_callback=self.log_callback,
            ).run(
                video_path=current_video,
                output_dir=paths.transcript_json.parent,
                model_size=whisper_model,
                language=source_language,
                cancel_event=cancel_event,
            ),
            source_video=str(current_video),
            model=whisper_model,
            requested_language="auto" if source_language is None else source_language,
        )
        if transcription_result is not None:
            generated_transcript = Path(transcription_result[0]).resolve()
            if generated_transcript != paths.transcript_json:
                current_transcript = generated_transcript
                paths = MovieProjectPaths.from_transcript(current_transcript)
                manager = ProjectManager(current_transcript)

        # Re-read the durable transcript, never rely on in-memory partial state.
        if not paths.transcript_json.is_file():
            raise PipelineError("Whisper chưa tạo transcript JSON.")
        transcript_data = read_json(paths.transcript_json)
        if transcript_data.get("is_complete") is False:
            raise PipelineError(
                "Transcript vẫn chưa hoàn tất; các chunk đã nhận diện được lưu cache."
            )

        # 2. Whole-film context profile.
        run_stage(
            "context",
            lambda: MovieContextService(
                api_key=gemini_api_key,
                model=gemini_model,
                progress_callback=stage_progress("context", 33, 17),
                log_callback=self.log_callback,
            ).run(
                paths.transcript_json,
                batch_segments=max(40, int(translation_batch_size) * 4),
                cancel_event=cancel_event,
            ),
            model=gemini_model,
            batch_segments=max(40, int(translation_batch_size) * 4),
        )

        # 3. Contextual translation and line-by-line QC.
        run_stage(
            "translation",
            lambda: FilmTranslationService(
                api_key=gemini_api_key,
                model=gemini_model,
                progress_callback=stage_progress("translation", 50, 25),
                log_callback=self.log_callback,
            ).run(
                paths.transcript_json,
                paths.context_json,
                batch_size=translation_batch_size,
                cancel_event=cancel_event,
            ),
            model=gemini_model,
            batch_size=int(translation_batch_size),
        )

        # Restore the last local VieNeu SDK voice configuration when reopening a project.
        if paths.tts_manifest.is_file():
            try:
                saved_tts = read_json(paths.tts_manifest)
                current_translation_hash = sha256_text(
                    paths.translated_json.read_text(encoding="utf-8")
                )
                if saved_tts.get("translation_fingerprint") == current_translation_hash:
                    if not voice_id:
                        voice_id = str(saved_tts.get("voice_id") or "")
                        voice_speed = float(saved_tts.get("speed", voice_speed))
            except (OSError, ValueError, TypeError):
                pass

        requested_voice_id = voice_id
        try:
            voice_id = VieNeuClient.resolve_voice_id(
                requested_voice_id,
                self.settings.vieneu_voice,
            )
        except Exception as exc:
            raise RuntimeError(f"Không tải được preset voice từ VieNeu SDK: {exc}") from exc
        if requested_voice_id != voice_id:
            self._log(
                f"Giọng lưu từ Cloud v4 '{requested_voice_id or 'trống'}' không có trong SDK local; "
                f"chuyển sang preset '{voice_id}'."
            )

        # 4. Per-line local VieNeu SDK synthesis. Completed WAV files are reused from cache.
        run_stage(
            "tts",
            lambda: VieNeuTTSService(
                progress_callback=stage_progress("tts", 75, 17),
                log_callback=self.log_callback,
            ).run(
                translated_json=paths.translated_json,
                voice_id=voice_id,
                speed=voice_speed,
                cancel_event=cancel_event,
            ),
            voice_id=voice_id,
            speed=float(voice_speed),
        )

        # 5. Assemble timeline, mix narration with source audio and render MP4.
        render_result = run_stage(
            "render",
            lambda: RenderService(
                progress_callback=stage_progress("render", 92, 8),
                log_callback=self.log_callback,
            ).run(
                translated_json=paths.translated_json,
                subtitle_mode=subtitle_mode,
                original_audio_mode=original_audio_mode,
                output_path=(
                    paths.transcript_json.parent
                    / (
                        paths.transcript_json.name.removesuffix(".transcript.json")
                        + ".vi.dubbed.mp4"
                    )
                ),
                cancel_event=cancel_event,
            ),
            subtitle_mode=subtitle_mode,
            original_audio_mode=original_audio_mode,
        )

        manager.refresh()
        output = (
            render_result[0]
            if render_result is not None
            else (
                paths.transcript_json.parent
                / (
                    paths.transcript_json.name.removesuffix(".transcript.json")
                    + ".vi.dubbed.mp4"
                )
            )
        )
        metadata = (
            render_result[1]
            if render_result is not None
            else output.with_suffix(".render.json")
        )
        if not output.is_file():
            raise PipelineError(
                "Project đánh dấu render hoàn tất nhưng không tìm thấy MP4 đầu ra."
            )

        self._emit_global(100, "Toàn bộ project đã hoàn tất.", 0)
        self._log(f"MP4 hoàn chỉnh: {output}")
        self._log(f"Project manifest: {manager.project_file}")
        return {
            "transcript": paths.transcript_json,
            "context": paths.context_json,
            "translation": paths.translated_json,
            "subtitle": paths.translated_srt,
            "tts_manifest": paths.tts_manifest,
            "video": output,
            "render_metadata": metadata,
            "project": manager.project_file,
        }

    @staticmethod
    def _is_stage_completed(
        manager: ProjectManager,
        stage: str,
        gemini_model: str,
        whisper_model: str,
        source_language: str | None,
        voice_id: str,
        voice_speed: float,
        translation_batch_size: int,
        subtitle_mode: str,
        original_audio_mode: str,
    ) -> bool:
        manager.refresh()
        if manager.state["stages"].get(stage, {}).get("status") != "completed":
            return False

        try:
            if stage == "context":
                # A complete manual translation has already been reviewed against
                # this transcript and intentionally bypasses Gemini context analysis.
                if manager._translation_complete():
                    manual_translation = read_json(manager.paths.translated_json)
                    if manual_translation.get("translation_source") == "manual":
                        return True
                context = read_json(manager.paths.context_json)
                transcript = read_json(manager.paths.transcript_json)
                total_segments = len(transcript.get("segments", []))
                processed_segments = int(
                    context.get("processed_segments", total_segments)
                )
                return (
                    context.get("is_complete") is True
                    and context.get("transcript_fingerprint")
                    == sha256_file(manager.paths.transcript_json)
                    and processed_segments == total_segments
                )
            if stage == "translation":
                translated = read_json(manager.paths.translated_json)
                if translated.get("translation_source") == "manual":
                    return manager._translation_complete()
                prompt_signature = (
                    load_prompt("system_prompt.txt")
                    + load_prompt("translation.txt")
                    + load_prompt("quality_control.txt")
                )
                transcript_fingerprint = sha256_file(manager.paths.transcript_json)
                context_fingerprint = sha256_file(manager.paths.context_json)
                batch_size = int(translation_batch_size)
                expected_signature = sha256_text(
                    transcript_fingerprint
                    + context_fingerprint
                    + str(batch_size)
                    + prompt_signature
                )
                legacy_signatures = {
                    sha256_text(
                        transcript_fingerprint
                        + context_fingerprint
                        + str(batch_size)
                        + prompt_signature
                    ),
                    sha256_text(
                        transcript_fingerprint
                        + context_fingerprint
                        + str(translated.get("model") or "")
                        + str(batch_size)
                        + prompt_signature
                    ),
                }
                return (
                    int(translated.get("batch_size", 0)) == batch_size
                    and translated.get("config_signature")
                    in ({expected_signature} | legacy_signatures)
                )
            if stage == "tts":
                manifest = read_json(manager.paths.tts_manifest)
                return (
                    manifest.get("voice_id") == voice_id
                    and abs(float(manifest.get("speed", -1)) - float(voice_speed)) < 0.001
                )
            if stage == "render":
                output = manager.paths.transcript_json.parent / (
                    manager.paths.transcript_json.name.removesuffix(".transcript.json")
                    + ".vi.dubbed.mp4"
                )
                metadata = read_json(output.with_suffix(".render.json"))
                return (
                    metadata.get("subtitle_mode") == subtitle_mode
                    and metadata.get("original_audio_mode") == original_audio_mode
                )
            if stage == "transcription":
                transcript = read_json(manager.paths.transcript_json)
                requested = (source_language or "auto").strip().lower()
                actual_requested = str(transcript.get("requested_language") or "auto")
                source_video = Path(
                    str(transcript.get("source_video", ""))
                ).expanduser()
                if not source_video.is_file():
                    return False
                expected_signature = sha256_text(
                    f"{sha256_file(source_video)}|{whisper_model}|{requested}|240|"
                    "mono16k-pcm|word-timestamps-v1"
                )
                return (
                    transcript.get("model") == whisper_model
                    and actual_requested == requested
                    and transcript.get("transcription_signature") == expected_signature
                )
        except (OSError, ValueError, TypeError):
            return False
        return False

    @staticmethod
    def _write_existing_transcript(path: Path, payload: dict[str, Any]) -> None:
        from video_translator.gemini.json_utils import write_json_atomic

        write_json_atomic(path, payload)

    def _download_progress(self, progress: DownloadProgress) -> None:
        percent = progress.percent
        global_percent = 0 if percent is None else int(percent * 0.08)
        self._emit_global(
            global_percent,
            f"Download: {progress.message}",
            0,
        )

    def _emit_stage(
        self,
        stage: str,
        percent: int,
        message: str,
        segments_done: int,
        cancel_event: threading.Event,
    ) -> None:
        self._check_cancel(cancel_event)
        offset, weight = self._WEIGHTS.get(stage, (0, 100))
        self._emit_global(
            offset + int(weight * percent / 100),
            f"{stage}: {message}",
            segments_done,
        )

    def _emit_global(self, percent: int, message: str, segments_done: int) -> None:
        if self.progress_callback:
            self.progress_callback(
                TranscriptionProgress(
                    max(0, min(100, int(percent))),
                    message,
                    segments_done,
                )
            )

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    @staticmethod
    def _check_cancel(cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            raise PipelineError("Đã hủy pipeline. Checkpoint đã được giữ để tiếp tục.")
