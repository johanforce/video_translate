from __future__ import annotations

import importlib.resources
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from video_translator.core.errors import VideoTranslatorError


LogCallback = Callable[[str], None]
LOCAL_ENGINE = "v3turbo"


class VieNeuError(VideoTranslatorError):
    """Raised when the on-device VieNeu SDK is unavailable or synthesis fails."""


class VieNeuClient:
    """Local VieNeu-TTS SDK wrapper. This class never calls the VieNeu Cloud API."""

    def __init__(self) -> None:
        self._engine: Any = None

    @staticmethod
    def list_voices() -> list[dict[str, Any]]:
        """Read the SDK's bundled voice catalogue without initializing/downloading a model."""
        try:
            resource = importlib.resources.files("vieneu").joinpath(
                "assets/voices_v3_turbo.json"
            )
            with resource.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (ImportError, ModuleNotFoundError) as exc:
            raise VieNeuError(
                "Chưa cài VieNeu SDK local. Hãy chạy: python -m pip install vieneu"
            ) from exc
        except (OSError, ValueError, TypeError) as exc:
            raise VieNeuError(
                "Không đọc được danh sách giọng VieNeu SDK. "
                "Hãy cập nhật package vieneu lên phiên bản mới."
            ) from exc

        presets = payload.get("presets") if isinstance(payload, dict) else None
        if not isinstance(presets, dict):
            raise VieNeuError("Danh sách preset voices của VieNeu SDK không hợp lệ.")

        def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
            name, info = item
            featured = info.get("featured") if isinstance(info, dict) else None
            try:
                rank = int(featured) if featured is not None else 10_000
            except (TypeError, ValueError):
                rank = 10_000
            return rank, name.casefold()

        result: list[dict[str, Any]] = []
        for name, info in sorted(presets.items(), key=sort_key):
            if not isinstance(info, dict):
                continue
            result.append(
                {
                    "id": str(name),
                    "name": str(name),
                    "description": str(info.get("description") or ""),
                    "gender": str(info.get("gender") or ""),
                    "region": str(info.get("region") or ""),
                    "kind": "preset",
                    "engine": LOCAL_ENGINE,
                }
            )
        return result

    @classmethod
    def resolve_voice_id(
        cls,
        requested_voice: str = "",
        fallback_voice: str = "Hải Đăng",
    ) -> str:
        """Resolve legacy Cloud voice IDs to a valid local SDK preset."""
        voices = cls.list_voices()
        available = {str(item["id"]) for item in voices}
        requested = (requested_voice or "").strip()
        if requested in available:
            return requested
        fallback = (fallback_voice or "").strip()
        if fallback in available:
            return fallback
        if voices:
            return str(voices[0]["id"])
        raise VieNeuError("VieNeu SDK không có preset voice nào khả dụng.")

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from vieneu import Vieneu
        except ImportError as exc:
            raise VieNeuError(
                "Chưa cài VieNeu SDK local. Hãy chạy: python -m pip install vieneu"
            ) from exc

        self._log(None, "VieNeu: khởi tạo model v3 Turbo local...")
        try:
            # The official SDK runs ONNX/CPU by default and selects PyTorch on
            # supported CUDA systems. No Cloud API or API key is involved.
            self._engine = Vieneu(mode="v3turbo")
        except Exception as exc:
            raise VieNeuError(
                "Không khởi tạo được VieNeu SDK local. "
                "Kiểm tra eSpeak NG, môi trường Python và model cache. "
                f"Chi tiết: {exc}"
            ) from exc
        return self._engine

    def synthesize_segment(
        self,
        text: str,
        voice_id: str,
        speed: float,
        segment_id: int,
        request_key: str,
        output_dir: Path,
        cancel_event: threading.Event,
        log_callback: LogCallback | None = None,
        job_callback: Callable[..., None] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        """Generate one WAV locally, optionally time-stretching it with FFmpeg."""
        clean_text = (text or "").strip()
        if not clean_text:
            raise VieNeuError(f"Segment {segment_id} không có văn bản để đọc.")
        if len(clean_text) > 50_000:
            raise VieNeuError(f"Segment {segment_id} vượt giới hạn 50.000 ký tự.")
        if not 0.5 <= float(speed) <= 3.0:
            raise VieNeuError("Tốc độ VieNeu phải nằm trong khoảng 0.5–3.0.")
        if cancel_event.is_set():
            raise VieNeuError("Đã hủy tạo giọng đọc.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"segment_{segment_id:06d}_{request_key[:10]}.wav"
        raw_path = output_dir / f"segment_{segment_id:06d}_{request_key[:10]}.raw.wav"

        engine = self._get_engine()
        self._log(log_callback, f"VieNeu SDK local: tạo segment {segment_id}...")
        try:
            audio = engine.infer(clean_text, voice=voice_id)
            if cancel_event.is_set():
                raise VieNeuError("Đã hủy tạo giọng đọc.")
            engine.save(audio, str(raw_path))
            if not raw_path.is_file() or raw_path.stat().st_size < 64:
                raise VieNeuError(f"VieNeu SDK không tạo được audio segment {segment_id}.")

            applied_speed = round(float(speed), 2)
            if abs(applied_speed - 1.0) < 0.001:
                raw_path.replace(final_path)
            else:
                self._time_stretch(raw_path, final_path, applied_speed)
                raw_path.unlink(missing_ok=True)
        except VieNeuError:
            raw_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            raw_path.unlink(missing_ok=True)
            raise VieNeuError(
                f"VieNeu SDK lỗi ở segment {segment_id}: {exc}"
            ) from exc

        sample_rate = int(getattr(engine, "sample_rate", 48_000))
        metadata = {
            "status": "completed",
            "audio_path": str(final_path),
            "duration_seconds": None,
            "voice_id": voice_id,
            "speed": round(float(speed), 2),
            "request_key": request_key,
            "engine": LOCAL_ENGINE,
            "sample_rate": sample_rate,
        }
        self._log(log_callback, f"VieNeu local: hoàn tất segment {segment_id} -> {final_path.name}")
        return final_path, metadata

    @staticmethod
    def _time_stretch(source: Path, destination: Path, speed: float) -> None:
        """Preserve pitch while applying the app's speed control with FFmpeg atempo."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise VieNeuError(
                "Cần FFmpeg trong PATH để áp dụng tốc độ đọc khác 1.0."
            )
        remaining = float(speed)
        filters: list[str] = []
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.6f}")
        temporary = destination.with_name(destination.stem + ".tmp.wav")
        try:
            completed = subprocess.run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source),
                    "-filter:a", ",".join(filters),
                    "-c:a", "pcm_s16le",
                    str(temporary),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise VieNeuError(
                    "FFmpeg không điều chỉnh được tốc độ đọc: "
                    + (completed.stderr.strip()[-1200:] or "unknown error")
                )
            if not temporary.is_file() or temporary.stat().st_size < 64:
                raise VieNeuError("FFmpeg tạo audio sau time-stretch không hợp lệ.")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    @staticmethod
    def _log(callback: LogCallback | None, message: str) -> None:
        if callback:
            callback(message)
