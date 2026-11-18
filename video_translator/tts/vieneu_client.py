from __future__ import annotations

import hashlib
import json
import mimetypes
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from video_translator.core.errors import VideoTranslatorError


ProgressCallback = Callable[[int, str], None]
LogCallback = Callable[[str], None]

API_BASE = "https://api.vieneu.io/api/v1"


class VieNeuError(VideoTranslatorError):
    """Raised when VieNeu voice listing or synthesis fails."""


class VieNeuClient:
    """VieNeu v4 client using idempotent async jobs for resumable synthesis."""

    def __init__(self, api_key: str, timeout: int = 60) -> None:
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise VieNeuError(
                "Thiếu VieNeu API key. Nhập key trong ứng dụng hoặc đặt VIENEU_API_KEY trong .env."
            )
        self.timeout = max(15, int(timeout))
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "VideoTranslator/0.5",
            }
        )

    @staticmethod
    def list_voices(api_key: str | None = None) -> list[dict[str, Any]]:
        """Get the current v4 catalogue; authenticated calls may include private clones."""
        headers = {"Accept": "application/json"}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        try:
            response = requests.get(
                f"{API_BASE}/voices",
                params={"engine": "v4"},
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise VieNeuError(f"Không lấy được danh sách giọng VieNeu: {exc}") from exc
        except ValueError as exc:
            raise VieNeuError("VieNeu trả danh sách giọng không phải JSON.") from exc

        voices = payload.get("voices") if isinstance(payload, dict) else None
        if not isinstance(voices, list):
            raise VieNeuError("Cấu trúc danh sách giọng VieNeu không hợp lệ.")
        result: list[dict[str, Any]] = []
        for voice in voices:
            if not isinstance(voice, dict) or not voice.get("id"):
                continue
            result.append(
                {
                    "id": str(voice["id"]),
                    "name": str(voice.get("name") or voice["id"]),
                    "description": str(voice.get("description") or ""),
                    "gender": str(voice.get("gender") or ""),
                    "region": str(voice.get("region") or ""),
                    "kind": str(voice.get("kind") or "catalog"),
                    "engine": str(voice.get("engine") or "v4"),
                }
            )
        return result

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
    ) -> tuple[Path, dict[str, Any]]:
        """Submit or resume one idempotent TTS job and download the generated audio."""
        clean_text = (text or "").strip()
        if not clean_text:
            raise VieNeuError(f"Segment {segment_id} không có văn bản để đọc.")
        if len(clean_text) > 50_000:
            raise VieNeuError(f"Segment {segment_id} vượt giới hạn 50.000 ký tự.")
        if not 0.5 <= float(speed) <= 3.0:
            raise VieNeuError("Tốc độ VieNeu phải nằm trong khoảng 0.5–3.0.")

        body = {
            "text": clean_text,
            "voiceId": voice_id,
            "speed": round(float(speed), 2),
            "engine": "v4",
        }
        headers = {"Idempotency-Key": request_key}
        self._log(log_callback, f"VieNeu: gửi segment {segment_id}...")
        response_data = self._submit_job(body, headers, cancel_event)
        job_id = response_data.get("jobId")
        status = str(response_data.get("status") or "").lower()
        audio_url = response_data.get("audioUrl")

        if status != "completed" or not audio_url:
            if not job_id:
                raise VieNeuError(
                    f"VieNeu không trả jobId/audioUrl cho segment {segment_id}: "
                    f"{json.dumps(response_data, ensure_ascii=False)}"
                )
            response_data = self._poll_job(str(job_id), cancel_event, segment_id, log_callback)
            audio_url = response_data.get("audioUrl")

        if not isinstance(audio_url, str) or not audio_url.startswith("https://"):
            raise VieNeuError(f"VieNeu không trả audioUrl HTTPS cho segment {segment_id}.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self._download_audio(audio_url, output_dir, segment_id, request_key)
        metadata = {
            "job_id": str(job_id or response_data.get("jobId") or ""),
            "status": "completed",
            "audio_path": str(audio_path),
            "duration_seconds": response_data.get("duration"),
            "voice_id": voice_id,
            "speed": round(float(speed), 2),
            "request_key": request_key,
        }
        self._log(log_callback, f"VieNeu: hoàn tất segment {segment_id} -> {audio_path.name}")
        return audio_path, metadata

    def _submit_job(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        url = f"{API_BASE}/tts"
        last_error: Exception | None = None
        for attempt in range(1, 6):
            if cancel_event.is_set():
                raise VieNeuError("Đã hủy tạo giọng đọc.")
            try:
                response = self.session.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=(15, self.timeout),
                )
                if response.status_code == 409:
                    retry_after = self._retry_after(response, 1)
                    self._wait(cancel_event, retry_after)
                    continue
                if response.status_code in {429, 500, 502, 503, 504}:
                    retry_after = self._retry_after(response, min(30, attempt * 3))
                    self._wait(cancel_event, retry_after)
                    continue
                self._raise_for_api_error(response)
                payload = response.json()
                if not isinstance(payload, dict):
                    raise VieNeuError("VieNeu trả job response không hợp lệ.")
                return payload
            except requests.RequestException as exc:
                # The same idempotency key is reused, so a lost response cannot create
                # a second billable job while the key remains valid on the provider.
                last_error = exc
                if attempt >= 5:
                    break
                self._wait(cancel_event, min(30, 2 ** attempt))
            except ValueError as exc:
                raise VieNeuError(f"VieNeu trả JSON job không hợp lệ: {exc}") from exc
        raise VieNeuError(f"Không gửi được job VieNeu sau nhiều lần thử: {last_error or 'quá thời gian chờ'}")

    def _poll_job(
        self,
        job_id: str,
        cancel_event: threading.Event,
        segment_id: int,
        log_callback: LogCallback | None,
    ) -> dict[str, Any]:
        url = f"{API_BASE}/tts/{job_id}"
        deadline = time.monotonic() + 30 * 60
        delay = 2.0
        while time.monotonic() < deadline:
            if cancel_event.is_set():
                raise VieNeuError("Đã hủy tạo giọng đọc. Job VieNeu vẫn được lưu để tiếp tục lần sau.")
            try:
                response = self.session.get(url, timeout=(15, 30))
                if response.status_code in {429, 500, 502, 503, 504}:
                    self._wait(cancel_event, self._retry_after(response, delay))
                    continue
                self._raise_for_api_error(response)
                payload = response.json()
            except requests.RequestException as exc:
                self._log(log_callback, f"Poll segment {segment_id} gặp lỗi mạng: {exc}")
                self._wait(cancel_event, delay)
                delay = min(15.0, delay * 1.5)
                continue
            except ValueError as exc:
                raise VieNeuError(f"VieNeu trả trạng thái job không hợp lệ: {exc}") from exc

            if not isinstance(payload, dict):
                raise VieNeuError(f"VieNeu trả trạng thái segment {segment_id} không hợp lệ.")
            status = str(payload.get("status") or "").lower()
            if status == "completed" and payload.get("audioUrl"):
                return payload
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise VieNeuError(
                    f"VieNeu job {job_id} thất bại ở segment {segment_id}: "
                    f"{payload.get('message') or payload.get('error') or status}"
                )
            self._log(log_callback, f"VieNeu segment {segment_id}: {status or 'đang xử lý'}...")
            self._wait(cancel_event, delay)
            delay = min(10.0, delay * 1.25)
        raise VieNeuError(
            f"VieNeu job {job_id} vẫn chưa hoàn tất sau 30 phút. "
            "Job được giữ trong manifest; chạy lại để tiếp tục poll."
        )

    @staticmethod
    def _raise_for_api_error(response: requests.Response) -> None:
        if response.ok:
            return
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        code = payload.get("code") or payload.get("error", {}).get("code") if isinstance(payload.get("error"), dict) else payload.get("code")
        message = payload.get("message") or payload.get("error", {}).get("message") if isinstance(payload.get("error"), dict) else payload.get("message")
        message = message or response.text[:1000] or response.reason
        if response.status_code == 403 or code in {
            "GRANT_EXPIRED", "GRANT_TOKENS_EXHAUSTED", "GRANT_DAILY_LIMIT",
            "GRANT_WEEKLY_LIMIT", "TRIAL_API_DAILY_LIMIT",
        }:
            raise VieNeuError(f"VieNeu từ chối do gói/quota ({code or response.status_code}): {message}")
        raise VieNeuError(f"VieNeu API HTTP {response.status_code} ({code or 'error'}): {message}")

    @staticmethod
    def _retry_after(response: requests.Response, default: float) -> float:
        try:
            return max(1.0, min(60.0, float(response.headers.get("Retry-After", default))))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _wait(cancel_event: threading.Event, seconds: float) -> None:
        if cancel_event.wait(max(0.5, seconds)):
            raise VieNeuError("Đã hủy tạo giọng đọc.")

    @staticmethod
    def _download_audio(
        audio_url: str,
        output_dir: Path,
        segment_id: int,
        request_key: str,
    ) -> Path:
        try:
            response = requests.get(audio_url, timeout=(15, 120), stream=True)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
            suffix = {
                "audio/wav": ".wav",
                "audio/x-wav": ".wav",
                "audio/mpeg": ".mp3",
                "audio/mp3": ".mp3",
                "audio/ogg": ".ogg",
                "audio/opus": ".opus",
                "audio/mp4": ".m4a",
                "audio/flac": ".flac",
            }.get(content_type)
            if suffix is None:
                suffix = Path(urlparse(audio_url).path).suffix.lower()
            if suffix not in {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac"}:
                suffix = mimetypes.guess_extension(content_type) or ".audio"

            final_path = output_dir / f"segment_{segment_id:06d}_{request_key[:10]}{suffix}"
            temporary = final_path.with_suffix(final_path.suffix + ".tmp")
            written = 0
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        stream.write(chunk)
                        written += len(chunk)
            if written < 64:
                temporary.unlink(missing_ok=True)
                raise VieNeuError(f"Audio segment {segment_id} quá nhỏ hoặc rỗng.")
            temporary.replace(final_path)
            return final_path
        except requests.RequestException as exc:
            raise VieNeuError(f"Không tải được audio segment {segment_id}: {exc}") from exc

    @staticmethod
    def _log(callback: LogCallback | None, message: str) -> None:
        if callback:
            callback(message)
