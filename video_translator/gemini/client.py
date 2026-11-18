from __future__ import annotations

import json
import re
import threading
from typing import Any

from video_translator.core.errors import VideoTranslatorError


class GeminiClientError(VideoTranslatorError):
    """Raised for Gemini API, response, or configuration errors."""


class GeminiJsonClient:
    """Small provider wrapper that requests JSON and retries transient failures."""

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash") -> None:
        key = (api_key or "").strip()
        if not key:
            raise GeminiClientError(
                "Thiếu Gemini API key. Nhập key trong ứng dụng hoặc đặt GEMINI_API_KEY trong .env."
            )
        self.model = (model or "gemini-3.6-flash").strip()
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiClientError(
                "Chưa cài Google Gen AI SDK. Hãy chạy pip install -r requirements.txt."
            ) from exc

        self._types = types
        self._client = genai.Client(api_key=key)

    def generate_json(
        self,
        prompt: str,
        system_instruction: str,
        cancel_event: threading.Event | None = None,
        max_attempts: int = 4,
    ) -> dict[str, Any]:
        cancel_event = cancel_event or threading.Event()
        config = self._types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=8192,
        )
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            if cancel_event.is_set():
                raise GeminiClientError("Đã hủy tác vụ Gemini.")
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                response_text = getattr(response, "text", None)
                if not response_text or not response_text.strip():
                    raise GeminiClientError("Gemini trả về nội dung rỗng.")
                return self._parse_json(response_text)
            except Exception as exc:
                last_error = exc
                if not self._is_retryable(exc) or attempt >= max_attempts:
                    if isinstance(exc, GeminiClientError):
                        raise
                    raise GeminiClientError(self._friendly_error(exc)) from exc
                delay = min(30, 2 ** (attempt - 1) * 2)
                if cancel_event.wait(delay):
                    raise GeminiClientError("Đã hủy tác vụ Gemini.") from exc
        raise GeminiClientError(self._friendly_error(last_error or RuntimeError("Unknown error")))

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        fence = chr(96) * 3
        if cleaned.startswith(fence):
            cleaned = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*" + re.escape(fence) + r"$", "", cleaned)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GeminiClientError(
                f"Gemini không trả về JSON hợp lệ (dòng {exc.lineno}, cột {exc.colno})."
            ) from exc
        if not isinstance(value, dict):
            raise GeminiClientError("Gemini phải trả về một JSON object.")
        return value

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "429", "500", "502", "503", "504",
                "timeout", "timed out", "temporarily unavailable",
                "connection reset", "connection error", "resource exhausted",
            )
        )

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()
        if "api_key_invalid" in lowered or "api key not valid" in lowered or "401" in lowered:
            return "Gemini API key không hợp lệ. Hãy kiểm tra GEMINI_API_KEY."
        if "429" in lowered or "resource exhausted" in lowered:
            return f"Gemini đang giới hạn tốc độ hoặc hết quota: {message}"
        if "404" in lowered or "not found" in lowered:
            return f"Không tìm thấy Gemini model hoặc endpoint: {message}"
        return f"Gemini API lỗi: {message}"
