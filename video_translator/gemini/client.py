from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from video_translator.core.errors import VideoTranslatorError


class GeminiClientError(VideoTranslatorError):
    """Raised for Gemini API, response, or configuration errors."""


class GeminiOutputLimitError(GeminiClientError):
    """Raised when Gemini truncates a response because it reached MAX_TOKENS."""


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
        debug_path: Path | None = None,
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
                    diagnostics = self._response_diagnostics(response)
                    if self._hit_output_limit(response):
                        message = "Gemini đã chạm MAX_TOKENS nhưng không trả về nội dung JSON."
                        if diagnostics:
                            message += f" [{diagnostics}]"
                        saved_path = self._save_raw_response(
                            debug_path, response_text or ""
                        )
                        if saved_path:
                            message += f" {saved_path}"
                        raise GeminiOutputLimitError(message)
                    raise GeminiClientError("Gemini trả về nội dung rỗng.")
                try:
                    return self._parse_json(response_text)
                except GeminiClientError as parse_error:
                    diagnostics = self._response_diagnostics(response)
                    saved_path = self._save_raw_response(debug_path, response_text)
                    details = [str(parse_error)]
                    if diagnostics:
                        details.append(diagnostics)
                    if saved_path:
                        details.append(saved_path)
                    message = " ".join(details)
                    if self._hit_output_limit(response):
                        raise GeminiOutputLimitError(message) from parse_error

                    # A normal STOP with malformed JSON is not an output-limit
                    # failure. Give the model one syntax-only repair attempt before
                    # failing the chapter; do not re-run the chapter analysis.
                    if cancel_event.is_set():
                        raise GeminiClientError("Đã hủy tác vụ Gemini.") from parse_error
                    repair_system = (
                        "You are a strict JSON syntax repair tool. "
                        "Return exactly one valid JSON object. Repair syntax only: "
                        "preserve every key, value, list item, and meaning from the "
                        "provided text. Do not summarize, translate, infer, add, or "
                        "remove information. Do not include Markdown fences or prose."
                    )
                    repair_prompt = (
                        "The following model response was intended to be a JSON object "
                        "but Python's JSON parser rejected it. Repair only the JSON "
                        "syntax and return the complete corrected object.\n\n"
                        f"Parser error: {parse_error}\n\n"
                        "INVALID RESPONSE TO REPAIR:\n"
                        + response_text
                    )
                    repair_config = self._types.GenerateContentConfig(
                        system_instruction=repair_system,
                        response_mime_type="application/json",
                        temperature=0.0,
                        max_output_tokens=8192,
                    )
                    try:
                        repaired_response = self._client.models.generate_content(
                            model=self.model,
                            contents=repair_prompt,
                            config=repair_config,
                        )
                    except Exception as repair_exc:
                        raise GeminiClientError(
                            message
                            + " Tự sửa JSON không thành công: "
                            + self._friendly_error(repair_exc)
                        ) from repair_exc

                    repaired_text = getattr(repaired_response, "text", None)
                    if not repaired_text or not repaired_text.strip():
                        if self._hit_output_limit(repaired_response):
                            repair_diagnostics = self._response_diagnostics(
                                repaired_response
                            )
                            raise GeminiOutputLimitError(
                                message
                                + " Lần tự sửa JSON chạm MAX_TOKENS. "
                                + repair_diagnostics
                            ) from parse_error
                        raise GeminiClientError(
                            message + " Gemini không trả nội dung trong lần tự sửa JSON."
                        ) from parse_error

                    try:
                        repaired_value = self._parse_json(repaired_text)
                    except GeminiClientError as repair_error:
                        repair_diagnostics = self._response_diagnostics(
                            repaired_response
                        )
                        repair_path = (
                            debug_path.with_suffix(".repair.raw.txt")
                            if debug_path is not None
                            else None
                        )
                        repair_saved = self._save_raw_response(
                            repair_path, repaired_text
                        )
                        repair_details = [
                            message,
                            "Gemini đã thử tự sửa JSON nhưng kết quả vẫn không hợp lệ:",
                            str(repair_error),
                        ]
                        if repair_diagnostics:
                            repair_details.append(repair_diagnostics)
                        if repair_saved:
                            repair_details.append(repair_saved)
                        if self._hit_output_limit(repaired_response):
                            raise GeminiOutputLimitError(
                                " ".join(repair_details)
                            ) from repair_error
                        raise GeminiClientError(
                            " ".join(repair_details)
                        ) from repair_error

                    if debug_path is not None:
                        repaired_path = debug_path.with_suffix(".repaired.json")
                        try:
                            repaired_path.write_text(
                                json.dumps(
                                    repaired_value,
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                encoding="utf-8",
                                newline="\n",
                            )
                        except OSError:
                            pass
                    return repaired_value
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
    def _response_diagnostics(response: Any) -> str:
        """Extract provider metadata useful for diagnosing incomplete JSON output."""
        details: list[str] = []
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            if finish_reason is not None:
                details.append(f"finish_reason={finish_reason}")
        usage = getattr(response, "usage_metadata", None)
        total_tokens = getattr(usage, "total_token_count", None) if usage else None
        if total_tokens is not None:
            details.append(f"total_token_count={total_tokens}")
        return f"[Gemini: {'; '.join(details)}]" if details else ""

    @staticmethod
    def _hit_output_limit(response: Any) -> bool:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return False
        finish_reason = getattr(candidates[0], "finish_reason", None)
        return "MAX_TOKENS" in str(finish_reason).upper()

    @staticmethod
    def _save_raw_response(debug_path: Path | None, response_text: str) -> str:
        """Save invalid model output without allowing a disk error to hide the parse error."""
        if debug_path is None:
            return ""
        path = Path(debug_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response_text, encoding="utf-8", newline="\n")
        except OSError as exc:
            return f"Không lưu được phản hồi gốc ({path}): {exc}"
        return f"Phản hồi Gemini gốc đã lưu: {path}"

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
