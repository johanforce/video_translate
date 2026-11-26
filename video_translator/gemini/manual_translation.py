from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from video_translator.gemini.json_utils import read_json, sha256_file, sha256_text, write_json_atomic
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.transcription.transcript_writer import format_srt_timestamp


TEMPLATE_TYPE = "video_translator.manual_translation"
TEMPLATE_SCHEMA_VERSION = 1

ALLOWED_EMOTIONS = {
    "neutral", "happy", "sad", "angry", "fearful", "surprised",
    "sarcastic", "humorous", "romantic", "disgusted", "curious",
    "reassuring", "threatening", "uncertain",
}
ALLOWED_DELIVERIES = {
    "normal", "shouting", "whispering", "crying", "laughing",
    "breathless", "trembling", "singing", "robotic", "soft", "emphatic",
}
ALLOWED_ISSUE_SEVERITIES = {"critical", "high"}
ALLOWED_ISSUE_CATEGORIES = {
    "accuracy", "omission", "addition", "terminology", "character_consistency",
    "addressing", "naturalness", "dubbing_fluency", "emotion", "delivery",
    "uncertainty",
}


class ManualTranslationError(ValueError):
    """Raised when a manual translation template cannot be safely imported."""


def create_manual_translation_template(
    transcript_path: Path,
    destination_path: Path,
) -> int:
    """Create an editable JSON template tied to the exact source transcript."""
    paths = MovieProjectPaths.from_transcript(transcript_path)
    destination = Path(destination_path).expanduser().resolve()
    protected_paths = {
        paths.transcript_json,
        paths.context_json,
        paths.translated_json,
        paths.translated_srt,
        paths.quality_json,
        paths.tts_manifest,
    }
    if destination in protected_paths:
        raise ManualTranslationError(
            "Không thể ghi file mẫu đè lên transcript hoặc artifact của project."
        )

    transcript = _load_transcript(paths.transcript_json)
    source_segments = _validated_source_segments(transcript)

    template = {
        "template_type": TEMPLATE_TYPE,
        "schema_version": TEMPLATE_SCHEMA_VERSION,
        "source_transcript": str(paths.transcript_json),
        "source_fingerprint": sha256_file(paths.transcript_json),
        "instructions": {
            "steps": [
                "Fill translated_text for every segment that has source_text.",
                "Set reviewed to true for every segment after checking its translation.",
                "Set quality_control.completed to true after the final manual review.",
                "Keep id, source_text, source_transcript and source_fingerprint unchanged.",
            ],
            "emotion_values": sorted(ALLOWED_EMOTIONS),
            "delivery_values": sorted(ALLOWED_DELIVERIES),
            "empty_source_rule": "Keep translated_text empty when source_text is empty.",
        },
        "quality_control": {
            "completed": False,
            "issues": [],
        },
        "segments": [
            {
                "id": int(item["id"]),
                "source_text": str(item.get("text", "")),
                "translated_text": "",
                "emotion": "neutral",
                "delivery": "normal",
                "reviewed": False,
            }
            for item in source_segments
        ],
    }
    write_json_atomic(destination, template)
    return len(source_segments)


def import_manual_translation(
    transcript_path: Path,
    template_path: Path,
) -> tuple[Path, Path, Path, int]:
    """Validate a manual JSON file and write the app's normal translation artifacts."""
    paths = MovieProjectPaths.from_transcript(transcript_path)
    transcript = _load_transcript(paths.transcript_json)
    source_segments = _validated_source_segments(transcript)
    source_fingerprint = sha256_file(paths.transcript_json)

    try:
        template = read_json(Path(template_path))
    except (OSError, ValueError) as exc:
        raise ManualTranslationError(f"Không đọc được file JSON dịch thủ công: {exc}") from exc

    if template.get("template_type") != TEMPLATE_TYPE:
        raise ManualTranslationError(
            "Không đúng định dạng template dịch thủ công của VideoTranslator."
        )
    if template.get("schema_version") != TEMPLATE_SCHEMA_VERSION:
        raise ManualTranslationError("Phiên bản template không được hỗ trợ.")
    if template.get("source_fingerprint") != source_fingerprint:
        raise ManualTranslationError(
            "Template được tạo từ transcript khác hoặc transcript nguồn đã thay đổi. "
            "Hãy xuất lại file mẫu từ project hiện tại."
        )

    rows = template.get("segments")
    if not isinstance(rows, list):
        raise ManualTranslationError("Trường segments phải là một danh sách.")
    if len(rows) != len(source_segments):
        raise ManualTranslationError(
            f"Số segment không khớp: file có {len(rows)}, transcript có {len(source_segments)}."
        )

    source_by_id = {int(item["id"]): item for item in source_segments}
    rows_by_id: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ManualTranslationError(f"Segment dòng {index} phải là object JSON.")
        raw_id = row.get("id")
        if isinstance(raw_id, bool):
            raise ManualTranslationError(f"Segment dòng {index} có ID không hợp lệ.")
        try:
            segment_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ManualTranslationError(f"Segment dòng {index} thiếu ID hợp lệ.") from exc
        if segment_id not in source_by_id:
            raise ManualTranslationError(f"File có segment ID không tồn tại: {segment_id}.")
        if segment_id in rows_by_id:
            raise ManualTranslationError(f"File bị trùng segment ID: {segment_id}.")
        rows_by_id[segment_id] = row

    missing = set(source_by_id) - set(rows_by_id)
    if missing:
        raise ManualTranslationError(
            "File còn thiếu segment ID: " + ", ".join(str(value) for value in sorted(missing))
        )

    translated_segments: list[dict[str, Any]] = []
    for source in source_segments:
        segment_id = int(source["id"])
        row = rows_by_id[segment_id]
        source_text = str(source.get("text", ""))
        if row.get("source_text") != source_text:
            raise ManualTranslationError(
                f"source_text của segment {segment_id} đã bị thay đổi. "
                "Không sửa nội dung nguồn; hãy chỉ điền translated_text."
            )

        translated_text = row.get("translated_text")
        if not isinstance(translated_text, str):
            raise ManualTranslationError(
                f"translated_text của segment {segment_id} phải là chuỗi."
            )
        translated_text = translated_text.strip()
        if source_text.strip() and not translated_text:
            raise ManualTranslationError(
                f"Segment {segment_id} chưa có bản dịch. "
                "Với câu được cố ý bỏ qua, hãy kiểm tra lại nội dung nguồn trước khi nhập."
            )
        if not source_text.strip() and translated_text:
            raise ManualTranslationError(
                f"Segment {segment_id} không có lời thoại nguồn nhưng lại có bản dịch."
            )
        if row.get("reviewed") is not True:
            raise ManualTranslationError(
                f"Segment {segment_id} chưa được đánh dấu reviewed: true."
            )

        emotion = row.get("emotion", "neutral")
        delivery = row.get("delivery", "normal")
        if not isinstance(emotion, str) or emotion not in ALLOWED_EMOTIONS:
            raise ManualTranslationError(
                f"emotion của segment {segment_id} không hợp lệ: {emotion!r}."
            )
        if not isinstance(delivery, str) or delivery not in ALLOWED_DELIVERIES:
            raise ManualTranslationError(
                f"delivery của segment {segment_id} không hợp lệ: {delivery!r}."
            )

        translated_segments.append(
            {
                "id": segment_id,
                "start": float(source["start"]),
                "end": float(source["end"]),
                "speaker_id": source.get("speaker_id"),
                "original_text": source_text,
                "translated_text": translated_text,
                "emotion": emotion,
                "delivery": delivery,
            }
        )

    qc = template.get("quality_control")
    if not isinstance(qc, dict) or qc.get("completed") is not True:
        raise ManualTranslationError(
            "Chưa xác nhận kiểm tra chất lượng cuối: quality_control.completed phải là true."
        )
    issues = _validate_issues(qc.get("issues", []), set(source_by_id))

    imported_at = datetime.now(timezone.utc).isoformat()
    review_metadata = {
        "completed": True,
        "reviewed_segments": len(translated_segments),
        "total_segments": len(source_segments),
        "imported_at": imported_at,
        "template_path": str(Path(template_path).expanduser().resolve()),
        "source_fingerprint": source_fingerprint,
    }
    translation_state = {
        "schema_version": 1,
        "source_transcript": str(paths.transcript_json),
        "source_fingerprint": source_fingerprint,
        "translation_source": "manual",
        "model": "manual",
        "config_signature": sha256_text(
            source_fingerprint + json.dumps(translated_segments, ensure_ascii=False, sort_keys=True)
        ),
        "completed_segments": len(translated_segments),
        "total_segments": len(source_segments),
        "is_complete": True,
        "segments": translated_segments,
        "issues": issues,
        "pending_batches": {},
        "manual_review": review_metadata,
    }

    # Validate everything before touching existing project artifacts.
    write_json_atomic(paths.translated_json, translation_state)
    _write_translated_srt(paths.translated_srt, translated_segments)
    write_json_atomic(
        paths.quality_json,
        {
            "schema_version": 1,
            "model": "manual",
            "source": "manual",
            "source_fingerprint": source_fingerprint,
            "translation_fingerprint": sha256_file(paths.translated_json),
            "manual_review_completed": True,
            "manual_review": review_metadata,
            "issues": issues,
            "summary": {
                "reviewed_segments": len(translated_segments),
                "issue_count": len(issues),
            },
        },
    )
    return paths.translated_json, paths.translated_srt, paths.quality_json, len(translated_segments)


def _load_transcript(path: Path) -> dict[str, Any]:
    try:
        transcript = read_json(path)
    except (OSError, ValueError) as exc:
        raise ManualTranslationError(f"Không đọc được transcript nguồn: {exc}") from exc
    if transcript.get("is_complete") is False:
        raise ManualTranslationError(
            "Transcript Whisper chưa hoàn tất; hãy hoàn tất nhận diện trước khi tạo file mẫu."
        )
    return transcript


def _validated_source_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ManualTranslationError("Transcript không có segments để dịch.")

    seen: set[int] = set()
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(segments, start=1):
        if not isinstance(raw, dict):
            raise ManualTranslationError(f"Transcript segment {index} không hợp lệ.")
        raw_id = raw.get("id")
        if isinstance(raw_id, bool):
            raise ManualTranslationError(f"Transcript segment {index} có ID không hợp lệ.")
        try:
            segment_id = int(raw_id)
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManualTranslationError(
                f"Transcript segment {index} thiếu ID/timestamp hợp lệ."
            ) from exc
        if segment_id in seen:
            raise ManualTranslationError(f"Transcript có ID segment bị trùng: {segment_id}.")
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ManualTranslationError(
                f"Timestamp của segment {segment_id} không hợp lệ."
            )
        text_value = raw.get("text", "")
        if not isinstance(text_value, str):
            raise ManualTranslationError(
                f"Nội dung nguồn của segment {segment_id} không phải chuỗi."
            )
        seen.add(segment_id)
        result.append(raw)
    return result


def _validate_issues(issues: Any, valid_ids: set[int]) -> list[dict[str, Any]]:
    if not isinstance(issues, list):
        raise ManualTranslationError("quality_control.issues phải là một danh sách.")
    validated: list[dict[str, Any]] = []
    for index, issue in enumerate(issues, start=1):
        if not isinstance(issue, dict):
            raise ManualTranslationError(f"QC issue {index} phải là object JSON.")
        raw_id = issue.get("segment_id", issue.get("id"))
        if isinstance(raw_id, bool):
            raise ManualTranslationError(f"QC issue {index} có segment_id không hợp lệ.")
        try:
            segment_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ManualTranslationError(f"QC issue {index} thiếu segment_id hợp lệ.") from exc
        if segment_id not in valid_ids:
            raise ManualTranslationError(
                f"QC issue {index} tham chiếu segment không tồn tại: {segment_id}."
            )
        severity = issue.get("severity", "high")
        category = issue.get("category", "uncertainty")
        if not isinstance(severity, str) or severity not in ALLOWED_ISSUE_SEVERITIES:
            raise ManualTranslationError(
                f"severity của QC issue {index} phải là critical hoặc high."
            )
        if not isinstance(category, str) or category not in ALLOWED_ISSUE_CATEGORIES:
            raise ManualTranslationError(f"category của QC issue {index} không hợp lệ.")
        item: dict[str, Any] = {
            "segment_id": segment_id,
            "severity": severity,
            "category": category,
        }
        note = issue.get("note")
        if note is not None:
            if not isinstance(note, str):
                raise ManualTranslationError(f"note của QC issue {index} phải là chuỗi.")
            item["note"] = note.strip()
        validated.append(item)
    return validated


def _write_translated_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    blocks: list[str] = []
    cue = 0
    for segment in segments:
        text_value = str(segment.get("translated_text", "")).strip()
        if not text_value:
            continue
        cue += 1
        blocks.append(
            f"{cue}\n"
            f"{format_srt_timestamp(float(segment['start']))} --> "
            f"{format_srt_timestamp(float(segment['end']))}\n"
            f"{text_value}"
        )
    content = "\n\n".join(blocks) + ("\n" if blocks else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
