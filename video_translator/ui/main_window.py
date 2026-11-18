from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from video_translator.config.settings import AppSettings
from video_translator.core.models import DownloadProgress, TranscriptionProgress
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.ui.download_worker import DownloadWorker
from video_translator.ui.gemini_worker import GeminiWorker
from video_translator.ui.transcription_worker import TranscriptionWorker
from video_translator.ui.vieneu_worker import TTSWorker, VoiceListWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self.worker: (
            DownloadWorker | TranscriptionWorker | GeminiWorker | TTSWorker | None
        ) = None
        self.voice_worker: VoiceListWorker | None = None
        self.context_path: Path | None = None
        self._busy = False
        self.setWindowTitle("VideoTranslator — Vietnamese Film Dubbing")
        self.resize(1040, 900)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(10)

        title = QLabel("VIDEO TRANSLATOR")
        title.setObjectName("title")
        subtitle = QLabel(
            "Dịch thoại phim theo ngữ cảnh, tạo giọng thuyết minh tiếng Việt."
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self._build_source_tab()
        self._build_gemini_tab()
        self._build_vieneu_tab()
        layout.addWidget(self.tabs, 1)

        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        actions = QHBoxLayout()
        log_title = QLabel("NHẬT KÝ")
        log_title.setObjectName("sectionTitle")
        actions.addWidget(log_title)
        actions.addStretch(1)
        self.cancel_button = QPushButton("Hủy tác vụ")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_task)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        self.log_view.setPlaceholderText("Tiến trình và lỗi sẽ hiển thị tại đây...")
        self.log_view.setMaximumHeight(190)
        layout.addWidget(self.log_view)

        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #111827;
                color: #E5E7EB;
                font-family: 'Segoe UI', sans-serif;
                font-size: 10pt;
            }
            QLabel#title {
                color: #F9FAFB;
                font-size: 22pt;
                font-weight: 700;
            }
            QLabel#subtitle {
                color: #9CA3AF;
                font-size: 10pt;
                padding-bottom: 5px;
            }
            QLabel#sectionTitle {
                color: #D1D5DB;
                font-weight: 700;
            }
            QLabel#status {
                color: #93C5FD;
                font-weight: 600;
            }
            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #1F2937;
                color: #F9FAFB;
                border: 1px solid #374151;
                border-radius: 7px;
                padding: 8px;
                selection-background-color: #2563EB;
            }
            QComboBox QAbstractItemView {
                background: #1F2937;
                color: #F9FAFB;
                selection-background-color: #2563EB;
            }
            QTabWidget::pane {
                border: 1px solid #374151;
                border-radius: 8px;
                top: -1px;
            }
            QTabBar::tab {
                background: #1F2937;
                color: #9CA3AF;
                padding: 10px 18px;
                margin-right: 4px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
            }
            QTabBar::tab:selected {
                background: #2563EB;
                color: #FFFFFF;
            }
            QPushButton {
                background: #374151;
                color: #F9FAFB;
                border: 0;
                border-radius: 7px;
                padding: 9px 14px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: #4B5563;
            }
            QPushButton:disabled {
                color: #6B7280;
                background: #1F2937;
            }
            QPushButton#primaryButton {
                background: #2563EB;
                font-weight: 700;
            }
            QPushButton#primaryButton:hover {
                background: #1D4ED8;
            }
            QProgressBar {
                background: #1F2937;
                border: 1px solid #374151;
                border-radius: 6px;
                color: #F9FAFB;
                text-align: center;
                min-height: 18px;
            }
            QProgressBar::chunk {
                background: #2563EB;
                border-radius: 5px;
            }
            """
        )
        self._update_action_states()

    def _build_source_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        form = QFormLayout()
        form.setSpacing(11)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Dán link YouTube hoặc Bilibili...")
        form.addRow("Video URL", self.url_input)

        self.output_input = QLineEdit(str(self.settings.output_dir))
        self.output_browse_button = QPushButton("Chọn thư mục")
        self.output_browse_button.clicked.connect(self._browse_output_dir)
        form.addRow("Thư mục dự án", self._row(self.output_input, self.output_browse_button))

        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("Chọn video đã tải hoặc video có sẵn...")
        self.video_browse_button = QPushButton("Chọn video")
        self.video_browse_button.clicked.connect(self._browse_video)
        form.addRow("Video nguồn", self._row(self.video_input, self.video_browse_button))

        self.transcript_input = QLineEdit()
        self.transcript_input.setPlaceholderText("Tự điền sau khi Whisper hoàn tất...")
        self.transcript_browse_button = QPushButton("Chọn JSON")
        self.transcript_browse_button.clicked.connect(self._browse_transcript)
        form.addRow(
            "Transcript JSON",
            self._row(self.transcript_input, self.transcript_browse_button),
        )

        whisper_row = QHBoxLayout()
        self.model_combo = QComboBox()
        for label, value in (
            ("Tiny — nhanh", "tiny"),
            ("Base — nhẹ", "base"),
            ("Small — khuyến nghị", "small"),
            ("Medium — chính xác hơn", "medium"),
            ("Large-v3 — nặng", "large-v3"),
        ):
            self.model_combo.addItem(label, value)
        self.model_combo.setCurrentIndex(2)

        self.language_combo = QComboBox()
        for label, value in (
            ("Tự nhận diện", "auto"),
            ("English", "en"),
            ("Chinese", "zh"),
            ("Japanese", "ja"),
            ("Korean", "ko"),
            ("Vietnamese", "vi"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("German", "de"),
            ("Russian", "ru"),
        ):
            self.language_combo.addItem(label, value)
        whisper_row.addWidget(self.model_combo, 2)
        whisper_row.addWidget(self.language_combo, 1)
        form.addRow("Whisper / ngôn ngữ", whisper_row)

        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.download_button = QPushButton("Tải MP4 tối đa 1080p")
        self.download_button.setObjectName("primaryButton")
        self.download_button.clicked.connect(self._start_download)
        self.transcribe_button = QPushButton("Nhận diện lời thoại")
        self.transcribe_button.clicked.connect(self._start_transcription)
        buttons.addWidget(self.download_button)
        buttons.addWidget(self.transcribe_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.source_hint = QLabel(
            "Whisper tạo transcript gốc có timestamp; file JSON là dữ liệu đầu vào cho Gemini."
        )
        self.source_hint.setWordWrap(True)
        self.source_hint.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(self.source_hint)
        layout.addStretch(1)
        self.tabs.addTab(tab, "1 · Video & Transcript")

    def _build_gemini_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        form = QFormLayout()
        form.setSpacing(11)

        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key_input.setPlaceholderText("Để trống để dùng GEMINI_API_KEY trong .env")
        form.addRow("Gemini API key", self.gemini_key_input)

        self.gemini_model_combo = QComboBox()
        for label, value in (
            ("Gemini 3.6 Flash — khuyến nghị", "gemini-3.6-flash"),
            ("Gemini 3.5 Flash", "gemini-3.5-flash"),
            ("Gemini 2.5 Flash", "gemini-2.5-flash"),
            ("Gemini 2.5 Pro", "gemini-2.5-pro"),
        ):
            self.gemini_model_combo.addItem(label, value)
        selected_index = self.gemini_model_combo.findData(self.settings.gemini_model)
        if selected_index >= 0:
            self.gemini_model_combo.setCurrentIndex(selected_index)
        form.addRow("Gemini model", self.gemini_model_combo)

        self.translation_batch_spin = QSpinBox()
        self.translation_batch_spin.setRange(5, 100)
        self.translation_batch_spin.setValue(30)
        form.addRow("Segment mỗi batch dịch", self.translation_batch_spin)

        self.context_status = QLabel("Context Profile: chưa phân tích")
        self.context_status.setWordWrap(True)
        self.context_status.setStyleSheet("color: #9CA3AF;")

        layout.addLayout(form)
        layout.addWidget(self.context_status)

        buttons = QHBoxLayout()
        self.context_button = QPushButton("Phân tích toàn phim")
        self.context_button.setObjectName("primaryButton")
        self.context_button.clicked.connect(lambda: self._start_gemini("context"))
        self.translate_button = QPushButton("Dịch phim + kiểm tra")
        self.translate_button.clicked.connect(lambda: self._start_gemini("translate"))
        buttons.addWidget(self.context_button)
        buttons.addWidget(self.translate_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        hint = QLabel(
            "Gemini phân tích transcript theo chapter, lưu hồ sơ nhân vật/quan hệ/thuật ngữ, "
            "sau đó dịch theo batch và QC từng batch. Có thể chạy lại để tiếp tục phần chưa xong."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(hint)
        layout.addStretch(1)
        self.tabs.addTab(tab, "2 · Gemini Dịch phim")

    def _build_vieneu_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        form = QFormLayout()
        form.setSpacing(11)

        self.translation_input = QLineEdit()
        self.translation_input.setPlaceholderText("Tự điền sau khi Gemini dịch...")
        self.translation_browse_button = QPushButton("Chọn JSON")
        self.translation_browse_button.clicked.connect(self._browse_translation)
        form.addRow(
            "Bản dịch JSON",
            self._row(self.translation_input, self.translation_browse_button),
        )

        self.vieneu_key_input = QLineEdit()
        self.vieneu_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.vieneu_key_input.setPlaceholderText("Để trống để dùng VIENEU_API_KEY trong .env")
        form.addRow("VieNeu API key", self.vieneu_key_input)

        voice_row = QHBoxLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(300)
        self.voice_refresh_button = QPushButton("Tải danh sách giọng")
        self.voice_refresh_button.clicked.connect(self._refresh_voices)
        voice_row.addWidget(self.voice_combo, 1)
        voice_row.addWidget(self.voice_refresh_button)
        form.addRow("Giọng đọc (engine v4)", voice_row)

        self.voice_speed_spin = QDoubleSpinBox()
        self.voice_speed_spin.setRange(0.5, 3.0)
        self.voice_speed_spin.setSingleStep(0.1)
        self.voice_speed_spin.setDecimals(1)
        self.voice_speed_spin.setValue(1.0)
        form.addRow("Tốc độ đọc", self.voice_speed_spin)

        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.tts_button = QPushButton("Tạo giọng đọc theo từng câu")
        self.tts_button.setObjectName("primaryButton")
        self.tts_button.clicked.connect(self._start_tts)
        buttons.addWidget(self.tts_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        hint = QLabel(
            "Mỗi câu thoại được tạo thành một audio riêng để đồng bộ timeline ở v0.6. "
            "Job và file đã hoàn thành được lưu manifest; chạy lại sẽ bỏ qua audio đã có."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(hint)
        layout.addStretch(1)
        self.tabs.addTab(tab, "3 · VieNeu Thuyết minh")

    @staticmethod
    def _row(widget: QWidget, button: QPushButton) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget, 1)
        row.addWidget(button)
        return container

    def _browse_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Chọn thư mục dự án",
            self.output_input.text().strip() or str(Path.home()),
        )
        if selected:
            self.output_input.setText(selected)

    def _browse_video(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video nguồn",
            str(Path.home()),
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All files (*)",
        )
        if selected:
            self.video_input.setText(selected)
            self._update_action_states()

    def _browse_transcript(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn transcript JSON",
            self.output_input.text().strip() or str(Path.home()),
            "Transcript JSON (*.json);;All files (*)",
        )
        if selected:
            self.transcript_input.setText(selected)
            self._refresh_context_status()
            self._update_action_states()

    def _browse_translation(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn bản dịch JSON",
            self.output_input.text().strip() or str(Path.home()),
            "Translation JSON (*.json);;All files (*)",
        )
        if selected:
            self.translation_input.setText(selected)
            self._update_action_states()

    def _start_download(self) -> None:
        url = self.url_input.text().strip()
        output_text = self.output_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Thiếu URL", "Vui lòng nhập link video.")
            return
        if not output_text:
            QMessageBox.warning(self, "Thiếu thư mục", "Vui lòng chọn thư mục dự án.")
            return

        self._begin_task("Đang chuẩn bị tải video...")
        self.worker = DownloadWorker(
            url=url,
            output_dir=Path(output_text).expanduser(),
            max_height=self.settings.max_height,
        )
        self._connect_common_worker(self.worker)
        self.worker.succeeded.connect(self._on_download_success)
        self.worker.start()

    def _start_transcription(self) -> None:
        video_path = Path(self.video_input.text().strip()).expanduser()
        output_text = self.output_input.text().strip()
        if not video_path.is_file():
            QMessageBox.warning(self, "Thiếu video", "Hãy chọn một file video hợp lệ.")
            return
        if not output_text:
            QMessageBox.warning(self, "Thiếu thư mục", "Vui lòng chọn thư mục dự án.")
            return

        language = self.language_combo.currentData()
        if language == "auto":
            language = None

        self._begin_task("Đang chuẩn bị nhận diện lời thoại...")
        self.worker = TranscriptionWorker(
            video_path=video_path,
            output_dir=Path(output_text).expanduser(),
            model_size=str(self.model_combo.currentData()),
            language=language,
        )
        self._connect_common_worker(self.worker)
        self.worker.succeeded.connect(self._on_transcription_success)
        self.worker.start()

    def _start_gemini(self, mode: str) -> None:
        transcript_path = Path(self.transcript_input.text().strip()).expanduser()
        if not transcript_path.is_file():
            QMessageBox.warning(
                self, "Thiếu transcript", "Hãy nhận diện lời thoại hoặc chọn transcript JSON."
            )
            return
        api_key = self.gemini_key_input.text().strip() or os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            QMessageBox.warning(
                self, "Thiếu Gemini API key",
                "Nhập API key tại đây hoặc cấu hình GEMINI_API_KEY trong file .env.",
            )
            return

        paths = MovieProjectPaths.from_transcript(transcript_path)
        context_path = paths.context_json
        if mode == "translate" and not context_path.is_file():
            QMessageBox.warning(
                self,
                "Chưa có Context Profile",
                "Hãy chạy Phân tích toàn phim trước khi dịch.",
            )
            return

        self._begin_task(
            "Gemini đang phân tích toàn phim..." if mode == "context"
            else "Gemini đang dịch và kiểm tra lời thoại..."
        )
        self.worker = GeminiWorker(
            mode=mode,
            api_key=api_key,
            transcript_path=transcript_path,
            context_path=context_path,
            model=str(self.gemini_model_combo.currentData()),
            batch_size=self.translation_batch_spin.value(),
        )
        self._connect_common_worker(self.worker)
        self.worker.context_succeeded.connect(self._on_context_success)
        self.worker.translation_succeeded.connect(self._on_translation_success)
        self.worker.start()

    def _refresh_voices(self) -> None:
        if self.voice_worker and self.voice_worker.isRunning():
            return
        api_key = self.vieneu_key_input.text().strip() or os.getenv("VIENEU_API_KEY", "").strip()
        self.voice_refresh_button.setEnabled(False)
        self._append_log("Đang tải danh sách VieNeu v4...")
        self.voice_worker = VoiceListWorker(api_key)
        self.voice_worker.loaded.connect(self._on_voices_loaded)
        self.voice_worker.failed.connect(self._on_voice_list_failed)
        self.voice_worker.finished.connect(
            lambda: self.voice_refresh_button.setEnabled(not self._busy)
        )
        self.voice_worker.start()

    def _start_tts(self) -> None:
        translated_path = Path(self.translation_input.text().strip()).expanduser()
        if not translated_path.is_file():
            QMessageBox.warning(self, "Thiếu bản dịch", "Hãy chọn file bản dịch JSON.")
            return
        api_key = self.vieneu_key_input.text().strip() or os.getenv("VIENEU_API_KEY", "").strip()
        if not api_key:
            QMessageBox.warning(
                self, "Thiếu VieNeu API key",
                "Nhập API key tại đây hoặc cấu hình VIENEU_API_KEY trong file .env.",
            )
            return
        voice_id = self.voice_combo.currentData()
        if not voice_id:
            QMessageBox.warning(self, "Chưa chọn giọng", "Hãy tải danh sách và chọn giọng VieNeu.")
            return

        try:
            payload = json.loads(translated_path.read_text(encoding="utf-8"))
            segments = payload.get("segments", [])
            pending = sum(
                1 for item in segments
                if isinstance(item, dict) and str(item.get("translated_text", "")).strip()
            )
        except (OSError, ValueError, AttributeError) as exc:
            QMessageBox.critical(self, "Bản dịch không hợp lệ", str(exc))
            return

        answer = QMessageBox.question(
            self,
            "Xác nhận tạo giọng",
            f"Sẽ tổng hợp tối đa {pending} đoạn thoại bằng VieNeu.\n"
            "Mỗi đoạn là một lượt API có tính phí theo ký tự. "
            "Gói dùng thử có thể giới hạn số lượt mỗi ngày.\n\n"
            "Các đoạn đã có audio sẽ được bỏ qua khi chạy lại. Tiếp tục?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._begin_task("VieNeu đang tạo giọng đọc...")
        self.worker = TTSWorker(
            api_key=api_key,
            translated_json=translated_path,
            voice_id=str(voice_id),
            speed=float(self.voice_speed_spin.value()),
        )
        self._connect_common_worker(self.worker)
        self.worker.succeeded.connect(self._on_tts_success)
        self.worker.start()

    def _connect_common_worker(self, worker: QThread) -> None:
        if hasattr(worker, "progress_changed"):
            worker.progress_changed.connect(self._on_progress)
        if hasattr(worker, "log_message"):
            worker.log_message.connect(self._append_log)
        if hasattr(worker, "failed"):
            worker.failed.connect(self._on_failure)
        if hasattr(worker, "cancelled"):
            worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(self._on_worker_finished)

    def _begin_task(self, message: str) -> None:
        self.log_view.clear()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self.status_label.setText(message)

    def _cancel_task(self) -> None:
        if self.worker and self.worker.isRunning():
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Đang yêu cầu hủy...")
            self.worker.cancel()

    def _on_progress(self, progress: DownloadProgress | TranscriptionProgress) -> None:
        self.status_label.setText(progress.message)
        if progress.percent is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, round(progress.percent))))

    def _on_download_success(self, output_path: str) -> None:
        self.video_input.setText(output_path)
        self.status_label.setText("Tải video hoàn tất.")
        self._append_log(f"Video nguồn: {output_path}")
        self._update_action_states()
        QMessageBox.information(self, "Hoàn tất", f"Đã tải video:\n{output_path}")

    def _on_transcription_success(
        self,
        json_path: str,
        srt_path: str,
        language: str,
        segment_count: int,
    ) -> None:
        self.transcript_input.setText(json_path)
        self.status_label.setText(
            f"Whisper hoàn tất: {segment_count} đoạn | ngôn ngữ: {language}"
        )
        self._append_log(f"Transcript JSON: {json_path}")
        self._append_log(f"Source SRT: {srt_path}")
        self._refresh_context_status()
        self._update_action_states()
        QMessageBox.information(
            self,
            "Whisper hoàn tất",
            f"Đã nhận diện {segment_count} đoạn.\n\nJSON: {json_path}\nSRT: {srt_path}",
        )

    def _on_context_success(self, context_path: str) -> None:
        self.context_path = Path(context_path)
        self.context_status.setText(f"Context Profile: {context_path}")
        self.status_label.setText("Phân tích ngữ cảnh phim hoàn tất.")
        self._append_log(f"Context Profile: {context_path}")
        QMessageBox.information(self, "Gemini hoàn tất", f"Đã lưu Context Profile:\n{context_path}")

    def _on_translation_success(self, json_path: str, srt_path: str, quality_path: str) -> None:
        self.translation_input.setText(json_path)
        self.status_label.setText("Dịch phim và kiểm tra chất lượng hoàn tất.")
        self._append_log(f"Vietnamese JSON: {json_path}")
        self._append_log(f"Vietnamese SRT: {srt_path}")
        self._append_log(f"QC report: {quality_path}")
        QMessageBox.information(
            self,
            "Dịch phim hoàn tất",
            f"JSON: {json_path}\nSRT: {srt_path}\nQC: {quality_path}",
        )

    def _on_voices_loaded(self, voices: list[dict]) -> None:
        selected = self.voice_combo.currentData() or self.settings.vieneu_voice
        self.voice_combo.clear()
        for voice in voices:
            label = voice["name"]
            description = voice.get("description", "")
            if description:
                label = f"{label} — {description}"
            if voice.get("kind") == "cloned":
                label = f"{label} [Clone]"
            self.voice_combo.addItem(label, voice["id"])
        selected_index = self.voice_combo.findData(selected)
        if selected_index >= 0:
            self.voice_combo.setCurrentIndex(selected_index)
        self._append_log(f"Đã tải {len(voices)} giọng VieNeu v4.")
        if not voices:
            QMessageBox.warning(self, "VieNeu", "Danh sách giọng hiện đang trống.")

    def _on_voice_list_failed(self, message: str) -> None:
        self._append_log(f"LỖI tải danh sách giọng: {message}")
        QMessageBox.critical(self, "Không tải được danh sách giọng", message)

    def _on_tts_success(self, manifest_path: str, completed: int, total: int) -> None:
        self.status_label.setText(f"VieNeu hoàn tất: {completed}/{total} đoạn.")
        self._append_log(f"TTS manifest: {manifest_path}")
        QMessageBox.information(
            self,
            "VieNeu hoàn tất",
            f"Đã có audio cho {completed}/{total} đoạn.\n\nManifest: {manifest_path}",
        )

    def _on_failure(self, message: str) -> None:
        self.status_label.setText("Tác vụ thất bại.")
        self._append_log(f"LỖI: {message}")
        QMessageBox.critical(self, "Không thể hoàn thành tác vụ", message)

    def _on_cancelled(self) -> None:
        self.status_label.setText("Đã hủy tác vụ.")
        self._append_log("Tác vụ đã được hủy. Dữ liệu checkpoint vẫn được giữ.")

    def _on_worker_finished(self) -> None:
        self.worker = None
        self._set_busy(False)
        self._refresh_context_status()
        self._update_action_states()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for widget in (
            self.url_input,
            self.output_input,
            self.output_browse_button,
            self.video_input,
            self.video_browse_button,
            self.transcript_input,
            self.transcript_browse_button,
            self.model_combo,
            self.language_combo,
            self.download_button,
            self.transcribe_button,
            self.gemini_key_input,
            self.gemini_model_combo,
            self.translation_batch_spin,
            self.context_button,
            self.translate_button,
            self.translation_input,
            self.translation_browse_button,
            self.vieneu_key_input,
            self.voice_combo,
            self.voice_refresh_button,
            self.voice_speed_spin,
            self.tts_button,
        ):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _refresh_context_status(self) -> None:
        transcript = Path(self.transcript_input.text().strip()).expanduser()
        if transcript.is_file():
            paths = MovieProjectPaths.from_transcript(transcript)
            if paths.context_json.is_file():
                self.context_path = paths.context_json
                try:
                    payload = json.loads(paths.context_json.read_text(encoding="utf-8"))
                    count = payload.get("processed_segments", 0)
                    total = payload.get("total_segments", 0)
                    complete = payload.get("is_complete", False)
                    self.context_status.setText(
                        f"Context Profile: {'hoàn tất' if complete else 'đang dở'} "
                        f"({count}/{total}) — {paths.context_json}"
                    )
                except (OSError, ValueError):
                    self.context_status.setText(f"Context Profile: {paths.context_json}")
                return
        self.context_path = None
        self.context_status.setText("Context Profile: chưa phân tích")

    def _update_action_states(self) -> None:
        video_path = Path(self.video_input.text().strip()).expanduser()
        transcript_path = Path(self.transcript_input.text().strip()).expanduser()
        translated_path = Path(self.translation_input.text().strip()).expanduser()
        has_transcript = transcript_path.is_file()
        has_context = (
            self.context_path is not None and self.context_path.is_file()
        )
        if has_transcript and not has_context:
            has_context = MovieProjectPaths.from_transcript(transcript_path).context_json.is_file()
        self.transcribe_button.setEnabled(not self._busy and video_path.is_file())
        self.context_button.setEnabled(not self._busy and has_transcript)
        self.translate_button.setEnabled(not self._busy and has_transcript and has_context)
        self.tts_button.setEnabled(
            not self._busy and translated_path.is_file() and bool(self.voice_combo.currentData())
        )

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
