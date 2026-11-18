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
from video_translator.core.project_manager import ProjectManager
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.ui.download_worker import DownloadWorker
from video_translator.ui.gemini_worker import GeminiWorker
from video_translator.ui.pipeline_worker import PipelineWorker
from video_translator.ui.render_worker import RenderWorker
from video_translator.ui.transcription_worker import TranscriptionWorker
from video_translator.ui.vieneu_worker import TTSWorker, VoiceListWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self.worker: (
            DownloadWorker
            | TranscriptionWorker
            | GeminiWorker
            | TTSWorker
            | RenderWorker
            | PipelineWorker
            | None
        ) = None
        self.voice_worker: VoiceListWorker | None = None
        self.context_path: Path | None = None
        self.project_manager: ProjectManager | None = None
        self._context_ready = False
        self._busy = False
        self._active_stage: str | None = None
        self._last_project_progress = -1
        self.setWindowTitle("VideoTranslator — Vietnamese Film Dubbing Studio")
        self.resize(1080, 940)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(10)

        title = QLabel("VIDEO TRANSLATOR")
        title.setObjectName("title")
        subtitle = QLabel(
            "Dịch thoại phim theo ngữ cảnh · Thuyết minh VieNeu · Phụ đề Việt · MP4"
        )
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self._build_source_tab()
        self._build_gemini_tab()
        self._build_vieneu_tab()
        self._build_export_tab()
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
        self.log_view.setMaximumBlockCount(5000)
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
                padding: 10px 15px;
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
        self.transcript_input.setPlaceholderText("Chọn hoặc tạo transcript JSON...")
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

        hint = QLabel(
            "Whisper tạo transcript gốc có timestamp. Chọn lại transcript cũ để khôi phục "
            "project và tiếp tục từ các checkpoint đã lưu."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(hint)
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
            ("Gemini 3.8 Flash — khuyến nghị", "gemini-3.8-flash"),
            ("Gemini 3.6 Flash", "gemini-3.6-flash"),
            ("Gemini 3.5 Flash", "gemini-3.5-flash"),
            ("Gemini 3.5 Flash-Lite", "gemini-3.5-flash-lite"),
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
            "Checkpoint theo chapter và batch. Khi Gemini lỗi hoặc ứng dụng bị tắt, "
            "chạy lại cùng project sẽ dùng phần đã lưu; bản dịch đã trả về được lưu "
            "ngay trước bước QC để tránh gọi lại không cần thiết."
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
        self.tts_button = QPushButton("Tạo / tiếp tục giọng đọc theo câu")
        self.tts_button.setObjectName("primaryButton")
        self.tts_button.clicked.connect(self._start_tts)
        layout.addWidget(self.tts_button)

        hint = QLabel(
            "Mỗi câu có job ID, idempotency key và file cache riêng. Nếu ứng dụng dừng "
            "sau khi VieNeu nhận job, lần chạy sau sẽ poll lại job cũ thay vì gửi job mới."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(hint)
        layout.addStretch(1)
        self.tabs.addTab(tab, "3 · VieNeu Thuyết minh")

    def _build_export_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)

        project_form = QFormLayout()
        self.project_transcript_input = QLineEdit()
        self.project_transcript_input.setPlaceholderText("Chọn transcript để khôi phục project...")
        self.project_browse_button = QPushButton("Mở project")
        self.project_browse_button.clicked.connect(self._browse_project_transcript)
        project_form.addRow(
            "Project source",
            self._row(self.project_transcript_input, self.project_browse_button),
        )
        layout.addLayout(project_form)

        self.project_status_label = QLabel("Chưa nạp project")
        self.project_status_label.setWordWrap(True)
        self.project_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.project_status_label.setStyleSheet(
            "background: #1F2937; border: 1px solid #374151; "
            "border-radius: 7px; padding: 10px; color: #D1D5DB;"
        )
        layout.addWidget(self.project_status_label)

        project_buttons = QHBoxLayout()
        self.project_refresh_button = QPushButton("Kiểm tra checkpoint")
        self.project_refresh_button.clicked.connect(self._refresh_project)
        self.resume_button = QPushButton("Chạy toàn bộ / Tiếp tục project")
        self.resume_button.setObjectName("primaryButton")
        self.resume_button.clicked.connect(self._resume_project)
        project_buttons.addWidget(self.project_refresh_button)
        project_buttons.addWidget(self.resume_button)
        project_buttons.addStretch(1)
        layout.addLayout(project_buttons)

        layout.addSpacing(10)
        render_form = QFormLayout()
        self.render_translation_input = QLineEdit()
        self.render_translation_input.setPlaceholderText("Bản dịch JSON đã hoàn thành...")
        self.render_browse_button = QPushButton("Chọn JSON")
        self.render_browse_button.clicked.connect(self._browse_render_translation)
        render_form.addRow(
            "Bản dịch",
            self._row(self.render_translation_input, self.render_browse_button),
        )

        self.subtitle_mode_combo = QComboBox()
        self.subtitle_mode_combo.addItem("Cứng + mềm (khuyến nghị)", "both")
        self.subtitle_mode_combo.addItem("Chỉ phụ đề cứng", "hard")
        self.subtitle_mode_combo.addItem("Chỉ phụ đề mềm", "soft")
        self.subtitle_mode_combo.addItem("Không gắn phụ đề vào MP4", "none")
        render_form.addRow("Phụ đề", self.subtitle_mode_combo)

        self.original_audio_combo = QComboBox()
        self.original_audio_combo.addItem("Giảm âm nền gốc", "duck")
        self.original_audio_combo.addItem("Giữ nguyên âm gốc", "mix")
        self.original_audio_combo.addItem("Tắt âm gốc", "mute")
        render_form.addRow("Âm thanh gốc", self.original_audio_combo)
        layout.addLayout(render_form)

        self.render_button = QPushButton("Render video MP4 tiếng Việt")
        self.render_button.setObjectName("primaryButton")
        self.render_button.clicked.connect(self._start_render)
        layout.addWidget(self.render_button)

        render_hint = QLabel(
            "Render đặt từng câu thuyết minh vào timeline gốc, trộn âm nền, xuất H.264/AAC "
            "và SRT riêng. Phụ đề cứng cần FFmpeg build có libass; phụ đề mềm dùng MP4 mov_text."
        )
        render_hint.setWordWrap(True)
        render_hint.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(render_hint)
        layout.addStretch(1)
        self.tabs.addTab(tab, "4 · Project & Xuất MP4")

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
            self, "Chọn thư mục dự án",
            self.output_input.text().strip() or str(Path.home()),
        )
        if selected:
            self.output_input.setText(selected)

    def _browse_video(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Chọn video nguồn", str(Path.home()),
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All files (*)",
        )
        if selected:
            self.video_input.setText(selected)
            self._update_action_states()

    def _browse_transcript(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Chọn transcript JSON",
            self.output_input.text().strip() or str(Path.home()),
            "Transcript JSON (*.json);;All files (*)",
        )
        if selected:
            self.transcript_input.setText(selected)
            self._set_project_source(Path(selected))
            self._refresh_context_status()
            self._update_action_states()

    def _browse_project_transcript(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Mở transcript của project",
            self.project_transcript_input.text().strip() or str(Path.home()),
            "Transcript JSON (*.json);;All files (*)",
        )
        if selected:
            self.project_transcript_input.setText(selected)
            self.transcript_input.setText(selected)
            self._set_project_source(Path(selected))
            self._refresh_context_status()
            self._update_action_states()

    def _browse_translation(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Chọn bản dịch JSON",
            self.output_input.text().strip() or str(Path.home()),
            "Translation JSON (*.json);;All files (*)",
        )
        if selected:
            self.translation_input.setText(selected)
            self.render_translation_input.setText(selected)
            self._set_project_from_translation(Path(selected))
            self._update_action_states()

    def _browse_render_translation(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Chọn bản dịch JSON",
            self.render_translation_input.text().strip() or str(Path.home()),
            "Translation JSON (*.json);;All files (*)",
        )
        if selected:
            self.render_translation_input.setText(selected)
            self.translation_input.setText(selected)
            self._set_project_from_translation(Path(selected))
            self._update_action_states()

    def _start_download(self) -> None:
        url = self.url_input.text().strip()
        output_text = self.output_input.text().strip()
        if not url or not output_text:
            QMessageBox.warning(self, "Thiếu thông tin", "Nhập URL và thư mục dự án.")
            return
        self._begin_task("Đang tải video...")
        self._active_stage = None
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
        if not video_path.is_file() or not output_text:
            QMessageBox.warning(self, "Thiếu video", "Chọn video nguồn và thư mục dự án.")
            return
        language = self.language_combo.currentData()
        if language == "auto":
            language = None
        self._begin_task("Đang nhận diện lời thoại...")
        self._active_stage = "transcription"
        self._mark_stage_running("transcription")
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
                "Nhập API key tại đây hoặc cấu hình GEMINI_API_KEY trong .env.",
            )
            return

        paths = MovieProjectPaths.from_transcript(transcript_path)
        if mode == "translate" and not paths.context_json.is_file():
            QMessageBox.warning(
                self, "Chưa có Context Profile",
                "Hãy chạy Phân tích toàn phim trước khi dịch.",
            )
            return

        stage = "context" if mode == "context" else "translation"
        self._begin_task(
            "Gemini đang phân tích toàn phim..." if mode == "context"
            else "Gemini đang dịch và kiểm tra lời thoại..."
        )
        self._active_stage = stage
        self._mark_stage_running(stage)
        self.worker = GeminiWorker(
            mode=mode,
            api_key=api_key,
            transcript_path=transcript_path,
            context_path=paths.context_json,
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
                "Nhập API key tại đây hoặc cấu hình VIENEU_API_KEY trong .env.",
            )
            return
        voice_id = self.voice_combo.currentData()
        if not voice_id:
            QMessageBox.warning(self, "Chưa chọn giọng", "Hãy tải danh sách và chọn giọng VieNeu.")
            self.tabs.setCurrentIndex(2)
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
            f"Sẽ kiểm tra/tổng hợp tối đa {pending} đoạn thoại bằng VieNeu.\n"
            "Các đoạn đã có audio sẽ không gọi API lại. Job đã nhận sẽ được resume bằng ID cũ.\n"
            "Mỗi đoạn mới có thể bị tính phí theo ký tự. Tiếp tục?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._set_project_from_translation(translated_path)
        self._begin_task("VieNeu đang tạo / tiếp tục giọng đọc...")
        self._active_stage = "tts"
        self._mark_stage_running("tts")
        self.worker = TTSWorker(
            api_key=api_key,
            translated_json=translated_path,
            voice_id=str(voice_id),
            speed=float(self.voice_speed_spin.value()),
        )
        self._connect_common_worker(self.worker)
        self.worker.succeeded.connect(self._on_tts_success)
        self.worker.start()

    def _start_render(self) -> None:
        translated_path = Path(self.render_translation_input.text().strip()).expanduser()
        if not translated_path.is_file():
            QMessageBox.warning(self, "Thiếu bản dịch", "Chọn bản dịch JSON hoàn chỉnh.")
            return
        self._set_project_from_translation(translated_path)
        paths = self._paths_for_translation(translated_path)
        if not paths.tts_manifest.is_file():
            QMessageBox.warning(self, "Thiếu audio", "Hãy tạo giọng đọc VieNeu trước.")
            return
        self._begin_task("Đang ghép timeline và render MP4...")
        self._active_stage = "render"
        self._mark_stage_running("render")
        self.worker = RenderWorker(
            translated_json=translated_path,
            subtitle_mode=str(self.subtitle_mode_combo.currentData()),
            original_audio_mode=str(self.original_audio_combo.currentData()),
        )
        self._connect_common_worker(self.worker)
        self.worker.succeeded.connect(self._on_render_success)
        self.worker.start()

    def _resume_project(self) -> None:
        self._start_full_pipeline()

    def _start_full_pipeline(self) -> None:
        output_text = self.output_input.text().strip()
        if not output_text:
            QMessageBox.warning(self, "Thiếu thư mục", "Hãy chọn thư mục dự án.")
            self.tabs.setCurrentIndex(0)
            return

        transcript_text = (
            self.project_transcript_input.text().strip()
            or self.transcript_input.text().strip()
        )
        transcript_path = Path(transcript_text).expanduser() if transcript_text else None
        if transcript_path is not None and not transcript_path.is_file():
            transcript_path = None

        video_text = self.video_input.text().strip()
        video_path = Path(video_text).expanduser() if video_text else None
        if video_path is not None and not video_path.is_file():
            video_path = None
        video_url = self.url_input.text().strip()

        if transcript_path is None and video_path is None and not video_url:
            QMessageBox.warning(
                self,
                "Thiếu đầu vào",
                "Chọn video, nhập URL YouTube/Bilibili hoặc mở transcript của project.",
            )
            self.tabs.setCurrentIndex(0)
            return

        gemini_key = (
            self.gemini_key_input.text().strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
        )
        vieneu_key = (
            self.vieneu_key_input.text().strip()
            or os.getenv("VIENEU_API_KEY", "").strip()
        )
        source_language = self.language_combo.currentData()
        if source_language == "auto":
            source_language = None

        selected_voice = self.voice_combo.currentData()
        voice_id = str(selected_voice) if selected_voice else ""
        speed = float(self.voice_speed_spin.value())

        # When reopening a project, the user may not have loaded the VieNeu catalogue.
        # Reuse the saved voice configuration for a matching transcript/translation.
        if transcript_path is not None and transcript_path.is_file():
            try:
                project_paths = MovieProjectPaths.from_transcript(transcript_path)
                if project_paths.tts_manifest.is_file():
                    tts_state = json.loads(
                        project_paths.tts_manifest.read_text(encoding="utf-8")
                    )
                    if not voice_id:
                        voice_id = str(tts_state.get("voice_id") or "")
                        speed = float(tts_state.get("speed", speed))
                        voice_index = self.voice_combo.findData(voice_id)
                        if voice_index >= 0:
                            self.voice_combo.setCurrentIndex(voice_index)
                        self.voice_speed_spin.setValue(speed)
            except (OSError, ValueError, TypeError):
                pass

        # Existing completed TTS cache is not billed again. For a new TTS stage,
        # warn before submitting any voice jobs because the service may charge by text.
        pending_voice_count = None
        if transcript_path is not None and transcript_path.is_file():
            try:
                project_paths = MovieProjectPaths.from_transcript(transcript_path)
                translation_state = json.loads(
                    project_paths.translated_json.read_text(encoding="utf-8")
                ) if project_paths.translated_json.is_file() else {}
                tts_state = json.loads(
                    project_paths.tts_manifest.read_text(encoding="utf-8")
                ) if project_paths.tts_manifest.is_file() else {}
                if translation_state and tts_state.get("is_complete") is not True:
                    cached = tts_state.get("segments", {})
                    pending_voice_count = sum(
                        1 for segment in translation_state.get("segments", [])
                        if isinstance(segment, dict)
                        and str(segment.get("translated_text", "")).strip()
                        and not (
                            isinstance(cached.get(str(segment.get("id"))), dict)
                            and cached[str(segment.get("id"))].get("status") == "completed"
                            and Path(str(cached[str(segment.get("id"))].get("audio_path", ""))).is_file()
                        )
                    )
            except (OSError, ValueError, TypeError, AttributeError):
                pass

        message = (
            "Chạy hoặc tiếp tục toàn bộ quy trình. Các transcript, chapter Gemini, "
            "batch dịch/QC, job VieNeu và audio đã hoàn thành sẽ được dùng lại từ cache."
        )
        if pending_voice_count:
            message += (
                f"\n\nHiện có tối đa {pending_voice_count} câu cần tạo giọng mới. "
                "VieNeu có thể tính phí theo ký tự; các job đã nhận sẽ được resume bằng job ID cũ."
            )
        answer = QMessageBox.question(
            self,
            "Chạy / tiếp tục project",
            message + "\n\nTiếp tục?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._begin_task("Đang kiểm tra cache và khôi phục project...")
        self._active_stage = None
        self.worker = PipelineWorker(
            output_dir=Path(output_text).expanduser(),
            video_path=video_path,
            video_url=video_url,
            transcript_path=transcript_path,
            whisper_model=str(self.model_combo.currentData()),
            source_language=source_language,
            gemini_api_key=gemini_key,
            gemini_model=str(self.gemini_model_combo.currentData()),
            translation_batch_size=self.translation_batch_spin.value(),
            vieneu_api_key=vieneu_key,
            voice_id=voice_id,
            voice_speed=speed,
            subtitle_mode=str(self.subtitle_mode_combo.currentData()),
            original_audio_mode=str(self.original_audio_combo.currentData()),
        )
        self._connect_common_worker(self.worker)
        self.worker.succeeded.connect(self._on_pipeline_success)
        self.worker.start()

    def _on_pipeline_success(self, result: dict[str, str]) -> None:
        transcript = Path(result["transcript"])
        self.transcript_input.setText(str(transcript))
        self.project_transcript_input.setText(str(transcript))
        self.translation_input.setText(result["translation"])
        self.render_translation_input.setText(result["translation"])
        self.video_input.setText(str(
            json.loads(transcript.read_text(encoding="utf-8")).get("source_video", "")
        ))
        self._set_project_source(transcript)
        self.status_label.setText("Toàn bộ quy trình đã hoàn tất.")
        self._append_log(f"MP4: {result['video']}")
        self._append_log(f"Phụ đề Việt: {result['subtitle']}")
        self._append_log(f"Project state: {result['project']}")
        self._refresh_project()
        self._update_action_states()
        QMessageBox.information(
            self,
            "Video đã hoàn tất",
            f"MP4: {result['video']}\n\nPhụ đề: {result['subtitle']}",
        )

    def _connect_common_worker(self, worker: object) -> None:
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
        self._last_project_progress = -1
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
            if (
                self.project_manager
                and self._active_stage
                and self._active_stage != "download"
                and abs(int(progress.percent) - self._last_project_progress) >= 5
            ):
                self.project_manager.update_progress(
                    self._active_stage,
                    int(progress.percent),
                    progress.message,
                )
                self._last_project_progress = int(progress.percent)

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
        self.project_transcript_input.setText(json_path)
        self._set_project_source(Path(json_path))
        self._complete_stage("transcription", segments=segment_count, language=language)
        self.status_label.setText(
            f"Whisper hoàn tất: {segment_count} đoạn | ngôn ngữ: {language}"
        )
        self._append_log(f"Transcript JSON: {json_path}")
        self._append_log(f"Source SRT: {srt_path}")
        self._refresh_context_status()
        self._refresh_project()
        self._update_action_states()
        QMessageBox.information(
            self, "Whisper hoàn tất",
            f"Đã nhận diện {segment_count} đoạn.\n\nJSON: {json_path}\nSRT: {srt_path}",
        )

    def _on_context_success(self, context_path: str) -> None:
        self.context_path = Path(context_path)
        self._context_ready = True
        self._complete_stage("context", context_path=context_path)
        self.context_status.setText(f"Context Profile: {context_path}")
        self.status_label.setText("Phân tích ngữ cảnh phim hoàn tất.")
        self._append_log(f"Context Profile: {context_path}")
        self._refresh_project()
        self._update_action_states()
        QMessageBox.information(self, "Gemini hoàn tất", f"Đã lưu Context Profile:\n{context_path}")

    def _on_translation_success(self, json_path: str, srt_path: str, quality_path: str) -> None:
        self.translation_input.setText(json_path)
        self.render_translation_input.setText(json_path)
        self._complete_stage("translation", translation_path=json_path)
        self.status_label.setText("Dịch phim và kiểm tra chất lượng hoàn tất.")
        self._append_log(f"Vietnamese JSON: {json_path}")
        self._append_log(f"Vietnamese SRT: {srt_path}")
        self._append_log(f"QC report: {quality_path}")
        self._refresh_project()
        self._update_action_states()
        QMessageBox.information(
            self, "Dịch phim hoàn tất",
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
        self._update_action_states()
        if not voices:
            QMessageBox.warning(self, "VieNeu", "Danh sách giọng hiện đang trống.")

    def _on_voice_list_failed(self, message: str) -> None:
        self._append_log(f"LỖI tải danh sách giọng: {message}")
        QMessageBox.critical(self, "Không tải được danh sách giọng", message)

    def _on_tts_success(self, manifest_path: str, completed: int, total: int) -> None:
        self._complete_stage("tts", manifest_path=manifest_path, segments=completed)
        self.status_label.setText(f"VieNeu hoàn tất: {completed}/{total} đoạn.")
        self._append_log(f"TTS manifest: {manifest_path}")
        self._refresh_project()
        self._update_action_states()
        QMessageBox.information(
            self, "VieNeu hoàn tất",
            f"Đã có audio cho {completed}/{total} đoạn.\n\nManifest: {manifest_path}",
        )

    def _on_render_success(self, output_path: str, metadata_path: str, cached: bool) -> None:
        self._complete_stage("render", output_path=output_path, cache_hit=cached)
        self.status_label.setText("Video MP4 tiếng Việt đã hoàn tất.")
        self._append_log(f"MP4: {output_path}")
        self._append_log(f"Metadata: {metadata_path}")
        self._refresh_project()
        QMessageBox.information(
            self, "Render hoàn tất",
            ("Đã dùng lại file render cache.\n\n" if cached else "Đã render video thành công.\n\n")
            + output_path,
        )

    def _on_failure(self, message: str) -> None:
        if self._active_stage and self.project_manager:
            self.project_manager.fail_stage(self._active_stage, message)
            self._refresh_project()
        self.status_label.setText("Tác vụ thất bại. Checkpoint đã lưu nếu công đoạn hỗ trợ resume.")
        self._append_log(f"LỖI: {message}")
        QMessageBox.critical(self, "Không thể hoàn thành tác vụ", message)

    def _on_cancelled(self) -> None:
        if self._active_stage and self.project_manager:
            self.project_manager.fail_stage(self._active_stage, "Đã hủy tác vụ.")
            self._refresh_project()
        self.status_label.setText("Đã hủy tác vụ.")
        self._append_log("Đã hủy. Chạy lại cùng project để tiếp tục từ checkpoint.")

    def _on_worker_finished(self) -> None:
        self.worker = None
        self._active_stage = None
        self._set_busy(False)
        self._refresh_context_status()
        self._refresh_project()
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
            self.project_transcript_input,
            self.project_browse_button,
            self.project_refresh_button,
            self.resume_button,
            self.render_translation_input,
            self.render_browse_button,
            self.subtitle_mode_combo,
            self.original_audio_combo,
            self.render_button,
        ):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def _set_project_source(self, transcript_path: Path) -> None:
        try:
            self.project_manager = ProjectManager(transcript_path)
            paths = MovieProjectPaths.from_transcript(transcript_path)
            self.project_transcript_input.setText(str(transcript_path))
            self.transcript_input.setText(str(transcript_path))
            if paths.context_json.is_file():
                self.context_path = paths.context_json
                try:
                    context_state = json.loads(paths.context_json.read_text(encoding="utf-8"))
                    self._context_ready = context_state.get("is_complete") is True
                except (OSError, ValueError):
                    self._context_ready = False
            else:
                self.context_path = None
                self._context_ready = False
            if paths.translated_json.is_file():
                self.translation_input.setText(str(paths.translated_json))
                self.render_translation_input.setText(str(paths.translated_json))
            source_video = Path("")
            try:
                transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
                source_video = Path(str(transcript_data.get("source_video", ""))).expanduser()
            except (OSError, ValueError):
                pass
            if source_video.is_file():
                self.video_input.setText(str(source_video))
            self._refresh_context_status()
            self._refresh_project()
            self._update_action_states()
        except Exception as exc:
            self.project_manager = None
            self._append_log(f"Không thể nạp project state: {exc}")

    def _set_project_from_translation(self, translated_path: Path) -> None:
        try:
            payload = json.loads(translated_path.read_text(encoding="utf-8"))
            source = payload.get("source_transcript")
            if source and Path(source).is_file():
                self._set_project_source(Path(source))
                if not self.transcript_input.text().strip():
                    self.transcript_input.setText(source)
        except (OSError, ValueError, AttributeError):
            pass

    def _paths_for_translation(self, translated_path: Path) -> MovieProjectPaths:
        payload = json.loads(translated_path.read_text(encoding="utf-8"))
        source = payload.get("source_transcript")
        if not source:
            raise ValueError("Bản dịch thiếu source_transcript.")
        return MovieProjectPaths.from_transcript(Path(source))

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
                    self._context_ready = payload.get("is_complete") is True
                    state_label = "hoàn tất" if self._context_ready else "đã lưu một phần"
                    self.context_status.setText(
                        f"Context Profile: {state_label} ({count}/{total}) — {paths.context_json}"
                    )
                except (OSError, ValueError):
                    self._context_ready = False
                    self.context_status.setText(f"Context Profile lỗi: {paths.context_json}")
                return
        self.context_path = None
        self._context_ready = False
        self.context_status.setText("Context Profile: chưa phân tích")

    def _refresh_project(self) -> None:
        transcript = Path(self.project_transcript_input.text().strip()).expanduser()
        if not transcript.is_file():
            transcript = Path(self.transcript_input.text().strip()).expanduser()
        if not transcript.is_file():
            self.project_status_label.setText("Chưa nạp project. Chọn transcript JSON đã tạo ở bước Whisper.")
            has_input = (
                Path(self.video_input.text().strip()).expanduser().is_file()
                or bool(self.url_input.text().strip())
            )
            self.resume_button.setEnabled(not self._busy and has_input)
            self.project_status_label.setText(
                "Chưa có transcript. Nhấn Chạy toàn bộ để tạo project mới từ video hoặc URL."
            )
            return
        try:
            self.project_manager = ProjectManager(transcript)
            state = self.project_manager.state
            lines = [f"Project: {state.get('project_name', transcript.stem)}"]
            labels = {
                "transcription": "Whisper",
                "context": "Gemini Context",
                "translation": "Gemini Translation + QC",
                "tts": "VieNeu TTS",
                "render": "MP4 Render",
            }
            for stage, label in labels.items():
                entry = state.get("stages", {}).get(stage, {})
                status = entry.get("status", "not_started")
                progress = int(entry.get("progress", 0))
                error = str(entry.get("last_error", "")).strip()
                lines.append(f"{label}: {status} ({progress}%)")
                if error and status in {"failed", "interrupted"}:
                    lines.append(f"  Lỗi gần nhất: {error[:180]}")
            lines.append(f"Project state: {self.project_manager.project_file}")
            next_stage = self.project_manager.next_incomplete_stage()
            self.resume_button.setEnabled(not self._busy and next_stage is not None)
            self.resume_button.setText(
                "Chạy toàn bộ / Tiếp tục project"
                if next_stage is not None
                else "Project đã hoàn tất"
            )
            if next_stage is None:
                lines.append("Trạng thái: TẤT CẢ CÔNG ĐOẠN ĐÃ HOÀN THÀNH")
            else:
                lines.append(f"Bước tiếp theo: {labels.get(next_stage, next_stage)}")
            self.project_status_label.setText("\n".join(lines))
        except Exception as exc:
            self.project_manager = None
            self.project_status_label.setText(f"Không đọc được project: {exc}")
            self.resume_button.setEnabled(False)

    def _mark_stage_running(self, stage: str) -> None:
        if self.project_manager:
            self.project_manager.start_stage(stage)
            self._refresh_project()

    def _complete_stage(self, stage: str, **details: object) -> None:
        if self.project_manager:
            self.project_manager.complete_stage(stage, **details)

    def _update_action_states(self) -> None:
        video_path = Path(self.video_input.text().strip()).expanduser()
        transcript_path = Path(self.transcript_input.text().strip()).expanduser()
        translated_path = Path(self.translation_input.text().strip()).expanduser()
        render_path = Path(self.render_translation_input.text().strip()).expanduser()
        has_transcript = transcript_path.is_file()
        has_context = self._context_ready and (
            self.context_path is not None and self.context_path.is_file()
        )
        if has_transcript and not has_context:
            context_file = MovieProjectPaths.from_transcript(transcript_path).context_json
            if context_file.is_file():
                try:
                    context_state = json.loads(context_file.read_text(encoding="utf-8"))
                    has_context = context_state.get("is_complete") is True
                    self._context_ready = has_context
                    self.context_path = context_file
                except (OSError, ValueError):
                    has_context = False
        self.transcribe_button.setEnabled(not self._busy and video_path.is_file())
        has_pipeline_input = (
            video_path.is_file()
            or bool(self.url_input.text().strip())
            or transcript_path.is_file()
            or Path(self.project_transcript_input.text().strip()).expanduser().is_file()
        )
        next_stage = (
            self.project_manager.next_incomplete_stage()
            if self.project_manager is not None
            else "transcription" if has_pipeline_input else None
        )
        self.resume_button.setEnabled(not self._busy and has_pipeline_input and next_stage is not None)
        self.context_button.setEnabled(not self._busy and has_transcript)
        self.translate_button.setEnabled(not self._busy and has_transcript and has_context)
        self.tts_button.setEnabled(
            not self._busy and translated_path.is_file() and bool(self.voice_combo.currentData())
        )
        self.render_button.setEnabled(not self._busy and render_path.is_file())

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
