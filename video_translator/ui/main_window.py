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
from video_translator.download.url_validator import validate_video_url
from video_translator.core.project_manager import ProjectManager
from video_translator.core.project_registry import ProjectRegistry
from video_translator.gemini.project_paths import MovieProjectPaths
from video_translator.gemini.manual_translation import (
    ManualTranslationError,
    create_manual_translation_template,
    import_manual_translation,
)
from video_translator.ui.batch_pipeline_worker import BatchPipelineWorker
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
            | BatchPipelineWorker
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
        # Populate the dropdown from the installed local SDK at startup. This only
        # reads the bundled voice catalogue; the TTS model itself loads on first use.
        self._refresh_voices()

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
        self._build_batch_tab()
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
        self.url_input.textChanged.connect(self._update_action_states)
        form.addRow("Video URL", self.url_input)

        self.output_input = QLineEdit(str(self.settings.output_dir))
        self.output_browse_button = QPushButton("Chọn thư mục")
        self.output_browse_button.clicked.connect(self._browse_output_dir)
        form.addRow("Thư mục dự án", self._row(self.output_input, self.output_browse_button))

        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("Chọn video đã tải hoặc video có sẵn...")
        self.video_input.textChanged.connect(self._update_action_states)
        self.video_input.editingFinished.connect(self._resolve_video_project)
        self.video_browse_button = QPushButton("Chọn video")
        self.video_browse_button.clicked.connect(self._browse_video)
        form.addRow("Video nguồn", self._row(self.video_input, self.video_browse_button))

        self.transcript_input = QLineEdit()
        self.transcript_input.setPlaceholderText("Chọn hoặc tạo transcript JSON...")
        self.transcript_input.textChanged.connect(self._update_action_states)
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
            "Whisper tạo transcript gốc có timestamp. Khi chọn lại video nguồn, ứng dụng "
            "tự tìm project đã lưu và tiếp tục checkpoint; cũng có thể mở transcript JSON thủ công."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(hint)
        layout.addStretch(1)
        self.tabs.addTab(tab, "1 · Video & Transcript")

    def _build_batch_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("CHẠY HÀNG LOẠT VIDEO")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        intro = QLabel(
            "Dán mỗi URL trên một dòng. Nhập danh sách Gemini API key, ngăn cách bằng dấu phẩy hoặc dấu chấm phẩy. "
            "Tool sẽ tự tải, nhận diện, phân tích/dịch/QC, tạo giọng VieNeu và render MP4 "
            "cho từng URL theo thứ tự."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(intro)

        layout.addWidget(QLabel("DANH SÁCH VIDEO URL"))
        self.batch_urls_input = QPlainTextEdit()
        self.batch_urls_input.setPlaceholderText(
            "https://www.youtube.com/watch?v=...\nhttps://youtu.be/...\nhttps://www.bilibili.com/video/..."
        )
        self.batch_urls_input.setMinimumHeight(150)
        self.batch_urls_input.textChanged.connect(self._update_action_states)
        layout.addWidget(self.batch_urls_input)

        layout.addWidget(QLabel("DANH SÁCH GEMINI API KEY"))
        self.batch_keys_input = QLineEdit()
        self.batch_keys_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.batch_keys_input.setPlaceholderText(
            "Nhập các key, phân tách bằng dấu phẩy hoặc dấu chấm phẩy"
        )
        self.batch_keys_input.textChanged.connect(self._update_action_states)
        layout.addWidget(self.batch_keys_input)

        layout.addWidget(QLabel("GIỌNG ĐỌC CHO TOÀN BỘ HÀNG ĐỢI"))
        self.batch_voice_combo = QComboBox()
        self.batch_voice_combo.setMinimumWidth(300)
        self.batch_voice_combo.setPlaceholderText("Đang tải danh sách giọng VieNeu SDK...")
        self.batch_voice_combo.setToolTip(
            "Giọng này được dùng cho tất cả video trong hàng đợi chạy hàng loạt."
        )
        layout.addWidget(self.batch_voice_combo)

        retry_hint = QLabel(
            "Chính sách tự động: mỗi lần lỗi chờ 10 giây rồi thử lại; sau 5 lần lỗi "
            "liên tiếp với một key sẽ chuyển sang key kế tiếp. Hết key thì dừng toàn bộ "
            "hàng đợi. Project được lưu checkpoint, nên lần retry không làm lại các bước "
            "đã hoàn tất. Key chỉ giữ trong RAM và không được ghi vào project/log."
        )
        retry_hint.setWordWrap(True)
        retry_hint.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(retry_hint)

        self.batch_start_button = QPushButton("Bắt đầu chạy hàng loạt")
        self.batch_start_button.setObjectName("primaryButton")
        self.batch_start_button.clicked.connect(self._start_batch_pipeline)
        layout.addWidget(self.batch_start_button)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Chạy hàng loạt")

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
            "chạy lại cùng project sẽ dùng phần đã lưu. Đổi Gemini model không xóa kết quả "
            "đã hoàn tất; model mới chỉ xử lý các phần còn dang dở."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(hint)

        manual_title = QLabel("DỊCH THỦ CÔNG (KHÔNG CẦN GEMINI API)")
        manual_title.setObjectName("sectionTitle")
        layout.addWidget(manual_title)
        manual_hint = QLabel(
            "Xuất một file JSON chứa ID và lời thoại gốc, điền bản dịch rồi nhập lại. "
            "Ứng dụng sẽ kiểm tra transcript có đúng project, đủ ID, bản dịch, nhãn giọng "
            "và xác nhận QC trước khi hoàn tất công đoạn Translation."
        )
        manual_hint.setWordWrap(True)
        manual_hint.setStyleSheet("color: #9CA3AF;")
        layout.addWidget(manual_hint)

        manual_buttons = QHBoxLayout()
        self.manual_template_button = QPushButton("1. Xuất file mẫu JSON")
        self.manual_template_button.clicked.connect(self._export_manual_template)
        self.manual_import_button = QPushButton("2. Nhập bản dịch & xác nhận QC")
        self.manual_import_button.setObjectName("primaryButton")
        self.manual_import_button.clicked.connect(self._import_manual_translation)
        manual_buttons.addWidget(self.manual_template_button)
        manual_buttons.addWidget(self.manual_import_button)
        manual_buttons.addStretch(1)
        layout.addLayout(manual_buttons)

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

        voice_row = QHBoxLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(300)
        self.voice_refresh_button = QPushButton("Nạp lại giọng SDK")
        self.voice_refresh_button.clicked.connect(self._refresh_voices)
        voice_row.addWidget(self.voice_combo, 1)
        voice_row.addWidget(self.voice_refresh_button)
        form.addRow("Giọng đọc có sẵn trong SDK local", voice_row)

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
            "VieNeu chạy trực tiếp trên máy bằng SDK v3 Turbo, không cần API key và không "
            "tính phí theo ký tự. Model được tải ở lần chạy đầu; các lần sau có thể chạy offline. "
            "Audio từng câu được lưu cache để tiếp tục nếu ứng dụng bị dừng."
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
            video_path = Path(selected).expanduser().resolve()
            self.video_input.setText(str(video_path))
            self._attach_project_for_video(video_path)
            self._update_action_states()

    def _resolve_video_project(self) -> None:
        video_path = Path(self.video_input.text().strip()).expanduser()
        if video_path.is_file():
            self._attach_project_for_video(video_path.resolve())

    def _find_project_for_video(self, video_path: Path) -> Path | None:
        directories: list[Path] = []
        for value in (
            self.output_input.text().strip(),
            str(video_path.parent),
            str(self.settings.output_dir),
            str(Path.home() / "Videos" / "VideoTranslator"),
        ):
            if value:
                directory = Path(value).expanduser()
                if directory not in directories:
                    directories.append(directory)
        return ProjectRegistry().find_for_video(
            video_path,
            tuple(directories),
        )

    @staticmethod
    def _transcript_matches_video(transcript_path: Path, video_path: Path) -> bool:
        try:
            payload = json.loads(transcript_path.read_text(encoding="utf-8"))
            source_text = str(payload.get("source_video") or "").strip()
            if not source_text:
                return transcript_path.name == f"{video_path.stem}.transcript.json"
            recorded = Path(source_text).expanduser()
            same_path = os.path.normcase(str(recorded.resolve())) == os.path.normcase(
                str(video_path.resolve())
            )
            moved_same_name = (
                not recorded.is_file()
                and recorded.name.casefold() == video_path.name.casefold()
                and transcript_path.name == f"{video_path.stem}.transcript.json"
            )
            return same_path or moved_same_name
        except (OSError, ValueError, TypeError):
            return False

    def _attach_project_for_video(self, video_path: Path) -> bool:
        transcript = self._find_project_for_video(video_path)
        if transcript is not None:
            self.output_input.setText(str(transcript.parent))
            self.transcript_input.setText(str(transcript))
            self.project_transcript_input.setText(str(transcript))
            self._set_project_source(transcript)
            self._append_log(
                f"[RESUME] Đã tìm thấy project theo video nguồn: {transcript}"
            )
            return True

        current_text = self.transcript_input.text().strip()
        current = Path(current_text).expanduser() if current_text else None
        if current is not None and current.is_file() and not self._transcript_matches_video(
            current, video_path
        ):
            self.transcript_input.clear()
            self.project_transcript_input.clear()
            self.translation_input.clear()
            self.render_translation_input.clear()
            self.project_manager = None
            self.context_path = None
            self._context_ready = False
            self._append_log(
                "Video nguồn khác project đang mở; đã bỏ liên kết transcript cũ để tránh dùng nhầm dữ liệu."
            )
        self._refresh_context_status()
        self._refresh_project()
        return False

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

    def _export_manual_template(self) -> None:
        transcript_path = Path(self.transcript_input.text().strip()).expanduser()
        if not transcript_path.is_file():
            QMessageBox.warning(
                self, "Thiếu transcript",
                "Hãy chọn transcript JSON đã hoàn tất ở tab Video & Transcript.",
            )
            return

        default_path = transcript_path.parent / (
            transcript_path.name.removesuffix(".transcript.json")
            + ".manual-translation.json"
        )
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất mẫu dịch thủ công",
            str(default_path),
            "Manual translation JSON (*.json)",
        )
        if not selected:
            return
        destination = Path(selected).expanduser()
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        try:
            count = create_manual_translation_template(transcript_path, destination)
        except (ManualTranslationError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "Không thể xuất file mẫu", str(exc))
            self._append_log(f"LỖI xuất mẫu dịch thủ công: {exc}")
            return

        self._append_log(f"Đã xuất mẫu dịch thủ công: {destination} ({count} segments)")
        QMessageBox.information(
            self,
            "Đã tạo file mẫu",
            f"Đã tạo file JSON gồm {count} đoạn thoại:\n{destination}\n\n"
            "Mở file bằng VS Code hoặc Notepad++, điền translated_text cho từng đoạn, "
            "đặt reviewed = true sau khi kiểm tra từng câu, rồi đặt "
            "quality_control.completed = true sau khi rà soát toàn bộ.",
        )

    def _import_manual_translation(self) -> None:
        transcript_path = Path(self.transcript_input.text().strip()).expanduser()
        if not transcript_path.is_file():
            QMessageBox.warning(
                self, "Thiếu transcript",
                "Hãy chọn đúng transcript JSON dùng để xuất mẫu.",
            )
            return

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Nhập bản dịch thủ công",
            str(transcript_path.parent),
            "Manual translation JSON (*.json);;All files (*)",
        )
        if not selected:
            return

        try:
            json_path, srt_path, quality_path, count = import_manual_translation(
                transcript_path,
                Path(selected),
            )
            ProjectRegistry().register(transcript_path)
            self.project_manager = ProjectManager(transcript_path)
            self.translation_input.setText(str(json_path))
            self.render_translation_input.setText(str(json_path))
            self._complete_stage("translation", translation_path=str(json_path), source="manual")
        except (ManualTranslationError, OSError, ValueError, TypeError) as exc:
            self._append_log(f"LỖI nhập bản dịch thủ công: {exc}")
            QMessageBox.critical(
                self,
                "File dịch thủ công chưa đạt yêu cầu",
                f"{exc}\n\nChưa xác nhận công đoạn Translation. "
                "Sửa file JSON rồi nhập lại.",
            )
            return

        self.status_label.setText(f"Đã xác nhận dịch thủ công: {count} đoạn.")
        self._append_log(f"Manual Vietnamese JSON: {json_path}")
        self._append_log(f"Vietnamese SRT: {srt_path}")
        self._append_log(f"Manual QC report: {quality_path}")
        self._refresh_project()
        self._update_action_states()
        QMessageBox.information(
            self,
            "Đã xác nhận công đoạn dịch",
            f"Đã kiểm tra và nhập thành công {count} đoạn.\n\n"
            f"JSON: {json_path}\nSRT: {srt_path}\nQC: {quality_path}\n\n"
            "Project Manager đã đánh dấu Translation = completed. "
            "Bạn có thể chuyển sang tab VieNeu Thuyết minh.",
        )

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
        self.voice_refresh_button.setEnabled(False)
        self._append_log("Đang đọc danh sách preset của VieNeu SDK local...")
        self.voice_worker = VoiceListWorker()
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
            f"Sẽ tổng hợp tối đa {pending} đoạn thoại bằng VieNeu SDK v3 Turbo ngay trên máy.\n"
            "Các audio đã có sẽ được dùng lại từ cache. Lần chạy đầu có thể cần tải model.\n"
            "Không gửi nội dung lên Cloud API và không tính phí theo ký tự. Tiếp tục?",
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

    def _start_batch_pipeline(self) -> None:
        urls = [
            line.strip()
            for line in self.batch_urls_input.toPlainText().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        raw_keys = (
            self.batch_keys_input.text()
            .replace(",", "\n")
            .replace(";", "\n")
        )
        api_keys = list(dict.fromkeys(
            line.strip()
            for line in raw_keys.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ))

        if not urls:
            QMessageBox.warning(self, "Thiếu URL", "Hãy nhập ít nhất một URL video.")
            return
        if not api_keys:
            QMessageBox.warning(self, "Thiếu API key", "Hãy nhập ít nhất một Gemini API key.")
            return

        normalized_urls: list[str] = []
        for index, url in enumerate(urls, start=1):
            try:
                normalized_urls.append(validate_video_url(url))
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "URL không hợp lệ",
                    f"URL dòng {index} không hợp lệ:\n{url}\n\n{exc}",
                )
                return

        output_text = self.output_input.text().strip()
        if not output_text:
            QMessageBox.warning(
                self,
                "Thiếu thư mục đầu ra",
                "Hãy chọn thư mục dự án trong tab Video & Transcript.",
            )
            return

        source_language = self.language_combo.currentData()
        if source_language == "auto":
            source_language = None
        selected_voice = self.batch_voice_combo.currentData()
        voice_id = str(selected_voice) if selected_voice else self.settings.vieneu_voice

        answer = QMessageBox.question(
            self,
            "Xác nhận chạy hàng loạt",
            f"Sẽ xử lý {len(normalized_urls)} video theo thứ tự bằng "
            f"{len(api_keys)} Gemini API key.\n\n"
            "Khi lỗi: chờ 10 giây rồi retry; sau 5 lần lỗi với một key sẽ chuyển key. "
            "Nếu hết key, toàn bộ hàng đợi sẽ dừng.\n\n"
            "API key chỉ được giữ trong bộ nhớ của phiên chạy. Bắt đầu?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._begin_task("Đang khởi chạy hàng đợi video...")
        self._active_stage = None
        self.worker = BatchPipelineWorker(
            urls=normalized_urls,
            api_keys=api_keys,
            output_dir=Path(output_text).expanduser(),
            whisper_model=str(self.model_combo.currentData()),
            source_language=source_language,
            gemini_model=str(self.gemini_model_combo.currentData()),
            translation_batch_size=self.translation_batch_spin.value(),
            voice_id=voice_id,
            voice_speed=float(self.voice_speed_spin.value()),
            subtitle_mode=str(self.subtitle_mode_combo.currentData()),
            original_audio_mode=str(self.original_audio_combo.currentData()),
        )
        self._connect_common_worker(self.worker)
        self.worker.succeeded.connect(self._on_batch_pipeline_success)
        self.worker.start()

    def _on_batch_pipeline_success(self, results: list[dict[str, object]]) -> None:
        completed = len(results)
        self.status_label.setText(f"Chạy hàng loạt hoàn tất: {completed} video.")
        self._append_log(f"[BATCH] Hoàn tất {completed} video.")
        summary_lines = []
        for index, result in enumerate(results, start=1):
            video_path = str(result.get("video") or "")
            summary_lines.append(f"{index}. {video_path}")
        if results:
            last_result = results[-1]
            transcript_value = last_result.get("transcript")
            if transcript_value:
                transcript = Path(str(transcript_value))
                if transcript.is_file():
                    self.transcript_input.setText(str(transcript))
                    self.project_transcript_input.setText(str(transcript))
                    self.translation_input.setText(str(last_result.get("translation") or ""))
                    self.render_translation_input.setText(str(last_result.get("translation") or ""))
                    self._set_project_source(transcript)
        self._update_action_states()
        QMessageBox.information(
            self,
            "Chạy hàng loạt hoàn tất",
            f"Đã tạo thành công {completed} video MP4:\n\n"
            + "\n".join(summary_lines),
        )

    def _resume_project(self) -> None:
        self._start_full_pipeline()

    def _start_full_pipeline(self) -> None:
        output_text = self.output_input.text().strip()

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

        # Selecting only the original video must still reopen its existing project.
        if video_path is not None:
            matching_transcript = self._find_project_for_video(video_path)
            transcript_matches = (
                transcript_path is not None
                and self._transcript_matches_video(transcript_path, video_path)
            )
            if matching_transcript is not None:
                transcript_path = matching_transcript
                self.output_input.setText(str(matching_transcript.parent))
                self.transcript_input.setText(str(matching_transcript))
                self.project_transcript_input.setText(str(matching_transcript))
                self._set_project_source(matching_transcript)
            elif not transcript_matches:
                transcript_path = None
                self.transcript_input.clear()
                self.project_transcript_input.clear()
                self.project_manager = None
                self.context_path = None
                self._context_ready = False
                self._refresh_context_status()
                self._refresh_project()

        # Project discovery may have switched to the folder containing the saved transcript.
        output_text = self.output_input.text().strip()
        if not output_text:
            QMessageBox.warning(self, "Thiếu thư mục", "Hãy chọn thư mục dự án.")
            self.tabs.setCurrentIndex(0)
            return
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
        if not voice_id:
            voice_id = self.settings.vieneu_voice

        # Completed local audio is reused. A new TTS stage runs on this machine only.
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
            "batch dịch/QC và audio VieNeu SDK local đã hoàn thành sẽ được dùng lại từ cache."
        )
        if pending_voice_count:
            message += (
                f"\n\nHiện có tối đa {pending_voice_count} câu cần tạo audio local mới. "
                "Lần đầu có thể cần tải model VieNeu SDK."
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
        alias_to_id = {
            str(alias): str(voice["id"])
            for voice in voices
            for alias in voice.get("aliases", [])
        }

        def populate_voice_combo(combo: QComboBox, preferred_voice: str) -> None:
            preferred_voice = alias_to_id.get(preferred_voice, preferred_voice)
            combo.clear()
            for voice in voices:
                label = voice["name"]
                description = voice.get("description", "")
                if description:
                    label = f"{label} — {description}"
                combo.addItem(label, voice["id"])
            selected_index = combo.findData(preferred_voice)
            if selected_index < 0 and voices:
                selected_index = combo.findData(self.settings.vieneu_voice)
                if selected_index < 0:
                    selected_index = 0
            if selected_index >= 0:
                combo.setCurrentIndex(selected_index)
            return selected_index

        selected_voice = str(
            self.voice_combo.currentData() or self.settings.vieneu_voice
        )
        batch_selected_voice = str(
            self.batch_voice_combo.currentData()
            or self.voice_combo.currentData()
            or self.settings.vieneu_voice
        )
        selected_index = populate_voice_combo(self.voice_combo, selected_voice)
        batch_selected_index = populate_voice_combo(
            self.batch_voice_combo,
            batch_selected_voice,
        )
        if selected_index < 0 and voices:
            self._append_log("Không khôi phục được giọng TTS chính; đã dùng preset đầu tiên.")
        if batch_selected_index < 0 and voices:
            self._append_log("Không khôi phục được giọng chạy hàng loạt; đã dùng preset đầu tiên.")
        self._append_log(f"Đã tải {len(voices)} giọng preset VieNeu SDK v3 Turbo.")
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
            self.batch_urls_input,
            self.batch_keys_input,
            self.batch_start_button,
            self.batch_voice_combo,
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
            self.manual_template_button,
            self.manual_import_button,
            self.translation_input,
            self.translation_browse_button,
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
            transcript_path = transcript_path.expanduser().resolve()
            self.output_input.setText(str(transcript_path.parent))
            ProjectRegistry().register(transcript_path)
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
            self._restore_project_settings(transcript_path, paths)
            self._refresh_context_status()
            self._refresh_project()
            self._update_action_states()
        except Exception as exc:
            self.project_manager = None
            self._append_log(f"Không thể nạp project state: {exc}")

    def _restore_project_settings(
        self,
        transcript_path: Path,
        paths: MovieProjectPaths,
    ) -> None:
        """Restore project-specific choices so a restart uses the same cache keys."""
        if not self.project_manager:
            return
        stages = self.project_manager.state.get("stages", {})

        transcription = stages.get("transcription", {})
        context = stages.get("context", {})
        translation = stages.get("translation", {})

        try:
            transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            transcript_data = {}

        whisper_model = transcription.get("model") or transcript_data.get("model")
        if whisper_model:
            index = self.model_combo.findData(str(whisper_model))
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

        requested_language = (
            transcription.get("requested_language")
            or transcript_data.get("requested_language")
        )
        if requested_language:
            language_index = self.language_combo.findData(str(requested_language))
            if language_index >= 0:
                self.language_combo.setCurrentIndex(language_index)

        context_state: dict = {}
        if paths.context_json.is_file():
            try:
                context_state = json.loads(paths.context_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                context_state = {}
        translation_state: dict = {}
        if paths.translated_json.is_file():
            try:
                translation_state = json.loads(paths.translated_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                translation_state = {}

        # Keep the currently selected Gemini model. Project restoration should
        # restore progress/settings, not silently override a model the user chose.

        batch_size = (
            translation.get("batch_size")
            or translation_state.get("batch_size")
        )
        if batch_size is None:
            chapter_size = context.get("batch_segments") or context_state.get("batch_segments")
            if chapter_size:
                batch_size = max(5, min(100, int(chapter_size) // 6))
        if batch_size is not None:
            self.translation_batch_spin.setValue(
                max(5, min(100, int(batch_size)))
            )

        tts_state: dict = {}
        if paths.tts_manifest.is_file():
            try:
                tts_state = json.loads(
                    paths.tts_manifest.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                tts_state = {}

        tts_stage = stages.get("tts", {})
        saved_voice = (
            tts_stage.get("voice_id")
            or tts_state.get("voice_id")
        )
        saved_speed = (
            tts_stage.get("speed")
            if tts_stage.get("speed") is not None
            else tts_state.get("speed")
        )
        if saved_voice:
            voice_id = str(saved_voice)
            index = self.voice_combo.findData(voice_id)
            if index >= 0:
                self.voice_combo.setCurrentIndex(index)
            else:
                fallback_index = self.voice_combo.findData(self.settings.vieneu_voice)
                if fallback_index >= 0:
                    self.voice_combo.setCurrentIndex(fallback_index)
                    self._append_log(
                        f"Project cũ dùng giọng '{voice_id}' của Cloud v4; "
                        f"SDK local sẽ dùng preset '{self.settings.vieneu_voice}'."
                    )
        if saved_speed is not None:
            try:
                self.voice_speed_spin.setValue(float(saved_speed))
            except (TypeError, ValueError):
                pass

        render = stages.get("render", {})
        subtitle_mode = render.get("subtitle_mode")
        original_audio_mode = render.get("original_audio_mode")
        if not subtitle_mode or not original_audio_mode:
            output = paths.transcript_json.parent / (
                paths.transcript_json.name.removesuffix(".transcript.json")
                + ".vi.dubbed.mp4"
            )
            metadata_path = output.with_suffix(".render.json")
            if metadata_path.is_file():
                try:
                    render_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    subtitle_mode = subtitle_mode or render_metadata.get("subtitle_mode")
                    original_audio_mode = (
                        original_audio_mode or render_metadata.get("original_audio_mode")
                    )
                except (OSError, ValueError):
                    pass
        if subtitle_mode:
            index = self.subtitle_mode_combo.findData(str(subtitle_mode))
            if index >= 0:
                self.subtitle_mode_combo.setCurrentIndex(index)
        if original_audio_mode:
            index = self.original_audio_combo.findData(str(original_audio_mode))
            if index >= 0:
                self.original_audio_combo.setCurrentIndex(index)

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
        if self.project_manager is not None:
            next_stage = self.project_manager.next_incomplete_stage()
        else:
            next_stage = "transcription" if has_pipeline_input else None
        self.resume_button.setEnabled(
            not self._busy and has_pipeline_input and next_stage is not None
        )
        self.context_button.setEnabled(not self._busy and has_transcript)
        self.translate_button.setEnabled(not self._busy and has_transcript and has_context)
        self.manual_template_button.setEnabled(not self._busy and has_transcript)
        self.manual_import_button.setEnabled(not self._busy and has_transcript)
        self.tts_button.setEnabled(
            not self._busy and translated_path.is_file() and bool(self.voice_combo.currentData())
        )
        self.render_button.setEnabled(not self._busy and render_path.is_file())
        self.batch_start_button.setEnabled(
            not self._busy
            and bool(self.batch_urls_input.toPlainText().strip())
            and bool(self.batch_keys_input.text().strip())
        )

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
