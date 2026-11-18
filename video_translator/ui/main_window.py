from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
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
    QVBoxLayout,
    QWidget,
)

from video_translator.config.settings import AppSettings
from video_translator.core.models import DownloadProgress, TranscriptionProgress
from video_translator.ui.download_worker import DownloadWorker
from video_translator.ui.transcription_worker import TranscriptionWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self.worker: DownloadWorker | TranscriptionWorker | None = None
        self.setWindowTitle("VideoTranslator — Vietnamese Film Dubbing")
        self.resize(880, 700)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("VIDEO TRANSLATOR")
        title.setObjectName("title")
        subtitle = QLabel(
            "Tải video, nhận diện lời thoại và tạo transcript có timestamp "
            "để chuẩn bị dịch phim."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setSpacing(10)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Dán link YouTube hoặc Bilibili...")
        form.addRow("Video URL", self.url_input)

        output_row = QHBoxLayout()
        self.output_input = QLineEdit(str(self.settings.output_dir))
        self.output_browse_button = QPushButton("Chọn thư mục")
        self.output_browse_button.clicked.connect(self._browse_output_dir)
        output_row.addWidget(self.output_input, 1)
        output_row.addWidget(self.output_browse_button)
        form.addRow("Thư mục lưu kết quả", output_row)

        video_row = QHBoxLayout()
        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("Chọn video MP4 đã tải hoặc video có sẵn...")
        self.video_browse_button = QPushButton("Chọn video")
        self.video_browse_button.clicked.connect(self._browse_video)
        video_row.addWidget(self.video_input, 1)
        video_row.addWidget(self.video_browse_button)
        form.addRow("Video nguồn", video_row)

        whisper_row = QHBoxLayout()
        self.model_combo = QComboBox()
        for label, value in (
            ("Tiny — nhanh, độ chính xác thấp hơn", "tiny"),
            ("Base — nhẹ", "base"),
            ("Small — khuyến nghị", "small"),
            ("Medium — chính xác hơn, chậm hơn", "medium"),
            ("Large v3 — chất lượng cao, cần tài nguyên", "large-v3"),
        ):
            self.model_combo.addItem(label, value)
        self.model_combo.setCurrentIndex(2)

        self.language_combo = QComboBox()
        language_options = (
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
        )
        for label, value in language_options:
            self.language_combo.addItem(label, value)
        whisper_row.addWidget(self.model_combo, 2)
        whisper_row.addWidget(self.language_combo, 1)
        form.addRow("Whisper model / ngôn ngữ", whisper_row)

        layout.addLayout(form)

        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        actions = QHBoxLayout()
        self.download_button = QPushButton("Tải MP4 tối đa 1080p")
        self.download_button.setObjectName("primaryButton")
        self.download_button.clicked.connect(self._start_download)

        self.transcribe_button = QPushButton("Nhận diện lời thoại")
        self.transcribe_button.clicked.connect(self._start_transcription)
        self.transcribe_button.setEnabled(False)

        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_task)

        actions.addWidget(self.download_button)
        actions.addWidget(self.transcribe_button)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)

        log_title = QLabel("NHẬT KÝ XỬ LÝ")
        log_title.setObjectName("sectionTitle")
        layout.addWidget(log_title)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText(
            "Tiến trình download, FFmpeg và Whisper sẽ hiển thị tại đây..."
        )
        layout.addWidget(self.log_view, 1)

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
                padding-bottom: 8px;
            }
            QLabel#sectionTitle {
                color: #D1D5DB;
                font-weight: 700;
                padding-top: 8px;
            }
            QLabel#status {
                color: #93C5FD;
                font-weight: 600;
            }
            QLineEdit, QPlainTextEdit, QComboBox {
                background: #1F2937;
                color: #F9FAFB;
                border: 1px solid #374151;
                border-radius: 7px;
                padding: 9px;
                selection-background-color: #2563EB;
            }
            QComboBox QAbstractItemView {
                background: #1F2937;
                color: #F9FAFB;
                selection-background-color: #2563EB;
            }
            QPushButton {
                background: #374151;
                color: #F9FAFB;
                border: 0;
                border-radius: 7px;
                padding: 10px 16px;
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

    def _browse_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Chọn thư mục lưu kết quả",
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
            self._update_transcribe_enabled()

    def _start_download(self) -> None:
        url = self.url_input.text().strip()
        output_text = self.output_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Thiếu URL", "Vui lòng nhập link video.")
            return
        if not output_text:
            QMessageBox.warning(self, "Thiếu thư mục", "Vui lòng chọn thư mục lưu.")
            return

        self.log_view.clear()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self.status_label.setText("Đang chuẩn bị tải video...")

        self.worker = DownloadWorker(
            url=url,
            output_dir=Path(output_text).expanduser(),
            max_height=self.settings.max_height,
        )
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.log_message.connect(self._append_log)
        self.worker.succeeded.connect(self._on_download_success)
        self.worker.failed.connect(self._on_failure)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _start_transcription(self) -> None:
        video_text = self.video_input.text().strip()
        output_text = self.output_input.text().strip()
        video_path = Path(video_text).expanduser()
        if not video_text or not video_path.is_file():
            QMessageBox.warning(self, "Thiếu video", "Hãy chọn một file video hợp lệ.")
            return
        if not output_text:
            QMessageBox.warning(self, "Thiếu thư mục", "Vui lòng chọn thư mục lưu.")
            return

        language = self.language_combo.currentData()
        if language == "auto":
            language = None

        self.log_view.clear()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self.status_label.setText("Đang chuẩn bị nhận diện lời thoại...")

        self.worker = TranscriptionWorker(
            video_path=video_path,
            output_dir=Path(output_text).expanduser(),
            model_size=str(self.model_combo.currentData()),
            language=language,
        )
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.log_message.connect(self._append_log)
        self.worker.succeeded.connect(self._on_transcription_success)
        self.worker.failed.connect(self._on_failure)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

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

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _on_download_success(self, output_path: str) -> None:
        self.video_input.setText(output_path)
        self.status_label.setText("Tải video thành công. Có thể nhận diện lời thoại.")
        self._append_log(f"Video nguồn: {output_path}")
        QMessageBox.information(
            self,
            "Hoàn tất download",
            f"Video đã được tải:\n{output_path}\n\nBạn có thể chạy Nhận diện lời thoại.",
        )

    def _on_transcription_success(
        self,
        json_path: str,
        srt_path: str,
        language: str,
        segment_count: int,
    ) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_label.setText(
            f"Nhận diện hoàn tất: {segment_count} đoạn | ngôn ngữ: {language}"
        )
        self._append_log(f"Transcript JSON: {json_path}")
        self._append_log(f"Source SRT: {srt_path}")
        QMessageBox.information(
            self,
            "Hoàn tất transcription",
            f"Đã nhận diện {segment_count} đoạn lời thoại.\n\n"
            f"JSON: {json_path}\nSRT: {srt_path}",
        )

    def _on_failure(self, message: str) -> None:
        self.status_label.setText("Tác vụ thất bại.")
        self._append_log(f"LỖI: {message}")
        QMessageBox.critical(self, "Không thể hoàn thành tác vụ", message)

    def _on_cancelled(self) -> None:
        self.status_label.setText("Đã hủy tác vụ.")
        self._append_log("Tác vụ đã được hủy.")

    def _on_worker_finished(self) -> None:
        self.worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.url_input,
            self.output_input,
            self.output_browse_button,
            self.video_input,
            self.video_browse_button,
            self.model_combo,
            self.language_combo,
            self.download_button,
        ):
            widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self._update_transcribe_enabled()

    def _update_transcribe_enabled(self) -> None:
        if self.worker and self.worker.isRunning():
            self.transcribe_button.setEnabled(False)
            return
        video_path = Path(self.video_input.text().strip()).expanduser()
        self.transcribe_button.setEnabled(video_path.is_file())
