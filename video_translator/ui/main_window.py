from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
from video_translator.core.models import DownloadProgress
from video_translator.ui.download_worker import DownloadWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self.worker: DownloadWorker | None = None
        self.setWindowTitle("VideoTranslator — Vietnamese Film Dubbing")
        self.resize(820, 620)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("VIDEO TRANSLATOR")
        title.setObjectName("title")
        subtitle = QLabel(
            "Tải video nguồn và chuẩn bị dữ liệu cho quy trình dịch phim, "
            "thuyết minh tiếng Việt."
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
        browse_button = QPushButton("Chọn thư mục")
        browse_button.clicked.connect(self._browse_output_dir)
        output_row.addWidget(self.output_input, 1)
        output_row.addWidget(browse_button)
        form.addRow("Thư mục lưu", output_row)

        layout.addLayout(form)

        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        actions = QHBoxLayout()
        self.download_button = QPushButton("Tải MP4 tối đa 1080p")
        self.download_button.setObjectName("primaryButton")
        self.download_button.clicked.connect(self._start_download)
        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_download)
        actions.addWidget(self.download_button)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)

        log_title = QLabel("NHẬT KÝ XỬ LÝ")
        log_title.setObjectName("sectionTitle")
        layout.addWidget(log_title)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Thông tin tải xuống sẽ hiển thị tại đây...")
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
            QLineEdit, QPlainTextEdit {
                background: #1F2937;
                color: #F9FAFB;
                border: 1px solid #374151;
                border-radius: 7px;
                padding: 9px;
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
            "Chọn thư mục lưu video",
            self.output_input.text().strip() or str(Path.home()),
        )
        if selected:
            self.output_input.setText(selected)

    def _start_download(self) -> None:
        url = self.url_input.text().strip()
        output_text = self.output_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Thiếu URL", "Vui lòng nhập link video.")
            return
        if not output_text:
            QMessageBox.warning(
                self, "Thiếu thư mục", "Vui lòng chọn thư mục lưu video."
            )
            return

        output_dir = Path(output_text).expanduser()
        self.log_view.clear()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self.status_label.setText("Đang chuẩn bị tải video...")

        self.worker = DownloadWorker(url, output_dir, self.settings.max_height)
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.log_message.connect(self._append_log)
        self.worker.succeeded.connect(self._on_success)
        self.worker.failed.connect(self._on_failure)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _cancel_download(self) -> None:
        if self.worker and self.worker.isRunning():
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Đang yêu cầu hủy...")
            self.worker.cancel()

    def _on_progress(self, progress: DownloadProgress) -> None:
        self.status_label.setText(progress.message)
        if progress.percent is None:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, round(progress.percent))))

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _on_success(self, output_path: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_label.setText("Tải video thành công.")
        self._append_log(f"File đầu ra: {output_path}")
        QMessageBox.information(
            self,
            "Hoàn tất",
            f"Video đã được tải và lưu thành MP4:\n{output_path}",
        )

    def _on_failure(self, message: str) -> None:
        self.status_label.setText("Tải video thất bại.")
        self._append_log(f"LỖI: {message}")
        QMessageBox.critical(self, "Không thể tải video", message)

    def _on_cancelled(self) -> None:
        self.status_label.setText("Đã hủy tải xuống.")
        self._append_log("Tác vụ đã được hủy.")

    def _on_worker_finished(self) -> None:
        self._set_busy(False)
        self.worker = None

    def _set_busy(self, busy: bool) -> None:
        self.download_button.setEnabled(not busy)
        self.url_input.setEnabled(not busy)
        self.output_input.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
