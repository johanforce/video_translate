from __future__ import annotations

import sys

from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication

from video_translator.ui.main_window import MainWindow


def main() -> int:
    load_dotenv()
    app = QApplication(sys.argv)
    app.setApplicationName("VideoTranslator")
    app.setOrganizationName("VideoTranslator")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
