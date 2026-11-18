import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from video_translator.ui.main_window import MainWindow


class MainWindowSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_constructs_and_pipeline_action_exists(self) -> None:
        window = MainWindow()
        self.assertTrue(window.windowTitle())
        self.assertTrue(hasattr(window, "resume_button"))
        self.assertTrue(hasattr(window, "cancel_button"))
        window.close()


if __name__ == "__main__":
    unittest.main()
