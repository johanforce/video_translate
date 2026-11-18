class VideoTranslatorError(Exception):
    """Base exception for expected application errors."""


class InvalidVideoUrlError(VideoTranslatorError):
    """Raised when a URL is not from a supported video platform."""


class DownloadCancelledError(VideoTranslatorError):
    """Raised when a download is cancelled by the user."""


class TranscriptionCancelledError(VideoTranslatorError):
    """Raised when audio extraction or transcription is cancelled."""


class VideoDownloadError(VideoTranslatorError):
    """Raised when yt-dlp or FFmpeg cannot complete a download."""
