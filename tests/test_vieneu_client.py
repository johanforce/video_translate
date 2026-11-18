import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from video_translator.tts.vieneu_client import VieNeuClient, VieNeuError


class VieNeuClientTests(unittest.TestCase):
    @patch("video_translator.tts.vieneu_client.requests.get")
    def test_lists_v4_voices(self, get_mock: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "voices": [
                {
                    "id": "Ngọc Lan",
                    "name": "Ngọc Lan",
                    "description": "Vietnamese southern female voice",
                    "gender": "female",
                    "region": "south",
                    "engine": "v4",
                    "kind": "catalog",
                }
            ]
        }
        response.raise_for_status.return_value = None
        get_mock.return_value = response

        voices = VieNeuClient.list_voices("key")
        self.assertEqual(voices[0]["id"], "Ngọc Lan")
        self.assertEqual(voices[0]["engine"], "v4")
        self.assertEqual(get_mock.call_args.kwargs["params"], {"engine": "v4"})

    def test_quota_error_is_not_retried(self) -> None:
        client = VieNeuClient("test-key")
        response = Mock()
        response.status_code = 429
        response.ok = False
        response.headers = {}
        response.text = "quota exceeded"
        response.reason = "Too Many Requests"
        response.json.return_value = {
            "code": "TRIAL_API_DAILY_LIMIT",
            "message": "Daily trial limit reached",
        }
        client.session.post = Mock(return_value=response)

        with self.assertRaisesRegex(VieNeuError, "TRIAL_API_DAILY_LIMIT"):
            client._submit_job(
                {"text": "Hello", "voiceId": "Ngọc Lan", "engine": "v4"},
                {"Idempotency-Key": "12345678"},
                threading.Event(),
            )
        self.assertEqual(client.session.post.call_count, 1)

    def test_rejects_non_https_audio_url(self) -> None:
        client = VieNeuClient("test-key")
        client._poll_job = Mock(
            return_value={"status": "completed", "audioUrl": "http://unsafe"}
        )
        with self.assertRaises(VieNeuError):
            client.resume_segment(
                job_id="job",
                segment_id=1,
                request_key="12345678",
                voice_id="Ngọc Lan",
                speed=1.0,
                output_dir=Path("."),
                cancel_event=threading.Event(),
            )


if __name__ == "__main__":
    unittest.main()
