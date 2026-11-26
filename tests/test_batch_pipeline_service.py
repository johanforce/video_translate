import tempfile
import threading
import unittest
from pathlib import Path

from video_translator.core.batch_pipeline_service import (
    BatchApiKeysExhaustedError,
    BatchPipelineCancelledError,
    BatchPipelineService,
)


class FakePipeline:
    def __init__(self, factory, **_kwargs):
        self.factory = factory

    def run(self, **kwargs):
        self.factory.calls.append(kwargs["gemini_api_key"])
        if self.factory.errors:
            error = self.factory.errors.pop(0)
            raise RuntimeError(error)
        return {"video": Path("/output/final.mp4")}


class FakeFactory:
    def __init__(self, errors=None):
        self.errors = list(errors or [])
        self.calls = []

    def __call__(self, **kwargs):
        return FakePipeline(self, **kwargs)


class BatchPipelineServiceTests(unittest.TestCase):
    def _run(self, factory, urls=None, keys=None, **overrides):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = BatchPipelineService(pipeline_factory=factory)
            arguments = {
                "urls": urls or ["https://youtu.be/abc123"],
                "api_keys": keys or ["key-one"],
                "output_dir": Path(temp_dir),
                "retry_delay_seconds": 0,
            }
            arguments.update(overrides)
            return service.run(**arguments)

    def test_retries_same_url_with_same_key_until_success(self):
        factory = FakeFactory(["temporary error 1", "temporary error 2"])

        results = self._run(factory)

        self.assertEqual(factory.calls, ["key-one", "key-one", "key-one"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["video"], Path("/output/final.mp4"))

    def test_rotates_key_after_five_failures_and_resumes_same_url(self):
        factory = FakeFactory(["failure"] * 5)

        results = self._run(
            factory,
            keys=["key-one", "key-two"],
        )

        self.assertEqual(
            factory.calls,
            ["key-one"] * 5 + ["key-two"],
        )
        self.assertEqual(len(results), 1)

    def test_stops_entire_queue_when_all_keys_are_exhausted(self):
        factory = FakeFactory(["failure"] * 10)

        with self.assertRaises(BatchApiKeysExhaustedError):
            self._run(
                factory,
                urls=["https://youtu.be/abc123", "https://youtu.be/def456"],
                keys=["key-one", "key-two"],
            )

        self.assertEqual(factory.calls, ["key-one"] * 5 + ["key-two"] * 5)

    def test_duplicate_keys_are_not_retried_as_separate_keys(self):
        factory = FakeFactory(["failure"] * 5)

        with self.assertRaises(BatchApiKeysExhaustedError):
            self._run(
                factory,
                keys=["key-one", "key-one", "key-one"],
            )

        self.assertEqual(factory.calls, ["key-one"] * 5)

    def test_rejects_invalid_url_before_running_pipeline(self):
        factory = FakeFactory()

        with self.assertRaises(Exception):
            self._run(factory, urls=["https://example.com/not-a-video"])

        self.assertEqual(factory.calls, [])

    def test_cancellation_during_retry_wait_stops_queue(self):
        factory = FakeFactory(["temporary error"])
        cancel_event = threading.Event()

        def cancel_on_log(message):
            if "Chờ" in message:
                cancel_event.set()

        with tempfile.TemporaryDirectory() as temp_dir:
            service = BatchPipelineService(
                pipeline_factory=factory,
                log_callback=cancel_on_log,
            )
            with self.assertRaises(BatchPipelineCancelledError):
                service.run(
                    urls=["https://youtu.be/abc123"],
                    api_keys=["key-one"],
                    output_dir=Path(temp_dir),
                    retry_delay_seconds=0,
                    cancel_event=cancel_event,
                )

        self.assertEqual(factory.calls, ["key-one"])


if __name__ == "__main__":
    unittest.main()
