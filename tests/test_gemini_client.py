import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from video_translator.gemini.client import (
    GeminiClientError,
    GeminiJsonClient,
    GeminiOutputLimitError,
)
from video_translator.gemini.prompt_loader import render_prompt


class GeminiClientTests(unittest.TestCase):
    def test_parse_plain_json(self) -> None:
        self.assertEqual(GeminiJsonClient._parse_json('{"segments": []}'), {"segments": []})

    def test_parse_fenced_json(self) -> None:
        fence = chr(96) * 3
        content = fence + 'json\n{"film_profile": {"genre": "Drama"}}\n' + fence
        self.assertEqual(
            GeminiJsonClient._parse_json(content),
            {"film_profile": {"genre": "Drama"}},
        )

    def test_rejects_non_object_json(self) -> None:
        with self.assertRaises(GeminiClientError):
            GeminiJsonClient._parse_json("[]")

    def test_invalid_response_is_saved_with_finish_reason(self) -> None:
        raw_response = '{"characters": [\n  {"name": "A"}'
        response = SimpleNamespace(
            text=raw_response,
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
            usage_metadata=SimpleNamespace(total_token_count=8192),
        )

        class FakeModels:
            def generate_content(self, **kwargs):
                return response

        client = GeminiJsonClient.__new__(GeminiJsonClient)
        client.model = "test-model"
        client._types = SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
        client._client = SimpleNamespace(models=FakeModels())

        with tempfile.TemporaryDirectory() as temp_dir:
            debug_path = Path(temp_dir) / "chapter-03.raw.txt"
            with self.assertRaisesRegex(
                GeminiOutputLimitError,
                "finish_reason=MAX_TOKENS",
            ) as raised:
                client.generate_json(
                    "prompt",
                    "system",
                    debug_path=debug_path,
                )

            self.assertIn("dòng 2", str(raised.exception))
            self.assertIn(str(debug_path), str(raised.exception))
            self.assertEqual(debug_path.read_text(encoding="utf-8"), raw_response)

    def test_invalid_json_with_stop_is_repaired_once(self) -> None:
        first = SimpleNamespace(
            text='{"characters": [\\n  {"name": "A",}\\n]}',
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(total_token_count=9166),
        )
        repaired = SimpleNamespace(
            text='{"characters": [{"name": "A"}]}',
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(total_token_count=120),
        )

        class FakeModels:
            def __init__(self):
                self.responses = [first, repaired]
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return self.responses.pop(0)

        client = GeminiJsonClient.__new__(GeminiJsonClient)
        client.model = "test-model"
        client._types = SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: kwargs
        )
        models = FakeModels()
        client._client = SimpleNamespace(models=models)

        with tempfile.TemporaryDirectory() as temp_dir:
            debug_path = Path(temp_dir) / "chapter-05.raw.txt"
            result = client.generate_json(
                "chapter prompt",
                "system prompt",
                debug_path=debug_path,
            )

            self.assertEqual(result, {"characters": [{"name": "A"}]})
            self.assertEqual(len(models.calls), 2)
            self.assertIn("Repair only the JSON syntax", models.calls[1]["contents"])
            self.assertTrue(debug_path.is_file())
            self.assertEqual(
                debug_path.with_suffix(".repaired.json").read_text(encoding="utf-8"),
                '{\\n  "characters": [\\n    {\\n      "name": "A"\\n    }\\n  ]\\n}',
            )

    def test_json_repair_max_tokens_is_reported_as_output_limit(self) -> None:
        first = SimpleNamespace(
            text='{"characters": [',
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(total_token_count=100),
        )
        repair = SimpleNamespace(
            text='',
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
            usage_metadata=SimpleNamespace(total_token_count=8192),
        )

        class FakeModels:
            def __init__(self):
                self.responses = [first, repair]

            def generate_content(self, **kwargs):
                return self.responses.pop(0)

        client = GeminiJsonClient.__new__(GeminiJsonClient)
        client.model = "test-model"
        client._types = SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: kwargs
        )
        client._client = SimpleNamespace(models=FakeModels())

        with self.assertRaises(GeminiOutputLimitError):
            client.generate_json("prompt", "system")

    def test_prompt_replaces_only_named_tokens(self) -> None:
        template = '{"data": {{DATA}}, "literal": "{not-a-token}"}'
        rendered = render_prompt(template, {"DATA": '[{"id": 1}]'})
        self.assertIn('"id": 1', rendered)
        self.assertIn("{not-a-token}", rendered)


if __name__ == "__main__":
    unittest.main()
