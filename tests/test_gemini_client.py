import unittest

from video_translator.gemini.client import GeminiClientError, GeminiJsonClient
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

    def test_prompt_replaces_only_named_tokens(self) -> None:
        template = '{"data": {{DATA}}, "literal": "{not-a-token}"}'
        rendered = render_prompt(template, {"DATA": '[{"id": 1}]'})
        self.assertIn('"id": 1', rendered)
        self.assertIn("{not-a-token}", rendered)


if __name__ == "__main__":
    unittest.main()
