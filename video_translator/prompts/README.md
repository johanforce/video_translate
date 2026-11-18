# Gemini film-translation prompts

These prompt assets define the film-localization workflow. They are deliberately separated from the Gemini API client so the prompts can be revised without coupling them to provider code.

## Intended request sequence

1. **Context analysis** — send the full transcript when it fits. For long films, send ordered transcript chapters with stable segment IDs, collect chapter observations, then ask Gemini to consolidate them into one project-level context profile. Never treat a partial chapter as the whole film.
2. **Translation** — translate timestamped batches. Every request includes the same current context profile, the batch, and neighboring dialogue for continuity. Neighboring dialogue is context only and must not be emitted again.
3. **Quality control** — validate the translation against the source batch and context profile. The validator must check exact ID coverage and unchanged timestamps before accepting the response.

## Template variables

The future prompt builder will replace these literal tokens:

- \`{{TRANSCRIPT_JSON}}\`
- \`{{CONTEXT_PROFILE_JSON}}\`
- \`{{SEGMENTS_JSON}}\`
- \`{{PREVIOUS_CONTEXT_JSON}}\`
- \`{{NEXT_CONTEXT_JSON}}\`
- \`{{TRANSLATION_JSON}}\`

Do not use Python \`str.format()\` on prompt bodies unless the JSON examples are escaped. A safe token replacement function should replace only the named placeholders.

## Structured response contracts

- Context analysis: one JSON object containing \`film_profile\`, \`characters\`, \`relationships\`, \`terminology\`, \`addressing_rules\`, and \`uncertainties\`.
- Translation: one JSON object with a \`segments\` array. Each input segment must occur exactly once, with \`id\`, \`start\`, \`end\`, \`speaker_id\`, \`translated_text\`, \`emotion\`, and \`delivery\`.
- Quality control: one JSON object with \`segments\`, \`issues\`, and \`summary\`.

The application, not Gemini, remains responsible for parsing, schema validation, timestamp equality, segment coverage, caching, and retry behavior.
