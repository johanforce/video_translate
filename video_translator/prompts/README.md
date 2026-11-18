# Gemini film-translation prompts

These prompt assets define the Gemini workflow. They favor compact inputs and sparse outputs; the application restores source metadata and applies only the model's accepted changes.

## Request sequence

1. **Context analysis** — analyze ordered transcript chapters and return only new or materially changed observations. The application merges and de-duplicates profile data locally. There is no second Gemini finalization request.
2. **Translation** — translate the current batch using a compact context profile and neighboring dialogue. Return compact [segment ID, translation] rows; include performance metadata only when non-default.
3. **Quality control** — review the entire batch but return only corrections and unresolved high/critical issues. Never echo unchanged translation lines.

## Template variables

- `{{TRANSCRIPT_JSON}}`
- `{{CONTEXT_PROFILE_JSON}}`
- `{{SEGMENTS_JSON}}`
- `{{PREVIOUS_CONTEXT_JSON}}`
- `{{NEXT_CONTEXT_JSON}}`
- `{{TRANSLATION_JSON}}`

The prompt loader replaces only these named placeholders.

## Response contracts

- **Context analysis:** incremental object with compact `film_profile`, `characters`, `relationships`, `terminology`, `style_guide`, and `uncertainties` fields. Evidence IDs and confidence scores are omitted.
- **Translation:** `segments: [[id, translated_text], ...]`; optional sparse `performance: [[id, emotion, delivery], ...]`. Source speaker IDs and timestamps are reconstructed locally.
- **Quality control:** optional `corrections: [[id, replacement_text], ...]`, `performance: [[id, emotion, delivery], ...]`, and `issues: [[id, severity, category], ...]`. An acceptable batch returns only `{"corrections":[]}`.

The application validates IDs, restores timestamps and source fields, preserves checkpoints, and accepts legacy object-shaped cached responses.
