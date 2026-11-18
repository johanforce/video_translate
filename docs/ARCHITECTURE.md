# Architecture and translation workflow

## Product direction

VideoTranslator is a film translation and Vietnamese dubbing tool. Its core quality goal is not literal subtitle translation: it is faithful, emotionally appropriate Vietnamese dialogue that sounds natural when spoken.

## Pipeline boundaries

\`\`\`text
URL
  -> downloader
  -> source media
  -> audio extraction
  -> transcription / subtitle import
  -> timestamped source segments
  -> whole-film context analysis (Gemini)
  -> persistent context profile
  -> contextual batch translation (Gemini)
  -> translation validation / quality control
  -> Vietnamese TTS (VieNeu)
  -> audio timeline and mix
  -> subtitles + MP4 render
\`\`\`

## Context-first translation

The context analyzer receives the full transcript when feasible. For long films, the transcript is divided into ordered chapters, analyzed incrementally, and consolidated into one project-level context profile. The profile stores film setting, narrative tone, character identities, relationship evidence, forms of address, terminology, continuity facts, and unresolved ambiguity.

Every translation batch receives:
- the current context profile;
- current source segments with stable IDs and timestamps;
- a small previous-dialogue window;
- a small next-dialogue window.

Neighboring dialogue is context only and must not be emitted again.

## Data integrity

Gemini is not the authority for IDs or timestamps. The application must validate:
- exact segment count;
- exact, unique ID set;
- original segment order;
- exact equality of start/end values with source;
- valid enum values;
- non-empty translation for meaningful source speech.

Reject malformed or incomplete output and retry only the failed batch. Never silently accept dropped or duplicated dialogue.

## Long-form project resilience

The future project store should persist each stage independently:
- source video metadata and local path;
- source transcript and subtitle imports;
- context profile and its revision;
- translation batches and validation status;
- TTS assets keyed by segment and voice configuration;
- render settings and final output.

A changed translation should invalidate only the affected TTS segment and downstream render, not force the source download or transcription to run again.

## Media handling

The initial downloader targets source formats up to 1080p and merges/remuxes to MP4. It must never claim native 1080p when the source is lower resolution. Audio extraction, transcription, TTS, and rendering are separate services and must not be implemented inside UI widgets.

## Provider boundaries

- Gemini is accessed through a translation provider interface in a later milestone.
- VieNeu is accessed through a TTS provider interface in a later milestone.
- Whisper is accessed through a transcription provider interface in a later milestone.

This keeps the core pipeline independent from any one model or API provider.
