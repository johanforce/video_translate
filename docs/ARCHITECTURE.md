# Architecture and crash recovery

## Pipeline

```text
URL / local video
  -> yt-dlp download (max source height 1080p)
  -> FFmpeg source audio extraction
  -> persistent audio chunks
  -> faster-whisper chunk transcription
  -> timestamped transcript JSON/SRT
  -> Gemini film-context profile
  -> Gemini batch translation
  -> per-batch QC
  -> VieNeu v4 per-segment TTS
  -> prepared TTS clip cache
  -> absolute audio timeline + narration mix
  -> original-audio mix + Vietnamese subtitles
  -> final MP4 + render metadata
```

## Cache locations

For a video named `movie.mp4` and output directory `PROJECT_DIR`:

- `PROJECT_DIR/.video_translator_cache/movie/` — extracted audio, chunk WAV files and Whisper checkpoint.
- `PROJECT_DIR/movie.transcript.json` — partial or complete transcript.
- `PROJECT_DIR/movie.context.json` — cumulative film profile and chapter checkpoint.
- `PROJECT_DIR/movie.vi.translation.json` — batch output, pending responses, completed translations.
- `PROJECT_DIR/movie.tts/manifest.json` — VieNeu jobs and audio clips.
- `PROJECT_DIR/movie.tts/prepared/` — normalized TTS clips and timeline cache.
- `PROJECT_DIR/movie.project.json` — stage status/history.
- `PROJECT_DIR/movie.vi.dubbed.mp4` — final output and `movie.vi.dubbed.render.json` metadata.

## Recovery rules

### Whisper
The source video and selected model/language/chunk-size produce a transcription signature. Extracted audio is cached independently by source-video fingerprint. Every completed chunk stores:
- the chunk's fingerprint;
- absolute-time segments and word timestamps;
- detected language and probability.

The transcript JSON/SRT is rewritten atomically after each chunk with `is_complete=false`. A final write sets `is_complete=true`. On restart, only missing or fingerprint-mismatched chunks are submitted to Whisper. A partial transcript cannot be sent to Gemini.

### Gemini context
Each chapter is analyzed in chronological order. The profile is merged cumulatively and saved after every chapter. The checkpoint signature includes transcript fingerprint, model, chapter size and prompt text. A matching completed profile is reused. A changed transcript/model/prompt invalidates only the context stage and downstream artifacts.

### Gemini translation and QC
Each batch has its own pending state:
- `translation` is written atomically immediately after the translation response is validated;
- `quality_control` and `issues` are written after QC;
- the batch is merged into the completed segment list and its pending record is cleared.

If translation succeeds but QC fails, restart reads the saved translation and retries QC only. Completed batches are skipped. IDs, order, segment count and timestamps are validated against source data.

### VieNeu
Before each submission, the segment's stable request signature and idempotency key are written to the manifest. When VieNeu returns a job ID, it is checkpointed immediately. On restart:
- completed audio files are reused;
- accepted jobs are polled by saved ID;
- if the process stopped after submission but before the ID was saved, the same idempotency key/body is resubmitted, preventing duplicate jobs while the provider's idempotency record remains valid.

Manifest fingerprint binds all audio to the exact translated JSON, voice and speed. Editing the translation invalidates dependent TTS only; unchanged segment audio retains its per-segment cache.

### Audio timeline
Every prepared voice clip has its own signature derived from TTS request, source audio file, timestamp window and preparation settings. A completed prepared clip is reused. The narration mix is regenerated locally from prepared clips if its own process was interrupted; it does not call VieNeu.

### MP4 render
Rendering happens to a `.partial.mp4` file. The final name is replaced only after ffprobe confirms video and audio streams. A render metadata file records hashes of source video, translation, subtitle, TTS manifest and output. If an interrupted render leaves a partial file, it is discarded on next run; cached narration is reused and FFmpeg performs only the final mux/render again.

## Project manifest
`movie.project.json` is a reconciled view of stage state. A stage left in `running` state becomes `interrupted` when reopened. Completion is not based only on the manifest: `ProjectManager` checks the underlying artifacts and their fingerprints, preventing stale stages from being considered complete.

## Service boundaries
- UI: PySide6 and QThread workers only.
- Download: yt-dlp.
- Audio/transcription: FFmpeg, ffprobe, faster-whisper.
- Context/translation/QC: Google Gen AI SDK.
- TTS: VieNeu Cloud API v4.
- Timeline/render: FFmpeg and NumPy.
- Project state/cache: atomic JSON files and SHA-256 artifact fingerprints.
