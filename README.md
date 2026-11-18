# VideoTranslator

Python desktop application for contextual Vietnamese film translation and dubbing.

## Current release: v0.7

### v0.1 — Download
- YouTube and Bilibili through yt-dlp.
- MP4 output, prefer source streams at or below 1080p.
- FFmpeg merges/remuxes video and audio. Lower-resolution sources are not upscaled.

### v0.2 — Transcription
- FFmpeg extracts mono 16 kHz PCM audio.
- faster-whisper transcribes in persistent chunks (default 4 minutes).
- Automatic language detection or manual source-language selection.
- Segment and word timestamps are converted to absolute video time.
- Partial transcript JSON/SRT is saved after each completed chunk.
- Restart skips chunks whose source/model/language fingerprints still match.

### v0.3 — Movie Context
- Gemini analyzes the transcript chapter by chapter (default 120 segments when translation batches are 30).
- Builds a persistent profile of the story, characters, relationships, Vietnamese forms of address, terminology, continuity facts and uncertainties.
- Saves the profile after each accepted chapter/subchapter.
- If Gemini reaches `MAX_TOKENS`, the application automatically bisects that chapter and retries smaller ranges; each accepted subchapter is checkpointed before continuing.
- Context output is incremental: the prompt asks Gemini to report only new or materially updated facts, not repeat the entire accumulated profile.
- Changing Gemini model or reducing chapter size keeps the already-processed segment count/profile. Only the unfinished tail is sent again.
- A matching completed profile is reused without another Gemini request.

### v0.4 — Film Translation
- Translates batches using the film profile and neighboring dialogue.
- Natural spoken Vietnamese, with character voice and performance metadata.
- Quality control runs separately for every batch.
- Source segment IDs are validated and matched by ID; timestamps are always restored from the original Whisper transcript, so Gemini cannot alter the dubbing timeline.
- Translation response is checkpointed immediately, before QC. If Gemini fails during QC, restart reuses the translation and retries only QC.
- Exports Vietnamese JSON, SRT and a quality report.

### v0.5 — VieNeu TTS
- Loads the current VieNeu v4 voice catalogue.
- Creates one audio file per translated dialogue segment.
- Persists request keys and job IDs before/after submitting TTS jobs.
- Reuses completed audio and resumes polling accepted jobs.
- Legacy v0.5 cache manifests are migrated without regenerating matching audio.
- Only supported v4 inline performance cues are sent to the TTS engine.

### v0.6 — Audio timeline and final MP4
- Places each prepared speech clip at its absolute transcript timestamp.
- Caches every prepared clip and the completed narration mix.
- Adjusts overlong lines moderately; reports when dialogue may overlap.
- Mixes narration with original audio (duck, mix or mute).
- Supports hard subtitles, soft MP4 subtitles, both, or no embedded subtitles.
- Renders H.264/AAC MP4 and verifies video/audio/subtitle streams.

### v0.7 — Project recovery
- Writes a `movie.project.json` manifest beside the transcript.
- Tracks each stage, progress, errors, checkpoints and recent history.
- Reconciles stage status with file hashes when a project is reopened.
- Resume starts at the first incomplete stage; each service also checks its own per-chunk/per-batch cache.
- Selecting a source video searches for its registered transcript/project, including after restarting the app. The registry is stored at `~/.video_translator/projects.json`; opening an existing transcript once registers older projects.
- Context chapter size is four times the translation batch setting (minimum 40 segments); with the default batch size of 30, the initial chapter size is 120 segments. Oversized responses are split further automatically.
- Gemini model selection is not part of the resumable context/translation cache signature. Completed chapters and batches remain reusable; the selected model handles only unfinished requests. Completed Gemini stages are not regenerated just because the model changed.
- Changed transcript/context/translation/voice/render settings invalidate only dependent artifacts, not unrelated earlier stages.

## Requirements

- Python 3.10+.
- FFmpeg and ffprobe available on PATH.
- FFmpeg build with `libass` for hard/burned subtitles.
- Gemini API key.
- VieNeu Cloud API key.
- Internet access for source download, first Whisper model download and API calls.

## Windows setup

```powershell
py -3.11 -m venv .venv
.venv`Scripts`Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python main.py
```

Edit `.env` and set `GEMINI_API_KEY` and `VIENEU_API_KEY`. Never commit `.env`.

## macOS / Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
cp .env.example .env
python main.py
```

## Typical workflow

1. **Video & Transcript** — paste a YouTube/Bilibili URL or choose a local video; download and transcribe.
2. **Gemini Film Translation** — analyze the whole transcript, then translate and QC.
3. **VieNeu Dubbing** — load v4 voices, choose a voice and speed, then synthesize each segment.
4. **Project & Export** — open an existing transcript or enter a video URL/local file, choose subtitle/audio options, then press **Chạy toàn bộ / Tiếp tục project**.

The single pipeline action checks every stage in order and skips matching completed artifacts. On a new project it downloads/transcribes/analyzes/translates/dubs/renders; on restart it resumes only incomplete work. If TTS has already submitted jobs, it restores the saved voice configuration and resumes the existing job IDs.

## Cache and recovery behavior

All durable project data is stored beside the output project:

- `.video_translator_cache/<movie>/source_16k_mono.wav` — reusable extracted audio.
- `.video_translator_cache/<movie>/chunks/` — split audio chunks.
- `.video_translator_cache/<movie>/transcription_state.json` — Whisper chunk results and fingerprints.
- `<movie>.transcript.json` / `<movie>.source.srt` — current partial or complete transcript.
- `<movie>.context.json` — chapter-by-chapter Gemini context checkpoint.
- `<movie>.vi.translation.json` — translation, pending translation/QC responses and completed batches.
- `<movie>.vi.srt` / `<movie>.vi.quality.json` — Vietnamese subtitles and QC report.
- `<movie>.tts/manifest.json` — VieNeu job IDs, idempotency keys, request signatures and completed audio paths.
- `<movie>.tts/prepared/` — audio timeline clip cache and prepared timeline state.
- `<movie>.tts/narration.wav` / `timeline.json` — completed speech mix and timeline.
- `<movie>.vi.dubbed.mp4` / `.render.json` — final render and artifact fingerprints.
- `<movie>.project.json` — project stages and history.

If Gemini returns a valid translation and then the QC call fails, the translation response is checkpointed before QC; the next run retries QC only. Context chapter responses and the final context response are also saved before profile consolidation. If the application closes between Whisper chunks, completed chunks are skipped. VieNeu request keys are written before submission and job IDs immediately after acceptance, so a restarted run polls the same job rather than creating a duplicate. If the application closes during MP4 rendering, the incomplete output is discarded and rendering is repeated from cached narration/TTS/prepared clips—not from Gemini or VieNeu.

Do not delete `.video_translator_cache`, `.tts` or project JSON files if you want to resume from previous work. API keys are read from `.env` or can be entered in the app; they are not written to project files.

## Output and synchronization notes

- MP4 video resolution follows the source resolution up to 1080p.
- Soft subtitles use MP4 `mov_text` and can be toggled by the player.
- Hard subtitles are burned into the picture and require FFmpeg `libass`.
- “Duck original audio” currently lowers the source track to 18% under narration; it is a fixed reduction rather than dynamic sidechain ducking.
- VieNeu TTS audio is generated one segment at a time. A very long spoken line can still overlap the next dialogue; the timeline builder applies a moderate speed-up and records warnings rather than changing source timestamps.
- Speaker diarization is not included yet; `speaker_id` remains null unless supplied by a future upstream stage.

## Tests

```bash
python -m compileall -q main.py video_translator tests
python -m unittest discover -s tests -v
```

CI runs Python 3.11 and 3.12 and installs FFmpeg so the final MP4 render path is integration-tested with generated sample media. Tests mock Gemini/VieNeu requests; no live API calls or paid credits are used in CI.

## Roadmap

- [x] 0.1 Download and desktop foundation
- [x] 0.2 Chunked Whisper transcription with resume
- [x] 0.3 Gemini movie context
- [x] 0.4 Contextual translation and QC
- [x] 0.5 VieNeu TTS
- [x] 0.6 Audio timeline, subtitles and MP4 rendering
- [x] 0.7 Project state and crash recovery

## Rights and access

Only download and process videos you are authorized to use. The application does not bypass DRM, paywalls, or platform access controls.
