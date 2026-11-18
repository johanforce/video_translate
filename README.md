# VideoTranslator

Python desktop application for translating and dubbing authorized YouTube/Bilibili video content into Vietnamese.

The application is developed incrementally. The current version includes video downloading and timestamped speech transcription. Gemini film-context analysis, Vietnamese dialogue translation, VieNeu dubbing, and final rendering are subsequent milestones.

## Current milestone

### v0.1 — Project foundation and video downloader
- PySide6 desktop interface.
- YouTube and Bilibili URL validation.
- Video download through yt-dlp.
- Prefer source streams at or below 1080p; never upscale lower-resolution sources.
- Merge/remux to MP4 using FFmpeg.

### v0.2 — Audio extraction and Whisper transcription
- Extract the first audio stream using FFmpeg.
- Normalize to mono, 16 kHz PCM WAV in a temporary workspace.
- Transcribe with faster-whisper.
- Automatic language detection or manual source-language selection.
- Segment-level timestamps and optional word-level timestamps.
- Export UTF-8 transcript JSON and source-language SRT.
- Run transcription in a background worker and allow cancellation between processing stages/segments.

The default model is \`small\`. The first run downloads the selected Whisper model and may take additional time. Larger models can improve recognition quality but require more memory and processing time.

## Requirements

- Python 3.10 or newer.
- FFmpeg available on PATH.
- Internet access for downloading video and the Whisper model on first use.
- Access to the video and permission to download/process it.

## Setup

### Windows

\`\`\`powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
\`\`\`

### macOS / Linux

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
\`\`\`

## v0.2 usage

1. Paste a YouTube/Bilibili URL and click **Tải MP4 tối đa 1080p**, or choose a local video file.
2. Select a Whisper model. \`small\` is the default balance of quality and speed.
3. Leave language set to **Tự nhận diện**, or choose the known source language.
4. Click **Nhận diện lời thoại**.
5. The output folder receives:
   - \`<video-name>.transcript.json\` — language metadata, segments, timestamps, and word timings.
   - \`<video-name>.source.srt\` — source-language subtitles.

The JSON is the canonical input for later Gemini context analysis and translation. Segment IDs are assigned in chronological order. Speaker IDs remain null because this milestone does not perform speaker diarization.

## Run tests

\`\`\`bash
python -m unittest discover -s tests -v
\`\`\`

## Download behavior

- The downloader selects the best available video/audio streams with a maximum source height of 1080p.
- MP4-compatible streams are preferred; FFmpeg is used to merge/remux streams.
- If the source is only 720p (or lower), the output remains at that source resolution.
- Playlist downloads are disabled.
- The downloader does not bypass DRM, paywalls, or platform access controls.

## Gemini film translation design

Prompt assets are located in \`video_translator/prompts/\`. The next translation milestones will:
1. Analyze the transcript and build a persistent film context profile.
2. Translate timestamped batches using that profile and neighboring dialogue.
3. Check omissions, terminology, character consistency, natural spoken Vietnamese, and dubbing suitability.

## Project structure

\`\`\`text
video_translator/
  config/          Application settings
  core/            Shared data models and exceptions
  download/        YouTube/Bilibili downloader
  transcription/   FFmpeg extraction, faster-whisper, JSON/SRT
  prompts/         Gemini film-translation prompt assets
  ui/              PySide6 interface and background workers
tests/             Unit tests
main.py            Application entry point
\`\`\`

## Roadmap

- [x] 0.1 Foundation, desktop shell, and video downloader
- [x] 0.2 Audio extraction and Whisper transcription with timestamps
- [ ] 0.3 Gemini film-context analysis and context profile
- [ ] 0.4 Contextual Vietnamese film translation and quality control
- [ ] 0.5 VieNeu TTS integration
- [ ] 0.6 Audio timeline, subtitles, and FFmpeg rendering
- [ ] 0.7 Project history, resume/cache, preview, and packaging

## Security

Store API keys in environment variables or a local \`.env\` file. Never commit real API keys, cookies, or private credentials.
