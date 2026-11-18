# VideoTranslator

Python desktop application for translating and dubbing authorized YouTube/Bilibili video content into Vietnamese.

The project is being developed incrementally. The first milestone (0.1) establishes the desktop shell and a reliable MP4 downloader. Subsequent milestones add speech transcription, film-context analysis, Vietnamese translation, VieNeu voice generation, subtitle synchronization, and final rendering.

## Current milestone

### v0.1 — Project foundation and video downloader
- PySide6 desktop interface.
- YouTube and Bilibili URL validation.
- Video download through yt-dlp.
- Prefer source streams at or below 1080p; never upscale lower-resolution sources.
- Merge/remux to MP4 using FFmpeg.
- Download progress and readable error messages.

Gemini film-translation prompts are included as project assets and will be wired into the translation pipeline in a later milestone.

## Requirements

- Python 3.10 or newer.
- FFmpeg (both \`ffmpeg\` and \`ffprobe\`) available on PATH.
- Internet access to the selected video platform.
- Access to the video and permission to download/process it.

## Setup

### Windows
1. Install Python 3.10+.
2. Install FFmpeg and add its \`bin\` directory to PATH.
3. Open a terminal in this repository.
4. Create and activate a virtual environment:

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

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Download behavior

- The downloader selects the best available video/audio streams with a maximum source height of 1080p.
- MP4-compatible streams are preferred; FFmpeg is used to merge/remux streams.
- If the source is only 720p (or lower), the output remains at that source resolution. The application does not claim that an upscaled file is native 1080p.
- Playlist downloads are disabled for the initial milestone.
- The downloader does not bypass DRM, paywalls, or platform access controls.

## Gemini film translation design

Prompt assets are located in \`video_translator/prompts/\`.

The planned translation workflow is:
1. Analyze the full transcript and build a persistent film context profile.
2. Translate small timestamped batches using that profile and neighboring dialogue.
3. Run a quality-control pass for omissions, consistency, natural spoken Vietnamese, and dubbing suitability.

The translation contract preserves segment IDs and source timestamps. Gemini is instructed not to invent dialogue, merge segments, or change the timeline.

## Project structure

\`\`\`text
video_translator/
  config/       Application settings
  core/         Shared data models and errors
  download/     URL validation and yt-dlp integration
  prompts/      Gemini film translation prompt assets
  ui/           PySide6 desktop interface
main.py         Application entry point
\`\`\`

## Roadmap

- [x] 0.1 Foundation, desktop shell, and video downloader
- [ ] 0.2 Audio extraction and Whisper transcription with timestamps
- [ ] 0.3 Gemini film-context analysis and context profile
- [ ] 0.4 Contextual Vietnamese film translation and quality control
- [ ] 0.5 VieNeu TTS integration
- [ ] 0.6 Audio timeline, subtitles, and FFmpeg rendering
- [ ] 0.7 Project history, resume/cache, preview, and packaging

## Security

Store API keys in environment variables or a local \`.env\` file. Never commit real API keys, cookies, or private credentials.
