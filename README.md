# VideoTranslator

Ứng dụng desktop Python hỗ trợ dịch thoại phim theo ngữ cảnh và tạo giọng thuyết minh tiếng Việt.

## Phiên bản hiện tại

### v0.1 — Download
- YouTube và Bilibili qua yt-dlp.
- MP4, ưu tiên nguồn tối đa 1080p; không upscale nguồn thấp hơn.
- FFmpeg ghép/remux audio và video.

### v0.2 — Transcription
- FFmpeg trích xuất mono 16 kHz WAV.
- faster-whisper nhận diện lời thoại.
- Tự nhận diện hoặc chọn ngôn ngữ nguồn.
- Timestamp theo đoạn và theo từ.
- Xuất transcript JSON và SRT gốc.

### v0.3 — Movie Context
- Gemini đọc transcript theo chapter.
- Tạo Context Profile gồm nội dung phim, nhân vật, quan hệ, cách xưng hô, thuật ngữ, mạch liên tục và điểm chưa chắc chắn.
- Lưu checkpoint theo chapter; có thể chạy lại sau lỗi.
- Nếu transcript không đổi và profile đã hoàn tất, ứng dụng tái sử dụng kết quả, không gọi Gemini lần nữa.

### v0.4 — Film Translation
- Dịch lời thoại theo batch, có context profile và các câu lân cận.
- Việt hóa tự nhiên cho lồng tiếng, giữ tính cách và cảm xúc nhân vật.
- Kiểm tra chất lượng từng batch.
- Từ chối kết quả nếu Gemini làm đổi ID, thứ tự, số lượng hoặc timestamp.
- Xuất JSON tiếng Việt, SRT tiếng Việt và báo cáo QC.
- Lưu checkpoint; chỉ dịch lại batch còn thiếu khi chạy tiếp.

### v0.5 — VieNeu TTS
- Lấy danh mục giọng v4 trực tiếp từ VieNeu.
- Tạo một audio riêng cho mỗi câu thoại tiếng Việt.
- Điều chỉnh tốc độ đọc.
- Lưu manifest, job ID và idempotency key.
- Bỏ qua audio đã hoàn thành; tiếp tục poll job còn dang dở sau khi mở lại.
- Cảnh báo số request trước khi tạo giọng.

Lưu ý: v0.5 tạo audio cho từng câu thoại. Ghép audio theo timeline, mix âm thanh gốc và render MP4 sẽ được làm ở v0.6.

## Yêu cầu

- Python 3.10+.
- FFmpeg có cả ffmpeg và ffprobe trên PATH.
- Gemini API key.
- VieNeu Cloud API key.
- Kết nối mạng để tải video, model Whisper lần đầu và gọi API.

## Cài đặt Windows

\`\`\`powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python main.py
\`\`\`

Mở file .env và điền GEMINI_API_KEY, VIENEU_API_KEY. Không commit file .env.

## macOS / Linux

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
cp .env.example .env
python main.py
\`\`\`

Điền API keys vào .env trước khi chạy Gemini/VieNeu.

## Quy trình sử dụng

1. Tab **Video & Transcript**: tải video hoặc chọn file có sẵn; chạy Whisper và tạo transcript.
2. Tab **Gemini Dịch phim**: chọn API key/model; chạy **Phân tích toàn phim** trước.
3. Sau khi Context Profile hoàn tất, chạy **Dịch phim + kiểm tra**.
4. Tab **VieNeu Thuyết minh**: tải danh sách giọng v4, chọn giọng/tốc độ và tạo audio theo từng câu thoại.

## File đầu ra

Với video movie.mp4, các file chính được lưu trong thư mục dự án:

- movie.transcript.json — transcript gốc, timestamp và word timings.
- movie.source.srt — phụ đề ngôn ngữ gốc.
- movie.context.json — Context Profile và checkpoint phân tích.
- movie.vi.translation.json — bản dịch theo segment, metadata cảm xúc và trạng thái.
- movie.vi.srt — phụ đề tiếng Việt.
- movie.vi.quality.json — lỗi và báo cáo kiểm tra chất lượng.
- movie.tts/manifest.json — trạng thái tạo giọng.
- movie.tts/segment_*.mp3|wav|... — audio từng câu (phần mở rộng theo định dạng trả về).

## Model Whisper

Mặc định là small. medium và large-v3 thường cần nhiều RAM/VRAM hơn và xử lý chậm hơn. Lần đầu chọn model sẽ tải model về máy.

## API cost và giới hạn

- Gemini được gọi theo chapter để phân tích context; dịch và QC là hai lượt xử lý cho mỗi batch.
- VieNeu tạo một job cho mỗi câu thoại có nội dung; chi phí phụ thuộc số ký tự và gói VieNeu.
- Ứng dụng hiển thị số segment trước khi gọi TTS. Hãy kiểm tra quota và chi phí tài khoản VieNeu.
- Dừng tác vụ không xóa các job đã gửi lên nhà cung cấp. Manifest giữ ID để có thể tiếp tục poll.

## Kiểm thử

\`\`\`bash
python -m compileall -q main.py video_translator tests
python -m unittest discover -s tests -v
\`\`\`

CI chạy trên Python 3.11 và 3.12.

## Kiến trúc

\`\`\`text
video_translator/
  config/          Runtime settings
  core/            Models and errors
  download/        yt-dlp
  transcription/   FFmpeg + faster-whisper + JSON/SRT
  prompts/         Film translation prompts
  gemini/          Context analysis + translation + QC
  tts/             VieNeu v4 async TTS
  ui/              PySide6 + background workers
tests/
main.py
\`\`\`

## Roadmap

- [x] 0.1 Download and desktop foundation
- [x] 0.2 Audio extraction and transcription
- [x] 0.3 Gemini movie context
- [x] 0.4 Contextual translation and QC
- [x] 0.5 VieNeu TTS per dialogue segment
- [ ] 0.6 Audio timeline, subtitle mux/burn, final MP4 rendering
- [ ] 0.7 Project history, editing, preview, packaging

## Quyền sử dụng nội dung

Chỉ tải và xử lý video mà bạn có quyền sử dụng; không vượt DRM, paywall hoặc cơ chế kiểm soát truy cập của nền tảng.
