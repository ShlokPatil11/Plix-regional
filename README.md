# Directory Language Studio

This is the deployable web application that embeds the processing code and required assets from both services under `vendor/`. It does not depend on sibling checkout paths.

```text
video upload
  -> Service A: translated on-screen video + visual SRT
  -> analyze: transcript, shot cuts, face candidates, background music (Shazam -> YouTube -> MP3)
  -> pause: user confirms which face to lip-sync (status awaiting_face_selection)
  -> Service B: voice-only dub, lip-sync on shots with the locked face only
  -> final mix: voice + music x slider% + Service A visual SRT track
  -> final MP4 + separate Service A and Service B SRT downloads
```

The two subtitle files describe different content. Service B never receives Service A's visual SRT as its transcript. Service B transcribes and dubs the spoken audio from Service A's rendered video. Its returned video keeps Service A's burned visual translation; the final mux also attaches the actual translated visual track from Service A as a selectable subtitle track.

## Run locally

From this directory, install `requirements.txt`, install FFmpeg and Tesseract, configure the environment variables below, then run:

```sh
uvicorn dir_language_pipeline.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Choose **Masking** to run Service A alone, **Dubbing** to run Service B alone on the uploaded video, or **Combine** to run Service A followed by Service B and mux their outputs. Each action accepts the selected target language and optional spoken source language. `POST /api/jobs` accepts the same fields plus `operation` (`masking`, `dubbing`, or `combine`; defaults to `combine`). It returns a job ID; poll `GET /api/jobs/{job_id}` and download the artifacts returned when processing completes.

## Run with Docker

Create `.env` from `.env.example` and add the provider keys you need. From this directory, build and start the app with:

```sh
docker compose up --build
```

Open `http://localhost:8000`. Compose keeps job data in the `directory-language-data` volume mounted at `/data`; stop the app with Ctrl+C. To run the image without Compose, build and start it directly:

```sh
docker build -t directory-language-studio .
docker run --env-file .env -p 8000:8000 -v directory-language-data:/data directory-language-studio
```

## Face lock and background music

Dubbing and Combine now pause after analysis. The job page shows the faces that appear while someone speaks; confirm the speaker and the job resumes (`POST /api/jobs/{job_id}/face` with `{"candidate_id": "0"}`). Only shots where that person is visible are lip-synced; Sync Labs receives per-frame bounding boxes of the locked face. If no face is found the video is dubbed audio-only. Jobs waiting for confirmation expire after 24 hours.

The ad's background music is identified with Shazam (`shazamio`, no API key), downloaded from YouTube with `yt-dlp`, aligned to the video, and mixed with the dubbed voice. Use the **Music level** slider on the result page (or `POST /api/jobs/{job_id}/remix` with `{"music_percent": 0-100}`) to change the mix; this runs only ffmpeg. If the music is not identified or cannot be downloaded the video is voice-only and the slider is disabled. `music.json` records the artist, title, and YouTube URL used.

Downloading commercial tracks from YouTube is a licensing and terms-of-service decision for ads that will run; review it before shipping.

To run the tests, use:

```sh
.venv/bin/python -m pytest tests -q
```

Configure the required secrets in `.env`:

- `GEMINI_API_KEY` for Service A visual translation (default model: `gemini-3.6-flash`). Create the key in Google AI Studio and set it as a deployment secret. Set `TRANSLATION_PROVIDER=claude` plus `ANTHROPIC_API_KEY` and `TRANSLATION_MODEL` only if you want Claude instead.
- `ELEVENLABS_API_KEY` for Service B.
- `SYNC_API_KEY` when using its Sync Labs lip-sync stage.
- `HF_TOKEN` is optional for the configured Hugging Face lip-sync path.

Jobs and generated artifacts live in the mounted `/data` volume. `PIPELINE_MAX_UPLOAD_BYTES` controls the upload limit. Run one API process per data volume because the embedded services use process-wide Python module state; the included worker serializes media jobs.

## Process logs and failures

The worker prefixes each job's Python output and stage changes with its job ID and streams them to the terminal/container log where Uvicorn is running. It also saves the complete stdout, stderr, Python logging output, and failure traceback in that job's `process.log`. Open **View complete process log** on the job page, or fetch `GET /api/jobs/{job_id}/logs`. The job status response includes the underlying Service A or Service B error when available.

On a fresh deployment, Service B downloads its faster-whisper model at runtime. The host must allow outbound access to the configured AI providers and model service.

## Artifacts

- `final_video`: final MP4, with Service A visual subtitles burned in and also attached as a selectable subtitle track.
- `visual_subtitles`: the translated visual SRT from Service A.
- `audio_subtitles`: the independent transcript/dub SRT from Service B.
- `voice_video`: voice-only video before the music mix.
- `music_track`: aligned MP3.
- `music_info`: `music.json`.
- `service_a_video`, `service_b_video`, and `service_b_audio`: retained stage outputs for inspection.

The orchestrator stores stage status and artifact URLs in SQLite and each job's `manifest.json`. The host filesystem paths stay internal to the container. If the app restarts during a running job, it queues that job again from the uploaded source.
