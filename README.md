# Directory Language Studio

This is the deployable web application that embeds the processing code and required assets from both services under `vendor/`. It does not depend on sibling checkout paths.

```text
video upload
  -> analyze: transcript, face candidates, background music (Shazam -> YouTube -> MP3)
  -> pause: user confirms which face to lip-sync (status awaiting_face_selection)
  -> Service B: one continuous voice dub + full-video Sync Labs lip-sync
     -> restore classified B-roll ranges from the untouched source video
  -> Service A: translate/mask on-screen text on the lip-synced video + visual SRT
  -> final mix: dubbed voice + music x slider% + Service A visual SRT track
  -> final MP4 + separate Service A and Service B SRT downloads
```

The two subtitle files describe different content. Service B never receives Service A's visual SRT as its transcript. Service B transcribes and dubs the spoken audio from Service A's rendered video. Its returned video keeps Service A's burned visual translation; the final mux also attaches the actual translated visual track from Service A as a selectable subtitle track.

## Run locally

From this directory, install `requirements.txt`, install FFmpeg and Tesseract, configure the environment variables below, then run:

```sh
uvicorn dir_language_pipeline.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Choose **Masking** to run Service A alone, **Dubbing** to run Service B alone on the uploaded video, or **Combine** to run Service A followed by Service B and mux their outputs. Each action accepts the selected target language and optional spoken source language. `POST /api/jobs` accepts the same fields plus `operation` (`masking`, `dubbing`, or `combine`; defaults to `combine`). It returns a job ID; Dubbing and Combine pause at `awaiting_face_selection` after analysis until the face is confirmed (in the UI, or via `POST /api/jobs/{job_id}/face`), then continue; a job in which no usable face is found skips the pause. Masking is unchanged and does not pause. Poll `GET /api/jobs/{job_id}` and download the artifacts returned when processing completes.

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

Configure the required secrets in `.env`:

- `GEMINI_API_KEY` for Service A visual translation (default model: `gemini-3.6-flash`). Create the key in Google AI Studio and set it as a deployment secret. Set `TRANSLATION_PROVIDER=claude` plus `ANTHROPIC_API_KEY` and `TRANSLATION_MODEL` only if you want Claude instead.
- `ELEVENLABS_API_KEY` for Service B.
- `SYNC_API_KEY` when using its Sync Labs lip-sync stage.
- `HF_TOKEN` is optional for the configured Hugging Face lip-sync path.

Jobs and generated artifacts live in the mounted `/data` volume. `PIPELINE_MAX_UPLOAD_BYTES` controls the upload limit. Run one API process per data volume because the embedded services use process-wide Python module state; the included worker serializes media jobs.

## Run the tests

From the project root, run:

```sh
.venv/bin/python -m pytest tests -q
```

Always pass `tests` (not just `pytest -q`) because the project directory name contains a hyphen, which makes a bare `pytest` fail to collect from the hyphenated `Plix-Project-1` path.

## Face lock and background music

Dubbing and Combine pause for face confirmation. The UI shows the faces that appear while someone speaks; confirm the speaker there, or via `GET /api/jobs/{job_id}/faces`. The UI puts the job ID in the address bar, so if you reload or close the tab, reopen the app URL with `?job=<job_id>` (for example `http://localhost:8000/?job=<job_id>`) to pick the job up again. You can also resume via `POST /api/jobs/{job_id}/face` with `{"candidate_id": "0"}`. The complete video and extracted dubbed audio are sent to Sync Labs, with per-frame bounding boxes locking the selected face. After sync, ranges classified as B-roll by Service B are replaced with the same ranges from the untouched source video. If no face is found the video is dubbed audio-only and the job does not pause. Jobs waiting for confirmation expire after 24 hours.

The ad's background music is identified with Shazam (`shazamio`, no API key), downloaded from YouTube with `yt-dlp`, aligned to the video, and mixed with the dubbed voice. Use the **Music level** slider on the result page (or `POST /api/jobs/{job_id}/remix` with `{"music_percent": 0-100}`) to change the mix; the default level is 25%, and changing it runs only ffmpeg. If the music is not identified or cannot be downloaded the video is voice-only and the slider is disabled. `music.json` records the artist, title, and YouTube URL used.

**Custom music.** On a finished job, open **Use custom music instead** under the video to replace the detected track (or to add one when nothing was detected). Upload an MP3, WAV, M4A, AAC, OGG or FLAC file (up to 50 MB), or paste a single YouTube link (up to 15 minutes; playlists and other sites are rejected). The API is `POST /api/jobs/{job_id}/music` (multipart form: `file` or `url`, plus an optional `music_percent`). The custom track starts at 0:00 and is trimmed to the video length; it replaces the detected one, and the level slider and `/remix` then work on it as usual. `music.json` records `source: "custom"` and, for a YouTube link, its URL.

The job status response now includes a `music` field. Downloading commercial tracks from YouTube is a licensing and terms-of-service decision for ads that will run; review it before shipping.

## Process logs and failures

The worker prefixes each job's Python output and stage changes with its job ID and streams them to the terminal/container log where Uvicorn is running. It also saves the complete stdout, stderr, Python logging output, and failure traceback in that job's `process.log`. Open **View complete process log** in the UI, or fetch `GET /api/jobs/{job_id}/logs`. The job status response includes the underlying Service A or Service B error when available.

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
