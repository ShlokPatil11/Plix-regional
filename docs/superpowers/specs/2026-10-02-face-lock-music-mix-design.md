# Face-Lock Lip-Sync and Background-Music Mix: Design

Date: 2026-10-02
Status: Draft for review

## Goal

Convert UGC ads into another language end to end: translated on-screen text, a dubbed voice with accurate lip-sync on the right person, and the original background music restored at an adjustable level.

Manager's requests:

1. Subtitle overlay on the English on-screen text (already provided by Service A).
2. Proper lip-sync on the desired speaker, with an auto-lock face feature.
3. Detect the background music, fetch it from YouTube, convert to MP3, and mix it with the dubbed audio at a percentage chosen with a slider.

## Decisions already made

| Topic | Decision |
|---|---|
| Music slider | Lives on the result page. Pipeline outputs a voice-only video plus a separate music track. A remix endpoint re-mixes with ffmpeg in seconds. No paid step re-runs. |
| Face selection | Auto-detect candidates, then a **user confirm step** before any paid lip-sync runs. |
| Music identification | `shazamio`, behind a swappable `MusicIdentifier` interface. |
| No match / download fails | Ship voice-only, flag "music not identified", disable the slider. No stem separation (no Demucs/torch). |
| Pause mechanism | Two-phase job with disk-cache resume. The worker is released while awaiting confirmation. |

## Non-goals

- Manual face click on the upload form (the confirm step covers correction).
- Automatic ducking of music under speech.
- Source-separation of the original music (Demucs).
- Changes to Service A's masking behavior. A regression check covers it.
- Live drag-to-remix. The slider uses an Apply button.

## Job flow

```text
upload
 -> Service A: translated on-screen text (unchanged)
 -> ANALYZE (no paid dubbing yet)
      transcript (Whisper) + scene cuts + face candidates
      music ID (Shazam, speech-gap windows) -> YouTube -> mp3
 -> PAUSE: status = awaiting_face_selection
      UI shows candidate face crops
 -> user confirms a face -> job re-queued
 -> RENDER
      ElevenLabs dub, voice-only (background audio dropped)
      lip-sync only on shots where the locked face appears,
        sending Sync Labs per-frame bounding boxes of the locked face
      stitch -> voice-only video
 -> MIX: voice + music x slider% -> final_video.mp4
```

Applies to the Dubbing and Combine operations. Masking-only is unchanged.

Service B's engine already caches the transcript and `scenes.json` under `cache/service-b/<job_id>`, so the resume re-uses analyze-phase work.

## Component 1: Face lock (`faces.py`)

### Candidate discovery (analyze phase)

1. Sample frames across shots that overlap speech. Detect faces with YuNet. Keep faces at least 10% of frame height (existing `MIN_FACE_HEIGHT_RATIO`).
2. Compute SFace embeddings. Group into identities by cosine similarity at the existing 0.70 threshold.
3. Rank identities by speech-overlapping screen time, weighted by face size. Keep up to 4.
4. Save one crop per identity. The top-ranked identity is the recommended default.

### Confirmation

The result page shows crops with "appears in N shots, about X seconds on screen". The user picks one and presses **Confirm & continue**.

- One candidate: one-click confirm.
- No usable face: skip the pause, dub audio-only for the whole video, flag "no face found, no lip-sync".

### The lock

The anchor is the mean embedding of the chosen identity, saved to `face_lock.json` in the job folder. Re-runs and resumes reuse it.

### Render-phase rule, per shot with speech

- Run `FaceIDTracker.match_interval`. If the locked identity is present in at least 40% of sampled frames, the shot is lip-synced.
- For lip-synced shots, build per-frame `bounding_boxes` for the locked face (`[x1,y1,x2,y2]`, `null` where absent) and send them as `options.active_speaker_detection.bounding_boxes` with `auto_detect` off. Verified against Sync's Speaker Selection docs (https://docs.sync.so/developer-guides/speaker-selection). Fallback if boxes prove flaky: `frame_number` plus `coordinates` (must be sent together).
- Otherwise the shot is B-roll: dubbed audio over the original video.
- Existing fallbacks stay: clips under 1 second and Sync Labs errors get an audio-only mux.
- `classifier.classify_shot` is unchanged. The analyze phase runs before a face is locked, so the identity gate is applied at render time inside `lipsyncer.process_shots`. Per-shot decisions (`lip_sync`, `locked_face_ratio`) are written back to `scenes.json`.

### Risks

- Matching is weaker on extreme angles, sunglasses, and heavy filters. The mean embedding, the 40% rule, and the confirm step mitigate this.

## Component 2: Music (`music.py`)

### Identify

- From the source audio, choose up to 3 windows of about 12 seconds with the least speech, using Whisper word timestamps from stage 1.
- Run `shazamio` on each. The first confident match gives artist, title, and the position within the song. The exact offset field in Shazam's response is to be verified during implementation.
- No match: write `music.json` with `{"status": "not_identified"}`.

### Fetch

- Add `yt-dlp` to `requirements.txt`.
- Search "artist title audio". Choose by duration close to Shazam's track length, preferring official-audio or Topic uploads, and skipping titles containing live, cover, remix, or sped-up.
- Download best audio, convert to MP3 with ffmpeg. Failure records the reason and falls back to voice-only.

### Align

Use the match position to find where the ad's music starts inside the track. Trim from there to the video length with a short fade-out.

### Licensing note

Pulling commercial tracks from YouTube is a licensing and terms-of-service issue for ads that run. `music.json` records the source URL for an audit trail. Whether to ship this is the manager's decision.

## Component 3: Mixer (`mixer.py`)

- Inputs: voice-only video, aligned `music_track.mp3`, `music_percent`, visual SRT.
- Mix: `voice x 1.0 + music x (percent / 100)`. Music is loudness-normalized first, so 100% is about voice loudness. Default 25%. 0% is voice only.
- Video stream is copied, not re-encoded. The visual SRT is re-attached as a selectable subtitle track, as `FfmpegFinalAssembler` does today. This module replaces that mux step.
- Output is a new `final_video`. Nothing paid re-runs.

## API and state

- New job status: `awaiting_face_selection`.
- New job-record fields: face choice and music result.
- Endpoints:
  - `GET /api/jobs/{id}/faces`: candidate crops and stats.
  - `POST /api/jobs/{id}/face`: confirm a candidate. Idempotent. Re-queues the job.
  - `POST /api/jobs/{id}/remix` with `{music_percent}`: re-run the mix only.
- A job awaiting confirmation for 24 hours expires.
- A restart while awaiting confirmation leaves the job awaiting.
- New artifacts: `music_track.mp3`, `music.json`, `voice_video.mp4` (published in both Combine and Dubbing operations).

## UI (`static/index.html`)

- Upload form unchanged.
- Face panel when awaiting: crops (recommended pre-selected), "Music found: Artist - Title" or "not identified", **Confirm & continue**.
- Result panel: player, **Music level** slider (0-100, default 25) with **Apply**, existing downloads. Slider disabled with an explanation when no music was identified.

## Error handling

Every soft failure degrades to voice-only or audio-only and is written to `process.log`: music ID or download failure, no face found, shot without the locked face, Sync Labs rejection. ElevenLabs failure and corrupt uploads remain hard failures.

## Changes to existing code

- New modules in the app: `music.py`, `mixer.py`. `faces.py` and `analysis.py` live in `vendor/service_b/src/` because `lipsyncer.py` imports them and the face models live there.
- `api.py`, `job_store.py`, `in_process.py`, `service_adapters.py`: pause/resume, new endpoints, wiring.
- Vendored `engine.py`: split into analyze and render phases.
- Vendored `lipsyncer.py`: locked-face gating and `bounding_boxes`.
- Vendored `el_dubber.py`: request voice-only audio (drop background).
- `requirements.txt`: `yt-dlp`, `shazamio`, `pytest`.

## Testing

No paid APIs in automated tests.

- Identity clustering and ranking on fixture frames.
- Shot-gating rule and the Sync `bounding_boxes` payload builder.
- Music window picker, YouTube result chooser, ffmpeg trim and mix commands on generated sine-wave audio.
- Job-store pause/resume state machine.
- Remix endpoint via FastAPI's test client.
- Regression: the combined flow still burns translated text over the English and attaches the SRT track.
- Fakes for `MusicIdentifier` and the downloader.
- Manual end-to-end run with real Shazam, ElevenLabs, and Sync keys.

## Open verification items for implementation

1. Shazam response: exact field for the in-song offset.
2. ElevenLabs Dubbing API: confirm the parameter name for dropping background audio.
3. Sync `bounding_boxes` behavior on real clips. Fall back to coordinates if needed.
