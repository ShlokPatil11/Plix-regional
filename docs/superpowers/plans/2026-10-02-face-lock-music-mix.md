# Face-Lock Lip-Sync and Background-Music Mix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user confirm which on-screen face to lip-sync (auto-detected, paused for confirmation), identify the ad's background music with Shazam, fetch it from YouTube as MP3, and mix it with the dubbed voice at a percentage the user adjusts with a slider after the job.

**Architecture:** The dubbing job becomes two-phase. An analyze phase (transcript, scene cuts, face candidates, music ID and fetch) ends by pausing the job in a new `awaiting_face_selection` state. Confirming a face re-queues the job, which resumes from disk caches and renders: ElevenLabs voice-only dub, lip-sync only on shots containing the locked face (Sync Labs gets per-frame bounding boxes), then an ffmpeg mix of voice and music. A remix endpoint re-runs only the ffmpeg mix.

**Tech Stack:** Python 3.11/3.13, FastAPI, SQLite, OpenCV (YuNet + SFace), ffmpeg, `shazamio`, `yt-dlp`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-face-lock-music-mix-design.md`

All paths below are relative to the repo root `Plix-Project-1/`. Run every command from that directory. Use the project venv: `.venv/bin/python`.

## Spec deviations decided while planning (the spec is amended in the same commit as this plan)

1. `faces.py` lives in `vendor/service_b/src/` (not the app package), because `lipsyncer.py` imports it and the face models live in Service B.
2. `classifier.py` is **not** changed. The analyze phase runs before a face is locked, so the identity gate is applied at render time inside `lipsyncer.process_shots`. Per-shot decisions are still written to `scenes.json`.
3. The voice-only pre-mix video is published as `voice_video.mp4` in both operations.

## Global Constraints

- Face "same person" similarity threshold: **0.70** (SFace cosine, existing repo value). Locked face must be present in at least **40%** of a shot's frames to be lip-synced.
- Minimum face height for candidates and tracking: **10%** of frame height. Maximum **4** face candidates.
- Music slider range: integer **0-100**, default **25**. 0 means voice only. Music is loudness-normalized to **-16 LUFS** before the percentage gain.
- Music identification uses up to **3** windows of **12 s** with the least speech.
- Awaiting-confirmation jobs expire after **24 hours**.
- ElevenLabs Mode A requests `drop_background_audio=true`.
- Sync Labs `active_speaker_detection` is sent as `{"auto_detect": false, "bounding_boxes": [...]}`; each box is `[x1, y1, x2, y2]` or `null` per frame.
- Soft failures (music, faces) never fail the job; hard failures (ElevenLabs, corrupt upload) keep today's behavior.
- No automated test calls a paid or network API. Real Shazam, YouTube, ElevenLabs and Sync Labs calls are only in the manual end-to-end check (Task 12).
- Do not modify unrelated vendored code. Follow the existing style (plain functions, `print` for job-log output, `PipelineError` for actionable errors in Service B).

## Review Focus

Failure modes the spec implies that no single feature task would otherwise pin down. Each has a test in the task that owns the code.

1. Source video has **no audio track**: music identification must soft-fail to `not_identified`, not crash (Task 3).
2. Video **shorter than one 12 s window**: still gets exactly one window (Task 2).
3. Music starts **after** the video starts (negative offset into the song): music is delayed, not trimmed negative (Task 2).
4. **No usable face** in the video: job continues audio-only with no pause and no Sync Labs call (Task 8 and Task 6).
5. Face confirmed **twice**, confirmed for a **wrong id**, or confirmed after the job is no longer waiting (Tasks 9 and 10).
6. Remix at the **extremes** 0 and 100, with **out-of-range** values, and on a job with **no identified music** (Tasks 1 and 10).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | modify | add `yt-dlp`, `shazamio`, `audioop-lts` (py3.13 only), `pytest` in dev file |
| `requirements-dev.txt` | create | `-r requirements.txt` + `pytest` |
| `tests/pytest.ini` | create | import mode (must live in tests/, see Task 0) |
| `tests/conftest.py`, `tests/helpers.py` | create | package registration, ffmpeg media helpers |
| `mixer.py` | create | voice + music x percent mix, subtitle track attach |
| `music.py` | create | window picking, Shazam identifier, YouTube fetcher, alignment, `identify_and_fetch` |
| `vendor/service_b/src/faces.py` | create | candidates, clustering, lock, per-frame boxes, `FaceEngine` |
| `vendor/service_b/src/analysis.py` | create | analyze phase (transcript + scenes) |
| `vendor/service_b/src/lipsyncer.py` | modify | lock-gated lip-sync with `bounding_boxes` |
| `vendor/service_b/src/el_dubber.py` | modify | `drop_background_audio` |
| `vendor/service_b/src/engine.py` | modify | accept `face_lock`, persist per-shot decisions |
| `contracts.py` | modify | `AwaitingFaceSelection`, new `PipelineResult` fields |
| `runner.py` | modify | pass-through of pause, music artifacts, assembler kwargs |
| `media.py` | modify | `FfmpegFinalAssembler` uses the mixer |
| `service_adapters.py` | modify | `TextToAudioAdapter` analyze, music, pause, lock |
| `in_process.py` | modify | `face_choice` plumbing, new artifacts, dubbing-only mix |
| `job_store.py` | modify | `awaiting_face_selection`, `face_choice`, expiry |
| `api.py` | modify | pause handling, `/faces`, `/face`, `/remix`, music in status |
| `static/index.html` | modify | face panel, music slider |
| `README.md` | modify | document the new flow |
| `tests/test_*.py` | create | one file per unit, listed in each task |

---

### Task 0: Test scaffolding and dependencies

**Files:**
- Create: `requirements-dev.txt`, `tests/pytest.ini`, `tests/conftest.py`, `tests/helpers.py`, `tests/test_smoke.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces (used by every later test): `helpers.make_video(path, seconds=2.0, freq=300, with_audio=True, size="160x90", fps=10) -> Path`, `helpers.make_tone(path, seconds=3.0, freq=800) -> Path`, `helpers.mean_volume_db(path, start=None, duration=None) -> float`, `helpers.stream_types(path) -> list[str]`, `helpers.media_duration(path) -> float`. The package is importable as `dir_language_pipeline`; Service B modules import as `src.*`.

- [ ] **Step 1: Add dependencies**

Append to `requirements.txt`:

```text
yt-dlp>=2024.8.6
shazamio>=0.7.0,<1
audioop-lts>=0.2.1; python_version >= "3.13"
```

Create `requirements-dev.txt`:

```text
-r requirements.txt
pytest>=8,<10
```

Create `tests/pytest.ini` (it must live in `tests/`: the repo directory name `Plix-Project-1` contains a hyphen, so an ini at the repo root makes pytest try to import the root `__init__.py` as a package and fail). Always run pytest as `.venv/bin/python -m pytest tests ...` from the repo root; a bare `pytest -q` is not supported.

```ini
[pytest]
addopts = --import-mode=importlib
```

- [ ] **Step 2: Install and smoke-check the new libraries**

Run:
```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -c "import yt_dlp, shazamio, pytest; print('imports ok')"
```
Expected: `imports ok`. If `shazamio` fails to import on Python 3.13 because of `audioop`, confirm `audioop-lts` installed; if a version pin fails to resolve, loosen the lower bound in `requirements.txt` and re-run.

- [ ] **Step 3: Write `tests/helpers.py`**

```python
import re
import subprocess
from pathlib import Path


def _run(command):
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{command[0]} failed: {completed.stderr}")
    return completed


def make_video(path, seconds=2.0, freq=300, with_audio=True, size="160x90", fps=10):
    path = Path(path)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "lavfi", "-i", f"testsrc=s={size}:r={fps}:d={seconds}"]
    if with_audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        command += ["-c:a", "aac", "-shortest"]
    command.append(str(path))
    _run(command)
    return path


def make_tone(path, seconds=3.0, freq=800):
    path = Path(path)
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
          "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}", str(path)])
    return path


def mean_volume_db(path, start=None, duration=None):
    command = ["ffmpeg", "-hide_banner", "-nostdin"]
    if start is not None:
        command += ["-ss", str(start)]
    if duration is not None:
        command += ["-t", str(duration)]
    command += ["-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"]
    completed = _run(command)
    match = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", completed.stderr)
    if not match:
        raise RuntimeError(f"no volume stats in: {completed.stderr}")
    return float("-inf") if match.group(1) == "-inf" else float(match.group(1))


def stream_types(path):
    completed = _run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                      "-of", "csv=p=0", str(path)])
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def media_duration(path):
    completed = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", str(path)])
    return float(completed.stdout.strip())
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
import importlib.util
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
SERVICE_B = APP_ROOT / "vendor" / "service_b"


def _register_package():
    """The app is deployed as the package `dir_language_pipeline`; mirror that here."""
    if "dir_language_pipeline" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "dir_language_pipeline", APP_ROOT / "__init__.py",
        submodule_search_locations=[str(APP_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["dir_language_pipeline"] = module
    spec.loader.exec_module(module)


_register_package()
if str(SERVICE_B) not in sys.path:
    sys.path.insert(0, str(SERVICE_B))
```

- [ ] **Step 5: Write the smoke test and run it**

`tests/test_smoke.py`:

```python
from helpers import make_video, stream_types


def test_package_and_service_b_import():
    from dir_language_pipeline.contracts import Artifact
    from src.audio import ffmpeg

    assert Artifact is not None
    assert ffmpeg()


def test_helpers_build_media(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=1.0)
    assert stream_types(video) == ["video", "audio"]
```

Run: `.venv/bin/python -m pytest tests/test_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt tests/
git commit -m "test: add pytest scaffolding and media helpers

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 1: Mixer

**Files:**
- Create: `mixer.py`, `tests/test_mixer.py`

**Interfaces:**
- Produces: `DEFAULT_MUSIC_PERCENT = 25`; `MixError`; `music_gain(percent: int) -> float`; `build_mix_command(ffmpeg_binary, voice_video, music, percent, output, subtitles=None) -> list[str]`; `mix_final(voice_video, music, percent, output, *, subtitles=None, ffmpeg_binary="ffmpeg") -> Path` (writes atomically; `music=None` or `percent=0` means voice only).

- [ ] **Step 1: Write the failing tests**

`tests/test_mixer.py`:

```python
import pytest
from helpers import make_tone, make_video, mean_volume_db, stream_types

from dir_language_pipeline.mixer import MixError, build_mix_command, mix_final, music_gain


@pytest.fixture
def media(tmp_path):
    voice = make_video(tmp_path / "voice.mp4", seconds=3.0, freq=300)
    music = make_tone(tmp_path / "music.mp3", seconds=3.0, freq=800)
    subtitles = tmp_path / "s.srt"
    subtitles.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    return voice, music, subtitles


def test_music_gain_bounds():
    assert music_gain(0) == 0.0
    assert music_gain(100) == 1.0
    assert music_gain(25) == 0.25
    for bad in (-1, 101, 12.5, "50", True):
        with pytest.raises(ValueError):
            music_gain(bad)


def test_zero_percent_keeps_voice_only_level(media, tmp_path):
    voice, music, _ = media
    out = mix_final(voice, music, 0, tmp_path / "out0.mp4")
    assert abs(mean_volume_db(out) - mean_volume_db(voice)) < 1.5


def test_music_makes_the_mix_louder_and_slider_is_monotonic(media, tmp_path):
    voice, music, _ = media
    quiet = mean_volume_db(mix_final(voice, music, 0, tmp_path / "a.mp4"))
    half = mean_volume_db(mix_final(voice, music, 50, tmp_path / "b.mp4"))
    full = mean_volume_db(mix_final(voice, music, 100, tmp_path / "c.mp4"))
    assert half > quiet + 1.0
    assert full > half + 1.0


def test_subtitle_track_is_attached_with_and_without_music(media, tmp_path):
    voice, music, subtitles = media
    with_music = mix_final(voice, music, 40, tmp_path / "m.mp4", subtitles=subtitles)
    without = mix_final(voice, None, 40, tmp_path / "n.mp4", subtitles=subtitles)
    assert "subtitle" in stream_types(with_music)
    assert "subtitle" in stream_types(without)
    assert stream_types(with_music).count("audio") == 1


def test_video_stream_is_copied_not_reencoded(media, tmp_path):
    voice, music, _ = media
    command = build_mix_command("ffmpeg", voice, music, 30, tmp_path / "o.mp4")
    assert command[command.index("-c:v") + 1] == "copy"


def test_failure_raises_mix_error_and_leaves_no_partial_file(media, tmp_path):
    _, music, _ = media
    missing = tmp_path / "missing.mp4"
    out = tmp_path / "out.mp4"
    with pytest.raises(MixError):
        mix_final(missing, music, 30, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".out.*"))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_mixer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dir_language_pipeline.mixer'`.

- [ ] **Step 3: Implement `mixer.py`**

```python
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

DEFAULT_MUSIC_PERCENT = 25
SUBTITLE_TITLE = "Visual subtitles (Service A)"


class MixError(RuntimeError):
    """The final voice + music mix could not be produced."""


def music_gain(percent: int) -> float:
    if isinstance(percent, bool) or not isinstance(percent, int) or not 0 <= percent <= 100:
        raise ValueError("music percent must be an integer from 0 to 100")
    return percent / 100


def build_mix_command(
    ffmpeg_binary: str,
    voice_video: Path,
    music: Path | None,
    percent: int,
    output: Path,
    subtitles: Path | None = None,
) -> list[str]:
    """Voice at unity gain plus loudness-normalized music scaled by percent/100.

    The video stream is copied. amix halves each input, so volume=2 restores
    voice level without relying on newer ffmpeg options.
    """
    gain = music_gain(percent)
    use_music = music is not None and gain > 0
    command = [ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(voice_video)]
    next_input = 1
    if use_music:
        command += ["-i", str(music)]
        next_input += 1
    subtitle_input = None
    if subtitles is not None:
        command += ["-i", str(subtitles)]
        subtitle_input = next_input
    command += ["-map", "0:v:0"]
    if use_music:
        graph = (
            f"[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,volume={gain:.4f}[m];"
            "[0:a][m]amix=inputs=2:duration=first:dropout_transition=0,volume=2,alimiter=limit=0.95[a]"
        )
        command += ["-filter_complex", graph, "-map", "[a]"]
    else:
        command += ["-map", "0:a:0?"]
    if subtitle_input is not None:
        command += ["-map", f"{subtitle_input}:0"]
    command += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    if subtitle_input is not None:
        command += ["-c:s", "mov_text", "-metadata:s:s:0", "language=und",
                    "-metadata:s:s:0", f"title={SUBTITLE_TITLE}"]
    command += ["-movflags", "+faststart", str(output)]
    return command


def mix_final(
    voice_video: Path,
    music: Path | None,
    percent: int,
    output: Path,
    *,
    subtitles: Path | None = None,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    output = Path(output)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.{threading.get_ident()}.tmp{output.suffix}")
    command = build_mix_command(ffmpeg_binary, Path(voice_video), music, percent, temporary, subtitles)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        temporary.unlink(missing_ok=True)
        raise MixError(f"ffmpeg mix failed: {completed.stderr.strip()}")
    temporary.replace(output)
    return output
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_mixer.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add mixer.py tests/test_mixer.py
git commit -m "feat: add voice + music mixer with subtitle attach

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Music helpers (pure logic)

**Files:**
- Create: `music.py` (first half), `tests/test_music_helpers.py`

**Interfaces:**
- Produces: `MusicMatch(title, artist, offset_s, window_start_s=0.0, track_length_s=None)`; `FetchedTrack(path, url)`; `MusicResult(status, info, track=None)`; `MusicFetchError`; `pick_quiet_windows(words, duration_s, window_s=12.0, count=3, step_s=1.0) -> list[tuple[float, float]]` (quietest first); `music_song_start(match) -> float`; `build_align_command(ffmpeg_binary, track, song_start_s, duration_s, output) -> list[str]`; `choose_youtube_result(entries, *, artist, title, track_length_s=None) -> dict | None`; `parse_shazam_result(result: dict) -> MusicMatch | None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_music_helpers.py`:

```python
import subprocess

from helpers import make_tone, media_duration, mean_volume_db

from dir_language_pipeline.music import (
    MusicMatch,
    build_align_command,
    choose_youtube_result,
    music_song_start,
    parse_shazam_result,
    pick_quiet_windows,
)


def words(*spans):
    return [{"word": "w", "start": s, "end": e} for s, e in spans]


def test_quiet_windows_avoid_speech():
    spoken = words((0, 10))
    windows = pick_quiet_windows(spoken, duration_s=40, window_s=12, count=3)
    assert windows[0][0] >= 10
    for start, end in windows:
        assert end - start == 12


def test_windows_do_not_overlap_and_respect_count():
    windows = pick_quiet_windows([], duration_s=100, window_s=12, count=3)
    assert len(windows) == 3
    ordered = sorted(windows)
    for (_, previous_end), (next_start, _) in zip(ordered, ordered[1:]):
        assert next_start >= previous_end


def test_short_video_gets_one_window_covering_all_of_it():
    assert pick_quiet_windows(words((0, 2)), duration_s=5.0) == [(0.0, 5.0)]
    assert pick_quiet_windows([], duration_s=0) == []


def test_song_start_is_offset_minus_window_start():
    match = MusicMatch("t", "a", offset_s=40.0, window_start_s=10.0)
    assert music_song_start(match) == 30.0


def test_align_trims_from_song_start_and_limits_duration(tmp_path):
    track = make_tone(tmp_path / "track.mp3", seconds=6.0)
    out = tmp_path / "aligned.mp3"
    subprocess.run(build_align_command("ffmpeg", track, 1.0, 3.0, out), check=True, capture_output=True)
    assert abs(media_duration(out) - 3.0) < 0.2


def test_negative_song_start_delays_the_music(tmp_path):
    track = make_tone(tmp_path / "track.mp3", seconds=6.0)
    out = tmp_path / "aligned.mp3"
    subprocess.run(build_align_command("ffmpeg", track, -1.0, 3.0, out), check=True, capture_output=True)
    assert abs(media_duration(out) - 3.0) < 0.2
    assert mean_volume_db(out, start=0, duration=0.5) < -60


def entry(title, duration, channel="Some Channel", id_="abc"):
    return {"title": title, "duration": duration, "channel": channel, "id": id_}


def test_chooser_prefers_topic_channel_and_rejects_live_and_cover():
    entries = [
        entry("Artist - Song (Live at Arena)", 215, id_="live"),
        entry("Song cover by someone", 214, id_="cover"),
        entry("Artist - Song (Official Audio)", 215, channel="Artist - Topic", id_="topic"),
        entry("Artist - Song lyrics", 216, id_="lyrics"),
    ]
    best = choose_youtube_result(entries, artist="Artist", title="Song", track_length_s=215)
    assert best["id"] == "topic"


def test_chooser_rejects_wrong_duration_and_returns_none_when_nothing_fits():
    entries = [entry("Artist - Song", 600, id_="long"), entry("Artist - Song", None, id_="unknown")]
    assert choose_youtube_result(entries, artist="Artist", title="Song", track_length_s=215) is None


def test_chooser_without_known_length_still_filters_implausible_durations():
    entries = [entry("Artist - Song", 20, id_="short"), entry("Artist - Song", 200, id_="ok")]
    assert choose_youtube_result(entries, artist="Artist", title="Song")["id"] == "ok"


def test_chooser_allows_banned_word_when_the_song_title_has_it():
    entries = [entry("Artist - Live Forever (Official Audio)", 230, id_="ok")]
    best = choose_youtube_result(entries, artist="Artist", title="Live Forever")
    assert best["id"] == "ok"


def test_parse_shazam_result():
    result = {"matches": [{"offset": 42.5}], "track": {"title": "Song", "subtitle": "Artist"}}
    match = parse_shazam_result(result)
    assert (match.title, match.artist, match.offset_s) == ("Song", "Artist", 42.5)
    assert parse_shazam_result({"matches": []}) is None
    assert parse_shazam_result({"track": {"subtitle": "no title"}}) is None
    assert parse_shazam_result({"track": {"title": "T"}}).offset_s == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_music_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: ... music`.

- [ ] **Step 3: Implement the first half of `music.py`**

```python
from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol


class MusicFetchError(RuntimeError):
    """The identified track could not be downloaded or processed."""


@dataclass(frozen=True)
class MusicMatch:
    title: str
    artist: str
    offset_s: float
    window_start_s: float = 0.0
    track_length_s: float | None = None


@dataclass(frozen=True)
class FetchedTrack:
    path: Path
    url: str


@dataclass
class MusicResult:
    status: str  # "ok" | "not_identified" | "download_failed"
    info: dict
    track: Path | None = None


class MusicIdentifier(Protocol):
    def identify(self, clip: Path) -> MusicMatch | None: ...


class MusicFetcher(Protocol):
    def fetch(self, artist: str, title: str, dest_dir: Path, track_length_s: float | None = None) -> FetchedTrack: ...


def _speech_overlap(words: list[dict], start: float, end: float) -> float:
    total = 0.0
    for word in words:
        total += max(0.0, min(float(word["end"]), end) - max(float(word["start"]), start))
    return total


def pick_quiet_windows(
    words: list[dict],
    duration_s: float,
    window_s: float = 12.0,
    count: int = 3,
    step_s: float = 1.0,
) -> list[tuple[float, float]]:
    """Non-overlapping windows with the least speech, quietest first."""
    if duration_s <= 0:
        return []
    if duration_s <= window_s:
        return [(0.0, float(duration_s))]
    scored = []
    start = 0.0
    while start + window_s <= duration_s + 1e-9:
        scored.append((_speech_overlap(words, start, start + window_s), start))
        start += step_s
    scored.sort()
    chosen: list[tuple[float, float]] = []
    for _, begin in scored:
        if all(begin >= end or begin + window_s <= other for other, end in chosen):
            chosen.append((begin, begin + window_s))
        if len(chosen) == count:
            break
    return chosen


def music_song_start(match: MusicMatch) -> float:
    """Song position that lines up with video time 0 (negative: music starts later)."""
    return match.offset_s - match.window_start_s


def build_align_command(
    ffmpeg_binary: str, track: Path, song_start_s: float, duration_s: float, output: Path
) -> list[str]:
    fade = min(2.0, duration_s / 2)
    fade_filter = f"afade=t=out:st={max(0.0, duration_s - fade):.3f}:d={fade:.3f}"
    command = [ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if song_start_s >= 0:
        command += ["-ss", f"{song_start_s:.3f}", "-i", str(track)]
        audio_filter = fade_filter
    else:
        delay_ms = int(round(-song_start_s * 1000))
        command += ["-i", str(track)]
        audio_filter = f"adelay={delay_ms}:all=1,{fade_filter}"
    command += ["-t", f"{duration_s:.3f}", "-af", audio_filter, "-vn",
                "-c:a", "libmp3lame", "-q:a", "2", str(output)]
    return command


_BANNED = ("live", "cover", "remix", "sped up", "speed up", "slowed", "reverb",
           "nightcore", "karaoke", "instrumental", "8d", "reaction", "tutorial", "mashup")


def choose_youtube_result(
    entries: list[dict], *, artist: str, title: str, track_length_s: float | None = None
) -> dict | None:
    wanted = f"{artist} {title}".lower()
    best, best_score = None, None
    for entry in entries:
        duration = entry.get("duration")
        if not duration:
            continue
        name = (entry.get("title") or "").lower()
        channel = (entry.get("channel") or entry.get("uploader") or "").lower()
        if any(word in name and word not in wanted for word in _BANNED):
            continue
        if track_length_s:
            if abs(duration - track_length_s) > max(8.0, 0.08 * track_length_s):
                continue
            closeness = abs(duration - track_length_s) / 10.0
        else:
            if not 60 <= duration <= 600:
                continue
            closeness = 0.0
        score = -closeness
        if channel.endswith(" - topic"):
            score += 3
        if "official audio" in name:
            score += 2
        elif "official" in name:
            score += 1
        if artist.lower() in channel or artist.lower() in name:
            score += 1
        if best_score is None or score > best_score:
            best, best_score = entry, score
    return best


def parse_shazam_result(result: dict) -> MusicMatch | None:
    track = (result or {}).get("track") or {}
    title = track.get("title")
    if not title:
        return None
    matches = result.get("matches") or []
    offset = float(matches[0].get("offset", 0.0)) if matches else 0.0
    return MusicMatch(title=title, artist=track.get("subtitle", ""), offset_s=offset)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_music_helpers.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add music.py tests/test_music_helpers.py
git commit -m "feat: add music window picking, alignment and YouTube result chooser

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Music identification, fetch and orchestration

**Files:**
- Modify: `music.py` (append)
- Test: `tests/test_music_pipeline.py`

**Interfaces:**
- Consumes: everything produced in Task 2.
- Produces: `ShazamIdentifier().identify(clip) -> MusicMatch | None`; `YouTubeFetcher(ffmpeg_location=None, search_count=8, ydl_factory=None).fetch(artist, title, dest_dir, track_length_s=None) -> FetchedTrack`; `identify_and_fetch(source_video, words, duration_s, music_dir, identifier, fetcher, *, ffmpeg_binary="ffmpeg") -> MusicResult`. `music_dir` receives `music.json` (always) and `music_track.mp3` (status ok). `music.json` keys: `status`, plus `title`, `artist`, `offset_s`, `window_start_s`; `ok` adds `source_url`, `song_start_s`; `download_failed` adds `error`. Never raises for soft failures. Results are cached: a second call with the same `music_dir` returns the stored result.

- [ ] **Step 1: Write the failing tests**

`tests/test_music_pipeline.py`:

```python
import asyncio
import json

from helpers import make_tone, make_video, media_duration

from dir_language_pipeline.music import (
    FetchedTrack,
    MusicFetchError,
    MusicMatch,
    ShazamIdentifier,
    YouTubeFetcher,
    identify_and_fetch,
)


class FakeIdentifier:
    def __init__(self, match=None, error=None):
        self.match, self.error, self.calls = match, error, 0

    def identify(self, clip):
        self.calls += 1
        if self.error:
            raise self.error
        return self.match


class FakeFetcher:
    def __init__(self, tmp_path, error=None):
        self.tmp_path, self.error, self.calls = tmp_path, error, []

    def fetch(self, artist, title, dest_dir, track_length_s=None):
        self.calls.append((artist, title))
        if self.error:
            raise self.error
        path = make_tone(dest_dir / "yt_source.mp3", seconds=8.0, freq=700)
        return FetchedTrack(path=path, url="https://www.youtube.com/watch?v=abc")


WORDS = [{"word": "hi", "start": 0.0, "end": 1.0}]


def test_identified_track_is_fetched_aligned_and_recorded(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=5.0)
    identifier = FakeIdentifier(MusicMatch("Song", "Artist", offset_s=3.0))
    result = identify_and_fetch(source, WORDS, 5.0, tmp_path / "music", identifier, FakeFetcher(tmp_path))
    assert result.status == "ok"
    assert result.track.name == "music_track.mp3"
    assert abs(media_duration(result.track) - 5.0) < 0.3
    saved = json.loads((tmp_path / "music" / "music.json").read_text(encoding="utf-8"))
    assert saved["status"] == "ok"
    assert saved["title"] == "Song" and saved["artist"] == "Artist"
    assert saved["source_url"].startswith("https://www.youtube.com/")


def test_not_identified_is_a_soft_result(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=5.0)
    result = identify_and_fetch(source, WORDS, 5.0, tmp_path / "music", FakeIdentifier(None), FakeFetcher(tmp_path))
    assert result.status == "not_identified" and result.track is None
    assert json.loads((tmp_path / "music" / "music.json").read_text())["status"] == "not_identified"


def test_download_failure_is_a_soft_result_with_reason(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=5.0)
    result = identify_and_fetch(
        source, WORDS, 5.0, tmp_path / "music",
        FakeIdentifier(MusicMatch("Song", "Artist", 1.0)),
        FakeFetcher(tmp_path, error=MusicFetchError("no suitable result")),
    )
    assert result.status == "download_failed"
    assert "no suitable result" in result.info["error"]
    assert result.info["title"] == "Song"


def test_identifier_exception_counts_as_no_match(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=5.0)
    result = identify_and_fetch(
        source, WORDS, 5.0, tmp_path / "music", FakeIdentifier(error=RuntimeError("network down")), FakeFetcher(tmp_path)
    )
    assert result.status == "not_identified"


def test_source_without_audio_track_is_not_identified_not_a_crash(tmp_path):
    silent = make_video(tmp_path / "silent.mp4", seconds=5.0, with_audio=False)
    identifier = FakeIdentifier(MusicMatch("Song", "Artist", 1.0))
    result = identify_and_fetch(silent, [], 5.0, tmp_path / "music", identifier, FakeFetcher(tmp_path))
    assert result.status == "not_identified"
    assert identifier.calls == 0


def test_result_is_cached_for_resume(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=5.0)
    identifier = FakeIdentifier(MusicMatch("Song", "Artist", 1.0))
    fetcher = FakeFetcher(tmp_path)
    identify_and_fetch(source, WORDS, 5.0, tmp_path / "music", identifier, fetcher)
    again = identify_and_fetch(source, WORDS, 5.0, tmp_path / "music", identifier, fetcher)
    assert again.status == "ok" and again.track.is_file()
    assert identifier.calls == 1 and len(fetcher.calls) == 1


def test_shazam_identifier_wraps_the_async_client(tmp_path, monkeypatch):
    identifier = ShazamIdentifier()

    async def fake_recognize(self, clip):
        return {"matches": [{"offset": 12.0}], "track": {"title": "Song", "subtitle": "Artist"}}

    monkeypatch.setattr(ShazamIdentifier, "_recognize", fake_recognize)
    match = identifier.identify(tmp_path / "clip.wav")
    assert (match.title, match.artist, match.offset_s) == ("Song", "Artist", 12.0)


class FakeYDL:
    searches = []
    downloads = []
    entries = []

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, query, download=False):
        FakeYDL.searches.append(query)
        return {"entries": FakeYDL.entries}

    def download(self, urls):
        FakeYDL.downloads.append(urls)
        out = self.options["outtmpl"].replace("%(ext)s", "mp3")
        open(out, "wb").write(b"x" * 100)


def test_youtube_fetcher_searches_picks_best_and_downloads(tmp_path):
    FakeYDL.searches, FakeYDL.downloads = [], []
    FakeYDL.entries = [
        {"title": "Artist - Song (Official Audio)", "duration": 200, "channel": "Artist - Topic", "id": "good"},
        {"title": "Song cover", "duration": 200, "channel": "x", "id": "bad"},
    ]
    fetched = YouTubeFetcher(ydl_factory=FakeYDL).fetch("Artist", "Song", tmp_path, track_length_s=200)
    assert FakeYDL.searches == ["ytsearch8:Artist Song audio"]
    assert FakeYDL.downloads == [["https://www.youtube.com/watch?v=good"]]
    assert fetched.path.name == "yt_source.mp3" and fetched.url.endswith("good")


def test_youtube_fetcher_raises_when_nothing_suitable(tmp_path):
    import pytest

    FakeYDL.entries = [{"title": "Song cover", "duration": 200, "channel": "x", "id": "bad"}]
    with pytest.raises(MusicFetchError):
        YouTubeFetcher(ydl_factory=FakeYDL).fetch("Artist", "Song", tmp_path)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_music_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name 'ShazamIdentifier'`.

- [ ] **Step 3: Append the implementation to `music.py`**

```python
class ShazamIdentifier:
    """Fingerprint a short clip with Shazam (via `shazamio`).

    The library is unofficial; keep it behind MusicIdentifier so it can be swapped.
    """

    async def _recognize(self, clip: Path) -> dict:
        from shazamio import Shazam

        shazam = Shazam()
        recognize = getattr(shazam, "recognize", None) or shazam.recognize_song
        return await recognize(str(clip))

    def identify(self, clip: Path) -> MusicMatch | None:
        return parse_shazam_result(asyncio.run(self._recognize(Path(clip))))


class YouTubeFetcher:
    def __init__(
        self,
        ffmpeg_location: str | None = None,
        search_count: int = 8,
        ydl_factory: Callable[[dict], Any] | None = None,
    ):
        self.ffmpeg_location = ffmpeg_location
        self.search_count = search_count
        self.ydl_factory = ydl_factory

    def _ydl(self, options: dict):
        if self.ydl_factory:
            return self.ydl_factory(options)
        import yt_dlp

        return yt_dlp.YoutubeDL(options)

    def fetch(self, artist: str, title: str, dest_dir: Path, track_length_s: float | None = None) -> FetchedTrack:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        search = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "skip_download": True}
        with self._ydl(search) as ydl:
            info = ydl.extract_info(f"ytsearch{self.search_count}:{artist} {title} audio", download=False)
        entries = [entry for entry in (info or {}).get("entries", []) if entry]
        best = choose_youtube_result(entries, artist=artist, title=title, track_length_s=track_length_s)
        if best is None:
            raise MusicFetchError(f"No suitable YouTube result for '{artist} - {title}'.")
        url = best.get("webpage_url") or f"https://www.youtube.com/watch?v={best['id']}"
        target = dest_dir / "yt_source.mp3"
        target.unlink(missing_ok=True)
        download = {
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "format": "bestaudio/best",
            "outtmpl": str(dest_dir / "yt_source.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        }
        if self.ffmpeg_location:
            download["ffmpeg_location"] = self.ffmpeg_location
        with self._ydl(download) as ydl:
            ydl.download([url])
        if not target.is_file():
            raise MusicFetchError("yt-dlp did not produce an MP3 file.")
        return FetchedTrack(path=target, url=url)


def _run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise MusicFetchError(f"ffmpeg failed: {completed.stderr.strip()}")


def _cut_window(ffmpeg_binary: str, source: Path, start: float, end: float, output: Path) -> None:
    _run_ffmpeg([
        ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", str(output),
    ])
    if not output.is_file() or output.stat().st_size < 1000:
        raise MusicFetchError("No audio in this window.")


def _load_cached(music_dir: Path) -> MusicResult | None:
    path = music_dir / "music.json"
    if not path.is_file():
        return None
    info = json.loads(path.read_text(encoding="utf-8"))
    track = music_dir / "music_track.mp3"
    if info.get("status") == "ok":
        return MusicResult("ok", info, track) if track.is_file() else None
    return MusicResult(info.get("status", "not_identified"), info)


def _identify_and_fetch(source_video, words, duration_s, music_dir, identifier, fetcher, ffmpeg_binary) -> MusicResult:
    match = None
    for index, (start, end) in enumerate(pick_quiet_windows(words, duration_s)):
        clip = music_dir / f"window_{index}.wav"
        try:
            _cut_window(ffmpeg_binary, source_video, start, end, clip)
            found = identifier.identify(clip)
        except Exception as exc:
            print(f"[Music] window {start:.1f}-{end:.1f}s skipped: {exc}")
            continue
        if found:
            match = replace(found, window_start_s=start)
            break
    if match is None:
        print("[Music] No music identified — the video will be voice-only.")
        return MusicResult("not_identified", {"status": "not_identified"})
    print(f"[Music] Identified: {match.artist} - {match.title}")
    base = {"title": match.title, "artist": match.artist,
            "offset_s": match.offset_s, "window_start_s": match.window_start_s}
    try:
        fetched = fetcher.fetch(match.artist, match.title, music_dir, match.track_length_s)
        song_start = music_song_start(match)
        track = music_dir / "music_track.mp3"
        _run_ffmpeg(build_align_command(ffmpeg_binary, fetched.path, song_start, duration_s, track))
    except Exception as exc:
        print(f"[Music] Could not fetch the track: {exc}")
        return MusicResult("download_failed", {**base, "status": "download_failed", "error": str(exc)})
    info = {**base, "status": "ok", "source_url": fetched.url, "song_start_s": round(song_start, 3)}
    return MusicResult("ok", info, track)


def identify_and_fetch(
    source_video: Path,
    words: list[dict],
    duration_s: float,
    music_dir: Path,
    identifier: MusicIdentifier,
    fetcher: MusicFetcher,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> MusicResult:
    """Identify the ad's music, fetch it, and align it to the video. Never raises for soft failures."""
    music_dir = Path(music_dir)
    music_dir.mkdir(parents=True, exist_ok=True)
    cached = _load_cached(music_dir)
    if cached is not None:
        return cached
    result = _identify_and_fetch(Path(source_video), words, duration_s, music_dir, identifier, fetcher, ffmpeg_binary)
    (music_dir / "music.json").write_text(json.dumps(result.info, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_music_pipeline.py tests/test_music_helpers.py -v`
Expected: 21 passed. If `test_source_without_audio_track...` fails because ffmpeg exits 0 with an empty file, the `_cut_window` size check should still raise; adjust the 1000-byte threshold only if a valid short window is wrongly rejected.

- [ ] **Step 5: Commit**

```bash
git add music.py tests/test_music_pipeline.py
git commit -m "feat: identify background music with Shazam and fetch it from YouTube

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Faces core (clustering, ranking, lock)

**Files:**
- Create: `vendor/service_b/src/faces.py`, `tests/test_faces_core.py`

**Interfaces:**
- Produces: constants `SAME_PERSON=0.70`, `MIN_FACE_HEIGHT_RATIO=0.10`, `LOCK_MIN_PRESENCE=0.40`, `MAX_CANDIDATES=4`; `FaceObs(shot_index, time_s, box, frame_h, embedding, weight_s, crop=None)` with `box=(x, y, w, h)`; `FaceCandidate(id, embedding, shots, screen_seconds, score, crop_file=None, recommended=False, crop=None)` with `.to_json()` (with embedding) and `.public_view()` (without embedding, with `crop` filename); `cosine(a, b)`; `cluster_observations(observations, threshold=SAME_PERSON) -> list[list[FaceObs]]`; `rank_candidates(clusters, max_candidates=MAX_CANDIDATES, min_seconds=0.3) -> list[FaceCandidate]` (ids `"0".."n"`, best first, first is `recommended`); `FaceLock(embedding)` with `.enabled`, `.tag`, `.vector()`, `FaceLock.disabled()`, `FaceLock.from_candidate(dict)`; `select_lock(candidates: list[FaceCandidate], choice: str) -> FaceLock` (raises `PipelineError` for unknown id); `save_candidates(path, candidates)`, `load_candidates(path) -> list[FaceCandidate]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_faces_core.py`:

```python
import numpy as np
import pytest

from src.common import PipelineError
from src.faces import (
    FaceCandidate,
    FaceLock,
    FaceObs,
    cluster_observations,
    cosine,
    load_candidates,
    rank_candidates,
    save_candidates,
    select_lock,
)

rng = np.random.default_rng(7)


def identity(seed):
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


def noisy(vec, scale=0.05):
    return vec + rng.normal(scale=scale, size=vec.shape).astype(np.float32)


def obs(vec, shot=0, weight=1.0, height=60, frame_h=100):
    return FaceObs(shot_index=shot, time_s=0.0, box=(0, 0, 40, height), frame_h=frame_h,
                   embedding=noisy(vec), weight_s=weight)


def test_cosine():
    a = identity(1)
    assert cosine(a, a) == pytest.approx(1.0)
    assert abs(cosine(identity(1), identity(2))) < 0.4


def test_clustering_groups_same_person_and_separates_others():
    a, b = identity(1), identity(2)
    observations = [obs(a, 0), obs(b, 1), obs(a, 2), obs(b, 3), obs(a, 4)]
    clusters = cluster_observations(observations)
    assert sorted(len(c) for c in clusters) == [2, 3]


def test_ranking_orders_by_screen_time_times_size_and_marks_recommended():
    a, b = identity(1), identity(2)
    clusters = cluster_observations([obs(a, 0, weight=3.0), obs(a, 1, weight=3.0), obs(b, 2, weight=1.0)])
    ranked = rank_candidates(clusters)
    assert [c.id for c in ranked] == ["0", "1"]
    assert ranked[0].recommended and not ranked[1].recommended
    assert ranked[0].shots == [0, 1] and ranked[0].screen_seconds == 6.0


def test_ranking_drops_noise_and_caps_candidates():
    people = [identity(seed) for seed in range(10, 16)]
    clusters = cluster_observations([obs(p, i, weight=2.0 + i) for i, p in enumerate(people)]
                                    + [obs(identity(99), 9, weight=0.05)])
    ranked = rank_candidates(clusters, max_candidates=4)
    assert len(ranked) == 4
    assert all(c.screen_seconds >= 0.3 for c in ranked)


def test_no_observations_gives_no_candidates():
    assert rank_candidates(cluster_observations([])) == []


def test_lock_tag_and_enabled():
    disabled = FaceLock.disabled()
    assert not disabled.enabled and disabled.tag == "NOFACE"
    lock = FaceLock.from_candidate({"embedding": identity(1).tolist()})
    other = FaceLock.from_candidate({"embedding": identity(2).tolist()})
    assert lock.enabled and lock.tag.startswith("L") and lock.tag != other.tag
    assert lock.vector().shape == (128,)


def test_select_lock_and_unknown_choice(tmp_path):
    candidates = rank_candidates(cluster_observations([obs(identity(1), 0, weight=2.0)]))
    assert select_lock(candidates, "0").enabled
    with pytest.raises(PipelineError):
        select_lock(candidates, "7")


def test_candidates_round_trip_without_losing_embedding(tmp_path):
    candidates = rank_candidates(cluster_observations([obs(identity(1), 0, weight=2.0)]))
    path = tmp_path / "face_candidates.json"
    save_candidates(path, candidates)
    loaded = load_candidates(path)
    assert loaded[0].id == "0" and loaded[0].recommended
    assert cosine(loaded[0].embedding, candidates[0].embedding) == pytest.approx(1.0)
    view = loaded[0].public_view()
    assert "embedding" not in view and view["id"] == "0"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_faces_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.faces'`.

- [ ] **Step 3: Implement `vendor/service_b/src/faces.py` (core part)**

```python
"""
Face candidates and speaker lock.

Analyze phase: find the distinct faces that appear while someone is speaking and
rank them. Render phase: only shots containing the locked identity are lip-synced,
and Sync Labs receives per-frame bounding boxes of that identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import PipelineError

SAME_PERSON = 0.70          # SFace cosine similarity: same identity at or above this
MIN_FACE_HEIGHT_RATIO = 0.10  # ignore faces smaller than this share of frame height
LOCK_MIN_PRESENCE = 0.40    # share of a shot's frames the locked face must appear in
MAX_CANDIDATES = 4
MIN_CANDIDATE_SECONDS = 0.3


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def cosine(a, b) -> float:
    return float(np.dot(_unit(a), _unit(b)))


@dataclass
class FaceObs:
    shot_index: int
    time_s: float
    box: tuple  # (x, y, w, h)
    frame_h: int
    embedding: Any
    weight_s: float
    crop: Any = None


@dataclass
class FaceCandidate:
    id: str
    embedding: list
    shots: list
    screen_seconds: float
    score: float
    crop_file: str | None = None
    recommended: bool = False
    crop: Any = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict:
        return {"id": self.id, "embedding": list(self.embedding), "shots": self.shots,
                "screen_seconds": self.screen_seconds, "score": self.score,
                "crop_file": self.crop_file, "recommended": self.recommended}

    def public_view(self) -> dict:
        return {"id": self.id, "shots": len(self.shots), "seconds": self.screen_seconds,
                "recommended": self.recommended, "crop": self.crop_file}


def cluster_observations(observations: list, threshold: float = SAME_PERSON) -> list:
    clusters = []
    for observation in observations:
        vector = _unit(observation.embedding)
        best, best_similarity = None, threshold
        for cluster in clusters:
            similarity = float(np.dot(vector, _unit(cluster["sum"])))
            if similarity >= best_similarity:
                best, best_similarity = cluster, similarity
        if best is None:
            clusters.append({"sum": vector.copy(), "members": [observation]})
        else:
            best["sum"] = best["sum"] + vector
            best["members"].append(observation)
    return [cluster["members"] for cluster in clusters]


def rank_candidates(clusters: list, max_candidates: int = MAX_CANDIDATES,
                    min_seconds: float = MIN_CANDIDATE_SECONDS) -> list:
    scored = []
    for members in clusters:
        seconds = sum(m.weight_s for m in members)
        if seconds < min_seconds:
            continue
        score = sum(m.weight_s * (m.box[3] / m.frame_h) for m in members)
        embedding = _unit(np.mean([_unit(m.embedding) for m in members], axis=0))
        sharpest = max(members, key=lambda m: m.box[3])
        scored.append((score, embedding, members, seconds, sharpest.crop))
    scored.sort(key=lambda item: -item[0])
    candidates = []
    for rank, (score, embedding, members, seconds, crop) in enumerate(scored[:max_candidates]):
        candidates.append(FaceCandidate(
            id=str(rank), embedding=embedding.tolist(),
            shots=sorted({m.shot_index for m in members}),
            screen_seconds=round(seconds, 2), score=round(score, 4),
            recommended=(rank == 0), crop=crop,
        ))
    return candidates


@dataclass(frozen=True)
class FaceLock:
    embedding: tuple | None

    @classmethod
    def disabled(cls) -> "FaceLock":
        return cls(None)

    @classmethod
    def from_candidate(cls, candidate: dict) -> "FaceLock":
        return cls(tuple(float(v) for v in candidate["embedding"]))

    @property
    def enabled(self) -> bool:
        return self.embedding is not None

    @property
    def tag(self) -> str:
        if self.embedding is None:
            return "NOFACE"
        digest = hashlib.sha1(json.dumps([round(v, 4) for v in self.embedding]).encode()).hexdigest()
        return "L" + digest[:8]

    def vector(self) -> np.ndarray:
        return _unit(self.embedding)


def select_lock(candidates: list, choice: str) -> FaceLock:
    for candidate in candidates:
        if candidate.id == str(choice):
            return FaceLock.from_candidate(candidate.to_json())
    raise PipelineError(
        f"Face choice '{choice}' is not one of the detected candidates; submit the video again."
    )


def save_candidates(path, candidates: list) -> None:
    Path(path).write_text(
        json.dumps([c.to_json() for c in candidates], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_candidates(path) -> list:
    return [
        FaceCandidate(id=item["id"], embedding=item["embedding"], shots=item["shots"],
                      screen_seconds=item["screen_seconds"], score=item["score"],
                      crop_file=item.get("crop_file"), recommended=item.get("recommended", False))
        for item in json.loads(Path(path).read_text(encoding="utf-8"))
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_faces_core.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add vendor/service_b/src/faces.py tests/test_faces_core.py
git commit -m "feat: add face clustering, ranking and speaker lock

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Face discovery and per-frame bounding boxes

**Files:**
- Modify: `vendor/service_b/src/faces.py` (append)
- Test: `tests/test_faces_video.py`

**Interfaces:**
- Consumes: Task 4's `FaceObs`, `cluster_observations`, `rank_candidates`, `save_candidates`, `load_candidates`, `FaceLock`.
- Produces: `FaceEngine(tracker=None).detect(frame_bgr) -> list[((x, y, w, h), embedding)]`; `discover_candidates(video_path, scenes, words, job_dir, engine, *, samples_per_shot=6, max_candidates=MAX_CANDIDATES) -> list[FaceCandidate]` (writes `job_dir/faces/face_cand_<id>.jpg` and `job_dir/face_candidates.json`); `load_or_discover(video_path, scenes, words, job_dir, engine=None) -> list[FaceCandidate]` (reuses `face_candidates.json`; creates a `FaceEngine` only when discovering); `locked_boxes_for_clip(clip_path, engine, lock, threshold=SAME_PERSON) -> tuple[list[list[int] | None], float]` (boxes are `[x1, y1, x2, y2]` per frame, plus presence ratio); `build_sync_options(boxes) -> dict`.
- `engine` is any object with `detect(frame_bgr)` returning `[((x, y, w, h), embedding), ...]`, so tests use fakes.

- [ ] **Step 1: Write the failing tests**

`tests/test_faces_video.py`:

```python
import numpy as np
from helpers import make_video

from src.faces import (
    FaceLock,
    build_sync_options,
    discover_candidates,
    load_or_discover,
    locked_boxes_for_clip,
)


def identity(seed):
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


A, B = identity(1), identity(2)


class ScriptedEngine:
    """Always sees A (big); sees B (small) on every third call."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        faces = [((20, 10, 40, 60), A)]
        if self.calls % 3 == 0:
            faces.append(((90, 20, 30, 40), B))
        return faces


class FixedEngine:
    def __init__(self, vector):
        self.vector = vector

    def detect(self, frame):
        return [((10, 10, 30, 40), self.vector)]


class EmptyEngine:
    def detect(self, frame):
        return []


def scenes_and_words():
    return [{"start": 0.0, "end": 3.0}], [{"word": "hi", "start": 0.2, "end": 2.5}]


def test_discovery_finds_ranks_and_saves_candidates(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3.0, size="160x90")
    scenes, words = scenes_and_words()
    candidates = discover_candidates(video, scenes, words, tmp_path / "job", ScriptedEngine())
    assert len(candidates) == 2
    assert candidates[0].recommended and candidates[0].screen_seconds > candidates[1].screen_seconds
    assert (tmp_path / "job" / "faces" / "face_cand_0.jpg").is_file()
    assert (tmp_path / "job" / "face_candidates.json").is_file()


def test_shots_without_speech_are_ignored(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3.0)
    candidates = discover_candidates(video, [{"start": 0.0, "end": 3.0}], [], tmp_path / "job", ScriptedEngine())
    assert candidates == []


def test_video_with_no_faces_gives_no_candidates(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3.0)
    scenes, words = scenes_and_words()
    assert discover_candidates(video, scenes, words, tmp_path / "job", EmptyEngine()) == []


def test_load_or_discover_reuses_saved_candidates(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3.0)
    scenes, words = scenes_and_words()
    first = load_or_discover(video, scenes, words, tmp_path / "job", engine=ScriptedEngine())
    again = load_or_discover(video, scenes, words, tmp_path / "job", engine=EmptyEngine())
    assert [c.id for c in again] == [c.id for c in first] and len(again) == 2


def test_boxes_follow_the_locked_identity(tmp_path):
    clip = make_video(tmp_path / "clip.mp4", seconds=2.0)
    boxes, presence = locked_boxes_for_clip(clip, FixedEngine(A), FaceLock.from_candidate({"embedding": A.tolist()}))
    assert presence == 1.0 and len(boxes) >= 18
    assert boxes[0] == [10, 10, 40, 50]


def test_other_person_is_not_boxed(tmp_path):
    clip = make_video(tmp_path / "clip.mp4", seconds=2.0)
    boxes, presence = locked_boxes_for_clip(clip, FixedEngine(B), FaceLock.from_candidate({"embedding": A.tolist()}))
    assert presence == 0.0 and all(box is None for box in boxes)


def test_sync_options_payload_shape():
    options = build_sync_options([[1, 2, 3, 4], None])
    assert options == {"auto_detect": False, "bounding_boxes": [[1, 2, 3, 4], None]}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_faces_video.py -v`
Expected: FAIL with `ImportError: cannot import name 'discover_candidates'`.

- [ ] **Step 3: Append the implementation to `faces.py`**

```python
class FaceEngine:
    """YuNet detection + SFace embeddings, using the models shipped with Service B."""

    def __init__(self, tracker=None):
        if tracker is None:
            from .face_id_tracker import FaceIDTracker

            tracker = FaceIDTracker()
        self._detector = tracker.detector
        self._recognizer = tracker.recognizer

    def detect(self, frame_bgr) -> list:
        height, width = frame_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame_bgr)
        found = []
        for face in faces if faces is not None else []:
            aligned = self._recognizer.alignCrop(frame_bgr, face)
            embedding = self._recognizer.feature(aligned).reshape(-1)
            x, y, w, h = [int(v) for v in face[:4]]
            found.append(((x, y, w, h), embedding))
        return found


def _speech_overlaps(words: list, start: float, end: float) -> bool:
    return any(float(w["end"]) > start and float(w["start"]) < end for w in words)


def _crop(frame, box, margin: float = 0.25, max_side: int = 256):
    height, width = frame.shape[:2]
    x, y, w, h = box
    pad_x, pad_y = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    scale = max_side / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale)
    return crop.copy()


def discover_candidates(video_path, scenes: list, words: list, job_dir, engine, *,
                        samples_per_shot: int = 6, max_candidates: int = MAX_CANDIDATES) -> list:
    job_dir = Path(job_dir)
    faces_dir = job_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PipelineError(f"Cannot open video for face detection: {video_path}")
    observations = []
    try:
        for index, scene in enumerate(scenes):
            start, end = float(scene["start"]), float(scene["end"])
            if not _speech_overlaps(words, start, end):
                continue
            duration = end - start
            count = max(1, min(samples_per_shot, int(duration * 2)))
            weight = duration / count
            for k in range(count):
                moment = start + duration * (k + 0.5) / count
                capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame_h = frame.shape[0]
                for box, embedding in engine.detect(frame):
                    if box[3] < MIN_FACE_HEIGHT_RATIO * frame_h:
                        continue
                    observations.append(FaceObs(index, moment, box, frame_h, embedding, weight, _crop(frame, box)))
    finally:
        capture.release()
    candidates = rank_candidates(cluster_observations(observations), max_candidates)
    for candidate in candidates:
        if candidate.crop is not None:
            name = f"face_cand_{candidate.id}.jpg"
            cv2.imwrite(str(faces_dir / name), candidate.crop)
            candidate.crop_file = name
    save_candidates(job_dir / "face_candidates.json", candidates)
    print(f"[Faces] {len(candidates)} face candidate(s) found.")
    return candidates


def load_or_discover(video_path, scenes: list, words: list, job_dir, engine=None) -> list:
    saved = Path(job_dir) / "face_candidates.json"
    if saved.is_file():
        return load_candidates(saved)
    return discover_candidates(video_path, scenes, words, job_dir, engine or FaceEngine())


def locked_boxes_for_clip(clip_path, engine, lock: FaceLock, threshold: float = SAME_PERSON):
    """Per-frame [x1, y1, x2, y2] of the locked identity (None where absent) and the presence ratio."""
    anchor = lock.vector()
    capture = cv2.VideoCapture(str(clip_path))
    boxes = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            best, best_similarity = None, threshold
            for box, embedding in engine.detect(frame):
                similarity = cosine(anchor, embedding)
                if similarity >= best_similarity:
                    best, best_similarity = box, similarity
            if best is None:
                boxes.append(None)
            else:
                x, y, w, h = best
                boxes.append([max(0, x), max(0, y), max(0, x + w), max(0, y + h)])
    finally:
        capture.release()
    present = sum(box is not None for box in boxes)
    return boxes, (present / len(boxes) if boxes else 0.0)


def build_sync_options(boxes: list) -> dict:
    """Sync Labs active_speaker_detection payload: only the locked face is animated."""
    return {"auto_detect": False, "bounding_boxes": boxes}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_faces_video.py tests/test_faces_core.py -v`
Expected: 15 passed. If `test_discovery_finds_ranks...` finds only one candidate because B's short-lived detections fall under `MIN_CANDIDATE_SECONDS`, increase the clip length in the test to 6 seconds rather than lowering the production threshold.

- [ ] **Step 5: Commit**

```bash
git add vendor/service_b/src/faces.py tests/test_faces_video.py
git commit -m "feat: discover face candidates and build per-frame locked-face boxes

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Lock-gated lip-sync

**Files:**
- Modify: `vendor/service_b/src/lipsyncer.py`
- Test: `tests/test_lipsync_gating.py`

**Interfaces:**
- Consumes: Task 4/5 `FaceLock`, `LOCK_MIN_PRESENCE`, `build_sync_options`, `locked_boxes_for_clip`, `FaceEngine`.
- Produces: `process_shots(video_path, dubbed_audio, classified_scenes, job_dir, sync_model="lipsync-2-pro", dry_run=False, lock=None, face_engine=None) -> list[Path]`. With `lock=None` behavior is unchanged (auto-detect). A disabled lock mux-only for every A-roll shot. An enabled lock lip-syncs a shot only when the locked face is present in at least 40% of its frames, sending `bounding_boxes`. Each A-roll scene dict gets `lip_sync` set to one of `synced`, `cached`, `skipped_not_target`, `skipped_no_face_lock`, `skipped_short`, `sync_failed`, `dry_run`, and `locked_face_ratio` when a lock exists. Synced and cache file names carry the lock tag so a different face never reuses another face's clip.

- [ ] **Step 1: Write the failing tests**

`tests/test_lipsync_gating.py`:

```python
import shutil

import numpy as np
import pytest
from helpers import make_tone, make_video

from src import lipsyncer
from src.faces import FaceLock


def identity(seed):
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


A, B = identity(1), identity(2)


class FakeSync:
    instances = []

    def __init__(self, default_model=None):
        self.uploads, self.generate_calls = [], []
        FakeSync.instances.append(self)

    def upload_file(self, path, content_type=None):
        self.uploads.append(path)
        return f"https://sync.test/{path.name}"

    def generate(self, video_url, audio_url, model=None, options=None):
        self.generate_calls.append(options)
        return {"id": "gen1"}

    def wait_for_completion(self, generation_id, poll_interval=8):
        return {"outputUrl": "https://sync.test/out"}

    def download_result(self, url, output_path):
        shutil.copyfile(self.uploads[0], output_path)


class FaceEngineStub:
    def __init__(self, vector=None):
        self.vector = vector

    def detect(self, frame):
        return [] if self.vector is None else [((10, 10, 30, 40), self.vector)]


@pytest.fixture
def shot(tmp_path, monkeypatch):
    FakeSync.instances = []
    monkeypatch.setattr(lipsyncer, "SyncLabsClient", FakeSync)
    video = make_video(tmp_path / "src.mp4", seconds=2.0, fps=10)
    audio = make_tone(tmp_path / "dub.wav", seconds=2.0)
    scenes = [{"start": 0.0, "end": 2.0, "class": "A_ROLL"}]
    return video, audio, scenes, tmp_path / "job"


def run(shot, **kwargs):
    video, audio, scenes, job = shot
    clips = lipsyncer.process_shots(video, audio, scenes, job, **kwargs)
    return clips, scenes


def test_without_a_lock_behavior_is_unchanged_auto_detect(shot):
    clips, scenes = run(shot)
    options = FakeSync.instances[0].generate_calls[0]
    assert options["active_speaker_detection"] == {"auto_detect": True}
    assert scenes[0]["lip_sync"] == "synced"


def test_locked_face_present_syncs_with_bounding_boxes(shot):
    lock = FaceLock.from_candidate({"embedding": A.tolist()})
    clips, scenes = run(shot, lock=lock, face_engine=FaceEngineStub(A))
    detection = FakeSync.instances[0].generate_calls[0]["active_speaker_detection"]
    assert detection["auto_detect"] is False
    assert len(detection["bounding_boxes"]) >= 18
    assert detection["bounding_boxes"][0] == [10, 10, 40, 50]
    assert scenes[0]["lip_sync"] == "synced" and scenes[0]["locked_face_ratio"] == 1.0
    assert lock.tag in clips[0].name


def test_other_person_in_shot_is_not_lip_synced(shot):
    lock = FaceLock.from_candidate({"embedding": A.tolist()})
    clips, scenes = run(shot, lock=lock, face_engine=FaceEngineStub(B))
    assert FakeSync.instances[0].generate_calls == []
    assert scenes[0]["lip_sync"] == "skipped_not_target"
    assert clips[0].is_file()


def test_disabled_lock_means_audio_only_no_sync_call(shot):
    clips, scenes = run(shot, lock=FaceLock.disabled())
    assert FakeSync.instances[0].generate_calls == []
    assert scenes[0]["lip_sync"] == "skipped_no_face_lock"
    assert clips[0].is_file()


def test_cached_clip_is_not_reused_for_a_different_face(shot):
    lock_a = FaceLock.from_candidate({"embedding": A.tolist()})
    lock_b = FaceLock.from_candidate({"embedding": B.tolist()})
    first, _ = run(shot, lock=lock_a, face_engine=FaceEngineStub(A))
    second, scenes = run(shot, lock=lock_b, face_engine=FaceEngineStub(B))
    assert first[0] != second[0]
    assert scenes[0]["lip_sync"] == "synced"
    again, scenes = run(shot, lock=lock_a, face_engine=FaceEngineStub(A))
    assert again[0] == first[0] and scenes[0]["lip_sync"] == "cached"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_lipsync_gating.py -v`
Expected: FAIL with `TypeError: process_shots() got an unexpected keyword argument 'lock'`.

- [ ] **Step 3: Edit `lipsyncer.py` imports and helper**

In `vendor/service_b/src/lipsyncer.py`, after the line `from .audio import ffmpeg` add:

```python
from .faces import (
    FaceEngine,
    FaceLock,
    LOCK_MIN_PRESENCE,
    build_sync_options,
    locked_boxes_for_clip,
)
```

After the `_copy_source_clip` function, add:

```python
def _audio_only_clip(raw_clip: Path, audio_slice: Path, out_path: Path, duration_s: float) -> Path:
    """Dubbed audio over the untouched video (no mouth edit)."""
    if not out_path.exists():
        _mux_audio_onto_video(raw_clip, audio_slice, out_path, duration_s)
    return out_path
```

- [ ] **Step 4: Edit the `process_shots` signature, docstring and setup**

Replace the signature block

```python
def process_shots(
    video_path: Path,
    dubbed_audio: Path,
    classified_scenes: list[dict],
    job_dir: Path,
    sync_model: str = "lipsync-2-pro",
    dry_run: bool = False,
) -> list[Path]:
```

with

```python
def process_shots(
    video_path: Path,
    dubbed_audio: Path,
    classified_scenes: list[dict],
    job_dir: Path,
    sync_model: str = "lipsync-2-pro",
    dry_run: bool = False,
    lock: "FaceLock | None" = None,
    face_engine=None,
) -> list[Path]:
```

Append to the docstring (before the closing `"""`):

```text

    lock=None keeps the legacy behavior (Sync Labs auto-detects the speaker).
    A disabled lock (no face to lock) dubs audio-only. An enabled lock lip-syncs a
    shot only when the locked identity is present in >= LOCK_MIN_PRESENCE of its
    frames, sending that identity's per-frame bounding boxes to Sync Labs.
```

Directly after the line `output_clips: list[Path] = []` add:

```python
    lock_tag = f"_{lock.tag}" if lock is not None else ""
    if lock is not None and lock.enabled and face_engine is None and not dry_run:
        face_engine = FaceEngine()
```

- [ ] **Step 5: Replace the A-roll branch**

Replace everything from the line `        else:  # A_ROLL` down to (not including) the line `    return output_clips` with:

```python
        else:  # A_ROLL
            # Content-hash cache — reuse if already synced for this exact face lock
            clip_hash = _file_hash(raw_clip) + _file_hash(audio_slice)
            synced_out = seg_dir / f"{prefix}_{time_key}_{clip_hash}{lock_tag}_synced.mp4"

            synced_found = None
            if synced_out.exists():
                synced_found = synced_out
            else:
                candidates = sorted(seg_dir.glob(f"{prefix}_{time_key}_*{lock_tag}_synced.mp4"))
                if candidates:
                    synced_found = candidates[0]

            if synced_found:
                print(f"  [LipSync] A-Roll → cached synced clip reused: {synced_found.name} (dur={duration_s:.2f}s)")
                scene["lip_sync"] = "cached"
                output_clips.append(synced_found)
                continue

            if dry_run:
                # In dry-run: mux audio without lip-sync (for testing)
                dry_out = seg_dir / f"{prefix}_{time_key}_aroll_dryrun.mp4"
                _audio_only_clip(raw_clip, audio_slice, dry_out, duration_s)
                print(f"  [LipSync] A-Roll → DRY RUN, muxed without sync")
                scene["lip_sync"] = "dry_run"
                output_clips.append(dry_out)
                continue

            # --- Guard: check actual clip duration ---
            try:
                actual_dur = get_video_info(raw_clip)["duration"]
            except Exception:
                actual_dur = duration_s

            MIN_SYNC_DURATION = 1.0  # Sync Labs rejects clips shorter than 1.0s
            if actual_dur < MIN_SYNC_DURATION:
                print(f"  [LipSync] A-Roll → clip too short ({actual_dur:.2f}s < {MIN_SYNC_DURATION}s) "
                      f"— falling back to B-Roll mux")
                fallback_out = seg_dir / f"{prefix}_{time_key}_short_fallback.mp4"
                _audio_only_clip(raw_clip, audio_slice, fallback_out, actual_dur)
                scene["lip_sync"] = "skipped_short"
                output_clips.append(fallback_out)
                continue

            # --- Face lock gate ---
            options = {
                "sync_mode": "cut_off",
                "active_speaker_detection": {"auto_detect": True},
            }
            if lock is not None:
                if not lock.enabled:
                    print(f"  [LipSync] A-Roll → no face locked for this video; dubbing audio only")
                    out = seg_dir / f"{prefix}_{time_key}_noface.mp4"
                    _audio_only_clip(raw_clip, audio_slice, out, actual_dur)
                    scene["lip_sync"] = "skipped_no_face_lock"
                    output_clips.append(out)
                    continue
                boxes, presence = locked_boxes_for_clip(raw_clip, face_engine, lock)
                scene["locked_face_ratio"] = round(presence, 3)
                if presence < LOCK_MIN_PRESENCE or not any(box is not None for box in boxes):
                    print(f"  [LipSync] A-Roll → locked face present in {presence:.0%} of frames "
                          f"(< {LOCK_MIN_PRESENCE:.0%}); dubbing audio only")
                    out = seg_dir / f"{prefix}_{time_key}_notarget.mp4"
                    _audio_only_clip(raw_clip, audio_slice, out, actual_dur)
                    scene["lip_sync"] = "skipped_not_target"
                    output_clips.append(out)
                    continue
                options = {
                    "sync_mode": "cut_off",
                    "active_speaker_detection": build_sync_options(boxes),
                }

            # --- Submit to Sync Labs ---
            print(f"  [LipSync] A-Roll → uploading to Sync Labs {sync_model} "
                  f"({actual_dur:.2f}s, {raw_clip.stat().st_size // 1024}KB)...")
            try:
                video_url = sync_client.upload_file(raw_clip, content_type="video/mp4")
                audio_url = sync_client.upload_file(audio_slice, content_type="audio/wav")

                gen_data = sync_client.generate(video_url, audio_url,
                                                model=sync_model, options=options)
                gen_id = gen_data["id"]
                print(f"  [LipSync] Sync Labs Job: {gen_id} — waiting...")
                result = sync_client.wait_for_completion(gen_id, poll_interval=8)
                sync_client.download_result(result["outputUrl"], synced_out)
                print(f"  [LipSync] Downloaded → {synced_out.name}")
                scene["lip_sync"] = "synced"
                output_clips.append(synced_out)

            except PipelineError as e:
                # Graceful fallback: if Sync Labs rejects (422, 429, etc.), mux audio only
                print(f"  [LipSync] ⚠ Sync Labs failed for shot {i}: {e}")
                print(f"  [LipSync] Falling back to B-Roll mux (original video + dubbed audio)")
                fallback_out = seg_dir / f"{prefix}_{time_key}_syncfail_fallback.mp4"
                _audio_only_clip(raw_clip, audio_slice, fallback_out, actual_dur)
                scene["lip_sync"] = "sync_failed"
                output_clips.append(fallback_out)

```

(Keep the blank line before `    return output_clips`.)

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_lipsync_gating.py -v`
Expected: 5 passed. Note: the fake Sync client records the raw clip as the first upload; if `download_result` raises `IndexError`, confirm `uploads[0]` is the video upload (video is uploaded before audio).

- [ ] **Step 7: Run the whole suite and commit**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all previous tests still pass.

```bash
git add vendor/service_b/src/lipsyncer.py tests/test_lipsync_gating.py
git commit -m "feat: gate lip-sync on a locked face and send Sync Labs bounding boxes

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Voice-only dub, analyze phase, engine lock

**Files:**
- Modify: `vendor/service_b/src/el_dubber.py`, `vendor/service_b/src/engine.py`
- Create: `vendor/service_b/src/analysis.py`
- Test: `tests/test_service_b_changes.py`

**Interfaces:**
- Produces: `ElevenLabsDubber.create_job` now sends `drop_background_audio="true"`. `analysis.analyze_video(source_video, job_dir, source_lang="auto", scene_threshold=32.0, supplied_transcript=None) -> {"words": list[dict], "scenes": list[dict], "duration": float}` (reuses the existing transcript and `scenes.json` caches). `engine.run_dubbing_engine(..., face_lock=None)` passes the lock to `process_shots` and rewrites `scenes.json` afterwards so per-shot `lip_sync` decisions are persisted.

- [ ] **Step 1: Write the failing tests**

`tests/test_service_b_changes.py`:

```python
import inspect
import json

from helpers import make_video

from src import analysis, el_dubber, engine


class FakeResponse:
    ok = True
    headers = {}

    def json(self):
        return {"dubbing_id": "d1"}


def test_dubbing_request_drops_background_audio(tmp_path, monkeypatch):
    sent = {}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        sent["data"] = data
        return FakeResponse()

    monkeypatch.setattr(el_dubber.requests, "post", fake_post)
    video = make_video(tmp_path / "v.mp4", seconds=1.0)
    el_dubber.ElevenLabsDubber("key").create_job(video, "hi")
    assert sent["data"]["drop_background_audio"] == "true"
    assert sent["data"]["watermark"] == "false"


def test_analysis_returns_words_scenes_and_reuses_scene_cache(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4", seconds=2.0)
    job = tmp_path / "job"
    calls = {"classify": 0}
    words = [{"word": "hi", "start": 0.1, "end": 0.5}]

    def fake_transcript(video_path, job_dir, source_lang, supplied_srt):
        return job / "t.srt", words

    def fake_classify(video_path, job_dir, words, scene_threshold):
        calls["classify"] += 1
        scenes = [{"start": 0.0, "end": 2.0, "class": "A_ROLL"}]
        (job_dir / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")
        return scenes

    monkeypatch.setattr(analysis, "extract_transcript", fake_transcript)
    monkeypatch.setattr(analysis, "classify_video", fake_classify)
    first = analysis.analyze_video(video, job)
    second = analysis.analyze_video(video, job)
    assert first["words"] == words and first["scenes"][0]["class"] == "A_ROLL"
    assert 1.8 < first["duration"] < 2.3
    assert second["scenes"] == first["scenes"]
    assert calls["classify"] == 1


def test_engine_accepts_a_face_lock():
    assert "face_lock" in inspect.signature(engine.run_dubbing_engine).parameters
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_service_b_changes.py -v`
Expected: FAIL (`drop_background_audio` KeyError; `src.analysis` missing).

- [ ] **Step 3: Edit `el_dubber.py`**

In `ElevenLabsDubber.create_job`, change the `data` dict from

```python
            "num_speakers": str(int(num_speakers)),
            "watermark": "false",
        }
```

to

```python
            "num_speakers": str(int(num_speakers)),
            "watermark": "false",
            # Voice only: the original music is restored later from the identified
            # track, so the dub must not carry its own background audio.
            "drop_background_audio": "true",
        }
```

- [ ] **Step 4: Create `vendor/service_b/src/analysis.py`**

```python
"""
Analyze phase — everything needed to ask the user which face to lock,
with no paid dubbing call yet: transcript, shot cuts and scene classes.

Both steps reuse the engine's on-disk caches, so the render phase that follows
(after the user confirms a face) does not repeat the work.
"""
import json
from pathlib import Path

from .classifier import classify_video
from .scene_analyzer import get_video_info
from .transcriber import extract_transcript


def analyze_video(source_video, job_dir, source_lang="auto", scene_threshold=32.0, supplied_transcript=None):
    source_video = Path(source_video).resolve()
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_info(source_video)["duration"]

    _, words = extract_transcript(
        video_path=source_video,
        job_dir=job_dir,
        source_lang=source_lang,
        supplied_srt=supplied_transcript,
    )

    scenes_path = job_dir / "scenes.json"
    if scenes_path.exists():
        scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
    else:
        scenes = classify_video(
            video_path=source_video,
            job_dir=job_dir,
            words=words,
            scene_threshold=scene_threshold,
        )
    return {"words": words, "scenes": scenes, "duration": duration}
```

- [ ] **Step 5: Edit `engine.py`**

In the `run_dubbing_engine` signature, add `face_lock=None,` after `skip_stages: list[int] = None,`. In the docstring Args list add:

```text
        face_lock           : FaceLock choosing which face to lip-sync (None = legacy auto-detect;
                              a disabled lock dubs audio-only)
```

Replace the Stage 4 call

```python
        output_clips = process_shots(
            video_path=source_video,
            dubbed_audio=dubbed_audio,
            classified_scenes=classified_scenes,
            job_dir=job_dir,
            sync_model=sync_model,
            dry_run=dry_run,
        )
```

with

```python
        output_clips = process_shots(
            video_path=source_video,
            dubbed_audio=dubbed_audio,
            classified_scenes=classified_scenes,
            job_dir=job_dir,
            sync_model=sync_model,
            dry_run=dry_run,
            lock=face_lock,
        )
        # Persist per-shot lip-sync decisions (lip_sync, locked_face_ratio) for review.
        (job_dir / "scenes.json").write_text(
            json.dumps(classified_scenes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_service_b_changes.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add vendor/service_b/src/el_dubber.py vendor/service_b/src/engine.py vendor/service_b/src/analysis.py tests/test_service_b_changes.py
git commit -m "feat: request voice-only dubs, add analyze phase, pass face lock to engine

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Pipeline wiring (pause, music artifacts, mixer in the assembler)

**Files:**
- Modify: `contracts.py`, `__init__.py`, `runner.py`, `media.py`, `service_adapters.py`, `in_process.py`
- Test: `tests/test_pipeline_wiring.py`, `tests/test_adapter_pause.py`

**Interfaces:**
- Consumes: `mixer.mix_final`, `mixer.DEFAULT_MUSIC_PERCENT` (Task 1); `music.identify_and_fetch`, `ShazamIdentifier`, `YouTubeFetcher`, `MusicResult` (Task 3); `src.analysis.analyze_video` (Task 7); `src.faces.load_or_discover`, `select_lock`, `FaceLock` (Tasks 4-5); `engine.run_dubbing_engine(..., face_lock=)` (Task 7).
- Produces:
  - `contracts.AwaitingFaceSelection(RuntimeError)`; `PipelineResult` gains `music_track: Artifact | None = None`, `music_info: Artifact | None = None`.
  - `FinalVideoAssembler.assemble(video, audio, visual_subtitles, *, job_id, music=None, music_percent=DEFAULT_MUSIC_PERCENT)`.
  - `TextToAudioAdapter(backend, service_root=SERVICE_B_ROOT, face_choice=None, music_identifier=None, music_fetcher=None)`. `run(...)` returns `video`, `audio`?, `audio_subtitles`?, plus `music_info` (always) and `music_track` (status ok). When candidates exist and `face_choice is None` it publishes `faces.json`, `face_cand_<id>.jpg` and `music.json` into the backend, then raises `AwaitingFaceSelection(job_id)`. When no candidates exist it uses `FaceLock.disabled()` and does not pause. `faces.json` is `{"candidates": [{"id", "shots", "seconds", "recommended", "crop"}]}`.
  - `in_process.run_artifact(..., face_choice=None)` and `run_operation(..., face_choice=None)`; both operations' artifact dicts gain `voice_video`, `music_track`, `music_info` where available. In dubbing-only mode `final_video`/`dubbed_video` is the voice + music mix at 25%.

- [ ] **Step 1: Write the failing runner/assembler tests**

`tests/test_pipeline_wiring.py`:

```python
import pytest
from helpers import make_tone, make_video, mean_volume_db, stream_types

from dir_language_pipeline.contracts import Artifact, AwaitingFaceSelection, PipelineStageError
from dir_language_pipeline.media import FfmpegFinalAssembler, FileArtifactBackend
from dir_language_pipeline.runner import DirectoryLanguagePipeline, PipelineServices


def artifact(backend, path, name, content_type):
    return backend.upload(path, filename=name, content_type=content_type)


class FakeVisual:
    def __init__(self, backend, tmp_path):
        self.backend, self.tmp_path = backend, tmp_path

    def run(self, video, **kwargs):
        srt = self.tmp_path / "v.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        return {"video": video, "visual_subtitles": artifact(self.backend, srt, "visual-subtitles.srt", "application/x-subrip")}


class PausingAudio:
    def run(self, video, **kwargs):
        raise AwaitingFaceSelection("job1")


class MusicAudio:
    def __init__(self, backend, tmp_path):
        self.backend, self.tmp_path = backend, tmp_path

    def run(self, video, **kwargs):
        voice = make_video(self.tmp_path / "voice.mp4", seconds=2.0)
        music = make_tone(self.tmp_path / "m.mp3", seconds=2.0)
        return {
            "video": artifact(self.backend, voice, "service-b-video.mp4", "video/mp4"),
            "music_track": artifact(self.backend, music, "music_track.mp3", "audio/mpeg"),
        }


class RecordingAssembler:
    def __init__(self, inner):
        self.inner, self.kwargs = inner, None

    def assemble(self, video, audio, visual_subtitles, **kwargs):
        self.kwargs = kwargs
        return self.inner.assemble(video, audio, visual_subtitles, **kwargs)


def services(backend, audio, assembler):
    visual = FakeVisual(backend, backend.root.parent)
    return PipelineServices(visual_subtitles=visual, audio_dubbing=audio, assembler=assembler, publisher=backend)


def test_pause_is_not_wrapped_in_a_stage_error(tmp_path):
    backend = FileArtifactBackend(tmp_path / "art")
    source = artifact(backend, make_video(tmp_path / "in.mp4"), "in.mp4", "video/mp4")
    pipeline = DirectoryLanguagePipeline(services(backend, PausingAudio(), FfmpegFinalAssembler(backend)))
    with pytest.raises(AwaitingFaceSelection):
        pipeline.run(source, target_language="hi", job_id="job1")


def test_music_flows_to_the_assembler_and_into_the_result(tmp_path):
    backend = FileArtifactBackend(tmp_path / "art")
    source = artifact(backend, make_video(tmp_path / "in.mp4"), "in.mp4", "video/mp4")
    assembler = RecordingAssembler(FfmpegFinalAssembler(backend))
    pipeline = DirectoryLanguagePipeline(services(backend, MusicAudio(backend, tmp_path), assembler))
    result = pipeline.run(source, target_language="hi", job_id="job1")
    assert assembler.kwargs["music"].filename.startswith("music_track")
    assert assembler.kwargs["music_percent"] == 25
    assert result.music_track is not None
    final = result.final_video.metadata["path"]
    assert "subtitle" in stream_types(final)


def test_assembler_mixes_music_at_the_requested_percent(tmp_path):
    backend = FileArtifactBackend(tmp_path / "art")
    voice = artifact(backend, make_video(tmp_path / "v.mp4", seconds=2.0), "v.mp4", "video/mp4")
    music = artifact(backend, make_tone(tmp_path / "m.mp3", seconds=2.0), "m.mp3", "audio/mpeg")
    srt_path = tmp_path / "s.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    srt = artifact(backend, srt_path, "s.srt", "application/x-subrip")
    assembler = FfmpegFinalAssembler(backend)
    plain = assembler.assemble(voice, None, srt, job_id="j")
    mixed = assembler.assemble(voice, None, srt, job_id="j", music=music, music_percent=80)
    assert mean_volume_db(mixed.metadata["path"]) > mean_volume_db(plain.metadata["path"]) + 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_pipeline_wiring.py -v`
Expected: FAIL with `ImportError: cannot import name 'AwaitingFaceSelection'`.

- [ ] **Step 3: Edit `contracts.py`**

Add after `PipelineStageError`:

```python
class AwaitingFaceSelection(RuntimeError):
    """Analysis finished; the user must confirm which face to lip-sync before rendering."""
```

In `PipelineResult`, add two defaulted fields after `manifest`:

```python
    manifest: Mapping[str, Any]
    music_track: Artifact | None = None
    music_info: Artifact | None = None
```

In `__init__.py` change the contracts import to
`from .contracts import Artifact, AwaitingFaceSelection, PipelineResult, PipelineStageError` and add `"AwaitingFaceSelection",` to `__all__`.

- [ ] **Step 4: Edit `runner.py`**

Add imports: `from .contracts import Artifact, AwaitingFaceSelection, PipelineResult, PipelineStageError` and `from .mixer import DEFAULT_MUSIC_PERCENT`.

Replace the `FinalVideoAssembler` protocol with:

```python
class FinalVideoAssembler(Protocol):
    def assemble(
        self,
        video: Artifact,
        audio: Artifact | None,
        visual_subtitles: Artifact,
        *,
        job_id: str,
        music: Artifact | None = None,
        music_percent: int = DEFAULT_MUSIC_PERCENT,
    ) -> Artifact: ...
```

In the Service B block, change the `except` to re-raise the pause, and read the music artifacts:

```python
            dubbed_video = audio.get("video")
            dubbed_audio = audio.get("audio")
            audio_srt = audio.get("audio_subtitles")
            music_track = audio.get("music_track")
            music_info = audio.get("music_info")
            if on_stage:
                on_stage("service_b_completed")
        except Exception as exc:
            if isinstance(exc, (PipelineStageError, AwaitingFaceSelection)):
                raise
            raise PipelineStageError("service_b", "audio dubbing failed", cause=exc) from exc
```

Change the assemble call to:

```python
            final_video = self.services.assembler.assemble(
                assembly_video,
                None if dubbed_video else dubbed_audio,
                visual_srt,
                job_id=job_id,
                music=music_track,
                music_percent=DEFAULT_MUSIC_PERCENT,
            )
```

Add `"music_track": music_track.uri if music_track else None,` to `manifest`, and pass `music_track=music_track, music_info=music_info,` into the `PipelineResult(...)` constructor.

- [ ] **Step 5: Edit `media.py`**

Add `from .mixer import DEFAULT_MUSIC_PERCENT, mix_final` to the imports of `media.py`. Then replace the whole `assemble` method of `FfmpegFinalAssembler` (from `    def assemble(` through its final `return self.backend.upload(output, filename="final_video.mp4", content_type="video/mp4")`) with the version below. When the dubbed video already carries its audio (`audio is None`, the normal path) the mixer builds the final file; the legacy separate-audio path is kept verbatim.

```python
    def assemble(
        self,
        video: Artifact,
        audio: Artifact | None,
        visual_subtitles: Artifact,
        *,
        job_id: str,
        music: Artifact | None = None,
        music_percent: int = DEFAULT_MUSIC_PERCENT,
    ) -> Artifact:
        with tempfile.TemporaryDirectory(prefix=f"dir-language-{job_id}-") as temp:
            root = Path(temp)
            video_path = self.backend.download(video, root / "service-a-video.mp4")
            subtitle_path = self.backend.download(visual_subtitles, root / "visual.srt")

            if audio is None:
                music_path = self.backend.download(music, root / "music.mp3") if music else None
                mixed = mix_final(
                    video_path, music_path, music_percent, root / "final_video.mp4",
                    subtitles=subtitle_path, ffmpeg_binary=self.ffmpeg_binary,
                )
                return self.backend.upload(mixed, filename="final_video.mp4", content_type="video/mp4")

            # Legacy path: Service B returned audio separately from the video.
            audio_path = self.backend.download(audio, root / audio.filename)
            output = root / "final_video.mp4"
            command = [self.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                       "-i", str(video_path), "-i", str(audio_path), "-i", str(subtitle_path),
                       "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
                       "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text",
                       "-metadata:s:s:0", "language=und",
                       "-metadata:s:s:0", "title=Visual subtitles (Service A)",
                       "-movflags", "+faststart", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"Final video mux failed: {completed.stderr.strip()}")
            return self.backend.upload(output, filename="final_video.mp4", content_type="video/mp4")
```

- [ ] **Step 6: Run the runner tests**

Run: `.venv/bin/python -m pytest tests/test_pipeline_wiring.py -v`
Expected: 3 passed.

- [ ] **Step 7: Write the failing adapter tests**

`tests/test_adapter_pause.py`:

```python
import json

import numpy as np
import pytest
from helpers import make_tone, make_video

from dir_language_pipeline import service_adapters
from dir_language_pipeline.contracts import AwaitingFaceSelection
from dir_language_pipeline.media import FileArtifactBackend
from dir_language_pipeline.music import MusicResult
from dir_language_pipeline.service_adapters import TextToAudioAdapter
from src import analysis, engine, faces
from src.common import PipelineError
from src.faces import FaceCandidate


def candidate(identity_seed, cid="0"):
    vec = np.random.default_rng(identity_seed).normal(size=128).astype(np.float32)
    return FaceCandidate(id=cid, embedding=vec.tolist(), shots=[0, 1], screen_seconds=4.0,
                         score=2.0, crop_file=f"face_cand_{cid}.jpg", recommended=(cid == "0"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    backend = FileArtifactBackend(tmp_path / "artifacts")
    source = backend.upload(make_video(tmp_path / "in.mp4", seconds=3.0), filename="in.mp4", content_type="video/mp4")
    job_dir = tmp_path / "data" / "cache" / "service-b" / "job1" / "job"
    (job_dir / "faces").mkdir(parents=True)
    for crop in ("face_cand_0.jpg", "face_cand_1.jpg"):
        (job_dir / "faces" / crop).write_bytes(b"\xff\xd8jpeg")
    music_track = make_tone(tmp_path / "music_track.mp3", seconds=3.0)

    monkeypatch.setattr(analysis, "analyze_video", lambda *a, **k: {"words": [], "scenes": [], "duration": 3.0})
    monkeypatch.setattr(
        service_adapters, "identify_and_fetch",
        lambda *a, **k: MusicResult("ok", {"status": "ok", "title": "Song", "artist": "Artist"}, music_track),
    )
    seen = {}

    def fake_engine(**kwargs):
        seen["lock"] = kwargs.get("face_lock")
        out = make_video(tmp_path / "dubbed.mp4", seconds=3.0)
        return out

    monkeypatch.setattr(engine, "run_dubbing_engine", fake_engine)
    return backend, source, seen, monkeypatch


def run(adapter, source):
    return adapter.run(source, target_language="hi", source_language="auto", job_id="job1")


def test_candidates_without_a_choice_pause_and_publish_the_panel_data(env):
    backend, source, seen, monkeypatch = env
    monkeypatch.setattr(faces, "load_or_discover", lambda *a, **k: [candidate(1, "0"), candidate(2, "1")])
    with pytest.raises(AwaitingFaceSelection):
        run(TextToAudioAdapter(backend), source)
    published = json.loads((backend.root / "faces.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in published["candidates"]] == ["0", "1"]
    assert "embedding" not in published["candidates"][0]
    assert (backend.root / "face_cand_0.jpg").is_file()
    assert json.loads((backend.root / "music.json").read_text())["status"] == "ok"
    assert "lock" not in seen  # nothing was rendered


def test_a_choice_renders_with_that_face_locked_and_returns_music(env):
    backend, source, seen, monkeypatch = env
    monkeypatch.setattr(faces, "load_or_discover", lambda *a, **k: [candidate(1, "0"), candidate(2, "1")])
    result = run(TextToAudioAdapter(backend, face_choice="1"), source)
    assert seen["lock"].enabled
    assert seen["lock"].tag == faces.select_lock([candidate(1, "0"), candidate(2, "1")], "1").tag
    assert result["music_track"].filename == "music_track.mp3"
    assert "video" in result and "music_info" in result


def test_no_face_found_skips_the_pause_and_disables_lip_sync(env):
    backend, source, seen, monkeypatch = env
    monkeypatch.setattr(faces, "load_or_discover", lambda *a, **k: [])
    result = run(TextToAudioAdapter(backend), source)
    assert seen["lock"].enabled is False
    assert "video" in result


def test_stale_choice_fails_with_a_clear_message(env):
    backend, source, seen, monkeypatch = env
    monkeypatch.setattr(faces, "load_or_discover", lambda *a, **k: [candidate(1, "0")])
    with pytest.raises(PipelineError, match="not one of the detected candidates"):
        run(TextToAudioAdapter(backend, face_choice="9"), source)


def test_music_failure_does_not_block_rendering(env):
    backend, source, seen, monkeypatch = env
    monkeypatch.setattr(faces, "load_or_discover", lambda *a, **k: [])
    monkeypatch.setattr(
        service_adapters, "identify_and_fetch",
        lambda *a, **k: MusicResult("not_identified", {"status": "not_identified"}),
    )
    result = run(TextToAudioAdapter(backend), source)
    assert "music_track" not in result
    assert result["music_info"].filename == "music.json"
```

- [ ] **Step 8: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_adapter_pause.py -v`
Expected: FAIL (`TextToAudioAdapter() got an unexpected keyword argument 'face_choice'` / missing `identify_and_fetch` attribute).

- [ ] **Step 9: Edit `service_adapters.py`**

Add to the imports:

```python
from .contracts import Artifact, AwaitingFaceSelection
from .music import ShazamIdentifier, YouTubeFetcher, identify_and_fetch
```

(`Artifact` is already imported from `.contracts`; merge into one import line.)

Replace the whole `TextToAudioAdapter` class with:

```python
class TextToAudioAdapter:
    """Runs Service B on Service A's rendered video.

    Analyze first (transcript, shots, face candidates, music). If faces were found
    and the user has not chosen one yet, publish the choices and raise
    AwaitingFaceSelection; the job is re-queued with `face_choice` once they confirm,
    and this adapter then resumes from its on-disk caches and renders.
    """

    def __init__(
        self,
        backend: ArtifactBackend,
        service_root: Path = SERVICE_B_ROOT,
        face_choice: str | None = None,
        music_identifier: Any = None,
        music_fetcher: Any = None,
    ):
        self.backend = backend
        self.service_root = Path(service_root)
        self.face_choice = face_choice
        self.music_identifier = music_identifier
        self.music_fetcher = music_fetcher

    def _publish(self, path: Path, name: str, content_type: str, job_id: str) -> Artifact:
        artifact = Artifact(uri=f"artifact://{name}", filename=name, content_type=content_type,
                            metadata={"path": str(path)})
        return self.backend.publish(artifact, name=name, job_id=job_id)

    def _publish_music(self, music, music_dir: Path, job_id: str) -> dict[str, Artifact]:
        music_dir.mkdir(parents=True, exist_ok=True)
        info_path = music_dir / "music.json"
        info_path.write_text(json.dumps(music.info, ensure_ascii=False, indent=2), encoding="utf-8")
        published = {"music_info": self._publish(info_path, "music.json", "application/json", job_id)}
        if music.track is not None:
            published["music_track"] = self._publish(music.track, "music_track.mp3", "audio/mpeg", job_id)
        return published

    def _publish_face_choices(self, candidates, job_dir: Path, job_id: str) -> None:
        for candidate in candidates:
            if candidate.crop_file:
                self._publish(job_dir / "faces" / candidate.crop_file, candidate.crop_file, "image/jpeg", job_id)
        panel = job_dir / "faces.json"
        panel.write_text(
            json.dumps({"candidates": [c.public_view() for c in candidates]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._publish(panel, "faces.json", "application/json", job_id)

    def run(self, video: Artifact, *, target_language: str, source_language: str = "auto", job_id: str) -> dict[str, Artifact]:
        _add_import_root(self.service_root)
        cached = sys.modules.get("src")
        if cached is not None and not Path(cached.__file__).resolve().is_relative_to(self.service_root.resolve()):
            raise RuntimeError("A different top-level 'src' package is already loaded; isolate this application process.")
        engine = importlib.import_module("src.engine")
        analysis = importlib.import_module("src.analysis")
        faces = importlib.import_module("src.faces")
        audio_tools = importlib.import_module("src.audio")
        data_dir = Path(os.getenv("PIPELINE_DATA_DIR", "/data"))
        root = data_dir / "cache" / "service-b" / job_id
        root.mkdir(parents=True, exist_ok=True)
        source = self.backend.download(video, root / "service-a-video.mp4")
        job_dir = root / "job"
        ffmpeg_binary = audio_tools.ffmpeg()

        # --- Analyze (cached on disk, so a resume after confirmation is cheap) ---
        info = analysis.analyze_video(source, job_dir, source_language)
        music_dir = root / "music"
        music = identify_and_fetch(
            source, info["words"], info["duration"], music_dir,
            self.music_identifier or ShazamIdentifier(),
            self.music_fetcher or YouTubeFetcher(ffmpeg_location=ffmpeg_binary),
            ffmpeg_binary=ffmpeg_binary,
        )
        print(f"[Music] status={music.status}")
        extras = self._publish_music(music, music_dir, job_id)

        candidates = faces.load_or_discover(source, info["scenes"], info["words"], job_dir)
        if not candidates:
            print("[Faces] No usable face found — dubbing audio-only, no lip-sync.")
            lock = faces.FaceLock.disabled()
        elif self.face_choice is None:
            self._publish_face_choices(candidates, job_dir, job_id)
            print(f"[Faces] {len(candidates)} candidate(s) found — waiting for the user to confirm one.")
            raise AwaitingFaceSelection(job_id)
        else:
            lock = faces.select_lock(candidates, self.face_choice)
            print(f"[Faces] Locked face candidate {self.face_choice}.")

        # --- Render ---
        output = engine.run_dubbing_engine(
            source_video_path=source,
            target_lang=target_language,
            source_lang=source_language,
            job_dir=job_dir,
            dry_run=os.getenv("DUBBING_DRY_RUN", "false").lower() == "true",
            face_lock=lock,
        )
        output = Path(output)
        result: dict[str, Artifact] = {
            "video": self.backend.upload(output, filename="service-b-video.mp4", content_type="video/mp4")
        }
        audio_path = output.parent / f"dubbed_master_{target_language}.mp3"
        if audio_path.is_file():
            result["audio"] = self.backend.upload(
                audio_path, filename="service-b-audio.mp3", content_type="audio/mpeg"
            )
        audio_srt = output.parent / f"transcript_{target_language}.srt"
        if audio_srt.is_file():
            result["audio_subtitles"] = self.backend.upload(
                audio_srt, filename="audio-subtitles.srt", content_type="application/x-subrip"
            )
        result.update(extras)
        return result
```

Add `import json` to the module imports.

- [ ] **Step 10: Edit `in_process.py`**

Add imports: `import tempfile`, `from pathlib import Path`, `from .mixer import DEFAULT_MUSIC_PERCENT, mix_final`.

Add a helper above `run_artifact`:

```python
def _mix_dubbed_only(backend, voice: Artifact, music: Artifact, job_id: str) -> Artifact:
    """Dubbing-only has no subtitle track; mix the voice video with the music."""
    with tempfile.TemporaryDirectory(prefix=f"dub-mix-{job_id}-") as temp:
        root = Path(temp)
        voice_path = backend.download(voice, root / "voice.mp4")
        music_path = backend.download(music, root / "music.mp3")
        mixed = mix_final(voice_path, music_path, DEFAULT_MUSIC_PERCENT, root / "dubbed.mp4")
        artifact = Artifact(uri="artifact://dubbed-mixed.mp4", filename="dubbed-mixed.mp4",
                            content_type="video/mp4", metadata={"path": str(mixed)})
        return backend.publish(artifact, name="dubbed_video.mp4", job_id=job_id)
```

Add `face_choice: str | None = None,` to the signatures of both `run_artifact` and `run_operation`; in `run_artifact` build `audio_dubbing=TextToAudioAdapter(backend, face_choice=face_choice)`; in `run_operation` pass `face_choice=face_choice` to `run_artifact` and to `TextToAudioAdapter(backend, face_choice=face_choice)`.

In the `combine` branch, after the existing optional artifacts, add:

```python
        if result.service_b_video:
            artifacts["voice_video"] = backend.publish(
                result.service_b_video, name="voice_video.mp4", job_id=job_id
            )
        if result.music_track:
            artifacts["music_track"] = result.music_track
        if result.music_info:
            artifacts["music_info"] = result.music_info
```

Replace the body of the `dubbing` branch after `output = TextToAudioAdapter(...).run(...)` with:

```python
        voice = output["video"]
        music = output.get("music_track")
        if music is not None:
            video = _mix_dubbed_only(backend, voice, music, job_id)
        else:
            video = backend.publish(voice, name="dubbed_video.mp4", job_id=job_id)
        if on_stage:
            on_stage("dubbing_completed")
        artifacts = {
            "final_video": video,
            "dubbed_video": video,
            "voice_video": backend.publish(voice, name="voice_video.mp4", job_id=job_id),
        }
        if output.get("music_track"):
            artifacts["music_track"] = output["music_track"]
        if output.get("music_info"):
            artifacts["music_info"] = output["music_info"]
        if output.get("audio_subtitles"):
            artifacts["audio_subtitles"] = backend.publish(
                output["audio_subtitles"], name="audio_subtitles.srt", job_id=job_id
            )
        if output.get("audio"):
            artifacts["service_b_audio"] = backend.publish(
                output["audio"], name="dubbed_audio.mp3", job_id=job_id
            )
        return artifacts
```

(The existing `video = backend.publish(output["video"], name="dubbed_video.mp4", ...)` and its `on_stage("dubbing_completed")`/`artifacts = {...}` lines are replaced by the block above.)

- [ ] **Step 11: Run adapter tests and the whole suite**

Run: `.venv/bin/python -m pytest tests/test_adapter_pause.py tests/test_pipeline_wiring.py -v` then `.venv/bin/python -m pytest tests -q`
Expected: all pass. If `test_a_choice_renders...` errors on `kwargs.get("face_lock")`, confirm the adapter passes `face_lock=lock` by keyword.

- [ ] **Step 12: Commit**

```bash
git add contracts.py __init__.py runner.py media.py service_adapters.py in_process.py tests/test_pipeline_wiring.py tests/test_adapter_pause.py
git commit -m "feat: pause for face confirmation, thread music through the pipeline

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Job store pause, confirm and expiry

**Files:**
- Modify: `job_store.py`
- Test: `tests/test_job_store.py`

**Interfaces:**
- Produces: `AWAITING = "awaiting_face_selection"`; `JobStore.await_face(job_id)`; `JobStore.confirm_face(job_id, candidate_id) -> str` returning `"queued"` (was waiting, now re-queued), `"already_confirmed"` (same choice already recorded and job is queued/running/completed), `"not_awaiting"` (anything else), or `"unknown_job"`; `JobStore.expire_stale(max_age_seconds=86400) -> list[str]` (fails waiting jobs older than the limit with stage `expired`). `get()` and `claim()` rows include `face_choice`. A new `JobStore` on the same file leaves waiting jobs waiting (only `running` jobs are re-queued on restart).

- [ ] **Step 1: Write the failing tests**

`tests/test_job_store.py`:

```python
import time

from dir_language_pipeline.job_store import AWAITING, JobStore


def make_store(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create("j1", "hi", "auto", tmp_path / "in.mp4", "in.mp4", "combine")
    return store


def test_await_then_confirm_requeues_with_the_choice(tmp_path):
    store = make_store(tmp_path)
    claimed = store.claim()
    assert claimed["face_choice"] is None
    store.await_face("j1")
    assert store.get("j1")["status"] == AWAITING
    assert store.claim() is None  # a waiting job is not picked up by the worker
    assert store.confirm_face("j1", "1") == "queued"
    resumed = store.claim()
    assert resumed["job_id"] == "j1" and resumed["face_choice"] == "1"


def test_confirming_twice_is_idempotent(tmp_path):
    store = make_store(tmp_path)
    store.claim()
    store.await_face("j1")
    assert store.confirm_face("j1", "0") == "queued"
    assert store.confirm_face("j1", "0") == "already_confirmed"
    assert store.get("j1")["status"] == "queued"


def test_confirm_with_a_different_choice_after_resume_is_rejected(tmp_path):
    store = make_store(tmp_path)
    store.claim()
    store.await_face("j1")
    store.confirm_face("j1", "0")
    assert store.confirm_face("j1", "1") == "not_awaiting"
    assert store.get("j1")["face_choice"] == "0"


def test_confirm_on_a_job_that_is_not_waiting_or_unknown(tmp_path):
    store = make_store(tmp_path)
    assert store.confirm_face("j1", "0") == "not_awaiting"  # still queued
    assert store.confirm_face("nope", "0") == "unknown_job"


def test_waiting_job_survives_a_restart_but_running_job_is_requeued(tmp_path):
    store = make_store(tmp_path)
    store.create("j2", "hi", "auto", tmp_path / "in2.mp4", "in2.mp4", "combine")
    store.claim()
    store.await_face("j1")
    store.claim()  # j2 running
    reopened = JobStore(tmp_path / "jobs.sqlite3")
    assert reopened.get("j1")["status"] == AWAITING
    assert reopened.get("j2")["status"] == "queued"


def test_expiry_fails_only_stale_waiting_jobs(tmp_path):
    store = make_store(tmp_path)
    store.claim()
    store.await_face("j1")
    assert store.expire_stale(max_age_seconds=3600) == []
    time.sleep(0.05)
    assert store.expire_stale(max_age_seconds=0) == ["j1"]
    record = store.get("j1")
    assert record["status"] == "failed" and record["stage"] == "expired"
    assert "24 hours" in record["error"]


def test_old_databases_are_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE jobs (job_id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
            target_language TEXT NOT NULL, source_language TEXT NOT NULL, operation TEXT NOT NULL DEFAULT 'combine',
            input_path TEXT NOT NULL, input_name TEXT NOT NULL, artifacts TEXT NOT NULL DEFAULT '{}',
            error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    store = JobStore(path)
    store.create("j1", "hi", "auto", tmp_path / "i.mp4", "i.mp4")
    assert store.get("j1")["face_choice"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_job_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'AWAITING'`.

- [ ] **Step 3: Edit `job_store.py`**

Add at module level after the imports: `AWAITING = "awaiting_face_selection"`.

In the `CREATE TABLE` statement, add `face_choice TEXT,` and `awaiting_since TEXT,` before `error TEXT,`. After the existing `operation` migration, add:

```python
            if "face_choice" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN face_choice TEXT")
            if "awaiting_since" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN awaiting_since TEXT")
```

Add these methods to `JobStore` (before `get`):

```python
    def await_face(self, job_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,stage=?,awaiting_since=?,updated_at=? WHERE job_id=?",
                (AWAITING, AWAITING, self._now(), self._now(), job_id),
            )
        self._write_manifest(job_id)

    def confirm_face(self, job_id: str, candidate_id: str) -> str:
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status, face_choice FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                db.commit()
                return "unknown_job"
            if row["status"] == AWAITING:
                db.execute(
                    "UPDATE jobs SET status='queued',stage='queued',face_choice=?,awaiting_since=NULL,updated_at=? WHERE job_id=?",
                    (candidate_id, self._now(), job_id),
                )
                db.commit()
                outcome = "queued"
            elif row["face_choice"] == candidate_id and row["status"] in ("queued", "running", "completed"):
                db.commit()
                return "already_confirmed"
            else:
                db.commit()
                return "not_awaiting"
        self._write_manifest(job_id)
        return outcome

    def expire_stale(self, max_age_seconds: float = 24 * 3600) -> list[str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT job_id FROM jobs WHERE status=? AND awaiting_since <= ?", (AWAITING, cutoff)
            ).fetchall()
            expired = [row["job_id"] for row in rows]
            for job_id in expired:
                db.execute(
                    "UPDATE jobs SET status='failed',stage='expired',error=?,updated_at=? WHERE job_id=?",
                    ("Face confirmation timed out after 24 hours; submit the video again.", self._now(), job_id),
                )
        for job_id in expired:
            self._write_manifest(job_id)
        return expired
```

Change the datetime import to `from datetime import datetime, timedelta, timezone`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_job_store.py -v`
Expected: 7 passed. (The 0.05 s sleep in the expiry test guarantees `awaiting_since <= cutoff` at `max_age_seconds=0`.)

- [ ] **Step 5: Commit**

```bash
git add job_store.py tests/test_job_store.py
git commit -m "feat: add awaiting_face_selection state, confirmation and expiry to job store

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: API endpoints and worker pause handling

**Files:**
- Modify: `api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `JobStore` (Task 9), `mixer.mix_final` (Task 1), `contracts.AwaitingFaceSelection`, the artifact names published in Task 8 (`faces.json`, `face_cand_<id>.jpg`, `music.json`, `voice_video.mp4`, `music_track.mp3`, `visual_subtitles.srt`).
- Produces:
  - `GET /api/jobs/{id}/faces` → `{"status", "candidates": [...], "music": {...}|null}`; 404 if the job has no `faces.json` yet.
  - `GET /api/jobs/{id}/faces/{filename}` serves a crop (`face_cand_<digits>.jpg` only).
  - `POST /api/jobs/{id}/face` body `{"candidate_id": "0"}` → `{"status": ...}`; 404 unknown job, 409 if no choices are published or the job is not waiting, 422 if the id is not a candidate; calling it again with the same id returns 200.
  - `POST /api/jobs/{id}/remix` body `{"music_percent": 0-100}` → `{"music_percent", "final_video"}`; 404 unknown job, 409 if not completed or no identified music, 422 for out-of-range values. It overwrites the file registered as `final_video` and records `percent` in `music.json`.
  - `GET /api/jobs/{id}` additionally returns `"music"` (the parsed `music.json` or `null`).
  - Worker: catches `AwaitingFaceSelection` → `store.await_face`; passes `face_choice` into `run_operation`; calls `store.expire_stale()` on each idle tick.

- [ ] **Step 1: Write the failing tests**

`tests/test_api.py`:

```python
import importlib
import json

import pytest
from fastapi.testclient import TestClient
from helpers import make_tone, make_video, mean_volume_db


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    from dir_language_pipeline import api as api_module

    module = importlib.reload(api_module)
    return module, TestClient(module.app)  # no lifespan: the worker thread is not started


def new_job(module, tmp_path, job_id="j1", status=None):
    module.store.create(job_id, "hi", "auto", tmp_path / "in.mp4", "in.mp4", "combine")
    art = module.JOB_ROOT / job_id / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    if status == "awaiting":
        module.store.claim()
        module.store.await_face(job_id)
    return art


def publish_faces(art):
    (art / "faces.json").write_text(json.dumps({"candidates": [
        {"id": "0", "shots": 3, "seconds": 5.0, "recommended": True, "crop": "face_cand_0.jpg"},
        {"id": "1", "shots": 1, "seconds": 1.0, "recommended": False, "crop": "face_cand_1.jpg"},
    ]}), encoding="utf-8")
    (art / "face_cand_0.jpg").write_bytes(b"\xff\xd8jpeg0")
    (art / "music.json").write_text(json.dumps({"status": "ok", "title": "Song", "artist": "Artist"}), encoding="utf-8")


def test_faces_endpoint_404_until_published_then_lists_candidates_and_music(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, status="awaiting")
    assert client.get("/api/jobs/j1/faces").status_code == 404
    publish_faces(art)
    body = client.get("/api/jobs/j1/faces").json()
    assert body["status"] == "awaiting_face_selection"
    assert [c["id"] for c in body["candidates"]] == ["0", "1"]
    assert body["music"]["title"] == "Song"


def test_crop_endpoint_serves_crops_and_rejects_path_tricks(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, status="awaiting")
    publish_faces(art)
    ok = client.get("/api/jobs/j1/faces/face_cand_0.jpg")
    assert ok.status_code == 200 and ok.content.startswith(b"\xff\xd8")
    assert client.get("/api/jobs/j1/faces/..%2Fmusic.json").status_code in (404, 422)
    assert client.get("/api/jobs/j1/faces/music.json").status_code == 404


def test_confirm_face_requeues_and_is_idempotent(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, status="awaiting")
    publish_faces(art)
    first = client.post("/api/jobs/j1/face", json={"candidate_id": "1"})
    assert first.status_code == 200 and first.json()["status"] == "queued"
    again = client.post("/api/jobs/j1/face", json={"candidate_id": "1"})
    assert again.status_code == 200
    assert module.store.get("j1")["face_choice"] == "1"


def test_confirm_rejects_unknown_candidate_unknown_job_and_not_waiting(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, status="awaiting")
    publish_faces(art)
    assert client.post("/api/jobs/j1/face", json={"candidate_id": "9"}).status_code == 422
    assert client.post("/api/jobs/nope/face", json={"candidate_id": "0"}).status_code == 404
    new_job(module, tmp_path, job_id="j2")  # queued, not waiting
    assert client.post("/api/jobs/j2/face", json={"candidate_id": "0"}).status_code == 409


def completed_job(module, tmp_path, with_music=True):
    art = new_job(module, tmp_path)
    voice = make_video(art / "voice_video.mp4", seconds=2.0)
    final = art / "final_video.mp4"
    final.write_bytes(voice.read_bytes())
    artifacts = {"final_video": str(final), "voice_video": str(voice)}
    if with_music:
        track = make_tone(art / "music_track.mp3", seconds=2.0)
        (art / "music.json").write_text(json.dumps({"status": "ok", "title": "Song", "artist": "Artist"}), encoding="utf-8")
        artifacts["music_track"] = str(track)
    else:
        (art / "music.json").write_text(json.dumps({"status": "not_identified"}), encoding="utf-8")
    module.store.claim()
    module.store.finish("j1", artifacts)
    return art, final


def test_remix_rebuilds_the_final_video_at_the_new_percent(api, tmp_path):
    module, client = api
    art, final = completed_job(module, tmp_path)
    client.post("/api/jobs/j1/remix", json={"music_percent": 0})
    quiet = mean_volume_db(final)
    response = client.post("/api/jobs/j1/remix", json={"music_percent": 100})
    assert response.status_code == 200
    assert response.json() == {"music_percent": 100, "final_video": "/api/jobs/j1/artifacts/final_video"}
    assert mean_volume_db(final) > quiet + 1.0
    assert json.loads((art / "music.json").read_text())["percent"] == 100


def test_remix_rejects_bad_values_missing_music_and_unfinished_jobs(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path, with_music=False)
    assert client.post("/api/jobs/j1/remix", json={"music_percent": 30}).status_code == 409
    assert client.post("/api/jobs/j1/remix", json={"music_percent": 101}).status_code == 422
    assert client.post("/api/jobs/j1/remix", json={"music_percent": -1}).status_code == 422
    assert client.post("/api/jobs/nope/remix", json={"music_percent": 30}).status_code == 404
    new_job(module, tmp_path, job_id="j2")
    assert client.post("/api/jobs/j2/remix", json={"music_percent": 30}).status_code == 409


def test_status_includes_music_info(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path)
    body = client.get("/api/jobs/j1").json()
    assert body["music"]["title"] == "Song"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: FAIL (`404` on the new routes / missing `store.await_face` wiring).

- [ ] **Step 3: Edit `api.py`**

Update imports:

```python
import json
import re
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .contracts import AwaitingFaceSelection, PipelineStageError
from .mixer import MixError, mix_final
```

(keep the existing imports; `json` and `re` are new.) Add near the top-level constants:

```python
CROP_NAME = re.compile(r"^face_cand_\d+\.jpg$")
remix_lock = threading.Lock()


class FaceChoice(BaseModel):
    candidate_id: str


class RemixRequest(BaseModel):
    music_percent: int = Field(ge=0, le=100)


def _artifacts_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id / "artifacts"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
```

In `_process_job`, pass the choice and handle the pause. Change the `run_operation(...)` call to add `face_choice=job.get("face_choice"),`. Add this `except` clause **before** the existing `except Exception as exc:`:

```python
    except AwaitingFaceSelection:
        store.await_face(job_id)
        print(f"[job {job_id}] waiting for the user to confirm which face to lip-sync",
              file=sys.__stderr__ or sys.stderr, flush=True)
        return
```

In `_worker`, add `store.expire_stale()` as the first statement inside the `try:` of the loop (before `job = store.claim()`).

Add `"music": _read_json(_artifacts_dir(job_id) / "music.json"),` to the dict returned by `job_status`.

Extend the media-type detection in `get_artifact` to:

```python
    suffix = path.suffix.lower()
    media = {
        ".mp4": "video/mp4", ".srt": "application/x-subrip", ".mp3": "audio/mpeg",
        ".json": "application/json", ".jpg": "image/jpeg",
    }.get(suffix, "application/octet-stream")
```

Add the new endpoints (after `get_artifact`):

```python
@app.get("/api/jobs/{job_id}/faces")
def get_faces(job_id: str):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    panel = _read_json(_artifacts_dir(job_id) / "faces.json")
    if panel is None:
        raise HTTPException(404, "No face choices available for this job")
    return {
        "status": record["status"],
        "candidates": panel["candidates"],
        "music": _read_json(_artifacts_dir(job_id) / "music.json"),
    }


@app.get("/api/jobs/{job_id}/faces/{filename}")
def get_face_crop(job_id: str, filename: str):
    if store.get(job_id) is None:
        raise HTTPException(404, "Job not found")
    if not CROP_NAME.match(filename):
        raise HTTPException(404, "Crop not found")
    path = _artifacts_dir(job_id) / filename
    if not path.is_file():
        raise HTTPException(404, "Crop not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/jobs/{job_id}/face")
def confirm_face(job_id: str, body: FaceChoice):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    panel = _read_json(_artifacts_dir(job_id) / "faces.json")
    if panel is None:
        raise HTTPException(409, "This job has no face choices to confirm")
    if body.candidate_id not in {candidate["id"] for candidate in panel["candidates"]}:
        raise HTTPException(422, "Unknown face candidate")
    outcome = store.confirm_face(job_id, body.candidate_id)
    if outcome == "not_awaiting":
        raise HTTPException(409, "This job is not waiting for a face choice")
    return {"status": store.get(job_id)["status"]}


@app.post("/api/jobs/{job_id}/remix")
def remix(job_id: str, body: RemixRequest):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    if record["status"] != "completed":
        raise HTTPException(409, "The job has not completed yet")
    art = _artifacts_dir(job_id)
    info = _read_json(art / "music.json")
    voice, track = art / "voice_video.mp4", art / "music_track.mp3"
    if not info or info.get("status") != "ok" or not voice.is_file() or not track.is_file():
        raise HTTPException(409, "No identified music track for this job")
    final_path = record["artifacts"].get("final_video")
    if not final_path:
        raise HTTPException(409, "The job has no final video to rebuild")
    subtitles = art / "visual_subtitles.srt"
    try:
        with remix_lock:
            mix_final(voice, track, body.music_percent, Path(final_path),
                      subtitles=subtitles if subtitles.is_file() else None)
            info["percent"] = body.music_percent
            (art / "music.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    except MixError as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"music_percent": body.music_percent, "final_video": f"/api/jobs/{job_id}/artifacts/final_video"}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_api.py -v` then `.venv/bin/python -m pytest tests -q`
Expected: all pass. If `importlib.reload(api_module)` raises because the module is not importable as `dir_language_pipeline.api`, confirm `tests/conftest.py` registers the package (Task 0).

- [ ] **Step 5: Commit**

```bash
git add api.py tests/test_api.py
git commit -m "feat: add face confirmation and music remix endpoints

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 11: UI — face panel and music slider

**Files:**
- Modify: `static/index.html`
- Test: manual (no JS test infrastructure exists in this repo); a Python test pins the markup hooks.

**Interfaces:**
- Consumes: `GET /api/jobs/{id}` (`status`, `music`, `artifacts`), `GET /api/jobs/{id}/faces`, `POST /api/jobs/{id}/face`, `POST /api/jobs/{id}/remix`.
- Produces: the face panel (candidate crops, recommended one pre-selected, music line, **Confirm & continue**) and the music panel (slider 0-100 default 25, **Apply**) described in the spec. All dynamic text is set with `textContent` (no `innerHTML`).

- [ ] **Step 1: Write the failing markup test**

`tests/test_ui_markup.py`:

```python
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")


def test_ui_has_face_panel_and_music_slider_hooks():
    for needle in ("awaiting_face_selection", "/faces`", "/face`", "/remix`", "type = 'range'",
                   "Confirm & continue", "Apply", "music_percent"):
        assert needle in HTML, needle


def test_ui_never_uses_inner_html_for_dynamic_text():
    assert "innerHTML" not in HTML
```

Run: `.venv/bin/python -m pytest tests/test_ui_markup.py -v`
Expected: FAIL (first test).

- [ ] **Step 2: Add styles**

Inside the existing `<style>` block add (adjust colors to match existing variables if present):

```css
.faces { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
.face { border: 2px solid #ccc; border-radius: 8px; padding: 6px; cursor: pointer; text-align: center; max-width: 160px; }
.face.selected { border-color: #2563eb; }
.face img { width: 140px; height: 140px; object-fit: cover; border-radius: 6px; display: block; }
.music-panel { margin: 12px 0; }
.music-panel input[type=range] { width: 260px; vertical-align: middle; }
```

- [ ] **Step 3: Replace the `poll` function**

In the `<script>`, inside the submit handler, replace the whole `const poll = async () => { ... };` definition (and keep the following `poll();` call) with:

```js
    const jobId = data.job_id;
    const poll = async () => {
      const statusResponse = await fetch(data.status_url);
      const job = await statusResponse.json();
      result.replaceChildren();
      const heading = document.createElement('p');
      heading.textContent = `${job.operation}: ${job.status} · ${job.stage}`;
      result.append(heading);
      const logLink = document.createElement('a'); logLink.href = job.log_url; logLink.textContent = 'View complete process log'; result.append(logLink);
      if (job.error) { const error = document.createElement('p'); error.textContent = job.error; result.append(error); }
      if (job.status === 'awaiting_face_selection') { await renderFacePanel(jobId, poll); return; }
      if (job.status === 'completed') {
        const links = document.createElement('div'); links.className = 'links';
        const labels = {final_video:'Processed video',visual_subtitles:'Visual subtitles',audio_subtitles:'Audio subtitles',masking_video:'Masking video',dubbed_video:'Dubbed video',service_a_video:'Service A video',service_b_video:'Service B video',service_b_audio:'Dubbed audio',voice_video:'Voice-only video',music_track:'Music track',music_info:'Music info'};
        for (const [name, url] of Object.entries(job.artifacts)) {
          const link = document.createElement('a'); link.href = url; link.textContent = labels[name] || name.replaceAll('_',' '); link.download = ''; links.append(link);
        }
        const manifest = document.createElement('a'); manifest.href = job.manifest_url; manifest.textContent = 'job manifest'; links.append(manifest);
        result.append(links);
        const video = document.createElement('video'); video.controls = true; video.src = job.artifacts.final_video; result.append(video);
        renderMusicPanel(jobId, job, video);
        buttons.forEach(button => button.disabled = false); return;
      }
      if (job.status === 'failed') { buttons.forEach(button => button.disabled = false); return; }
      setTimeout(poll, 2500);
    };
```

- [ ] **Step 4: Add the two panel functions**

Add before the `form.addEventListener('submit', ...)` line:

```js
async function renderFacePanel(jobId, resume) {
  const response = await fetch(`/api/jobs/${jobId}/faces`);
  const panel = document.createElement('section');
  if (!response.ok) { panel.textContent = 'Face choices are not ready yet…'; result.append(panel); setTimeout(resume, 2500); return; }
  const data = await response.json();
  const title = document.createElement('p');
  title.textContent = 'Which person should be lip-synced? Pick the speaker, then continue.';
  panel.append(title);
  const music = document.createElement('p');
  if (data.music && data.music.status === 'ok') music.textContent = `Music found: ${data.music.artist} – ${data.music.title}`;
  else music.textContent = 'Music not identified — the video will be voice-only.';
  panel.append(music);
  const row = document.createElement('div'); row.className = 'faces';
  let chosen = (data.candidates.find(c => c.recommended) || data.candidates[0]).id;
  for (const candidate of data.candidates) {
    const card = document.createElement('div'); card.className = 'face' + (candidate.id === chosen ? ' selected' : '');
    const image = document.createElement('img'); image.src = `/api/jobs/${jobId}/faces/${candidate.crop}`; image.alt = `Face ${candidate.id}`;
    const caption = document.createElement('small');
    caption.textContent = `${candidate.shots} shot(s) · ${Number(candidate.seconds).toFixed(1)}s${candidate.recommended ? ' · recommended' : ''}`;
    card.append(image, caption);
    card.addEventListener('click', () => {
      chosen = candidate.id;
      row.querySelectorAll('.face').forEach(el => el.classList.remove('selected'));
      card.classList.add('selected');
    });
    row.append(card);
  }
  panel.append(row);
  const confirm = document.createElement('button'); confirm.type = 'button'; confirm.textContent = 'Confirm & continue';
  const note = document.createElement('p');
  confirm.addEventListener('click', async () => {
    confirm.disabled = true;
    const reply = await fetch(`/api/jobs/${jobId}/face`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({candidate_id: chosen})});
    if (!reply.ok) { const err = await reply.json().catch(() => ({})); note.textContent = err.detail || 'Could not confirm the face.'; confirm.disabled = false; return; }
    resume();
  });
  panel.append(confirm, note);
  result.append(panel);
}

function renderMusicPanel(jobId, job, video) {
  const music = job.music;
  if (!music) return;
  const panel = document.createElement('div'); panel.className = 'music-panel';
  if (music.status !== 'ok') {
    panel.textContent = music.status === 'download_failed'
      ? `Music identified (${music.artist} – ${music.title}) but could not be downloaded — voice-only video.`
      : 'Music not identified — voice-only video.';
    result.append(panel); return;
  }
  const label = document.createElement('p'); label.textContent = `Music: ${music.artist} – ${music.title}`;
  const slider = document.createElement('input'); slider.type = 'range'; slider.min = 0; slider.max = 100;
  slider.value = music.percent ?? 25;
  const value = document.createElement('output'); value.textContent = `${slider.value}%`;
  slider.addEventListener('input', () => { value.textContent = `${slider.value}%`; });
  const apply = document.createElement('button'); apply.type = 'button'; apply.textContent = 'Apply';
  const status = document.createElement('small');
  apply.addEventListener('click', async () => {
    apply.disabled = true; status.textContent = 'Mixing…';
    const reply = await fetch(`/api/jobs/${jobId}/remix`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({music_percent: Number(slider.value)})});
    const body = await reply.json().catch(() => ({}));
    if (reply.ok) { video.src = `${body.final_video}?v=${Date.now()}`; video.load(); status.textContent = `Applied ${body.music_percent}%`; }
    else status.textContent = body.detail || 'Mix failed';
    apply.disabled = false;
  });
  panel.append(label, slider, value, apply, status);
  result.append(panel);
}
```

- [ ] **Step 5: Run the markup test**

Run: `.venv/bin/python -m pytest tests/test_ui_markup.py -v`
Expected: 2 passed.

- [ ] **Step 6: Manual UI check (with a stub job)**

Run the app: `PIPELINE_DATA_DIR=/tmp/ui-check .venv/bin/python -m uvicorn dir_language_pipeline.api:app --port 8000` from the **parent** directory of `Plix-Project-1` after symlinking: `ln -s Plix-Project-1 dir_language_pipeline`. In another shell create a job in the SQLite store in the `awaiting_face_selection` state with `faces.json` and a crop (same files as the API test fixture) and open `http://localhost:8000`. Confirm: crops render, the recommended card is highlighted, clicking another card moves the highlight, **Confirm & continue** posts and the page resumes polling. Then mark the job completed with a voice video and music track and confirm the slider appears, **Apply** swaps the video source. Remove the symlink afterwards (`rm dir_language_pipeline`).

- [ ] **Step 7: Commit**

```bash
git add static/index.html tests/test_ui_markup.py
git commit -m "feat: add face confirmation panel and music level slider

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Docs, full verification, manual end-to-end

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update `README.md`**

Replace the pipeline diagram code block with:

```text
video upload
  -> Service A: translated on-screen video + visual SRT
  -> analyze: transcript, shot cuts, face candidates, background music (Shazam -> YouTube -> MP3)
  -> pause: user confirms which face to lip-sync (status awaiting_face_selection)
  -> Service B: voice-only dub, lip-sync on shots with the locked face only
  -> final mix: voice + music x slider% + Service A visual SRT track
  -> final MP4 + separate Service A and Service B SRT downloads
```

Add a new section after "Run with Docker":

```markdown
## Face lock and background music

Dubbing and Combine now pause after analysis. The job page shows the faces that appear while someone speaks; confirm the speaker and the job resumes (`POST /api/jobs/{job_id}/face` with `{"candidate_id": "0"}`). Only shots where that person is visible are lip-synced; Sync Labs receives per-frame bounding boxes of the locked face. If no face is found the video is dubbed audio-only. Jobs waiting for confirmation expire after 24 hours.

The ad's background music is identified with Shazam (`shazamio`, no API key), downloaded from YouTube with `yt-dlp`, aligned to the video, and mixed with the dubbed voice. Use the **Music level** slider on the result page (or `POST /api/jobs/{job_id}/remix` with `{"music_percent": 0-100}`) to change the mix; this runs only ffmpeg. If the music is not identified or cannot be downloaded the video is voice-only and the slider is disabled. `music.json` records the artist, title, and YouTube URL used.

Downloading commercial tracks from YouTube is a licensing and terms-of-service decision for ads that will run; review it before shipping.
```

Add to the Artifacts list: `voice_video` (voice-only video before the music mix), `music_track` (aligned MP3), `music_info` (`music.json`).

- [ ] **Step 2: Run the entire test suite**

Run: `.venv/bin/python -m pytest tests -v`
Expected: every test passes (roughly 85 tests). No test touches the network.

- [ ] **Step 3: Check nothing paid or secret is committed**

Run: `git status --short` and `git diff --stat main..HEAD`
Expected: only intended files; `.env` is ignored.

- [ ] **Step 4: Commit the docs**

```bash
git add README.md
git commit -m "docs: document face lock and background music mix

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Manual end-to-end check (needs real keys, uses paid credits)**

With `ELEVENLABS_API_KEY`, `SYNC_API_KEY`, `GEMINI_API_KEY` set in `.env`, run the app via Docker or uvicorn and submit a short (15-30 s) UGC ad that has music under a speaker, with Combine and a supported target language (for example Hindi). Verify and record the result of each:

1. Job reaches `awaiting_face_selection`; the panel shows the speaker's face (and any other person) and a music line.
2. Confirming the face resumes; `process.log` shows `[Faces] Locked face candidate N` and per-shot `[LipSync]` lines. In `cache/service-b/<job>/job/scenes.json` shots without the speaker show `lip_sync: skipped_not_target`.
3. The final video plays with the dubbed voice, lip-sync on the chosen person only, translated on-screen text, and the music at 25%.
4. Move the slider to 0, 60, 100 and Apply: the audio changes accordingly and no new ElevenLabs or Sync Labs calls appear in the log.
5. `music.json` has the right artist/title. Open its `source_url` and confirm it is the same song and that the music in the video lines up (the alignment depends on Shazam's `offset`; if it is consistently off, log the raw Shazam `matches[0]` and revisit `music_song_start`).
6. Negative checks: an ad with no music shows the voice-only message and a disabled slider; an ad with no visible face skips the pause and logs `No usable face found`.

Report any deviation as a bug before merging.
