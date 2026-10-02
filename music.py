from __future__ import annotations

import asyncio
import json
import re
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
        if any(re.search(rf"\b{re.escape(word)}\b", name) and not re.search(rf"\b{re.escape(word)}\b", wanted) for word in _BANNED):
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
