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
