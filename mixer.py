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
            f"[1:a]loudnorm=I=-16:TP=-1.5:LRA=11,volume={gain:.4f},apad[m];"
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
