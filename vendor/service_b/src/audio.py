import math
import os
from pathlib import Path
import shutil
import subprocess
import wave
from .common import PipelineError

RATE = 24000


def ffmpeg():
    executable = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise PipelineError("FFmpeg is missing. Install requirements.txt or set FFMPEG_BINARY.") from None


def convert(source, destination, speed=1.0):
    args = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source)]
    if speed != 1:
        args += ["-af", f"atempo={speed:.8f}"]
    args += ["-ar", str(RATE), "-ac", "1", "-c:a", "pcm_s16le", str(destination)]
    run(args)


def run(args):
    try:
        result = subprocess.run(args, capture_output=True, timeout=300,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        raise PipelineError("Could not run FFmpeg; check installation and input audio.") from None
    if result.returncode:
        raise PipelineError("FFmpeg could not process this audio. Check the saved source segment.")


def frames(path):
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() != RATE or audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise PipelineError("Audio must be mono 24 kHz, 16-bit PCM.")
        if audio.getnframes() == 0:
            raise PipelineError("Generated clip is empty.")
        return audio.getnframes()


def fit(source, destination, available_ms, max_speed=1.10):
    if not math.isfinite(max_speed) or not 1 <= max_speed <= 1.25:
        raise PipelineError("Maximum speed must be between 1.0 and 1.25.")
    original_frames = frames(source)
    budget = available_ms * RATE // 1000
    original_ms = original_frames * 1000 / RATE
    required = original_frames / budget
    speed = 1.0
    if required <= 1:
        shutil.copyfile(source, destination)
    elif required <= max_speed:
        # Small headroom for atempo rounding; never truncate to hide an overflow.
        speed = min(max_speed, required * 1.003)
        convert(source, destination, speed)
    else:
        return {"status": "needs_review", "original_ms": original_ms,
                "duration_ms": original_ms, "speed": 1.0,
                "overflow_ms": original_ms - available_ms, "audio": str(source)}
    actual_frames = frames(destination)
    duration = actual_frames * 1000 / RATE
    return {"status": ("fits" if speed == 1 else "adjusted") if actual_frames <= budget else "needs_review",
            "original_ms": original_ms, "duration_ms": duration, "speed": speed,
            "overflow_ms": max(0, duration - available_ms), "audio": str(destination)}


def assemble(segments, destination, total_ms):
    """Stream PCM in absolute timeline order; memory use is bounded."""
    position = 0
    with wave.open(str(destination), "wb") as output:
        output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        def silence(count):
            while count:
                chunk = min(count, RATE)
                output.writeframesraw(b"\0\0" * chunk)
                count -= chunk
        for item in segments:
            if item.get("skip"):
                continue
            if item.get("status") not in ("fits", "adjusted"):
                raise PipelineError("Cannot assemble unresolved speech.")
            start = item["start_ms"] * RATE // 1000
            end = item["end_ms"] * RATE // 1000
            size = frames(item["audio"])
            if start < position or start + size > end:
                raise PipelineError("Final timeline contains speech overlap or overflow.")
            silence(start - position)
            with wave.open(item["audio"], "rb") as clip:
                while True:
                    data = clip.readframes(RATE)
                    if not data:
                        break
                    output.writeframesraw(data)
            position = start + size
        end = total_ms * RATE // 1000
        if position > end:
            raise PipelineError("Final audio exceeds timeline.")
        silence(end - position)
    return frames(destination) * 1000 / RATE


def export_mp3(source, destination):
    run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", str(source), "-c:a", "libmp3lame", "-b:a", "128k", str(destination)])
    decoded = destination.with_suffix(".verification.wav")
    try:
        convert(destination, decoded)
        difference = abs(frames(decoded) - frames(source)) * 1000 / RATE
        if difference > 25:
            raise PipelineError("MP3 decoded duration differs by more than 25 ms; use the WAV output.")
        return difference
    finally:
        decoded.unlink(missing_ok=True)
