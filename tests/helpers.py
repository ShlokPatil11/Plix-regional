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
