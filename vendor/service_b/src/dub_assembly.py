"""Frame-bounded assembly that retains each provider result's audio."""
from pathlib import Path
import subprocess
from .audio import ffmpeg
from .common import PipelineError
from .scene_analyzer import get_video_info


def route_model(item, presenter_model, broll_model):
    # A rejected or unverified face must never be animated using narration.
    # The broll_model argument is retained for callers of the older interface.
    return presenter_model if item.get("is_speech") and item.get("classification") != "b_roll" else None


def assemble_segment(video, audio, output, frames, fps, audio_start=0, video_start=0):
    if get_video_info(video)["duration"] - video_start + 0.5 / fps < frames / fps:
        raise PipelineError(f"Segment too short for its timeline slot: {video}")
    duration = frames / fps
    cmd = [ffmpeg(), "-v", "error", "-nostdin", "-y", "-ss", f"{video_start:.9f}", "-i", str(video),
           "-ss", f"{audio_start:.9f}", "-i", str(audio),
           "-map", "0:v:0", "-map", "1:a:0",
           "-vf", f"setpts=PTS-STARTPTS,fps={fps},trim=end_frame={frames},setpts=PTS-STARTPTS",
           "-af", f"atrim=duration={duration:.9f},asetpts=PTS-STARTPTS,apad,atrim=duration={duration:.9f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2", str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise PipelineError(f"Cannot assemble segment: {result.stderr}")
    if get_video_info(output)["frame_count"] != frames:
        raise PipelineError(f"Incorrect assembled frame count: {output}")
    return Path(output)


def join_segments(segments, output, total_frames, fps):
    listing = Path(output).with_suffix(".concat.txt")
    listing.write_text("\n".join("file '" + Path(p).resolve().as_posix().replace("'", "'\\''") + "'" for p in segments), encoding="utf-8")
    result = subprocess.run([ffmpeg(), "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0",
                             "-i", str(listing), "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                             "-t", f"{total_frames / fps:.9f}", str(output)], capture_output=True, text=True)
    if result.returncode:
        raise PipelineError(f"Cannot join segments: {result.stderr}")
    if get_video_info(output)["frame_count"] != total_frames:
        raise PipelineError("Final video frame count does not match source")
