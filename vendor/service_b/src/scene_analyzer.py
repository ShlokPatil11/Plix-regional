from pathlib import Path
import subprocess
import cv2
from .audio import ffmpeg
from .common import PipelineError


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {
        "fps": fps,
        "frame_count": int(frame_count),
        "width": width,
        "height": height,
        "duration": duration
    }


def detect_scenes(video_path, threshold=32.0, min_scene_len_sec=0.6, sample_fps=10):
    """
    Detect camera shots / scene cuts using frame difference analysis.
    Returns a list of dicts: [{'index': 0, 'start': 0.0, 'end': 3.2, 'duration': 3.2}, ...]
    """
    info = get_video_info(video_path)
    total_duration = info["duration"]
    if total_duration <= 0:
        raise PipelineError("Video has 0 duration.")

    cap = cv2.VideoCapture(str(video_path))
    video_fps = info["fps"]
    step = max(1, int(round(video_fps / sample_fps)))
    
    prev_gray = None
    cuts = [0.0]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % step == 0:
            timestamp = frame_idx / video_fps
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if prev_gray is not None:
                diff = cv2.norm(prev_gray, gray, cv2.NORM_L1) / (160 * 90)
                if diff > threshold and (timestamp - cuts[-1]) >= min_scene_len_sec:
                    cuts.append(round(timestamp, 3))
            
            prev_gray = gray
        frame_idx += 1

    cap.release()

    if cuts[-1] < total_duration:
        cuts.append(round(total_duration, 3))

    scenes = []
    for i in range(len(cuts) - 1):
        start = cuts[i]
        end = cuts[i + 1]
        if end - start >= 0.1:
            scenes.append({
                "index": i,
                "start": start,
                "end": end,
                "duration": round(end - start, 3)
            })

    if not scenes:
        scenes = [{"index": 0, "start": 0.0, "end": total_duration, "duration": total_duration}]

    return scenes


def extract_subclip(video_path, start_sec, end_sec, output_path, target_fps=25):
    """
    Extracts a subclip from start_sec to end_sec with frame-accurate boundaries.

    Boundaries are converted to integer frame numbers at target_fps so that
    rounding cannot accumulate across segments.  The PTS of the output is reset
    to start at zero (setpts=PTS-STARTPTS) and exactly (end_frame - start_frame)
    frames are written, ensuring every raw clip has start_time=0 and the correct
    frame count.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert boundaries to integer frame numbers to eliminate float rounding.
    # Using round() so that e.g. 5.589 * 23.976... correctly maps to frame 134.
    start_frame = round(start_sec * target_fps)
    end_frame = round(end_sec * target_fps)
    n_frames = end_frame - start_frame
    if n_frames <= 0:
        raise PipelineError(
            f"Zero-length subclip requested [{start_sec}-{end_sec}] "
            f"(frames {start_frame}-{end_frame})"
        )

    # Express the start position as an exact frame timestamp (avoids seek imprecision
    # for very short offsets while still using fast input seek for longer ones).
    start_pts = start_frame / target_fps

    # Build the video filter chain:
    #   fps=          – re-stamp frames at target_fps
    #   trim=         – keep only the first n_frames frames (exact count)
    #   setpts=PTS-STARTPTS  – reset PTS to 0 so start_time=0 in the container
    vf = f"fps={target_fps},trim=end_frame={n_frames},setpts=PTS-STARTPTS"

    args = [
        ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-ss", f"{start_pts:.6f}",
        "-i", str(video_path),
        "-vf", vf,
        "-af", f"atrim=duration={n_frames / target_fps:.9f},asetpts=PTS-STARTPTS",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "48000",
        "-b:a", "320k",
        str(output_path)
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise PipelineError(
            f"Failed to extract subclip [{start_sec}-{end_sec}]: {res.stderr}"
        )
    return output_path


def extract_frame(video_path, timestamp_sec, output_image_path):
    """
    Extract a single frame image at timestamp_sec.
    """
    output_image_path = Path(output_image_path)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-ss", f"{timestamp_sec:.3f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
        str(output_image_path)
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise PipelineError(f"Failed to extract frame at {timestamp_sec}s: {res.stderr}")
    return output_image_path


def concat_segments(segment_paths, output_path):
    """
    Concatenate video segments losslessly using FFmpeg concat demuxer.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    list_file = output_path.with_suffix(".concat_list.txt")
    lines = []
    for seg in segment_paths:
        seg_str = Path(seg).resolve().as_posix()
        lines.append(f"file '{seg_str}'")
    list_file.write_text("\n".join(lines), encoding="utf-8")

    args = [
        ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path)
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    try:
        list_file.unlink(missing_ok=True)
    except Exception:
        pass

    if res.returncode != 0:
        raise PipelineError(f"Failed to concatenate segments: {res.stderr}")
    return output_path


def mux_audio_video(video_path, audio_path, output_path):
    """
    Replaces the audio in video_path with audio_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "320k",
        "-ar", "48000",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path)
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise PipelineError(f"Failed to mux audio and video: {res.stderr}")
    return output_path

