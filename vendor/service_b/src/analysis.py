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
