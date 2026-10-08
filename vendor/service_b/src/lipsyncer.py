"""
Stage 4 — Targeted Lip-Sync
For each A-Roll shot:
  1. Extract video clip from source (exact frame bounds)
  2. Extract dubbed audio slice from master dubbed audio
  3. Submit to Sync Labs lipsync-2-pro (active_speaker_detection on)
  4. Cache by sha256(video_clip + audio_clip) — never re-calls if synced file exists

For each B-Roll shot:
  1. Extract video clip (untouched original visuals)
  2. Extract dubbed audio slice
  3. Mux audio onto original video clip

For SILENT shots:
  1. Copy source video clip untouched (no audio change)
"""
import hashlib
import subprocess
from pathlib import Path

from .common import PipelineError
from .scene_analyzer import get_video_info, extract_subclip
from .synclabs_client import SyncLabsClient
from .audio import ffmpeg
from .faces import (
    FaceEngine,
    FaceLock,
    LOCK_MIN_PRESENCE,
    build_sync_options,
    locked_boxes_for_clip,
)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(2 * 1024 * 1024))  # First 2MB sufficient
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:12]


def _extract_audio_slice(
    audio_path: Path, start_s: float, duration_s: float, out_path: Path
) -> Path:
    """Cut an exact slice of audio to WAV for Sync Labs input."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg(), "-y", "-nostdin",
        "-ss", f"{start_s:.6f}", "-t", f"{duration_s:.6f}",
        "-i", str(audio_path),
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"Audio slice failed: {r.stderr}")
    return out_path


def _mux_audio_onto_video(
    video_path: Path, audio_path: Path, out_path: Path, duration_s: float
) -> Path:
    """Replace video's audio track with dubbed audio (for B-Roll shots)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg(), "-y", "-nostdin",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-af", f"atrim=duration={duration_s:.6f},asetpts=PTS-STARTPTS,apad,atrim=duration={duration_s:.6f}",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"Audio mux failed: {r.stderr}")
    return out_path


def _copy_source_clip(
    video_path: Path, start_s: float, duration_s: float,
    fps: float, out_path: Path
) -> Path:
    """Copy source clip bit-for-bit (SILENT or untouched shots)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extract_subclip(video_path, start_s, start_s + duration_s, out_path, target_fps=fps)
    return out_path


def _audio_only_clip(raw_clip: Path, audio_slice: Path, out_path: Path, duration_s: float) -> Path:
    """Dubbed audio over the untouched video (no mouth edit)."""
    if not out_path.exists():
        _mux_audio_onto_video(raw_clip, audio_slice, out_path, duration_s)
    return out_path


def process_full_video(
    video_path: Path,
    dubbed_audio: Path,
    job_dir: Path,
    sync_model: str = "lipsync-2-pro",
    dry_run: bool = False,
    lock: "FaceLock | None" = None,
    face_engine=None,
) -> Path:
    """Submit one continuous video and dubbed-audio track to Sync Labs."""
    job_dir = Path(job_dir)
    full_dir = job_dir / "full_video_sync"
    full_dir.mkdir(parents=True, exist_ok=True)
    duration = get_video_info(video_path)["duration"]
    audio_wav = full_dir / "dubbed_full.wav"
    if not audio_wav.exists():
        _extract_audio_slice(dubbed_audio, 0.0, duration, audio_wav)

    fallback = full_dir / "full_audio_only.mp4"
    if dry_run or (lock is not None and not lock.enabled):
        reason = "DRY RUN" if dry_run else "no face locked"
        print(f"[LipSync] Full-video sync skipped ({reason}); dubbing audio only")
        return _audio_only_clip(video_path, audio_wav, fallback, duration)

    lock_tag = f"_{lock.tag}" if lock is not None else "_AUTO"
    content_hash = _file_hash(video_path) + _file_hash(audio_wav)
    synced_out = full_dir / f"full_{content_hash}{lock_tag}_synced.mp4"
    if synced_out.exists():
        print(f"[LipSync] Reusing cached full-video Sync result: {synced_out.name}")
        return synced_out

    options = {
        "sync_mode": "cut_off",
        "active_speaker_detection": {"auto_detect": True},
    }
    if lock is not None and lock.enabled:
        engine = face_engine or FaceEngine()
        boxes, presence = locked_boxes_for_clip(video_path, engine, lock)
        print(f"[LipSync] Selected face is visible in {presence:.0%} of full-video frames")
        if not any(box is not None for box in boxes):
            print("[LipSync] Selected face was not found; dubbing audio only")
            return _audio_only_clip(video_path, audio_wav, fallback, duration)
        options["active_speaker_detection"] = build_sync_options(boxes)

    print(f"[LipSync] Uploading complete {duration:.2f}s video and dubbed audio to Sync Labs {sync_model}...")
    client = SyncLabsClient(default_model=sync_model)
    try:
        video_url = client.upload_file(video_path, content_type="video/mp4")
        audio_url = client.upload_file(audio_wav, content_type="audio/wav")
        generation = client.generate(video_url, audio_url, model=sync_model, options=options)
        print(f"[LipSync] Full-video Sync job: {generation['id']} — waiting...")
        result = client.wait_for_completion(generation["id"], poll_interval=8)
        client.download_result(result["outputUrl"], synced_out)
        print(f"[LipSync] Downloaded full-video result → {synced_out.name}")
        return synced_out
    except PipelineError as exc:
        print(f"[LipSync] Full-video Sync failed ({exc}); using dubbed audio without mouth edits")
        return _audio_only_clip(video_path, audio_wav, fallback, duration)


def restore_broll_from_source(
    synced_video: Path,
    source_video: Path,
    classified_scenes: list[dict],
    job_dir: Path,
) -> list[Path]:
    """Split a full-video Sync result and replace B-roll with original frames.

    Sync Labs gets the continuous video so that face lock and mouth motion have
    full temporal context.  After it returns, Service B's existing scene map is
    used only as an edit decision list: B_ROLL ranges come from ``source_video``
    and every other range comes from ``synced_video``.  Stage 5 subsequently
    lays the continuous dubbed master audio over these video-only decisions.
    """
    synced_video = Path(synced_video)
    source_video = Path(source_video)
    restore_dir = Path(job_dir) / "full_video_sync" / "restored_segments"
    restore_dir.mkdir(parents=True, exist_ok=True)

    source_info = get_video_info(source_video)
    fps = source_info["fps"]
    source_duration = source_info["duration"]
    output_clips: list[Path] = []
    synced_hash = _file_hash(synced_video)
    source_hash = _file_hash(source_video)

    for i, scene in enumerate(classified_scenes):
        start_s = max(0.0, float(scene["start"]))
        end_s = min(float(scene["end"]), source_duration)
        if end_s <= start_s:
            continue

        scene_class = scene.get("class", "B_ROLL")
        use_original = scene_class in {"B_ROLL", "SILENT"}
        class_tag = "broll" if scene_class == "B_ROLL" else scene_class.lower()
        selected_video = source_video if use_original else synced_video
        kind = f"original_{class_tag}" if use_original else "synced"
        media_hash = source_hash if use_original else synced_hash
        time_key = f"{int(round(start_s * 1000))}_{int(round(end_s * 1000))}"
        out = restore_dir / f"seg_{i:03d}_{time_key}_{media_hash}_{kind}.mp4"
        if not out.exists():
            extract_subclip(selected_video, start_s, end_s, out, target_fps=fps)

        scene["lip_sync"] = f"restored_original_{class_tag}" if use_original else "full_video"
        output_clips.append(out)

    if not output_clips:
        raise PipelineError("Full-video B-roll restoration produced no segments.")

    broll_restored = sum(1 for scene in classified_scenes if scene.get("class") == "B_ROLL")
    silent_restored = sum(1 for scene in classified_scenes if scene.get("class") == "SILENT")
    print(f"[LipSync] Restored {broll_restored} B-roll and {silent_restored} silent "
          "section(s) from the untouched source")
    return output_clips


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
    """
    Stage 4 entry point.
    Processes all shots and returns ordered list of output video clip paths.

    A_ROLL → Sync Labs lip-sync (cached by content hash)
    B_ROLL → Original video + dubbed audio mux
    SILENT → Original video clip (untouched)

    lock=None keeps the legacy behavior (Sync Labs auto-detects the speaker).
    A disabled lock (no face to lock) dubs audio-only. An enabled lock lip-syncs a
    shot only when the locked identity is present in >= LOCK_MIN_PRESENCE of its
    frames, sending that identity's per-frame bounding boxes to Sync Labs.
    """
    job_dir = Path(job_dir)
    seg_dir = job_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(video_path)
    fps = info["fps"]
    source_duration = info["duration"]

    sync_client = None
    if not dry_run:
        sync_client = SyncLabsClient(default_model=sync_model)

    output_clips: list[Path] = []
    lock_tag = f"_{lock.tag}" if lock is not None else ""
    if lock is not None and lock.enabled and face_engine is None and not dry_run:
        face_engine = FaceEngine()

    for i, scene in enumerate(classified_scenes):
        start_s = scene["start"]
        end_s = min(scene["end"], source_duration)
        duration_s = end_s - start_s
        label = scene.get("class", "B_ROLL")
        prefix = f"seg_{i:03d}"
        # Time-keyed suffix so files are strictly bound to their exact millisecond range
        time_key = f"{int(round(start_s * 1000))}_{int(round(end_s * 1000))}"

        print(f"\n[LipSync] Shot {i:02d}/{len(classified_scenes)-1} "
              f"[{start_s:.2f}s-{end_s:.2f}s] ({duration_s:.2f}s) → {label}")

        # --- Extract source video clip (strictly time-keyed) ---
        raw_clip = seg_dir / f"{prefix}_{time_key}_raw.mp4"
        if not raw_clip.exists():
            extract_subclip(video_path, start_s, end_s, raw_clip, target_fps=fps)

        # --- Extract dubbed audio slice (strictly time-keyed) ---
        audio_slice = seg_dir / f"{prefix}_{time_key}_dubbed.wav"
        if not audio_slice.exists():
            _extract_audio_slice(dubbed_audio, start_s, duration_s, audio_slice)

        # --------------------------------------------------------
        if label == "SILENT":
            out = seg_dir / f"{prefix}_{time_key}_silent.mp4"
            if not out.exists():
                _copy_source_clip(video_path, start_s, duration_s, fps, out)
            print(f"  [LipSync] SILENT → copied source clip ({duration_s:.2f}s)")
            output_clips.append(out)

        elif label == "B_ROLL":
            out = seg_dir / f"{prefix}_{time_key}_broll.mp4"
            if not out.exists():
                _mux_audio_onto_video(raw_clip, audio_slice, out, duration_s)
            print(f"  [LipSync] B-Roll → original video + dubbed audio muxed")
            output_clips.append(out)

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
                try:
                    boxes, presence = locked_boxes_for_clip(raw_clip, face_engine, lock)
                except PipelineError as e:
                    print(f"  [LipSync] A-Roll → face check failed ({e}); dubbing audio only")
                    out = seg_dir / f"{prefix}_{time_key}_facecheckfail.mp4"
                    _audio_only_clip(raw_clip, audio_slice, out, actual_dur)
                    scene["lip_sync"] = "skipped_face_check_failed"
                    output_clips.append(out)
                    continue
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

    return output_clips
