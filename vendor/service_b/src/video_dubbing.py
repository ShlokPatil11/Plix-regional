import json
from pathlib import Path
import cv2
from .scene_analyzer import (
    detect_scenes,
    extract_subclip,
    extract_frame,
    concat_segments,
    mux_audio_video,
    get_video_info
)
from .face_tracker import (
    classify_shot,
    extract_reference_signature_from_image,
    detect_faces,
    compute_face_signature
)
from .lipsync_hf import sync_clip
from .audio import ffmpeg
from .common import PipelineError, write_json
import subprocess


def extract_audio_segment(master_audio_path, start_sec, end_sec, output_path):
    """
    Extracts an exact audio chunk from the master audio track.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec

    args = [
        ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(master_audio_path),
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        str(output_path)
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise PipelineError(f"Failed to slice audio chunk [{start_sec}-{end_sec}]: {res.stderr}")
    return output_path


def auto_extract_reference_character(video_path, output_image_path):
    """
    Automatically finds the primary face in the video to anchor character-locking.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps))  # sample once every second

    best_crop = None
    best_area = 0

    for idx in range(0, min(frame_count, int(fps * 30)), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        faces = detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]
            area = w * h
            if area > best_area:
                best_area = area
                best_crop = frame[y:y+h, x:x+w]

    cap.release()

    if best_crop is not None:
        output_image_path = Path(output_image_path)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), best_crop)
        return compute_face_signature(best_crop)
    return None


def run_video_dubbing(
    source_video_path,
    master_audio_path,
    job_dir,
    reference_image_path=None,
    dry_run=False,
    space_name=None,
    hf_token=None,
    scene_threshold=32.0,
    min_scene_sec=0.6
):
    """
    Main orchestrator for Character-Locked, B-Roll Aware Video Lip-Sync.
    """
    source_video_path = Path(source_video_path).resolve()
    master_audio_path = Path(master_audio_path).resolve()
    job_dir = Path(job_dir).resolve()
    seg_dir = job_dir / "video_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(source_video_path)
    print(f"[Video Pipeline] Source video loaded: {info['width']}x{info['height']} @ {info['fps']}fps, duration: {info['duration']:.2f}s")

    # Step 1: Detect Shots & Camera Cuts
    print("[Video Pipeline] Step 1: Analyzing video shots and camera cuts...")
    scenes = detect_scenes(source_video_path, threshold=scene_threshold, min_scene_len_sec=min_scene_sec)
    print(f"[Video Pipeline] Found {len(scenes)} distinct shots.")

    # Step 2: Character Anchor Setup
    print("[Video Pipeline] Step 2: Initializing Character-Locking anchor...")
    ref_sig = None
    if reference_image_path and Path(reference_image_path).exists():
        print(f"[Video Pipeline] Using supplied reference image: {reference_image_path}")
        ref_sig = extract_reference_signature_from_image(reference_image_path)
    else:
        auto_ref_path = job_dir / "character_anchor.jpg"
        print(f"[Video Pipeline] Auto-detecting primary character anchor from video...")
        ref_sig = auto_extract_reference_character(source_video_path, auto_ref_path)
        if ref_sig is not None:
            print(f"[Video Pipeline] Character anchor saved to: {auto_ref_path.name}")
        else:
            print("[Video Pipeline] Notice: No clear face detected for anchor. Proceeding with face presence detection.")

    # Step 3: Shot-by-Shot Analysis & Targeted Lip-Sync
    print("[Video Pipeline] Step 3: Classifying shots (A-Roll vs B-Roll) and generating lip-sync...")
    shot_reports = []
    final_segment_paths = []

    for shot in scenes:
        idx = shot["index"]
        start_s = shot["start"]
        end_s = shot["end"]
        duration = shot["duration"]

        print(f"\n--- Shot #{idx + 1}/{len(scenes)}: [{start_s:.2f}s - {end_s:.2f}s] ({duration:.2f}s) ---")
        
        # Extract subclips for this shot
        raw_clip = seg_dir / f"shot_{idx:03d}_raw.mp4"
        extract_subclip(source_video_path, start_s, end_s, raw_clip)

        audio_clip = seg_dir / f"shot_{idx:03d}_audio.wav"
        extract_audio_segment(master_audio_path, start_s, end_s, audio_clip)

        # Classify Shot
        classification = classify_shot(source_video_path, start_s, end_s, reference_sig=ref_sig)
        shot_type = classification["type"]
        reason = classification["reason"]
        print(f"Classification: {shot_type.upper()} | Reason: {reason}")

        shot_info = {
            "index": idx,
            "start": start_s,
            "end": end_s,
            "duration": duration,
            "type": shot_type,
            "reason": reason,
            "score": classification.get("score", 0.0)
        }

        if shot_type == "b_roll":
            # B-roll: Product, drink, mannequin, or cutaway. Keep video frames untouched!
            shot_info["action"] = "untouched_b_roll"
            final_segment_paths.append(raw_clip)
        else:
            # A-roll: Target character speaking on camera. Perform Lip-Sync!
            shot_info["action"] = "lip_synced_a_roll"
            synced_clip = seg_dir / f"shot_{idx:03d}_synced.mp4"
            print(f"Action: Lip-syncing target character via Hugging Face...")
            sync_clip(
                video_path=raw_clip,
                audio_path=audio_clip,
                output_path=synced_clip,
                space_name=space_name,
                hf_token=hf_token,
                dry_run=dry_run
            )
            final_segment_paths.append(synced_clip)

        shot_reports.append(shot_info)

    # Step 4: Lossless Video Stitching
    print("\n[Video Pipeline] Step 4: Stitching all segments back together...")
    stitched_video = job_dir / "stitched_temp.mp4"
    concat_segments(final_segment_paths, stitched_video)

    # Step 5: Master Audio Muxing
    print("[Video Pipeline] Step 5: Muxing master audio track...")
    final_output_video = job_dir / "final_character_locked_lipsynced.mp4"
    mux_audio_video(stitched_video, master_audio_path, final_output_video)

    # Cleanup temporary stitched file
    try:
        stitched_video.unlink(missing_ok=True)
    except Exception:
        pass

    # Save manifest
    manifest = {
        "source_video": str(source_video_path),
        "master_audio": str(master_audio_path),
        "final_output": str(final_output_video),
        "total_duration": info["duration"],
        "shots_count": len(scenes),
        "a_roll_count": sum(1 for s in shot_reports if s["type"] == "a_roll"),
        "b_roll_count": sum(1 for s in shot_reports if s["type"] == "b_roll"),
        "shots": shot_reports
    }
    write_json(job_dir / "video_dubbing_manifest.json", manifest)

    print(f"\n[Video Pipeline] Completed successfully!")
    print(f"Output Video: {final_output_video}")
    print(f"A-Roll Shots: {manifest['a_roll_count']} | B-Roll Shots (Preserved): {manifest['b_roll_count']}")

    return final_output_video, manifest


def export_aroll_package(source_video_path, master_audio_path, job_dir):
    """
    Extracts only the A-Roll speaking segment and corresponding audio for high-speed,
    high-quality diffusion sync (in Colab or ZeroGPU), while saving all B-roll cuts locally.
    """
    source_video_path = Path(source_video_path).resolve()
    master_audio_path = Path(master_audio_path).resolve()
    job_dir = Path(job_dir).resolve()
    seg_dir = job_dir / "video_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(source_video_path)
    scenes = detect_scenes(source_video_path, threshold=28.0, min_scene_len_sec=0.7)
    
    auto_ref_path = job_dir / "character_anchor.jpg"
    ref_sig = auto_extract_reference_character(source_video_path, auto_ref_path)

    aroll_cuts = []
    all_segments = []

    for shot in scenes:
        idx = shot["index"]
        start_s = shot["start"]
        end_s = shot["end"]
        raw_clip = seg_dir / f"shot_{idx:03d}_raw.mp4"
        audio_clip = seg_dir / f"shot_{idx:03d}_audio.wav"

        extract_subclip(source_video_path, start_s, end_s, raw_clip)
        extract_audio_segment(master_audio_path, start_s, end_s, audio_clip)

        c = classify_shot(source_video_path, start_s, end_s, reference_sig=ref_sig)
        is_aroll = (c["type"] == "a_roll")

        seg_info = {
            "index": idx,
            "start": start_s,
            "end": end_s,
            "type": c["type"],
            "raw_video": str(raw_clip),
            "audio": str(audio_clip),
            "is_aroll": is_aroll
        }
        all_segments.append(seg_info)
        if is_aroll:
            aroll_cuts.append(seg_info)

    export_info = {
        "source_video": str(source_video_path),
        "master_audio": str(master_audio_path),
        "aroll_clips": aroll_cuts,
        "all_segments": all_segments
    }
    write_json(job_dir / "package_state.json", export_info)

    print(f"\n[A-Roll Package Exported]")
    print(f"Total Shots: {len(scenes)} | A-Roll Speaking Shots to Sync: {len(aroll_cuts)}")
    for a in aroll_cuts:
        print(f" -> A-Roll Video to upload: {a['raw_video']}")
        print(f" -> A-Roll Audio to upload: {a['audio']}")

    return export_info


def assemble_from_synced(job_dir, synced_aroll_video):
    """
    Takes the high-resolution lip-synced A-roll video, losslessly recombines it
    with the untouched B-roll shots, and muxes with the 48kHz master audio.
    """
    job_dir = Path(job_dir).resolve()
    state_file = job_dir / "package_state.json"
    if not state_file.exists():
        raise PipelineError(f"Missing package_state.json in {job_dir}. Run export first.")
    
    from .common import read_json
    state = read_json(state_file)
    master_audio = Path(state["master_audio"])
    synced_clip = Path(synced_aroll_video).resolve()

    final_segments = []
    aroll_used = False

    for seg in state["all_segments"]:
        if seg["is_aroll"] and not aroll_used:
            final_segments.append(synced_clip)
            aroll_used = True
        else:
            final_segments.append(Path(seg["raw_video"]))

    stitched_video = job_dir / "stitched_temp.mp4"
    concat_segments(final_segments, stitched_video)

    final_output = job_dir / "final_1080p_studio_lipsynced.mp4"
    mux_audio_video(stitched_video, master_audio, final_output)

    try:
        stitched_video.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"\n[Studio Assembly Complete!]")
    print(f"Master Output Video: {final_output}")
    return final_output


