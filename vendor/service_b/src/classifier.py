"""
Stage 2 — Scene Classification
Detects camera cuts, then classifies each shot as:
  A_ROLL   — target speaker confirmed visible & mouth moving  → send to lipsync
  B_ROLL   — product/background shot, no target speaker      → dubbed audio + original video
  SILENT   — no speech in this interval                      → copy untouched

Uses:
  - PySceneDetect / ffmpeg for cut detection
  - OpenCV YuNet + SFace for deep face identity (replaces legacy skin-color tracker)
  - Mouth motion score to separate speaking vs posed shots
"""
import json
import cv2
import numpy as np
import subprocess
from pathlib import Path
from typing import Optional

from .common import PipelineError
from .scene_analyzer import detect_scenes, get_video_info

SERVICE_ROOT = Path(__file__).resolve().parent.parent
YUNET_PATH = SERVICE_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = SERVICE_ROOT / "models" / "face_recognition_sface_2021dec.onnx"

AROLL_THRESHOLD = 0.70   # SFace cosine similarity ≥ this → target speaker present
REJECT_THRESHOLD = 0.40  # Below this → definitely not the target speaker
MOTION_THRESHOLD = 4.0   # Mouth motion score above this → speaker is talking


class SceneClassifier:
    """
    Classifies video shots against a locked identity anchor.
    Call set_anchor() first, then classify_scenes() for all shots.
    """

    def __init__(self, yunet_path: Path = None, sface_path: Path = None,
                 conf_threshold: float = 0.7):
        y = str(yunet_path or YUNET_PATH)
        s = str(sface_path or SFACE_PATH)
        self._has_models = False
        self.detector = None
        self.recognizer = None
        if Path(y).exists() and Path(s).exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(y, "", (640, 640), conf_threshold, 0.3, 5000)
                self.recognizer = cv2.FaceRecognizerSF.create(s, "")
                self._has_models = True
            except Exception as exc:
                print(f"[Classifier] Model load warning ({exc}) — falling back to motion-only classification.")
        self.anchor_embedding = None

    # ------------------------------------------------------------------
    # Anchor setup
    # ------------------------------------------------------------------
    def set_anchor_from_video(self, video_path: Path, target_time: float = None,
                               target_point: tuple = None):
        """
        Extract identity anchor from the video itself.
        target_time: timestamp (seconds) showing the speaker — uses midpoint if None.
        target_point: (x, y) pixel point inside her face when multiple faces visible.
        """
        if not self._has_models:
            print("[Classifier] YuNet/SFace models not found — face identity disabled.")
            return None

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Probe multiple candidate frames and pick the one with the clearest face
        probe_times = []
        if target_time is not None:
            probe_times = [target_time, target_time + 0.5, target_time - 0.5]
        else:
            dur = total / fps
            probe_times = [dur * 0.2, dur * 0.35, dur * 0.5, dur * 0.65]

        for t in probe_times:
            if t < 0:
                continue
            frame_idx = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            if faces is None or len(faces) == 0:
                continue

            selected = None
            if target_point:
                px, py = target_point
                for f in faces:
                    fx, fy, fw, fh = f[:4]
                    if fx <= px <= fx + fw and fy <= py <= fy + fh:
                        selected = f
                        break
                if selected is None:
                    continue
            else:
                selected = max(faces, key=lambda f: f[2] * f[3])

            aligned = self.recognizer.alignCrop(frame, selected)
            self.anchor_embedding = self.recognizer.feature(aligned)
            box = [int(v) for v in selected[:4]]
            cap.release()
            print(f"[Classifier] Anchor set from t={t:.2f}s, face box={box}")
            return box

        cap.release()
        print("[Classifier] Warning: could not set identity anchor — will classify by motion only.")
        return None

    def set_anchor_from_image(self, image_path: Path):
        """Set anchor from a reference photo of the speaker."""
        if not self._has_models:
            return None
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise PipelineError(f"Cannot load reference image: {image_path}")
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            raise PipelineError("No face detected in reference image.")
        selected = max(faces, key=lambda f: f[2] * f[3])
        aligned = self.recognizer.alignCrop(frame, selected)
        self.anchor_embedding = self.recognizer.feature(aligned)
        print(f"[Classifier] Anchor set from reference image: {image_path.name}")

    # ------------------------------------------------------------------
    # Per-frame matching
    # ------------------------------------------------------------------
    def _similarity(self, embedding) -> float:
        if self.anchor_embedding is None:
            return -1.0
        return float(self.recognizer.match(
            self.anchor_embedding, embedding, cv2.FaceRecognizerSF_FR_COSINE
        ))

    def _sample_frames(self, video_path: Path, start_s: float, end_s: float,
                        n_samples: int = 8) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        first = int(start_s * fps)
        last = int(end_s * fps)
        step = max(1, (last - first) // n_samples)
        frames = []
        for idx in range(first, last, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()
        return frames

    def _mouth_motion_score(self, frames: list[np.ndarray]) -> float:
        """Estimate mouth region motion via frame-diff in lower-face region."""
        if len(frames) < 2:
            return 0.0
        scores = []
        for i in range(1, len(frames)):
            h, w = frames[i].shape[:2]
            # Lower third of frame (mouth area)
            r1 = cv2.cvtColor(frames[i - 1][h * 2 // 3:, w // 4: w * 3 // 4], cv2.COLOR_BGR2GRAY)
            r2 = cv2.cvtColor(frames[i][h * 2 // 3:, w // 4: w * 3 // 4], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(r1.astype(np.float32), r2.astype(np.float32))
            scores.append(float(diff.mean()))
        return float(np.mean(scores)) if scores else 0.0

    def classify_shot(self, video_path: Path, start_s: float, end_s: float,
                       speech_intervals: list[tuple] = None) -> dict:
        """
        Classify a single shot.
        speech_intervals: list of (start_s, end_s) pairs from transcript where speech occurs.
        Returns dict: {class: 'A_ROLL'|'B_ROLL'|'SILENT', similarity, motion, face_coords}
        """
        # Check if any speech falls in this window
        has_speech = True
        if speech_intervals is not None:
            has_speech = any(
                not (end_s <= si[0] or start_s >= si[1])
                for si in speech_intervals
            )

        if not has_speech:
            return {
                "class": "SILENT",
                "similarity": -1.0,
                "motion": 0.0,
                "face_coords": None,
                "reason": "No speech in this interval",
            }

        frames = self._sample_frames(video_path, start_s, end_s)
        motion = self._mouth_motion_score(frames)

        if not self._has_models or self.anchor_embedding is None:
            # Fallback: classify by motion only
            label = "A_ROLL" if motion >= MOTION_THRESHOLD else "B_ROLL"
            return {
                "class": label,
                "similarity": -1.0,
                "motion": round(motion, 2),
                "face_coords": None,
                "reason": f"Motion-only: score={motion:.2f}",
            }

        # Deep face identity check
        best_sim = -1.0
        best_coords = None
        for frame in frames:
            h, w = frame.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)
            if faces is None:
                continue
            for face in faces:
                aligned = self.recognizer.alignCrop(frame, face)
                emb = self.recognizer.feature(aligned)
                sim = self._similarity(emb)
                if sim > best_sim:
                    best_sim = sim
                    best_coords = [int(v) for v in face[:4]]

        if best_sim >= AROLL_THRESHOLD and motion >= MOTION_THRESHOLD:
            label = "A_ROLL"
            reason = f"Target speaker confirmed (sim={best_sim:.2f}, motion={motion:.2f})"
        elif best_sim >= AROLL_THRESHOLD:
            # Face present but not talking (posed/listening shot)
            label = "B_ROLL"
            reason = f"Target face visible but not speaking (motion={motion:.2f})"
        elif best_sim >= REJECT_THRESHOLD:
            label = "B_ROLL"
            reason = f"Uncertain face (sim={best_sim:.2f}) — treating as B-roll"
        else:
            label = "B_ROLL"
            reason = f"Non-target face or no face (sim={best_sim:.2f})"

        return {
            "class": label,
            "similarity": round(best_sim, 3),
            "motion": round(motion, 2),
            "face_coords": best_coords,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Full scene classification
    # ------------------------------------------------------------------
    def classify_scenes(self, video_path: Path, scenes: list[dict],
                         words: list[dict] = None) -> list[dict]:
        """
        Classify all scenes. Returns enriched scene list with 'class', 'reason', etc.
        words: transcript words with {start, end} for speech interval detection.
        """
        speech_intervals = None
        if words:
            speech_intervals = [(w["start"], w["end"]) for w in words]

        results = []
        for i, scene in enumerate(scenes):
            start_s = scene["start"]
            end_s = scene["end"]
            result = self.classify_shot(video_path, start_s, end_s, speech_intervals)
            enriched = {**scene, **result}
            results.append(enriched)
            print(f"[Classifier] Shot {i:02d} [{start_s:.2f}s-{end_s:.2f}s] → "
                  f"{result['class']} | {result['reason']}")
        return results


def classify_video(
    video_path: Path,
    job_dir: Path,
    words: list[dict],
    scene_threshold: float = 32.0,
    target_time: float = None,
    target_point: tuple = None,
    reference_image: Path = None,
) -> list[dict]:
    """
    Top-level Stage 2 entry point.
    Detects scenes, classifies each shot, saves scenes.json.
    Returns enriched scene list.
    """
    job_dir = Path(job_dir)
    scenes_path = job_dir / "scenes.json"

    # Detect scenes (cached by scene_analyzer)
    print("[Classifier] Detecting camera cuts...")
    scenes = detect_scenes(video_path, threshold=scene_threshold)
    info = get_video_info(video_path)
    print(f"[Classifier] Found {len(scenes)} shots in {info['duration']:.2f}s video.")

    clf = SceneClassifier()

    # Set identity anchor
    if reference_image and Path(reference_image).exists():
        clf.set_anchor_from_image(Path(reference_image))
    elif clf._has_models:
        clf.set_anchor_from_video(video_path, target_time=target_time,
                                   target_point=target_point)

    # Classify
    classified = clf.classify_scenes(video_path, scenes, words=words)

    # Save
    scenes_path.write_text(
        json.dumps(classified, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    aroll = sum(1 for s in classified if s["class"] == "A_ROLL")
    broll = sum(1 for s in classified if s["class"] == "B_ROLL")
    silent = sum(1 for s in classified if s["class"] == "SILENT")
    print(f"[Classifier] Classified: {aroll} A-Roll | {broll} B-Roll | {silent} Silent")
    print(f"[Classifier] Saved → {scenes_path.name}")
    return classified
