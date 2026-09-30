"""
Deep Face Identification & Character Locking Module
Uses OpenCV YuNet (Face Detection) and SFace (128-D Feature Embeddings)
to lock onto the intended speaker with high cosine similarity (>=0.70)
and reject other persons/models (<0.40).
"""
import os
import cv2
import numpy as np
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
YUNET_PATH = SERVICE_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = SERVICE_ROOT / "models" / "face_recognition_sface_2021dec.onnx"


class FaceIDTracker:
    def __init__(self, yunet_path=None, sface_path=None, conf_threshold=0.7, score_threshold=0.70):
        y_path = str(yunet_path or YUNET_PATH)
        s_path = str(sface_path or SFACE_PATH)
        if not os.path.exists(y_path) or not os.path.exists(s_path):
            raise FileNotFoundError(f"YuNet or SFace model not found at {y_path} / {s_path}")

        # Initialize detector with default resolution, resized dynamically per frame
        self.detector = cv2.FaceDetectorYN.create(y_path, "", (640, 640), conf_threshold, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(s_path, "")
        self.score_threshold = score_threshold
        self.anchor_embedding = None
        self.anchor_box = None

    def set_anchor_from_frame(self, frame_bgr, target_box=None):
        """
        Extracts and stores the 128-D facial embedding from a reference frame.
        If target_box is provided (x, y, w, h), locks to the closest face in that box.
        Otherwise selects the largest/most centered face.
        """
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            raise ValueError("No face detected in reference frame.")

        selected_face = None
        if target_box:
            tx, ty, tw, th = target_box
            tcx, tcy = tx + tw / 2, ty + th / 2
            best_dist = float("inf")
            for f in faces:
                fx, fy, fw, fh = f[:4]
                fcx, fcy = fx + fw / 2, fy + fh / 2
                dist = (tcx - fcx) ** 2 + (tcy - fcy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    selected_face = f
        else:
            # Pick the largest face by area
            selected_face = max(faces, key=lambda f: f[2] * f[3])

        aligned = self.recognizer.alignCrop(frame_bgr, selected_face)
        self.anchor_embedding = self.recognizer.feature(aligned)
        self.anchor_box = [int(v) for v in selected_face[:4]]
        return self.anchor_embedding, self.anchor_box

    def set_anchor_from_point(self, frame_bgr, x, y):
        """Choose only the face containing the user's point; never guess a bystander."""
        height, width = frame_bgr.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame_bgr)
        for face in faces if faces is not None else []:
            fx, fy, fw, fh = face[:4]
            if fx <= x <= fx + fw and fy <= y <= fy + fh:
                return self.set_anchor_from_frame(frame_bgr, tuple(face[:4]))
        raise ValueError("The selected point is not inside a detected face.")

    def match_interval(self, video_path, start_sec, end_sec):
        """Return presence and a stable target-face point for a short shot."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        first = round(start_sec * fps)
        last = round(end_sec * fps)
        step = max(1, (last - first) // 8)
        sampled = matched = 0
        best = None
        for index in range(first, last, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            sampled += 1
            matches = self.detect_and_match(frame)
            target = [m for m in matches if m['is_target'] and m['box'][3] >= .15 * frame.shape[0]]
            if target:
                matched += 1
                candidate = max(target, key=lambda m: m['similarity'])
                if best is None or candidate['similarity'] > best['similarity']:
                    x, y, w, h = candidate['box']
                    best = {'similarity': candidate['similarity'],
                            'frame_number': index - first,
                            'coordinates': [int(x + w / 2), int(y + h / 2)]}
        cap.release()
        ratio = matched / sampled if sampled else 0
        return {'is_target': ratio >= .4, 'target_ratio': ratio, 'best': best}

    def detect_and_match(self, frame_bgr):
        """
        Detects all faces in frame and compares each against the anchor embedding.
        Returns list of dicts:
        [{'box': (x,y,w,h), 'conf': float, 'similarity': float, 'is_target': bool}, ...]
        """
        if self.anchor_embedding is None:
            raise RuntimeError("Anchor embedding not set. Call set_anchor_from_frame first.")

        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            box = tuple(int(v) for v in face[:4])
            conf = float(face[-1])
            aligned = self.recognizer.alignCrop(frame_bgr, face)
            feat = self.recognizer.feature(aligned)
            sim = float(self.recognizer.match(self.anchor_embedding, feat, cv2.FaceRecognizerSF_FR_COSINE))
            is_target = sim >= self.score_threshold
            results.append({
                "box": box,
                "conf": conf,
                "similarity": sim,
                "is_target": is_target
            })
        return results

    def is_target_present_in_scene(self, video_path, start_sec, end_sec, min_target_ratio=0.40):
        """
        Evaluates a video scene to determine whether the target speaker is visibly present on full-screen.
        Returns (is_target_present, average_similarity, max_similarity, target_ratio).
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_f = int(round(start_sec * fps))
        end_f = int(round(end_sec * fps))
        dur_f = max(1, end_f - start_f)
        stride = max(1, dur_f // 6)

        target_matches = 0
        total_samples = 0
        sims = []

        for f_idx in range(start_f, end_f, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            total_samples += 1
            matches = self.detect_and_match(frame)
            # Filter for target character only when face is reasonably sized (full screen, not tiny inset)
            h_f = frame.shape[0]
            target_faces = [m for m in matches if m["is_target"] and m["box"][3] > (0.15 * h_f)]
            if target_faces:
                target_matches += 1
                sims.append(max(m["similarity"] for m in target_faces))

        cap.release()
        if total_samples == 0:
            return False, 0.0, 0.0, 0.0

        target_ratio = target_matches / total_samples
        avg_sim = float(np.mean(sims)) if sims else 0.0
        max_sim = float(np.max(sims)) if sims else 0.0
        is_target = target_ratio >= min_target_ratio and avg_sim >= self.score_threshold
        return is_target, avg_sim, max_sim, target_ratio
