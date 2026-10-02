"""
Face candidates and speaker lock.

Analyze phase: find the distinct faces that appear while someone is speaking and
rank them. Render phase: only shots containing the locked identity are lip-synced,
and Sync Labs receives per-frame bounding boxes of that identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import PipelineError

SAME_PERSON = 0.70          # SFace cosine similarity: same identity at or above this
MIN_FACE_HEIGHT_RATIO = 0.10  # ignore faces smaller than this share of frame height
LOCK_MIN_PRESENCE = 0.40    # share of a shot's frames the locked face must appear in
MAX_CANDIDATES = 4
MIN_CANDIDATE_SECONDS = 0.3


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def cosine(a, b) -> float:
    return float(np.dot(_unit(a), _unit(b)))


@dataclass
class FaceObs:
    shot_index: int
    time_s: float
    box: tuple  # (x, y, w, h)
    frame_h: int
    embedding: Any
    weight_s: float
    crop: Any = None


@dataclass
class FaceCandidate:
    id: str
    embedding: list
    shots: list
    screen_seconds: float
    score: float
    crop_file: str | None = None
    recommended: bool = False
    crop: Any = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict:
        return {"id": self.id, "embedding": list(self.embedding), "shots": self.shots,
                "screen_seconds": self.screen_seconds, "score": self.score,
                "crop_file": self.crop_file, "recommended": self.recommended}

    def public_view(self) -> dict:
        return {"id": self.id, "shots": len(self.shots), "seconds": self.screen_seconds,
                "recommended": self.recommended, "crop": self.crop_file}


def cluster_observations(observations: list, threshold: float = SAME_PERSON) -> list:
    clusters = []
    for observation in observations:
        vector = _unit(observation.embedding)
        best, best_similarity = None, threshold
        for cluster in clusters:
            similarity = float(np.dot(vector, _unit(cluster["sum"])))
            if similarity >= best_similarity:
                best, best_similarity = cluster, similarity
        if best is None:
            clusters.append({"sum": vector.copy(), "members": [observation]})
        else:
            best["sum"] = best["sum"] + vector
            best["members"].append(observation)
    return [cluster["members"] for cluster in clusters]


def rank_candidates(clusters: list, max_candidates: int = MAX_CANDIDATES,
                    min_seconds: float = MIN_CANDIDATE_SECONDS) -> list:
    scored = []
    for members in clusters:
        seconds = sum(m.weight_s for m in members)
        if seconds < min_seconds:
            continue
        score = sum(m.weight_s * (m.box[3] / m.frame_h) for m in members)
        embedding = _unit(np.mean([_unit(m.embedding) for m in members], axis=0))
        sharpest = max(members, key=lambda m: m.box[3])
        scored.append((score, embedding, members, seconds, sharpest.crop))
    scored.sort(key=lambda item: -item[0])
    candidates = []
    for rank, (score, embedding, members, seconds, crop) in enumerate(scored[:max_candidates]):
        candidates.append(FaceCandidate(
            id=str(rank), embedding=embedding.tolist(),
            shots=sorted({m.shot_index for m in members}),
            screen_seconds=round(seconds, 2), score=round(score, 4),
            recommended=(rank == 0), crop=crop,
        ))
    return candidates


@dataclass(frozen=True)
class FaceLock:
    embedding: tuple | None

    @classmethod
    def disabled(cls) -> "FaceLock":
        return cls(None)

    @classmethod
    def from_candidate(cls, candidate: dict) -> "FaceLock":
        return cls(tuple(float(v) for v in candidate["embedding"]))

    @property
    def enabled(self) -> bool:
        return self.embedding is not None

    @property
    def tag(self) -> str:
        if self.embedding is None:
            return "NOFACE"
        digest = hashlib.sha1(json.dumps([round(v, 4) for v in self.embedding]).encode()).hexdigest()
        return "L" + digest[:8]

    def vector(self) -> np.ndarray:
        return _unit(self.embedding)


def select_lock(candidates: list, choice: str) -> FaceLock:
    for candidate in candidates:
        if candidate.id == str(choice):
            return FaceLock.from_candidate(candidate.to_json())
    raise PipelineError(
        f"Face choice '{choice}' is not one of the detected candidates; submit the video again."
    )


def save_candidates(path, candidates: list) -> None:
    Path(path).write_text(
        json.dumps([c.to_json() for c in candidates], ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_candidates(path) -> list:
    return [
        FaceCandidate(id=item["id"], embedding=item["embedding"], shots=item["shots"],
                      screen_seconds=item["screen_seconds"], score=item["score"],
                      crop_file=item.get("crop_file"), recommended=item.get("recommended", False))
        for item in json.loads(Path(path).read_text(encoding="utf-8"))
    ]


class FaceEngine:
    """YuNet detection + SFace embeddings, using the models shipped with Service B."""

    def __init__(self, tracker=None):
        if tracker is None:
            from .face_id_tracker import FaceIDTracker

            tracker = FaceIDTracker()
        self._detector = tracker.detector
        self._recognizer = tracker.recognizer

    def detect(self, frame_bgr) -> list:
        height, width = frame_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame_bgr)
        found = []
        for face in faces if faces is not None else []:
            aligned = self._recognizer.alignCrop(frame_bgr, face)
            embedding = self._recognizer.feature(aligned).reshape(-1)
            x, y, w, h = [int(v) for v in face[:4]]
            found.append(((x, y, w, h), embedding))
        return found


def _speech_overlaps(words: list, start: float, end: float) -> bool:
    return any(float(w["end"]) > start and float(w["start"]) < end for w in words)


def _crop(frame, box, margin: float = 0.25, max_side: int = 256):
    height, width = frame.shape[:2]
    x, y, w, h = box
    pad_x, pad_y = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    scale = max_side / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale)
    return crop.copy()


def discover_candidates(video_path, scenes: list, words: list, job_dir, engine, *,
                        samples_per_shot: int = 6, max_candidates: int = MAX_CANDIDATES) -> list:
    job_dir = Path(job_dir)
    faces_dir = job_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PipelineError(f"Cannot open video for face detection: {video_path}")
    observations = []
    try:
        for index, scene in enumerate(scenes):
            start, end = float(scene["start"]), float(scene["end"])
            if not _speech_overlaps(words, start, end):
                continue
            duration = end - start
            count = max(1, min(samples_per_shot, int(duration * 2)))
            weight = duration / count
            for k in range(count):
                moment = start + duration * (k + 0.5) / count
                capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame_h = frame.shape[0]
                for box, embedding in engine.detect(frame):
                    if box[3] < MIN_FACE_HEIGHT_RATIO * frame_h:
                        continue
                    observations.append(FaceObs(index, moment, box, frame_h, embedding, weight, _crop(frame, box)))
    finally:
        capture.release()
    candidates = rank_candidates(cluster_observations(observations), max_candidates)
    for candidate in candidates:
        if candidate.crop is not None:
            name = f"face_cand_{candidate.id}.jpg"
            if cv2.imwrite(str(faces_dir / name), candidate.crop):
                candidate.crop_file = name
            else:
                print(f"[Faces] WARNING: could not write the crop for candidate {candidate.id}.")
    save_candidates(job_dir / "face_candidates.json", candidates)
    print(f"[Faces] {len(candidates)} face candidate(s) found.")
    return candidates


def load_or_discover(video_path, scenes: list, words: list, job_dir, engine=None) -> list:
    saved = Path(job_dir) / "face_candidates.json"
    if saved.is_file():
        return load_candidates(saved)
    return discover_candidates(video_path, scenes, words, job_dir, engine or FaceEngine())


def locked_boxes_for_clip(clip_path, engine, lock: FaceLock, threshold: float = SAME_PERSON):
    """Per-frame [x1, y1, x2, y2] of the locked identity (None where absent) and the presence ratio."""
    anchor = lock.vector()
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise PipelineError(f"Cannot open clip for face tracking: {clip_path}")
    boxes = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_h, frame_w = frame.shape[:2]
            best, best_similarity = None, threshold
            for box, embedding in engine.detect(frame):
                if box[3] < MIN_FACE_HEIGHT_RATIO * frame_h:
                    continue
                similarity = cosine(anchor, embedding)
                if similarity >= best_similarity:
                    best, best_similarity = box, similarity
            if best is None:
                boxes.append(None)
            else:
                x, y, w, h = best
                x1, y1 = min(max(0, x), frame_w), min(max(0, y), frame_h)
                x2, y2 = min(max(0, x + w), frame_w), min(max(0, y + h), frame_h)
                # Detections can extend past the frame; Sync Labs wants in-frame boxes.
                boxes.append([x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None)
    finally:
        capture.release()
    if not boxes:
        raise PipelineError(f"No frames read from clip: {clip_path}")
    present = sum(box is not None for box in boxes)
    return boxes, (present / len(boxes) if boxes else 0.0)


def build_sync_options(boxes: list) -> dict:
    """Sync Labs active_speaker_detection payload: only the locked face is animated."""
    return {"auto_detect": False, "bounding_boxes": boxes}
