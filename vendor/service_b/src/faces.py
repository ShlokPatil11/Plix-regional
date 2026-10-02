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
