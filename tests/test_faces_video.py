import numpy as np
from helpers import make_video

from src.faces import (
    FaceLock,
    build_sync_options,
    discover_candidates,
    load_or_discover,
    locked_boxes_for_clip,
)


def identity(seed):
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


A, B = identity(1), identity(2)


class ScriptedEngine:
    """Always sees A (big); sees B (small) on every third call."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        faces = [((20, 10, 40, 60), A)]
        if self.calls % 3 == 0:
            faces.append(((90, 20, 30, 40), B))
        return faces


class FixedEngine:
    def __init__(self, vector):
        self.vector = vector

    def detect(self, frame):
        return [((10, 10, 30, 40), self.vector)]


class EmptyEngine:
    def detect(self, frame):
        return []


def scenes_and_words():
    return [{"start": 0.0, "end": 3.0}], [{"word": "hi", "start": 0.2, "end": 2.5}]


def test_discovery_finds_ranks_and_saves_candidates(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3.0, size="160x90")
    scenes, words = scenes_and_words()
    candidates = discover_candidates(video, scenes, words, tmp_path / "job", ScriptedEngine())
    assert len(candidates) == 2
    assert candidates[0].recommended and candidates[0].screen_seconds > candidates[1].screen_seconds
    assert (tmp_path / "job" / "faces" / "face_cand_0.jpg").is_file()
    assert (tmp_path / "job" / "face_candidates.json").is_file()


def test_shots_without_speech_are_ignored(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3.0)
    candidates = discover_candidates(video, [{"start": 0.0, "end": 3.0}], [], tmp_path / "job", ScriptedEngine())
    assert candidates == []


def test_video_with_no_faces_gives_no_candidates(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3.0)
    scenes, words = scenes_and_words()
    assert discover_candidates(video, scenes, words, tmp_path / "job", EmptyEngine()) == []


def test_load_or_discover_reuses_saved_candidates(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3.0)
    scenes, words = scenes_and_words()
    first = load_or_discover(video, scenes, words, tmp_path / "job", engine=ScriptedEngine())
    again = load_or_discover(video, scenes, words, tmp_path / "job", engine=EmptyEngine())
    assert [c.id for c in again] == [c.id for c in first] and len(again) == 2


def test_boxes_follow_the_locked_identity(tmp_path):
    clip = make_video(tmp_path / "clip.mp4", seconds=2.0)
    boxes, presence = locked_boxes_for_clip(clip, FixedEngine(A), FaceLock.from_candidate({"embedding": A.tolist()}))
    assert presence == 1.0 and len(boxes) >= 18
    assert boxes[0] == [10, 10, 40, 50]


def test_other_person_is_not_boxed(tmp_path):
    clip = make_video(tmp_path / "clip.mp4", seconds=2.0)
    boxes, presence = locked_boxes_for_clip(clip, FixedEngine(B), FaceLock.from_candidate({"embedding": A.tolist()}))
    assert presence == 0.0 and all(box is None for box in boxes)


def test_sync_options_payload_shape():
    options = build_sync_options([[1, 2, 3, 4], None])
    assert options == {"auto_detect": False, "bounding_boxes": [[1, 2, 3, 4], None]}
