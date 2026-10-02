import numpy as np
import pytest
from helpers import make_video

from src.common import PipelineError
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


# Test (a): Two faces in the same frame
class TwoFacesEngine:
    """Always sees A (box (10,10,30,40)) and B (box (80,10,30,40)) in same frame."""

    def detect(self, frame):
        return [((10, 10, 30, 40), A), ((80, 10, 30, 40), B)]


def test_two_faces_in_same_frame_boxes_follow_locked_identity(tmp_path):
    clip = make_video(tmp_path / "clip.mp4", seconds=2.0)
    boxes, presence = locked_boxes_for_clip(clip, TwoFacesEngine(), FaceLock.from_candidate({"embedding": A.tolist()}))
    assert presence == 1.0 and len(boxes) >= 18
    # All boxes should follow identity A
    assert all(box == [10, 10, 40, 50] for box in boxes), f"Expected all boxes [10, 10, 40, 50], got {boxes}"


# Test (b): Face smaller than 10% frame height
class TinyFaceEngine:
    """Face with height 5 in a 90px frame (less than 10% threshold)."""

    def __init__(self, vector):
        self.vector = vector

    def detect(self, frame):
        # Box (10, 10, 30, 5) - height 5, frame height is 90 from make_video
        return [((10, 10, 30, 5), self.vector)]


def test_faces_below_min_height_not_boxed(tmp_path):
    clip = make_video(tmp_path / "clip.mp4", seconds=2.0, size="160x90")  # frame height = 90
    boxes, presence = locked_boxes_for_clip(clip, TinyFaceEngine(A), FaceLock.from_candidate({"embedding": A.tolist()}))
    assert presence == 0.0 and all(box is None for box in boxes)


# Test (c): Discovery ignores faces under minimum height
class SmallFaceOnlyEngine:
    """Only detects small faces below threshold."""

    def detect(self, frame):
        # Box height 5 (less than 10% of 90)
        return [((10, 10, 30, 5), identity(99))]


def test_discovery_ignores_faces_below_min_height(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3.0, size="160x90")
    scenes, words = scenes_and_words()
    candidates = discover_candidates(video, scenes, words, tmp_path / "job", SmallFaceOnlyEngine())
    assert candidates == []


# Test (d): max_candidates caps discovery
class ThreeIdentitiesEngine:
    """Yields 3 distinct identities, all big enough."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        faces = []
        if self.calls % 3 == 1:
            faces.append(((20, 10, 40, 60), identity(10)))
        elif self.calls % 3 == 2:
            faces.append(((20, 10, 40, 60), identity(11)))
        else:
            faces.append(((20, 10, 40, 60), identity(12)))
        return faces


def test_max_candidates_caps_discovery(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=3.0)
    scenes, words = scenes_and_words()
    candidates = discover_candidates(video, scenes, words, tmp_path / "job", ThreeIdentitiesEngine(), max_candidates=2)
    assert len(candidates) == 2


# Test (e): locked_boxes_for_clip on missing file raises PipelineError
def test_locked_boxes_for_clip_missing_file_raises_error(tmp_path):
    missing_clip = tmp_path / "nonexistent.mp4"
    with pytest.raises(PipelineError, match="Cannot open clip for face tracking"):
        locked_boxes_for_clip(missing_clip, FixedEngine(A), FaceLock.from_candidate({"embedding": A.tolist()}))


# Test (f): FaceEngine with fake tracker object
class FakeDetector:
    def setInputSize(self, size):
        pass

    def detect(self, frame):
        # Return (None, array of faces) format like real YuNet
        # Each face is [x, y, w, h, ...]
        return (None, np.array([[10, 20, 30, 40, 0.99]]))


class FakeRecognizer:
    def alignCrop(self, frame, face):
        # Return aligned face crop
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def feature(self, aligned):
        # Return embedding as (1, 128) array
        return np.ones((1, 128), dtype=np.float32)


class FakeTracker:
    def __init__(self):
        self.detector = FakeDetector()
        self.recognizer = FakeRecognizer()


def test_face_engine_with_fake_tracker():
    from src.faces import FaceEngine

    tracker = FakeTracker()
    engine = FaceEngine(tracker=tracker)
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    detections = engine.detect(frame)
    assert len(detections) == 1
    box, embedding = detections[0]
    assert box == (10, 20, 30, 40)
    assert embedding.shape == (128,)
    assert np.all(embedding == 1.0)


def test_face_engine_with_fake_tracker_no_faces():
    from src.faces import FaceEngine

    class FakeDetectorNoFaces:
        def setInputSize(self, size):
            pass

        def detect(self, frame):
            return (None, None)

    class FakeTrackerNoFaces:
        def __init__(self):
            self.detector = FakeDetectorNoFaces()
            self.recognizer = None

    tracker = FakeTrackerNoFaces()
    engine = FaceEngine(tracker=tracker)
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    detections = engine.detect(frame)
    assert detections == []
