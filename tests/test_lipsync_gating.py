import shutil

import numpy as np
import pytest
from helpers import make_tone, make_video

from src import lipsyncer
from src.common import PipelineError
from src.faces import FaceLock


def identity(seed):
    return np.random.default_rng(seed).normal(size=128).astype(np.float32)


A, B = identity(1), identity(2)


class FakeSync:
    instances = []

    def __init__(self, default_model=None):
        self.uploads, self.generate_calls = [], []
        FakeSync.instances.append(self)

    def upload_file(self, path, content_type=None):
        self.uploads.append(path)
        return f"https://sync.test/{path.name}"

    def generate(self, video_url, audio_url, model=None, options=None):
        self.generate_calls.append(options)
        return {"id": "gen1"}

    def wait_for_completion(self, generation_id, poll_interval=8):
        return {"outputUrl": "https://sync.test/out"}

    def download_result(self, url, output_path):
        shutil.copyfile(self.uploads[0], output_path)


class FaceEngineStub:
    def __init__(self, vector=None):
        self.vector = vector

    def detect(self, frame):
        return [] if self.vector is None else [((10, 10, 30, 40), self.vector)]


@pytest.fixture
def shot(tmp_path, monkeypatch):
    FakeSync.instances = []
    monkeypatch.setattr(lipsyncer, "SyncLabsClient", FakeSync)
    video = make_video(tmp_path / "src.mp4", seconds=2.0, fps=10)
    audio = make_tone(tmp_path / "dub.wav", seconds=2.0)
    scenes = [{"start": 0.0, "end": 2.0, "class": "A_ROLL"}]
    return video, audio, scenes, tmp_path / "job"


def run(shot, **kwargs):
    video, audio, scenes, job = shot
    clips = lipsyncer.process_shots(video, audio, scenes, job, **kwargs)
    return clips, scenes


def test_without_a_lock_behavior_is_unchanged_auto_detect(shot):
    clips, scenes = run(shot)
    options = FakeSync.instances[0].generate_calls[0]
    assert options["active_speaker_detection"] == {"auto_detect": True}
    assert scenes[0]["lip_sync"] == "synced"


def test_locked_face_present_syncs_with_bounding_boxes(shot):
    lock = FaceLock.from_candidate({"embedding": A.tolist()})
    clips, scenes = run(shot, lock=lock, face_engine=FaceEngineStub(A))
    detection = FakeSync.instances[0].generate_calls[0]["active_speaker_detection"]
    assert detection["auto_detect"] is False
    assert len(detection["bounding_boxes"]) >= 18
    assert detection["bounding_boxes"][0] == [10, 10, 40, 50]
    assert scenes[0]["lip_sync"] == "synced" and scenes[0]["locked_face_ratio"] == 1.0
    assert lock.tag in clips[0].name


def test_other_person_in_shot_is_not_lip_synced(shot):
    lock = FaceLock.from_candidate({"embedding": A.tolist()})
    clips, scenes = run(shot, lock=lock, face_engine=FaceEngineStub(B))
    assert FakeSync.instances[0].generate_calls == []
    assert scenes[0]["lip_sync"] == "skipped_not_target"
    assert clips[0].is_file()


def test_disabled_lock_means_audio_only_no_sync_call(shot):
    clips, scenes = run(shot, lock=FaceLock.disabled())
    assert FakeSync.instances[0].generate_calls == []
    assert scenes[0]["lip_sync"] == "skipped_no_face_lock"
    assert clips[0].is_file()


def test_cached_clip_is_not_reused_for_a_different_face(shot):
    lock_a = FaceLock.from_candidate({"embedding": A.tolist()})
    lock_b = FaceLock.from_candidate({"embedding": B.tolist()})
    first, _ = run(shot, lock=lock_a, face_engine=FaceEngineStub(A))
    second, scenes = run(shot, lock=lock_b, face_engine=FaceEngineStub(B))
    assert first[0] != second[0]
    assert scenes[0]["lip_sync"] == "synced"
    again, scenes = run(shot, lock=lock_a, face_engine=FaceEngineStub(A))
    assert again[0] == first[0] and scenes[0]["lip_sync"] == "cached"


def test_unreadable_clip_for_face_check_falls_back_to_audio_only(shot, monkeypatch):
    def boom(*args, **kwargs):
        raise PipelineError("boom")

    monkeypatch.setattr(lipsyncer, "locked_boxes_for_clip", boom)
    lock = FaceLock.from_candidate({"embedding": A.tolist()})
    clips, scenes = run(shot, lock=lock, face_engine=FaceEngineStub(A))
    assert FakeSync.instances[0].generate_calls == []
    assert scenes[0]["lip_sync"] == "skipped_face_check_failed"
    assert clips[0].is_file()
    assert "facecheckfail" in clips[0].name
