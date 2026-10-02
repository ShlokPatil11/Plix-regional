import json

import numpy as np
import pytest
from helpers import make_tone, make_video

from dir_language_pipeline import service_adapters
from dir_language_pipeline.contracts import AwaitingFaceSelection
from dir_language_pipeline.media import FileArtifactBackend
from dir_language_pipeline.music import MusicResult
from dir_language_pipeline.service_adapters import TextToAudioAdapter
from src import analysis, engine, faces
from src.common import PipelineError
from src.faces import FaceCandidate


def candidate(identity_seed, cid="0"):
    vec = np.random.default_rng(identity_seed).normal(size=128).astype(np.float32)
    return FaceCandidate(id=cid, embedding=vec.tolist(), shots=[0, 1], screen_seconds=4.0,
                         score=2.0, crop_file=f"face_cand_{cid}.jpg", recommended=(cid == "0"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    backend = FileArtifactBackend(tmp_path / "artifacts")
    source = backend.upload(make_video(tmp_path / "in.mp4", seconds=3.0), filename="in.mp4", content_type="video/mp4")
    job_dir = tmp_path / "data" / "cache" / "service-b" / "job1" / "job"
    (job_dir / "faces").mkdir(parents=True)
    for crop in ("face_cand_0.jpg", "face_cand_1.jpg"):
        (job_dir / "faces" / crop).write_bytes(b"\xff\xd8jpeg")
    music_track = make_tone(tmp_path / "music_track.mp3", seconds=3.0)

    monkeypatch.setattr(analysis, "analyze_video", lambda *a, **k: {"words": [], "scenes": [], "duration": 3.0})
    monkeypatch.setattr(
        service_adapters, "identify_and_fetch",
        lambda *a, **k: MusicResult("ok", {"status": "ok", "title": "Song", "artist": "Artist"}, music_track),
    )
    seen = {}

    def fake_engine(**kwargs):
        seen["lock"] = kwargs.get("face_lock")
        out = make_video(tmp_path / "dubbed.mp4", seconds=3.0)
        return out

    monkeypatch.setattr(engine, "run_dubbing_engine", fake_engine)
    return backend, source, seen, monkeypatch


def run(adapter, source):
    return adapter.run(source, target_language="hi", source_language="auto", job_id="job1")


def test_candidates_without_a_choice_pause_and_publish_the_panel_data(env):
    backend, source, seen, monkeypatch = env
    monkeypatch.setattr(faces, "load_or_discover", lambda *a, **k: [candidate(1, "0"), candidate(2, "1")])
    with pytest.raises(AwaitingFaceSelection):
        run(TextToAudioAdapter(backend), source)
    published = json.loads((backend.root / "faces.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in published["candidates"]] == ["0", "1"]
    assert "embedding" not in published["candidates"][0]
    assert (backend.root / "face_cand_0.jpg").is_file()
    assert json.loads((backend.root / "music.json").read_text())["status"] == "ok"
    assert "lock" not in seen  # nothing was rendered


def test_a_choice_renders_with_that_face_locked_and_returns_music(env):
    backend, source, seen, monkeypatch = env
    monkeypatch.setattr(faces, "load_or_discover", lambda *a, **k: [candidate(1, "0"), candidate(2, "1")])
    result = run(TextToAudioAdapter(backend, face_choice="1"), source)
    assert seen["lock"].enabled
    assert seen["lock"].tag == faces.select_lock([candidate(1, "0"), candidate(2, "1")], "1").tag
    assert result["music_track"].filename == "music_track.mp3"
    assert "video" in result and "music_info" in result


def test_no_face_found_skips_the_pause_and_disables_lip_sync(env):
    backend, source, seen, monkeypatch = env
    monkeypatch.setattr(faces, "load_or_discover", lambda *a, **k: [])
    result = run(TextToAudioAdapter(backend), source)
    assert seen["lock"].enabled is False
    assert "video" in result


def test_stale_choice_fails_with_a_clear_message(env):
    backend, source, seen, monkeypatch = env
    monkeypatch.setattr(faces, "load_or_discover", lambda *a, **k: [candidate(1, "0")])
    with pytest.raises(PipelineError, match="not one of the detected candidates"):
        run(TextToAudioAdapter(backend, face_choice="9"), source)


def test_music_failure_does_not_block_rendering(env):
    backend, source, seen, monkeypatch = env
    monkeypatch.setattr(faces, "load_or_discover", lambda *a, **k: [])
    monkeypatch.setattr(
        service_adapters, "identify_and_fetch",
        lambda *a, **k: MusicResult("not_identified", {"status": "not_identified"}),
    )
    result = run(TextToAudioAdapter(backend), source)
    assert "music_track" not in result
    assert result["music_info"].filename == "music.json"
