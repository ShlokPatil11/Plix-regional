import importlib
import json
import time

import pytest
from fastapi.testclient import TestClient
from helpers import make_tone, make_video, mean_volume_db


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    from dir_language_pipeline import api as api_module

    module = importlib.reload(api_module)
    return module, TestClient(module.app)  # no lifespan: the worker thread is not started


def new_job(module, tmp_path, job_id="j1", status=None):
    module.store.create(job_id, "hi", "auto", tmp_path / "in.mp4", "in.mp4", "combine")
    art = module.JOB_ROOT / job_id / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    if status == "awaiting":
        module.store.claim()
        module.store.await_face(job_id)
    return art


def publish_faces(art):
    (art / "faces.json").write_text(json.dumps({"candidates": [
        {"id": "0", "shots": 3, "seconds": 5.0, "recommended": True, "crop": "face_cand_0.jpg"},
        {"id": "1", "shots": 1, "seconds": 1.0, "recommended": False, "crop": "face_cand_1.jpg"},
    ]}), encoding="utf-8")
    (art / "face_cand_0.jpg").write_bytes(b"\xff\xd8jpeg0")
    (art / "music.json").write_text(json.dumps({"status": "ok", "title": "Song", "artist": "Artist"}), encoding="utf-8")


def test_faces_endpoint_404_until_published_then_lists_candidates_and_music(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, status="awaiting")
    assert client.get("/api/jobs/j1/faces").status_code == 404
    publish_faces(art)
    body = client.get("/api/jobs/j1/faces").json()
    assert body["status"] == "awaiting_face_selection"
    assert [c["id"] for c in body["candidates"]] == ["0", "1"]
    assert body["music"]["title"] == "Song"


def test_crop_endpoint_serves_crops_and_rejects_path_tricks(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, status="awaiting")
    publish_faces(art)
    ok = client.get("/api/jobs/j1/faces/face_cand_0.jpg")
    assert ok.status_code == 200 and ok.content.startswith(b"\xff\xd8")
    assert client.get("/api/jobs/j1/faces/..%2Fmusic.json").status_code in (404, 422)
    assert client.get("/api/jobs/j1/faces/music.json").status_code == 404


def test_confirm_face_requeues_and_is_idempotent(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, status="awaiting")
    publish_faces(art)
    first = client.post("/api/jobs/j1/face", json={"candidate_id": "1"})
    assert first.status_code == 200 and first.json()["status"] == "queued"
    again = client.post("/api/jobs/j1/face", json={"candidate_id": "1"})
    assert again.status_code == 200
    assert module.store.get("j1")["face_choice"] == "1"


def test_confirm_rejects_unknown_candidate_unknown_job_and_not_waiting(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, status="awaiting")
    publish_faces(art)
    assert client.post("/api/jobs/j1/face", json={"candidate_id": "9"}).status_code == 422
    assert client.post("/api/jobs/nope/face", json={"candidate_id": "0"}).status_code == 404
    new_job(module, tmp_path, job_id="j2")  # queued, not waiting, and no choices published
    assert client.post("/api/jobs/j2/face", json={"candidate_id": "0"}).status_code == 409


def test_confirm_rejects_a_queued_job_even_when_choices_are_published(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, job_id="j2")  # still queued, never paused
    publish_faces(art)
    assert module.store.get("j2")["status"] == "queued"
    assert client.post("/api/jobs/j2/face", json={"candidate_id": "0"}).status_code == 409
    assert module.store.get("j2")["face_choice"] is None


def test_confirm_rejects_a_different_choice_after_confirmation_but_accepts_the_same(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, status="awaiting")
    publish_faces(art)
    first = client.post("/api/jobs/j1/face", json={"candidate_id": "1"})
    assert first.status_code == 200 and first.json()["status"] == "queued"
    assert client.post("/api/jobs/j1/face", json={"candidate_id": "0"}).status_code == 409
    assert client.post("/api/jobs/j1/face", json={"candidate_id": "1"}).status_code == 200
    assert module.store.get("j1")["face_choice"] == "1"


def test_confirm_rejects_an_expired_job(api, tmp_path):
    module, client = api
    art = new_job(module, tmp_path, status="awaiting")
    publish_faces(art)
    time.sleep(0.05)
    assert module.store.expire_stale(max_age_seconds=0) == ["j1"]
    assert module.store.get("j1")["status"] == "failed"
    assert client.post("/api/jobs/j1/face", json={"candidate_id": "0"}).status_code == 409


def completed_job(module, tmp_path, with_music=True):
    art = new_job(module, tmp_path)
    voice = make_video(art / "voice_video.mp4", seconds=2.0)
    final = art / "final_video.mp4"
    final.write_bytes(voice.read_bytes())
    artifacts = {"final_video": str(final), "voice_video": str(voice)}
    if with_music:
        track = make_tone(art / "music_track.mp3", seconds=2.0)
        (art / "music.json").write_text(json.dumps({"status": "ok", "title": "Song", "artist": "Artist"}), encoding="utf-8")
        artifacts["music_track"] = str(track)
    else:
        (art / "music.json").write_text(json.dumps({"status": "not_identified"}), encoding="utf-8")
    module.store.claim()
    module.store.finish("j1", artifacts)
    return art, final


def test_remix_rebuilds_the_final_video_at_the_new_percent(api, tmp_path):
    module, client = api
    art, final = completed_job(module, tmp_path)
    client.post("/api/jobs/j1/remix", json={"music_percent": 0})
    quiet = mean_volume_db(final)
    response = client.post("/api/jobs/j1/remix", json={"music_percent": 100})
    assert response.status_code == 200
    assert response.json() == {"music_percent": 100, "final_video": "/api/jobs/j1/artifacts/final_video"}
    assert mean_volume_db(final) > quiet + 1.0
    assert json.loads((art / "music.json").read_text())["percent"] == 100


def test_remix_rejects_bad_values_missing_music_and_unfinished_jobs(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path, with_music=False)
    assert client.post("/api/jobs/j1/remix", json={"music_percent": 30}).status_code == 409
    assert client.post("/api/jobs/j1/remix", json={"music_percent": 101}).status_code == 422
    assert client.post("/api/jobs/j1/remix", json={"music_percent": -1}).status_code == 422
    assert client.post("/api/jobs/nope/remix", json={"music_percent": 30}).status_code == 404
    new_job(module, tmp_path, job_id="j2")
    assert client.post("/api/jobs/j2/remix", json={"music_percent": 30}).status_code == 409


def test_status_includes_music_info(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path)
    body = client.get("/api/jobs/j1").json()
    assert body["music"]["title"] == "Song"
