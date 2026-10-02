import json

from dir_language_pipeline.job_store import JobStore


def make_store(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create("j1", "hi", "auto", tmp_path / "in.mp4", "in.mp4", "combine")
    return store


def test_set_artifact_adds_and_overwrites_without_touching_other_keys(tmp_path):
    store = make_store(tmp_path)
    store.claim()
    store.finish("j1", {"final_video": "/a/final.mp4"})
    store.set_artifact("j1", "music_track", "/a/music_track.mp3")
    assert store.get("j1")["artifacts"] == {"final_video": "/a/final.mp4", "music_track": "/a/music_track.mp3"}
    store.set_artifact("j1", "music_track", "/a/other.mp3")
    assert store.get("j1")["artifacts"]["music_track"] == "/a/other.mp3"
    assert store.get("j1")["artifacts"]["final_video"] == "/a/final.mp4"
    assert store.get("j1")["status"] == "completed"


def test_set_artifact_shows_up_in_the_manifest(tmp_path):
    store = make_store(tmp_path)
    store.claim()
    store.finish("j1", {})
    store.set_artifact("j1", "music_info", "/a/music.json")
    manifest = json.loads((tmp_path / "jobs" / "j1" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["music_info"] == "/api/jobs/j1/artifacts/music_info"


def test_set_artifact_on_an_unknown_job_is_a_no_op(tmp_path):
    store = make_store(tmp_path)
    store.set_artifact("nope", "music_track", "/a/x.mp3")
    assert store.get("nope") is None
