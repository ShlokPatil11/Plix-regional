import importlib
import json

import pytest
from fastapi.testclient import TestClient
from helpers import make_tone, make_video, mean_volume_db, stream_types

from dir_language_pipeline.music import FetchedTrack, MusicFetchError


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_DATA_DIR", str(tmp_path / "data"))
    from dir_language_pipeline import api as api_module

    module = importlib.reload(api_module)
    return module, TestClient(module.app)


SRT = "1\n00:00:00,000 --> 00:00:01,000\nhello\n"


def completed_job(module, tmp_path, music="ok", subtitles=False, voice=True, operation="combine"):
    module.store.create("j1", "hi", "auto", tmp_path / "in.mp4", "in.mp4", operation)
    art = module.JOB_ROOT / "j1" / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    if voice:
        voice_path = make_video(art / "voice_video.mp4", seconds=3.0)
        final = art / "final_video.mp4"
        final.write_bytes(voice_path.read_bytes())
        artifacts = {"final_video": str(final), "voice_video": str(voice_path)}
    if music == "ok":
        artifacts["music_track"] = str(make_tone(art / "music_track.mp3", seconds=3.0, freq=440))
        info = {"status": "ok", "title": "Detected", "artist": "Someone", "percent": 25}
    else:
        info = {"status": "not_identified"}
    (art / "music.json").write_text(json.dumps(info), encoding="utf-8")
    artifacts["music_info"] = str(art / "music.json")
    if subtitles:
        (art / "visual_subtitles.srt").write_text(SRT, encoding="utf-8")
    module.store.claim()
    module.store.finish("j1", artifacts)
    return art


def song(tmp_path, name="my song.mp3", seconds=6.0, freq=900):
    path = make_tone(tmp_path / name, seconds=seconds, freq=freq)
    return path.name, path.read_bytes()


def post_file(client, name, data, percent=None, job="j1"):
    form = {} if percent is None else {"music_percent": str(percent)}
    return client.post(f"/api/jobs/{job}/music", files={"file": (name, data, "application/octet-stream")}, data=form)


class FakeFetcher:
    def __init__(self, tmp_path=None, error=None):
        self.calls, self.error = [], error

    def fetch_url(self, url, dest_dir):
        self.calls.append(url)
        if self.error:
            raise self.error
        path = make_tone(dest_dir / "custom_source.mp3", seconds=5.0, freq=700)
        return FetchedTrack(path=path, url=url, title="Cool Song")


def test_uploaded_song_replaces_the_music_and_is_mixed_in(api, tmp_path):
    module, client = api
    art = completed_job(module, tmp_path)
    final = module.JOB_ROOT / "j1" / "artifacts" / "final_video.mp4"
    voice_only = mean_volume_db(art / "voice_video.mp4")
    name, data = song(tmp_path)

    response = post_file(client, name, data, percent=100)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["music_percent"] == 100
    assert body["final_video"] == "/api/jobs/j1/artifacts/final_video"
    assert body["music"]["title"] == "my song.mp3" and body["music"]["artist"] == "Custom"
    assert body["music"]["source"] == "custom" and body["music"]["status"] == "ok"
    assert mean_volume_db(final) > voice_only + 1.0
    saved = json.loads((art / "music.json").read_text(encoding="utf-8"))
    assert saved["source"] == "custom" and saved["percent"] == 100
    assert module.store.get("j1")["artifacts"]["music_track"] == str(art / "music_track.mp3")
    assert client.get("/api/jobs/j1").json()["music"]["title"] == "my song.mp3"


def test_custom_music_works_when_nothing_was_detected_and_remix_then_works(api, tmp_path):
    module, client = api
    art = completed_job(module, tmp_path, music="none")
    assert "music_track" not in module.store.get("j1")["artifacts"]
    name, data = song(tmp_path)

    assert post_file(client, name, data, percent=60).status_code == 200

    assert (art / "music_track.mp3").is_file()
    assert module.store.get("j1")["artifacts"]["music_track"] == str(art / "music_track.mp3")
    again = client.post("/api/jobs/j1/remix", json={"music_percent": 10})
    assert again.status_code == 200 and again.json()["music_percent"] == 10


def test_existing_masking_job_uses_final_video_when_voice_artifact_is_missing(api, tmp_path):
    module, client = api
    art = completed_job(module, tmp_path, music="none", operation="masking")
    (art / "voice_video.mp4").unlink()
    name, data = song(tmp_path)

    response = post_file(client, name, data, percent=35)

    assert response.status_code == 200, response.text
    assert response.json()["music_percent"] == 35


def test_default_percent_keeps_the_previous_slider_value_or_25(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path)
    name, data = song(tmp_path)
    assert post_file(client, name, data).json()["music_percent"] == 25  # music.json had percent 25


def test_subtitle_track_survives_the_swap(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path, subtitles=True)
    name, data = song(tmp_path)
    assert post_file(client, name, data, percent=50).status_code == 200
    final = module.JOB_ROOT / "j1" / "artifacts" / "final_video.mp4"
    assert "subtitle" in stream_types(final)


def test_youtube_link_is_fetched_and_mixed(api, tmp_path):
    module, client = api
    art = completed_job(module, tmp_path)
    fetcher = FakeFetcher()
    module.make_youtube_fetcher = lambda: fetcher

    response = client.post("/api/jobs/j1/music", data={"url": "https://youtu.be/abc123", "music_percent": "40"})

    assert response.status_code == 200, response.text
    assert fetcher.calls == ["https://youtu.be/abc123"]
    music = response.json()["music"]
    assert music["title"] == "Cool Song" and music["source_url"] == "https://youtu.be/abc123"
    assert json.loads((art / "music.json").read_text())["source"] == "custom"


def test_non_youtube_link_is_rejected_before_any_download(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path)
    response = client.post("/api/jobs/j1/music", data={"url": "https://example.com/some.mp3"})
    assert response.status_code == 422
    assert "youtube" in response.json()["detail"].lower()


def test_fetch_errors_become_422_with_the_message(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path)
    module.make_youtube_fetcher = lambda: FakeFetcher(error=MusicFetchError("The video must be 15 minutes or shorter."))
    response = client.post("/api/jobs/j1/music", data={"url": "https://youtu.be/abc123"})
    assert response.status_code == 422 and "15 minutes" in response.json()["detail"]


def test_must_send_exactly_one_of_file_or_link(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path)
    name, data = song(tmp_path)
    assert client.post("/api/jobs/j1/music").status_code == 422
    both = client.post("/api/jobs/j1/music", files={"file": (name, data)}, data={"url": "https://youtu.be/abc123"})
    assert both.status_code == 422


def test_unsupported_extension_is_rejected(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path)
    assert post_file(client, "notes.txt", b"hello").status_code == 415
    assert post_file(client, "noextension", b"hello").status_code == 415


def test_renamed_non_audio_file_is_rejected_and_nothing_changes(api, tmp_path):
    module, client = api
    art = completed_job(module, tmp_path)
    final = art / "final_video.mp4"
    before = (final.read_bytes(), (art / "music.json").read_text(), (art / "music_track.mp3").read_bytes())

    response = post_file(client, "fake.mp3", b"definitely not audio " * 100)

    assert response.status_code == 422
    assert "audio" in response.json()["detail"].lower()
    assert (final.read_bytes(), (art / "music.json").read_text(), (art / "music_track.mp3").read_bytes()) == before


def test_oversized_upload_is_rejected(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path)
    module.MAX_MUSIC_UPLOAD_BYTES = 1000
    response = post_file(client, "big.mp3", b"x" * 5000)
    assert response.status_code == 413


def test_percent_must_be_between_0_and_100(api, tmp_path):
    module, client = api
    completed_job(module, tmp_path)
    name, data = song(tmp_path)
    assert post_file(client, name, data, percent=101).status_code == 422
    assert post_file(client, name, data, percent=-1).status_code == 422


def test_unknown_unfinished_and_voiceless_jobs(api, tmp_path):
    module, client = api
    name, data = song(tmp_path)
    assert post_file(client, name, data, job="nope").status_code == 404
    completed_job(module, tmp_path, voice=False)
    assert post_file(client, name, data, job="j1").status_code == 409  # completed, but no voice-only video
    module.store.create("q1", "hi", "auto", tmp_path / "in.mp4", "in.mp4", "combine")
    assert post_file(client, name, data, job="q1").status_code == 409  # still queued


def test_no_scratch_files_are_left_behind(api, tmp_path):
    module, client = api
    art = completed_job(module, tmp_path)
    name, data = song(tmp_path)
    post_file(client, name, data)
    post_file(client, "fake.mp3", b"nope" * 100)
    assert not [p for p in art.iterdir() if p.name.startswith(".")]


def test_uploaded_file_name_cannot_escape_the_job_folder(api, tmp_path):
    module, client = api
    art = completed_job(module, tmp_path)
    name, data = song(tmp_path)
    response = post_file(client, "../../evil.mp3", data)
    assert response.status_code == 200
    assert response.json()["music"]["title"] == "evil.mp3"
    assert not (module.JOB_ROOT / "evil.mp3").exists() and not (tmp_path / "evil.mp3").exists()
    assert (art / "music_track.mp3").is_file()
