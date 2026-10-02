import inspect
import json

from helpers import make_video

from src import analysis, el_dubber, engine


class FakeResponse:
    ok = True
    headers = {}

    def json(self):
        return {"dubbing_id": "d1"}


def test_dubbing_request_drops_background_audio(tmp_path, monkeypatch):
    sent = {}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        sent["data"] = data
        return FakeResponse()

    monkeypatch.setattr(el_dubber.requests, "post", fake_post)
    video = make_video(tmp_path / "v.mp4", seconds=1.0)
    el_dubber.ElevenLabsDubber("key").create_job(video, "hi")
    assert sent["data"]["drop_background_audio"] == "true"
    assert sent["data"]["watermark"] == "false"


def test_analysis_returns_words_scenes_and_reuses_scene_cache(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4", seconds=2.0)
    job = tmp_path / "job"
    calls = {"classify": 0}
    words = [{"word": "hi", "start": 0.1, "end": 0.5}]

    def fake_transcript(video_path, job_dir, source_lang, supplied_srt):
        return job / "t.srt", words

    def fake_classify(video_path, job_dir, words, scene_threshold):
        calls["classify"] += 1
        scenes = [{"start": 0.0, "end": 2.0, "class": "A_ROLL"}]
        (job_dir / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")
        return scenes

    monkeypatch.setattr(analysis, "extract_transcript", fake_transcript)
    monkeypatch.setattr(analysis, "classify_video", fake_classify)
    first = analysis.analyze_video(video, job)
    second = analysis.analyze_video(video, job)
    assert first["words"] == words and first["scenes"][0]["class"] == "A_ROLL"
    assert 1.8 < first["duration"] < 2.3
    assert second["scenes"] == first["scenes"]
    assert calls["classify"] == 1


def test_engine_accepts_a_face_lock():
    assert "face_lock" in inspect.signature(engine.run_dubbing_engine).parameters
