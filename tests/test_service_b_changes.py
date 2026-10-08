import inspect
import json
import sys
import types
from pathlib import Path

from helpers import make_tone, make_video, mean_volume_db, media_duration

from src import analysis, el_dubber, engine, lipsyncer, transcriber
from dir_language_pipeline import clean_translate
from src.common import PipelineError
from src.srt_parser import Cue


class FakeResponse:
    ok = True
    headers = {}

    def json(self):
        return {"dubbing_id": "d1"}


def test_magnific_media_url_supports_nested_creation_shapes():
    assert clean_translate._creation_media_url({"creation": {"url": "https://cdn.test/a.mp4"}}) == "https://cdn.test/a.mp4"
    assert clean_translate._creation_media_url({
        "creations": [{"outputs": [{"videoUrl": "https://cdn.test/b.mp4"}]}]
    }) == "https://cdn.test/b.mp4"
    assert clean_translate._creation_media_url({"creation": {"status": "completed"}}) is None


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


def test_google_translation_rejects_unchanged_non_telugu_track(monkeypatch):
    class UnchangedResponse:
        ok = True
        status_code = 200

        def json(self):
            return [[["original Hindi text", "original Hindi text"]]]

    monkeypatch.setattr(el_dubber.requests, "get", lambda *args, **kwargs: UnchangedResponse())
    cue = Cue(1, 0, 1000, "original Hindi text")
    try:
        el_dubber._translate_cues_google([cue], "te")
    except PipelineError as exc:
        assert "expected Telugu" in str(exc)
    else:
        raise AssertionError("unchanged source text must not be accepted as Telugu")


def test_target_script_validation_accepts_telugu_and_rejects_urdu():
    el_dubber._validate_target_script([Cue(1, 0, 1000, "ఇది తెలుగు వాక్యం")], "te")
    try:
        el_dubber._validate_target_script([Cue(1, 0, 1000, "یہ اردو متن ہے")], "te")
    except PipelineError as exc:
        assert "expected Telugu" in str(exc)
    else:
        raise AssertionError("Urdu text must not pass Telugu validation")


def test_contextual_translation_repairs_noisy_change_phrase(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["prompt"] = contents
            return types.SimpleNamespace(text=json.dumps([
                {"id": 1, "text": "त्यांनी त्यांच्या दिनचर्येत फक्त एक बदल केला"},
                {"id": 2, "text": "प्लिक्स एसीव्ही समाविष्ट केले"},
            ], ensure_ascii=False))

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeClient
    genai_module.types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
    )
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_module.types)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    cues = [
        Cue(1, 0, 1000, "बस एक چینچ किया"),
        Cue(2, 1000, 2000, "PLIX ACV add किया"),
    ]
    translated = el_dubber._translate_cues_gemini(cues, "mr", "auto")

    assert [cue.text for cue in translated] == [
        "त्यांनी त्यांच्या दिनचर्येत फक्त एक बदल केला",
        "प्लिक्स एसीव्ही समाविष्ट केले",
    ]
    assert "चिमटा" not in " ".join(cue.text for cue in translated)
    assert "PLIX ACV add" in captured["prompt"]


def test_transcript_groups_long_unpunctuated_speech_into_readable_cues():
    words = [
        {"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4}
        for i in range(24)
    ]
    srt = transcriber._words_to_srt(words)
    assert srt.count(" --> ") == 3


def test_tts_assembly_time_fits_cues_without_overlap(tmp_path):
    first = make_tone(tmp_path / "first.mp3", seconds=2.0, freq=440)
    second = make_tone(tmp_path / "second.mp3", seconds=2.0, freq=880)
    output = tmp_path / "assembled.mp3"
    cues = [
        Cue(1, 0, 1000, "one"),
        Cue(2, 1000, 2000, "two"),
    ]

    el_dubber._assemble_tts_audio(cues, [first, second], 2.5, output)

    assert 2.45 <= media_duration(output) <= 2.55
    first_db = mean_volume_db(output, start=0.1, duration=0.8)
    second_db = mean_volume_db(output, start=1.1, duration=0.8)
    assert abs(first_db - second_db) < 1.0


def test_engine_passes_the_face_lock_to_process_shots_and_saves_the_decisions(tmp_path, monkeypatch):
    from src.faces import FaceLock

    video = make_video(tmp_path / "v.mp4", seconds=2.0)
    job = tmp_path / "job"
    lock = FaceLock.disabled()
    seen = {}

    monkeypatch.setattr(engine, "extract_transcript",
                        lambda **kwargs: (job / "t.srt", [{"word": "hi", "start": 0.1, "end": 0.5}]))
    monkeypatch.setattr(
        engine, "classify_video",
        lambda **kwargs: [{"start": 0.0, "end": 2.0, "class": "A_ROLL"}],
    )
    monkeypatch.setattr(
        engine, "run_dubbing",
        lambda **kwargs: (job / "dub.mp3", job / "dub.srt", None),
    )

    def fake_process_shots(**kwargs):
        seen["lock"] = kwargs.get("lock")
        kwargs["classified_scenes"][0]["lip_sync"] = False
        return [job / "clip.mp4"]

    monkeypatch.setattr(engine, "process_shots", fake_process_shots)
    monkeypatch.setattr(engine, "stitch_segments", lambda **kwargs: None)

    engine.run_dubbing_engine(
        video, "hi", job_dir=job, dry_run=False, face_lock=lock,
        full_video_sync=False,
    )
    assert seen["lock"] is lock
    saved = json.loads((job / "scenes.json").read_text(encoding="utf-8"))
    assert saved[0]["lip_sync"] is False


def test_full_video_sync_submits_complete_video_and_audio(tmp_path, monkeypatch):
    video = make_video(tmp_path / "video.mp4", seconds=2.0)
    audio = make_tone(tmp_path / "dub.mp3", seconds=2.0)
    uploads = []

    class FakeSync:
        def __init__(self, default_model):
            self.default_model = default_model

        def upload_file(self, path, content_type):
            uploads.append((Path(path), content_type))
            return f"uploaded://{Path(path).name}"

        def generate(self, video_url, audio_url, model, options):
            assert video_url.endswith("video.mp4")
            assert audio_url.endswith("dubbed_full.wav")
            return {"id": "full-1"}

        def wait_for_completion(self, generation_id, poll_interval):
            return {"outputUrl": "result://full"}

        def download_result(self, url, destination):
            import shutil
            shutil.copyfile(video, destination)

    monkeypatch.setattr(lipsyncer, "SyncLabsClient", FakeSync)
    output = lipsyncer.process_full_video(video, audio, tmp_path / "job", lock=None)

    assert output.is_file()
    assert uploads[0] == (video, "video/mp4")
    assert uploads[1][0].name == "dubbed_full.wav"
    assert uploads[1][1] == "audio/wav"


def test_full_video_restore_uses_original_only_for_broll(tmp_path, monkeypatch):
    source = make_video(tmp_path / "source.mp4", seconds=3.0)
    synced = make_video(tmp_path / "synced.mp4", seconds=3.0, freq=600)
    scenes = [
        {"start": 0.0, "end": 1.0, "class": "A_ROLL"},
        {"start": 1.0, "end": 2.0, "class": "B_ROLL"},
        {"start": 2.0, "end": 3.0, "class": "SILENT"},
    ]
    extracted_from = []

    def fake_extract(video_path, start, end, output_path, target_fps):
        extracted_from.append((Path(video_path), start, end))
        Path(output_path).write_bytes(b"x" * 1001)

    monkeypatch.setattr(lipsyncer, "extract_subclip", fake_extract)
    clips = lipsyncer.restore_broll_from_source(
        synced, source, scenes, tmp_path / "job"
    )

    assert [item[0] for item in extracted_from] == [synced, source, source]
    assert len(clips) == 3
    assert scenes[0]["lip_sync"] == "full_video"
    assert scenes[1]["lip_sync"] == "restored_original_broll"
    assert scenes[2]["lip_sync"] == "restored_original_silent"


def test_engine_restores_broll_after_full_video_sync(tmp_path, monkeypatch):
    video = make_video(tmp_path / "v.mp4", seconds=2.0)
    job = tmp_path / "job"
    synced = tmp_path / "synced.mp4"
    seen = {}

    monkeypatch.setattr(engine, "extract_transcript",
                        lambda **kwargs: (job / "t.srt", [{"word": "hi", "start": 0.1, "end": 0.5}]))
    monkeypatch.setattr(
        engine, "classify_video",
        lambda **kwargs: [{"start": 0.0, "end": 1.0, "class": "A_ROLL"},
                          {"start": 1.0, "end": 2.0, "class": "B_ROLL"}],
    )
    monkeypatch.setattr(
        engine, "run_dubbing",
        lambda **kwargs: (job / "dub.mp3", job / "dub.srt", None),
    )
    monkeypatch.setattr(engine, "process_full_video", lambda **kwargs: synced)

    def fake_restore(**kwargs):
        seen.update(kwargs)
        kwargs["classified_scenes"][1]["lip_sync"] = "restored_original_broll"
        return [tmp_path / "a.mp4", tmp_path / "b.mp4"]

    monkeypatch.setattr(engine, "restore_broll_from_source", fake_restore)
    monkeypatch.setattr(engine, "stitch_segments", lambda **kwargs: seen.setdefault("stitch", kwargs))

    engine.run_dubbing_engine(video, "hi", job_dir=job, full_video_sync=True)

    assert seen["synced_video"] == synced
    assert seen["source_video"] == video.resolve()
    assert len(seen["stitch"]["segments"]) == 2
    saved = json.loads((job / "scenes.json").read_text(encoding="utf-8"))
    assert saved[1]["lip_sync"] == "restored_original_broll"
