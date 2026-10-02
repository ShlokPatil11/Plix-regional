import json

import pytest
from helpers import make_tone, make_video, mean_volume_db

from dir_language_pipeline import in_process
from dir_language_pipeline.contracts import AwaitingFaceSelection
from dir_language_pipeline.media import FileArtifactBackend


def make_fake_adapter(tmp_path, *, corrupt_music=False, pause=False):
    class FakeTextToAudioAdapter:
        def __init__(self, backend, face_choice=None, **kwargs):
            self.backend = backend

        def run(self, video, *, target_language, source_language="auto", job_id):
            if pause:
                raise AwaitingFaceSelection(job_id)
            voice = make_video(tmp_path / "gen_voice.mp4", seconds=2.0, freq=300)
            if corrupt_music:
                music = tmp_path / "gen_music.mp3"
                music.write_bytes(b"\x00garbage" * 25)
            else:
                music = make_tone(tmp_path / "gen_music.mp3", seconds=2.0, freq=900)
            info = tmp_path / "gen_music.json"
            info.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            up = self.backend.upload
            return {
                "video": up(voice, filename="service-b-video.mp4", content_type="video/mp4"),
                "music_track": up(music, filename="music_track.mp3", content_type="audio/mpeg"),
                "music_info": up(info, filename="music.json", content_type="application/json"),
            }

    return FakeTextToAudioAdapter


def test_dubbing_run_operation_mixes_music_and_publishes_the_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(in_process, "TextToAudioAdapter", make_fake_adapter(tmp_path))
    backend = FileArtifactBackend(tmp_path / "art")
    source = backend.upload(make_video(tmp_path / "in.mp4"), filename="in.mp4", content_type="video/mp4")
    artifacts = in_process.run_operation(
        source, operation="dubbing", target_language="hi", backend=backend, job_id="job1"
    )
    for name in ("final_video", "dubbed_video", "voice_video", "music_track", "music_info"):
        assert name in artifacts
    final = artifacts["final_video"].metadata["path"]
    voice = artifacts["voice_video"].metadata["path"]
    assert mean_volume_db(final) > mean_volume_db(voice) + 0.5


def test_dubbing_with_a_corrupt_music_track_completes_voice_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(in_process, "TextToAudioAdapter", make_fake_adapter(tmp_path, corrupt_music=True))
    backend = FileArtifactBackend(tmp_path / "art")
    source = backend.upload(make_video(tmp_path / "in.mp4"), filename="in.mp4", content_type="video/mp4")
    artifacts = in_process.run_operation(
        source, operation="dubbing", target_language="hi", backend=backend, job_id="job1"
    )
    assert artifacts["final_video"].filename == "dubbed_video.mp4"
    assert (backend.root / "dubbed_video.mp4").stat().st_size > 0
    assert "voice-only" in capsys.readouterr().out


def test_face_selection_pause_propagates_out_of_run_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(in_process, "TextToAudioAdapter", make_fake_adapter(tmp_path, pause=True))
    backend = FileArtifactBackend(tmp_path / "art")
    source = backend.upload(make_video(tmp_path / "in.mp4"), filename="in.mp4", content_type="video/mp4")
    with pytest.raises(AwaitingFaceSelection):
        in_process.run_operation(source, operation="dubbing", target_language="hi", backend=backend, job_id="job1")
