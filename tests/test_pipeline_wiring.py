import pytest
from helpers import make_tone, make_video, mean_volume_db, stream_types

from dir_language_pipeline.contracts import Artifact, AwaitingFaceSelection, PipelineStageError
from dir_language_pipeline.media import FfmpegFinalAssembler, FileArtifactBackend
from dir_language_pipeline.runner import DirectoryLanguagePipeline, PipelineServices


def artifact(backend, path, name, content_type):
    return backend.upload(path, filename=name, content_type=content_type)


class FakeVisual:
    def __init__(self, backend, tmp_path):
        self.backend, self.tmp_path = backend, tmp_path

    def run(self, video, **kwargs):
        srt = self.tmp_path / "v.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
        return {"video": video, "visual_subtitles": artifact(self.backend, srt, "visual-subtitles.srt", "application/x-subrip")}


class PausingAudio:
    def run(self, video, **kwargs):
        raise AwaitingFaceSelection("job1")


class MusicAudio:
    def __init__(self, backend, tmp_path):
        self.backend, self.tmp_path = backend, tmp_path

    def run(self, video, **kwargs):
        voice = make_video(self.tmp_path / "voice.mp4", seconds=2.0)
        music = make_tone(self.tmp_path / "m.mp3", seconds=2.0)
        return {
            "video": artifact(self.backend, voice, "service-b-video.mp4", "video/mp4"),
            "music_track": artifact(self.backend, music, "music_track.mp3", "audio/mpeg"),
        }


class RecordingAssembler:
    def __init__(self, inner):
        self.inner, self.kwargs = inner, None

    def assemble(self, video, audio, visual_subtitles, **kwargs):
        self.kwargs = kwargs
        return self.inner.assemble(video, audio, visual_subtitles, **kwargs)


def services(backend, audio, assembler):
    visual = FakeVisual(backend, backend.root.parent)
    return PipelineServices(visual_subtitles=visual, audio_dubbing=audio, assembler=assembler, publisher=backend)


def test_pause_is_not_wrapped_in_a_stage_error(tmp_path):
    backend = FileArtifactBackend(tmp_path / "art")
    source = artifact(backend, make_video(tmp_path / "in.mp4"), "in.mp4", "video/mp4")
    pipeline = DirectoryLanguagePipeline(services(backend, PausingAudio(), FfmpegFinalAssembler(backend)))
    with pytest.raises(AwaitingFaceSelection):
        pipeline.run(source, target_language="hi", job_id="job1")


def test_music_flows_to_the_assembler_and_into_the_result(tmp_path):
    backend = FileArtifactBackend(tmp_path / "art")
    source = artifact(backend, make_video(tmp_path / "in.mp4"), "in.mp4", "video/mp4")
    assembler = RecordingAssembler(FfmpegFinalAssembler(backend))
    pipeline = DirectoryLanguagePipeline(services(backend, MusicAudio(backend, tmp_path), assembler))
    result = pipeline.run(source, target_language="hi", job_id="job1")
    assert assembler.kwargs["music"].filename.startswith("music_track")
    assert assembler.kwargs["music_percent"] == 25
    assert result.music_track is not None
    final = result.final_video.metadata["path"]
    assert "subtitle" in stream_types(final)


def test_assembler_mixes_music_at_the_requested_percent(tmp_path):
    backend = FileArtifactBackend(tmp_path / "art")
    voice = artifact(backend, make_video(tmp_path / "v.mp4", seconds=2.0), "v.mp4", "video/mp4")
    music = artifact(backend, make_tone(tmp_path / "m.mp3", seconds=2.0), "m.mp3", "audio/mpeg")
    srt_path = tmp_path / "s.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    srt = artifact(backend, srt_path, "s.srt", "application/x-subrip")
    assembler = FfmpegFinalAssembler(backend)
    plain = assembler.assemble(voice, None, srt, job_id="j")
    mixed = assembler.assemble(voice, None, srt, job_id="j", music=music, music_percent=80)
    assert mean_volume_db(mixed.metadata["path"]) > mean_volume_db(plain.metadata["path"]) + 1.0
