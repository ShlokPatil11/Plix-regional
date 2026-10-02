from helpers import make_video, stream_types


def test_package_and_service_b_import():
    from dir_language_pipeline.contracts import Artifact
    from src.audio import ffmpeg

    assert Artifact is not None
    assert ffmpeg()


def test_helpers_build_media(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=1.0)
    assert stream_types(video) == ["video", "audio"]
