import pytest
from helpers import make_tone, make_video, mean_volume_db, stream_types

from dir_language_pipeline.mixer import MixError, build_mix_command, mix_final, music_gain


@pytest.fixture
def media(tmp_path):
    voice = make_video(tmp_path / "voice.mp4", seconds=3.0, freq=300)
    music = make_tone(tmp_path / "music.mp3", seconds=3.0, freq=800)
    subtitles = tmp_path / "s.srt"
    subtitles.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    return voice, music, subtitles


def test_music_gain_bounds():
    assert music_gain(0) == 0.0
    assert music_gain(100) == 1.0
    assert music_gain(25) == 0.25
    for bad in (-1, 101, 12.5, "50", True):
        with pytest.raises(ValueError):
            music_gain(bad)


def test_zero_percent_keeps_voice_only_level(media, tmp_path):
    voice, music, _ = media
    out = mix_final(voice, music, 0, tmp_path / "out0.mp4")
    assert abs(mean_volume_db(out) - mean_volume_db(voice)) < 1.5


def test_music_makes_the_mix_louder_and_slider_is_monotonic(media, tmp_path):
    voice, music, _ = media
    quiet = mean_volume_db(mix_final(voice, music, 0, tmp_path / "a.mp4"))
    half = mean_volume_db(mix_final(voice, music, 50, tmp_path / "b.mp4"))
    full = mean_volume_db(mix_final(voice, music, 100, tmp_path / "c.mp4"))
    assert half > quiet + 1.0
    assert full > half + 1.0


def test_subtitle_track_is_attached_with_and_without_music(media, tmp_path):
    voice, music, subtitles = media
    with_music = mix_final(voice, music, 40, tmp_path / "m.mp4", subtitles=subtitles)
    without = mix_final(voice, None, 40, tmp_path / "n.mp4", subtitles=subtitles)
    assert "subtitle" in stream_types(with_music)
    assert "subtitle" in stream_types(without)
    assert stream_types(with_music).count("audio") == 1


def test_video_stream_is_copied_not_reencoded(media, tmp_path):
    voice, music, _ = media
    command = build_mix_command("ffmpeg", voice, music, 30, tmp_path / "o.mp4")
    assert command[command.index("-c:v") + 1] == "copy"


def test_failure_raises_mix_error_and_leaves_no_partial_file(media, tmp_path):
    _, music, _ = media
    missing = tmp_path / "missing.mp4"
    out = tmp_path / "out.mp4"
    with pytest.raises(MixError):
        mix_final(missing, music, 30, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".out.*"))
