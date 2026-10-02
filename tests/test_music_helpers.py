import subprocess

from helpers import make_tone, media_duration, mean_volume_db

from dir_language_pipeline.music import (
    MusicMatch,
    build_align_command,
    choose_youtube_result,
    music_song_start,
    parse_shazam_result,
    pick_quiet_windows,
)


def words(*spans):
    return [{"word": "w", "start": s, "end": e} for s, e in spans]


def test_quiet_windows_avoid_speech():
    spoken = words((0, 10))
    windows = pick_quiet_windows(spoken, duration_s=40, window_s=12, count=3)
    assert windows[0][0] >= 10
    for start, end in windows:
        assert end - start == 12


def test_windows_do_not_overlap_and_respect_count():
    windows = pick_quiet_windows([], duration_s=100, window_s=12, count=3)
    assert len(windows) == 3
    ordered = sorted(windows)
    for (_, previous_end), (next_start, _) in zip(ordered, ordered[1:]):
        assert next_start >= previous_end


def test_short_video_gets_one_window_covering_all_of_it():
    assert pick_quiet_windows(words((0, 2)), duration_s=5.0) == [(0.0, 5.0)]
    assert pick_quiet_windows([], duration_s=0) == []


def test_song_start_is_offset_minus_window_start():
    match = MusicMatch("t", "a", offset_s=40.0, window_start_s=10.0)
    assert music_song_start(match) == 30.0


def test_align_trims_from_song_start_and_limits_duration(tmp_path):
    track = make_tone(tmp_path / "track.mp3", seconds=6.0)
    out = tmp_path / "aligned.mp3"
    subprocess.run(build_align_command("ffmpeg", track, 1.0, 3.0, out), check=True, capture_output=True)
    assert abs(media_duration(out) - 3.0) < 0.2


def test_negative_song_start_delays_the_music(tmp_path):
    track = make_tone(tmp_path / "track.mp3", seconds=6.0)
    out = tmp_path / "aligned.mp3"
    subprocess.run(build_align_command("ffmpeg", track, -1.0, 3.0, out), check=True, capture_output=True)
    assert abs(media_duration(out) - 3.0) < 0.2
    assert mean_volume_db(out, start=0, duration=0.5) < -60


def entry(title, duration, channel="Some Channel", id_="abc"):
    return {"title": title, "duration": duration, "channel": channel, "id": id_}


def test_chooser_prefers_topic_channel_and_rejects_live_and_cover():
    entries = [
        entry("Artist - Song (Live at Arena)", 215, id_="live"),
        entry("Song cover by someone", 214, id_="cover"),
        entry("Artist - Song (Official Audio)", 215, channel="Artist - Topic", id_="topic"),
        entry("Artist - Song lyrics", 216, id_="lyrics"),
    ]
    best = choose_youtube_result(entries, artist="Artist", title="Song", track_length_s=215)
    assert best["id"] == "topic"


def test_chooser_rejects_wrong_duration_and_returns_none_when_nothing_fits():
    entries = [entry("Artist - Song", 600, id_="long"), entry("Artist - Song", None, id_="unknown")]
    assert choose_youtube_result(entries, artist="Artist", title="Song", track_length_s=215) is None


def test_chooser_without_known_length_still_filters_implausible_durations():
    entries = [entry("Artist - Song", 20, id_="short"), entry("Artist - Song", 200, id_="ok")]
    assert choose_youtube_result(entries, artist="Artist", title="Song")["id"] == "ok"


def test_chooser_allows_banned_word_when_the_song_title_has_it():
    entries = [entry("Artist - Live Forever (Official Audio)", 230, id_="ok")]
    best = choose_youtube_result(entries, artist="Artist", title="Live Forever")
    assert best["id"] == "ok"


def test_parse_shazam_result():
    result = {"matches": [{"offset": 42.5}], "track": {"title": "Song", "subtitle": "Artist"}}
    match = parse_shazam_result(result)
    assert (match.title, match.artist, match.offset_s) == ("Song", "Artist", 42.5)
    assert parse_shazam_result({"matches": []}) is None
    assert parse_shazam_result({"track": {"subtitle": "no title"}}) is None
    assert parse_shazam_result({"track": {"title": "T"}}).offset_s == 0.0
