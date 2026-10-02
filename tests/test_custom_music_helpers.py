import pytest
from helpers import make_tone, make_video, media_duration

from dir_language_pipeline.music import (
    MusicFetchError,
    YouTubeFetcher,
    is_youtube_url,
    prepare_custom_track,
)


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc123",
    "http://youtube.com/watch?v=abc123",
    "https://m.youtube.com/watch?v=abc123",
    "https://music.youtube.com/watch?v=abc123",
    "https://youtu.be/abc123",
    "  https://youtu.be/abc123  ",
])
def test_youtube_urls_are_accepted(url):
    assert is_youtube_url(url)


@pytest.mark.parametrize("url", [
    "https://example.com/watch?v=abc",
    "https://youtube.com.evil.example/watch?v=abc",
    "https://evil.example/youtube.com",
    "ftp://youtube.com/watch?v=abc",
    "file:///etc/passwd",
    "javascript:alert(1)",
    "youtube.com/watch?v=abc",
    "",
    "not a url",
])
def test_other_urls_are_rejected(url):
    assert not is_youtube_url(url)


def test_custom_track_is_trimmed_to_the_video_length(tmp_path):
    source = make_tone(tmp_path / "song.wav", seconds=8.0)
    out = tmp_path / "music_track.mp3"
    prepare_custom_track(source, 3.0, out)
    assert abs(media_duration(out) - 3.0) < 0.3


def test_custom_track_shorter_than_the_video_is_kept_as_is(tmp_path):
    source = make_tone(tmp_path / "short.wav", seconds=2.0)
    out = tmp_path / "music_track.mp3"
    prepare_custom_track(source, 6.0, out)
    assert 1.5 < media_duration(out) < 2.6


def test_custom_track_accepts_a_video_file_and_uses_its_audio(tmp_path):
    source = make_video(tmp_path / "clip.mp4", seconds=4.0)
    out = tmp_path / "music_track.mp3"
    prepare_custom_track(source, 3.0, out)
    assert abs(media_duration(out) - 3.0) < 0.3


def test_non_audio_file_is_rejected_without_leaking_paths_or_leftovers(tmp_path):
    source = tmp_path / "secret-folder-name.mp3"
    source.write_bytes(b"this is not audio at all " * 50)
    out = tmp_path / "music_track.mp3"
    with pytest.raises(MusicFetchError) as error:
        prepare_custom_track(source, 3.0, out)
    assert "secret-folder-name" not in str(error.value)
    assert not out.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_video_without_audio_is_rejected(tmp_path):
    source = make_video(tmp_path / "silent.mp4", seconds=3.0, with_audio=False)
    out = tmp_path / "music_track.mp3"
    with pytest.raises(MusicFetchError):
        prepare_custom_track(source, 3.0, out)
    assert not out.exists()


class FakeYDL:
    info = {"title": "Cool Song", "duration": 200}
    downloads = []
    probes = []

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        FakeYDL.probes.append((url, self.options))
        return FakeYDL.info

    def download(self, urls):
        FakeYDL.downloads.append((urls, self.options))
        out = self.options["outtmpl"].replace("%(ext)s", "mp3")
        open(out, "wb").write(b"x" * 100)


@pytest.fixture(autouse=True)
def reset_fake():
    FakeYDL.info = {"title": "Cool Song", "duration": 200}
    FakeYDL.downloads, FakeYDL.probes = [], []


def test_fetch_url_downloads_a_single_video_with_its_title(tmp_path):
    fetched = YouTubeFetcher(ydl_factory=FakeYDL).fetch_url("https://youtu.be/abc123", tmp_path)
    assert fetched.title == "Cool Song"
    assert fetched.url == "https://youtu.be/abc123"
    assert fetched.path.name == "custom_source.mp3" and fetched.path.is_file()
    urls, options = FakeYDL.downloads[0]
    assert urls == ["https://youtu.be/abc123"]
    assert options["noplaylist"] is True
    assert options["socket_timeout"] > 0


def test_fetch_url_refuses_non_youtube_links_before_any_network_use(tmp_path):
    with pytest.raises(MusicFetchError, match="youtube"):
        YouTubeFetcher(ydl_factory=FakeYDL).fetch_url("https://example.com/video", tmp_path)
    assert FakeYDL.probes == [] and FakeYDL.downloads == []


@pytest.mark.parametrize("info", [
    {"title": "Long", "duration": 16 * 60},
    {"title": "Unknown length"},
    {"title": "Live stream", "duration": None},
])
def test_fetch_url_rejects_long_or_unknown_length_videos_without_downloading(tmp_path, info):
    FakeYDL.info = info
    with pytest.raises(MusicFetchError):
        YouTubeFetcher(ydl_factory=FakeYDL).fetch_url("https://youtu.be/abc123", tmp_path)
    assert FakeYDL.downloads == []


def test_fetch_url_rejects_playlists(tmp_path):
    FakeYDL.info = {"_type": "playlist", "entries": [{"id": "a"}], "title": "Mix"}
    with pytest.raises(MusicFetchError, match="single video"):
        YouTubeFetcher(ydl_factory=FakeYDL).fetch_url("https://www.youtube.com/playlist?list=PL1", tmp_path)
    assert FakeYDL.downloads == []


def test_fetch_url_turns_downloader_errors_into_fetch_errors(tmp_path):
    class Boom(FakeYDL):
        def extract_info(self, url, download=False):
            raise RuntimeError("ERROR: Video unavailable /private/path")

    with pytest.raises(MusicFetchError) as error:
        YouTubeFetcher(ydl_factory=Boom).fetch_url("https://youtu.be/abc123", tmp_path)
    assert "/private/path" not in str(error.value)
