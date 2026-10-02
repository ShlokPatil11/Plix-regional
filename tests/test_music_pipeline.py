import asyncio
import json

from helpers import make_tone, make_video, media_duration

from dir_language_pipeline.music import (
    FetchedTrack,
    MusicFetchError,
    MusicMatch,
    ShazamIdentifier,
    YouTubeFetcher,
    identify_and_fetch,
)


class FakeIdentifier:
    def __init__(self, match=None, error=None):
        self.match, self.error, self.calls = match, error, 0

    def identify(self, clip):
        self.calls += 1
        if self.error:
            raise self.error
        return self.match


class FakeFetcher:
    def __init__(self, tmp_path, error=None):
        self.tmp_path, self.error, self.calls = tmp_path, error, []

    def fetch(self, artist, title, dest_dir, track_length_s=None):
        self.calls.append((artist, title))
        if self.error:
            raise self.error
        path = make_tone(dest_dir / "yt_source.mp3", seconds=8.0, freq=700)
        return FetchedTrack(path=path, url="https://www.youtube.com/watch?v=abc")


WORDS = [{"word": "hi", "start": 0.0, "end": 1.0}]


def test_identified_track_is_fetched_aligned_and_recorded(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=5.0)
    identifier = FakeIdentifier(MusicMatch("Song", "Artist", offset_s=3.0))
    result = identify_and_fetch(source, WORDS, 5.0, tmp_path / "music", identifier, FakeFetcher(tmp_path))
    assert result.status == "ok"
    assert result.track.name == "music_track.mp3"
    assert abs(media_duration(result.track) - 5.0) < 0.3
    saved = json.loads((tmp_path / "music" / "music.json").read_text(encoding="utf-8"))
    assert saved["status"] == "ok"
    assert saved["title"] == "Song" and saved["artist"] == "Artist"
    assert saved["source_url"].startswith("https://www.youtube.com/")


def test_not_identified_is_a_soft_result(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=5.0)
    result = identify_and_fetch(source, WORDS, 5.0, tmp_path / "music", FakeIdentifier(None), FakeFetcher(tmp_path))
    assert result.status == "not_identified" and result.track is None
    assert json.loads((tmp_path / "music" / "music.json").read_text())["status"] == "not_identified"


def test_download_failure_is_a_soft_result_with_reason(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=5.0)
    result = identify_and_fetch(
        source, WORDS, 5.0, tmp_path / "music",
        FakeIdentifier(MusicMatch("Song", "Artist", 1.0)),
        FakeFetcher(tmp_path, error=MusicFetchError("no suitable result")),
    )
    assert result.status == "download_failed"
    assert "no suitable result" in result.info["error"]
    assert result.info["title"] == "Song"


def test_identifier_exception_counts_as_no_match(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=5.0)
    result = identify_and_fetch(
        source, WORDS, 5.0, tmp_path / "music", FakeIdentifier(error=RuntimeError("network down")), FakeFetcher(tmp_path)
    )
    assert result.status == "not_identified"


def test_source_without_audio_track_is_not_identified_not_a_crash(tmp_path):
    silent = make_video(tmp_path / "silent.mp4", seconds=5.0, with_audio=False)
    identifier = FakeIdentifier(MusicMatch("Song", "Artist", 1.0))
    result = identify_and_fetch(silent, [], 5.0, tmp_path / "music", identifier, FakeFetcher(tmp_path))
    assert result.status == "not_identified"
    assert identifier.calls == 0


def test_result_is_cached_for_resume(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=5.0)
    identifier = FakeIdentifier(MusicMatch("Song", "Artist", 1.0))
    fetcher = FakeFetcher(tmp_path)
    identify_and_fetch(source, WORDS, 5.0, tmp_path / "music", identifier, fetcher)
    again = identify_and_fetch(source, WORDS, 5.0, tmp_path / "music", identifier, fetcher)
    assert again.status == "ok" and again.track.is_file()
    assert identifier.calls == 1 and len(fetcher.calls) == 1


def test_shazam_identifier_wraps_the_async_client(tmp_path, monkeypatch):
    identifier = ShazamIdentifier()

    async def fake_recognize(self, clip):
        return {"matches": [{"offset": 12.0}], "track": {"title": "Song", "subtitle": "Artist"}}

    monkeypatch.setattr(ShazamIdentifier, "_recognize", fake_recognize)
    match = identifier.identify(tmp_path / "clip.wav")
    assert (match.title, match.artist, match.offset_s) == ("Song", "Artist", 12.0)


class FakeYDL:
    searches = []
    downloads = []
    entries = []

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, query, download=False):
        FakeYDL.searches.append(query)
        return {"entries": FakeYDL.entries}

    def download(self, urls):
        FakeYDL.downloads.append(urls)
        out = self.options["outtmpl"].replace("%(ext)s", "mp3")
        open(out, "wb").write(b"x" * 100)


def test_youtube_fetcher_searches_picks_best_and_downloads(tmp_path):
    FakeYDL.searches, FakeYDL.downloads = [], []
    FakeYDL.entries = [
        {"title": "Artist - Song (Official Audio)", "duration": 200, "channel": "Artist - Topic", "id": "good"},
        {"title": "Song cover", "duration": 200, "channel": "x", "id": "bad"},
    ]
    fetched = YouTubeFetcher(ydl_factory=FakeYDL).fetch("Artist", "Song", tmp_path, track_length_s=200)
    assert FakeYDL.searches == ["ytsearch8:Artist Song audio"]
    assert FakeYDL.downloads == [["https://www.youtube.com/watch?v=good"]]
    assert fetched.path.name == "yt_source.mp3" and fetched.url.endswith("good")


def test_youtube_fetcher_raises_when_nothing_suitable(tmp_path):
    import pytest

    FakeYDL.entries = [{"title": "Song cover", "duration": 200, "channel": "x", "id": "bad"}]
    with pytest.raises(MusicFetchError):
        YouTubeFetcher(ydl_factory=FakeYDL).fetch("Artist", "Song", tmp_path)
