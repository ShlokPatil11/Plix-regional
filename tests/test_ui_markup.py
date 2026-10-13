from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")


def test_ui_has_face_panel_and_music_slider_hooks():
    for needle in ("awaiting_face_selection", "/faces`", "/face`", "/remix`", "type = 'range'",
                   "Confirm & continue", "Apply", "music_percent"):
        assert needle in HTML, needle


def test_ui_never_uses_inner_html_for_dynamic_text():
    assert "innerHTML" not in HTML


def test_ui_resumes_a_job_from_the_url_after_a_reload():
    for needle in ("URLSearchParams", "replaceState", "?job=", "startPolling"):
        assert needle in HTML, needle


def test_ui_handles_network_errors_in_confirm_and_apply():
    assert HTML.count("} catch (") >= 3  # submit, confirm and apply handlers
    assert "confirm.disabled = false" in HTML and "apply.disabled = false" in HTML


def test_ui_offers_custom_music_by_file_or_youtube_link():
    for needle in ("Use custom music", "/music`", "type = 'file'", "YouTube link", "new FormData()",
                   "form.append('file', file)", "form.append('url', link)", "music_percent"):
        assert needle in HTML, needle


def test_custom_music_is_offered_even_when_no_music_was_identified():
    # The custom section is appended after the ok / not-ok branches, not inside the ok branch.
    ok_branch_end = HTML.index("panel.append(label, slider, value, apply, status);")
    assert HTML.index("panel.append(customMusicSection(") > ok_branch_end
    assert "Music not identified — voice-only video." in HTML


def test_custom_music_is_offered_when_detection_was_not_run():
    assert "job.music || {status: 'not_analyzed', percent: 25}" in HTML
    assert "Automatic music detection was not run for this masking job." in HTML


def test_detected_masking_music_is_not_mixed_twice():
    assert "music.mixable !== false" in HTML
    assert "The original masking audio already contains this track." in HTML


def test_detector_requests_simultaneous_lower_third_overlays():
    detector = (Path(__file__).resolve().parents[1] / "vendor/service_a/pipeline/openai_card_detector.py").read_text(encoding="utf-8")
    assert 'self.model = os.getenv("OPENAI_VISION_MODEL", "gpt-6-luna")' in detector
    assert "When top and bottom overlays coexist, return BOTH" in detector
    assert "Never omit a long lower-third caption" in detector


def test_ui_has_dedicated_create_queue_and_history_views():
    for needle in ('id="create-view"', 'id="queue-view"', 'id="history-view"',
                   'data-view="queue"', 'data-view="history"', "method: 'DELETE'", 'Remove'):
        assert needle in HTML, needle


def test_ui_plays_boot_video_before_revealing_the_app():
    for needle in ('id="boot-screen"', 'id="boot-video"', 'autoplay muted playsinline',
                   '/boot-media/Green_liquid_filling_frame_1080p_20261005174418_gwr_video_mvp.mp4',
                   "addEventListener('ended', finishBoot)", 'Skip intro'):
        assert needle in HTML, needle


def test_ui_offers_audio_transcription_caption_mode():
    for needle in ('value="transcript"', "Auto captions", "transcribe audio, translate"):
        assert needle in HTML, needle
