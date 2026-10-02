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
