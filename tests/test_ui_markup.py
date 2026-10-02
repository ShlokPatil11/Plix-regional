from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")


def test_ui_has_face_panel_and_music_slider_hooks():
    for needle in ("awaiting_face_selection", "/faces`", "/face`", "/remix`", "type = 'range'",
                   "Confirm & continue", "Apply", "music_percent"):
        assert needle in HTML, needle


def test_ui_never_uses_inner_html_for_dynamic_text():
    assert "innerHTML" not in HTML
