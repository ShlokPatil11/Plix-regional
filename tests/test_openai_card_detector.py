import importlib
import sys
from pathlib import Path


SERVICE_A = Path(__file__).resolve().parents[1] / "vendor" / "service_a"
if str(SERVICE_A) not in sys.path:
    sys.path.insert(0, str(SERVICE_A))

detector_module = importlib.import_module("pipeline.openai_card_detector")
Observation = detector_module.Observation
OpenAICardDetector = detector_module.OpenAICardDetector


def detector(sample_seconds=0.5):
    instance = object.__new__(OpenAICardDetector)
    instance.sample_seconds = sample_seconds
    instance.mask_preroll = 0.3
    instance.mask_postroll = 0.2
    return instance


def test_simultaneous_top_and_bottom_observations_become_independent_segments():
    observations = [
        Observation(1.0, "Day 14", (400, 240, 650, 330), "#000000", "top"),
        Observation(1.0, "A long lower caption", (100, 1500, 980, 1660), "#000000", "bottom"),
        Observation(1.5, "Day 14", (402, 242, 652, 332), "#000000", "top"),
        Observation(1.5, "A long lower caption", (102, 1502, 978, 1658), "#000000", "bottom"),
    ]

    segments = detector()._segments(observations, 3.0)

    assert len(segments) == 2
    assert {segment.source_text for segment in segments} == {"Day 14", "A long lower caption"}
    assert {"top", "bottom"} == {"top" if segment.bbox.y < 500 else "bottom" for segment in segments}
    assert all(segment.start == 0.7 for segment in segments)
    assert all(segment.end == 2.2 for segment in segments)


def test_same_text_does_not_bridge_a_long_period_when_overlay_disappears():
    observations = [
        Observation(0.0, "Before", (100, 200, 400, 300), "#000000", "top"),
        Observation(2.0, "Before", (100, 200, 400, 300), "#000000", "top"),
    ]

    segments = detector()._segments(observations, 3.0)

    assert len(segments) == 2


def test_short_changing_labels_are_not_merged_and_handoff_without_overlap():
    observations = [
        Observation(1.0, "Day 4", (400, 240, 650, 330), "#000000", "top"),
        Observation(1.25, "Day 4", (400, 240, 650, 330), "#000000", "top"),
        Observation(1.5, "Day 8", (400, 240, 650, 330), "#000000", "top"),
        Observation(1.75, "Day 8", (400, 240, 650, 330), "#000000", "top"),
    ]

    segments = detector(0.25)._segments(observations, 3.0)

    assert [segment.source_text for segment in segments] == ["Day 4", "Day 8"]
    assert segments[0].end == segments[1].start


def test_simultaneous_top_and_bottom_cards_are_not_clipped_as_replacements():
    observations = [
        Observation(1.0, "Day 4", (400, 240, 650, 330), "#000000", "top"),
        Observation(1.25, "Day 4", (400, 240, 650, 330), "#000000", "top"),
        Observation(1.25, "Lower caption", (100, 1500, 980, 1660), "#000000", "bottom"),
    ]

    segments = detector(0.25)._segments(observations, 3.0)
    top = next(segment for segment in segments if segment.source_text == "Day 4")
    bottom = next(segment for segment in segments if segment.source_text == "Lower caption")

    assert top.end > bottom.start


def test_simultaneous_cards_in_the_same_lane_remain_covered():
    observations = [
        Observation(1.0, "Old card", (80, 240, 330, 330), "#000000", "top"),
        Observation(1.25, "Old card", (80, 240, 330, 330), "#000000", "top"),
        Observation(1.25, "New card", (430, 240, 680, 330), "#000000", "top"),
        Observation(1.5, "New card", (430, 240, 680, 330), "#000000", "top"),
    ]

    segments = detector(0.25)._segments(observations, 3.0)
    old = next(segment for segment in segments if segment.source_text == "Old card")
    new = next(segment for segment in segments if segment.source_text == "New card")

    assert old.end > new.start
