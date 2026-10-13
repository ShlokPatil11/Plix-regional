from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from models.subtitle import BoundingBox, SubtitleSegment


logger = logging.getLogger(__name__)


@dataclass
class Observation:
    timestamp: float
    text: str
    box: tuple[int, int, int, int]
    background: str
    region: str


class OpenAICardDetector:
    """Find every editorial text overlay using sampled frames and GPT vision."""

    def __init__(self, config):
        self.config = config
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for GPT vision masking")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_VISION_MODEL", "gpt-6-luna")
        base_interval = max(0.2, float(os.getenv("OPENAI_VISION_SAMPLE_SECONDS", "0.5")))
        # Sampling more densely is equivalent to slowing fast footage for visual analysis,
        # while leaving the source and rendered output at their original playback speed.
        slowdown = max(1.0, float(os.getenv("OPENAI_VISION_ANALYSIS_SLOWDOWN", "2.0")))
        self.sample_seconds = max(0.1, base_interval / slowdown)
        self.mask_preroll = max(0.0, float(os.getenv("OPENAI_VISION_MASK_PREROLL_SECONDS", "0.30")))
        self.mask_postroll = max(0.0, float(os.getenv("OPENAI_VISION_MASK_POSTROLL_SECONDS", "0.20")))

    def _frame(self, video: Path, timestamp: float) -> Image.Image:
        completed = subprocess.run(
            [getattr(self.config, "ffmpeg", "ffmpeg"), "-loglevel", "error", "-ss", f"{timestamp:.3f}",
             "-i", str(video), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
            capture_output=True, timeout=20,
        )
        if completed.returncode or not completed.stdout:
            raise RuntimeError(completed.stderr.decode(errors="replace") or "Could not extract video frame")
        image = Image.open(io.BytesIO(completed.stdout)).convert("RGB")
        image.load()
        return image

    @staticmethod
    def _data_url(image: Image.Image) -> str:
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=88, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")

    @staticmethod
    def _schema() -> dict:
        overlay = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "box_2d": {
                    "type": "array", "items": {"type": "integer"},
                    "minItems": 4, "maxItems": 4,
                    "description": "[ymin,xmin,ymax,xmax], normalized 0-1000",
                },
                "background_hex": {"type": "string"},
                "region": {"type": "string", "enum": ["top", "middle", "bottom"]},
            },
            "required": ["text", "box_2d", "background_hex", "region"],
            "additionalProperties": False,
        }
        frame = {
            "type": "object",
            "properties": {
                "frame_index": {"type": "integer"},
                "overlays": {"type": "array", "items": overlay},
            },
            "required": ["frame_index", "overlays"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"frames": {"type": "array", "items": frame}},
            "required": ["frames"],
            "additionalProperties": False,
        }

    def _inspect_batch(self, frames: list[tuple[float, Image.Image]]) -> list[Observation]:
        prompt = (
            "Inspect every frame independently and return ALL changing editorial text overlays. "
            "Editorial overlays include spoken-dialogue subtitles, lower-third captions, before/after labels, "
            "day counters, benefits, and temporary title cards. When top and bottom overlays coexist, return BOTH. "
            "Never omit a long lower-third caption because a short top label exists. Exclude text physically printed "
            "on products or scenery, permanent logos/watermarks, player controls, and static end-card advertising. "
            "Transcribe the visible text exactly in its source script. The bounding box must cover the entire opaque "
            "card/background, not only glyphs. box_2d is [ymin,xmin,ymax,xmax] normalized from 0 to 1000. "
            "frame_index is zero-based in the order supplied. Return an empty overlays list when none exists."
        )
        content = [{"type": "input_text", "text": prompt}]
        for index, (_, image) in enumerate(frames):
            content.append({"type": "input_text", "text": f"Frame {index}"})
            content.append({"type": "input_image", "image_url": self._data_url(image), "detail": "high"})
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "video_text_overlays",
                             "strict": True, "schema": self._schema()}},
        )
        import json

        payload = json.loads(response.output_text)
        observations = []
        for result in payload.get("frames", []):
            index = int(result.get("frame_index", -1))
            if not 0 <= index < len(frames):
                continue
            timestamp, image = frames[index]
            width, height = image.size
            for overlay in result.get("overlays", []):
                text = " ".join(str(overlay.get("text", "")).split())
                box = overlay.get("box_2d", [])
                if not text or len(box) != 4:
                    continue
                ymin, xmin, ymax, xmax = [max(0, min(1000, int(value))) for value in box]
                left, top = int(xmin * width / 1000), int(ymin * height / 1000)
                right, bottom = int(xmax * width / 1000), int(ymax * height / 1000)
                if right <= left or bottom <= top:
                    continue
                observations.append(Observation(timestamp, text, (left, top, right, bottom),
                                                str(overlay.get("background_hex") or "#000000"),
                                                str(overlay.get("region") or "middle")))
        return observations

    @staticmethod
    def _same_overlay(left: Observation, right: Observation) -> bool:
        if left.region != right.region:
            return False
        similarity = SequenceMatcher(None, left.text.casefold(), right.text.casefold()).ratio()
        lx1, ly1, lx2, ly2 = left.box
        rx1, ry1, rx2, ry2 = right.box
        intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(0, min(ly2, ry2) - max(ly1, ry1))
        union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
        # A permissive threshold joins OCR variations, but also incorrectly joins
        # short changing labels such as "Day 4", "Day 8", and "Day 14". Long
        # captions tolerate a little OCR noise; short cards must match closely.
        text_threshold = 0.93 if max(len(left.text), len(right.text)) <= 24 else 0.88
        return similarity >= text_threshold and union > 0 and intersection / union >= 0.35

    @staticmethod
    def _same_visual_lane(left: SubtitleSegment, right: SubtitleSegment) -> bool:
        if not left.bbox or not right.bbox:
            return False
        left_center = left.bbox.y + left.bbox.height / 2
        right_center = right.bbox.y + right.bbox.height / 2
        tolerance = max(80.0, max(left.bbox.height, right.bbox.height) * 1.5)
        return abs(left_center - right_center) <= tolerance

    def _clip_replaced_overlays(self, segments: list[SubtitleSegment]) -> None:
        """End an old card exactly when a new card takes over the same lane."""
        for index, current in enumerate(segments):
            replacements = [candidate for candidate in segments[index + 1:]
                            if current.start < candidate.start < current.end
                            and self._same_visual_lane(current, candidate)
                            # A transition can show two separate cards in the same
                            # lane at once. Keep masking both when the detector
                            # actually observed them concurrently; only clip a
                            # stale padded tail after the old card disappeared.
                            and getattr(candidate, "_first_observed_at", candidate.start)
                            > getattr(current, "_last_observed_at", current.end)]
            if replacements:
                current.end = min(current.end, min(item.start for item in replacements))

    def _segments(self, observations: list[Observation], duration: float) -> list[SubtitleSegment]:
        active: list[dict] = []
        completed: list[dict] = []
        for timestamp in sorted({item.timestamp for item in observations}):
            current = [item for item in observations if item.timestamp == timestamp]
            matched = set()
            for track in active:
                if timestamp - track["last"].timestamp > self.sample_seconds * 1.6:
                    completed.append(track)
                    continue
                match_index = next((i for i, item in enumerate(current)
                                    if i not in matched and self._same_overlay(track["last"], item)), None)
                if match_index is None:
                    continue
                item = current[match_index]
                matched.add(match_index)
                track["last"] = item
                track["items"].append(item)
            active = [track for track in active if track not in completed]
            for index, item in enumerate(current):
                if index not in matched:
                    active.append({"first": item, "last": item, "items": [item]})
        completed.extend(active)

        segments = []
        for track in completed:
            items = track["items"]
            first, last = items[0], items[-1]
            left = min(item.box[0] for item in items) - 18
            top = min(item.box[1] for item in items) - 6
            right = max(item.box[2] for item in items) + 18
            bottom = max(item.box[3] for item in items) + 6
            raw = next((item.background.strip().lstrip("#") for item in items
                        if len(item.background.strip().lstrip("#")) == 6), "000000")
            box = BoundingBox(max(0, left), max(0, top), max(1, right - max(0, left)),
                              max(1, bottom - max(0, top)), patch_color=f"0x{raw.upper()}")
            start = max(0.0, first.timestamp - self.mask_preroll)
            end = min(duration, last.timestamp + self.sample_seconds + self.mask_postroll)
            if end > start + 0.15:
                segment = SubtitleSegment(len(segments) + 1, start, end,
                                          first.text, bbox=box, ocr_confidence=1.0)
                segment._first_observed_at = first.timestamp
                segment._last_observed_at = last.timestamp
                segments.append(segment)
        segments.sort(key=lambda segment: (segment.start, segment.bbox.y if segment.bbox else 0))
        self._clip_replaced_overlays(segments)
        segments = [segment for segment in segments if segment.end > segment.start + 0.05]
        for index, segment in enumerate(segments, 1):
            segment.id = index
        return segments

    def detect_subtitles(self, video_path: str | Path, metadata) -> list[SubtitleSegment]:
        video = Path(video_path)
        duration = float(getattr(metadata, "duration", 0.0) or 0.0)
        if duration <= 0:
            raise ValueError("Video duration is unavailable")
        timestamps = []
        timestamp = 0.0
        while timestamp < duration:
            timestamps.append(round(timestamp, 3))
            timestamp += self.sample_seconds
        observations = []
        batch_size = 6
        for offset in range(0, len(timestamps), batch_size):
            frames = [(ts, self._frame(video, min(ts, max(0.0, duration - 0.02))))
                      for ts in timestamps[offset:offset + batch_size]]
            observations.extend(self._inspect_batch(frames))
        segments = self._segments(observations, duration)
        if not segments:
            logger.warning("GPT vision found no editorial overlay text")
        return segments
