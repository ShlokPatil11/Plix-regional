import io
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from models.subtitle import BoundingBox, SubtitleSegment

logger = logging.getLogger(__name__)

@dataclass
class AgentCardResult:
    card_detected: bool
    bbox: BoundingBox | None
    bg_hex_color: str | None
    text_hex_color: str | None
    description: str = ''

class AgentCardDetector:
    """Uses Gemini Multimodal Vision to visually locate the floating title pill / card,
    extracting its precise bounding box, background color, and text styling.
    """
    def __init__(self, config):
        self.config = config
        api_key = getattr(config, 'gemini_api_key', '') or os.getenv('GEMINI_API_KEY', '')
        if not api_key:
            raise ValueError('GEMINI_API_KEY is required for AgentCardDetector')
        from google import genai
        self.client = genai.Client(api_key=api_key)
        pref = getattr(config, 'gemini_model', 'gemini-3.5-flash-lite') or 'gemini-3.5-flash-lite'
        if pref in ('gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-2.5-flash-lite'):
            pref = 'gemini-3.5-flash-lite'
        self.preferred_model = pref

    def extract_keyframe(self, video_path: str | Path, timestamp: float = 0.5) -> Image.Image:
        ffmpeg_bin = getattr(self.config, 'ffmpeg', 'ffmpeg')
        cmd = [
            ffmpeg_bin,
            '-loglevel', 'error',
            '-ss', str(timestamp),
            '-i', str(video_path),
            '-vframes', '1',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            'pipe:1'
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=15)
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(f'Failed to extract keyframe at {timestamp}s: {proc.stderr.decode(errors="replace")}')
        return Image.open(io.BytesIO(proc.stdout))

    def detect_master_card(self, video_path: str | Path, metadata, sample_timestamps: list[float] | None = None) -> dict:
        """Analyzes 1 or more keyframes from the video to locate the stable master title card."""
        from google.genai import types

        timestamps = sample_timestamps or [1.0, 2.5, 4.0]
        prompt = (
            "Analyze this video keyframe. Detect the on-screen spoken dialogue or subtitle card/pill container.\n"
            "The subtitle/card container can appear anywhere: lower area (subtitles), middle, or top.\n"
            "The container is a graphic overlay added on top of the footage (a pill, bar, or box), NOT part of the physical scene. "
            "NEVER select text printed on a physical object/prop in the scene (e.g. a product bottle, box, tube, or ingredient label "
            "someone is holding or that is on display), speaker name badges, logos, or brand splash screens - set card_detected to "
            "false if no genuine subtitle overlay is visible.\n"
            "CRITICAL: The bounding box MUST encompass the ENTIRE card/pill container from its very top boundary "
            "to its bottom boundary, and full width from left curve/edge to right curve/edge. Do not restrict the box to just the inner text.\n"
            "Return a JSON object with:\n"
            "- card_detected: boolean\n"
            "- box_2d: [ymin, xmin, ymax, xmax] normalized to 1000 (integers between 0 and 1000)\n"
            "- bg_hex_color: hex color of the card background (e.g. #000000 or #1C1C1E)\n"
            "- text_hex_color: hex color of the title text (e.g. #FFFFFF)\n"
            "- description: short description\n"
        )
        gen_config = types.GenerateContentConfig(
            response_mime_type='application/json',
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        for ts in timestamps:
            try:
                img = self.extract_keyframe(video_path, timestamp=ts)
                w, h = img.size

                # Attempt with preferred model, fallback if busy
                response = None
                for model_name in [self.preferred_model, 'gemini-2.5-flash', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']:
                    try:
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=[img, prompt],
                            config=gen_config
                        )
                        if response and response.text:
                            break
                    except Exception as exc:
                        logger.warning('Gemini model %s failed: %s', model_name, exc)
                        continue

                if not response or not response.text:
                    continue

                raw = json.loads(response.text)
                if isinstance(raw, list) and len(raw) > 0:
                    raw = raw[0]
                if not isinstance(raw, dict) or not raw.get('card_detected'):
                    continue

                box = raw.get('box_2d', [])
                if len(box) == 4:
                    ymin, xmin, ymax, xmax = [float(v) for v in box]
                    abs_x = int(xmin / 1000.0 * w)
                    abs_y = int(ymin / 1000.0 * h)
                    abs_w = int((xmax - xmin) / 1000.0 * w)
                    abs_h = int((ymax - ymin) / 1000.0 * h)

                    # Normalize color to 0xRRGGBB format
                    raw_bg = raw.get('bg_hex_color', '#F4D9B9') or '#F4D9B9'
                    clean_hex = raw_bg.lstrip('#')
                    patch_color = f"0x{clean_hex.upper()}"

                    text_color = raw.get('text_hex_color', '#1F1916')
                    if not text_color.startswith('#'):
                        text_color = f"#{text_color.lstrip('0x')}"

                    bbox = BoundingBox(abs_x, abs_y, abs_w, abs_h, patch_color=patch_color)
                    logger.info('Agent detected master title card: %r (bg=%s, text=%s)', bbox, patch_color, text_color)
                    return {
                        'card_detected': True,
                        'bbox': bbox,
                        'bg_hex_color': patch_color,
                        'text_hex_color': text_color,
                        'description': raw.get('description', '')
                    }
            except Exception as exc:
                logger.warning('Agent detection attempt at %ss failed: %s', ts, exc)

        logger.info('Agent did not detect a distinct title card pill.')
        return {'card_detected': False, 'bbox': None, 'bg_hex_color': None, 'text_hex_color': None}

    def detect_subtitles(self, video_path: str | Path, metadata) -> list[SubtitleSegment]:
        """Detects dialogue subtitle cards across the video using video transcript extraction
        combined with batched multimodal vision for exact per-card bounding box tracking.
        """
        from google.genai import types
        import tempfile, time, shutil
        from difflib import SequenceMatcher

        w = getattr(metadata, 'width', 1080) or 1080
        h = getattr(metadata, 'height', 1920) or 1920
        dur = float(getattr(metadata, 'duration', 0.0) or 9999.0)
        ffmpeg_bin = getattr(self.config, 'ffmpeg', 'ffmpeg')
        video_path = Path(video_path)

        # Step 1: Extract all subtitle card events across the full video timeline in 1 API call
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            upload_path = td_path / "proxy_timecode.mp4"
            font_opt = "fontfile='C\\:/Windows/Fonts/arial.ttf':" if Path("C:/Windows/Fonts/arial.ttf").exists() else ""
            vf_filter = f"scale=-2:480,drawtext={font_opt}text='%{{pts\\:hms}}':x=(w-tw)/2:y=20:fontsize=32:fontcolor=yellow:box=1:boxcolor=black@0.9:boxborderw=5"
            cmd = [
                ffmpeg_bin, '-loglevel', 'error', '-y', '-i', str(video_path),
                '-vf', vf_filter, '-c:v', 'libx264',
                '-preset', 'ultrafast', '-crf', '28',
                '-c:a', 'aac', '-b:a', '64k',
                str(upload_path)
            ]
            try:
                subprocess.run(cmd, check=True, timeout=90)
            except Exception as e:
                logger.warning('Failed to render timecode proxy, falling back to clean proxy: %s', e)
                subprocess.run([
                    ffmpeg_bin, '-loglevel', 'error', '-y', '-i', str(video_path),
                    '-vf', 'scale=-2:480', '-c:v', 'libx264',
                    '-preset', 'ultrafast', '-crf', '28',
                    '-c:a', 'aac', '-b:a', '64k',
                    str(upload_path)
                ], check=True, timeout=60)

            upload = self.client.files.upload(file=str(upload_path))
            try:
                while self.client.files.get(name=upload.name).state.name == "PROCESSING":
                    time.sleep(1)

                prompt_video = (
                    "Watch this video carefully. A yellow HH:MM:SS timecode is burned in a black box at the top center.\n"
                    "USE THE BURNED-IN YELLOW TIMECODE TO RECORD EXACT START AND END TIMESTAMPS FOR EVERY ON-SCREEN SUBTITLE CARD.\n"
                    "The video has animated dialogue subtitle cards (pills) that change rapidly as the speaker talks.\n"
                    "CRITICAL RULES:\n"
                    "1. Every visual dialogue text change (typically every 1 to 2.5 seconds, 2 to 5 words) MUST be its own separate card.\n"
                    "   DO NOT merge multiple lines or sentences into one card! Each card corresponds to a separate on-screen pill.\n"
                    "2. Start and end timestamps must be precise numbers in seconds matching the yellow timecode when that specific card appears and disappears.\n"
                    "3. Only extract spoken dialogue cards. DO NOT extract doctor/speaker name tags, permanent logos, or brand splash screens.\n"
                    "4. DO NOT extract text printed on physical objects or props in the scene - e.g. product packaging, bottle/box/tube labels, "
                    "ingredient panels, or any writing on an item someone is holding or that is on display. That is product text, not a dialogue "
                    "subtitle, even if it is readable on screen at the same time as the speaker.\n"
                    "5. When spoken dialogue finishes, STOP. Do NOT create cards for the outro slides or product splash screens.\n"
                    "Return a JSON array of objects with:\n"
                    "  - start: float (seconds)\n"
                    "  - end: float (seconds)\n"
                    "  - text: string (exact words on that card)\n"
                )
                schema = {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "start": {"type": "NUMBER"},
                            "end": {"type": "NUMBER"},
                            "text": {"type": "STRING"}
                        },
                        "required": ["start", "end", "text"]
                    }
                }
                gen_config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
                resp_video = None
                for m_name in [self.preferred_model, 'gemini-2.5-flash', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']:
                    try:
                        resp_video = self.client.models.generate_content(
                            model=m_name,
                            contents=[upload, prompt_video],
                            config=gen_config
                        )
                        if resp_video and resp_video.text:
                            break
                    except Exception as exc:
                        logger.warning('Gemini model %s failed in video detection: %s', m_name, exc)

                clean_json = (resp_video.text if resp_video else '[]').strip()
                raw_cards = json.loads(clean_json)
            finally:
                try:
                    self.client.files.delete(name=upload.name)
                except Exception:
                    pass

            if not isinstance(raw_cards, list) or not raw_cards:
                logger.warning('Video subtitle card extraction returned no cards')
                return []

            # Sample the beginning, middle, and end of each subtitle interval so
            # the resulting mask covers cards that move during the segment.
            card_items = []
            for item in raw_cards:
                s_t = max(0.0, float(item.get('start', 0.0)))
                e_t = min(dur, float(item.get('end', s_t + 1.0)))
                txt = (item.get('text') or '').strip()
                if not txt or e_t <= s_t:
                    continue
                duration = e_t - s_t
                sample_times = sorted({round(s_t + min(duration * fraction, max(0.0, duration - 0.02)), 2)
                                       for fraction in (0.1, 0.5, 0.9)})
                images = []
                for sample_idx, sample_t in enumerate(sample_times):
                    kf_path = td_path / f"kf_{sample_t}_{len(card_items)}_{sample_idx}.jpg"
                    cmd = [
                        ffmpeg_bin, '-loglevel', 'error', '-y', '-ss', str(sample_t),
                        '-i', str(video_path), '-vframes', '1',
                        '-vf', 'scale=-2:720', '-q:v', '3', str(kf_path)
                    ]
                    try:
                        subprocess.run(cmd, check=True, timeout=15)
                        img = Image.open(kf_path)
                        img.load()
                        images.append(img)
                    except Exception as exc:
                        logger.debug('Could not extract keyframe at %.2fs: %s', sample_t, exc)
                card_items.append({'start': s_t, 'end': e_t, 'text': txt, 'images': images})

            if not card_items:
                return []

            # Step 3: Batch image inspection with targeted text grounding
            BATCH_SIZE = 8
            card_boxes = {}
            vision_items = [(card_idx, sample_idx, card['text'], img)
                            for card_idx, card in enumerate(card_items)
                            for sample_idx, img in enumerate(card['images'])]
            for i in range(0, len(vision_items), BATCH_SIZE):
                valid_batch = vision_items[i:i + BATCH_SIZE]
                if not valid_batch:
                    continue
                imgs = [entry[3] for entry in valid_batch]
                image_descriptions = []
                for sub_idx, (card_idx, sample_idx, text, _) in enumerate(valid_batch, 1):
                    image_descriptions.append(f"- Image {sub_idx}: Subtitle card {card_idx + 1}, sample {sample_idx + 1}, dialogue is \"{text}\"")
                img_desc_text = "\n".join(image_descriptions)

                prompt_batch = (
                    f"You are given {len(imgs)} video keyframes in order (Image 1 to Image {len(imgs)}).\n"
                    "For EACH image, find the exact on-screen container/pill displaying the specified spoken dialogue:\n"
                    f"{img_desc_text}\n\n"
                    "CRITICAL RULES:\n"
                    "1. ONLY locate the container/pill displaying the matching dialogue text above. The subtitle container is an overlay graphic "
                    "(a pill, bar, or box added on top of the footage) that is NOT part of the physical scene.\n"
                    "2. NEVER select speaker name badges, diagram labels, brand logos, or text printed on a physical object/prop in the scene "
                    "(e.g. a product bottle, box, tube, or ingredient label someone is holding or that is on display) even if that text happens "
                    "to overlap or share words with the dialogue above. Product packaging is part of the filmed scene, not a subtitle overlay.\n"
                    "3. If you cannot find a genuine subtitle overlay matching the dialogue text in an image, set card_detected to false for it "
                    "instead of guessing at the nearest text.\n"
                    "4. If the dialogue subtitle is at the bottom, return the bottom box. If it moved up into the middle or top, return the middle/top box where that dialogue appears.\n"
                    f"Return a JSON array of {len(imgs)} objects in exact matching order:\n"
                    "- image_index: integer (1 to {len(imgs)})\n"
                    "- card_detected: boolean (true if the matching dialogue subtitle card is visible)\n"
                    "- box_2d: [ymin, xmin, ymax, xmax] normalized 0-1000 (outer boundary of the subtitle container)\n"
                    "- bg_hex_color: hex color of the subtitle container\n"
                )
                try:
                    gen_cfg = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                    resp_b = None
                    for m_name in [self.preferred_model, 'gemini-2.5-flash', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']:
                        try:
                            resp_b = self.client.models.generate_content(
                                model=m_name,
                                contents=[prompt_batch, *imgs],
                                config=gen_cfg
                            )
                            if resp_b and resp_b.text:
                                break
                        except Exception as m_exc:
                            logger.warning('Model %s failed in batch keyframe detection: %s', m_name, m_exc)
                    b_res = json.loads(resp_b.text or '[]') if resp_b else []
                    if isinstance(b_res, list):
                        for res_item, (card_idx, _, _, _) in zip(b_res, valid_batch):
                            card_boxes.setdefault(card_idx, []).append(res_item)
                except Exception as exc:
                    logger.warning('Batch card detection failed for batch %d: %s', i // BATCH_SIZE, exc)
                time.sleep(0.5)

            # Step 4: Construct SubtitleSegments with verified per-card bounding boxes
            segments = []
            for idx, item in enumerate(card_items):
                boxes = []
                colors = []
                for box_info in card_boxes.get(idx, []):
                    box = box_info.get('box_2d', [])
                    if isinstance(box, list) and len(box) > 0 and isinstance(box[0], list):
                        box = box[0]
                    if box_info.get('card_detected') and isinstance(box, list) and len(box) == 4:
                        ymin, xmin, ymax, xmax = [float(v) for v in box]
                        left = max(0, int(xmin / 1000.0 * w) - 20)
                        top = max(0, int(ymin / 1000.0 * h) - 5)
                        right = min(w, int(xmax / 1000.0 * w) + 20)
                        bottom = min(h, int(ymax / 1000.0 * h) + 5)
                        if right > left and bottom > top:
                            boxes.append((left, top, right, bottom))
                            colors.append((box_info.get('bg_hex_color') or '').strip().lstrip('#'))

                if boxes:
                    abs_x = min(b[0] for b in boxes)
                    abs_y = min(b[1] for b in boxes)
                    abs_right = max(b[2] for b in boxes)
                    abs_bottom = max(b[3] for b in boxes)
                    abs_w, abs_h = abs_right - abs_x, abs_bottom - abs_y
                    raw_bg = next((c for c in colors if len(c) == 6), '')
                    if len(raw_bg) == 6:
                        try:
                            r, g, b = int(raw_bg[0:2], 16), int(raw_bg[2:4], 16), int(raw_bg[4:6], 16)
                            lum = 0.299 * r + 0.587 * g + 0.114 * b
                            if lum > 130:
                                patch_color = getattr(self.config, 'overlay_patch_color', '0x1F1F1F')
                            else:
                                patch_color = f"0x{raw_bg.upper()}"
                        except Exception:
                            patch_color = getattr(self.config, 'overlay_patch_color', '0x000000')
                    else:
                        patch_color = getattr(self.config, 'overlay_patch_color', '0x000000')
                    bbox = BoundingBox(abs_x, abs_y, abs_w, abs_h, patch_color=patch_color)
                else:
                    logger.warning('No subtitle-card box was confirmed for segment %d (%r); skipping this segment only', idx + 1, item['text'])
                    continue

                segments.append(SubtitleSegment(
                    len(segments) + 1,
                    item['start'],
                    item['end'],
                    item['text'],
                    bbox=bbox,
                    ocr_confidence=1.0
                ))

            # Step 5: Post-processing to ensure seamless coverage without gaps
            if segments and segments[0].start <= 0.8:
                segments[0].start = 0.0

            # Bridge gaps between consecutive segments so the background mask never flickers off
            # mid-sentence, including gaps left by a segment that was skipped in Step 4 for
            # lacking a confirmed box - better to hold the previous patch a bit longer than to
            # expose the original un-translated source text underneath.
            for i in range(len(segments) - 1):
                gap = segments[i + 1].start - segments[i].end
                if gap <= 3.0:
                    segments[i].end = segments[i + 1].start

            return [s for s in segments if s.end > s.start + 0.15]
