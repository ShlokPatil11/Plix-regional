from __future__ import annotations

import json
import importlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

import cv2
import requests
from PIL import Image

from .contracts import Artifact
from .magnific_oauth import MagnificAuth, MagnificMcp
from .service_adapters import LANGUAGES


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "Media command failed")


def _probe_duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _slots(duration: float) -> list[float]:
    count = max(1, math.ceil(duration / 10.0))
    while count > 1 and duration / count < 5.0:
        count -= 1
    return [duration / count] * count


def _creation_media_url(value) -> str | None:
    """Find a downloadable media URL in any supported Magnific response shape."""
    preferred = ("url", "outputUrl", "videoUrl", "downloadUrl", "mediaUrl")
    if isinstance(value, dict):
        for key in preferred:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.startswith(("https://", "http://")):
                return candidate
        for child in value.values():
            candidate = _creation_media_url(child)
            if candidate:
                return candidate
    elif isinstance(value, list):
        for child in value:
            candidate = _creation_media_url(child)
            if candidate:
                return candidate
    return None


def _response_shape(value) -> str:
    """Return response keys only, keeping errors useful without logging signed URLs."""
    if isinstance(value, dict):
        return "{" + ", ".join(sorted(str(key) for key in value)) + "}"
    return type(value).__name__


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _detect_visual_cues_with_openai(source: Path, duration: float) -> list[dict]:
    """Read burned overlay text from video pixels; never infer it from audio."""
    service_root = Path(__file__).resolve().parent / "vendor" / "service_a"
    root_string = str(service_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    detector_module = importlib.import_module("pipeline.openai_card_detector")
    detector = detector_module.OpenAICardDetector(SimpleNamespace(ffmpeg="ffmpeg"))
    segments = detector.detect_subtitles(source, SimpleNamespace(duration=duration))
    capture = cv2.VideoCapture(str(source))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    cues = []
    for segment in segments:
        cue = {"start": float(segment.start), "end": float(segment.end),
               "text": str(segment.source_text).strip()}
        if segment.bbox:
            cue["bbox"] = {"x": segment.bbox.x, "y": segment.bbox.y,
                           "width": segment.bbox.width, "height": segment.bbox.height,
                           "frame_width": frame_width, "frame_height": frame_height,
                           "patch_color": segment.bbox.patch_color or "0x000000"}
        if cue["text"]:
            cues.append(cue)
    if not cues:
        raise RuntimeError("GPT vision found no visible overlay text to translate")
    return cues


def _translate_cues_with_openai(cues: list[dict], target_language: str) -> list[dict]:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for translation")
    target = LANGUAGES.get(target_language, target_language)
    indexed = [{"index": index, "text": cue["text"]} for index, cue in enumerate(cues)]
    schema = {"name": "translated_cues", "strict": True, "schema": {
        "type": "object", "properties": {"translations": {"type": "array", "items": {
            "type": "object", "properties": {"index": {"type": "integer"}, "text": {"type": "string"}},
            "required": ["index", "text"], "additionalProperties": False}}},
        "required": ["translations"], "additionalProperties": False}}
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": (
                    f"Translate each subtitle into simple, natural {target}. Keep the meaning, product claims, "
                    "and brand names Plix and ACV. Read the complete ordered cue list as one visual-ad sequence "
                    "before translating individual cards. Use scene and neighboring-card context, not isolated literal "
                    "word meanings. In transformations or comparisons, translate paired temporal labels as the natural "
                    "'Before' and 'After' equivalents in the target language—not 'First', 'Earlier', 'Later', or 'Now'—"
                    "unless the surrounding sequence clearly has another meaning. Do not add explanations. Preserve every index.")},
                {"role": "user", "content": json.dumps(indexed, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "temperature": 0.2,
        },
        timeout=120,
    )
    response.raise_for_status()
    translated = json.loads(response.json()["choices"][0]["message"]["content"])["translations"]
    by_index = {int(item["index"]): str(item["text"]).strip() for item in translated}
    if len(by_index) != len(cues):
        raise RuntimeError("OpenAI translation omitted one or more subtitle cues")
    return [{**cue, "text": by_index[index]} for index, cue in enumerate(cues)]


def _font_family(font_path: Path) -> str:
    stem = font_path.stem
    for marker, family in (
        ("Devanagari", "Noto Sans Devanagari"),
        ("Tamil", "Noto Sans Tamil"),
        ("Malayalam", "Noto Sans Malayalam"),
        ("Kannada", "Noto Sans Kannada"),
        ("Bengali", "Noto Sans Bengali"),
        ("Telugu", "Noto Sans Telugu"),
        ("Arabic", "Noto Sans Arabic"),
        ("JP", "Noto Sans JP"),
    ):
        if marker in stem:
            return family
    return "Noto Sans"


def _pango_overlays(cues: list[dict], width: int, font_path: Path, root: Path) -> list[Image.Image]:
    pango = shutil.which("pango-view")
    if not pango:
        raise RuntimeError("pango-view is required for correctly shaped multilingual captions")
    fonts = font_path.parent.resolve()
    cache = root / "font-cache"
    cache.mkdir(parents=True, exist_ok=True)
    config = root / "fonts.conf"
    config.write_text(
        "<?xml version='1.0'?><!DOCTYPE fontconfig SYSTEM 'urn:fontconfig:fonts.dtd'>"
        f"<fontconfig><dir>{escape(str(fonts))}</dir><cachedir>{escape(str(cache))}</cachedir>"
        "<config></config></fontconfig>", encoding="utf-8")
    env = dict(os.environ)
    env.update({"FONTCONFIG_FILE": str(config), "XDG_CACHE_HOME": str(cache)})
    family = _font_family(font_path)
    overlays = []
    for index, cue in enumerate(cues):
        path = root / f"caption-{index:04d}.png"
        bbox = cue.get("bbox") or {}
        source_width = max(1, int(bbox.get("frame_width") or width))
        detected_width = int(bbox.get("width") or round(width * 0.86))
        label_width = min(round(width * 0.86), max(160, round(detected_width * width / source_width * 1.15)))
        command = [
            pango, "--no-display", "--backend=cairo", "--pixels",
            "--background=#000000ff", "--foreground=#ffffffff",
            "--align=center", "--wrap=word-char", f"--width={label_width}",
            "--margin", os.getenv("CAPTION_CARD_PADDING_PX", "10"),
            "--spacing=4", "--font", f"{family} SemiBold 38",
            "--text", str(cue["text"]), "--output", str(path),
        ]
        completed = subprocess.run(command, env=env, capture_output=True, text=True)
        if completed.returncode or not path.is_file():
            raise RuntimeError(completed.stderr.strip() or "Pango could not render a caption")
        overlays.append(Image.open(path).convert("RGBA"))
    return overlays


def _scaled_mask_box(bbox: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """Scale a detected source-card rectangle to the render frame and clamp it."""
    source_width = max(1, int(bbox.get("frame_width") or width))
    source_height = max(1, int(bbox.get("frame_height") or height))
    x1 = round(float(bbox.get("x", 0)) * width / source_width)
    y1 = round(float(bbox.get("y", 0)) * height / source_height)
    x2 = round((float(bbox.get("x", 0)) + float(bbox.get("width", 0))) * width / source_width)
    y2 = round((float(bbox.get("y", 0)) + float(bbox.get("height", 0))) * height / source_height)
    return (max(0, min(width, x1)), max(0, min(height, y1)),
            max(0, min(width, x2)), max(0, min(height, y2)))


def _patch_color(value: str) -> str:
    raw = str(value or "0x000000").strip()
    if raw.lower().startswith("0x"):
        raw = "#" + raw[2:]
    return raw if raw.startswith("#") and len(raw) in (4, 7, 9) else "#000000"


def _render_captions(video: Path, cues: list[dict], output: Path, font_path: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    silent = output.with_name("captioned-video-only.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    with tempfile.TemporaryDirectory(prefix="caption-overlays-", dir=output.parent) as temporary:
        overlays = _pango_overlays(cues, width, font_path, Path(temporary))
        safe_bottom = round(height * float(os.getenv("CAPTION_BOTTOM_SAFE_RATIO", "0.17")))
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            now = frame_number / fps
            active = [index for index, cue in enumerate(cues)
                      if float(cue["start"]) <= now < float(cue["end"])]
            if active:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
                for cue_index in active:
                    cue = cues[cue_index]
                    label = overlays[cue_index]
                    bbox = cue.get("bbox")
                    if bbox and bbox.get("frame_width") and bbox.get("frame_height"):
                        # Cover the complete detected source card first. Do not
                        # depend on the translated label's dimensions to hide it.
                        from PIL import ImageDraw
                        mask_box = _scaled_mask_box(bbox, width, height)
                        if mask_box[2] > mask_box[0] and mask_box[3] > mask_box[1]:
                            ImageDraw.Draw(image).rectangle(
                                mask_box, fill=_patch_color(bbox.get("patch_color")))
                        center_x = (mask_box[0] + mask_box[2]) / 2
                        center_y = (mask_box[1] + mask_box[3]) / 2
                        x = round(center_x - label.width / 2)
                        y = round(center_y - label.height / 2)
                    else:
                        x = (width - label.width) // 2
                        y = height - safe_bottom - label.height
                    x = max(0, min(width - label.width, x))
                    y = max(0, min(height - label.height, y))
                    image.alpha_composite(label, (x, y))
                frame = cv2.cvtColor(__import__("numpy").array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            writer.write(frame)
            frame_number += 1
    capture.release(); writer.release()
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(silent),
          "-i", str(video), "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium",
          "-crf", "18", "-c:a", "copy", "-movflags", "+faststart", "-shortest", str(output)])


def run_clean_translate(source: Path, target_language: str, work: Path, auth: MagnificAuth, on_stage=None) -> dict[str, Path]:
    work.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(source)
    slots = _slots(duration)
    if on_stage: on_stage("detecting_visual_text_with_openai")
    source_cues = _detect_visual_cues_with_openai(source, duration)
    if on_stage: on_stage("translating_with_openai")
    cues = _translate_cues_with_openai(source_cues, target_language)
    subtitle_path = work / "translated-captions.srt"
    subtitle_path.write_text("\n\n".join(
        f"{index}\n{_srt_time(float(cue['start']))} --> {_srt_time(float(cue['end']))}\n{cue['text']}"
        for index, cue in enumerate(cues, 1)) + "\n", encoding="utf-8")

    if on_stage: on_stage("splitting_for_wan_3")
    chunks = []
    offset = 0.0
    for index, length in enumerate(slots, 1):
        path = work / f"source-{index:03d}.mp4"
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-ss", f"{offset:.6f}",
              "-i", str(source), "-t", f"{length:.6f}", "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
              "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(path)])
        chunks.append(path); offset += length

    if on_stage: on_stage("wan_3_removing_captions")
    mcp = MagnificMcp(auth.access_token())
    upload = mcp.call("creations_request_upload", {"count": len(chunks), "mimeType": "video/mp4"})
    targets = upload.get("uploads") or [upload]
    for path, target in zip(chunks, targets):
        with path.open("rb") as stream:
            response = requests.put(target["proxyUploadUrl"], data=stream, headers={"Content-Type": "video/mp4"}, timeout=300)
            response.raise_for_status()
    finalized = mcp.call("creations_finalize_upload", {"uploads": [
        {"path": target["path"], "fileName": path.name} for path, target in zip(chunks, targets)], "visible": False})
    source_ids = [item["identifier"] for item in finalized["results"]]
    prompt = ("Preserve the supplied reference video exactly: identity, face, clothing, product, background, framing, "
              "motion, gestures, lighting, colors, timing, and camera. Remove only burned dialogue subtitles/captions "
              "and their caption backgrounds. Reconstruct pixels naturally. Preserve product labels, logos, and end cards. "
              "Do not add text, objects, transitions, or camera motion.")
    generation_state = work / "wan-generations.json"
    generated = []
    try:
        saved = json.loads(generation_state.read_text(encoding="utf-8"))
        if saved.get("slots") == slots:
            generated = [(item["identifier"], item["length"]) for item in saved["generations"]]
            print(f"[WAN] Resuming {len(generated)} previously submitted generation(s)")
    except (OSError, ValueError, KeyError, TypeError):
        generated = []
    if not generated:
        for source_id, length in zip(source_ids, slots):
            result = mcp.call("video_generate", {"slug": "wan-3-0", "prompt": prompt,
                "duration": max(5, min(10, round(length))), "aspectRatio": "9:16", "resolution": "720p",
                "references": [{"type": "video", "url": source_id}], "withSoundEffects": False})
            generated.append((result["creations"][0]["identifier"], length))
        generation_state.write_text(json.dumps({
            "slots": slots,
            "generations": [
                {"identifier": identifier, "length": length} for identifier, length in generated
            ],
        }, indent=2), encoding="utf-8")
    pending = {identifier for identifier, _ in generated}
    urls = {}
    completed_without_url = {identifier: 0 for identifier in pending}
    while pending:
        status = mcp.call("creations_wait", {"identifiers": list(pending), "timeoutSeconds": 25})
        for item in status["results"]:
            if item["status"] == "completed":
                details = mcp.call("creations_get", {"creationIdentifier": item["identifier"]})
                media_url = _creation_media_url(details) or _creation_media_url(item)
                if media_url:
                    urls[item["identifier"]] = media_url
                    pending.remove(item["identifier"])
                else:
                    completed_without_url[item["identifier"]] += 1
                    attempts = completed_without_url[item["identifier"]]
                    print(f"[WAN] Creation completed but media is not ready yet "
                          f"(attempt {attempts}, response {_response_shape(details)})")
                    if attempts >= 12:
                        raise RuntimeError(
                            "Magnific marked a WAN creation completed but returned no downloadable "
                            f"media after {attempts} checks; response shape {_response_shape(details)}"
                        )
            elif item["status"] in {"failed", "error"}:
                raise RuntimeError(item.get("failureReason") or "Wan 3.0 generation failed")
        if pending:
            time.sleep(5)

    processed = []
    for index, (identifier, length) in enumerate(generated, 1):
        path = work / f"clean-{index:03d}.mp4"
        if not path.is_file() or path.stat().st_size == 0:
            response = requests.get(urls[identifier], timeout=300)
            response.raise_for_status(); path.write_bytes(response.content)
        else:
            print(f"[WAN] Reusing downloaded part {index}: {path.name}")
        processed.append((path, length))
    clean = work / "clean-stitched-720p.mp4"
    inputs, filters = [], []
    for index, (path, length) in enumerate(processed):
        inputs += ["-i", str(path)]
        filters.append(f"[{index}:v]setpts={length / _probe_duration(path):.9f}*PTS,fps=30,format=yuv420p[v{index}]")
    filters.append("".join(f"[v{i}]" for i in range(len(processed))) + f"concat=n={len(processed)}:v=1:a=0[v]")
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *inputs, "-i", str(source),
          "-filter_complex", ";".join(filters), "-map", "[v]", "-map", f"{len(processed)}:a?", "-c:v", "libx264",
          "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.6f}", str(clean)])
    if on_stage: on_stage("rendering_translated_captions")
    final = work / "final-clean-translated-720p.mp4"
    font_names = {
        "hi": "NotoSansDevanagari-SemiBold.ttf", "mr": "NotoSansDevanagari-SemiBold.ttf",
        "ta": "NotoSansTamil-SemiBold.ttf", "ml": "NotoSansMalayalam-SemiBold.ttf",
        "kn": "NotoSansKannada-SemiBold.ttf", "bn": "NotoSansBengali-SemiBold.ttf",
        "te": "NotoSansTelugu-SemiBold.ttf", "ar": "NotoSansArabic-SemiBold.ttf",
        "ja": "NotoSansJP-SemiBold.ttf",
    }
    language_font = Path(__file__).resolve().parent / "vendor/service_a/fonts" / font_names.get(
        target_language, "NotoSans-SemiBold.ttf")
    _render_captions(clean, cues, final, language_font)
    return {"final_video": final, "clean_video": clean, "translated_captions": subtitle_path}
