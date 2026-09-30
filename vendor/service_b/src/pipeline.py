import contextlib
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import uuid

from .common import PipelineError, digest, read_json, write_json
from .srt_parser import parse_srt, render_srt
from .translation import LibreTranslator, language
from .input_formats import split_bilingual, pair_translated_srt, split_labeled_bilingual
from .elevenlabs_client import ElevenLabs
from .audio import convert, fit, assemble, export_mp3, ffmpeg


@contextlib.contextmanager
def job_lock(job):
    lock = job / ".lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise PipelineError("Job is already running. If a previous process crashed, remove its .lock file after confirming it stopped.") from None
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def prepare(source_file, output_root, source, target, provider="none",
            translations=None, locale=None, *, bilingual=False,
            source_lines=1, source_first=True, translated_srt=None,
            source_label=None, target_label=None):
    source_file, output_root = Path(source_file).resolve(), Path(output_root).resolve()
    # Outputs must not be written inside the directory containing source inputs.
    if output_root == source_file.parent or source_file.parent in output_root.parents:
        raise PipelineError("Use separate input and output directories.")
    if sum((bool(translations), bool(bilingual), bool(translated_srt))) > 1:
        raise PipelineError("Choose one translation input: bilingual SRT, translated SRT, or translation JSON.")
    raw_source = source_file.read_text(encoding="utf-8-sig")
    original_cues = parse_srt(raw_source)
    cues = original_cues
    supplied, translated_raw = None, None
    input_format = "single_language"
    if (source_label is None) != (target_label is None) or ((source_label is not None or target_label is not None) and not bilingual):
        raise PipelineError("Specify both labels together with bilingual mode.")
    if bilingual:
        if source_label is not None and target_label is not None:
            cues, supplied = split_labeled_bilingual(raw_source, source_label, target_label)
            input_format = "labeled_bilingual_srt"
        else:
            cues, supplied = split_bilingual(raw_source, source_lines, source_first)
            input_format = "bilingual_srt"
    elif translated_srt:
        translated_path = Path(translated_srt).resolve()
        if output_root == translated_path.parent or translated_path.parent in output_root.parents:
            raise PipelineError("Use separate input and output directories.")
        supplied, translated_raw = pair_translated_srt(cues, translated_path)
        input_format = "paired_srt"
    elif translations:
        supplied = read_json(translations)
        if supplied is None:
            raise PipelineError("Translation JSON must contain an object mapping cue IDs to text.")
        input_format = "translation_json"
    target = language(target)
    source = source.strip().lower()
    if supplied is not None and source == "auto":
        raise PipelineError("Specify the source language explicitly for supplied translations; no detection service is needed.")
    translator = LibreTranslator() if source == "auto" or (provider == "libre" and source != target and supplied is None) else None
    if source == "auto":
        source = translator.detect(cues)
    source = language(source)
    if supplied is not None:
        if not isinstance(supplied, dict) or set(supplied) != {str(c.id) for c in cues}:
            raise PipelineError("Translation JSON must map every source cue ID to target text, with no extra IDs.")
    elif source != target:
        if not translator:
            raise PipelineError("Provide --bilingual, --translated-srt, --translations, or --provider libre for different source/target languages.")
        if locale or "-" in source or "-" in target:
            raise PipelineError("Use base language codes with LibreTranslate. Supply reviewed translations for locale adaptation.")
        translator.validate(source, target)
    elif locale:
        raise PipelineError("Locale adaptation requires a reviewed translation JSON.")

    job = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    job.mkdir(parents=True)
    (job / "segments").mkdir()
    (job / "source.srt").write_text(raw_source, encoding="utf-8")
    if translated_raw is not None:
        (job / "translated_input.srt").write_text(translated_raw, encoding="utf-8")
    manifest = {
        "version": 1, "status": "preparing", "source_language": source, "target_language": target,
        "locale": locale, "translation_provider": "file" if supplied is not None else provider,
        "source_sha256": digest([c.to_dict() for c in original_cues]),
        "input_format": input_format,
        "bilingual_layout": {"source_lines": source_lines, "source_first": source_first,
                             "source_label": source_label, "target_label": target_label} if bilingual else None,
        "segments": [], "exports": {}, "usage": {"speech_requests": 0, "submitted_characters": 0, "cache_hits": 0},
        "notes": ["Review translation, names, numbers, non-speech captions and pronunciation before final use.",
                  "One cue per speaking segment preserves original windows.",
                  "Existing translated text is used directly; only target text is synthesized." if supplied is not None else
                  "Full surrounding source context is shown for human review."]
    }
    write_json(job / "manifest.json", manifest)
    try:
        for index, cue in enumerate(cues):
            context = [c.text for c in cues[max(0, index - 1):index + 2]]
            if supplied is not None:
                text = supplied[str(cue.id)]
            elif source == target:
                text = cue.text
            else:
                text = translator.translate(cue.text, source, target, context=context)
            if not isinstance(text, str) or not text.strip():
                raise PipelineError(f"Invalid translation for cue {cue.id}.")
            manifest["segments"].append({
                "id": cue.id, "source_cue_ids": [cue.id], "start_ms": cue.start_ms,
                "end_ms": cue.end_ms, "available_ms": cue.end_ms - cue.start_ms,
                "source_text": cue.text, "context": context, "target_text": text.strip(),
                "skip": False, "revision": 0, "status": "pending",
                "non_speech_candidate": bool(re.fullmatch(r"\[[^\]]+\]|\([^)]+\)|[♪♫].*", cue.text)),
            })
            write_json(job / "manifest.json", manifest)
        manifest["status"] = "prepared"
    except Exception:
        manifest["status"] = "translation_failed"
        write_json(job / "manifest.json", manifest)
        raise
    write_json(job / "manifest.json", manifest)
    return job


def validate_manifest(job, manifest):
    if manifest.get("version") != 1 or manifest.get("status") in ("preparing", "translation_failed"):
        raise PipelineError("Job preparation is incomplete or version is unsupported.")
    cues = parse_srt((job / "source.srt").read_text(encoding="utf-8-sig"))
    if digest([c.to_dict() for c in cues]) != manifest.get("source_sha256"):
        raise PipelineError("Source SRT changed; prepare a new job.")
    segments = manifest.get("segments", [])
    if len(segments) != len(cues):
        raise PipelineError("Segment count changed; prepare a new job.")
    for cue, segment in zip(cues, segments):
        if (segment.get("id"), segment.get("start_ms"), segment.get("end_ms")) != (cue.id, cue.start_ms, cue.end_ms):
            raise PipelineError("Cue IDs/timing changed; prepare a new job.")
        if not isinstance(segment.get("skip"), bool):
            raise PipelineError("skip must be true or false.")
        if not isinstance(segment.get("target_text"), str) or not segment["target_text"].strip():
            raise PipelineError(f"Cue {cue.id} requires target text, even when skipped.")
        if type(segment.get("revision")) is not int or segment["revision"] < 0:
            raise PipelineError("revision must be a non-negative integer.")
        segment["available_ms"] = cue.end_ms - cue.start_ms
    return segments


def render(job, voice, model, max_speed=1.10, mp3=False, progress=None, client=None):
    job = Path(job).resolve()
    if not math.isfinite(max_speed) or not 1 <= max_speed <= 1.25:
        raise PipelineError("Maximum speed must be between 1.0 and 1.25.")
    ffmpeg()  # Fail before paid generation if audio tools are unavailable.
    with job_lock(job):
        manifest = read_json(job / "manifest.json")
        segments = validate_manifest(job, manifest)
        target = language(manifest["target_language"])
        speech = client or ElevenLabs(model, voice, target)
        speech.preflight()
        manifest.update(status="generating", exports={}, voice_id=voice, model_id=model, max_speed=max_speed)
        manifest.pop("error", None)
        cache = job.parent / ".cache" / "speech"
        cache.mkdir(parents=True, exist_ok=True)
        report = []
        write_json(job / "manifest.json", manifest)
        for index, segment in enumerate(segments):
            for field in ("error", "audio", "original_ms", "duration_ms", "overflow_ms", "speed", "fit_ratio"):
                segment.pop(field, None)
            if segment["skip"]:
                segment["status"] = "skipped"
            else:
                try:
                    audio, alignment, hit = speech.generate(segment["target_text"], cache, segment["revision"])
                    usage = manifest["usage"]
                    usage["cache_hits"] += int(hit)
                    usage["speech_requests"] += int(not hit)
                    usage["submitted_characters"] += 0 if hit else len(segment["target_text"])
                    folder = job / "segments" / str(segment["id"])
                    folder.mkdir(parents=True, exist_ok=True)
                    raw, decoded, adjusted = folder / "original.mp3", folder / "decoded.wav", folder / "fitted.wav"
                    raw.write_bytes(audio)
                    write_json(folder / "alignment.json", alignment)
                    convert(raw, decoded)
                    segment.update(fit(decoded, adjusted, segment["available_ms"], max_speed))
                    segment["fit_ratio"] = segment["duration_ms"] / segment["available_ms"]
                    segment["listening_review_required"] = True
                except (PipelineError, OSError, ValueError) as error:
                    segment.update(status="failed", error=str(error))
            report.append({k: segment[k] for k in (
                "id", "start_ms", "end_ms", "available_ms", "status", "original_ms",
                "duration_ms", "overflow_ms", "speed", "fit_ratio", "error") if k in segment})
            write_json(job / "timing_report.json", report)
            write_json(job / "manifest.json", manifest)
            if progress:
                progress(index + 1, len(segments), segment["status"])
        if any(s["status"] in ("failed", "needs_review") for s in segments):
            manifest["status"] = "needs_review"
            write_json(job / "manifest.json", manifest)
            return manifest
        # Unique export paths ensure stale output cannot masquerade as the current render.
        export_dir = job / "exports" / uuid.uuid4().hex[:12]
        export_dir.mkdir(parents=True)
        wav = export_dir / "output.wav"
        try:
            duration = assemble(segments, wav, segments[-1]["end_ms"])
            subtitles = export_dir / "output.srt"
            subtitles.write_text(render_srt(segments), encoding="utf-8")
            manifest["exports"] = {"wav": str(wav), "srt": str(subtitles)}
            manifest["duration_ms"] = duration
            if mp3:
                mp3_path = export_dir / "output.mp3"
                manifest["mp3_duration_difference_ms"] = export_mp3(wav, mp3_path)
                manifest["exports"]["mp3"] = str(mp3_path)
            manifest["status"] = "ready_for_listening_review"
        except (PipelineError, OSError) as error:
            manifest["status"] = "export_failed"
            manifest["error"] = str(error)
        write_json(job / "manifest.json", manifest)
        return manifest


def edit_segment(job, cue_id, text=None, skip=None, regenerate=False):
    job = Path(job).resolve()
    with job_lock(job):
        manifest = read_json(job / "manifest.json")
        segments = validate_manifest(job, manifest)
        segment = next((s for s in segments if s["id"] == cue_id), None)
        if segment is None:
            raise PipelineError("Unknown cue ID.")
        if text is not None:
            if not text.strip():
                raise PipelineError("Target text cannot be empty.")
            segment["target_text"] = text.strip()
        if skip is not None:
            segment["skip"] = bool(skip)
        if regenerate:
            segment["revision"] += 1
        segment["status"] = "pending"
        manifest.update(status="prepared", exports={})
        write_json(job / "manifest.json", manifest)
