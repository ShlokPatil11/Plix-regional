from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
from pathlib import Path

from .contracts import Artifact, PipelineResult
from .media import FfmpegFinalAssembler
from .mixer import DEFAULT_MUSIC_PERCENT, MixError, mix_final
from .runner import DirectoryLanguagePipeline, PipelineServices
from .service_adapters import LocalSubtitleTranslatorAdapter, TextToAudioAdapter, analyze_masking_music
from .clean_translate import _render_captions, run_clean_translate
from .magnific_oauth import MagnificAuth


def _mix_dubbed_only(backend, voice: Artifact, music: Artifact, job_id: str) -> Artifact:
    """Dubbing-only has no subtitle track; mix the voice video with the music."""
    with tempfile.TemporaryDirectory(prefix=f"dub-mix-{job_id}-") as temp:
        root = Path(temp)
        voice_path = backend.download(voice, root / "voice.mp4")
        music_path = backend.download(music, root / "music.mp3")
        try:
            mixed = mix_final(voice_path, music_path, DEFAULT_MUSIC_PERCENT, root / "dubbed.mp4")
        except MixError as exc:
            print(f"[Mix] WARNING: music mix failed ({exc}); continuing voice-only.")
            mixed = mix_final(voice_path, None, DEFAULT_MUSIC_PERCENT, root / "dubbed.mp4")
        artifact = Artifact(uri="artifact://dubbed-mixed.mp4", filename="dubbed-mixed.mp4",
                            content_type="video/mp4", metadata={"path": str(mixed)})
        return backend.publish(artifact, name="dubbed_video.mp4", job_id=job_id)


def _run_clean_stage(input_artifact, target_language, backend, job_id, on_stage=None):
    data_dir = Path(__import__("os").getenv(
        "PIPELINE_DATA_DIR", str(Path(tempfile.gettempdir()) / "dir-language-pipeline"))).resolve()
    source = Path(str(input_artifact.metadata.get("path", "")))
    if not source.is_file():
        source = backend.download(input_artifact, Path(backend.root) / "clean-source.mp4")
    outputs = run_clean_translate(
        source, target_language, Path(backend.root) / "clean-work",
        MagnificAuth(data_dir / "magnific-oauth.json"), on_stage,
    )
    return {
        name: backend.upload(path, filename=path.name,
                             content_type="video/mp4" if path.suffix == ".mp4" else "application/x-subrip")
        for name, path in outputs.items()
    }


def _run_transcript_caption_stage(
    input_artifact, target_language, source_language, backend, job_id, on_stage=None
):
    """Transcribe speech, translate it, and burn timed captions onto an untouched video."""
    if on_stage:
        on_stage("transcribing_audio")
    service_root = Path(__file__).resolve().parent / "vendor" / "service_b"
    root_value = str(service_root)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)
    transcriber = importlib.import_module("src.transcriber")
    parser = importlib.import_module("src.srt_parser")
    dubber = importlib.import_module("src.el_dubber")

    work = Path(backend.root) / "transcript-caption-work"
    work.mkdir(parents=True, exist_ok=True)
    source = Path(str(input_artifact.metadata.get("path", "")))
    if not source.is_file():
        source = backend.download(input_artifact, work / input_artifact.filename)
    transcript_path, _ = transcriber.extract_transcript(
        source, work, source_lang=source_language
    )
    source_cues = parser.parse_srt(transcript_path.read_text(encoding="utf-8"))
    if not source_cues:
        raise RuntimeError("No speech was found to create captions.")

    if on_stage:
        on_stage("translating_transcript")
    translated = dubber._translate_cues_gemini(
        source_cues, target_language, source_language
    )
    captions = work / "translated-transcript.srt"
    captions.write_text(
        "\n\n".join(
            f"{index}\n{parser.stamp(cue.start_ms)} --> {parser.stamp(cue.end_ms)}\n{cue.text}"
            for index, cue in enumerate(translated, 1)
        ) + "\n",
        encoding="utf-8",
    )

    if on_stage:
        on_stage("rendering_transcript_captions")
    staged_source = work / f"source{source.suffix.lower() or '.mp4'}"
    if source.resolve() != staged_source.resolve():
        shutil.copy2(source, staged_source)
    output = work / "captioned-video.mp4"
    font_names = {
        "hi": "NotoSansDevanagari-SemiBold.ttf", "mr": "NotoSansDevanagari-SemiBold.ttf",
        "ta": "NotoSansTamil-SemiBold.ttf", "ml": "NotoSansMalayalam-SemiBold.ttf",
        "kn": "NotoSansKannada-SemiBold.ttf", "te": "NotoSansTelugu-SemiBold.ttf",
        "bn": "NotoSansBengali-SemiBold.ttf", "ar": "NotoSansArabic-SemiBold.ttf",
        "ur": "NotoSansArabic-SemiBold.ttf", "ja": "NotoSansJP-SemiBold.ttf",
    }
    fonts = Path(__file__).resolve().parent / "vendor" / "service_a" / "fonts"
    font = fonts / font_names.get(target_language.lower(), "NotoSans-SemiBold.ttf")
    if not font.is_file():
        font = fonts / "NotoSans-Regular.ttf"
    render_cues = [
        {"start": cue.start_ms / 1000.0, "end": cue.end_ms / 1000.0, "text": cue.text}
        for cue in translated
    ]
    _render_captions(staged_source, render_cues, output, font)
    return {
        "final_video": backend.upload(output, filename="captioned-video.mp4", content_type="video/mp4"),
        "translated_captions": backend.upload(
            captions, filename="translated-captions.srt", content_type="application/x-subrip"
        ),
    }


def run_artifact(
    input_artifact: Artifact,
    *,
    target_language: str,
    backend,
    source_language: str = "auto",
    job_id: str | None = None,
    on_stage=None,
    face_choice: str | None = None,
) -> PipelineResult:
    """Run both embedded services from an artifact provided by the host."""
    services = PipelineServices(
        visual_subtitles=LocalSubtitleTranslatorAdapter(backend),
        audio_dubbing=TextToAudioAdapter(backend, face_choice=face_choice),
        assembler=FfmpegFinalAssembler(backend),
        publisher=backend,
    )
    return DirectoryLanguagePipeline(services).run(
        input_artifact,
        target_language=target_language,
        source_language=source_language,
        job_id=job_id,
        on_stage=on_stage,
    )


def run_operation(
    input_artifact: Artifact,
    *,
    operation: str,
    masking_mode: str = "new",
    target_language: str,
    backend,
    source_language: str = "auto",
    job_id: str | None = None,
    on_stage=None,
    face_choice: str | None = None,
) -> dict[str, Artifact]:
    """Run masking, dubbing, or both and publish only that operation's outputs."""
    job_id = job_id or "job"
    if masking_mode not in {"overlay", "new", "transcript"}:
        raise ValueError(f"Unsupported masking mode: {masking_mode}")
    if operation == "combine" and masking_mode == "overlay":
        result = run_artifact(
            input_artifact, target_language=target_language, source_language=source_language,
            backend=backend, job_id=job_id, on_stage=on_stage, face_choice=face_choice,
        )
        artifacts = {
            "final_video": result.final_video,
            "masking_video": result.service_a_video,
            "service_a_video": result.service_a_video,
            "translated_captions": result.visual_subtitles,
        }
        for name, value in (
            ("service_b_video", result.service_b_video), ("voice_video", result.service_b_video),
            ("service_b_audio", result.service_b_audio), ("audio_subtitles", result.audio_subtitles),
            ("music_track", result.music_track), ("music_info", result.music_info),
        ):
            if value is not None:
                artifacts[name] = value
        return artifacts
    if operation == "combine":
        clean = (_run_transcript_caption_stage(
            input_artifact, target_language, source_language, backend, job_id, on_stage
        ) if masking_mode == "transcript" else
            _run_clean_stage(input_artifact, target_language, backend, job_id, on_stage))
        if on_stage:
            on_stage("dubbing_clean_video")
        dubbed = TextToAudioAdapter(backend, face_choice=face_choice).run(
            clean["final_video"], target_language=target_language,
            source_language=source_language, job_id=job_id,
        )
        voice = dubbed["video"]
        music = dubbed.get("music_track")
        if music is not None:
            combined = _mix_dubbed_only(backend, voice, music, job_id)
        else:
            combined = backend.publish(voice, name="dubbed_video.mp4", job_id=job_id)
        final = backend.publish(combined, name="final_video.mp4", job_id=job_id)
        artifacts = {"final_video": final, "masking_video": clean["final_video"],
                     "translated_captions": clean["translated_captions"],
                     "service_b_video": voice,
                     "voice_video": backend.publish(voice, name="voice_video.mp4", job_id=job_id)}
        if clean.get("clean_video"):
            artifacts["clean_video"] = clean["clean_video"]
        if dubbed.get("music_track"):
            artifacts["music_track"] = dubbed["music_track"]
        if dubbed.get("music_info"):
            artifacts["music_info"] = dubbed["music_info"]
        if dubbed.get("audio"):
            artifacts["service_b_audio"] = dubbed["audio"]
        if dubbed.get("audio_subtitles"):
            artifacts["audio_subtitles"] = dubbed["audio_subtitles"]
        return artifacts

    if operation == "masking":
        if masking_mode == "transcript":
            output = _run_transcript_caption_stage(
                input_artifact, target_language, source_language, backend, job_id, on_stage
            )
            video = backend.publish(output["final_video"], name="masking_video.mp4", job_id=job_id)
            captions = backend.publish(
                output["translated_captions"], name="translated_captions.srt", job_id=job_id
            )
            if on_stage:
                on_stage("masking_completed")
            voice = backend.publish(video, name="voice_video.mp4", job_id=job_id)
            artifacts = {"final_video": video, "masking_video": video, "voice_video": voice,
                         "translated_captions": captions}
            artifacts.update(analyze_masking_music(backend, input_artifact, job_id))
            return artifacts
        if masking_mode == "overlay":
            if on_stage:
                on_stage("service_a_running")
            output = LocalSubtitleTranslatorAdapter(backend).run(
                input_artifact, target_language=target_language,
                source_language=source_language, job_id=job_id,
            )
            video = backend.publish(output["video"], name="masking_video.mp4", job_id=job_id)
            captions = backend.publish(
                output["visual_subtitles"], name="translated_captions.srt", job_id=job_id
            )
            if on_stage:
                on_stage("masking_completed")
            voice = backend.publish(video, name="voice_video.mp4", job_id=job_id)
            artifacts = {"final_video": video, "masking_video": video, "voice_video": voice,
                         "service_a_video": output["video"], "translated_captions": captions}
            artifacts.update(analyze_masking_music(backend, input_artifact, job_id))
            return artifacts
        output = _run_clean_stage(input_artifact, target_language, backend, job_id, on_stage)
        video = backend.publish(output["final_video"], name="masking_video.mp4", job_id=job_id)
        voice = backend.publish(video, name="voice_video.mp4", job_id=job_id)
        if on_stage:
            on_stage("masking_completed")
        artifacts = {"final_video": video, "masking_video": video, "voice_video": voice,
                     "clean_video": output["clean_video"],
                     "translated_captions": output["translated_captions"]}
        artifacts.update(analyze_masking_music(backend, input_artifact, job_id))
        return artifacts

    if operation == "dubbing":
        if on_stage:
            on_stage("dubbing_running")
        output = TextToAudioAdapter(backend, face_choice=face_choice).run(
            input_artifact, target_language=target_language,
            source_language=source_language, job_id=job_id,
        )
        voice = output["video"]
        music = output.get("music_track")
        if music is not None:
            video = _mix_dubbed_only(backend, voice, music, job_id)
        else:
            video = backend.publish(voice, name="dubbed_video.mp4", job_id=job_id)
        if on_stage:
            on_stage("dubbing_completed")
        artifacts = {
            "final_video": video,
            "dubbed_video": video,
            "voice_video": backend.publish(voice, name="voice_video.mp4", job_id=job_id),
        }
        if output.get("music_track"):
            artifacts["music_track"] = output["music_track"]
        if output.get("music_info"):
            artifacts["music_info"] = output["music_info"]
        if output.get("audio_subtitles"):
            artifacts["audio_subtitles"] = backend.publish(
                output["audio_subtitles"], name="audio_subtitles.srt", job_id=job_id
            )
        if output.get("audio"):
            artifacts["service_b_audio"] = backend.publish(
                output["audio"], name="dubbed_audio.mp3", job_id=job_id
            )
        return artifacts

    raise ValueError(f"Unsupported operation: {operation}")
