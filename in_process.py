from __future__ import annotations

import tempfile
from pathlib import Path

from .contracts import Artifact, PipelineResult
from .media import FfmpegFinalAssembler
from .mixer import DEFAULT_MUSIC_PERCENT, MixError, mix_final
from .runner import DirectoryLanguagePipeline, PipelineServices
from .service_adapters import LocalSubtitleTranslatorAdapter, TextToAudioAdapter
from .clean_translate import run_clean_translate
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
    target_language: str,
    backend,
    source_language: str = "auto",
    job_id: str | None = None,
    on_stage=None,
    face_choice: str | None = None,
) -> dict[str, Artifact]:
    """Run masking, dubbing, or both and publish only that operation's outputs."""
    job_id = job_id or "job"
    if operation == "combine":
        clean = _run_clean_stage(input_artifact, target_language, backend, job_id, on_stage)
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
                     "clean_video": clean["clean_video"], "translated_captions": clean["translated_captions"],
                     "service_b_video": voice,
                     "voice_video": backend.publish(voice, name="voice_video.mp4", job_id=job_id)}
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
        output = _run_clean_stage(input_artifact, target_language, backend, job_id, on_stage)
        video = backend.publish(output["final_video"], name="masking_video.mp4", job_id=job_id)
        if on_stage:
            on_stage("masking_completed")
        return {"final_video": video, "masking_video": video, "clean_video": output["clean_video"],
                "translated_captions": output["translated_captions"]}

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
