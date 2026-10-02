from __future__ import annotations

import tempfile
from pathlib import Path

from .contracts import Artifact, PipelineResult
from .media import FfmpegFinalAssembler
from .mixer import DEFAULT_MUSIC_PERCENT, MixError, mix_final
from .runner import DirectoryLanguagePipeline, PipelineServices
from .service_adapters import LocalSubtitleTranslatorAdapter, TextToAudioAdapter


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
        result = run_artifact(
            input_artifact, target_language=target_language, backend=backend,
            source_language=source_language, job_id=job_id, on_stage=on_stage,
            face_choice=face_choice,
        )
        artifacts = {
            "final_video": result.final_video,
            "visual_subtitles": result.visual_subtitles,
            "service_a_video": result.service_a_video,
        }
        if result.audio_subtitles:
            artifacts["audio_subtitles"] = result.audio_subtitles
        if result.service_b_video:
            artifacts["service_b_video"] = result.service_b_video
        if result.service_b_audio:
            artifacts["service_b_audio"] = result.service_b_audio
        if result.service_b_video:
            artifacts["voice_video"] = backend.publish(
                result.service_b_video, name="voice_video.mp4", job_id=job_id
            )
        if result.music_track:
            artifacts["music_track"] = result.music_track
        if result.music_info:
            artifacts["music_info"] = result.music_info
        return {name: artifact for name, artifact in artifacts.items() if artifact is not None}

    if operation == "masking":
        if on_stage:
            on_stage("masking_running")
        output = LocalSubtitleTranslatorAdapter(backend).run(
            input_artifact, target_language=target_language,
            source_language=source_language, job_id=job_id,
        )
        video = backend.publish(output["video"], name="masking_video.mp4", job_id=job_id)
        subtitles = backend.publish(output["visual_subtitles"], name="visual_subtitles.srt", job_id=job_id)
        if on_stage:
            on_stage("masking_completed")
        return {"final_video": video, "masking_video": video, "visual_subtitles": subtitles}

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
