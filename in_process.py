from __future__ import annotations

from .contracts import Artifact, PipelineResult
from .media import FfmpegFinalAssembler
from .runner import DirectoryLanguagePipeline, PipelineServices
from .service_adapters import LocalSubtitleTranslatorAdapter, TextToAudioAdapter


def run_artifact(
    input_artifact: Artifact,
    *,
    target_language: str,
    backend,
    source_language: str = "auto",
    job_id: str | None = None,
    on_stage=None,
) -> PipelineResult:
    """Run both embedded services from an artifact provided by the host."""
    services = PipelineServices(
        visual_subtitles=LocalSubtitleTranslatorAdapter(backend),
        audio_dubbing=TextToAudioAdapter(backend),
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
) -> dict[str, Artifact]:
    """Run masking, dubbing, or both and publish only that operation's outputs."""
    job_id = job_id or "job"
    if operation == "combine":
        result = run_artifact(
            input_artifact, target_language=target_language, backend=backend,
            source_language=source_language, job_id=job_id, on_stage=on_stage,
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
        output = TextToAudioAdapter(backend).run(
            input_artifact, target_language=target_language,
            source_language=source_language, job_id=job_id,
        )
        video = backend.publish(output["video"], name="dubbed_video.mp4", job_id=job_id)
        if on_stage:
            on_stage("dubbing_completed")
        artifacts = {"final_video": video, "dubbed_video": video}
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
