from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from .contracts import Artifact, AwaitingFaceSelection, PipelineResult, PipelineStageError
from .mixer import DEFAULT_MUSIC_PERCENT


class VisualSubtitleService(Protocol):
    def run(self, video: Artifact, *, target_language: str, source_language: str, job_id: str) -> Mapping[str, Any]: ...


class AudioDubbingService(Protocol):
    def run(self, video: Artifact, *, target_language: str, source_language: str, job_id: str) -> Mapping[str, Any]: ...


class ArtifactPublisher(Protocol):
    def publish(self, artifact: Artifact, *, name: str, job_id: str) -> Artifact: ...


class FinalVideoAssembler(Protocol):
    def assemble(
        self,
        video: Artifact,
        audio: Artifact | None,
        visual_subtitles: Artifact,
        *,
        job_id: str,
        music: Artifact | None = None,
        music_percent: int = DEFAULT_MUSIC_PERCENT,
    ) -> Artifact: ...


@dataclass(frozen=True)
class PipelineServices:
    visual_subtitles: VisualSubtitleService
    audio_dubbing: AudioDubbingService
    assembler: FinalVideoAssembler
    publisher: ArtifactPublisher


class DirectoryLanguagePipeline:
    """Chain visual subtitle translation into audio dubbing.

    Service A and Service B keep separate subtitle artifacts. Only Service A's
    video is passed into Service B. The final assembler attaches Service A's
    visual SRT and uses Service B's audio result.
    """

    def __init__(self, services: PipelineServices):
        self.services = services

    def run(
        self,
        video: Artifact,
        *,
        target_language: str,
        source_language: str = "auto",
        job_id: str | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> PipelineResult:
        job_id = job_id or uuid4().hex
        if not target_language.strip():
            raise ValueError("target_language must not be empty")

        try:
            if on_stage:
                on_stage("service_a_running")
            visual = self._invoke(
                self.services.visual_subtitles, video,
                target_language=target_language, source_language=source_language, job_id=job_id,
            )
            service_a_video = self._required(visual, "video", "service_a")
            visual_srt = self._required(visual, "visual_subtitles", "service_a")
            if on_stage:
                on_stage("service_a_completed")
        except Exception as exc:
            if isinstance(exc, PipelineStageError):
                raise
            raise PipelineStageError("service_a", "visual subtitle translation failed", cause=exc) from exc

        try:
            if on_stage:
                on_stage("service_b_running")
            audio = self._invoke(
                self.services.audio_dubbing, service_a_video,
                target_language=target_language, source_language=source_language, job_id=job_id,
            )
            dubbed_video = audio.get("video")
            dubbed_audio = audio.get("audio")
            audio_srt = audio.get("audio_subtitles")
            music_track = audio.get("music_track")
            music_info = audio.get("music_info")
            if on_stage:
                on_stage("service_b_completed")
        except Exception as exc:
            if isinstance(exc, (PipelineStageError, AwaitingFaceSelection)):
                raise
            raise PipelineStageError("service_b", "audio dubbing failed", cause=exc) from exc

        try:
            if on_stage:
                on_stage("finalizing")
            assembly_video = dubbed_video or service_a_video
            final_video = self.services.assembler.assemble(
                assembly_video,
                None if dubbed_video else dubbed_audio,
                visual_srt,
                job_id=job_id,
                music=music_track,
                music_percent=DEFAULT_MUSIC_PERCENT,
            )
            final_video = self.services.publisher.publish(
                final_video, name="final_video.mp4", job_id=job_id
            )
            visual_srt = self.services.publisher.publish(
                visual_srt, name="visual_subtitles.srt", job_id=job_id
            )
            if audio_srt is not None:
                audio_srt = self.services.publisher.publish(
                    audio_srt, name="audio_subtitles.srt", job_id=job_id
                )
        except Exception as exc:
            if isinstance(exc, PipelineStageError):
                raise
            raise PipelineStageError("finalize", "final video assembly failed", cause=exc) from exc

        manifest = {
            "job_id": job_id,
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_language": target_language,
            "visual_subtitles": visual_srt.uri,
            "audio_subtitles": audio_srt.uri if audio_srt else None,
            "service_a_video": service_a_video.uri,
            "service_b_video": dubbed_video.uri if dubbed_video else None,
            "service_b_audio": dubbed_audio.uri if dubbed_audio else None,
            "music_track": music_track.uri if music_track else None,
            "final_video": final_video.uri,
        }
        return PipelineResult(
            job_id=job_id,
            target_language=target_language,
            final_video=final_video,
            visual_subtitles=visual_srt,
            audio_subtitles=audio_srt,
            service_a_video=service_a_video,
            service_b_video=dubbed_video,
            service_b_audio=dubbed_audio,
            manifest=manifest,
            music_track=music_track,
            music_info=music_info,
        )

    @staticmethod
    def _required(result: Mapping[str, Any], key: str, service: str) -> Artifact:
        value = result.get(key)
        if not isinstance(value, Artifact):
            raise PipelineStageError(service, f"missing required artifact: {key}")
        return value

    @staticmethod
    def _invoke(handler: Any, *args: Any, **kwargs: Any) -> Any:
        method = getattr(handler, "run", None)
        return method(*args, **kwargs) if method else handler(*args, **kwargs)
