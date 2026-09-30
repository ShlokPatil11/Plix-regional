from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Artifact:
    """A hosted artifact reference.

    Services exchange artifact references instead of machine-specific paths.
    The storage implementation is supplied by the host application.
    """

    uri: str
    filename: str
    content_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str, *, cause: Exception | None = None):
        detail = f" ({cause})" if cause else ""
        super().__init__(f"{stage}: {message}{detail}")
        self.stage = stage
        self.cause = cause


@dataclass(frozen=True)
class PipelineResult:
    job_id: str
    target_language: str
    final_video: Artifact
    visual_subtitles: Artifact
    audio_subtitles: Artifact | None
    service_a_video: Artifact
    service_b_video: Artifact | None
    service_b_audio: Artifact | None
    manifest: Mapping[str, Any]
