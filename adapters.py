from __future__ import annotations

from typing import Any, Callable, Mapping

from .contracts import Artifact


class CallableVisualSubtitleService:
    """Adapter for a hosted Service A callable."""

    def __init__(self, handler: Callable[..., Mapping[str, Any]]):
        self.handler = handler

    def run(self, video: Artifact, *, target_language: str, source_language: str = "auto", job_id: str) -> Mapping[str, Any]:
        return self.handler(video, target_language=target_language, source_language=source_language, job_id=job_id)


class CallableAudioDubbingService:
    """Adapter for a hosted Service B callable."""

    def __init__(self, handler: Callable[..., Mapping[str, Any]]):
        self.handler = handler

    def run(self, video: Artifact, *, target_language: str, source_language: str = "auto", job_id: str) -> Mapping[str, Any]:
        return self.handler(video, target_language=target_language, source_language=source_language, job_id=job_id)
