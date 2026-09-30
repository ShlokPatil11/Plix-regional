from __future__ import annotations

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Protocol

from .contracts import Artifact


class ArtifactBackend(Protocol):
    def download(self, artifact: Artifact, destination: Path) -> Path: ...

    def upload(self, source: Path, *, filename: str, content_type: str) -> Artifact: ...

    def publish(self, artifact: Artifact, *, name: str, job_id: str) -> Artifact: ...


class FileArtifactBackend:
    """Artifact backend rooted in the host's configured persistent storage.

    Production hosts can replace this with S3, GCS, Azure Blob, or their own
    artifact store without changing the pipeline or service adapters.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def download(self, artifact: Artifact, destination: Path) -> Path:
        source = Path(artifact.metadata.get("path", ""))
        if not source.is_file():
            raise FileNotFoundError(f"Artifact is unavailable: {artifact.uri}")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    def upload(self, source: Path, *, filename: str, content_type: str) -> Artifact:
        destination = self.root / filename
        counter = 1
        while destination.exists():
            destination = self.root / f"{Path(filename).stem}-{counter}{Path(filename).suffix}"
            counter += 1
        shutil.copyfile(source, destination)
        return Artifact(
            uri=f"artifact://{destination.name}",
            filename=destination.name,
            content_type=content_type,
            metadata={"path": str(destination)},
        )


    def publish(self, artifact: Artifact, *, name: str, job_id: str) -> Artifact:
        source = Path(str(artifact.metadata.get("path", "")))
        if not source.is_file():
            raise FileNotFoundError(f"Cannot publish missing artifact: {artifact.uri}")
        destination = self.root / name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        return Artifact(
            uri=f"artifact://{name}",
            filename=name,
            content_type=artifact.content_type,
            metadata={"path": str(destination)},
        )


# Kept as a compatibility alias for callers of the first orchestration draft.
TemporaryArtifactBackend = FileArtifactBackend


class FfmpegFinalAssembler:
    """Keep Service B's dubbed video/audio and attach Service A's visual SRT."""

    def __init__(self, backend: ArtifactBackend, ffmpeg_binary: str = "ffmpeg"):
        self.backend = backend
        self.ffmpeg_binary = ffmpeg_binary

    def assemble(
        self,
        video: Artifact,
        audio: Artifact | None,
        visual_subtitles: Artifact,
        *,
        job_id: str,
    ) -> Artifact:
        with tempfile.TemporaryDirectory(prefix=f"dir-language-{job_id}-") as temp:
            root = Path(temp)
            video_path = self.backend.download(video, root / "service-a-video.mp4")
            subtitle_path = self.backend.download(visual_subtitles, root / "visual.srt")
            audio_path = self.backend.download(audio, root / audio.filename) if audio else None
            output = root / "final_video.mp4"

            command = [self.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(video_path)]
            if audio_path:
                command += ["-i", str(audio_path)]
            command += ["-i", str(subtitle_path), "-map", "0:v:0"]
            if audio_path:
                command += ["-map", "1:a:0"]
            else:
                command += ["-map", "0:a:0?"]
            subtitle_input = 2 if audio_path else 1
            command += [
                "-map", f"{subtitle_input}:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-c:s", "mov_text",
                "-metadata:s:s:0", "language=und",
                "-metadata:s:s:0", "title=Visual subtitles (Service A)",
                "-movflags", "+faststart",
                str(output),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"Final video mux failed: {completed.stderr.strip()}")
            return self.backend.upload(output, filename="final_video.mp4", content_type="video/mp4")
