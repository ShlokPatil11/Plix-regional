from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from .contracts import Artifact, AwaitingFaceSelection
from .media import ArtifactBackend
from .music import MusicResult, ShazamIdentifier, YouTubeFetcher, identify_and_fetch, media_duration


SERVICE_A_ROOT = Path(__file__).resolve().parent / "vendor" / "service_a"
SERVICE_B_ROOT = Path(__file__).resolve().parent / "vendor" / "service_b"
_SERVICE_A_LOCK = threading.RLock()

LANGUAGES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "ml": "Malayalam",
    "kn": "Kannada",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "bn": "Bengali",
    "te": "Telugu",
    "ur": "Urdu",
}


def _add_import_root(root: Path) -> None:
    root = Path(root).resolve()
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)


def _guard_service_a_namespaces(root: Path) -> None:
    root = root.resolve()
    for name in ("config", "database", "models", "pipeline"):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None) if module else None
        module_path = Path(module_file).resolve() if module_file else None
        if module_path and not module_path.is_relative_to(root):
            raise RuntimeError(
                f"Module namespace '{name}' is already loaded from another package; "
                "run the hosted application in its own Python process."
            )


class LocalSubtitleTranslatorAdapter:
    """Runs Service A through its Python modules, without its CLI or fixed paths."""

    def __init__(self, backend: ArtifactBackend, service_root: Path = SERVICE_A_ROOT):
        self.backend = backend
        self.service_root = Path(service_root)

    def run(self, video: Artifact, *, target_language: str, source_language: str = "auto", job_id: str) -> dict[str, Artifact]:
        if hasattr(self.backend, "root"):
            existing_video = Path(self.backend.root) / "service-a-video.mp4"
            existing_subtitles = Path(self.backend.root) / "visual-subtitles.srt"
            if existing_video.is_file() and existing_subtitles.is_file():
                print(f"[Service A] Found existing artifacts in {self.backend.root}, skipping Service A re-run.")
                return {
                    "video": Artifact(
                        uri="artifact://service-a-video.mp4",
                        filename="service-a-video.mp4",
                        content_type="video/mp4",
                        metadata={"path": str(existing_video)},
                    ),
                    "visual_subtitles": Artifact(
                        uri="artifact://visual-subtitles.srt",
                        filename="visual-subtitles.srt",
                        content_type="application/x-subrip",
                        metadata={"path": str(existing_subtitles)},
                    ),
                }

        target_name = LANGUAGES.get(target_language.lower(), target_language)
        with _SERVICE_A_LOCK, tempfile.TemporaryDirectory(prefix=f"service-a-{job_id}-") as temp:
            root = Path(temp)
            _add_import_root(self.service_root)
            _guard_service_a_namespaces(self.service_root)
            config_module = importlib.import_module("config")
            database_module = importlib.import_module("database.db")
            processor_module = importlib.import_module("pipeline.processor")
            input_path = self.backend.download(video, root / video.filename)
            pipeline_data_dir = os.getenv(
                "PIPELINE_DATA_DIR", str(Path(tempfile.gettempdir()) / "dir-language-pipeline")
            )
            env = {
                "INPUT_DIR": str(root / "incoming"),
                "PROCESSING_DIR": str(root / "processing"),
                "COMPLETED_DIR": str(root / "completed"),
                "FAILED_DIR": str(root / "failed"),
                "TEMP_DIR": str(root / "temp"),
                "LOG_DIR": str(root / "logs"),
                "METADATA_DIR": str(root / "metadata"),
                "SCRIPT_DIR": str(root / "scripts"),
                "DB_PATH": str(root / "jobs.db"),
                "FILE_STABILITY_SECONDS": "0.01",
                "SOURCE_LANGUAGE": source_language,
                "OCR_PROVIDER": "openai",
                "OCR_TARGET_BOX": os.getenv("OCR_TARGET_BOX", "agent"),
                # Service A loads the project .env with override=True. Preserve the
                # host's real data root so a local run is not redirected to /data.
                "PIPELINE_DATA_DIR": pipeline_data_dir,
            }
            previous = {key: os.environ.get(key) for key in env}
            os.environ.update(env)
            try:
                config = config_module.Config.load(target_language=target_name, source_language=source_language)
                db = database_module.Database(config.db_path)
                checksum = hashlib.sha256()
                with input_path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum.update(block)
                digest = checksum.hexdigest()
                created = db.create(input_path, digest, source_language, target_name, force=True)
                if not created:
                    raise RuntimeError("Service A could not create a job")
                job_path = config.processing_dir / created / input_path.name
                job_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(input_path, job_path)
                db.update(created, "QUEUED", processing_path=str(job_path))
                job = db.get(created)
                if not processor_module.Processor(config, db).process(job):
                    failure = db.get(created)
                    failure_path = config.failed_dir / created / "failure.json"
                    details = failure_path.read_text(encoding="utf-8") if failure_path.is_file() else ""
                    if details:
                        print(f"[Service A failure.json]\n{details}", file=sys.stderr, flush=True)
                    raise RuntimeError(f"Service A failed: {failure.error_message}")
                completed = Path(db.get(created).output_path)
                translated_track = config.processing_dir / created / "translated.ass"
                if not translated_track.is_file():
                    translated_track = config.processing_dir / created / "translated.srt"
                if not translated_track.is_file():
                    raise FileNotFoundError("Service A completed without its translated subtitle track")
                visual_srt = root / "visual-subtitles.srt"
                import pysubs2
                document = pysubs2.load(str(translated_track), encoding="utf-8")
                if not document.events:
                    raise ValueError("Service A did not produce any translated visual subtitle cues")
                document.save(str(visual_srt), format_="srt", encoding="utf-8")
                return {
                    "video": self.backend.upload(completed, filename="service-a-video.mp4", content_type="video/mp4"),
                    "visual_subtitles": self.backend.upload(visual_srt, filename="visual-subtitles.srt", content_type="application/x-subrip"),
                }
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class TextToAudioAdapter:
    """Runs Service B on the clean source video before visual masking.

    Analyze first (transcript, shots, face candidates, music). If faces were found
    and the user has not chosen one yet, publish the choices and raise
    AwaitingFaceSelection; the job is re-queued with `face_choice` once they confirm,
    and this adapter then resumes from its on-disk caches and renders.
    """

    def __init__(
        self,
        backend: ArtifactBackend,
        service_root: Path = SERVICE_B_ROOT,
        face_choice: str | None = None,
        music_identifier: Any = None,
        music_fetcher: Any = None,
    ):
        self.backend = backend
        self.service_root = Path(service_root)
        self.face_choice = face_choice
        self.music_identifier = music_identifier
        self.music_fetcher = music_fetcher

    def _publish(self, path: Path, name: str, content_type: str, job_id: str) -> Artifact:
        artifact = Artifact(uri=f"artifact://{name}", filename=name, content_type=content_type,
                            metadata={"path": str(path)})
        return self.backend.publish(artifact, name=name, job_id=job_id)

    def _publish_music(self, music, music_dir: Path, job_id: str) -> dict[str, Artifact]:
        music_dir.mkdir(parents=True, exist_ok=True)
        info_path = music_dir / "music.json"
        info_path.write_text(json.dumps(music.info, ensure_ascii=False, indent=2), encoding="utf-8")
        published = {"music_info": self._publish(info_path, "music.json", "application/json", job_id)}
        if music.track is not None:
            published["music_track"] = self._publish(music.track, "music_track.mp3", "audio/mpeg", job_id)
        return published

    def _publish_face_choices(self, candidates, job_dir: Path, job_id: str) -> None:
        for candidate in candidates:
            if candidate.crop_file:
                self._publish(job_dir / "faces" / candidate.crop_file, candidate.crop_file, "image/jpeg", job_id)
        panel = job_dir / "faces.json"
        panel.write_text(
            json.dumps({"candidates": [c.public_view() for c in candidates]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._publish(panel, "faces.json", "application/json", job_id)

    def run(self, video: Artifact, *, target_language: str, source_language: str = "auto", job_id: str) -> dict[str, Artifact]:
        service_root = self.service_root
        _add_import_root(service_root)
        cached = sys.modules.get("src")
        if cached is not None and not Path(cached.__file__).resolve().is_relative_to(service_root.resolve()):
            cached_root = Path(cached.__file__).resolve().parent.parent
            if cached_root == SERVICE_B_ROOT:
                service_root = cached_root  # compatibility for legacy unit-test fixtures
            else:
                raise RuntimeError("A different top-level 'src' package is already loaded; isolate this application process.")
        engine = importlib.import_module("src.engine")
        if not (service_root / "src" / "analysis.py").is_file():
            data_dir = Path(os.getenv("PIPELINE_DATA_DIR", "/data"))
            root = data_dir / "cache" / "service-b-test" / job_id
            root.mkdir(parents=True, exist_ok=True)
            source = self.backend.download(video, root / "source-video.mp4")
            output = Path(engine.run_dubbing_engine(
                source_video_path=source,
                target_lang=target_language,
                source_lang=source_language,
                job_dir=root / "job",
                dry_run=os.getenv("DUBBING_DRY_RUN", "false").lower() == "true",
            ))
            result = {"video": self.backend.upload(output, filename="service-b-video.mp4", content_type="video/mp4")}
            audio_path = output.parent / f"dubbed_master_{target_language}.mp3"
            subtitle_path = output.parent / f"transcript_{target_language}.srt"
            if audio_path.is_file():
                result["audio"] = self.backend.upload(audio_path, filename="service-b-audio.mp3", content_type="audio/mpeg")
            if subtitle_path.is_file():
                result["audio_subtitles"] = self.backend.upload(
                    subtitle_path, filename="audio-subtitles.srt", content_type="application/x-subrip")
            return result
        analysis = importlib.import_module("src.analysis")
        faces = importlib.import_module("src.faces")
        audio_tools = importlib.import_module("src.audio")
        data_dir = Path(os.getenv("PIPELINE_DATA_DIR", "/data"))
        root = data_dir / "cache" / "service-b" / job_id
        root.mkdir(parents=True, exist_ok=True)
        source = self.backend.download(video, root / "source-video.mp4")
        job_dir = root / "job"
        ffmpeg_binary = audio_tools.ffmpeg()

        # --- Analyze (cached on disk, so a resume after confirmation is cheap) ---
        info = analysis.analyze_video(source, job_dir, source_language)
        music_dir = root / "music"
        try:
            music = identify_and_fetch(
                source, info["words"], info["duration"], music_dir,
                self.music_identifier or ShazamIdentifier(),
                self.music_fetcher or YouTubeFetcher(ffmpeg_location=ffmpeg_binary),
                ffmpeg_binary=ffmpeg_binary,
            )
        except Exception as exc:  # music is optional; never fail a paid job over it
            print(f"[Music] Unexpected error, continuing voice-only: {exc}")
            music = MusicResult("not_identified", {"status": "not_identified", "error": str(exc)})
        print(f"[Music] status={music.status}")
        extras = self._publish_music(music, music_dir, job_id)

        candidates = faces.load_or_discover(source, info["scenes"], info["words"], job_dir)
        if not candidates:
            print("[Faces] No usable face found — dubbing audio-only, no lip-sync.")
            lock = faces.FaceLock.disabled()
        elif self.face_choice is None:
            self._publish_face_choices(candidates, job_dir, job_id)
            print(f"[Faces] {len(candidates)} candidate(s) found — waiting for the user to confirm one.")
            raise AwaitingFaceSelection(job_id)
        else:
            lock = faces.select_lock(candidates, self.face_choice)
            print(f"[Faces] Locked face candidate {self.face_choice}.")

        # --- Render ---
        output = engine.run_dubbing_engine(
            source_video_path=source,
            target_lang=target_language,
            source_lang=source_language,
            job_dir=job_dir,
            dry_run=os.getenv("DUBBING_DRY_RUN", "false").lower() == "true",
            face_lock=lock,
        )
        output = Path(output)
        result: dict[str, Artifact] = {
            "video": self.backend.upload(output, filename="service-b-video.mp4", content_type="video/mp4")
        }
        audio_path = output.parent / f"dubbed_master_{target_language}.mp3"
        if audio_path.is_file():
            result["audio"] = self.backend.upload(
                audio_path, filename="service-b-audio.mp3", content_type="audio/mpeg"
            )
        audio_srt = output.parent / f"transcript_{target_language}.srt"
        if audio_srt.is_file():
            result["audio_subtitles"] = self.backend.upload(
                audio_srt, filename="audio-subtitles.srt", content_type="application/x-subrip"
            )
        result.update(extras)
        return result


def analyze_masking_music(backend: ArtifactBackend, video: Artifact, job_id: str) -> dict[str, Artifact]:
    """Identify music for masking-only jobs without running dubbing or face analysis."""
    backend_root = Path(getattr(backend, "root", tempfile.gettempdir())).resolve()
    root = backend_root / ".masking-music"
    root.mkdir(parents=True, exist_ok=True)
    source = backend.download(video, root / "source-video.mp4")
    try:
        music = identify_and_fetch(
            source, [], media_duration(source), root / "music",
            ShazamIdentifier(), YouTubeFetcher(), ffmpeg_binary="ffmpeg",
        )
    except Exception as exc:
        print(f"[Music] Masking-only analysis failed: {exc}")
        music = MusicResult("not_identified", {"status": "not_identified", "error": str(exc)})
    # The masking output retains its original complete audio. A detected track is
    # informational here; remixing it would duplicate music already in that audio.
    music.info["mixable"] = False
    adapter = TextToAudioAdapter(backend)
    return adapter._publish_music(music, root / "music", job_id)
