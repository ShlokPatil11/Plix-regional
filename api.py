from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
import json
import logging
import mimetypes
import os
from pathlib import Path

# Load .env from this file's directory (or any parent) before anything else
from dotenv import load_dotenv as _load_dotenv
_dotenv_path = Path(__file__).resolve().parent
for _ in range(6):
    if (_dotenv_path / '.env').exists():
        _load_dotenv(_dotenv_path / '.env', override=False)
        break
    _dotenv_path = _dotenv_path.parent
import re
import shutil
import sys
import tempfile
import threading
import traceback
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .contracts import AwaitingFaceSelection, PipelineStageError
from .in_process import run_operation
from .job_store import JobStore
from .media import FileArtifactBackend
from .magnific_oauth import MagnificAuth
from .mixer import DEFAULT_MUSIC_PERCENT, MixError, mix_final
from .music import MusicFetchError, YouTubeFetcher, media_duration, prepare_custom_track
from .service_adapters import LANGUAGES


DATA_DIR = Path(os.getenv("PIPELINE_DATA_DIR", str(Path(tempfile.gettempdir()) / "dir-language-pipeline"))).resolve()
MAX_UPLOAD_BYTES = int(os.getenv("PIPELINE_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
JOB_ROOT = DATA_DIR / "jobs"
JOB_ROOT.mkdir(parents=True, exist_ok=True)
store = JobStore(DATA_DIR / "jobs.sqlite3")
magnific_auth = MagnificAuth(DATA_DIR / "magnific-oauth.json")
stop_worker = threading.Event()
worker_thread: threading.Thread | None = None
CROP_NAME = re.compile(r"^face_cand_\d+\.jpg$")
remix_lock = threading.Lock()
ALLOWED_MUSIC_SUFFIXES = frozenset({".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"})
MAX_MUSIC_UPLOAD_BYTES = 50 * 1024 * 1024
MIN_VOICE_SECONDS = 0.5


def make_youtube_fetcher() -> YouTubeFetcher:
    return YouTubeFetcher()


class FaceChoice(BaseModel):
    candidate_id: str


class RemixRequest(BaseModel):
    music_percent: int = Field(ge=0, le=100)


def _artifacts_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id / "artifacts"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class _TerminalAndFileStream:
    """Mirror every worker print to the server terminal and a persistent log."""

    def __init__(self, terminal, logfile, job_id: str):
        self.terminal = terminal
        self.logfile = logfile
        self.prefix = f"[job {job_id}] "
        self.line_start = True
        self.lock = threading.RLock()

    def write(self, value: str) -> int:
        if not value:
            return 0
        with self.lock:
            offset = 0
            while offset < len(value):
                end = value.find("\n", offset)
                end = len(value) if end < 0 else end + 1
                piece = value[offset:end]
                if self.line_start:
                    self.terminal.write(self.prefix)
                    self.logfile.write(self.prefix)
                self.terminal.write(piece)
                self.logfile.write(piece)
                self.line_start = piece.endswith("\n")
                offset = end
            self.flush()
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())


def _process_job(job: dict) -> None:
    job_id = job["job_id"]
    job_root = JOB_ROOT / job_id
    artifacts_root = job_root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    backend = FileArtifactBackend(artifacts_root)
    log_path = job_root / "process.log"
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            terminal = sys.__stdout__ or sys.stdout
            stream = _TerminalAndFileStream(terminal, log, job_id)
            root_logger = logging.getLogger()
            previous_level = root_logger.level
            handler = logging.StreamHandler(stream)
            root_logger.addHandler(handler)
            if previous_level > logging.INFO:
                root_logger.setLevel(logging.INFO)
            try:
                with redirect_stdout(stream), redirect_stderr(stream):
                    print(f"Starting pipeline for {job['input_name']} | source={job['source_language']} | target={job['target_language']}")
                    source = Path(job["input_path"])
                    content_type = mimetypes.guess_type(job["input_name"])[0] or "application/octet-stream"
                    incoming = backend.upload(source, filename=job["input_name"], content_type=content_type)
                    result = run_operation(
                        incoming,
                        operation=job["operation"],
                        masking_mode=job.get("masking_mode", "new"),
                        target_language=job["target_language"],
                        source_language=job["source_language"],
                        backend=backend,
                        job_id=job_id,
                        face_choice=job.get("face_choice"),
                        on_stage=lambda value: (store.stage(job_id, value), print(f"--- stage: {value} ---", flush=True)),
                    )
            finally:
                root_logger.removeHandler(handler)
                root_logger.setLevel(previous_level)
        output_artifacts = {
            name: artifact.metadata["path"] for name, artifact in result.items()
        }
        store.finish(job_id, output_artifacts)
    except AwaitingFaceSelection:
        store.await_face(job_id)
        print(f"[job {job_id}] waiting for the user to confirm which face to lip-sync",
              file=sys.__stderr__ or sys.stderr, flush=True)
        return
    except Exception as exc:
        stage = exc.stage if isinstance(exc, PipelineStageError) else "processing"
        trace = traceback.format_exc()
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[job {job_id}] FAILED at {stage}: {exc}\n{trace}\n")
        error_console = sys.__stderr__ or sys.stderr
        print(f"[job {job_id}] FAILED at {stage}: {exc}\n{trace}", file=error_console, flush=True)
        store.fail(job_id, stage, str(exc))


def _worker() -> None:
    terminal = sys.__stderr__ or sys.stderr
    print("[pipeline-worker] started", file=terminal, flush=True)
    while not stop_worker.is_set():
        job = None
        try:
            store.expire_stale()
            job = store.claim()
            if job is None:
                stop_worker.wait(0.5)
                continue
            print(f"[pipeline-worker] claimed job {job['job_id']}", file=terminal, flush=True)
            _process_job(job)
        except Exception:
            details = traceback.format_exc()
            print(f"[pipeline-worker] loop error\n{details}", file=terminal, flush=True)
            if job is not None:
                try:
                    store.fail(job["job_id"], "worker", details.strip())
                except Exception:
                    print(f"[pipeline-worker] could not record failure for {job['job_id']}\n{traceback.format_exc()}", file=terminal, flush=True)
            stop_worker.wait(2)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global worker_thread
    stop_worker.clear()
    worker_thread = threading.Thread(target=_worker, name="directory-language-worker", daemon=True)
    worker_thread.start()
    try:
        yield
    finally:
        stop_worker.set()
        if worker_thread:
            await asyncio.to_thread(worker_thread.join, 30)


app = FastAPI(title="Directory Language Studio", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.mount("/images", StaticFiles(directory=Path(__file__).parent / "static" / "images"), name="images")
app.mount("/boot-media", StaticFiles(directory=Path(__file__).parent / "bg video"), name="boot-media")


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/api/languages")
def languages():
    return [{"code": code, "name": name} for code, name in LANGUAGES.items()]


@app.post("/api/detect-language")
async def detect_uploaded_language(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise HTTPException(415, "Upload an MP4, MOV, MKV, AVI, or WebM video.")
    with tempfile.TemporaryDirectory(prefix="language-detection-", dir=DATA_DIR) as temporary:
        root = Path(temporary)
        source = root / f"source{suffix}"
        written = 0
        try:
            with source.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Video exceeds configured upload size limit.")
                    output.write(chunk)
        finally:
            await file.close()
        try:
            from .vendor.service_b.src.transcriber import detect_language

            detected = await asyncio.to_thread(detect_language, source, root)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc
    code = detected["code"].lower()
    return {"code": code, "name": LANGUAGES.get(code, code.upper()),
            "confidence": round(detected["probability"], 3)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "worker_running": bool(worker_thread and worker_thread.is_alive()),
    }


@app.get("/api/magnific/status")
def magnific_status():
    return magnific_auth.status()


@app.get("/api/magnific/connect")
def magnific_connect(request: Request):
    redirect_uri = str(request.url_for("magnific_callback"))
    try:
        return RedirectResponse(magnific_auth.begin(redirect_uri))
    except Exception as exc:
        raise HTTPException(502, f"Could not start Magnific login: {exc}") from exc


@app.get("/api/magnific/callback", name="magnific_callback")
def magnific_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        raise HTTPException(400, f"Magnific denied the connection: {error}")
    if not code or not state:
        raise HTTPException(400, "Magnific callback is missing its authorization code or state")
    try:
        magnific_auth.finish(state, code)
    except Exception as exc:
        raise HTTPException(502, f"Could not complete Magnific login: {exc}") from exc
    return RedirectResponse("/?magnific=connected")


@app.post("/api/jobs", status_code=202)
async def submit_job(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    source_language: str | None = Form(None),
    operation: str = Form("combine"),
    masking_mode: str = Form("new"),
):
    if operation not in {"masking", "dubbing", "combine"}:
        raise HTTPException(422, "Unsupported operation.")
    if masking_mode not in {"overlay", "new"}:
        raise HTTPException(422, "Unsupported masking mode.")
    if operation in {"masking", "combine"} and masking_mode == "new" and not magnific_auth.status()["connected"]:
        raise HTTPException(409, "Connect Magnific before using the New masking method.")
    target = target_language.lower().strip()
    if target not in LANGUAGES:
        raise HTTPException(422, f"Unsupported target language code: {target}")
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise HTTPException(415, "Upload an MP4, MOV, MKV, AVI, or WebM video.")

    job_id = uuid4().hex
    job_root = JOB_ROOT / job_id
    input_dir = job_root / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    safe_name = Path(file.filename or f"video{suffix}").name
    destination = input_dir / safe_name
    written = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds configured upload size limit.")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    source = (source_language or os.getenv("PIPELINE_SOURCE_LANGUAGE", "auto")).strip() or "auto"
    store.create(job_id, target, source, destination, safe_name, operation, masking_mode)
    return {"job_id": job_id, "status": "queued", "operation": operation,
            "masking_mode": masking_mode, "status_url": f"/api/jobs/{job_id}"}


def _job_payload(record: dict) -> dict:
    job_id = record["job_id"]
    return {
        "job_id": job_id,
        "input_name": record["input_name"],
        "status": record["status"],
        "stage": record["stage"],
        "target_language": record["target_language"],
        "source_language": record["source_language"],
        "operation": record["operation"],
        "masking_mode": record.get("masking_mode", "new"),
        "error": record["error"],
        "music": _read_json(_artifacts_dir(job_id) / "music.json"),
        "artifacts": {
            name: f"/api/jobs/{job_id}/artifacts/{name}"
            for name in record["artifacts"]
        },
        "manifest_url": f"/api/jobs/{job_id}/manifest",
        "log_url": f"/api/jobs/{job_id}/logs",
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }


@app.get("/api/jobs")
def list_jobs(limit: int = 100):
    """List the durable queue and job history for browser restoration."""
    return [_job_payload(record) for record in store.list(limit)]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    return _job_payload(record)


@app.delete("/api/jobs/{job_id}")
def remove_queued_job(job_id: str):
    outcome = store.remove_queued(job_id)
    if outcome == "unknown_job":
        raise HTTPException(404, "Job not found")
    if outcome == "not_queued":
        raise HTTPException(409, "Only jobs that have not started can be removed from the queue.")
    job_root = (JOB_ROOT / job_id).resolve()
    if job_root.parent != JOB_ROOT.resolve():
        raise HTTPException(400, "Invalid job ID")
    shutil.rmtree(job_root, ignore_errors=True)
    return {"job_id": job_id, "status": "removed"}


@app.get("/api/jobs/{job_id}/logs")
def get_job_log(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(404, "Job not found")
    path = JOB_ROOT / job_id / "process.log"
    if not path.is_file():
        raise HTTPException(404, "Job log not available yet")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename="process.log")


@app.get("/api/jobs/{job_id}/manifest")
def get_manifest(job_id: str):
    if store.get(job_id) is None:
        raise HTTPException(404, "Job not found")
    path = JOB_ROOT / job_id / "manifest.json"
    if not path.is_file():
        raise HTTPException(404, "Manifest not available")
    return FileResponse(path, media_type="application/json", filename="manifest.json")


@app.get("/api/jobs/{job_id}/artifacts/{name}")
def get_artifact(job_id: str, name: str):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    if name not in record["artifacts"]:
        raise HTTPException(404, "Artifact not available")
    path = Path(record["artifacts"][name]).resolve()
    allowed = (JOB_ROOT / job_id / "artifacts").resolve()
    if allowed not in path.parents or not path.is_file():
        raise HTTPException(404, "Artifact not available")
    suffix = path.suffix.lower()
    media = {
        ".mp4": "video/mp4", ".srt": "application/x-subrip", ".mp3": "audio/mpeg",
        ".json": "application/json", ".jpg": "image/jpeg",
    }.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/api/jobs/{job_id}/faces")
def get_faces(job_id: str):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    panel = _read_json(_artifacts_dir(job_id) / "faces.json")
    if panel is None:
        raise HTTPException(404, "No face choices available for this job")
    return {
        "status": record["status"],
        "candidates": panel["candidates"],
        "music": _read_json(_artifacts_dir(job_id) / "music.json"),
    }


@app.get("/api/jobs/{job_id}/faces/{filename}")
def get_face_crop(job_id: str, filename: str):
    if store.get(job_id) is None:
        raise HTTPException(404, "Job not found")
    if not CROP_NAME.match(filename):
        raise HTTPException(404, "Crop not found")
    path = _artifacts_dir(job_id) / filename
    if not path.is_file():
        raise HTTPException(404, "Crop not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/jobs/{job_id}/face")
def confirm_face(job_id: str, body: FaceChoice):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    panel = _read_json(_artifacts_dir(job_id) / "faces.json")
    if panel is None:
        raise HTTPException(409, "This job has no face choices to confirm")
    if body.candidate_id not in {candidate["id"] for candidate in panel["candidates"]}:
        raise HTTPException(422, "Unknown face candidate")
    outcome = store.confirm_face(job_id, body.candidate_id)
    if outcome == "not_awaiting":
        raise HTTPException(409, "This job is not waiting for a face choice")
    return {"status": store.get(job_id)["status"]}


@app.post("/api/jobs/{job_id}/remix")
def remix(job_id: str, body: RemixRequest):
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    if record["status"] != "completed":
        raise HTTPException(409, "The job has not completed yet")
    art = _artifacts_dir(job_id)
    info = _read_json(art / "music.json")
    voice, track = art / "voice_video.mp4", art / "music_track.mp3"
    if not info or info.get("status") != "ok" or not voice.is_file() or not track.is_file():
        raise HTTPException(409, "No identified music track for this job")
    final_path = record["artifacts"].get("final_video")
    if not final_path:
        raise HTTPException(409, "The job has no final video to rebuild")
    subtitles = art / "visual_subtitles.srt"
    try:
        with remix_lock:
            mix_final(voice, track, body.music_percent, Path(final_path),
                      subtitles=subtitles if subtitles.is_file() else None)
            info["percent"] = body.music_percent
            (art / "music.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    except MixError as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"music_percent": body.music_percent, "final_video": f"/api/jobs/{job_id}/artifacts/final_video"}


@app.post("/api/jobs/{job_id}/music")
def set_custom_music(
    job_id: str,
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    music_percent: int | None = Form(None, ge=0, le=100),
):
    """Replace the job's music with an uploaded audio file or a single YouTube link, then re-mix."""
    record = store.get(job_id)
    if record is None:
        raise HTTPException(404, "Job not found")
    url = (url or "").strip() or None
    has_file = file is not None and bool(file.filename)
    if has_file == (url is not None):
        raise HTTPException(422, "Send either an audio file or a YouTube link, not both and not neither.")
    if record["status"] != "completed":
        raise HTTPException(409, "The job has not completed yet")
    art = _artifacts_dir(job_id)
    voice = art / "voice_video.mp4"
    final_path = record["artifacts"].get("final_video")
    if not voice.is_file() or not final_path:
        raise HTTPException(409, "This job has no voice-only video to add music to")
    duration = media_duration(voice)
    if duration < MIN_VOICE_SECONDS:
        raise HTTPException(409, "This job's voice-only video could not be read")

    scratch = art / f".custom_{uuid4().hex}"
    scratch.mkdir()
    try:
        if has_file:
            safe_name = Path(file.filename).name
            suffix = Path(safe_name).suffix.lower()
            if suffix not in ALLOWED_MUSIC_SUFFIXES:
                raise HTTPException(415, "Upload an MP3, WAV, M4A, AAC, OGG or FLAC audio file.")
            source = scratch / f"upload{suffix}"
            written = 0
            with source.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_MUSIC_UPLOAD_BYTES:
                        raise HTTPException(413, f"Audio file is larger than {MAX_MUSIC_UPLOAD_BYTES // (1024 * 1024)} MB.")
                    output.write(chunk)
            title, source_url = safe_name[:80], None
        else:
            fetched = make_youtube_fetcher().fetch_url(url, scratch)
            source, title, source_url = fetched.path, (fetched.title or "YouTube audio")[:80], fetched.url
        new_track = scratch / "music_track.mp3"
        prepare_custom_track(source, duration, new_track)

        previous = _read_json(art / "music.json") or {}
        percent = music_percent if music_percent is not None else previous.get("percent", DEFAULT_MUSIC_PERCENT)
        info = {"status": "ok", "source": "custom", "title": title, "artist": "Custom",
                "source_url": source_url, "percent": percent}
        subtitles = art / "visual_subtitles.srt"
        with remix_lock:
            mix_final(voice, new_track, percent, Path(final_path),
                      subtitles=subtitles if subtitles.is_file() else None)
            os.replace(new_track, art / "music_track.mp3")
            info_tmp = art / ".music.json.tmp"
            info_tmp.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(info_tmp, art / "music.json")
        store.set_artifact(job_id, "music_track", str(art / "music_track.mp3"))
        store.set_artifact(job_id, "music_info", str(art / "music.json"))
    except MusicFetchError as exc:
        raise HTTPException(422, str(exc)) from exc
    except MixError as exc:
        print(f"[job {job_id}] custom music mix failed: {exc}", file=sys.__stderr__ or sys.stderr, flush=True)
        raise HTTPException(500, "Could not mix that music into the video.") from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {"music_percent": percent, "final_video": f"/api/jobs/{job_id}/artifacts/final_video", "music": info}
