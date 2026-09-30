"""
Stage 3 — ElevenLabs Audio Generation
Two modes depending on whether ElevenLabs Dubbing API supports the target language:

MODE A — Dubbing API (auto, supported languages only):
  POST /v1/dubbing → full video → dubbed audio + timed SRT
  Supported: hi, es, fr, de, pt, ja, ko, zh, ar, etc.
  NOT supported: mr (Marathi) — falls through to Mode B automatically.

MODE B — Voice-Clone TTS (for unsupported languages like Marathi):
  1. Extract voice sample from video (ffmpeg)
  2. Clone voice via ElevenLabs Instant Voice Clone
  3. Translate transcript using a supplied translated SRT or Google Translate
  4. TTS each cue using eleven_v4 / eleven_v3 (both support Marathi)
  5. Assemble time-fitted audio matching source video duration

Cache: per (video_hash, target_lang). Never re-charges on re-run.

Supported Dubbing API languages (partial, check EL docs for full list):
  af, ar, bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, id, it, ja, ko,
  lt, lv, ms, nl, no, pl, pt, ro, ru, sk, sl, sv, ta, tr, uk, vi, zh, hi
NOT supported by Dubbing API (use Voice-Clone TTS instead):
  mr, gu, bn, pa, ml, te, kn, ur, ne, si, my, km, lo, ...
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from .common import PipelineError
from .srt_parser import parse_srt, Cue

# Languages supported by ElevenLabs Dubbing API
# (based on validation_error testing — mr returns unsupported_target_language)
EL_DUBBING_SUPPORTED = {
    "af", "ar", "bg", "ca", "cs", "da", "de", "el", "en", "es", "es-419",
    "et", "fi", "fr", "he", "hi", "hr", "hu", "id", "it", "ja", "ko",
    "lt", "lv", "ms", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl",
    "sq", "sr", "sv", "sw", "ta", "th", "tl", "tr", "uk", "vi", "zh",
}

# ElevenLabs TTS models that support Marathi and other Indian languages
EL_V4_MODEL = "eleven_v4"           # Best quality, Marathi supported
EL_V4_TURBO = "eleven_v4_turbo"     # Faster, Marathi supported
EL_V3_MODEL = "eleven_v3"           # Also Marathi supported


def _video_hash(video_path: Path) -> str:
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        h.update(f.read(2 * 1024 * 1024))
    h.update(str(video_path.stat().st_size).encode())
    return h.hexdigest()[:16]


def _extract_voice_sample(video_path: Path, out_wav: Path, duration_s: float = 20.0) -> Path:
    """Extract first N seconds of clean speech from video for voice cloning."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(video_path),
        "-vn", "-ss", "1.0", "-t", str(duration_s),
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        str(out_wav),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"Voice sample extraction failed: {r.stderr}")
    return out_wav


def _ensure_voice_clone(sample_wav: Path, voice_name: str, api_key: str) -> str:
    """Get existing clone by name, or create one. Returns voice_id."""
    headers = {"xi-api-key": api_key}
    r = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers, timeout=30)
    if r.ok:
        for v in r.json().get("voices", []):
            if v["name"] == voice_name:
                print(f"[Dubber] Reusing voice clone '{voice_name}' (ID: {v['voice_id']})")
                return v["voice_id"]

    print(f"[Dubber] Creating voice clone '{voice_name}' from {sample_wav.name}...")
    with open(sample_wav, "rb") as f:
        resp = requests.post(
            "https://api.elevenlabs.io/v1/voices/add",
            headers=headers,
            data={"name": voice_name, "description": "Auto-cloned for dubbing"},
            files={"files": ("sample.wav", f, "audio/wav")},
            timeout=60,
        )
    if not resp.ok:
        raise PipelineError(f"Voice clone failed: HTTP {resp.status_code} — {resp.text[:300]}")
    vid = resp.json()["voice_id"]
    print(f"[Dubber] Created voice clone ID: {vid}")
    return vid


def _translate_cues_google(cues: list[Cue], target_lang: str) -> list[Cue]:
    """Translate cue texts using unofficial Google Translate (no API key needed)."""
    from urllib.parse import quote
    translated = []
    for cue in cues:
        text = cue.text.replace("\n", " ").strip()
        url = (
            f"https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl=auto&tl={target_lang}&dt=t&q={quote(text)}"
        )
        try:
            r = requests.get(url, timeout=10)
            if r.ok:
                data = r.json()
                translated_text = "".join(part[0] for part in data[0] if part[0])
            else:
                translated_text = text
        except Exception:
            translated_text = text
        new_cue = Cue(cue.id, cue.start_ms, cue.end_ms, translated_text)
        translated.append(new_cue)
        print(f"  [{cue.id}] {text[:40]} → {translated_text[:40]}")
    return translated


def _tts_cue(voice_id: str, text: str, out_path: Path, model: str, api_key: str) -> Path:
    """Synthesize a single cue to audio file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.85,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers=headers, json=payload, timeout=60,
    )
    if not r.ok:
        raise PipelineError(f"TTS failed for '{text[:30]}...': HTTP {r.status_code} — {r.text[:200]}")
    out_path.write_bytes(r.content)
    return out_path


def _assemble_tts_audio(
    cues: list[Cue], cue_audio_paths: list[Path],
    total_duration_s: float, out_path: Path
) -> Path:
    """
    Assemble per-cue audio files into a single full-length audio track.
    Each cue is placed at its start_ms position. Gaps are silence.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Build ffmpeg complex filter
    inputs = []
    filter_parts = []
    delay_parts = []

    for i, (cue, audio_path) in enumerate(zip(cues, cue_audio_paths)):
        if not audio_path.exists():
            continue
        start_ms = int(cue.start_ms)
        inputs += ["-i", str(audio_path)]
        filter_parts.append(f"[{i}:a]adelay={start_ms}|{start_ms}[a{i}]")
        delay_parts.append(f"[a{i}]")

    if not delay_parts:
        raise PipelineError("No cue audio files were generated.")

    mix_filter = "".join(delay_parts) + f"amix=inputs={len(delay_parts)}:normalize=0[out]"
    full_filter = ";".join(filter_parts) + ";" + mix_filter

    cmd = [
        "ffmpeg", "-y", "-nostdin",
        *inputs,
        "-filter_complex", full_filter,
        "-map", "[out]",
        "-t", f"{total_duration_s:.6f}",
        "-acodec", "libmp3lame", "-q:a", "2",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"TTS audio assembly failed: {r.stderr}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Mode A — ElevenLabs Dubbing API
# ──────────────────────────────────────────────────────────────────────────────

class ElevenLabsDubber:
    BASE_URL = "https://api.elevenlabs.io/v1/dubbing"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise PipelineError("Set ELEVENLABS_API_KEY in .env or environment.")
        self.headers = {"xi-api-key": self.api_key}

    def create_job(self, video_path: Path, target_lang: str,
                   source_lang: str = "auto", num_speakers: int = 0) -> str:
        video_path = Path(video_path).resolve()
        suffix = video_path.suffix.lower()
        content_type = "video/mp4" if suffix in (".mp4", ".mov", ".mkv") else "audio/mpeg"
        data = {
            "target_lang": target_lang,
            "source_lang": source_lang or "auto",
            "num_speakers": str(int(num_speakers)),
            "watermark": "false",
        }
        print(f"[Dubber] Submitting to ElevenLabs Dubbing ({source_lang} → {target_lang})...")
        with open(video_path, "rb") as f:
            r = requests.post(self.BASE_URL, headers=self.headers, data=data,
                              files={"file": (video_path.name, f, content_type)}, timeout=300)
        if not r.ok:
            detail = r.json().get("detail", r.text) if r.headers.get("content-type","").startswith("application/json") else r.text
            raise PipelineError(f"ElevenLabs Dubbing create failed: HTTP {r.status_code} — {detail}")
        dubbing_id = r.json().get("dubbing_id")
        if not dubbing_id:
            raise PipelineError("ElevenLabs returned no dubbing_id.")
        print(f"[Dubber] Job created: {dubbing_id}")
        return dubbing_id

    def wait_for_completion(self, dubbing_id: str, poll_interval: int = 12, timeout: int = 1800) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            r = requests.get(f"{self.BASE_URL}/{dubbing_id}", headers=self.headers, timeout=30)
            if not r.ok:
                raise PipelineError(f"ElevenLabs status check failed: HTTP {r.status_code}")
            data = r.json()
            status = data.get("status", "")
            print(f"  [Dubber] Status: {status} ({int(time.time()-start)}s)...")
            if status == "dubbed":
                return data
            if status == "failed":
                raise PipelineError(f"ElevenLabs Dubbing failed: {data.get('error','unknown')}")
            time.sleep(poll_interval)
        raise PipelineError(f"ElevenLabs Dubbing timed out after {timeout}s.")

    def download_audio(self, dubbing_id: str, target_lang: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.BASE_URL}/{dubbing_id}/audio/{target_lang}"
        with requests.get(url, headers=self.headers, stream=True, timeout=120) as r:
            if not r.ok:
                raise PipelineError(f"Failed to download dubbed audio: HTTP {r.status_code}")
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)
        print(f"[Dubber] Dubbed audio → {output_path.name} ({output_path.stat().st_size//1024}KB)")
        return output_path

    def download_transcript(self, dubbing_id: str, target_lang: str, srt_path: Path) -> list:
        srt_path = Path(srt_path)
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.BASE_URL}/{dubbing_id}/transcript/{target_lang}?format_type=srt"
        r = requests.get(url, headers=self.headers, timeout=30)
        if not r.ok:
            raise PipelineError(f"Failed to download transcript: HTTP {r.status_code}")
        srt_path.write_text(r.text, encoding="utf-8")
        cues = parse_srt(r.text)
        print(f"[Dubber] Transcript → {srt_path.name} ({len(cues)} cues)")
        return cues


# ──────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_dubbing(
    video_path: Path,
    job_dir: Path,
    target_lang: str,
    source_lang: str = "auto",
    num_speakers: int = 0,
    source_cues: list = None,
    translated_srt: Path = None,
    tts_model: str = EL_V4_MODEL,
) -> tuple[Path, Path, list]:
    """
    Stage 3 unified entry point.
    Auto-selects Mode A (Dubbing API) or Mode B (Voice-Clone TTS) based on language support.

    Args:
        video_path      : Source video
        job_dir         : Output directory
        target_lang     : e.g. 'mr', 'hi', 'en'
        source_lang     : Source language or 'auto'
        num_speakers    : For Dubbing API
        source_cues     : Parsed source transcript cues (for Mode B translation)
        translated_srt  : Pre-translated SRT in target language (skips translation step)
        tts_model       : ElevenLabs TTS model for Mode B (default: eleven_v4)

    Returns:
        (dubbed_audio_path, dubbed_srt_path, cues)
    """
    video_path = Path(video_path).resolve()
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise PipelineError("Set ELEVENLABS_API_KEY in .env or environment.")

    dubbed_audio = job_dir / f"dubbed_master_{target_lang}.mp3"
    dubbed_srt = job_dir / f"transcript_{target_lang}.srt"
    job_id_path = job_dir / f"el_dubbing_id_{target_lang}.txt"

    # Cache check — ignore PLACEHOLDER files from dry-runs (< 500KB is likely a placeholder)
    REAL_AUDIO_MIN_BYTES = 500 * 1024  # 500KB
    if (dubbed_audio.exists()
            and dubbed_audio.stat().st_size >= REAL_AUDIO_MIN_BYTES
            and dubbed_srt.exists()
            and dubbed_srt.stat().st_size > 100
            and "PLACEHOLDER" not in dubbed_audio.name):
        print(f"[Dubber] Cache hit — reusing {dubbed_audio.name} ({dubbed_audio.stat().st_size // 1024}KB)")
        cues = parse_srt(dubbed_srt.read_text(encoding="utf-8")) if dubbed_srt.exists() else []
        return dubbed_audio, dubbed_srt, cues


    # ── Mode A: ElevenLabs Dubbing API (auto dub, voice-preserved) ────────────
    if target_lang in EL_DUBBING_SUPPORTED:
        print(f"[Dubber] Mode A: ElevenLabs Dubbing API ({target_lang} is supported)")
        client = ElevenLabsDubber(api_key)

        dubbing_id = None
        if job_id_path.exists():
            dubbing_id = job_id_path.read_text().strip()
            print(f"[Dubber] Resuming existing job: {dubbing_id}")

        if not dubbing_id:
            dubbing_id = client.create_job(video_path, target_lang, source_lang, num_speakers)
            job_id_path.write_text(dubbing_id, encoding="utf-8")

        client.wait_for_completion(dubbing_id)
        client.download_audio(dubbing_id, target_lang, dubbed_audio)
        cues = client.download_transcript(dubbing_id, target_lang, dubbed_srt)
        return dubbed_audio, dubbed_srt, cues

    # ── Mode B: Voice-Clone TTS (for unsupported languages like Marathi) ──────
    print(f"[Dubber] Mode B: Voice-Clone TTS ({target_lang} not supported by Dubbing API)")
    print(f"[Dubber] Using ElevenLabs TTS model: {tts_model}")

    # Step B1: Get translated cues
    target_cues: list[Cue] = []
    if translated_srt and Path(translated_srt).exists():
        print(f"[Dubber] Using supplied translated SRT: {Path(translated_srt).name}")
        target_cues = parse_srt(Path(translated_srt).read_text(encoding="utf-8"))
    elif source_cues:
        print(f"[Dubber] Translating {len(source_cues)} cues to {target_lang} via Google Translate...")
        target_cues = _translate_cues_google(source_cues, target_lang)
        # Save translated SRT
        srt_lines = []
        for i, cue in enumerate(target_cues, 1):
            def _fmt(ms):
                h, rem = divmod(int(ms), 3600000)
                m, rem = divmod(rem, 60000)
                s, ms_ = divmod(rem, 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms_:03d}"
            srt_lines.append(f"{i}\n{_fmt(cue.start_ms)} --> {_fmt(cue.end_ms)}\n{cue.text}\n")
        dubbed_srt.write_text("\n".join(srt_lines), encoding="utf-8")
    else:
        raise PipelineError(
            f"Language '{target_lang}' is not supported by ElevenLabs Dubbing API.\n"
            f"For Mode B (Voice-Clone TTS), supply one of:\n"
            f"  --transcript <source_srt>  (will be auto-translated)\n"
            f"  --reviewed-srt <marathi_srt>  (pre-translated Marathi SRT)"
        )

    # Step B2: Extract voice sample & clone
    sample_wav = job_dir / "voice_sample.wav"
    if not sample_wav.exists() or sample_wav.stat().st_size < 1000:
        _extract_voice_sample(video_path, sample_wav, duration_s=20.0)

    voice_name = f"DubClone_{video_path.stem[:20]}"
    voice_id = _ensure_voice_clone(sample_wav, voice_name, api_key)

    # Step B3: TTS each cue
    cue_dir = job_dir / f"tts_cues_{target_lang}"
    cue_dir.mkdir(parents=True, exist_ok=True)
    cue_audio_paths = []
    print(f"[Dubber] Generating {len(target_cues)} TTS cues in {target_lang}...")
    for cue in target_cues:
        cue_out = cue_dir / f"cue_{cue.id:04d}.mp3"
        if not cue_out.exists():
            text = cue.text.strip()
            if text:
                _tts_cue(voice_id, text, cue_out, tts_model, api_key)
        cue_audio_paths.append(cue_out)

    # Step B4: Assemble into full-length audio
    from .scene_analyzer import get_video_info
    info = get_video_info(video_path)
    total_dur = info["duration"]
    print(f"[Dubber] Assembling {len(cue_audio_paths)} cues into {total_dur:.2f}s audio track...")
    _assemble_tts_audio(target_cues, cue_audio_paths, total_dur, dubbed_audio)
    print(f"[Dubber] Mode B complete → {dubbed_audio.name} ({dubbed_audio.stat().st_size//1024}KB)")

    return dubbed_audio, dubbed_srt, target_cues
