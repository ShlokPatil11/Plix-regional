"""
Stage 1 — Transcript Extraction
Extracts word-level timestamped transcript from source video using faster-whisper (local).
Falls back to OpenAI Whisper API or a pre-supplied SRT if whisper is unavailable.
Cache key: sha256 of first 2MB of video file (fast, deterministic, avoids re-transcription).
"""
import hashlib
import json
import subprocess
from pathlib import Path

from .common import PipelineError
from .srt_parser import parse_srt


def _video_hash(video_path: Path, chunk_size: int = 2 * 1024 * 1024) -> str:
    """Fast partial hash of video file for cache keying."""
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        data = f.read(chunk_size)
        h.update(data)
    h.update(str(video_path.stat().st_size).encode())
    return h.hexdigest()[:16]


def _extract_audio_for_whisper(video_path: Path, out_wav: Path) -> Path:
    """Extract 16kHz mono WAV for Whisper input."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out_wav),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise PipelineError(f"Audio extraction for transcription failed: {r.stderr}")
    return out_wav


def _transcribe_with_faster_whisper(audio_path: Path, source_lang: str = None) -> list[dict]:
    """
    Transcribe using faster-whisper (local, no API cost).
    Returns list of word dicts: {word, start, end}.
    Raises ImportError if faster-whisper is not installed.
    """
    from faster_whisper import WhisperModel  # noqa: PLC0415

    model = WhisperModel("base", compute_type="int8")
    lang = None if not source_lang or source_lang == "auto" else source_lang
    segments, _ = model.transcribe(str(audio_path), language=lang, word_timestamps=True)

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
        else:
            # Fallback: use segment-level timing
            words.append({"word": seg.text.strip(), "start": seg.start, "end": seg.end})
    return words


def _words_to_srt(words: list[dict]) -> str:
    """Convert word list to SRT cue format (one cue per natural pause group)."""

    def _fmt(secs: float) -> str:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        ms = int(round((secs % 1) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # Group words into sentences by gaps > 0.5s or punctuation
    groups = []
    current: list[dict] = []
    for w in words:
        if current:
            gap = w["start"] - current[-1]["end"]
            text = current[-1]["word"]
            ends_sentence = text.endswith((".", "!", "?", "।"))
            if gap > 0.6 or ends_sentence:
                groups.append(current)
                current = []
        current.append(w)
    if current:
        groups.append(current)

    lines = []
    for i, group in enumerate(groups, 1):
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(w["word"] for w in group).strip()
        lines.append(f"{i}\n{_fmt(start)} --> {_fmt(end)}\n{text}\n")
    return "\n".join(lines)


def extract_transcript(
    video_path: Path,
    job_dir: Path,
    source_lang: str = "auto",
    supplied_srt: Path = None,
) -> tuple[Path, list[dict]]:
    """
    Main entry point for Stage 1.

    Returns:
        (srt_path, words)
        srt_path  — Path to the saved SRT file
        words     — List of {word, start, end} dicts
    """
    video_path = Path(video_path).resolve()
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    lang_tag = source_lang if source_lang and source_lang != "auto" else "src"
    srt_path = job_dir / f"transcript_{lang_tag}.srt"
    words_path = job_dir / f"transcript_{lang_tag}.json"
    hash_path = job_dir / f"transcript_{lang_tag}.hash"

    # --- Use supplied SRT if provided ---
    if supplied_srt and Path(supplied_srt).exists():
        text = Path(supplied_srt).read_text(encoding="utf-8")
        cues = parse_srt(text)
        words = [
            {"word": c.text.replace("\n", " ").strip(),
             "start": c.start_ms / 1000.0,
             "end": c.end_ms / 1000.0}
            for c in cues
        ]
        srt_path.write_text(text, encoding="utf-8")
        words_path.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Transcriber] Using supplied SRT: {supplied_srt.name} ({len(cues)} cues)")
        return srt_path, words

    # --- Check cache ---
    vid_hash = _video_hash(video_path)
    if srt_path.exists() and words_path.exists() and hash_path.exists():
        if hash_path.read_text().strip() == vid_hash:
            words = json.loads(words_path.read_text(encoding="utf-8"))
            print(f"[Transcriber] Cache hit — {len(words)} words loaded from {srt_path.name}")
            return srt_path, words

    # --- Transcribe ---
    print(f"[Transcriber] Extracting audio for transcription...")
    wav_path = job_dir / "whisper_input.wav"
    _extract_audio_for_whisper(video_path, wav_path)

    words = []
    try:
        print(f"[Transcriber] Running faster-whisper (local)...")
        words = _transcribe_with_faster_whisper(wav_path, source_lang)
        print(f"[Transcriber] Transcribed {len(words)} words.")
    except ImportError:
        raise PipelineError(
            "faster-whisper is not installed. Run:\n"
            "  .venv\\Scripts\\pip install faster-whisper\n"
            "Or supply a pre-made SRT with --transcript."
        )

    # Clean up temp audio
    wav_path.unlink(missing_ok=True)

    # --- Save ---
    srt_text = _words_to_srt(words)
    srt_path.write_text(srt_text, encoding="utf-8")
    words_path.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")
    hash_path.write_text(vid_hash, encoding="utf-8")
    print(f"[Transcriber] Saved transcript → {srt_path.name}")
    return srt_path, words
