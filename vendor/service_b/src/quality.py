"""Measured audio mastering; reference comparisons are not speaker-identity scores."""
import json
import math
from pathlib import Path
import re
import subprocess
import numpy as np
from .audio import ffmpeg
from .common import PipelineError, write_json


def command(args):
    result = subprocess.run([ffmpeg(), "-hide_banner", "-nostdin", *args],
                            capture_output=True, timeout=180,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise PipelineError("FFmpeg quality analysis/mastering failed.")
    return result


def loudness(path):
    result = command(["-i", str(path), "-af",
                      "loudnorm=I=-14:TP=-1.5:LRA=7:print_format=json", "-f", "null", "-"])
    blocks = re.findall(r"\{[^{}]+\}", result.stderr.decode("utf-8", errors="replace"))
    if not blocks:
        raise PipelineError("No loudness measurements returned.")
    stats = json.loads(blocks[-1])
    return {key: float(stats[key]) for key in ("input_i", "input_tp", "input_lra", "input_thresh")}


def signal_metrics(path):
    result = command(["-v", "error", "-i", str(path), "-ar", "24000",
                      "-ac", "1", "-f", "f32le", "-"])
    samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)
    if len(samples) < 480:
        raise PipelineError("Audio is too short for signal checks.")
    blocks = samples[:len(samples)//480*480].reshape(-1, 480)
    power = np.mean(blocks * blocks, axis=1)
    levels = 10 * np.log10(np.maximum(power, 1e-12))
    speech_reference_db = float(np.percentile(levels, 80))
    speech_mask = levels > speech_reference_db - 18
    noise_mask = levels < speech_reference_db - 35
    speech_power = float(np.mean(power[speech_mask])) if speech_mask.any() else 0.0
    noise_power = float(np.mean(power[noise_mask])) if noise_mask.any() else 0.0
    # Quiet frames are only candidates for noise; the files have no noise-only ground truth.
    estimated_snr = None
    if np.count_nonzero(noise_mask) >= 5 and noise_power > 1e-12 and speech_power > noise_power:
        estimated_snr = 10 * math.log10((speech_power - noise_power) / noise_power)
    active_indices = np.flatnonzero(speech_mask)
    internal_longest = 0
    current = 0
    if len(active_indices):
        for quiet in ~speech_mask[active_indices[0]:active_indices[-1] + 1]:
            current = current + 1 if quiet else 0
            internal_longest = max(internal_longest, current)
    third = max(1, len(levels)//3)
    beginning = levels[:third][speech_mask[:third]]
    ending = levels[-third:][speech_mask[-third:]]
    drift = float(np.median(ending) - np.median(beginning)) if len(beginning) and len(ending) else None
    return {
        "duration_seconds": len(samples)/24000,
        "rms_dbfs": float(10*np.log10(max(float(np.mean(samples*samples)), 1e-12))),
        "sample_peak_dbfs": float(20*np.log10(max(float(np.max(abs(samples))), 1e-12))),
        "clipped_sample_count": int(np.count_nonzero(abs(samples) >= 0.999969)),
        "estimated_snr_db": estimated_snr,
        "estimated_noise_dbfs": float(10*math.log10(max(noise_power, 1e-12))) if noise_mask.any() else None,
        "noise_candidate_seconds": int(np.count_nonzero(noise_mask))*0.02,
        "internal_longest_low_energy_gap_seconds": internal_longest*0.02,
        "first_to_last_speech_level_drift_db": drift,
        "snr_limitation": "Blind estimate from low-energy frames, not a clean-reference SNR or voice-likeness score. Null means insufficient trustworthy quiet energy.",
    }


def analyze(path):
    return {"loudness": loudness(path), "signal": signal_metrics(path)}


def error(metrics, target_lufs=-14, peak_ceiling=-1.0):
    loud = metrics["loudness"]
    return abs(loud["input_i"]-target_lufs) + 4*max(0, loud["input_tp"]-peak_ceiling)


def master_loop(source, folder, target_lufs=-14, attempts=3):
    """Bounded measure/process/measure loop; each attempt starts from the original."""
    source, folder = Path(source), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    before = analyze(source)
    history = [{"iteration": 0, "path": str(source), "metrics": before,
                "loudness_peak_error": error(before, target_lufs)}]
    target_adjustment = 0.0
    best = None
    for attempt in range(1, attempts+1):
        # Mild compression reduces quiet/loud swings; no denoising without evidence.
        preparation = "acompressor=threshold=0.1:ratio=1.8:attack=15:release=180:makeup=1"
        requested = target_lufs + target_adjustment
        normalization = f"loudnorm=I={requested:.3f}:TP=-1.5:LRA=7"
        first = command(["-i", str(source), "-af", preparation+","+normalization+":print_format=json",
                         "-f", "null", "-"])
        blocks = re.findall(r"\{[^{}]+\}", first.stderr.decode("utf-8", errors="replace"))
        stats = json.loads(blocks[-1])
        second = normalization + (
            f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
            f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
            f":offset={stats['target_offset']}:linear=true"
        )
        output = folder/f"master_iteration_{attempt}.wav"
        command(["-y", "-i", str(source), "-af", preparation+","+second,
                 "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(output)])
        metrics = analyze(output)
        score = error(metrics, target_lufs)
        row = {"iteration": attempt, "path": str(output), "metrics": metrics,
               "loudness_peak_error": score}
        history.append(row)
        if best is None or score < best["loudness_peak_error"]:
            best = row
        write_json(folder/"mastering_loop.json", history)
        if score <= 0.25 and metrics["signal"]["clipped_sample_count"] == 0:
            break
        target_adjustment += target_lufs - metrics["loudness"]["input_i"]
        target_adjustment = max(-2, min(2, target_adjustment))
    return best, history
