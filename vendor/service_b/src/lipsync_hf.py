import os
from pathlib import Path
import shutil
from .audio import ffmpeg
from .common import PipelineError

DEFAULT_SPACES = [
    "fffiloni/LatentSync",
    "manavisrani07/gradio-lipsync-wav2lip"
]


def sync_clip(video_path, audio_path, output_path, space_name=None, hf_token=None, dry_run=False):
    """
    Sends a single video cut and audio segment to a free Hugging Face Space.
    Downloads and writes the resulting lip-synced video to output_path.
    """
    video_path = Path(video_path).resolve()
    audio_path = Path(audio_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        from .scene_analyzer import mux_audio_video
        mux_audio_video(video_path, audio_path, output_path)
        return output_path

    token = hf_token or os.environ.get("HF_TOKEN") or None
    target_space = space_name or os.environ.get("HF_LIPSYNC_SPACE") or DEFAULT_SPACES[0]

    try:
        from gradio_client import Client, handle_file
    except ImportError:
        raise PipelineError("gradio_client is required. Run 'pip install gradio_client'.")

    last_error = None
    spaces_to_try = [target_space] + [s for s in DEFAULT_SPACES if s != target_space]

    for current_space in spaces_to_try:
        try:
            print(f"[HF Lip-Sync] Connecting to Space: {current_space}...")
            client = Client(current_space, token=token)
            
            result = None
            if "wav2lip" in current_space.lower():
                # Wav2Lip endpoint
                res = client.predict(
                    video=handle_file(str(video_path)),
                    audio=handle_file(str(audio_path)),
                    checkpoint="wav2lip_gan",
                    no_smooth=0,
                    resize_factor=1,
                    pad_top=0,
                    pad_bottom=0,
                    pad_left=0,
                    api_name="/generate"
                )
                if isinstance(res, dict) and "video" in res:
                    result = res["video"]
                else:
                    result = res
            else:
                # LatentSync endpoint
                try:
                    result = client.predict(
                        handle_file(str(video_path)),
                        handle_file(str(audio_path)),
                        api_name="/generate_lip_sync_video"
                    )
                except Exception:
                    result = client.predict(
                        handle_file(str(video_path)),
                        handle_file(str(audio_path)),
                        api_name="/predict"
                    )

            # Gradio returns either dict, tuple, list, or path string
            if isinstance(result, dict) and "video" in result:
                result_file = result["video"]
            elif isinstance(result, (tuple, list)):
                result_file = result[0]
            else:
                result_file = result

            if result_file and Path(result_file).exists():
                shutil.copyfile(result_file, output_path)
                print(f"[HF Lip-Sync] Successfully synchronized via {current_space}!")
                return output_path
            else:
                raise PipelineError(f"Space returned empty result file: {result}")

        except Exception as e:
            last_error = e
            print(f"[HF Lip-Sync] Notice on {current_space}: {e}. Trying fallback if available...")

    # If all spaces fail, provide actionable error guidance
    raise PipelineError(
        f"Free Hugging Face lip-sync failed across attempted spaces. Error: {last_error}.\n\n"
        "How to resolve:\n"
        "Open .env and set HF_TOKEN=hf_... to bypass anonymous duration caps."
    )
