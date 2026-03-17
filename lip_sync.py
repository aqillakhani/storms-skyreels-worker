"""
Wav2Lip lip sync wrapper.

Syncs lip movements in a video to match audio speech.
Uses Wav2Lip model for frame-by-frame mouth region modification.
"""

import os
import gc
import logging
import subprocess
import torch
from pathlib import Path

logger = logging.getLogger("storms-worker")


def sync_lips(
    video_path: str,
    audio_path: str,
    output_path: str,
    wav2lip_path: str = "/opt/Wav2Lip",
) -> str:
    """Sync lips in video to match audio using Wav2Lip.

    Args:
        video_path: Input video (face-swapped).
        audio_path: TTS audio to sync lips to.
        output_path: Output video with synced lips.
        wav2lip_path: Path to Wav2Lip repo.

    Returns:
        Path to lip-synced video.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model_cache = os.environ.get("MODEL_CACHE", "/workspace/models")
    checkpoint = os.path.join(model_cache, "wav2lip", "wav2lip_gan.pth")

    if not os.path.exists(checkpoint):
        _download_wav2lip_model(checkpoint)

    logger.info("Running Wav2Lip lip sync...")
    result = subprocess.run(
        [
            "python3", os.path.join(wav2lip_path, "inference.py"),
            "--checkpoint_path", checkpoint,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--resize_factor", "1",
            "--nosmooth",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=wav2lip_path,
    )

    if result.returncode != 0:
        logger.error(f"Wav2Lip failed: {result.stderr[-1500:]}")
        raise RuntimeError(f"Wav2Lip failed: {result.stderr[-500:]}")

    logger.info(f"Lip sync complete: {output_path}")
    return output_path


def _download_wav2lip_model(checkpoint_path: str):
    """Download Wav2Lip GAN model."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    from huggingface_hub import hf_hub_download
    logger.info("Downloading Wav2Lip GAN model...")
    downloaded = hf_hub_download(
        repo_id="camenduru/Wav2Lip",
        filename="checkpoints/wav2lip_gan.pth",
        local_dir=os.path.dirname(checkpoint_path),
    )
    # Move from subdirectory to expected path
    import shutil
    if downloaded != checkpoint_path and os.path.exists(downloaded):
        shutil.move(downloaded, checkpoint_path)
    logger.info(f"Wav2Lip model saved: {checkpoint_path}")
