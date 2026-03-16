"""
SkyReels V3 Talking Avatar inference wrapper.

Wraps SkyReels V3 TalkingAvatarPipeline into a simple function:
    generate_skyreels_video(audio_path, ref_image_path, output_path)
"""

import os
import sys
import logging
import time
import subprocess
import torch
import imageio
from pathlib import Path

# Add SkyReels V3 to path
sys.path.insert(0, "/opt/skyreels-v3")

from skyreels_v3.configs import WAN_CONFIGS
from skyreels_v3.modules import download_model
from skyreels_v3.pipelines import TalkingAvatarPipeline
from skyreels_v3.utils.avatar_preprocess import preprocess_audio

logger = logging.getLogger("storms-worker")

# Lazy-loaded pipeline
_skyreels_pipe = None
_model_path = None


def ensure_skyreels_models() -> str:
    """Download SkyReels V3 TalkingAvatar model if not present. Returns local model path."""
    global _model_path
    if _model_path is not None:
        return _model_path

    # Use network volume for model cache so models persist across cold starts
    model_cache = os.environ.get("MODEL_CACHE", "/runpod-volume/models")
    os.environ.setdefault("HF_HOME", os.path.join(model_cache, "huggingface"))

    logger.info("Ensuring SkyReels V3 TalkingAvatar model is downloaded...")
    # Use explicit cache_dir to ensure download goes to network volume
    from huggingface_hub import snapshot_download
    cache_dir = os.path.join(model_cache, "huggingface", "hub")
    os.makedirs(cache_dir, exist_ok=True)
    _model_path = snapshot_download(
        repo_id="Skywork/SkyReels-V3-A2V-19B",
        cache_dir=cache_dir,
    )
    logger.info(f"SkyReels V3 model ready at: {_model_path}")
    return _model_path


def _load_pipeline():
    """Load SkyReels V3 TalkingAvatarPipeline (cached after first call)."""
    global _skyreels_pipe

    if _skyreels_pipe is not None:
        return _skyreels_pipe

    model_path = ensure_skyreels_models()

    logger.info("Loading SkyReels V3 TalkingAvatarPipeline (full VRAM mode)...")
    config = WAN_CONFIGS["talking-avatar-19B"]

    _skyreels_pipe = TalkingAvatarPipeline(
        config=config,
        model_path=model_path,
        device_id=0,
        rank=0,
        use_usp=False,
        offload=False,
        low_vram=False,
    )

    logger.info("SkyReels V3 pipeline loaded")
    return _skyreels_pipe


def generate_skyreels_video(
    audio_path: str,
    ref_image_path: str,
    output_path: str,
    prompt: str = "A woman speaking directly to camera with natural expressions and gestures.",
    resolution: str = "540P",
    seed: int = 42,
    sampling_steps: int = 4,
) -> str:
    """Generate talking avatar video using SkyReels V3.

    Args:
        audio_path: Path to audio file (WAV or MP3)
        ref_image_path: Path to reference portrait image
        output_path: Path for output MP4 (with audio muxed in)
        prompt: Text prompt describing the scene
        resolution: Output resolution (480P, 540P, 720P)
        seed: Random seed
        sampling_steps: Number of sampling steps (4 default)

    Returns:
        Path to output video file with audio
    """
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not Path(ref_image_path).exists():
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model_path = ensure_skyreels_models()
    pipe = _load_pipeline()

    # Prepare input data
    input_data = {
        "prompt": prompt,
        "cond_image": ref_image_path,
        "cond_audio": {"person1": audio_path},
    }

    # Preprocess audio
    logger.info("Preprocessing audio for SkyReels V3...")
    input_data, _ = preprocess_audio(model_path, input_data, "processed_audio")

    # Generate video
    logger.info(f"Generating SkyReels V3 video: {resolution}, seed={seed}, steps={sampling_steps}")
    t0 = time.time()

    video_frames = pipe.generate(
        input_data=input_data,
        size_buckget=resolution,
        motion_frame=5,
        frame_num=41,
        drop_frame=6,
        shift=11,
        text_guide_scale=1.0,
        audio_guide_scale=1.0,
        seed=seed,
        sampling_steps=sampling_steps,
        max_frames_num=5000,
    )

    elapsed = time.time() - t0
    logger.info(f"SkyReels V3 inference complete in {elapsed:.1f}s, {len(video_frames)} frames")

    # Save raw video (no audio)
    raw_path = output_path.replace(".mp4", "_raw.mp4")
    imageio.mimwrite(
        raw_path,
        video_frames,
        fps=25,
        quality=8,
        output_params=["-loglevel", "error"],
    )

    # Mux audio into video with ffmpeg
    audio_for_mux = input_data.get("video_audio", audio_path)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", raw_path,
            "-i", audio_for_mux,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-shortest",
            output_path,
        ],
        check=True,
        timeout=120,
        capture_output=True,
    )

    # Cleanup raw file
    if os.path.exists(raw_path):
        os.remove(raw_path)

    logger.info(f"SkyReels V3 output: {output_path}")
    return output_path
