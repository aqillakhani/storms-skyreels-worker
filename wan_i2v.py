"""
Wan2.1 Image-to-Video wrapper.

Animates a still image into a short video clip with natural motion
guided by a text prompt (e.g., "woman turns to show outfit").
"""

import os
import gc
import sys
import logging
import time
import torch
import imageio
from pathlib import Path

logger = logging.getLogger("storms-worker")

_wan_pipe = None
_model_path = None


def ensure_wan_models() -> str:
    """Download Wan2.1 I2V model if not present. Returns local model path."""
    global _model_path
    if _model_path is not None:
        return _model_path

    model_cache = os.environ.get("MODEL_CACHE", "/workspace/models")
    cache_dir = os.path.join(model_cache, "huggingface", "hub")
    os.makedirs(cache_dir, exist_ok=True)

    from huggingface_hub import snapshot_download

    logger.info("Ensuring Wan2.1 I2V model is downloaded...")
    _model_path = snapshot_download(
        repo_id="Wan-AI/Wan2.1-I2V-14B-720P",
        cache_dir=cache_dir,
    )
    logger.info(f"Wan2.1 I2V model ready at: {_model_path}")
    return _model_path


def _load_wan_pipeline():
    """Load Wan2.1 I2V pipeline. Cached after first call."""
    global _wan_pipe
    if _wan_pipe is not None:
        return _wan_pipe

    sys.path.insert(0, "/opt/skyreels-v3")
    model_path = ensure_wan_models()

    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import load_image

    logger.info("Loading Wan2.1 I2V pipeline (offload mode)...")
    _wan_pipe = WanImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    _wan_pipe.enable_model_cpu_offload()
    logger.info("Wan2.1 I2V pipeline loaded")
    return _wan_pipe


def unload_wan():
    """Unload Wan pipeline to free VRAM."""
    global _wan_pipe
    if _wan_pipe is not None:
        del _wan_pipe
        _wan_pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Wan2.1 I2V pipeline unloaded")


def generate_video_from_image(
    image_path: str,
    output_path: str,
    prompt: str = "A woman showing off her outfit with natural gestures and movement.",
    num_frames: int = 81,
    fps: int = 16,
    seed: int = 42,
    num_steps: int = 30,
) -> str:
    """Animate a still image into a video clip.

    Args:
        image_path: Path to input still image.
        output_path: Path for output MP4.
        prompt: Motion description prompt.
        num_frames: Number of frames to generate (81 = ~5s at 16fps).
        fps: Frames per second.
        seed: Random seed.
        num_steps: Number of inference steps.

    Returns:
        Path to output video.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pipe = _load_wan_pipeline()

    from diffusers.utils import load_image
    image = load_image(image_path)
    # Resize to 720P aspect ratio
    image = image.resize((720, 1280))

    generator = torch.Generator("cpu").manual_seed(seed)

    logger.info(f"Generating I2V: {num_frames} frames, {num_steps} steps")
    t0 = time.time()

    result = pipe(
        image=image,
        prompt=prompt,
        negative_prompt="blurry, distorted face, extra limbs, low quality",
        num_frames=num_frames,
        guidance_scale=5.0,
        num_inference_steps=num_steps,
        generator=generator,
    )

    elapsed = time.time() - t0
    frames = result.frames[0]
    logger.info(f"I2V complete: {len(frames)} frames in {elapsed:.1f}s")

    # Save as MP4
    imageio.mimwrite(
        output_path,
        frames,
        fps=fps,
        quality=8,
        output_params=["-loglevel", "error"],
    )
    logger.info(f"I2V video saved: {output_path}")
    return output_path
