"""
LivePortrait wrapper — animates Ashley's reference photo with natural head motion.

Uses a pre-recorded "driving" motion video to transfer head movement (nods, turns, gestures)
onto the reference photo. Only motion is transferred, NOT identity.

VRAM: ~4-6GB
"""

import logging
import os
import sys
import subprocess
from pathlib import Path

logger = logging.getLogger("storms-worker")

LIVEPORTRAIT_DIR = "/opt/liveportrait"
LIVEPORTRAIT_WEIGHTS_DIR = "/workspace/models/liveportrait"
DRIVING_VIDEOS_DIR = "/workspace/assets/driving"

_liveportrait_pipeline = None


def ensure_liveportrait_models():
    """Download LivePortrait pretrained weights from HuggingFace if not present."""
    weights_dir = Path(LIVEPORTRAIT_WEIGHTS_DIR)
    marker = weights_dir / ".download_complete"

    if marker.exists():
        logger.info("LivePortrait weights already downloaded")
        return

    logger.info("Downloading LivePortrait weights from HuggingFace...")
    weights_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="KwaiVGI/LivePortrait",
        local_dir=str(weights_dir),
        local_dir_use_symlinks=False,
    )

    marker.touch()
    logger.info("LivePortrait weights downloaded")


def _load_liveportrait():
    """Lazy-load LivePortrait inference pipeline (cached after first call)."""
    global _liveportrait_pipeline
    if _liveportrait_pipeline is not None:
        return _liveportrait_pipeline

    ensure_liveportrait_models()

    logger.info("Loading LivePortrait pipeline...")

    # Add LivePortrait to Python path
    if LIVEPORTRAIT_DIR not in sys.path:
        sys.path.insert(0, LIVEPORTRAIT_DIR)

    from src.config.inference_config import InferenceConfig
    from src.live_portrait_pipeline import LivePortraitPipeline

    inference_cfg = InferenceConfig(
        flag_pasteback=True,
        flag_do_crop=True,
        flag_do_rot=True,
    )

    _liveportrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=None,
    )

    logger.info("LivePortrait pipeline loaded")
    return _liveportrait_pipeline


def _pick_driving_video() -> str:
    """Pick a random driving motion video from the assets directory."""
    import random

    driving_dir = Path(DRIVING_VIDEOS_DIR)
    if not driving_dir.exists():
        raise FileNotFoundError(
            f"Driving videos directory not found: {DRIVING_VIDEOS_DIR}. "
            "Please create it with 4-5 motion clips."
        )

    videos = list(driving_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(
            f"No .mp4 driving videos found in {DRIVING_VIDEOS_DIR}"
        )

    chosen = random.choice(videos)
    logger.info(f"Selected driving video: {chosen.name}")
    return str(chosen)


def animate_portrait(
    ref_image_path: str,
    driving_video_path: str,
    output_path: str,
) -> str:
    """Run LivePortrait inference: animate reference photo with driving video motion.

    Args:
        ref_image_path: Path to Ashley's reference photo
        driving_video_path: Path to driving motion video
        output_path: Path to save animated output video

    Returns:
        Path to the animated video (no audio)
    """
    if not Path(ref_image_path).exists():
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
    if not Path(driving_video_path).exists():
        raise FileNotFoundError(f"Driving video not found: {driving_video_path}")

    logger.info(
        f"LivePortrait: animating {ref_image_path} "
        f"with driving={Path(driving_video_path).name}"
    )

    pipeline = _load_liveportrait()

    # Run inference
    output_dir = str(Path(output_path).parent)
    pipeline.execute(
        input_source_path=ref_image_path,
        input_driving_path=driving_video_path,
        output_dir=output_dir,
    )

    # LivePortrait outputs to a default filename in output_dir —
    # find and rename to our expected output path
    output_dir_path = Path(output_dir)
    generated_files = sorted(
        output_dir_path.glob("*.mp4"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not generated_files:
        raise RuntimeError("LivePortrait produced no output video")

    latest = generated_files[0]
    if str(latest) != output_path:
        latest.rename(output_path)

    logger.info(f"LivePortrait done: {output_path}")
    return output_path


def unload_liveportrait():
    """Free LivePortrait from VRAM."""
    global _liveportrait_pipeline
    if _liveportrait_pipeline is not None:
        del _liveportrait_pipeline
        _liveportrait_pipeline = None

        import torch
        torch.cuda.empty_cache()
        logger.info("LivePortrait unloaded from VRAM")
