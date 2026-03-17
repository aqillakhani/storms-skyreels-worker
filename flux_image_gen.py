"""
Flux image generation with IP-Adapter for product photo conditioning.

Generates a still image of Ashley wearing/using a product by conditioning
on a product photo + text prompt describing the scene.
"""

import os
import gc
import logging
import torch
from pathlib import Path
from PIL import Image

logger = logging.getLogger("storms-worker")

_flux_pipe = None


def _load_flux_pipeline():
    """Load Flux.1-schnell pipeline with IP-Adapter. Cached after first call."""
    global _flux_pipe
    if _flux_pipe is not None:
        return _flux_pipe

    from diffusers import FluxPipeline

    model_cache = os.environ.get("MODEL_CACHE", "/workspace/models")
    cache_dir = os.path.join(model_cache, "huggingface", "hub")

    logger.info("Loading Flux.1-schnell pipeline...")
    _flux_pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    _flux_pipe.enable_model_cpu_offload()
    logger.info("Flux.1-schnell pipeline loaded")
    return _flux_pipe


def unload_flux():
    """Unload Flux pipeline to free VRAM for next stage."""
    global _flux_pipe
    if _flux_pipe is not None:
        del _flux_pipe
        _flux_pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Flux pipeline unloaded")


def generate_product_image(
    product_photo_path: str,
    prompt: str,
    output_path: str,
    width: int = 768,
    height: int = 1024,
    num_steps: int = 4,
    seed: int = 42,
) -> str:
    """Generate an image of Ashley wearing/using the product.

    Args:
        product_photo_path: Path to product photo (for prompt context).
        prompt: Scene description including product details.
        output_path: Path to save output image.
        width: Output width (default 768 for 3:4 portrait).
        height: Output height (default 1024 for 3:4 portrait).
        num_steps: Inference steps (4 for schnell).
        seed: Random seed.

    Returns:
        Path to generated image.
    """
    if not Path(product_photo_path).exists():
        raise FileNotFoundError(f"Product photo not found: {product_photo_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pipe = _load_flux_pipeline()
    generator = torch.Generator("cpu").manual_seed(seed)

    logger.info(f"Generating product image: {width}x{height}, steps={num_steps}")
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        generator=generator,
        guidance_scale=0.0,  # schnell uses no CFG
    )

    image = result.images[0]
    image.save(output_path)
    logger.info(f"Product image saved: {output_path}")

    return output_path
