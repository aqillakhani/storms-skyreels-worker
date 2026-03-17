"""
InsightFace face swap wrapper.

Swaps the face in every frame of a video with Ashley's reference face.
Uses insightface buffalo_l for detection + inswapper_128 for swapping.
"""

import os
import gc
import logging
import cv2
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger("storms-worker")

_face_app = None
_swapper = None


def _load_face_models():
    """Load InsightFace detection + swap models."""
    global _face_app, _swapper
    if _face_app is not None and _swapper is not None:
        return _face_app, _swapper

    import insightface
    from insightface.app import FaceAnalysis

    model_cache = os.environ.get("MODEL_CACHE", "/workspace/models")
    insightface_dir = os.path.join(model_cache, "insightface")
    os.makedirs(insightface_dir, exist_ok=True)

    logger.info("Loading InsightFace models...")
    _face_app = FaceAnalysis(
        name="buffalo_l",
        root=insightface_dir,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Load inswapper model
    swapper_path = os.path.join(insightface_dir, "models", "inswapper_128.onnx")
    if not os.path.exists(swapper_path):
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(swapper_path), exist_ok=True)
        hf_hub_download(
            repo_id="ezioruan/inswapper_128.onnx",
            filename="inswapper_128.onnx",
            local_dir=os.path.dirname(swapper_path),
        )

    _swapper = insightface.model_zoo.get_model(
        swapper_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    logger.info("InsightFace models loaded")
    return _face_app, _swapper


def unload_face_models():
    """Unload face models to free memory."""
    global _face_app, _swapper
    _face_app = None
    _swapper = None
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("InsightFace models unloaded")


def swap_face_in_video(
    video_path: str,
    ref_face_path: str,
    output_path: str,
) -> str:
    """Swap face in every frame of a video with the reference face.

    Args:
        video_path: Input video path.
        ref_face_path: Reference face image (Ashley).
        output_path: Output video path.

    Returns:
        Path to face-swapped video.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(ref_face_path).exists():
        raise FileNotFoundError(f"Reference face not found: {ref_face_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    face_app, swapper = _load_face_models()

    # Get reference face embedding
    ref_img = cv2.imread(ref_face_path)
    ref_faces = face_app.get(ref_img)
    if not ref_faces:
        raise RuntimeError("No face detected in reference image")
    ref_face = sorted(ref_faces, key=lambda x: x.bbox[2] - x.bbox[0], reverse=True)[0]

    # Process video frame by frame
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    swapped_count = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_app.get(frame)
        if faces:
            # Swap the largest face
            target_face = sorted(faces, key=lambda x: x.bbox[2] - x.bbox[0], reverse=True)[0]
            frame = swapper.get(frame, target_face, ref_face, paste_back=True)
            swapped_count += 1

        writer.write(frame)

    cap.release()
    writer.release()

    logger.info(f"Face swap complete: {swapped_count}/{total_frames} frames swapped")
    return output_path
