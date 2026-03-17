"""
Lightweight face detection check using OpenCV Haar cascade.

Used to validate that a generated image contains a detectable face
before proceeding to downstream face swap / lip sync stages.
"""

import logging
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger("storms-worker")

_face_cascade = None


def _get_cascade():
    """Load OpenCV Haar cascade for face detection (lightweight, no GPU)."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def image_has_face(image_path: str, min_face_size: int = 60) -> bool:
    """Check if an image contains at least one detectable face.

    Args:
        image_path: Path to image file.
        min_face_size: Minimum face size in pixels.

    Returns:
        True if at least one face is detected.
    """
    if not Path(image_path).exists():
        return False

    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_face_size, min_face_size),
    )

    count = len(faces)
    logger.info(f"Face check: {count} face(s) detected in {Path(image_path).name}")
    return count > 0
