"""
F5-TTS wrapper for voice-cloned text-to-speech.

Uses F5-TTS for natural English speech with voice cloning from a reference audio sample.
Falls back to Fish Speech if F5-TTS fails.
"""

import logging
import os
import soundfile as sf
from pathlib import Path

logger = logging.getLogger("storms-worker")

# Lazy-loaded model
_f5_model = None


def _load_f5_model():
    """Load F5-TTS model (cached after first call)."""
    global _f5_model
    if _f5_model is not None:
        return _f5_model

    logger.info("Loading F5-TTS model...")
    from f5_tts.api import F5TTS

    _f5_model = F5TTS()
    logger.info("F5-TTS model loaded")
    return _f5_model


def generate_tts_f5(
    script: str,
    ref_audio_path: str,
    output_path: str,
    ref_text: str = "",
) -> str:
    """Generate speech using F5-TTS with voice cloning.

    Args:
        script: Text to speak
        ref_audio_path: Path to reference audio for voice cloning (10-30s ideal)
        output_path: Path to save output WAV
        ref_text: Transcript of the reference audio (improves cloning quality)

    Returns:
        Path to generated audio file
    """
    if not Path(ref_audio_path).exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    logger.info(f"F5-TTS: generating {len(script)} chars with ref={ref_audio_path}")

    model = _load_f5_model()

    # Generate with voice cloning
    wav, sr, _ = model.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=script,
    )

    # Save output
    sf.write(output_path, wav, sr)

    duration_ms = int(len(wav) / sr * 1000)
    logger.info(f"F5-TTS complete: {output_path} ({duration_ms}ms @ {sr}Hz)")
    return output_path
