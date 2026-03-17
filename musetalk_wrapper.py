"""
MuseTalk v1.5 wrapper — adds precise lip sync to animated video.

Takes an animated video (from LivePortrait) and TTS audio, produces
a lip-synced video where the mouth moves precisely with the speech.

VRAM: ~8-12GB
"""

import logging
import os
import sys
import subprocess
from pathlib import Path

logger = logging.getLogger("storms-worker")

MUSETALK_DIR = "/opt/musetalk"
MUSETALK_MODELS_DIR = "/workspace/models/musetalk"

_musetalk_loaded = False


def ensure_musetalk_models():
    """Download MuseTalk model files if not present."""
    models_dir = Path(MUSETALK_MODELS_DIR)
    marker = models_dir / ".download_complete"

    if marker.exists():
        logger.info("MuseTalk models already downloaded")
        return

    logger.info("Downloading MuseTalk models...")
    models_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download, hf_hub_download

    # Main MuseTalk v1.5 model
    snapshot_download(
        repo_id="TMElyralab/MuseTalkV2",
        local_dir=str(models_dir / "musetalk"),
        local_dir_use_symlinks=False,
        allow_patterns=["musetalkV15/*"],
    )

    # DWPose model for face landmark detection
    snapshot_download(
        repo_id="yzd-v/DWPose",
        local_dir=str(models_dir / "dwpose"),
        local_dir_use_symlinks=False,
    )

    # Face parsing model
    snapshot_download(
        repo_id="jonathandinu/face-parsing",
        local_dir=str(models_dir / "face-parse-bisenet"),
        local_dir_use_symlinks=False,
    )

    # SD VAE for latent decoding
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse",
        local_dir=str(models_dir / "sd-vae-ft-mse"),
        local_dir_use_symlinks=False,
    )

    # Whisper tiny for audio feature extraction
    hf_hub_download(
        repo_id="openai/whisper-tiny",
        filename="model.safetensors",
        local_dir=str(models_dir / "whisper"),
    )

    marker.touch()
    logger.info("MuseTalk models downloaded")


def _load_musetalk():
    """Lazy-load MuseTalk inference components."""
    global _musetalk_loaded
    if _musetalk_loaded:
        return

    ensure_musetalk_models()

    logger.info("Loading MuseTalk pipeline...")

    # Add MuseTalk to Python path
    if MUSETALK_DIR not in sys.path:
        sys.path.insert(0, MUSETALK_DIR)

    _musetalk_loaded = True
    logger.info("MuseTalk pipeline loaded")


def sync_lips(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> str:
    """Run MuseTalk lip sync: overlay precise mouth movements onto video.

    Args:
        video_path: Path to animated video (from LivePortrait)
        audio_path: Path to TTS audio (from F5-TTS)
        output_path: Path to save final lip-synced video with audio

    Returns:
        Path to the lip-synced video
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Input audio not found: {audio_path}")

    logger.info(f"MuseTalk: lip-syncing {video_path} with {audio_path}")

    _load_musetalk()

    # MuseTalk inference via its CLI — most reliable integration path
    output_dir = str(Path(output_path).parent)
    result_name = Path(output_path).stem

    cmd = [
        sys.executable, "-m", "musetalk.inference",
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--output_dir", output_dir,
        "--result_name", result_name,
        "--model_dir", str(Path(MUSETALK_MODELS_DIR) / "musetalk" / "musetalkV15"),
        "--version", "v15",
    ]

    logger.info(f"MuseTalk cmd: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=MUSETALK_DIR,
        env={
            **os.environ,
            "PYTHONPATH": f"{MUSETALK_DIR}:{os.environ.get('PYTHONPATH', '')}",
        },
    )

    if proc.returncode != 0:
        logger.error(f"MuseTalk stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"MuseTalk failed (rc={proc.returncode}): {proc.stderr[-500:]}")

    # Find the output file — MuseTalk may output with its own naming
    output_dir_path = Path(output_dir)
    candidates = [
        output_dir_path / f"{result_name}.mp4",
        output_dir_path / "result.mp4",
    ]

    # Also check for any recently created mp4
    all_mp4s = sorted(
        output_dir_path.glob("*.mp4"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    result_file = None
    for candidate in candidates:
        if candidate.exists():
            result_file = candidate
            break

    if result_file is None and all_mp4s:
        # Use the most recently created mp4 that isn't the input
        for mp4 in all_mp4s:
            if str(mp4) != video_path:
                result_file = mp4
                break

    if result_file is None:
        raise RuntimeError("MuseTalk produced no output video")

    # Rename to expected output path if needed
    if str(result_file) != output_path:
        result_file.rename(output_path)

    # Ensure audio is muxed into the final output
    _ensure_audio_muxed(output_path, audio_path)

    logger.info(f"MuseTalk done: {output_path}")
    return output_path


def _ensure_audio_muxed(video_path: str, audio_path: str):
    """Verify the video has audio; if not, mux it in with FFmpeg."""
    import json

    # Check if video already has audio stream
    probe = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            video_path,
        ],
        capture_output=True,
        text=True,
    )

    has_audio = False
    if probe.returncode == 0:
        try:
            streams = json.loads(probe.stdout).get("streams", [])
            has_audio = any(s.get("codec_type") == "audio" for s in streams)
        except (json.JSONDecodeError, KeyError):
            pass

    if has_audio:
        return

    logger.info("Muxing audio into MuseTalk output...")
    temp_path = video_path + ".tmp.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            temp_path,
        ],
        check=True,
        timeout=120,
        capture_output=True,
    )
    os.replace(temp_path, video_path)


def unload_musetalk():
    """Free MuseTalk from VRAM."""
    global _musetalk_loaded
    if _musetalk_loaded:
        _musetalk_loaded = False

        import torch
        torch.cuda.empty_cache()
        logger.info("MuseTalk unloaded from VRAM")
