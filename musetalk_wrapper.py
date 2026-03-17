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

    from huggingface_hub import snapshot_download

    # Main MuseTalk models (v1 config + v1.5 weights) from TMElyralab/MuseTalk
    snapshot_download(
        repo_id="TMElyralab/MuseTalk",
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
    )

    # SD VAE for latent decoding
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse",
        local_dir=str(models_dir / "sd-vae-ft-mse"),
        local_dir_use_symlinks=False,
    )

    # Whisper tiny for audio feature extraction
    snapshot_download(
        repo_id="openai/whisper-tiny",
        local_dir=str(models_dir / "whisper"),
        local_dir_use_symlinks=False,
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

    # MuseTalk inference via scripts/inference.py CLI
    result_dir = str(Path(output_path).parent / "musetalk_out")
    os.makedirs(result_dir, exist_ok=True)
    output_vid_name = Path(output_path).name

    # Build inference config YAML for MuseTalk
    import yaml
    inference_cfg = {
        "task_0": {
            "video_path": video_path,
            "audio_path": audio_path,
        }
    }
    cfg_path = os.path.join(result_dir, "inference_cfg.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(inference_cfg, f)

    # Wrapper script patches torch.load for PyTorch 2.6+ (weights_only=True breaks DWPose)
    wrapper_path = os.path.join(result_dir, "_run_inference.py")
    inference_script = os.path.join(MUSETALK_DIR, "scripts", "inference.py")
    with open(wrapper_path, "w") as f:
        f.write(
            "import torch, sys\n"
            "_orig = torch.load\n"
            "def _patched(*a, **kw):\n"
            "    kw.setdefault('weights_only', False)\n"
            "    return _orig(*a, **kw)\n"
            "torch.load = _patched\n"
            f"sys.argv[0] = '{inference_script}'\n"
            f"exec(compile(open('{inference_script}').read(), '{inference_script}', 'exec'))\n"
        )

    cmd = [
        sys.executable, wrapper_path,
        "--inference_config", cfg_path,
        "--result_dir", result_dir,
        "--output_vid_name", output_vid_name,
        "--unet_model_path", str(Path(MUSETALK_MODELS_DIR) / "musetalkV15" / "unet.pth"),
        "--unet_config", str(Path(MUSETALK_MODELS_DIR) / "musetalk" / "musetalk.json"),
        "--whisper_dir", str(Path(MUSETALK_MODELS_DIR) / "whisper"),
        "--version", "v15",
        "--use_float16",
        "--fps", "25",
    ]

    # Build clean env for subprocess
    sub_env = {**os.environ}
    sub_env["PYTHONPATH"] = f"{MUSETALK_DIR}:{sub_env.get('PYTHONPATH', '')}"
    sub_env.pop("PYTHONHASHSEED", None)  # Avoid invalid hash seed crashes
    # PyTorch 2.6 defaults weights_only=True which breaks DWPose/mmengine loading
    sub_env["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

    logger.info(f"MuseTalk cmd: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=MUSETALK_DIR,
        env=sub_env,
    )

    if proc.returncode != 0:
        logger.error(f"MuseTalk stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"MuseTalk failed (rc={proc.returncode}): {proc.stderr[-500:]}")

    # Find the output file — MuseTalk outputs to result_dir/v15/
    result_dir_path = Path(result_dir)
    v15_dir = result_dir_path / "v15"
    search_dirs = [v15_dir, result_dir_path]

    result_file = None
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        # Check for our named output first
        named = search_dir / output_vid_name
        if named.exists():
            result_file = named
            break
        # Then check for any mp4
        all_mp4s = sorted(
            search_dir.glob("*.mp4"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for mp4 in all_mp4s:
            if str(mp4) != video_path:
                result_file = mp4
                break
        if result_file:
            break

    if result_file is None:
        raise RuntimeError(
            f"MuseTalk produced no output video. "
            f"Searched: {[str(d) for d in search_dirs]}. "
            f"stdout: {proc.stdout[-500:]}"
        )

    # Move to expected output path
    if str(result_file) != output_path:
        import shutil
        shutil.move(str(result_file), output_path)

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
