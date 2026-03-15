"""
Storms SkyReels V3 Worker — Talking Avatar Pipeline
F5-TTS voice cloning + SkyReels V3 TalkingAvatar

STARTUP-SAFE: No heavy imports at module level. All ML imports are deferred
to job time so the worker boots fast and doesn't crash/timeout on init.

Input contract:
  - script: str (required)
  - pipeline: "skyreels" (only mode)
  - voice_ref_audio_url: str | null
  - presigned_audio_url: str
  - presigned_video_url: str

Output contract:
  - audio_url: str
  - video_url: str
  - duration_ms: int
  - metadata: dict
"""

import runpod
import os
import sys
import logging
import tempfile
import subprocess
import time
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("storms-skyreels")

# ── Paths ──
MODEL_CACHE = os.environ.get("MODEL_CACHE", "/runpod-volume/models")
DEFAULT_REF_AUDIO = Path("/runpod-volume/assets/voice/ashley_ref.wav")
DEFAULT_REF_TEXT_PATH = Path("/runpod-volume/assets/voice/ashley_ref.txt")
REF_IMAGE = "/app/ashley_reference.png"


# ═══════════════════════════════════════════════════════════════
# File Transfer
# ═══════════════════════════════════════════════════════════════

def download_file(url: str, dest_path: str) -> str:
    """Download file from URL to local path."""
    logger.info(f"Downloading: {url[:80]}...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    logger.info(f"Downloaded {size_mb:.1f} MB -> {dest_path}")
    return dest_path


def upload_to_presigned_url(file_path: str, presigned_url: str, content_type: str):
    """Upload file to presigned URL via PUT."""
    file_size = os.path.getsize(file_path)
    logger.info(f"Uploading {file_size / (1024*1024):.1f} MB...")
    with open(file_path, "rb") as f:
        resp = requests.put(
            presigned_url,
            data=f,
            headers={"Content-Type": content_type},
            timeout=300,
        )
        resp.raise_for_status()
    logger.info("Upload complete")


# ═══════════════════════════════════════════════════════════════
# Main Handler
# ═══════════════════════════════════════════════════════════════

def handler(event):
    """RunPod serverless handler for SkyReels V3 talking avatar."""
    job_input = event.get("input", {})

    # Diagnostic modes (no heavy imports needed)
    mode = job_input.get("mode")
    if mode:
        return handle_diagnostic(job_input, mode)

    # ── Production pipeline ──
    script = job_input.get("script", "").strip()
    if not script:
        return {"error": "Missing 'script' in input"}

    voice_ref_url = job_input.get("voice_ref_audio_url")
    presigned_audio_url = job_input.get("presigned_audio_url")
    presigned_video_url = job_input.get("presigned_video_url")
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="skyreels_") as work_dir:
        try:
            # ── 1. Prepare reference audio ──
            if voice_ref_url:
                ref_audio = os.path.join(work_dir, "ref_audio.wav")
                download_file(voice_ref_url, ref_audio)
            elif DEFAULT_REF_AUDIO.exists():
                ref_audio = str(DEFAULT_REF_AUDIO)
            else:
                return {"error": "No voice reference audio available"}

            # ── 2. TTS (F5-TTS) — lazy import ──
            logger.info(f"Generating TTS (F5-TTS) for {len(script)} chars...")
            from f5_tts_wrapper import generate_tts_f5
            ref_text = ""
            if DEFAULT_REF_TEXT_PATH.exists():
                ref_text = DEFAULT_REF_TEXT_PATH.read_text().strip()
            audio_path = os.path.join(work_dir, "tts_output.wav")
            generate_tts_f5(script, ref_audio, audio_path, ref_text=ref_text)

            import torch
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)
            duration_ms = int(len(audio_data) / sample_rate * 1000)
            logger.info(f"TTS audio: {duration_ms}ms @ {sample_rate}Hz")

            torch.cuda.empty_cache()

            # ── 3. SkyReels V3 video generation — lazy import ──
            from skyreels_inference import generate_skyreels_video

            logger.info("Generating talking avatar video (SkyReels V3)...")
            raw_video_path = os.path.join(work_dir, "skyreels_out.mp4")
            generate_skyreels_video(
                audio_path=audio_path,
                ref_image_path=REF_IMAGE,
                output_path=raw_video_path,
                prompt="A beautiful young woman speaking to camera with confident expression and natural gestures.",
                resolution="540P",
            )
            torch.cuda.empty_cache()

            # ── 4. Re-encode to vertical 9:16 H.264 ──
            browser_video_path = os.path.join(work_dir, "final.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", raw_video_path,
                    "-vf", "crop=ih*9/16:ih,scale=1080:1920",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-movflags", "+faststart",
                    browser_video_path,
                ],
                check=True,
                timeout=120,
                capture_output=True,
            )
            logger.info(f"Video ready: {browser_video_path}")

            # ── 5. Upload results ──
            if presigned_audio_url:
                upload_to_presigned_url(audio_path, presigned_audio_url, "audio/wav")
            if presigned_video_url:
                upload_to_presigned_url(browser_video_path, presigned_video_url, "video/mp4")

            elapsed = time.time() - t0
            logger.info(f"Pipeline complete in {elapsed:.1f}s")

            return {
                "audio_url": presigned_audio_url or "",
                "video_url": presigned_video_url or "",
                "duration_ms": duration_ms,
                "metadata": {
                    "tts_model": "f5-tts",
                    "video_model": "skyreels_v3",
                    "pipeline": "skyreels",
                    "num_frames": int(duration_ms / 1000 * 25),
                    "script_length": len(script),
                    "elapsed_seconds": round(elapsed, 1),
                },
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Subprocess failed: {e.cmd}\nstderr: {e.stderr}")
            return {"error": f"Pipeline subprocess failed: {e.stderr[-1500:] if e.stderr else 'N/A'}"}
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Diagnostic Modes
# ═══════════════════════════════════════════════════════════════

def handle_diagnostic(job_input, mode):
    if mode == "echo":
        return {
            "status": "ok",
            "message": "SkyReels worker alive!",
            "python": sys.version,
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        }

    if mode == "check_gpu":
        try:
            import torch
            return {
                "status": "ok",
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1) if torch.cuda.is_available() else 0,
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    if mode == "check_imports":
        results = {}
        for mod_name in ["torch", "diffusers", "transformers", "f5_tts", "soundfile", "imageio"]:
            try:
                __import__(mod_name)
                results[mod_name] = "ok"
            except Exception as e:
                results[mod_name] = f"FAIL: {e}"
        # Check SkyReels V3 separately (needs sys.path)
        try:
            sys.path.insert(0, "/opt/skyreels-v3")
            from skyreels_v3.configs import WAN_CONFIGS
            results["skyreels_v3"] = "ok"
        except Exception as e:
            results["skyreels_v3"] = f"FAIL: {e}"
        return {"status": "ok", "imports": results}

    if mode == "check_disk":
        import shutil
        paths = {
            "/": None,
            "/runpod-volume": None,
            "/app": None,
        }
        for p in paths:
            try:
                usage = shutil.disk_usage(p)
                paths[p] = {
                    "total_gb": round(usage.total / (1024**3), 1),
                    "free_gb": round(usage.free / (1024**3), 1),
                }
            except Exception as e:
                paths[p] = str(e)
        return {"status": "ok", "disk": paths}

    if mode == "setup_models":
        try:
            import torch
            sys.path.insert(0, "/opt/skyreels-v3")
            from skyreels_v3.modules import download_model
            os.environ.setdefault("HF_HOME", os.path.join(MODEL_CACHE, "huggingface"))
            model_path = download_model("Skywork/SkyReels-V3-TalkingAvatar")
            return {"status": "ok", "model_path": str(model_path)}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    return {"status": "ok", "message": f"Unknown mode: {mode}"}


# ═══════════════════════════════════════════════════════════════
# Startup — MINIMAL, no heavy imports
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("STORMS SKYREELS V3 WORKER — STARTUP-SAFE")
print(f"Python: {sys.version}")
print(f"MODEL_CACHE: {MODEL_CACHE}")
print("=" * 60)

# Set HF cache to network volume so models persist across cold starts
os.environ.setdefault("HF_HOME", os.path.join(MODEL_CACHE, "huggingface"))
os.makedirs(MODEL_CACHE, exist_ok=True)

# Create voice assets directory
os.makedirs(str(DEFAULT_REF_AUDIO.parent), exist_ok=True)

runpod.serverless.start({"handler": handler})
