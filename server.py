"""
Storms LivePortrait + MuseTalk — FastAPI server for Vast.ai GPU deployment.

3-stage pipeline: F5-TTS → LivePortrait → MuseTalk → FFmpeg
Produces pixel-perfect talking head videos from Ashley's reference photo.

Endpoints:
  POST /generate  — Submit a generation job (returns job_id)
  GET  /status/{job_id} — Poll job status
  GET  /health    — Health check
"""

import os
import sys
import uuid
import time
import logging
import tempfile
import subprocess
import threading
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("storms-liveportrait")

# ── Env / Paths ──
MODEL_CACHE = os.environ.get("MODEL_CACHE", "/workspace/models")
DEFAULT_REF_AUDIO = Path("/workspace/assets/voice/ashley_ref.wav")
DEFAULT_REF_TEXT_PATH = Path("/workspace/assets/voice/ashley_ref.txt")
REF_IMAGE = "/app/ashley_reference.png"

# Disable problematic features
os.environ.setdefault("TRITON_DISABLE_AUTOTUNE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HOME", os.path.join(MODEL_CACHE, "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(MODEL_CACHE, "huggingface", "hub"))
os.environ.setdefault("TMPDIR", "/workspace/tmp")

os.makedirs(MODEL_CACHE, exist_ok=True)
os.makedirs("/workspace/tmp", exist_ok=True)
os.makedirs(str(DEFAULT_REF_AUDIO.parent), exist_ok=True)
os.makedirs("/workspace/assets/driving", exist_ok=True)

# ── Job Store (in-memory) ──
jobs: dict[str, dict] = {}

app = FastAPI(title="Storms LivePortrait + MuseTalk", version="2.0.0")


# ── Models ──

class GenerateRequest(BaseModel):
    script: str
    voice_ref_audio_url: Optional[str] = None
    presigned_audio_url: Optional[str] = None
    presigned_video_url: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str


class StatusResponse(BaseModel):
    job_id: str
    status: str  # PENDING, RUNNING, COMPLETED, FAILED
    output: Optional[dict] = None
    error: Optional[str] = None


# ── File Transfer ──

def download_file(url: str, dest_path: str) -> str:
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


# ── Pipeline Worker ──

def run_pipeline(job_id: str, job_input: dict):
    """Run the 3-stage pipeline in a background thread.

    Stages: F5-TTS → LivePortrait → MuseTalk → FFmpeg (encode)
    """
    jobs[job_id]["status"] = "RUNNING"
    jobs[job_id]["started_at"] = time.time()

    script = job_input["script"]
    voice_ref_url = job_input.get("voice_ref_audio_url")
    presigned_audio_url = job_input.get("presigned_audio_url")
    presigned_video_url = job_input.get("presigned_video_url")

    with tempfile.TemporaryDirectory(prefix="storms_") as work_dir:
        try:
            import torch

            # ── Stage 1/3: F5-TTS voice clone ──
            logger.info(f"[{job_id}] Stage 1/3: TTS ({len(script)} chars)")
            if voice_ref_url:
                ref_audio = os.path.join(work_dir, "ref_audio.wav")
                download_file(voice_ref_url, ref_audio)
            elif DEFAULT_REF_AUDIO.exists():
                ref_audio = str(DEFAULT_REF_AUDIO)
            else:
                raise RuntimeError("No voice reference audio available")

            from f5_tts_wrapper import generate_tts_f5
            ref_text = ""
            if DEFAULT_REF_TEXT_PATH.exists():
                ref_text = DEFAULT_REF_TEXT_PATH.read_text().strip()
            audio_path = os.path.join(work_dir, "tts_output.wav")
            generate_tts_f5(script, ref_audio, audio_path, ref_text=ref_text)

            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)
            duration_ms = int(len(audio_data) / sample_rate * 1000)
            logger.info(f"[{job_id}] TTS done: {duration_ms}ms")
            torch.cuda.empty_cache()

            # ── Stage 2/3: LivePortrait head animation ──
            logger.info(f"[{job_id}] Stage 2/3: LivePortrait animation")
            from liveportrait_wrapper import animate_portrait, unload_liveportrait, _pick_driving_video

            driving_video = _pick_driving_video()
            animated_path = os.path.join(work_dir, "animated.mp4")
            animate_portrait(REF_IMAGE, driving_video, animated_path)
            unload_liveportrait()
            logger.info(f"[{job_id}] LivePortrait done")

            # ── Stage 3/3: MuseTalk lip sync ──
            logger.info(f"[{job_id}] Stage 3/3: MuseTalk lip sync")
            from musetalk_wrapper import sync_lips, unload_musetalk

            synced_path = os.path.join(work_dir, "synced.mp4")
            sync_lips(animated_path, audio_path, synced_path)
            unload_musetalk()
            logger.info(f"[{job_id}] MuseTalk done")

            # ── Debug: log intermediate file durations ──
            for label, fpath in [("animated", animated_path), ("synced", synced_path), ("audio", audio_path)]:
                probe = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", fpath],
                    capture_output=True, text=True,
                )
                logger.info(f"[{job_id}] {label} duration: {probe.stdout.strip()}s")

            # Save debug copies
            debug_dir = f"/workspace/debug/{job_id}"
            os.makedirs(debug_dir, exist_ok=True)
            import shutil
            for label, fpath in [("animated.mp4", animated_path), ("synced.mp4", synced_path), ("audio.wav", audio_path)]:
                shutil.copy2(fpath, os.path.join(debug_dir, label))

            # ── FFmpeg encode to 9:16 ──
            logger.info(f"[{job_id}] FFmpeg: encoding final video")
            final_video_path = os.path.join(work_dir, "final.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", synced_path,
                    "-vf", "crop=ih*9/16:ih,scale=1080:1920",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    "-movflags", "+faststart",
                    final_video_path,
                ],
                check=True,
                timeout=120,
                capture_output=True,
            )
            shutil.copy2(final_video_path, os.path.join(debug_dir, "final.mp4"))

            # ── Upload results ──
            if presigned_audio_url:
                upload_to_presigned_url(audio_path, presigned_audio_url, "audio/wav")
            if presigned_video_url:
                upload_to_presigned_url(final_video_path, presigned_video_url, "video/mp4")

            elapsed = time.time() - jobs[job_id]["started_at"]
            logger.info(f"[{job_id}] Pipeline complete in {elapsed:.1f}s")

            jobs[job_id]["status"] = "COMPLETED"
            jobs[job_id]["output"] = {
                "audio_url": presigned_audio_url or "",
                "video_url": presigned_video_url or "",
                "duration_ms": duration_ms,
                "metadata": {
                    "tts_model": "f5-tts",
                    "video_model": "liveportrait+musetalk",
                    "pipeline": "talking_head_v2",
                    "script_length": len(script),
                    "elapsed_seconds": round(elapsed, 1),
                },
            }

        except Exception as e:
            logger.error(f"[{job_id}] Pipeline failed: {e}", exc_info=True)
            jobs[job_id]["status"] = "FAILED"
            jobs[job_id]["error"] = str(e)


# ── Endpoints ──

@app.get("/health")
async def health():
    import torch
    return {
        "status": "ok",
        "python": sys.version,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if torch.cuda.is_available() else 0,
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "RUNNING"),
        "pipeline": "liveportrait+musetalk",
    }


@app.post("/generate", response_model=JobResponse)
async def generate(req: GenerateRequest):
    script = req.script.strip()
    if not script:
        raise HTTPException(status_code=400, detail="Missing 'script'")

    # Only allow 1 concurrent job (single GPU)
    running = [j for j in jobs.values() if j["status"] == "RUNNING"]
    if running:
        raise HTTPException(status_code=429, detail="GPU busy — a job is already running")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "PENDING",
        "created_at": time.time(),
        "output": None,
        "error": None,
    }

    job_input = {
        "script": script,
        "voice_ref_audio_url": req.voice_ref_audio_url,
        "presigned_audio_url": req.presigned_audio_url,
        "presigned_video_url": req.presigned_video_url,
    }

    thread = threading.Thread(target=run_pipeline, args=(job_id, job_input), daemon=True)
    thread.start()

    return JobResponse(job_id=job_id, status="PENDING")


@app.get("/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        output=job.get("output"),
        error=job.get("error"),
    )


# ── Pre-load models at startup ──

def preload_models():
    """Pre-load F5-TTS at startup (other models loaded on-demand and unloaded between stages)."""
    logger.info("Pre-loading F5-TTS at startup...")
    try:
        from f5_tts_wrapper import _load_f5_model
        _load_f5_model()
        logger.info("F5-TTS model pre-loaded")
    except Exception as e:
        logger.error(f"Failed to pre-load F5-TTS: {e}", exc_info=True)


if __name__ == "__main__":
    print("=" * 60)
    print("STORMS LIVEPORTRAIT + MUSETALK — VAST.AI SERVER")
    print(f"Python: {sys.version}")
    print(f"MODEL_CACHE: {MODEL_CACHE}")
    print("Pipeline: F5-TTS → LivePortrait → MuseTalk → FFmpeg")
    print("=" * 60)

    preload_models()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
