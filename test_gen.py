#!/usr/bin/env python3
"""Generate a test video using the LivePortrait + MuseTalk pipeline.

Usage (on GPU server):
    python3 test_gen.py
"""
import os
import sys
import subprocess
import requests

os.chdir("/app")
sys.path.insert(0, "/app")

# Download voice ref
ref = "/workspace/test_ref.mp3"
if not os.path.exists(ref):
    r = requests.get(
        "https://wbyjuruknmtujkizayxy.supabase.co/storage/v1/object/public/media/voice/ashley_ref_birmingham.mp3"
    )
    open(ref, "wb").write(r.content)
    print(f"Downloaded voice ref: {len(r.content)} bytes")

# Stage 1: TTS
from f5_tts_wrapper import generate_tts_f5

print("Stage 1/3: F5-TTS...")
generate_tts_f5(
    "Hey yall! This is Ashley Storms. Let me tell you about freedom.",
    ref,
    "/workspace/test_audio.wav",
)
print("TTS done")

# Stage 2: LivePortrait
from liveportrait_wrapper import animate_portrait, unload_liveportrait, _pick_driving_video

print("Stage 2/3: LivePortrait animation...")
driving = _pick_driving_video()
animate_portrait(
    ref_image_path="/app/ashley_reference.png",
    driving_video_path=driving,
    output_path="/workspace/test_animated.mp4",
)
unload_liveportrait()
print("LivePortrait done")

# Stage 3: MuseTalk lip sync
from musetalk_wrapper import sync_lips, unload_musetalk

print("Stage 3/3: MuseTalk lip sync...")
sync_lips(
    video_path="/workspace/test_animated.mp4",
    audio_path="/workspace/test_audio.wav",
    output_path="/workspace/test_synced.mp4",
)
unload_musetalk()
print("MuseTalk done")

# FFmpeg encode to 9:16
print("Encoding to 9:16...")
subprocess.run(
    [
        "ffmpeg", "-y",
        "-i", "/workspace/test_synced.mp4",
        "-vf", "crop=ih*9/16:ih,scale=1080:1920",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        "/workspace/test_final.mp4",
    ],
    check=True,
)

size = os.path.getsize("/workspace/test_final.mp4") / (1024 * 1024)
print(f"DONE — /workspace/test_final.mp4 ({size:.1f} MB)")
