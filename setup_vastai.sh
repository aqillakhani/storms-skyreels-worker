#!/bin/bash
set -ex

mkdir -p /app /workspace

# System deps
apt-get update && apt-get install -y --no-install-recommends git wget ffmpeg python3-pip build-essential

# Clone repo into /app
rm -rf /app/*
git clone --branch master https://github.com/aqillakhani/storms-skyreels-worker.git /app

# Install PyTorch
pip3 install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Core deps
pip3 install fastapi==0.115.0 uvicorn pydantic transformers==4.53.2 tokenizers==0.21.4 accelerate==1.8.1 huggingface_hub numpy==1.26.4 requests soundfile==0.12.1 omegaconf==2.3.0 ftfy==6.3.1 imageio-ffmpeg==0.5.1 imageio easydict pyloudnorm librosa wget==3.2 sentencepiece av

# F5-TTS
pip3 uninstall -y torchcodec 2>/dev/null || true
pip3 install f5-tts
pip3 uninstall -y torchcodec triton 2>/dev/null || true

# Stubs
python3 /app/create_stubs.py

# LivePortrait
git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git /opt/liveportrait || true
pip3 install -r /opt/liveportrait/requirements.txt

# MuseTalk v1.5
git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /opt/musetalk || true
pip3 install -U openmim
mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
pip3 install -r /opt/musetalk/requirements.txt 2>/dev/null || true

echo "DEPS_DONE"

# Start server
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var before running}"
export HF_HUB_DISABLE_XET=1
export MODEL_CACHE=/workspace/models
export HF_HOME=/workspace/models/huggingface
export HF_HUB_CACHE=/workspace/models/huggingface/hub
export TMPDIR=/workspace/tmp
export PYTHONPATH=/opt/liveportrait:/opt/musetalk
mkdir -p /workspace/models /workspace/tmp /workspace/assets/voice /workspace/assets/driving

cd /app
python3 -u server.py
