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
pip3 install fastapi==0.115.0 uvicorn pydantic diffusers==0.34.0 transformers==4.53.2 tokenizers==0.21.4 accelerate==1.8.1 huggingface_hub numpy==1.26.4 requests soundfile==0.12.1 omegaconf==2.3.0 ftfy==6.3.1 imageio-ffmpeg==0.5.1 imageio easydict pyloudnorm librosa kornia wget==3.2 sentencepiece av xfuser

# F5-TTS
pip3 uninstall -y torchcodec 2>/dev/null || true
pip3 install f5-tts
pip3 uninstall -y torchcodec triton 2>/dev/null || true

# Stubs
python3 /app/create_stubs.py

# Clone SkyReels V3
git clone --depth 1 https://github.com/SkyworkAI/SkyReels-V3.git /opt/skyreels-v3 || true

# InsightFace + Wav2Lip
pip3 install insightface onnxruntime-gpu opencv-python-headless batch-face
git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /opt/Wav2Lip || true

echo "DEPS_DONE"

# Start server
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var before running}"
export HF_HUB_DISABLE_XET=1
export MODEL_CACHE=/workspace/models
export HF_HOME=/workspace/models/huggingface
export HF_HUB_CACHE=/workspace/models/huggingface/hub
export TMPDIR=/workspace/tmp
export PYTHONPATH=/opt/skyreels-v3
mkdir -p /workspace/models /workspace/tmp /workspace/assets/voice

cd /app
python3 -u server.py
