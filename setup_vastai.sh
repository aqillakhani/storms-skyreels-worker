#!/bin/bash
set -ex

mkdir -p /app /workspace

# ── System deps ──
apt-get update && apt-get install -y --no-install-recommends git wget ffmpeg python3-pip build-essential

# ── Clone repo into /app ──
rm -rf /app/*
git clone --branch master https://github.com/aqillakhani/storms-skyreels-worker.git /app

# ── Install PyTorch 2.6 with CUDA 12.4 ──
pip3 install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# ── Core deps ──
pip3 install \
    fastapi==0.115.0 uvicorn pydantic \
    transformers==4.53.2 tokenizers==0.21.4 accelerate==1.8.1 huggingface_hub \
    numpy==1.26.4 requests soundfile==0.12.1 \
    omegaconf==2.3.0 ftfy==6.3.1 imageio-ffmpeg==0.5.1 imageio \
    easydict pyloudnorm librosa wget==3.2 sentencepiece av yaml gdown

# ── F5-TTS (remove torchcodec/triton which cause crashes) ──
pip3 uninstall -y torchcodec 2>/dev/null || true
pip3 install f5-tts
pip3 uninstall -y torchcodec triton 2>/dev/null || true

# ── Stubs for torchcodec/torchao ──
python3 /app/create_stubs.py

# ── LivePortrait ──
git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git /opt/liveportrait || true
pip3 install -r /opt/liveportrait/requirements.txt

# ── MuseTalk v1.5 ──
git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /opt/musetalk || true
pip3 install -U openmim
mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
pip3 install -r /opt/musetalk/requirements.txt 2>/dev/null || true

echo "=== DEPS INSTALLED ==="

# ── Create workspace directories ──
mkdir -p /workspace/models /workspace/tmp \
    /workspace/assets/voice /workspace/assets/driving

# ── Set environment ──
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var before running}"
export HF_HUB_DISABLE_XET=1
export MODEL_CACHE=/workspace/models
export HF_HOME=/workspace/models/huggingface
export HF_HUB_CACHE=/workspace/models/huggingface/hub
export TMPDIR=/workspace/tmp
export PYTHONPATH=/opt/liveportrait:/opt/musetalk

# ── Download MuseTalk models (if not already present) ──
MUSETALK_MODELS=/workspace/models/musetalk
if [ ! -f "$MUSETALK_MODELS/.download_complete" ]; then
    echo "Downloading MuseTalk models..."
    python3 -c "
from huggingface_hub import snapshot_download

# MuseTalk v1 + v1.5 weights
snapshot_download('TMElyralab/MuseTalk', local_dir='$MUSETALK_MODELS')

# SD VAE
snapshot_download('stabilityai/sd-vae-ft-mse', local_dir='$MUSETALK_MODELS/sd-vae-ft-mse')

# Whisper tiny
snapshot_download('openai/whisper-tiny', local_dir='$MUSETALK_MODELS/whisper')

# DWPose
snapshot_download('yzd-v/DWPose', local_dir='$MUSETALK_MODELS/dwpose')

# Face parsing (HF version — we also need manual files below)
snapshot_download('jonathandinu/face-parsing', local_dir='$MUSETALK_MODELS/face-parse-bisenet')
"

    # Face parsing requires BiSeNet weights (not on HuggingFace)
    wget -q -O "$MUSETALK_MODELS/face-parse-bisenet/resnet18-5c106cde.pth" \
        "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    python3 -c "import gdown; gdown.download(id='154JgKpzCPW82qINcVieuPH3fZ2e0P812', output='$MUSETALK_MODELS/face-parse-bisenet/79999_iter.pth')"

    touch "$MUSETALK_MODELS/.download_complete"
    echo "MuseTalk models downloaded"
else
    echo "MuseTalk models already present"
fi

# ── Create symlinks (MuseTalk uses hardcoded relative ./models/ paths) ──
ln -sfn /workspace/models/musetalk /opt/musetalk/models
ln -sfn /workspace/models/musetalk/face-parse-bisenet /opt/musetalk/models/face-parse-bisent
ln -sfn /workspace/models/musetalk/sd-vae-ft-mse /opt/musetalk/models/sd-vae

echo "=== MODELS READY ==="

# ── Download voice reference (if not present) ──
if [ ! -f "/workspace/assets/voice/ashley_ref.wav" ]; then
    echo "Downloading voice reference..."
    wget -q -O /workspace/assets/voice/ashley_ref.mp3 \
        "https://wbyjuruknmtujkizayxy.supabase.co/storage/v1/object/public/media/voice/ashley_ref_birmingham.mp3"
    # Convert mp3 to wav
    ffmpeg -y -i /workspace/assets/voice/ashley_ref.mp3 -ar 24000 -ac 1 /workspace/assets/voice/ashley_ref.wav
    echo "Please call Stella. Ask her to bring these things with her from the store: six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station." > /workspace/assets/voice/ashley_ref.txt
    echo "Voice reference downloaded"
fi

echo "=== SETUP COMPLETE ==="

# ── Start Cloudflare tunnel (for external access from Railway backend) ──
if command -v cloudflared &>/dev/null; then
    nohup cloudflared tunnel --url http://localhost:8000 > /workspace/cloudflared.log 2>&1 &
    sleep 5
    TUNNEL_URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /workspace/cloudflared.log | head -1)
    echo "=== TUNNEL URL: $TUNNEL_URL ==="
    echo "Set GPU_WORKER_URL=$TUNNEL_URL on Railway backend"
fi

# ── Start server ──
cd /app
python3 -u server.py
