FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /app

# System deps + FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    build-essential \
    git wget ffmpeg && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# PyTorch 2.6 with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Core dependencies (torchao pinned to 0.7.0 for torch 2.6 compat)
RUN pip install --no-cache-dir \
    runpod==1.7.0 \
    diffusers==0.34.0 \
    transformers==4.53.2 \
    tokenizers==0.21.4 \
    accelerate==1.8.1 \
    huggingface_hub>=0.26.0 \
    numpy==1.26.4 \
    requests>=2.32.0 \
    soundfile==0.12.1 \
    omegaconf==2.3.0 \
    ftfy==6.3.1 \
    imageio-ffmpeg==0.5.1 \
    imageio \
    easydict \
    pyloudnorm \
    librosa \
    kornia \
    wget==3.2 \
    sentencepiece \
    av \
    xfuser

# Verify diffusers deep imports work (no torchao needed)
RUN python -c "from diffusers.models.modeling_utils import ModelMixin; print('diffusers ModelMixin OK')"

# F5-TTS for voice cloning
# Remove torchcodec BEFORE and AFTER install to prevent ABI mismatch crashes
RUN pip uninstall -y torchcodec 2>/dev/null || true && \
    pip install --no-cache-dir f5-tts && \
    pip uninstall -y torchcodec 2>/dev/null || true

# Remove triton — its JIT compilation fails without runtime libcuda.so
# SkyReels V3 doesn't need triton for inference
RUN pip uninstall -y triton 2>/dev/null || true

# Create stub modules for torchcodec and torchao (avoid import crashes)
# torchao is imported unconditionally by SkyReels V3 but only used for low_vram quantization
RUN python -c "\
import site, os;\
sp=site.getsitepackages()[0];\
for mod in ['torchcodec','torchao']:\
    p=os.path.join(sp,mod);\
    os.makedirs(p,exist_ok=True);\
    open(os.path.join(p,'__init__.py'),'w').write('# stub\n');\
os.makedirs(os.path.join(sp,'torchao','quantization'),exist_ok=True);\
open(os.path.join(sp,'torchao','quantization','__init__.py'),'w').write('def float8_weight_only(*a,**k): pass\ndef quantize_(*a,**k): pass\n');\
print('stubs created')"

# Clone SkyReels V3
RUN git clone --depth 1 https://github.com/SkyworkAI/SkyReels-V3.git /opt/skyreels-v3
ENV PYTHONPATH="/opt/skyreels-v3:${PYTHONPATH}"

# Set model cache to network volume (will be mounted at runtime)
ENV MODEL_CACHE="/runpod-volume/models"
ENV HF_HOME="/runpod-volume/models/huggingface"
ENV HF_HUB_CACHE="/runpod-volume/models/huggingface/hub"
ENV TMPDIR="/runpod-volume/tmp"
# HF_TOKEN should be set as an env var in the RunPod endpoint config

# Copy worker files
COPY handler.py .
COPY skyreels_inference.py .
COPY f5_tts_wrapper.py .
COPY ashley_reference.png .

CMD ["python", "-u", "handler.py"]
