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

# Core dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    pydantic>=2.0.0 \
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
    wget==3.2 \
    sentencepiece \
    av

# F5-TTS for voice cloning
# Remove torchcodec BEFORE and AFTER install to prevent ABI mismatch crashes
RUN pip uninstall -y torchcodec 2>/dev/null || true && \
    pip install --no-cache-dir f5-tts && \
    pip uninstall -y torchcodec 2>/dev/null || true

# Remove triton — its JIT compilation fails without runtime libcuda.so
RUN pip uninstall -y triton 2>/dev/null || true

# Create stub modules for torchcodec and torchao (avoid import crashes)
COPY create_stubs.py /tmp/create_stubs.py
RUN python /tmp/create_stubs.py && rm /tmp/create_stubs.py

# LivePortrait
RUN git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git /opt/liveportrait
RUN pip install --no-cache-dir -r /opt/liveportrait/requirements.txt

# MuseTalk v1.5
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /opt/musetalk
RUN pip install --no-cache-dir -U openmim
RUN mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
RUN pip install --no-cache-dir -r /opt/musetalk/requirements.txt 2>/dev/null || true

# Vast.ai uses /workspace as persistent storage
ENV MODEL_CACHE="/workspace/models"
ENV HF_HOME="/workspace/models/huggingface"
ENV HF_HUB_CACHE="/workspace/models/huggingface/hub"
ENV TMPDIR="/workspace/tmp"
ENV PYTHONPATH="/opt/liveportrait:/opt/musetalk:${PYTHONPATH}"

# Copy worker files
COPY server.py .
COPY f5_tts_wrapper.py .
COPY liveportrait_wrapper.py .
COPY musetalk_wrapper.py .
COPY ashley_reference.png .

EXPOSE 8000

CMD ["python", "-u", "server.py"]
