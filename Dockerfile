FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# System deps + FFmpeg 6 (needed by torchcodec)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    git wget software-properties-common && \
    add-apt-repository ppa:ubuntuhandbook1/ffmpeg6 -y && \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# PyTorch 2.6 with CUDA 12.4 (latest available on PyPI for cu124)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# SkyReels V3 dependencies (from their requirements.txt)
RUN pip install --no-cache-dir \
    runpod==1.7.0 \
    diffusers==0.34.0 \
    "transformers>=4.53.0,<5.0.0" \
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
    torchao \
    xfuser

# F5-TTS for voice cloning (then remove torchcodec — optional dep with PyTorch ABI issues)
RUN pip install --no-cache-dir f5-tts && \
    pip uninstall -y torchcodec 2>/dev/null || true

# Clone SkyReels V3
RUN git clone --depth 1 https://github.com/SkyworkAI/SkyReels-V3.git /opt/skyreels-v3
ENV PYTHONPATH="/opt/skyreels-v3:${PYTHONPATH}"

# Copy worker files
COPY handler.py .
COPY skyreels_inference.py .
COPY f5_tts_wrapper.py .
COPY ashley_reference.png .

CMD ["python", "-u", "handler.py"]
