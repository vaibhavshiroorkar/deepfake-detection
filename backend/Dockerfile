FROM python:3.11-slim

# Hugging Face Spaces mounts the repo at /home/user/app and runs as UID 1000.
# We mirror that convention so the image works both locally and on Spaces.

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    VERITAS_FORCE_CPU=1 \
    VERITAS_WHISPER_MODEL=openai/whisper-tiny \
    PORT=7860

WORKDIR /home/user/app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

# Pre-download all model weights into the image layer so HF Spaces
# cold-starts do not pay a 60 to 80 second download cost each time.
RUN python scripts/prefetch_models.py || true

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
