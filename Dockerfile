FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY chatterbox /app/chatterbox

# Pre-download model weights so cold starts are fast.
# Build with --build-arg MODEL_TYPE=multilingual for the multilingual model.
ARG MODEL_TYPE=multilingual
ENV MODEL_TYPE=${MODEL_TYPE}

ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

RUN if [ "$MODEL_TYPE" = "multilingual" ]; then \
        python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cpu')"; \
    else \
        python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')"; \
    fi

# Clear the token from the image after download
ENV HF_TOKEN=""

COPY *.py .

CMD ["python", "handler.py"]
