# Chatterbox TTS — Voice Generation Model

A deployment-ready Text-to-Speech system built on [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) by ResembleAI. Supports **English** and **multilingual** (23 languages) voice synthesis with voice cloning capabilities.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Running Locally](#running-locally)
  - [FastAPI Server](#fastapi-server)
  - [RunPod Serverless Handler](#runpod-serverless-handler)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
  - [Health Check](#get-health)
  - [Generate Audio (Multipart)](#post-generate)
  - [Generate Audio (JSON)](#post-generate-json)
  - [Clone Voice](#post-clone-voice)
  - [RunPod Serverless](#runpod-serverless)
- [Generation Parameters](#generation-parameters)
- [Supported Languages](#supported-languages)
- [Usage Examples](#usage-examples)
  - [Voice Cloning Workflow](#voice-cloning-workflow)
  - [cURL Examples](#curl-examples)
  - [Python Client Examples](#python-client-examples)

---

## Features

- **Voice cloning** — Clone any voice from a short reference `.wav` file
- **Batch generation** — Generate multiple audio clips in a single request
- **Multilingual support** — 23 languages including English, Spanish, French, Japanese, Chinese, and more
- **Two deployment modes** — FastAPI server for self-hosting or RunPod serverless handler for GPU-on-demand
- **Loudness normalization** — Automatic loudness enhancement with soft limiting on generated audio
- **Pre-baked model weights** — Dockerfile pre-downloads weights at build time for fast cold starts

## Architecture

```
Reference .wav ──► Voice Encoder + S3Gen ──► Voice Conditionals (.pt file)
                                                      │
Text input ──► Tokenizer ──► T3 Transformer ──► Speech Tokens ──► S3Gen Vocoder ──► WAV audio
                                   ▲                                    ▲
                                   │                                    │
                              Voice Conds (.pt)                   Voice Conds (.pt)
```

**Pipeline:**

1. **Voice cloning (one-time):** A reference `.wav` is processed through a voice encoder and S3Gen to produce voice conditionals, saved as a `.pt` file.
2. **Text-to-speech:** Text is tokenized, passed through the T3 transformer to produce speech tokens, then decoded by the S3Gen vocoder into a WAV waveform.
3. **Post-processing:** Output audio is loudness-normalized to -14 dBFS with soft limiting.

## Project Structure

```
voice_generation_model/
├── app.py              # FastAPI server (4 endpoints)
├── handler.py          # RunPod serverless handler
├── tts_engine.py       # Shared TTS engine (model loading, generation, cloning)
├── Dockerfile          # RunPod-optimized Docker image
├── requirements.txt    # Python dependencies
└── chatterbox/         # Chatterbox TTS model package
    ├── __init__.py
    ├── tts.py           # English TTS (ChatterboxTTS)
    ├── mtl_tts.py       # Multilingual TTS (ChatterboxMultilingualTTS)
    ├── vc.py            # Voice conversion (ChatterboxVC)
    └── models/          # Neural network components
        ├── t3/          # T3 transformer (text → speech tokens)
        ├── s3gen/       # S3Gen vocoder (speech tokens → audio)
        ├── s3tokenizer/ # Speech tokenizer
        ├── tokenizers/  # Text tokenizers (English + multilingual)
        ├── voice_encoder/ # Speaker embedding encoder
        └── utils.py
```

## Prerequisites

- **Python** 3.11+
- **PyTorch** 2.4+ with CUDA support (for GPU inference)
- **CUDA** 12.4+ (recommended for production)
- **ffmpeg** and **libsndfile1** (for audio processing)
- **HuggingFace account** with access to `ResembleAI/chatterbox` (set `HF_TOKEN` if the repo is gated)

## Environment Variables

| Variable     | Default        | Description                                                    |
|--------------|----------------|----------------------------------------------------------------|
| `MODEL_TYPE` | `multilingual` | Model variant to load: `multilingual` or `en`                  |
| `PORT`       | `8000`         | Port for the FastAPI server                                    |
| `HF_TOKEN`   | *(none)*       | HuggingFace token (required if the model repo is gated)        |

## Running Locally

### Install Dependencies

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install fastapi uvicorn
pip install -r requirements.txt
```

### FastAPI Server

```bash
# English model
MODEL_TYPE=en python app.py

# Multilingual model (default)
MODEL_TYPE=multilingual python app.py

# Custom port
PORT=9000 python app.py
```

The server starts at `http://localhost:8000` (or your custom port). Interactive API docs are available at `http://localhost:8000/docs`.

### RunPod Serverless Handler

```bash
# Used as the entrypoint for RunPod serverless deployments
python handler.py
```

This loads the model and starts the RunPod serverless worker loop.

## Docker Deployment

### Build the Image

```bash
# Multilingual model (default)
docker build -t chatterbox-tts .

# English-only model
docker build --build-arg MODEL_TYPE=en -t chatterbox-tts .
```

Model weights are downloaded and baked into the image during build, so cold starts are fast.

### Run with Docker

```bash
# RunPod serverless handler (default CMD)
docker run --gpus all chatterbox-tts

# FastAPI server instead
docker run --gpus all -p 8000:8000 chatterbox-tts python app.py
```

### Deploy to RunPod

1. Push your image to a container registry (Docker Hub, GHCR, etc.)
2. Create a new **Serverless Endpoint** on [RunPod](https://www.runpod.io/)
3. Point it to your image
4. The handler auto-starts via the default `CMD ["python", "handler.py"]`

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_type": "multilingual",
  "device": "cuda"
}
```

---

### `POST /generate`

Generate audio from text using a cloned voice. Accepts multipart form data, returns a ZIP archive of WAV files.

**Request (multipart/form-data):**

| Field          | Type           | Required | Default | Description                                    |
|----------------|----------------|----------|---------|------------------------------------------------|
| `voice_file`   | File (.pt)     | Yes      | —       | Voice conditionals file from `/clone-voice`    |
| `texts`        | string (JSON)  | Yes      | —       | JSON array of strings, e.g. `["Hello","World"]`|
| `temperature`  | float          | No       | 0.8     | Sampling temperature                           |
| `exaggeration` | float          | No       | 0.5     | Emotion/expression intensity                   |
| `cfg_weight`   | float          | No       | 0.5     | Classifier-free guidance weight                |
| `language`     | string         | No       | null    | Language code (multilingual model only)        |

**Response:** `application/zip` containing `audio_0.wav`, `audio_1.wav`, etc.

---

### `POST /generate-json`

Same as `/generate` but with a JSON body and base64-encoded I/O.

**Request (application/json):**
```json
{
  "voice_file_base64": "<base64-encoded .pt file>",
  "texts": ["Hello world", "How are you?"],
  "temperature": 0.8,
  "exaggeration": 0.5,
  "cfg_weight": 0.5,
  "language": "en"
}
```

**Response:**
```json
{
  "audio_files": ["<base64-encoded WAV>", "<base64-encoded WAV>"],
  "sample_rate": 24000
}
```

---

### `POST /clone-voice`

Clone a voice from reference audio. Returns a `.pt` conditionals file to use with the generation endpoints.

**Request (multipart/form-data):**

| Field          | Type       | Required | Default | Description                   |
|----------------|------------|----------|---------|-------------------------------|
| `audio_file`   | File (.wav)| Yes      | —       | Reference audio for cloning   |
| `exaggeration` | float      | No       | 0.5     | Emotion exaggeration factor   |

**Response:** `application/octet-stream` — binary `.pt` file (save as `voice.pt`).

---

### RunPod Serverless

When deployed on RunPod, send jobs via the RunPod API.

**Input:**
```json
{
  "input": {
    "texts": ["Hello world", "How are you?"],
    "voice_file_base64": "<base64-encoded .pt file>",
    "temperature": 0.8,
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "language": "en"
  }
}
```

**Output:**
```json
{
  "audio_files": ["<base64-encoded WAV>", ...],
  "sample_rate": 24000
}
```

## Generation Parameters

| Parameter      | Range / Type | Default | Description                                                                 |
|----------------|--------------|---------|-----------------------------------------------------------------------------|
| `temperature`  | float > 0    | 0.8     | Controls randomness. Lower = more deterministic, higher = more varied.      |
| `exaggeration` | float 0-1    | 0.5     | Controls emotional expressiveness. 0 = neutral, 1 = highly expressive.      |
| `cfg_weight`   | float >= 0   | 0.5     | Classifier-free guidance strength. Higher = more faithful to text prompt.   |
| `language`     | string/null  | null    | ISO 639-1 language code. Required for multilingual model, ignored for `en`. |

## Supported Languages

The multilingual model (`MODEL_TYPE=multilingual`) supports the following 23 languages:

| Code | Language   | Code | Language   | Code | Language   |
|------|------------|------|------------|------|------------|
| `ar` | Arabic     | `he` | Hebrew     | `no` | Norwegian  |
| `da` | Danish     | `hi` | Hindi      | `pl` | Polish     |
| `de` | German     | `it` | Italian    | `pt` | Portuguese |
| `el` | Greek      | `ja` | Japanese   | `ru` | Russian    |
| `en` | English    | `ko` | Korean     | `sv` | Swedish    |
| `es` | Spanish    | `ms` | Malay      | `sw` | Swahili    |
| `fi` | Finnish    | `nl` | Dutch      | `tr` | Turkish    |
| `fr` | French     |      |            | `zh` | Chinese    |

## Usage Examples

### Voice Cloning Workflow

The typical workflow is two steps:

1. **Clone a voice** — Upload a reference `.wav` file to get a `.pt` voice conditionals file.
2. **Generate speech** — Use the `.pt` file with any text to generate audio in that voice.

The `.pt` file can be saved and reused indefinitely without re-cloning.

### cURL Examples

**Clone a voice:**
```bash
curl -X POST http://localhost:8000/clone-voice \
  -F "audio_file=@reference.wav" \
  -F "exaggeration=0.5" \
  --output voice.pt
```

**Generate audio (multipart):**
```bash
curl -X POST http://localhost:8000/generate \
  -F "voice_file=@voice.pt" \
  -F 'texts=["Hello world!", "How are you today?"]' \
  -F "temperature=0.8" \
  -F "exaggeration=0.5" \
  -F "language=en" \
  --output generated_audio.zip
```

**Generate audio (JSON):**
```bash
# Encode the .pt file to base64
VOICE_B64=$(base64 -i voice.pt)

curl -X POST http://localhost:8000/generate-json \
  -H "Content-Type: application/json" \
  -d "{
    \"voice_file_base64\": \"$VOICE_B64\",
    \"texts\": [\"Hello world!\", \"How are you today?\"],
    \"language\": \"en\"
  }"
```

### Python Client Examples

**Clone and generate:**
```python
import requests
import base64
import json

BASE_URL = "http://localhost:8000"

# Step 1: Clone a voice
with open("reference.wav", "rb") as f:
    resp = requests.post(f"{BASE_URL}/clone-voice", files={"audio_file": f})
    resp.raise_for_status()

with open("voice.pt", "wb") as f:
    f.write(resp.content)

# Step 2: Generate speech (multipart)
with open("voice.pt", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/generate",
        files={"voice_file": f},
        data={
            "texts": json.dumps(["Hello!", "Welcome to Chatterbox."]),
            "temperature": 0.8,
            "exaggeration": 0.5,
            "language": "en",
        },
    )
    resp.raise_for_status()

with open("output.zip", "wb") as f:
    f.write(resp.content)
```

**Generate via JSON endpoint:**
```python
import requests
import base64

BASE_URL = "http://localhost:8000"

with open("voice.pt", "rb") as f:
    voice_b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    f"{BASE_URL}/generate-json",
    json={
        "voice_file_base64": voice_b64,
        "texts": ["Hello!", "Welcome to Chatterbox."],
        "language": "en",
    },
)
resp.raise_for_status()
data = resp.json()

# Decode and save each audio file
for i, audio_b64 in enumerate(data["audio_files"]):
    with open(f"audio_{i}.wav", "wb") as f:
        f.write(base64.b64decode(audio_b64))

print(f"Sample rate: {data['sample_rate']} Hz")
```

**RunPod client:**
```python
import runpod
import base64

runpod.api_key = "YOUR_RUNPOD_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

with open("voice.pt", "rb") as f:
    voice_b64 = base64.b64encode(f.read()).decode()

run = endpoint.run_sync({
    "texts": ["Hello from RunPod!", "This is a test."],
    "voice_file_base64": voice_b64,
    "temperature": 0.8,
    "language": "en",
})

for i, audio_b64 in enumerate(run["audio_files"]):
    with open(f"audio_{i}.wav", "wb") as f:
        f.write(base64.b64decode(audio_b64))
```

---

## Notes

- The server uses an **async lock** to serialize GPU inference requests, preventing OOM errors from concurrent generation.
- Audio output is **24 kHz, mono, WAV** format.
- The loudness enhancer normalizes to **-14 dBFS** with a soft limiter at **-1 dBFS**.
- Model weights are downloaded from HuggingFace Hub on first run (or at Docker build time).
- The `.pt` voice conditionals files are portable and can be shared across deployments using the same model type.
