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
- [Upgrading to V3 Checkpoints](#upgrading-to-v3-checkpoints)
- [Dependency Pinning](#dependency-pinning)
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
- **Runtime weight fetch** — weights are pulled from HuggingFace on first start and cached on the pod volume

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
- **PyTorch** exactly 2.4.0 with CUDA support (supplied by the base image; newer torch is
  untested here and older `transformers` pins depend on it — see [Dependency Pinning](#dependency-pinning))
- **CUDA** 12.4+ (recommended for production)
- **ffmpeg** and **libsndfile1** (for audio processing)
- **HuggingFace account** with access to `ResembleAI/chatterbox` (set `HF_TOKEN` if the repo is gated)

## Environment Variables

| Variable      | Default        | Description                                                                     |
|---------------|----------------|---------------------------------------------------------------------------------|
| `MODEL_TYPE`  | `multilingual` | Model variant to load: `multilingual` or `en`                                   |
| `T3_MODEL`    | `v2`           | Multilingual T3 checkpoint: `v2` or `v3`. Ignored when `MODEL_TYPE=en`          |
| `S3GEN_MODEL` | `v1`           | S3Gen vocoder checkpoint: `v1` or `v3`. Ignored when `MODEL_TYPE=en`            |
| `PORT`        | `8000`         | Port for the FastAPI server                                                     |
| `HF_TOKEN`    | *(none)*       | HuggingFace token (required if the model repo is gated)                         |

The defaults reproduce the exact behaviour this service had before V3 support was
added, so existing callers are unaffected. See [Upgrading to V3 Checkpoints](#upgrading-to-v3-checkpoints).

## Running Locally

### Install Dependencies

```bash
# torch must be 2.4.0 to match the base image — see the note in requirements.txt
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install fastapi uvicorn
pip install -r requirements.txt
```

> **Dependencies are pinned.** `requirements.txt` pins every package to an exact
> version. Do not relax the pins casually — in particular **`transformers` must stay
> `<=5.13.1`** on this base image. See [Dependency Pinning](#dependency-pinning).

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

Model weights are **not** baked into the image — they are downloaded from HuggingFace on the
first start and cached, so the first cold start after a fresh deploy is slow.

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

## Upgrading to V3 Checkpoints

ResembleAI published two newer checkpoints to `ResembleAI/chatterbox`. Both are
**opt-in** here and **off by default** — this service keeps running T3 v2 + S3Gen v1
until you explicitly set an environment variable.

| Env var | Value | Weights file | What it changes |
|---------|-------|--------------|-----------------|
| `T3_MODEL` | `v3` | `t3_mtl23ls_v3.safetensors` | Multilingual V3 text→token model: better speaker similarity, fewer hallucinations |
| `S3GEN_MODEL` | `v3` | `s3gen_v3.pt` | Retrained HiFTNet vocoder (`mel2wav`) — waveform quality only |

The two are independent; you can enable either or both.

### Your cached `.pt` voice files stay valid

`s3gen_v3` differs from `s3gen` **only** in the `mel2wav` vocoder submodule. The speech
tokenizer, the speaker encoder (x-vector), and the flow matching decoder are byte-identical.
Voice conditionals are produced by `embed_ref()`, which touches only those unchanged parts,
so **every `.pt` file already held by downstream services keeps working — no re-cloning
required** when switching either checkpoint.

### Step-by-step: ship it to RunPod

**1. Commit and push the code.** The weights themselves are *not* in the repo (`*.pt` and
`*.safetensors` are gitignored) and are *not* baked into the image — they are pulled from
HuggingFace on the pod's first cold start.

```bash
git add -A
git commit -m "Add opt-in V3 checkpoint support"
git push origin main
```

**2. Rebuild and push the image.** Required once, because the checkpoint-selection code is
new. After this, switching versions never needs another rebuild.

```bash
docker build -t <your-registry>/chatterbox-tts:v3-support .
docker push <your-registry>/chatterbox-tts:v3-support
```

**3. Point the endpoint at the new image.** In the RunPod console: *Serverless → your
endpoint → Edit → Container Image*, set the new tag and save.

**4. Set the environment variables.** Same edit screen, under *Environment Variables*:

| Key | Value |
|-----|-------|
| `T3_MODEL` | `v3` |
| `S3GEN_MODEL` | `v3` |

Leave either one out to keep that component on the old checkpoint. Save — RunPod rolls the
workers automatically.

**5. Expect a slow first request.** The new checkpoints are downloaded on the first cold
start (~2.1 GB for T3 v3, ~1.0 GB for S3Gen v3), so the first job after the switch can take
several minutes. Subsequent cold starts reuse the cached volume.

**6. Verify which weights the worker actually loaded** with the `info` action:

```bash
curl -X POST https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "info"}}'
```

```json
{"output": {"model_type": "multilingual", "device": "cuda",
            "t3_model": "v3", "s3gen_model": "v3", "sample_rate": 24000}}
```

The worker also logs the selection at startup:
`[tts_engine] Loaded multilingual model on cuda (t3=v3, s3gen=v3)`

**7. Generate as usual** — the request format is unchanged:

```bash
python test_endpoint.py --voice-pt srk.pt --texts "Testing the v3 checkpoints."
```

### Rolling back

Set `T3_MODEL=v2` and `S3GEN_MODEL=v1` (or delete both variables) and save. No rebuild, no
image change — the previous behaviour is restored exactly.

### Recommended rollout

Because other services depend on this endpoint, upgrade on a **separate endpoint** first:
create a second RunPod serverless endpoint from the same image with the V3 variables set,
compare output against the current one using the same `.pt` voice, then move traffic. Both
endpoints can run side by side from one image — only the env vars differ.

### Behaviour bundled with `T3_MODEL=v3`

Three changes ship with V3 upstream and are applied **only** when `T3_MODEL=v3`, so the v2
path is untouched:

- The alignment stream analyzer is disabled — its forced-EOS heuristics were tuned on v2 and
  can truncate v3 output.
- Default `repetition_penalty` drops from `2.0` to `1.2`. An explicit value in the request
  still wins.
- The final speech token is trimmed (~40 ms of noise emitted just before EOS).

### Local testing before you push

```bash
T3_MODEL=v3 S3GEN_MODEL=v3 MODEL_TYPE=multilingual python app.py
curl localhost:8000/health   # confirms t3_model / s3gen_model
```

## Dependency Pinning

`requirements.txt` pins every package to an exact version. This is deliberate: the file
used to be fully unpinned, so each rebuild silently picked up whatever was newest on PyPI.
A rebuild in August 2026 pulled `transformers` 5.14 and every worker crashed on startup
with:

```
ImportError: cannot import name 'DTensor' from 'torch.distributed.tensor'
```

**Cause.** The base image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1` provides **torch 2.4.0**,
where `torch/distributed/tensor/__init__.py` is an empty placeholder — `DTensor` only
entered that public namespace in torch 2.5. transformers 5.14.0 added a module-level
`from torch.distributed.tensor import DTensor` that is reached unconditionally through
`activations.py` when you import any model class, so `import transformers` itself fails.
transformers still declares `torch>=2.4` in its metadata, so pip resolves it happily and
the failure only appears at runtime.

**The rule:** on this base image, **`transformers` must stay `<= 5.13.1`.** Verified by
running the real import chain against torch 2.4.0:

| transformers | Result on torch 2.4.0 |
|--------------|------------------------|
| 5.2.0 *(pinned)* | imports fine — also what upstream chatterbox pins |
| 5.13.1 | imports fine — highest usable version |
| 5.14.0 | **fails** — first version with the bad import |
| 5.14.1 | **fails** |

`torch` is deliberately absent from `requirements.txt`: `torchaudio==2.4.0` requires
`torch==2.4.0` exactly, which locks it transitively while letting pip keep the CUDA 12.4
build already in the image instead of replacing it with a CPU wheel from PyPI.

### Changing a pin

1. Check it against torch 2.4.0 before committing:
   ```bash
   docker run --rm python:3.11-slim bash -c \
     "pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu && \
      pip install -q transformers==<new-version> && \
      python -c 'from transformers import LlamaModel, GPT2Model; print(\"ok\")'"
   ```
2. If you need a newer transformers than 5.13.1, you must move to a torch 2.5+ base image,
   which revalidates the whole CUDA stack. Upstream chatterbox now targets torch 2.6.0 +
   transformers 5.2.0.

**Residual gap:** only direct dependencies are pinned; deeper transitive packages still
float, though `numpy==1.26.4` constrains the risky `librosa → numba → numpy` chain. For
a fully reproducible build, generate a complete lock with `pip freeze` inside a successful
image and commit that instead.

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_type": "multilingual",
  "device": "cuda",
  "t3_model": "v2",
  "s3gen_model": "v1"
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

The handler also accepts `"action": "clone_voice"` and `"action": "info"`. The latter reports
which checkpoints the worker loaded — see
[Upgrading to V3 Checkpoints](#upgrading-to-v3-checkpoints).

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



### NOTE
This image can run on ada and amphere gpu arch and cuda>= 12.4
So remember to keep these in gpu settings
