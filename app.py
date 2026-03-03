"""
FastAPI server for Chatterbox TTS.

Endpoints:
  POST /generate       — multipart form: .pt voice file + JSON texts → ZIP of WAVs
  POST /generate-json  — JSON body: base64 .pt + texts → JSON with base64 WAVs
  POST /clone-voice    — multipart form: reference .wav → .pt voice file
  GET  /health         — health check
"""

import asyncio
import base64
import io
import json
import zipfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from contextlib import asynccontextmanager

from tts_engine import (
    DEVICE,
    MODEL_TYPE,
    SAMPLE_RATE,
    clone_voice_from_audio,
    generate_batch,
    load_model,
    wav_tensor_to_bytes,
)

_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Chatterbox TTS API",
    description="Text-to-speech with voice cloning via Chatterbox",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_type": MODEL_TYPE, "device": DEVICE}


@app.post("/generate")
async def generate(
    voice_file: UploadFile = File(..., description=".pt voice conditionals file"),
    texts: str = Form(..., description='JSON array of strings, e.g. ["Hello", "World"]'),
    temperature: float = Form(0.8),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    language: Optional[str] = Form(None, description="Language code (multilingual model only)"),
):
    try:
        text_list = json.loads(texts)
        if not isinstance(text_list, list) or not all(isinstance(t, str) for t in text_list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(400, "texts must be a JSON array of strings")

    pt_bytes = await voice_file.read()

    async with _lock:
        audio_results = generate_batch(
            texts=text_list,
            voice_pt_bytes=pt_bytes,
            language=language,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for idx, wav in enumerate(audio_results):
            zf.writestr(f"audio_{idx}.wav", wav_tensor_to_bytes(wav))
    zip_buf.seek(0)

    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=generated_audio.zip"},
    )


class GenerateJSONRequest(BaseModel):
    voice_file_base64: str
    texts: list[str]
    temperature: float = 0.8
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    language: Optional[str] = None


@app.post("/generate-json")
async def generate_json(req: GenerateJSONRequest):
    """JSON-based endpoint — accepts base64 .pt file, returns base64 WAVs."""
    try:
        pt_bytes = base64.b64decode(req.voice_file_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 in voice_file_base64")

    async with _lock:
        audio_results = generate_batch(
            texts=req.texts,
            voice_pt_bytes=pt_bytes,
            language=req.language,
            temperature=req.temperature,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
        )

    return {
        "audio_files": [
            base64.b64encode(wav_tensor_to_bytes(wav)).decode()
            for wav in audio_results
        ],
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/clone-voice")
async def clone_voice(
    audio_file: UploadFile = File(..., description="Reference audio (.wav)"),
    exaggeration: float = Form(0.5),
):
    """Clone a voice from reference audio. Returns a .pt conditionals file."""
    audio_bytes = await audio_file.read()

    async with _lock:
        pt_bytes = clone_voice_from_audio(audio_bytes, exaggeration=exaggeration)

    return Response(
        content=pt_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=voice.pt"},
    )


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
