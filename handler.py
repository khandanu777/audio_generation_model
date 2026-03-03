"""
RunPod Serverless Handler for Chatterbox TTS.

Supports two actions via the "action" field:

1. Generate audio (default):
{
    "input": {
        "action": "generate",
        "texts": ["Hello world", "How are you?"],
        "voice_file_base64": "<base64-encoded .pt file>",
        "temperature": 0.8,
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "language": "en"
    }
}
→ {"audio_files": ["<base64 wav>", ...], "sample_rate": 24000}

2. Clone voice:
{
    "input": {
        "action": "clone_voice",
        "audio_file_base64": "<base64-encoded .wav file>",
        "exaggeration": 0.5
    }
}
→ {"voice_file_base64": "<base64-encoded .pt file>"}
"""

import base64

import runpod

from tts_engine import (
    SAMPLE_RATE,
    clone_voice_from_audio,
    generate_batch,
    load_model,
    wav_tensor_to_bytes,
)


def _handle_generate(job_input: dict) -> dict:
    texts = job_input.get("texts")
    if not texts or not isinstance(texts, list):
        return {"error": "texts must be a non-empty list of strings"}

    voice_b64 = job_input.get("voice_file_base64")
    if not voice_b64:
        return {"error": "voice_file_base64 is required"}

    try:
        pt_bytes = base64.b64decode(voice_b64)
    except Exception:
        return {"error": "Invalid base64 in voice_file_base64"}

    audio_results = generate_batch(
        texts=texts,
        voice_pt_bytes=pt_bytes,
        language=job_input.get("language"),
        temperature=job_input.get("temperature", 0.8),
        exaggeration=job_input.get("exaggeration", 0.5),
        cfg_weight=job_input.get("cfg_weight", 0.5),
    )

    return {
        "audio_files": [
            base64.b64encode(wav_tensor_to_bytes(wav)).decode()
            for wav in audio_results
        ],
        "sample_rate": SAMPLE_RATE,
    }


def _handle_clone_voice(job_input: dict) -> dict:
    audio_b64 = job_input.get("audio_file_base64")
    if not audio_b64:
        return {"error": "audio_file_base64 is required"}

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        return {"error": "Invalid base64 in audio_file_base64"}

    exaggeration = job_input.get("exaggeration", 0.5)
    pt_bytes = clone_voice_from_audio(audio_bytes, exaggeration=exaggeration)

    return {
        "voice_file_base64": base64.b64encode(pt_bytes).decode(),
    }


def handler(job):
    job_input = job["input"]
    action = job_input.get("action", "generate")

    try:
        if action == "generate":
            return _handle_generate(job_input)
        elif action == "clone_voice":
            return _handle_clone_voice(job_input)
        else:
            return {"error": f"Unknown action: {action}. Use 'generate' or 'clone_voice'."}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
