"""
RunPod Serverless Handler for Chatterbox TTS.

Use this as the entrypoint for RunPod serverless GPU deployment.
Set CMD in Dockerfile to: ["python", "handler.py"]

Input schema:
{
    "input": {
        "texts": ["Hello world", "How are you?"],
        "voice_file_base64": "<base64-encoded .pt file>",
        "temperature": 0.8,
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "language": null
    }
}

Output schema:
{
    "audio_files": ["<base64 wav>", ...],
    "sample_rate": 24000
}
"""

import base64

import runpod

from tts_engine import (
    SAMPLE_RATE,
    generate_batch,
    load_model,
    wav_tensor_to_bytes,
)


def handler(job):
    job_input = job["input"]

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

    temperature = job_input.get("temperature", 0.8)
    exaggeration = job_input.get("exaggeration", 0.5)
    cfg_weight = job_input.get("cfg_weight", 0.5)
    language = job_input.get("language")

    try:
        audio_results = generate_batch(
            texts=texts,
            voice_pt_bytes=pt_bytes,
            language=language,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
    except Exception as e:
        return {"error": str(e)}

    return {
        "audio_files": [
            base64.b64encode(wav_tensor_to_bytes(wav)).decode()
            for wav in audio_results
        ],
        "sample_rate": SAMPLE_RATE,
    }


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
