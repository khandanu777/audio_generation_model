"""
Core Chatterbox TTS engine.

Handles model loading, voice conditionals, and audio generation.
Shared by both the FastAPI server (app.py) and RunPod handler (handler.py).
"""

import io
import os
import tempfile
from typing import Optional

import torch
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000
MODEL_TYPE = os.getenv("MODEL_TYPE", "multilingual")  # "multilingual" or "en"

_model = None


def load_model():
    """Load the Chatterbox TTS model. Call once at startup."""
    global _model
    if MODEL_TYPE == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        _model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    else:
        from chatterbox.tts import ChatterboxTTS
        _model = ChatterboxTTS.from_pretrained(device=DEVICE)
    print(f"[tts_engine] Loaded {MODEL_TYPE} model on {DEVICE}")
    return _model


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


def set_voice_from_path(pt_path: str):
    """Load cached voice conditionals from a .pt file path."""
    if MODEL_TYPE == "multilingual":
        from chatterbox.mtl_tts import Conditionals
    else:
        from chatterbox.tts import Conditionals
    model = get_model()
    model.conds = Conditionals.load(pt_path, map_location=DEVICE).to(DEVICE)


def set_voice_from_bytes(pt_bytes: bytes):
    """Load cached voice conditionals from raw .pt bytes."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(pt_bytes)
        temp_path = f.name
    try:
        set_voice_from_path(temp_path)
    finally:
        os.unlink(temp_path)


def enhance_loudness(audio_tensor, target_db=-14.0, limiter_threshold=-1.0):
    audio = audio_tensor.float()
    original_shape = audio.shape
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms < 1e-8:
        return audio_tensor
    current_db = 20 * torch.log10(rms)
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)
    audio_enhanced = audio * gain_linear
    limiter_threshold_linear = 10 ** (limiter_threshold / 20)
    mask = torch.abs(audio_enhanced) > limiter_threshold_linear
    if mask.any():
        exceeded = audio_enhanced[mask]
        sign = torch.sign(exceeded)
        magnitude = torch.abs(exceeded)
        compressed = limiter_threshold_linear + (1 - limiter_threshold_linear) * torch.tanh(
            (magnitude - limiter_threshold_linear) / (1 - limiter_threshold_linear)
        )
        audio_enhanced[mask] = sign * compressed
    peak = torch.max(torch.abs(audio_enhanced))
    if peak > 1.0:
        audio_enhanced = audio_enhanced / peak * 0.99
    if len(original_shape) == 1:
        audio_enhanced = audio_enhanced.squeeze(0)
    return audio_enhanced


def generate_single(
    text: str,
    language: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> torch.Tensor:
    """Generate a single audio clip. Returns tensor of shape [1, samples]."""
    model = get_model()
    if MODEL_TYPE == "multilingual" and language:
        wav = model.generate(
            text=text,
            language_id=language,
            audio_prompt_path=None,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )
    else:
        wav = model.generate(
            text=text,
            audio_prompt_path=None,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )
    return enhance_loudness(wav)


def generate_batch(
    texts: list[str],
    voice_pt_bytes: bytes,
    language: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> list[torch.Tensor]:
    """Generate audio for a list of texts using the given voice .pt file."""
    set_voice_from_bytes(voice_pt_bytes)
    return [
        generate_single(text, language, temperature, exaggeration, cfg_weight)
        for text in texts
    ]


def wav_tensor_to_bytes(wav: torch.Tensor) -> bytes:
    """Convert a wav tensor to WAV file bytes."""
    buf = io.BytesIO()
    torchaudio.save(buf, wav, SAMPLE_RATE, format="wav")
    buf.seek(0)
    return buf.read()


def clone_voice_from_audio(audio_bytes: bytes, exaggeration: float = 0.5) -> bytes:
    """Clone a voice from reference audio bytes, return .pt file bytes."""
    model = get_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
        af.write(audio_bytes)
        audio_path = af.name
    pt_path = audio_path.replace(".wav", ".pt")
    try:
        model.prepare_conditionals(audio_path, exaggeration=exaggeration)
        model.conds.save(pt_path)
        with open(pt_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(audio_path)
        if os.path.exists(pt_path):
            os.unlink(pt_path)
