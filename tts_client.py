"""
Chatterbox TTS client for RunPod serverless endpoint.

Usage:
    from tts_client import ChatterboxClient

    client = ChatterboxClient(api_key="...", endpoint_id="...")

    # Clone a voice from reference audio
    voice_pt = client.clone_voice("reference.wav")
    # or save it for reuse:
    client.clone_voice("reference.wav", save_to="my_voice.pt")

    # Generate audio
    wav_files = client.generate(
        texts=["Hello world!", "How are you?"],
        voice_pt_path="my_voice.pt",
        language="en",
    )
    # wav_files = ["output/audio_0.wav", "output/audio_1.wav"]
"""

import base64
import os
import time
from pathlib import Path
from typing import Optional, Union

import requests


class ChatterboxClient:
    def __init__(
        self,
        api_key: str,
        endpoint_id: str,
        timeout: int = 300,
        poll_interval: float = 2.0,
    ):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _submit(self, payload: dict) -> str:
        resp = requests.post(f"{self.base_url}/run", json=payload, headers=self._headers)
        resp.raise_for_status()
        data = resp.json()
        if "id" not in data:
            raise RuntimeError(f"RunPod submit failed: {data}")
        return data["id"]

    def _poll(self, job_id: str) -> dict:
        url = f"{self.base_url}/status/{job_id}"
        start = time.time()
        interval = self.poll_interval

        while time.time() - start < self.timeout:
            resp = requests.get(url, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "UNKNOWN")

            if status == "COMPLETED":
                output = data.get("output", {})
                if "error" in output:
                    raise RuntimeError(f"Job returned error: {output['error']}")
                return output
            elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                error = data.get("error", data.get("output", {}).get("error", "Unknown"))
                raise RuntimeError(f"Job {status}: {error}")

            time.sleep(interval)
            interval = min(interval * 1.5, 10)

        raise TimeoutError(f"Job {job_id} timed out after {self.timeout}s")

    def _run(self, payload: dict) -> dict:
        job_id = self._submit(payload)
        return self._poll(job_id)

    def clone_voice(
        self,
        audio_path: str,
        save_to: Optional[str] = None,
        exaggeration: float = 0.5,
    ) -> bytes:
        """
        Clone a voice from a reference .wav file.

        Args:
            audio_path: Path to reference audio (.wav).
            save_to: If provided, saves the .pt file to this path.
            exaggeration: Emotion exaggeration factor (0-1).

        Returns:
            Raw .pt file bytes.
        """
        audio_bytes = Path(audio_path).read_bytes()
        audio_b64 = base64.b64encode(audio_bytes).decode()

        output = self._run({
            "input": {
                "action": "clone_voice",
                "audio_file_base64": audio_b64,
                "exaggeration": exaggeration,
            }
        })

        pt_bytes = base64.b64decode(output["voice_file_base64"])

        if save_to:
            Path(save_to).write_bytes(pt_bytes)

        return pt_bytes

    def _build_generate_payload(
        self,
        texts: list[str],
        voice_pt_path: Optional[str],
        voice_pt_bytes: Optional[bytes],
        language: str,
        temperature: Union[float, list[float]],
        exaggeration: Union[float, list[float]],
        cfg_weight: Union[float, list[float]],
    ) -> dict:
        if voice_pt_path:
            voice_b64 = base64.b64encode(Path(voice_pt_path).read_bytes()).decode()
        elif voice_pt_bytes:
            voice_b64 = base64.b64encode(voice_pt_bytes).decode()
        else:
            raise ValueError("Provide either voice_pt_path or voice_pt_bytes")

        return {
            "input": {
                "action": "generate",
                "texts": texts,
                "voice_file_base64": voice_b64,
                "language": language,
                "temperature": temperature,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            }
        }

    def generate(
        self,
        texts: list[str],
        voice_pt_path: Optional[str] = None,
        voice_pt_bytes: Optional[bytes] = None,
        language: str = "en",
        temperature: Union[float, list[float]] = 0.8,
        exaggeration: Union[float, list[float]] = 0.5,
        cfg_weight: Union[float, list[float]] = 0.5,
        output_dir: str = "output",
    ) -> list[str]:
        """
        Generate audio for a list of texts using a cloned voice.

        Args:
            texts: List of text strings to synthesize.
            voice_pt_path: Path to a .pt voice file.
            voice_pt_bytes: Raw .pt bytes (alternative to voice_pt_path).
            language: Language code (e.g. "en", "es", "fr", "ja").
            temperature: Single float for all texts, or list[float] one per text.
            exaggeration: Single float for all texts, or list[float] one per text.
            cfg_weight: Single float for all texts, or list[float] one per text.
            output_dir: Directory to save generated .wav files.

        Returns:
            List of saved .wav file paths.
        """
        payload = self._build_generate_payload(
            texts, voice_pt_path, voice_pt_bytes,
            language, temperature, exaggeration, cfg_weight,
        )
        output = self._run(payload)

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        for i, audio_b64 in enumerate(output["audio_files"]):
            fpath = os.path.join(output_dir, f"audio_{i}.wav")
            Path(fpath).write_bytes(base64.b64decode(audio_b64))
            saved_paths.append(fpath)

        return saved_paths

    def generate_to_bytes(
        self,
        texts: list[str],
        voice_pt_path: Optional[str] = None,
        voice_pt_bytes: Optional[bytes] = None,
        language: str = "en",
        temperature: Union[float, list[float]] = 0.8,
        exaggeration: Union[float, list[float]] = 0.5,
        cfg_weight: Union[float, list[float]] = 0.5,
    ) -> list[bytes]:
        """
        Same as generate() but returns raw WAV bytes instead of saving to disk.
        Useful for piping into other processing steps.

        Args:
            texts: List of text strings to synthesize.
            voice_pt_path: Path to a .pt voice file.
            voice_pt_bytes: Raw .pt bytes (alternative to voice_pt_path).
            language: Language code (e.g. "en", "es", "fr", "ja").
            temperature: Single float for all texts, or list[float] one per text.
            exaggeration: Single float for all texts, or list[float] one per text.
            cfg_weight: Single float for all texts, or list[float] one per text.

        Returns:
            List of WAV file bytes.
        """
        payload = self._build_generate_payload(
            texts, voice_pt_path, voice_pt_bytes,
            language, temperature, exaggeration, cfg_weight,
        )
        output = self._run(payload)

        return [base64.b64decode(a) for a in output["audio_files"]]
