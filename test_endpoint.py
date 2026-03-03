"""
Test script for the Chatterbox TTS RunPod serverless endpoint.

Usage:
  1. Fill in your .env file with RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID.
  2. Place a voice.pt file in this directory (or set VOICE_PT_PATH).
  3. Run: python test_endpoint.py
"""

import argparse
import base64
import json
import os
import sys
import time

from dotenv import load_dotenv
import requests

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
BASE_URL = "https://api.runpod.ai/v2"

OUTPUT_DIR = "test_output"


def submit_job(endpoint_id: str, headers: dict, payload: dict) -> str:
    url = f"{BASE_URL}/{endpoint_id}/run"
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if "id" not in data:
        print(f"Unexpected response: {json.dumps(data, indent=2)}")
        sys.exit(1)
    return data["id"]


def poll_result(endpoint_id: str, job_id: str, headers: dict, timeout: int = 300):
    url = f"{BASE_URL}/{endpoint_id}/status/{job_id}"
    start = time.time()
    interval = 2

    while time.time() - start < timeout:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "UNKNOWN")

        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>3d}s] Status: {status}")

        if status == "COMPLETED":
            return data.get("output", {})
        elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(f"\nJob failed: {json.dumps(data, indent=2)}")
            sys.exit(1)

        time.sleep(interval)
        interval = min(interval * 1.5, 10)

    print(f"\nTimed out after {timeout}s")
    sys.exit(1)


def save_audio_files(output: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = output.get("audio_files", [])
    sample_rate = output.get("sample_rate", 24000)

    if not audio_files:
        print("No audio files in response!")
        if "error" in output:
            print(f"Error: {output['error']}")
        return

    print(f"\nReceived {len(audio_files)} audio file(s) at {sample_rate} Hz")
    saved = []
    for i, audio_b64 in enumerate(audio_files):
        fpath = os.path.join(output_dir, f"audio_{i}.wav")
        with open(fpath, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        size_kb = os.path.getsize(fpath) / 1024
        saved.append(fpath)
        print(f"  Saved: {fpath} ({size_kb:.1f} KB)")

    return saved


def run_test(
    api_key: str,
    endpoint_id: str,
    voice_pt_path: str,
    texts: list[str],
    language: str,
    temperature: float,
    exaggeration: float,
    cfg_weight: float,
    timeout: int,
):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Encode voice file
    print(f"Loading voice file: {voice_pt_path}")
    with open(voice_pt_path, "rb") as f:
        voice_b64 = base64.b64encode(f.read()).decode()
    print(f"  Voice file size: {len(voice_b64) // 1024} KB (base64)")

    payload = {
        "input": {
            "texts": texts,
            "voice_file_base64": voice_b64,
            "temperature": temperature,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
    }
    if language:
        payload["input"]["language"] = language

    # Submit
    print(f"\nSubmitting job to endpoint: {endpoint_id}")
    print(f"  Texts: {texts}")
    print(f"  Language: {language or '(default)'}")
    print(f"  Temperature: {temperature}, Exaggeration: {exaggeration}, CFG: {cfg_weight}")

    job_id = submit_job(endpoint_id, headers, payload)
    print(f"  Job ID: {job_id}\n")

    # Poll
    print("Polling for result...")
    output = poll_result(endpoint_id, job_id, headers, timeout=timeout)

    # Save
    save_audio_files(output, OUTPUT_DIR)
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Test Chatterbox TTS RunPod endpoint")
    parser.add_argument(
        "--api-key",
        default=RUNPOD_API_KEY,
        help="RunPod API key (or set RUNPOD_API_KEY env var)",
    )
    parser.add_argument(
        "--endpoint-id",
        default=ENDPOINT_ID,
        help="RunPod endpoint ID (or set RUNPOD_ENDPOINT_ID env var)",
    )
    parser.add_argument(
        "--voice-pt",
        default=os.getenv("VOICE_PT_PATH", "voice.pt"),
        help="Path to .pt voice conditionals file (default: voice.pt)",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=[
            "Hello! This is a test of the Chatterbox text to speech system.",
            "How are you doing today? I hope everything is going well.",
            "I am a software engineer and I love to code.",
            "My name is John Doe and I am a software engineer.",
            "what is the weather in Tokyo?",
            "what is the weather in London?",
            "what is the weather in Paris?",
            "what is the weather in Berlin?",
            "what is the weather in Rome?",
            "what is the weather in Madrid?",
            "what is the weather in Milan?",
            "what is the weather in Amsterdam?",
            "what is the weather in Vienna?",
            "what is the weather in Zurich?",
            ],
        help="Texts to generate audio for",
    )
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--exaggeration", type=float, default=0.6)
    parser.add_argument("--cfg-weight", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=300, help="Max wait time in seconds")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: RunPod API key required. Use --api-key or set RUNPOD_API_KEY env var.")
        sys.exit(1)
    if not args.endpoint_id:
        print("Error: Endpoint ID required. Use --endpoint-id or set RUNPOD_ENDPOINT_ID env var.")
        sys.exit(1)
    if not os.path.exists(args.voice_pt):
        print(f"Error: Voice file not found: {args.voice_pt}")
        print("  Provide a .pt file via --voice-pt or place voice.pt in this directory.")
        sys.exit(1)

    run_test(
        api_key=args.api_key,
        endpoint_id=args.endpoint_id,
        voice_pt_path=args.voice_pt,
        texts=args.texts,
        language=args.language,
        temperature=args.temperature,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()


