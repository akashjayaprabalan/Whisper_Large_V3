from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_ASR_REPO = "Reza2kn/openai_whisper-large-v3-NVFP4"
DEFAULT_TRANSLATION_REPO = "PontifexMaximus/opus-mt-iir-en-finetuned-fa-to-en"
DEFAULT_ASR_DIR = Path("models/asr")
DEFAULT_TRANSLATION_DIR = Path("models/translation_fa_en")

def default_model_location(local_dir: Path, repo_id: str) -> str:
    return str(local_dir) if local_dir.exists() else repo_id

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe speech with Whisper large-v3 NVFP4, then translate the "
            "transcript to English with a small text model."
        )
    )
    parser.add_argument("audio", type=Path, help="Audio or video file to transcribe.")
    parser.add_argument(
        "--asr-model",
        default=None,
        help="Local ASR model directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--translation-model",
        default=None,
        help="Local translation model directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--language",
        default="fa",
        help="Whisper source language. Use 'auto' to let Whisper detect it.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device for both models.",
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Only produce the source-language transcript.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for the full JSON result.",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        help="Optional path for the English text, or transcript if translation is skipped.",
    )
    return parser.parse_args()

def main() -> None:
    parse_args()


if __name__ == "__main__":
    main()
