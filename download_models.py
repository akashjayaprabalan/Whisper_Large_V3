from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ASR_REPO = "Reza2kn/openai_whisper-large-v3-NVFP4"

ASR_ALLOW_PATTERNS = [
    "README.md",
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the local ASR and text-translation models."
    )
    parser.add_argument("--asr-repo", default=DEFAULT_ASR_REPO)
    parser.add_argument("--asr-dir", type=Path, default=Path("models/asr"))
    return parser.parse_args()


def main() -> None:
    parse_args()


if __name__ == "__main__":
    main()
