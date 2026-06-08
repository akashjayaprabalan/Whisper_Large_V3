from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_ASR_REPO = "Reza2kn/openai_whisper-large-v3-NVFP4"
DEFAULT_TRANSLATION_REPO = "PontifexMaximus/opus-mt-iir-en-finetuned-fa-to-en"

ASR_ALLOW_PATTERNS = [
    "README.md",
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.yaml",
]

TRANSLATION_ALLOW_PATTERNS = [
    "README.md",
    "*.json",
    "*.spm",
    "*.model",
    "*.txt",
    "pytorch_model.bin",
    "vocab.json",
]


def download_repo(repo_id: str, local_dir: Path, allow_patterns: list[str]) -> str:
    local_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the local ASR and text-translation models."
    )
    parser.add_argument("--asr-repo", default=DEFAULT_ASR_REPO)
    parser.add_argument("--asr-dir", type=Path, default=Path("models/asr"))
    parser.add_argument("--translation-repo", default=DEFAULT_TRANSLATION_REPO)
    parser.add_argument(
        "--translation-dir", type=Path, default=Path("models/translation_fa_en")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Downloading ASR model: {args.asr_repo} -> {args.asr_dir}")
    asr_path = download_repo(args.asr_repo, args.asr_dir, ASR_ALLOW_PATTERNS)
    print(f"ASR model ready at {asr_path}")


if __name__ == "__main__":
    main()
