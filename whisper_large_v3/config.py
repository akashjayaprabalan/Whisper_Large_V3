from __future__ import annotations

from pathlib import Path


DEFAULT_ASR_REPO = "Reza2kn/openai_whisper-large-v3-NVFP4"
DEFAULT_TRANSLATION_REPO = "PontifexMaximus/opus-mt-iir-en-finetuned-fa-to-en"
DEFAULT_ASR_DIR = Path("models/asr")
DEFAULT_TRANSLATION_DIR = Path("models/translation_fa_en")

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


def default_model_location(local_dir: Path, repo_id: str) -> str:
    """Prefer an existing local model directory, otherwise use the HF repo id."""
    return str(local_dir) if local_dir.exists() else repo_id
