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

def import_runtime() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            AutoTokenizer,
            pipeline,
        )
        from transformers.utils.quantization_config import CompressedTensorsConfig
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependencies. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    return (
        torch,
        AutoModelForSeq2SeqLM,
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        AutoTokenizer,
        pipeline,
        CompressedTensorsConfig,
    )

def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but torch cannot see a CUDA device.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested, but torch cannot see an MPS device.")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def pipeline_device(torch: Any, device: str) -> Any:
    if device == "cuda":
        return 0
    if device == "mps":
        return torch.device("mps")
    return -1

def load_asr_pipeline(
    torch: Any,
    auto_model_cls: Any,
    processor_cls: Any,
    pipeline_fn: Any,
    compressed_tensors_config_cls: Any,
    model_id: str,
    requested_device: str,
) -> tuple[Any, str]:
    candidates = [requested_device]
    if requested_device != "cpu":
        candidates.append("cpu")

    last_error: Exception | None = None
    for device in candidates:
        try:
            processor = processor_cls.from_pretrained(model_id)
            model = auto_model_cls.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                quantization_config=compressed_tensors_config_cls(
                    run_compressed=False
                ),
                dtype=torch.bfloat16,
            )
            model.to(device)
            model.eval()
            pipe = pipeline_fn(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=pipeline_device(torch, device),
                dtype=torch.bfloat16,
            )
            return pipe, device
        except Exception as exc:
            last_error = exc
            if device != "cpu":
                print(
                    f"Could not load ASR on {device}; retrying on CPU. "
                    f"Reason: {exc}",
                    file=sys.stderr,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                break

    raise RuntimeError(f"Could not load ASR model {model_id!r}") from last_error

def transcribe(
    asr_pipe: Any,
    audio_path: Path,
    language: str,
    chunk_length_s: float,
    stride_length_s: float,
    max_new_tokens: int | None,
) -> dict[str, Any]:
    generate_kwargs: dict[str, Any] = {"task": "transcribe"}
    if language.lower() != "auto":
        generate_kwargs["language"] = language
    if max_new_tokens is not None:
        generate_kwargs["max_new_tokens"] = max_new_tokens

    result = asr_pipe(
        str(audio_path),
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
    )
    return dict(result)

def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?؟؛۔])\s+", text)
        if part.strip()
    ]

def main() -> None:
    parse_args()


if __name__ == "__main__":
    main()
