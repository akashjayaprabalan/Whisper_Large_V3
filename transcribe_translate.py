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

def chunk_for_translation(text: str, tokenizer: Any, max_input_tokens: int) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []

    def token_count(value: str) -> int:
        return len(tokenizer(value, add_special_tokens=True).input_ids)

    for sentence in sentences:
        candidate = " ".join([*current, sentence]).strip()
        if current and token_count(candidate) > max_input_tokens:
            chunks.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)

        while current and token_count(" ".join(current)) > max_input_tokens:
            too_long = current.pop()
            words = too_long.split()
            piece: list[str] = []
            for word in words:
                candidate_piece = " ".join([*piece, word]).strip()
                if piece and token_count(candidate_piece) > max_input_tokens:
                    chunks.append(" ".join(piece).strip())
                    piece = [word]
                else:
                    piece.append(word)
            if piece:
                current.insert(0, " ".join(piece).strip())

    if current:
        chunks.append(" ".join(current).strip())
    return chunks

def translate_to_english(
    torch: Any,
    model_cls: Any,
    tokenizer_cls: Any,
    model_id: str,
    text: str,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    tokenizer = tokenizer_cls.from_pretrained(model_id)
    model = model_cls.from_pretrained(model_id)
    model.to(device)
    model.eval()

    chunks = chunk_for_translation(text, tokenizer, max_input_tokens)
    translated: list[str] = []

    with torch.inference_mode():
        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            inputs = inputs.to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            translated.extend(
                tokenizer.batch_decode(generated, skip_special_tokens=True)
            )

    return "\n".join(part.strip() for part in translated if part.strip())

def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    output_text = payload.get("english") or payload["transcript"]
    if args.output_text:
        args.output_text.parent.mkdir(parents=True, exist_ok=True)
        args.output_text.write_text(output_text, encoding="utf-8")

    if not args.output_json and not args.output_text:
        print("\nTranscript:\n")
        print(payload["transcript"])
        if payload.get("english"):
            print("\nEnglish:\n")
            print(payload["english"])

def main() -> None:
    args = parse_args()
    if not args.audio.exists():
        raise SystemExit(f"Audio file not found: {args.audio}")

    (
        torch,
        seq2seq_model_cls,
        speech_model_cls,
        processor_cls,
        tokenizer_cls,
        pipeline_fn,
        compressed_tensors_config_cls,
    ) = import_runtime()

    requested_device = choose_device(torch, args.device)
    asr_model = args.asr_model or default_model_location(DEFAULT_ASR_DIR, DEFAULT_ASR_REPO)
    translation_model = args.translation_model or default_model_location(
        DEFAULT_TRANSLATION_DIR,
        DEFAULT_TRANSLATION_REPO,
    )

    print(f"Loading ASR model: {asr_model}")
    asr_pipe, asr_device = load_asr_pipeline(
        torch,
        speech_model_cls,
        processor_cls,
        pipeline_fn,
        compressed_tensors_config_cls,
        asr_model,
        requested_device,
    )
    print(f"Transcribing on {asr_device}: {args.audio}")
    asr_result = transcribe(
        asr_pipe,
        args.audio,
        args.language,
        30.0,
        5.0,
        None,
    )
    transcript = asr_result["text"].strip()

    english = None
    if not args.skip_translation and transcript:
        print(f"Loading translation model: {translation_model}")
        english = translate_to_english(
            torch,
            seq2seq_model_cls,
            tokenizer_cls,
            translation_model,
            transcript,
            requested_device,
            420,
            256,
            4,
        )

    payload = {
        "audio": str(args.audio),
        "language": args.language,
        "asr_model": asr_model,
        "translation_model": None if args.skip_translation else translation_model,
        "transcript": transcript,
        "english": english,
        "chunks": asr_result.get("chunks", []),
    }
    write_outputs(payload, args)


if __name__ == "__main__":
    main()
