from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .asr import load_asr_pipeline, transcribe
from .config import (
    DEFAULT_ASR_DIR,
    DEFAULT_ASR_REPO,
    DEFAULT_TRANSLATION_DIR,
    DEFAULT_TRANSLATION_REPO,
    default_model_location,
)
from .devices import choose_device
from .outputs import build_payload, write_outputs
from .runtime import import_runtime
from .translation import translate_to_english


def build_parser() -> argparse.ArgumentParser:
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
        "--chunk-length-s",
        type=float,
        default=30.0,
        help="ASR chunk length in seconds for long audio.",
    )
    parser.add_argument(
        "--stride-length-s",
        type=float,
        default=5.0,
        help="ASR overlap in seconds between chunks.",
    )
    parser.add_argument(
        "--translation-max-input-tokens",
        type=int,
        default=420,
        help="Maximum tokens per translation chunk.",
    )
    parser.add_argument(
        "--translation-max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per translation chunk.",
    )
    parser.add_argument(
        "--asr-max-new-tokens",
        type=int,
        help="Optional cap for Whisper decode length. Useful for smoke tests.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam count for translation generation.",
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.audio.exists():
        raise SystemExit(f"Audio file not found: {args.audio}")

    runtime = import_runtime()
    requested_device = choose_device(runtime.torch, args.device)
    asr_model = args.asr_model or default_model_location(DEFAULT_ASR_DIR, DEFAULT_ASR_REPO)
    translation_model = args.translation_model or default_model_location(
        DEFAULT_TRANSLATION_DIR,
        DEFAULT_TRANSLATION_REPO,
    )

    print(f"Loading ASR model: {asr_model}")
    asr_pipe, asr_device = load_asr_pipeline(
        runtime.torch,
        runtime.speech_model_cls,
        runtime.processor_cls,
        runtime.pipeline_fn,
        runtime.compressed_tensors_config_cls,
        asr_model,
        requested_device,
    )
    print(f"Transcribing on {asr_device}: {args.audio}")
    asr_result = transcribe(
        asr_pipe,
        args.audio,
        args.language,
        args.chunk_length_s,
        args.stride_length_s,
        args.asr_max_new_tokens,
    )
    transcript = asr_result["text"].strip()

    english = None
    if not args.skip_translation and transcript:
        print(f"Loading translation model: {translation_model}")
        english = translate_to_english(
            runtime.torch,
            runtime.seq2seq_model_cls,
            runtime.tokenizer_cls,
            translation_model,
            transcript,
            requested_device,
            args.translation_max_input_tokens,
            args.translation_max_new_tokens,
            args.num_beams,
        )

    payload = build_payload(
        audio_path=args.audio,
        language=args.language,
        asr_model=asr_model,
        translation_model=translation_model,
        skip_translation=args.skip_translation,
        transcript=transcript,
        english=english,
        asr_result=asr_result,
    )
    write_outputs(payload, args.output_json, args.output_text)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))
