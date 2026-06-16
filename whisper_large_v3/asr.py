from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

from .devices import pipeline_device


def _clear_cuda_cache(torch: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
                _clear_cuda_cache(torch)
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
