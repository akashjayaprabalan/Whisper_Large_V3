from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeDependencies:
    torch: Any
    seq2seq_model_cls: Any
    speech_model_cls: Any
    processor_cls: Any
    tokenizer_cls: Any
    pipeline_fn: Any
    compressed_tensors_config_cls: Any


def import_runtime() -> RuntimeDependencies:
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

    return RuntimeDependencies(
        torch=torch,
        seq2seq_model_cls=AutoModelForSeq2SeqLM,
        speech_model_cls=AutoModelForSpeechSeq2Seq,
        processor_cls=AutoProcessor,
        tokenizer_cls=AutoTokenizer,
        pipeline_fn=pipeline,
        compressed_tensors_config_cls=CompressedTensorsConfig,
    )
