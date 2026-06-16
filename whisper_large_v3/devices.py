from __future__ import annotations

from typing import Any


def _mps_available(torch: Any) -> bool:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps and mps.is_available())


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but torch cannot see a CUDA device.")
        if requested == "mps" and not _mps_available(torch):
            raise SystemExit("MPS was requested, but torch cannot see an MPS device.")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def pipeline_device(torch: Any, device: str) -> Any:
    if device == "cuda":
        return 0
    if device == "mps":
        return torch.device("mps")
    return -1
