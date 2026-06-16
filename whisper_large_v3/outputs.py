from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_payload(
    audio_path: Path,
    language: str,
    asr_model: str,
    translation_model: str,
    skip_translation: bool,
    transcript: str,
    english: str | None,
    asr_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "audio": str(audio_path),
        "language": language,
        "asr_model": asr_model,
        "translation_model": None if skip_translation else translation_model,
        "transcript": transcript,
        "english": english,
        "chunks": asr_result.get("chunks", []),
    }


def payload_text(payload: dict[str, Any]) -> str:
    return payload.get("english") or payload["transcript"]


def write_outputs(
    payload: dict[str, Any],
    output_json: Path | None = None,
    output_text: Path | None = None,
) -> None:
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if output_text:
        output_text.parent.mkdir(parents=True, exist_ok=True)
        output_text.write_text(payload_text(payload), encoding="utf-8")

    if not output_json and not output_text:
        print("\nTranscript:\n")
        print(payload["transcript"])
        if payload.get("english"):
            print("\nEnglish:\n")
            print(payload["english"])
