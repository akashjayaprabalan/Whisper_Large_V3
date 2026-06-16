from __future__ import annotations

from typing import Any

from .text import chunk_for_translation


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
