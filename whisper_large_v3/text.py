from __future__ import annotations

import re
from typing import Any


SENTENCE_END_RE = re.compile(r"(?<=[.!?\u061f\u061b\u06d4])\s+")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [part.strip() for part in SENTENCE_END_RE.split(text) if part.strip()]


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
