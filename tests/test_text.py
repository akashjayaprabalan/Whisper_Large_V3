from __future__ import annotations

import unittest

from whisper_large_v3.text import chunk_for_translation, split_sentences


class _TokenResult:
    def __init__(self, input_ids: list[str]) -> None:
        self.input_ids = input_ids


class _WhitespaceTokenizer:
    def __call__(self, value: str, add_special_tokens: bool = True) -> _TokenResult:
        del add_special_tokens
        return _TokenResult(value.split())


class TextTests(unittest.TestCase):
    def test_split_sentences_handles_latin_and_persian_punctuation(self) -> None:
        persian_question = (
            "\u062d\u0627\u0644 \u0634\u0645\u0627 "
            "\u0686\u0637\u0648\u0631 \u0627\u0633\u062a\u061f"
        )
        self.assertEqual(
            split_sentences(f"  Hello world.   {persian_question}  Fine! "),
            ["Hello world.", persian_question, "Fine!"],
        )

    def test_chunk_for_translation_groups_sentences_under_token_limit(self) -> None:
        chunks = chunk_for_translation(
            "one two. three four. five six.",
            _WhitespaceTokenizer(),
            max_input_tokens=4,
        )

        self.assertEqual(chunks, ["one two. three four.", "five six."])

    def test_chunk_for_translation_splits_long_sentence_by_words(self) -> None:
        chunks = chunk_for_translation(
            "one two three four five.",
            _WhitespaceTokenizer(),
            max_input_tokens=3,
        )

        self.assertEqual(chunks, ["one two three", "four five."])


if __name__ == "__main__":
    unittest.main()
