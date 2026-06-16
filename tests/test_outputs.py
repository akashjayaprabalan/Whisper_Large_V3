from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from whisper_large_v3.outputs import build_payload, payload_text, write_outputs


class OutputTests(unittest.TestCase):
    def test_build_payload_omits_translation_model_when_skipped(self) -> None:
        source_text = "\u0633\u0644\u0627\u0645"
        payload = build_payload(
            audio_path=Path("sample.wav"),
            language="fa",
            asr_model="asr",
            translation_model="translator",
            skip_translation=True,
            transcript=source_text,
            english=None,
            asr_result={"chunks": [{"text": source_text}]},
        )

        self.assertIsNone(payload["translation_model"])
        self.assertEqual(payload["chunks"], [{"text": source_text}])

    def test_payload_text_prefers_english(self) -> None:
        self.assertEqual(
            payload_text({"transcript": "source", "english": "translated"}),
            "translated",
        )
        self.assertEqual(payload_text({"transcript": "source", "english": None}), "source")

    def test_write_outputs_writes_json_and_text(self) -> None:
        payload = {"transcript": "source", "english": "translated"}
        with tempfile.TemporaryDirectory() as tmp:
            output_json = Path(tmp) / "nested" / "result.json"
            output_text = Path(tmp) / "nested" / "result.txt"

            write_outputs(payload, output_json, output_text)

            self.assertEqual(json.loads(output_json.read_text(encoding="utf-8")), payload)
            self.assertEqual(output_text.read_text(encoding="utf-8"), "translated")


if __name__ == "__main__":
    unittest.main()
