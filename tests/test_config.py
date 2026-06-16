from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from whisper_large_v3.config import default_model_location


class ConfigTests(unittest.TestCase):
    def test_default_model_location_prefers_existing_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()

            self.assertEqual(
                default_model_location(model_dir, "owner/repo"),
                str(model_dir),
            )

    def test_default_model_location_uses_repo_when_local_directory_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = Path(tmp) / "missing"

            self.assertEqual(
                default_model_location(missing_dir, "owner/repo"),
                "owner/repo",
            )


if __name__ == "__main__":
    unittest.main()
