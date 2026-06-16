from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from whisper_large_v3.download_cli import download_repo, parse_args


class DownloadCliTests(unittest.TestCase):
    def test_parse_args_supports_skip_translation(self) -> None:
        args = parse_args(["--skip-translation", "--asr-repo", "owner/asr"])

        self.assertTrue(args.skip_translation)
        self.assertEqual(args.asr_repo, "owner/asr")

    def test_download_repo_creates_destination_and_passes_allow_patterns(self) -> None:
        calls = []

        def fake_snapshot_download(**kwargs: object) -> str:
            calls.append(kwargs)
            return "/tmp/snapshot"

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "model"
            result = download_repo(
                "owner/repo",
                target,
                ["*.json"],
                snapshot_download=fake_snapshot_download,
            )

            self.assertEqual(result, "/tmp/snapshot")
            self.assertTrue(calls)
            self.assertEqual(calls[0]["repo_id"], "owner/repo")
            self.assertEqual(calls[0]["local_dir"], target)
            self.assertEqual(calls[0]["allow_patterns"], ["*.json"])
            self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
