from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from .config import (
    ASR_ALLOW_PATTERNS,
    DEFAULT_ASR_REPO,
    DEFAULT_TRANSLATION_REPO,
    TRANSLATION_ALLOW_PATTERNS,
)

SnapshotDownload = Callable[..., str]


def import_snapshot_download() -> SnapshotDownload:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing Hugging Face dependency. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    return snapshot_download


def download_repo(
    repo_id: str,
    local_dir: Path,
    allow_patterns: list[str],
    snapshot_download: SnapshotDownload | None = None,
) -> str:
    local_dir.mkdir(parents=True, exist_ok=True)
    download = snapshot_download or import_snapshot_download()
    return download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the local ASR and text-translation models."
    )
    parser.add_argument("--asr-repo", default=DEFAULT_ASR_REPO)
    parser.add_argument("--asr-dir", type=Path, default=Path("models/asr"))
    parser.add_argument("--translation-repo", default=DEFAULT_TRANSLATION_REPO)
    parser.add_argument(
        "--translation-dir", type=Path, default=Path("models/translation_fa_en")
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Only download the ASR model.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> None:
    print(f"Downloading ASR model: {args.asr_repo} -> {args.asr_dir}")
    asr_path = download_repo(args.asr_repo, args.asr_dir, ASR_ALLOW_PATTERNS)
    print(f"ASR model ready at {asr_path}")

    if args.skip_translation:
        return

    print(
        "Downloading translation model: "
        f"{args.translation_repo} -> {args.translation_dir}"
    )
    translation_path = download_repo(
        args.translation_repo,
        args.translation_dir,
        TRANSLATION_ALLOW_PATTERNS,
    )
    print(f"Translation model ready at {translation_path}")


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))
