from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the local ASR and text-translation models."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()


if __name__ == "__main__":
    main()
