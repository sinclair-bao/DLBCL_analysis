#!/usr/bin/env python3
"""DLBCL analysis pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DLBCL imaging analysis pipeline")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "segment", "analyze", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Data root containing raw/interim/processed",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results",
        help="Results output root",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"[DLBCL] stage={args.stage}")
    print(f"[DLBCL] data_root={args.data_root}")
    print(f"[DLBCL] results_root={args.results_root}")
    # Wire scripts.processing / scripts.analysis modules here as they mature.
    if args.stage in ("preprocess", "all"):
        print("[DLBCL] preprocess: not implemented yet")
    if args.stage in ("segment", "all"):
        print("[DLBCL] segment: not implemented yet")
    if args.stage in ("analyze", "all"):
        print("[DLBCL] analyze: not implemented yet")


if __name__ == "__main__":
    main()
