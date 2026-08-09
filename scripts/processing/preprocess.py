"""Image preprocessing utilities (normalization, resampling, etc.)."""

from __future__ import annotations

from pathlib import Path


def preprocess_subject(raw_dir: Path, out_dir: Path) -> None:
    """Placeholder: convert/normalize one subject from raw to interim."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(f"preprocess_subject({raw_dir}, {out_dir})")


if __name__ == "__main__":
    print("Run via main.py or import preprocess_subject().")
