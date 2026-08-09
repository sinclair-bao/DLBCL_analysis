"""Segmentation inference / classical segmentation placeholders."""

from __future__ import annotations

from pathlib import Path


def run_segmentation(image_path: Path, mask_out: Path) -> None:
    """Placeholder: produce a segmentation mask for one image."""
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(f"run_segmentation({image_path}, {mask_out})")


if __name__ == "__main__":
    print("Run via main.py or import run_segmentation().")
