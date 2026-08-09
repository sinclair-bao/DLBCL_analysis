"""Plot feature distributions and summary figures."""

from __future__ import annotations

from pathlib import Path


def plot_feature_distributions(table_csv: Path, figure_out: Path) -> None:
    """Placeholder: read a feature table and write a figure."""
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(f"plot_feature_distributions({table_csv}, {figure_out})")


if __name__ == "__main__":
    print("Run via main.py or import plot_feature_distributions().")
