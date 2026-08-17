#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""最多三列冠状 MIP：基线 / 中期 / 末期，叠本底（红）与映射（青）mask。"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from display_utils import MAPPED_RGB, NATIVE_RGB, apply_pet_cmap, coronal_mip
from volume_io import VolumeSet


class EvolutionStrip(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.fig = Figure(figsize=(9, 3.2), facecolor="#121212")
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setMinimumHeight(220)

    def clear(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#121212")
        ax.axis("off")
        ax.text(0.5, 0.5, "指定时间点后显示 MIP 演变", ha="center", va="center", color="#888888")
        self.canvas.draw_idle()

    def set_volumes(
        self,
        items: list[tuple[str, VolumeSet]],
        *,
        suv_max: float = 6.0,
        show_native: bool = True,
        show_mapped: bool = True,
    ) -> None:
        self.fig.clear()
        if not items:
            self.clear()
            return
        n = len(items)
        axes = self.fig.subplots(1, n) if n > 1 else [self.fig.add_subplot(111)]
        if not isinstance(axes, np.ndarray) and not isinstance(axes, list):
            axes = [axes]
        for ax, (title, vol) in zip(axes, items):
            ax.set_facecolor("#000000")
            pet_mip = coronal_mip(np.clip(vol.pet, 0.0, suv_max))
            rgb = apply_pet_cmap(pet_mip, 0.0, suv_max)
            if show_native and vol.native is not None:
                rgb = _blend_mask(rgb, coronal_mip(vol.native.astype(np.float32)), NATIVE_RGB)
            if show_mapped and vol.mapped is not None:
                rgb = _blend_mask(rgb, coronal_mip(vol.mapped.astype(np.float32)), MAPPED_RGB)
            ax.imshow(np.clip(rgb, 0, 1), aspect="auto", interpolation="bilinear")
            ax.set_title(title, color="white", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        self.fig.tight_layout(pad=0.3)
        self.canvas.draw_idle()

    def save_png(self, path) -> None:
        self.fig.savefig(str(path), dpi=120, facecolor=self.fig.get_facecolor())


def _blend_mask(rgb: np.ndarray, mask_mip: np.ndarray, color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    hit = mask_mip > 0
    if not np.any(hit):
        return rgb
    out = rgb.copy()
    out[hit] = out[hit] * (1.0 - alpha) + color * alpha
    return out
