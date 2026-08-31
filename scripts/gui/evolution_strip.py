#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""最多三列冠状 MIP：基线 / 中期 / 末期；等比例、统一缩放。"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from display_utils import MAPPED_RGB, NATIVE_RGB, apply_pet_gray, coronal_mip
from volume_io import VolumeSet

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)


class _MipView(pg.GraphicsLayoutWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setMinimumHeight(120)
        self.box = self.addViewBox(lockAspect=True, invertY=True)
        self.box.setMenuEnabled(False)
        self.box.disableAutoRange()
        self.box.setMouseEnabled(x=False, y=False)
        self.item = pg.ImageItem()
        self.box.addItem(self.item)
        self._label = pg.LabelItem(title, color="#dddddd")
        self.addItem(self._label, row=1, col=0)

    def set_title(self, text: str, *, highlight: bool = False) -> None:
        color = "#ffd24a" if highlight else "#dddddd"
        self._label.setText(text, color=color)

    def set_rgb(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self.item.clear()
            return
        img = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        cur = self.item.image
        if cur is not None and cur.shape == img.shape and cur.dtype == img.dtype:
            np.copyto(cur, img)
            self.item.updateImage()
        else:
            self.item.setImage(img, autoLevels=False)

    def set_zoom(self, percent: int) -> None:
        if self.item.image is None:
            return
        h, w = self.item.image.shape[:2]
        scale = max(percent, 10) / 100.0
        half_w = (w / 2.0) / scale
        half_h = (h / 2.0) / scale
        self.box.setRange(
            xRange=(w / 2.0 - half_w, w / 2.0 + half_w),
            yRange=(h / 2.0 - half_h, h / 2.0 + half_h),
            padding=0,
        )


class EvolutionStrip(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._views = [_MipView("—") for _ in range(3)]
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for v in self._views:
            row.addWidget(v, 1)

        self.spin_zoom = QSpinBox()
        self.spin_zoom.setRange(50, 400)
        self.spin_zoom.setValue(100)
        self.spin_zoom.setSuffix("%")
        self.spin_zoom.valueChanged.connect(self._apply_zoom)
        btn_reset = QPushButton("复位")
        btn_reset.clicked.connect(lambda: self.spin_zoom.setValue(100))
        bar = QHBoxLayout()
        bar.addWidget(QLabel("MIP 等比例缩放"))
        bar.addWidget(self.spin_zoom)
        bar.addWidget(btn_reset)
        bar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row, 1)
        layout.addLayout(bar)
        self.setMinimumHeight(140)
        self._highlight: str | None = None

    def wheelEvent(self, ev) -> None:
        step = 10 if ev.angleDelta().y() > 0 else -10
        self.spin_zoom.setValue(int(np.clip(self.spin_zoom.value() + step, 50, 400)))
        ev.accept()

    def clear(self) -> None:
        for v in self._views:
            v.set_rgb(None)
            v.set_title("指定时间点后显示 MIP 演变")

    def set_highlight(self, role: str | None) -> None:
        self._highlight = role

    def set_volumes(
        self,
        items: list[tuple[str, VolumeSet]],
        *,
        suv_max: float = 6.0,
        show_native: bool = True,
        show_mapped: bool = True,
        highlight_role: str | None = None,
    ) -> None:
        if highlight_role is not None:
            self._highlight = highlight_role
        if not items:
            self.clear()
            return
        for i, v in enumerate(self._views):
            if i >= len(items):
                v.set_rgb(None)
                v.set_title("—")
                continue
            title, vol = items[i]
            pet_mip = coronal_mip(np.clip(vol.pet, 0.0, suv_max))
            rgb = apply_pet_gray(pet_mip, 0.0, suv_max)
            if show_native and vol.native is not None:
                rgb = _blend_mask(rgb, coronal_mip(vol.native.astype(np.float32)), NATIVE_RGB)
            if show_mapped and vol.mapped is not None:
                rgb = _blend_mask(rgb, coronal_mip(vol.mapped.astype(np.float32)), MAPPED_RGB)
            v.set_rgb(rgb)
            v.set_title(title, highlight=(vol.role == self._highlight))
        self._apply_zoom(self.spin_zoom.value())

    def _apply_zoom(self, percent: int) -> None:
        for v in self._views:
            v.set_zoom(int(percent))

    def save_png(self, path) -> None:
        self.grab().save(str(path))


def _blend_mask(rgb: np.ndarray, mask_mip: np.ndarray, color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    hit = mask_mip > 0
    if not np.any(hit):
        return rgb
    out = rgb.copy()
    out[hit] = out[hit] * (1.0 - alpha) + color * alpha
    return out
