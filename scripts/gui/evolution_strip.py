#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""任意数量时间点的冠状 MIP；等比例、统一缩放。"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from display_utils import (
    DEFAULT_PET_CMAP,
    MAPPED_RGB,
    NATIVE_RGB,
    apply_pet_cmap,
    voxel_to_display,
)
from ortho_viewer import _LateralityMixin
from volume_io import MipData

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)


class _MipView(_LateralityMixin, pg.GraphicsLayoutWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setMinimumHeight(120)
        self.setMinimumWidth(220)
        self.box = self.addViewBox(lockAspect=True, invertY=True)
        self.box.setMenuEnabled(False)
        self.box.disableAutoRange()
        self.box.setMouseEnabled(x=False, y=False)
        self.item = pg.ImageItem()
        self.box.addItem(self.item)
        pen = pg.mkPen("#00e5ff", width=1)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        for line in (self.vline, self.hline):
            line.setZValue(10)
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            line.setVisible(False)
            self.box.addItem(line)
        self._zoom_percent = 100
        self._label = pg.LabelItem(title, color="#dddddd")
        self.addItem(self._label, row=1, col=0)
        self.setToolTip("冠状 MIP 十字线同步当前检查的左右、头脚位置；前后深度在投影中合并。")
        self._init_laterality("coronal")

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._place_laterality()
        if hasattr(self, "box") and hasattr(self, "_zoom_percent"):
            self.set_zoom(self._zoom_percent)

    def set_crosshair(self, col: float, row: float, visible: bool) -> None:
        self.vline.setVisible(visible)
        self.hline.setVisible(visible)
        if visible:
            self.vline.setPos(float(col))
            self.hline.setPos(float(row))

    def set_title(self, text: str, *, highlight: bool = False) -> None:
        color = "#ffd24a" if highlight else "#dddddd"
        self._label.setText(text, color=color)

    def set_rgb(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self.item.clear()
            self.set_crosshair(0, 0, False)
            return
        img = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        cur = self.item.image
        shape_changed = cur is None or cur.shape != img.shape
        if cur is not None and cur.shape == img.shape and cur.dtype == img.dtype:
            np.copyto(cur, img)
            self.item.updateImage()
        else:
            self.item.setImage(img, autoLevels=False)
        if shape_changed:
            self.set_zoom(self._zoom_percent)

    def set_zoom(self, percent: int) -> None:
        self._zoom_percent = int(percent)
        if self.item.image is None:
            return
        h, w = self.item.image.shape[:2]
        scale = max(percent, 10) / 100.0
        aspect = max(self.box.width(), 1.0) / max(self.box.height(), 1.0)
        fit_w = max(float(w), float(h) * aspect)
        fit_h = fit_w / aspect
        half_w = fit_w / (2.0 * scale)
        half_h = fit_h / (2.0 * scale)
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
        self._row = row
        row.setContentsMargins(0, 0, 0, 0)
        for v in self._views:
            row.addWidget(v)
        self._mip_widget = QWidget()
        self._mip_widget.setLayout(row)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setWidget(self._mip_widget)

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
        layout.addWidget(self._scroll, 1)
        layout.addLayout(bar)
        self.setMinimumHeight(140)
        self._highlight: str | None = None
        self._view_mips: list[MipData | None] = []
        self._cursor_date: str | None = None
        self._cursor_ijk = (0, 0, 0)
        self._cursor_visible = False

    def wheelEvent(self, ev) -> None:
        step = 10 if ev.angleDelta().y() > 0 else -10
        self.spin_zoom.setValue(int(np.clip(self.spin_zoom.value() + step, 50, 400)))
        ev.accept()

    def clear(self) -> None:
        self._view_mips = []
        for v in self._views:
            v.set_rgb(None)
            v.set_title("选择患者及检查后显示 MIP")

    def set_cursor(self, study_date: str | None, i: int, j: int, k: int,
                   visible: bool) -> None:
        self._cursor_date = study_date
        self._cursor_ijk = (i, j, k)
        self._cursor_visible = visible
        self._update_cursor()

    def _update_cursor(self) -> None:
        for view, mip in zip(self._views, self._view_mips):
            if (not self._cursor_visible or mip is None or
                    mip.study_date != self._cursor_date):
                view.set_crosshair(0, 0, False)
                continue
            col, row = voxel_to_display("coronal", *self._cursor_ijk,
                                        mip.volume_shape)
            view.set_crosshair(col, row, True)

    def set_highlight(self, role: str | None) -> None:
        self._highlight = role

    def set_volumes(
        self,
        items: list[tuple[str, MipData | None]],
        *,
        suv_max: float = 6.0,
        show_native: bool = True,
        show_mapped: bool = True,
        highlight_role: str | None = None,
        highlight_date: str | None = None,
        pet_cmap: str = DEFAULT_PET_CMAP,
    ) -> None:
        if highlight_role is not None:
            self._highlight = highlight_role
        if not items:
            self.clear()
            return
        while len(self._views) < len(items):
            view = _MipView("—")
            self._views.append(view)
            self._row.addWidget(view)
        self._view_mips = [mip for _, mip in items]
        for i, v in enumerate(self._views):
            if i >= len(items):
                v.set_rgb(None)
                v.set_title("—")
                v.hide()
                continue
            v.show()
            title, mip = items[i]
            if mip is None:
                v.set_rgb(None)
                v.set_title(title)
                continue
            pet_mip = np.clip(mip.pet, 0.0, suv_max)
            rgb = apply_pet_cmap(pet_mip, 0.0, suv_max, pet_cmap)
            if show_native:
                rgb = _blend_mask(rgb, mip.native, NATIVE_RGB)
            if show_mapped and mip.mapped is not None:
                rgb = _blend_mask(rgb, mip.mapped, MAPPED_RGB)
            v.set_rgb(rgb)
            v.set_title(title, highlight=(mip.study_date == highlight_date if highlight_date
                                          else mip.role == self._highlight))
        self._mip_widget.adjustSize()
        self._apply_zoom(self.spin_zoom.value())
        self._update_cursor()

    def _apply_zoom(self, percent: int) -> None:
        for v in self._views:
            v.set_zoom(int(percent))

    def save_png(self, path) -> None:
        self._mip_widget.grab().save(str(path))


def _blend_mask(rgb: np.ndarray, mask_mip: np.ndarray, color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    hit = mask_mip > 0
    if not np.any(hit):
        return rgb
    out = rgb.copy()
    out[hit] = out[hit] * (1.0 - alpha) + color * alpha
    return out
