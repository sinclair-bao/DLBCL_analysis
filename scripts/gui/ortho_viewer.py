#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""轴 / 冠 / 矢 联动三视图：CT + PET 融合 + 本底/映射 mask。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from display_utils import compose_rgb, slice_axial, slice_coronal, slice_sagittal
from volume_io import VolumeSet

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)


class _RgbView(pg.GraphicsLayoutWidget):
    clicked = Signal(float, float)

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setMinimumHeight(180)
        self.view = self.addViewBox(lockAspect=True, invertY=True)
        self.view.setMenuEnabled(False)
        self.item = pg.ImageItem()
        self.view.addItem(self.item)
        self.label = pg.LabelItem(title, color="#dddddd")
        self.addItem(self.label, row=1, col=0)
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_rgb(self, rgb: np.ndarray) -> None:
        img = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        self.item.setImage(img, autoLevels=False)

    def _on_click(self, ev) -> None:
        if self.item.image is None:
            return
        pos = self.item.mapFromScene(ev.scenePos())
        self.clicked.emit(float(pos.x()), float(pos.y()))


class OrthoViewer(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._vol: Optional[VolumeSet] = None
        self._i = self._j = self._k = 0

        self.axial = _RgbView("Axial")
        self.coronal = _RgbView("Coronal")
        self.sagittal = _RgbView("Sagittal")
        self.axial.clicked.connect(lambda x, y: self._click("axial", x, y))
        self.coronal.clicked.connect(lambda x, y: self._click("coronal", x, y))
        self.sagittal.clicked.connect(lambda x, y: self._click("sagittal", x, y))

        views = QHBoxLayout()
        views.addWidget(self.axial, 1)
        views.addWidget(self.coronal, 1)
        views.addWidget(self.sagittal, 1)

        self.slider_k = QSlider(Qt.Orientation.Horizontal)
        self.slider_j = QSlider(Qt.Orientation.Horizontal)
        self.slider_i = QSlider(Qt.Orientation.Horizontal)
        for sl in (self.slider_k, self.slider_j, self.slider_i):
            sl.valueChanged.connect(self._sliders_changed)

        self.chk_native = QCheckBox("本底 mask")
        self.chk_mapped = QCheckBox("映射 mask")
        self.chk_native.setChecked(True)
        self.chk_mapped.setChecked(True)
        self.chk_native.toggled.connect(self.refresh)
        self.chk_mapped.toggled.connect(self.refresh)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setValue(0.55)
        self.spin_alpha.valueChanged.connect(self.refresh)

        self.spin_suv = QDoubleSpinBox()
        self.spin_suv.setRange(1.0, 30.0)
        self.spin_suv.setSingleStep(0.5)
        self.spin_suv.setValue(6.0)
        self.spin_suv.valueChanged.connect(self.refresh)

        self.lbl_pos = QLabel("—")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("轴位"))
        controls.addWidget(self.slider_k, 1)
        controls.addWidget(QLabel("冠状"))
        controls.addWidget(self.slider_j, 1)
        controls.addWidget(QLabel("矢状"))
        controls.addWidget(self.slider_i, 1)
        controls.addWidget(self.chk_native)
        controls.addWidget(self.chk_mapped)
        controls.addWidget(QLabel("PET 透明度"))
        controls.addWidget(self.spin_alpha)
        controls.addWidget(QLabel("SUV 上限"))
        controls.addWidget(self.spin_suv)
        controls.addWidget(self.lbl_pos)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(views, 1)
        layout.addLayout(controls)

    def set_volumes(self, vol: Optional[VolumeSet]) -> None:
        self._vol = vol
        if vol is None:
            self.lbl_pos.setText("无图像")
            for view in (self.axial, self.coronal, self.sagittal):
                view.item.clear()
            return
        nx, ny, nz = vol.ct.shape
        self.slider_i.blockSignals(True)
        self.slider_j.blockSignals(True)
        self.slider_k.blockSignals(True)
        self.slider_i.setRange(0, max(nx - 1, 0))
        self.slider_j.setRange(0, max(ny - 1, 0))
        self.slider_k.setRange(0, max(nz - 1, 0))
        self._i, self._j, self._k = nx // 2, ny // 2, nz // 2
        self.slider_i.setValue(self._i)
        self.slider_j.setValue(self._j)
        self.slider_k.setValue(self._k)
        self.slider_i.blockSignals(False)
        self.slider_j.blockSignals(False)
        self.slider_k.blockSignals(False)
        self.refresh()

    def _sliders_changed(self) -> None:
        self._i = self.slider_i.value()
        self._j = self.slider_j.value()
        self._k = self.slider_k.value()
        self.refresh()

    def _click(self, which: str, x: float, y: float) -> None:
        if self._vol is None:
            return
        # 点击坐标是显示图的列/行，与 orient_* 输出一致；只作粗略跳层。
        if which == "axial":
            self.slider_k.setValue(self._k)
        elif which == "coronal":
            ny = self._vol.ct.shape[1]
            self.slider_j.setValue(int(np.clip(ny - 1 - y, 0, ny - 1)))
        else:
            nx = self._vol.ct.shape[0]
            self.slider_i.setValue(int(np.clip(x, 0, nx - 1)))

    def refresh(self) -> None:
        vol = self._vol
        if vol is None:
            return
        args = dict(
            pet_alpha=float(self.spin_alpha.value()),
            suv_max=float(self.spin_suv.value()),
            show_native=self.chk_native.isChecked(),
            show_mapped=self.chk_mapped.isChecked(),
        )
        native = vol.native
        mapped = vol.mapped
        ax = compose_rgb(
            slice_axial(vol.ct, self._k),
            slice_axial(vol.pet, self._k),
            slice_axial(native, self._k) if native is not None else None,
            slice_axial(mapped, self._k) if mapped is not None else None,
            **args,
        )
        co = compose_rgb(
            slice_coronal(vol.ct, self._j),
            slice_coronal(vol.pet, self._j),
            slice_coronal(native, self._j) if native is not None else None,
            slice_coronal(mapped, self._j) if mapped is not None else None,
            **args,
        )
        sa = compose_rgb(
            slice_sagittal(vol.ct, self._i),
            slice_sagittal(vol.pet, self._i),
            slice_sagittal(native, self._i) if native is not None else None,
            slice_sagittal(mapped, self._i) if mapped is not None else None,
            **args,
        )
        self.axial.set_rgb(ax)
        self.coronal.set_rgb(co)
        self.sagittal.set_rgb(sa)
        suv = float(vol.pet[self._i, self._j, self._k])
        hu = float(vol.ct[self._i, self._j, self._k])
        self.lbl_pos.setText(
            f"{vol.study_date}  ijk=({self._i},{self._j},{self._k})  HU={hu:.0f}  SUV={suv:.2f}"
        )
