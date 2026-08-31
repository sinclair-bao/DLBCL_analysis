#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""轴 / 冠 / 矢 联动三视图：显示模式、窗宽窗位、缩放、二维画笔。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from display_utils import (
    DEFAULT_WL,
    DEFAULT_WW,
    compose_rgb,
    ct_window_from_wl,
    slice_axial,
    slice_coronal,
    slice_sagittal,
    voxel_to_display,
)
from mask_ops import paint_disk, promote_new_islands
from volume_io import VolumeSet

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)


class _PaintItem(pg.ImageItem):
    paint_at = Signal(float, float, bool)
    clicked_xy = Signal(float, float)
    stroke_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.edit_mode = False

    def mouseClickEvent(self, ev) -> None:
        pos = ev.pos()
        if self.edit_mode:
            erase = ev.button() == Qt.MouseButton.RightButton
            self.paint_at.emit(float(pos.x()), float(pos.y()), erase)
            self.stroke_finished.emit()
            ev.accept()
            return
        self.clicked_xy.emit(float(pos.x()), float(pos.y()))
        ev.accept()

    def mouseDragEvent(self, ev) -> None:
        if not self.edit_mode:
            ev.ignore()
            return
        pos = ev.pos()
        erase = ev.button() == Qt.MouseButton.RightButton
        self.paint_at.emit(float(pos.x()), float(pos.y()), erase)
        if ev.isFinish():
            self.stroke_finished.emit()
        ev.accept()


class _RgbView(pg.GraphicsLayoutWidget):
    def __init__(self, title: str, view_name: str) -> None:
        super().__init__()
        self.view_name = view_name
        self.setMinimumHeight(120)
        self.box = self.addViewBox(lockAspect=True, invertY=True)
        self.box.setMenuEnabled(False)
        self.box.disableAutoRange()
        self.item = _PaintItem()
        self.box.addItem(self.item)
        pen = pg.mkPen("#00e5ff", width=1)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self.vline.setZValue(10)
        self.hline.setZValue(10)
        self.vline.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.hline.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.box.addItem(self.vline)
        self.box.addItem(self.hline)
        self.addItem(pg.LabelItem(title, color="#dddddd"), row=1, col=0)

    def set_crosshair(self, col: float, row: float) -> None:
        pos = (float(col), float(row))
        if getattr(self, "_xh", None) == pos:
            return
        self._xh = pos
        self.vline.setPos(pos[0])
        self.hline.setPos(pos[1])

    def set_crosshair_visible(self, on: bool) -> None:
        self.vline.setVisible(on)
        self.hline.setVisible(on)

    def set_rgb(self, rgb: np.ndarray) -> None:
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


class OrthoViewer(QWidget):
    mask_changed = Signal()
    role_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._vol: Optional[VolumeSet] = None
        self._i = self._j = self._k = 0
        self.highlight_label = 0
        self.current_label = 1
        self.brush_radius = 5
        self.active_view = "axial"
        self._stroke = False
        self._stroke_erase = False

        self.axial = _RgbView("Axial", "axial")
        self.coronal = _RgbView("Coronal", "coronal")
        self.sagittal = _RgbView("Sagittal", "sagittal")
        for v in (self.axial, self.coronal, self.sagittal):
            v.item.clicked_xy.connect(lambda x, y, vv=v: self._click(vv.view_name, x, y))
            v.item.paint_at.connect(lambda x, y, er, vv=v: self._paint(vv.view_name, x, y, er))
            v.item.stroke_finished.connect(self._end_stroke)

        views = QHBoxLayout()
        views.addWidget(self.axial, 1)
        views.addWidget(self.coronal, 1)
        views.addWidget(self.sagittal, 1)

        self.slider_k = QSlider(Qt.Orientation.Horizontal)
        self.slider_j = QSlider(Qt.Orientation.Horizontal)
        self.slider_i = QSlider(Qt.Orientation.Horizontal)
        for sl in (self.slider_k, self.slider_j, self.slider_i):
            sl.valueChanged.connect(self._sliders_changed)

        self.radio_fusion = QRadioButton("PET/CT")
        self.radio_ct = QRadioButton("仅 CT")
        self.radio_pet = QRadioButton("仅 PET")
        self.radio_fusion.setChecked(True)
        self._mode_group = QButtonGroup(self)
        for r in (self.radio_fusion, self.radio_ct, self.radio_pet):
            self._mode_group.addButton(r)
            r.toggled.connect(self.refresh)

        self.radio_bl = QRadioButton("基线")
        self.radio_in = QRadioButton("中期")
        self.radio_end = QRadioButton("末期")
        self._role_group = QButtonGroup(self)
        self._role_group.setExclusive(True)
        for r, role in (
            (self.radio_bl, "baseline"),
            (self.radio_in, "interim"),
            (self.radio_end, "end"),
        ):
            self._role_group.addButton(r)
            r.setEnabled(False)
            r.toggled.connect(lambda on, role=role: on and self.role_requested.emit(role))

        self.chk_native = QCheckBox("本底 mask")
        self.chk_mapped = QCheckBox("映射 mask")
        self.chk_edit = QCheckBox("编辑 mask")
        self.chk_crosshair = QCheckBox("十字线")
        self.chk_native.setChecked(True)
        self.chk_mapped.setChecked(True)
        self.chk_crosshair.setChecked(True)
        self.chk_native.toggled.connect(self.refresh)
        self.chk_mapped.toggled.connect(self.refresh)
        self.chk_edit.toggled.connect(self._toggle_edit)
        self.chk_crosshair.toggled.connect(self.refresh)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setValue(0.55)
        self.spin_alpha.valueChanged.connect(self.refresh)

        self.spin_suv_min = QDoubleSpinBox()
        self.spin_suv_min.setRange(0.0, 20.0)
        self.spin_suv_min.setValue(0.0)
        self.spin_suv_max = QDoubleSpinBox()
        self.spin_suv_max.setRange(0.5, 40.0)
        self.spin_suv_max.setValue(6.0)
        self.spin_suv_min.valueChanged.connect(self.refresh)
        self.spin_suv_max.valueChanged.connect(self.refresh)
        self.spin_suv = self.spin_suv_max

        self.spin_wl = QSpinBox()
        self.spin_wl.setRange(-1000, 3000)
        self.spin_wl.setValue(int(DEFAULT_WL))
        self.spin_ww = QSpinBox()
        self.spin_ww.setRange(1, 4000)
        self.spin_ww.setValue(int(DEFAULT_WW))
        self.spin_wl.valueChanged.connect(self.refresh)
        self.spin_ww.valueChanged.connect(self.refresh)

        self.spin_zoom = QSpinBox()
        self.spin_zoom.setRange(50, 400)
        self.spin_zoom.setValue(100)
        self.spin_zoom.setSuffix("%")
        self.spin_zoom.valueChanged.connect(self._apply_zoom)
        self.btn_zoom_reset = QPushButton("复位")
        self.btn_zoom_reset.clicked.connect(lambda: self.spin_zoom.setValue(100))

        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(3, 15)
        self.spin_brush.setValue(5)
        self.spin_brush.valueChanged.connect(lambda v: setattr(self, "brush_radius", int(v)))

        self.lbl_pos = QLabel("—")
        self.lbl_pos.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        for spin in (
            self.spin_alpha,
            self.spin_suv_min,
            self.spin_suv_max,
            self.spin_wl,
            self.spin_ww,
            self.spin_zoom,
            self.spin_brush,
        ):
            spin.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            spin.setMaximumWidth(90)

        slice_row = QHBoxLayout()
        slice_row.addWidget(QLabel("轴位"))
        slice_row.addWidget(self.slider_k, 1)
        slice_row.addWidget(QLabel("冠状"))
        slice_row.addWidget(self.slider_j, 1)
        slice_row.addWidget(QLabel("矢状"))
        slice_row.addWidget(self.slider_i, 1)
        slice_row.addWidget(self.radio_fusion)
        slice_row.addWidget(self.radio_ct)
        slice_row.addWidget(self.radio_pet)
        slice_row.addWidget(QLabel("查看"))
        slice_row.addWidget(self.radio_bl)
        slice_row.addWidget(self.radio_in)
        slice_row.addWidget(self.radio_end)

        tools = QGridLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setHorizontalSpacing(6)
        tools.setVerticalSpacing(4)
        r1 = [
            self.chk_native,
            self.chk_mapped,
            self.chk_edit,
            self.chk_crosshair,
            QLabel("笔刷"),
            self.spin_brush,
            QLabel("窗位"),
            self.spin_wl,
            QLabel("窗宽"),
            self.spin_ww,
        ]
        r2 = [
            QLabel("SUV"),
            self.spin_suv_min,
            self.spin_suv_max,
            QLabel("融合"),
            self.spin_alpha,
            QLabel("缩放"),
            self.spin_zoom,
            self.btn_zoom_reset,
            self.lbl_pos,
        ]
        for col, w in enumerate(r1):
            tools.addWidget(w, 0, col)
        for col, w in enumerate(r2):
            tools.addWidget(w, 1, col)
        tools.setColumnStretch(8, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(views, 1)
        layout.addLayout(slice_row)
        layout.addLayout(tools)

    def set_role_buttons(self, available: dict[str, bool], current: str | None) -> None:
        mapping = {
            "baseline": self.radio_bl,
            "interim": self.radio_in,
            "end": self.radio_end,
        }
        for role, btn in mapping.items():
            btn.blockSignals(True)
            btn.setEnabled(bool(available.get(role)))
            btn.setChecked(role == current and bool(available.get(role)))
            btn.blockSignals(False)

    def display_mode(self) -> str:
        if self.radio_ct.isChecked():
            return "ct"
        if self.radio_pet.isChecked():
            return "pet"
        return "fusion"

    def _toggle_edit(self, on: bool) -> None:
        for v in (self.axial, self.coronal, self.sagittal):
            v.item.edit_mode = bool(on)

    def set_volumes(self, vol: Optional[VolumeSet]) -> None:
        self._vol = vol
        if vol is None:
            self.lbl_pos.setText("无图像")
            for view in (self.axial, self.coronal, self.sagittal):
                view.item.clear()
                view.set_crosshair_visible(False)
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
        self._apply_zoom(self.spin_zoom.value())

    def volumes(self) -> Optional[VolumeSet]:
        return self._vol

    def _sliders_changed(self) -> None:
        self._i = self.slider_i.value()
        self._j = self.slider_j.value()
        self._k = self.slider_k.value()
        self.refresh()

    def _click(self, which: str, x: float, y: float) -> None:
        self.active_view = which
        if self._vol is None:
            return
        from display_utils import display_to_voxel

        ii, jj, kk = display_to_voxel(
            which, x, y, self._i, self._j, self._k, self._vol.ct.shape
        )
        self.slider_i.setValue(ii)
        self.slider_j.setValue(jj)
        self.slider_k.setValue(kk)

    def _push_undo(self) -> None:
        if self._vol is None:
            return
        self._vol._undo.append(self._vol.native.copy())
        if len(self._vol._undo) > 20:
            self._vol._undo.pop(0)

    def undo(self) -> None:
        if self._vol is None or not self._vol._undo:
            return
        self._vol.native = self._vol._undo.pop()
        self._vol.dirty = True
        self.refresh()
        self.mask_changed.emit()

    def _paint(self, which: str, x: float, y: float, erase: bool) -> None:
        vol = self._vol
        if vol is None:
            return
        self.active_view = which
        if vol.native is None:
            vol.native = np.zeros(vol.ct.shape, dtype=np.uint16)
        if not self._stroke:
            self._push_undo()
            self._stroke = True
            self._stroke_erase = bool(erase)
        label = 0 if erase else max(int(self.current_label), 1)
        paint_disk(
            vol.native,
            which,
            self._i,
            self._j,
            self._k,
            x,
            y,
            self.brush_radius,
            label,
        )
        vol.dirty = True
        self.refresh()

    def _end_stroke(self) -> None:
        vol = self._vol
        if vol is not None and self._stroke and not self._stroke_erase and vol._undo:
            promote_new_islands(vol.native, vol._undo[-1])
            self.refresh()
        self._stroke = False
        self.mask_changed.emit()

    def apply_mask(self, mask: np.ndarray, *, undo: bool = True) -> None:
        if self._vol is None:
            return
        if undo:
            self._push_undo()
        self._vol.native = np.asarray(mask, dtype=np.uint16)
        self._vol.dirty = True
        self.refresh()
        self.mask_changed.emit()

    def ijk(self) -> tuple[int, int, int]:
        return self._i, self._j, self._k

    def _apply_zoom(self, percent: int) -> None:
        for v in (self.axial, self.coronal, self.sagittal):
            v.set_zoom(int(percent))

    def refresh(self) -> None:
        vol = self._vol
        if vol is None:
            return
        args = dict(
            mode=self.display_mode(),
            pet_alpha=float(self.spin_alpha.value()),
            suv_min=float(self.spin_suv_min.value()),
            suv_max=float(self.spin_suv_max.value()),
            ct_window=ct_window_from_wl(float(self.spin_wl.value()), float(self.spin_ww.value())),
            show_native=self.chk_native.isChecked(),
            show_mapped=self.chk_mapped.isChecked(),
            highlight_label=int(self.highlight_label),
        )
        native = vol.native
        mapped = vol.mapped
        self.axial.set_rgb(
            compose_rgb(
                slice_axial(vol.ct, self._k),
                slice_axial(vol.pet, self._k),
                slice_axial(native, self._k) if native is not None else None,
                slice_axial(mapped, self._k) if mapped is not None else None,
                **args,
            )
        )
        self.coronal.set_rgb(
            compose_rgb(
                slice_coronal(vol.ct, self._j),
                slice_coronal(vol.pet, self._j),
                slice_coronal(native, self._j) if native is not None else None,
                slice_coronal(mapped, self._j) if mapped is not None else None,
                **args,
            )
        )
        self.sagittal.set_rgb(
            compose_rgb(
                slice_sagittal(vol.ct, self._i),
                slice_sagittal(vol.pet, self._i),
                slice_sagittal(native, self._i) if native is not None else None,
                slice_sagittal(mapped, self._i) if mapped is not None else None,
                **args,
            )
        )
        suv = float(vol.pet[self._i, self._j, self._k])
        hu = float(vol.ct[self._i, self._j, self._k])
        lid = int(native[self._i, self._j, self._k]) if native is not None else 0
        extra = f"  灶#{lid}" if lid else ""
        self.lbl_pos.setText(
            f"{vol.study_date}  ijk=({self._i},{self._j},{self._k})  "
            f"HU={hu:.0f}  SUV={suv:.2f}{extra}"
        )
        self._update_crosshairs()

    def _update_crosshairs(self) -> None:
        vol = self._vol
        views = (self.axial, self.coronal, self.sagittal)
        on = self.chk_crosshair.isChecked()
        for v in views:
            v.set_crosshair_visible(on)
        if vol is None or not on:
            return
        shape = vol.ct.shape
        for v in views:
            col, row = voxel_to_display(v.view_name, self._i, self._j, self._k, shape)
            v.set_crosshair(col, row)
