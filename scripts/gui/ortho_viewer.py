#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""轴 / 冠 / 矢 联动三视图：显示模式、窗宽窗位、缩放、二维画笔。"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
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
    DEFAULT_PET_CMAP,
    DEFAULT_WL,
    DEFAULT_WW,
    PET_CMAP_CHOICES,
    ct_window_from_wl,
    voxel_to_display,
)
from mask_ops import paint_disk, promote_new_islands
from render_backend import GpuVolumeCache, compose_plane_cpu, gpu_available
from volume_io import VolumeSet

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
_LOG = logging.getLogger(__name__)

_LAT_STYLE = (
    "QLabel { color: #f2f2f2; background: rgba(10, 10, 10, 165); "
    "padding: 1px 6px; border-radius: 3px; font-weight: 600; font-size: 12px; }"
)


class _LateralityMixin:
    """视口左右标记：轴位/冠状/MIP 为 R·L，矢状为 P·A。贴在屏幕边，不随缩放走。"""

    def _init_laterality(self, view_name: str) -> None:
        self._lat_left = QLabel(self)
        self._lat_right = QLabel(self)
        for lbl in (self._lat_left, self._lat_right):
            lbl.setStyleSheet(_LAT_STYLE)
            lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.set_laterality(view_name)

    def set_laterality(self, view_name: str) -> None:
        left, right = ("P", "A") if view_name == "sagittal" else ("R", "L")
        self._lat_left.setText(left)
        self._lat_right.setText(right)
        self._lat_left.adjustSize()
        self._lat_right.adjustSize()
        self._place_laterality()

    def _place_laterality(self) -> None:
        if not hasattr(self, "_lat_left"):
            return
        margin = 8
        title_h = 22
        area_h = max(self.height() - title_h, 1)
        y = max(margin, area_h // 2 - self._lat_left.height() // 2)
        self._lat_left.move(margin, y)
        self._lat_right.move(
            max(margin, self.width() - self._lat_right.width() - margin), y
        )
        self._lat_left.raise_()
        self._lat_right.raise_()


def configure_render_combo(combo: QComboBox, current: str = "cpu") -> QLabel:
    """CPU / GPU 下拉。无 CUDA 时禁用 GPU 并旁注。"""
    combo.addItem("CPU", "cpu")
    combo.addItem("GPU", "gpu")
    note = QLabel("")
    want = current if current in ("cpu", "gpu") else "cpu"
    if not gpu_available():
        idx = combo.findData("gpu")
        model = combo.model()
        getter = getattr(model, "item", None)
        item = getter(idx) if callable(getter) and idx >= 0 else None
        if item is not None:
            item.setEnabled(False)
        note.setText("无 CUDA")
        note.setStyleSheet("color: #888;")
        want = "cpu"
    idx = combo.findData(want)
    combo.setCurrentIndex(idx if idx >= 0 else 0)
    return note


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


class _RgbView(_LateralityMixin, pg.GraphicsLayoutWidget):
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
        self._init_laterality(view_name)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._place_laterality()

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
        shape_changed = cur is None or cur.shape != img.shape
        if cur is not None and cur.shape == img.shape and cur.dtype == img.dtype:
            np.copyto(cur, img)
            self.item.updateImage()
        else:
            self.item.setImage(img, autoLevels=False)
        if shape_changed:
            self.set_zoom(100)

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
        self._gpu = GpuVolumeCache()
        self._masks_dirty = True
        self._gpu_failed = False

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

        self.combo_cmap = QComboBox()
        for label, key in PET_CMAP_CHOICES:
            self.combo_cmap.addItem(label, key)
        idx = self.combo_cmap.findData(DEFAULT_PET_CMAP)
        self.combo_cmap.setCurrentIndex(idx if idx >= 0 else 1)
        self.combo_cmap.currentIndexChanged.connect(self.refresh)

        self.combo_render = QComboBox()
        self.lbl_cuda = configure_render_combo(self.combo_render)
        self.combo_render.currentIndexChanged.connect(self._on_render_device)

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
            QLabel("配色"),
            self.combo_cmap,
            QLabel("融合"),
            self.spin_alpha,
            QLabel("缩放"),
            self.spin_zoom,
            self.btn_zoom_reset,
            QLabel("渲染"),
            self.combo_render,
            self.lbl_cuda,
            self.lbl_pos,
        ]
        for col, w in enumerate(r1):
            tools.addWidget(w, 0, col)
        for col, w in enumerate(r2):
            tools.addWidget(w, 1, col)
        tools.setColumnStretch(max(len(r1), len(r2)) - 1, 1)

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

    def pet_cmap(self) -> str:
        data = self.combo_cmap.currentData()
        return str(data) if data else DEFAULT_PET_CMAP

    def render_device(self) -> str:
        if self._gpu_failed or not gpu_available():
            return "cpu"
        data = self.combo_render.currentData()
        return str(data) if data else "cpu"

    def mark_masks_dirty(self) -> None:
        self._masks_dirty = True

    def _on_render_device(self, _index: int = 0) -> None:
        self._gpu_failed = False
        self._gpu.clear()
        self._masks_dirty = True
        self._bind_gpu()
        self.refresh()

    def _bind_gpu(self) -> None:
        vol = self._vol
        if vol is None or self.render_device() != "gpu":
            return
        if self._gpu.bind_volumes(vol):
            self._gpu.sync_masks(vol)
            self._masks_dirty = False

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
        self._gpu.clear()
        self._gpu_failed = False
        self._masks_dirty = True
        if vol is None:
            self.lbl_pos.setText("无图像")
            for view in (self.axial, self.coronal, self.sagittal):
                view.item.clear()
                view.set_crosshair_visible(False)
            return
        self._bind_gpu()
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
        self.mark_masks_dirty()
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
        self.mark_masks_dirty()
        self.refresh()

    def _end_stroke(self) -> None:
        vol = self._vol
        if vol is not None and self._stroke and not self._stroke_erase and vol._undo:
            promote_new_islands(vol.native, vol._undo[-1])
            self.mark_masks_dirty()
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
        self.mark_masks_dirty()
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
            pet_cmap=self.pet_cmap(),
        )
        native = vol.native
        device = self.render_device()
        gpu_cache = None
        if device == "gpu":
            try:
                if not self._gpu.bind_volumes(vol):
                    raise RuntimeError("GPU 不可用")
                if self._masks_dirty:
                    self._gpu.sync_masks(vol)
                    self._masks_dirty = False
                gpu_cache = self._gpu
            except Exception:
                _LOG.exception("GPU 体积上传失败，回退 CPU")
                self._gpu_failed = True
                device = "cpu"
        planes = (
            (self.axial, "axial"),
            (self.coronal, "coronal"),
            (self.sagittal, "sagittal"),
        )
        used_gpu = device == "gpu" and gpu_cache is not None
        if used_gpu:
            try:
                for view, plane in planes:
                    view.set_rgb(
                        gpu_cache.compose_plane(plane, self._i, self._j, self._k, **args)
                    )
            except Exception:
                _LOG.exception("GPU 合成失败，回退 CPU")
                self._gpu_failed = True
                used_gpu = False
        if not used_gpu:
            for view, plane in planes:
                view.set_rgb(
                    compose_plane_cpu(vol, plane, self._i, self._j, self._k, **args)
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
