#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""分割编辑弹窗：CT / PET / 融合三格；轴/冠/矢由用户点选。"""

from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from display_utils import (
    DEFAULT_PET_CMAP,
    DEFAULT_WL,
    DEFAULT_WW,
    PET_CMAP_CHOICES,
    ct_window_from_wl,
    voxel_to_display,
)
from mask_ops import (
    lesion_stats,
    morph_labels,
    next_label,
    paint_disk,
    promote_new_islands,
    relabel_by_volume,
    threshold_pet_mask,
)
from ortho_viewer import _RgbView, configure_render_combo
from render_backend import GpuVolumeCache, compose_plane_cpu, gpu_available
from volume_io import VolumeSet

_LOG = logging.getLogger(__name__)
_MODES = ("ct", "pet", "fusion")
_MODE_LABEL = {"ct": "CT", "pet": "PET", "fusion": "融合"}
_PLANE_LABEL = {"axial": "轴位", "coronal": "冠状", "sagittal": "矢状"}


class SegmentEditorDialog(QDialog):
    def __init__(
        self,
        vol: VolumeSet,
        parent=None,
        *,
        pet_cmap: str = DEFAULT_PET_CMAP,
        render_device: str = "cpu",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"分割编辑  {vol.patient_id}  {vol.study_date}")
        self.setModal(True)
        self._vol = VolumeSet(
            ct=vol.ct,
            pet=vol.pet,
            native=np.array(vol.native, copy=True, dtype=np.uint16),
            mapped=vol.mapped,
            study_date=vol.study_date,
            patient_id=vol.patient_id,
            affine=vol.affine,
            role=vol.role,
        )
        nx, ny, nz = vol.ct.shape
        self._i, self._j, self._k = nx // 2, ny // 2, nz // 2
        self.current_label = 1
        self.highlight_label = 0
        self.brush_radius = 5
        self.plane = "axial"
        self.active_view = "axial"
        self._stroke = False
        self._stroke_erase = False
        self._gpu = GpuVolumeCache()
        self._masks_dirty = True
        self._gpu_failed = False
        pet = np.asarray(vol.pet, dtype=np.float32)
        self._pet_peak = float(np.nanmax(pet)) if pet.size else 0.0

        views_row = QHBoxLayout()
        self._cells: dict[str, _RgbView] = {}
        for mode in _MODES:
            view = _RgbView(_MODE_LABEL[mode], "axial")
            view.setMinimumHeight(160)
            view.item.edit_mode = True
            view.item.clicked_xy.connect(lambda x, y: self._click(x, y))
            view.item.paint_at.connect(lambda x, y, er: self._paint(x, y, er))
            view.item.stroke_finished.connect(self._end_stroke)
            self._cells[mode] = view
            views_row.addWidget(view, 1)

        self.radio_ax = QRadioButton("轴位")
        self.radio_co = QRadioButton("冠状")
        self.radio_sa = QRadioButton("矢状")
        self.radio_ax.setChecked(True)
        self._plane_group = QButtonGroup(self)
        for r, plane in ((self.radio_ax, "axial"), (self.radio_co, "coronal"), (self.radio_sa, "sagittal")):
            self._plane_group.addButton(r)
            r.toggled.connect(lambda on, p=plane: on and self._set_plane(p))

        self.slider_k = QSlider(Qt.Orientation.Horizontal)
        self.slider_j = QSlider(Qt.Orientation.Horizontal)
        self.slider_i = QSlider(Qt.Orientation.Horizontal)
        self.slider_i.setRange(0, max(nx - 1, 0))
        self.slider_j.setRange(0, max(ny - 1, 0))
        self.slider_k.setRange(0, max(nz - 1, 0))
        self.slider_i.setValue(self._i)
        self.slider_j.setValue(self._j)
        self.slider_k.setValue(self._k)
        for sl in (self.slider_k, self.slider_j, self.slider_i):
            sl.valueChanged.connect(self._sliders_changed)

        self.chk_crosshair = QCheckBox("十字线")
        self.chk_crosshair.setChecked(True)
        self.chk_crosshair.toggled.connect(lambda _: self.refresh(False))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(3, 15)
        self.spin_brush.setValue(5)
        self.spin_brush.valueChanged.connect(lambda v: setattr(self, "brush_radius", int(v)))
        self.spin_wl = QSpinBox()
        self.spin_wl.setRange(-1000, 3000)
        self.spin_wl.setValue(int(DEFAULT_WL))
        self.spin_ww = QSpinBox()
        self.spin_ww.setRange(1, 4000)
        self.spin_ww.setValue(int(DEFAULT_WW))
        self.spin_suv_min = QDoubleSpinBox()
        self.spin_suv_min.setRange(0.0, 20.0)
        self.spin_suv_min.setValue(0.0)
        self.spin_suv_max = QDoubleSpinBox()
        self.spin_suv_max.setRange(0.5, 40.0)
        self.spin_suv_max.setValue(6.0)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setValue(0.55)
        self.combo_cmap = QComboBox()
        for label, key in PET_CMAP_CHOICES:
            self.combo_cmap.addItem(label, key)
        idx = self.combo_cmap.findData(pet_cmap)
        self.combo_cmap.setCurrentIndex(idx if idx >= 0 else 1)
        for w in (self.spin_wl, self.spin_ww, self.spin_suv_min, self.spin_suv_max, self.spin_alpha):
            w.valueChanged.connect(lambda _: self.refresh(False))
        self.combo_cmap.currentIndexChanged.connect(lambda _: self.refresh(False))

        self.combo_render = QComboBox()
        self.lbl_cuda = configure_render_combo(self.combo_render, render_device)
        self.combo_render.currentIndexChanged.connect(self._on_render_device)

        self.lbl_pos = QLabel("—")

        plane_row = QHBoxLayout()
        plane_row.addWidget(QLabel("平面"))
        plane_row.addWidget(self.radio_ax)
        plane_row.addWidget(self.radio_co)
        plane_row.addWidget(self.radio_sa)
        plane_row.addWidget(QLabel("轴位"))
        plane_row.addWidget(self.slider_k, 1)
        plane_row.addWidget(QLabel("冠状"))
        plane_row.addWidget(self.slider_j, 1)
        plane_row.addWidget(QLabel("矢状"))
        plane_row.addWidget(self.slider_i, 1)

        ctrl = QHBoxLayout()
        ctrl.addWidget(self.chk_crosshair)
        ctrl.addWidget(QLabel("笔刷"))
        ctrl.addWidget(self.spin_brush)
        ctrl.addWidget(QLabel("窗位"))
        ctrl.addWidget(self.spin_wl)
        ctrl.addWidget(QLabel("窗宽"))
        ctrl.addWidget(self.spin_ww)
        ctrl.addWidget(QLabel("SUV"))
        ctrl.addWidget(self.spin_suv_min)
        ctrl.addWidget(self.spin_suv_max)
        ctrl.addWidget(QLabel("融合"))
        ctrl.addWidget(self.spin_alpha)
        ctrl.addWidget(QLabel("配色"))
        ctrl.addWidget(self.combo_cmap)
        ctrl.addWidget(QLabel("渲染"))
        ctrl.addWidget(self.combo_render)
        ctrl.addWidget(self.lbl_cuda)

        self.radio_thr_rel = QRadioButton("41% SUVmax")
        self.radio_thr_abs = QRadioButton("固定 SUV")
        self.radio_thr_rel.setChecked(True)
        self._thr_group = QButtonGroup(self)
        self._thr_group.addButton(self.radio_thr_rel)
        self._thr_group.addButton(self.radio_thr_abs)
        self.slider_thr = QSlider(Qt.Orientation.Horizontal)
        self.slider_thr.setRange(5, 150)
        self.slider_thr.setValue(25)
        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.5, 15.0)
        self.spin_thr.setSingleStep(0.1)
        self.spin_thr.setDecimals(1)
        self.spin_thr.setValue(2.5)
        self.lbl_thr = QLabel("—")
        self.radio_thr_rel.toggled.connect(lambda on: on and self._on_thr_mode())
        self.radio_thr_abs.toggled.connect(lambda on: on and self._on_thr_mode())
        self.slider_thr.valueChanged.connect(self._thr_slider_moved)
        self.slider_thr.sliderReleased.connect(self._apply_threshold_abs)
        self.spin_thr.valueChanged.connect(self._thr_spin_moved)
        self.spin_thr.editingFinished.connect(self._apply_threshold_abs)
        self._sync_thr_ui()

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("SUV 阈值"))
        thr_row.addWidget(self.radio_thr_rel)
        thr_row.addWidget(self.radio_thr_abs)
        thr_row.addWidget(self.slider_thr, 1)
        thr_row.addWidget(self.spin_thr)
        thr_row.addWidget(self.lbl_thr, 1)

        morph = QHBoxLayout()
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(1, 5)
        self.spin_radius.setValue(1)
        self.combo_scope = QComboBox()
        self.combo_scope.addItem("当前层 2D", "slice")
        self.combo_scope.addItem("三维", "3d")
        self.combo_target = QComboBox()
        self.combo_target.addItem("当前病灶", "current")
        self.combo_target.addItem("全部病灶", "all")
        for text, op in (("膨胀", "dilate"), ("腐蚀", "erode"), ("开", "open"), ("闭", "close")):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, o=op: self._morph(o))
            morph.addWidget(btn)
        morph.addWidget(QLabel("半径"))
        morph.addWidget(self.spin_radius)
        morph.addWidget(self.combo_scope)
        morph.addWidget(self.combo_target)
        btn_undo = QPushButton("撤销")
        btn_undo.clicked.connect(self.undo)
        btn_relabel = QPushButton("按体积重新编号")
        btn_relabel.clicked.connect(self._relabel)
        morph.addWidget(btn_undo)
        morph.addWidget(btn_relabel)
        morph.addWidget(self.lbl_pos, 1)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["编号", "体素", "体积 mL", "SUVmax"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(140)
        self.table.itemSelectionChanged.connect(self._on_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存并关闭")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        hint = QLabel(
            "点选轴位/冠状/矢状后，三格显示该平面的 CT、PET、融合。"
            "左键涂抹、右键擦除。切换阈值模式或松开滑杆会重算 mask。"
        )
        hint.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(views_row, 1)
        layout.addLayout(plane_row)
        layout.addLayout(ctrl)
        layout.addLayout(thr_row)
        layout.addLayout(morph)
        layout.addWidget(self.table)
        layout.addWidget(hint)
        layout.addWidget(buttons)

        self._fit_to_screen()
        self._bind_gpu()
        self.refresh(update_table=True)

    def _fit_to_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 800)
            return
        geo = screen.availableGeometry()
        w = min(geo.width(), max(int(geo.width() * 0.94), 1100))
        h = min(geo.height(), max(int(geo.height() * 0.92), 680))
        self.resize(w, h)
        self.move(geo.x() + (geo.width() - w) // 2, geo.y() + (geo.height() - h) // 2)

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        for view in self._cells.values():
            view.set_zoom(100)

    def edited_mask(self) -> np.ndarray:
        return self._vol.native

    def _pet_cmap(self) -> str:
        data = self.combo_cmap.currentData()
        return str(data) if data else DEFAULT_PET_CMAP

    def render_device(self) -> str:
        if self._gpu_failed or not gpu_available():
            return "cpu"
        data = self.combo_render.currentData()
        return str(data) if data else "cpu"

    def _mark_masks_dirty(self) -> None:
        self._masks_dirty = True

    def _on_render_device(self, _index: int = 0) -> None:
        self._gpu_failed = False
        self._gpu.clear()
        self._masks_dirty = True
        self._bind_gpu()
        self.refresh(update_table=False)

    def _bind_gpu(self) -> None:
        if self.render_device() != "gpu":
            return
        if self._gpu.bind_volumes(self._vol):
            self._gpu.sync_masks(self._vol)
            self._masks_dirty = False

    def _sync_thr_ui(self) -> None:
        rel = self.radio_thr_rel.isChecked()
        self.slider_thr.setEnabled(not rel)
        self.spin_thr.setEnabled(not rel)
        peak = self._pet_peak
        if rel:
            self.lbl_thr.setText(f"0.41 × SUVmax {peak:.2f} = {0.41 * peak:.2f}")
        else:
            self.lbl_thr.setText(f"SUV ≥ {float(self.spin_thr.value()):.2f}")

    def _on_thr_mode(self) -> None:
        self._sync_thr_ui()
        self._apply_threshold()

    def _thr_slider_moved(self, raw: int) -> None:
        val = max(raw, 5) / 10.0
        self.spin_thr.blockSignals(True)
        self.spin_thr.setValue(val)
        self.spin_thr.blockSignals(False)
        self._sync_thr_ui()

    def _thr_spin_moved(self, val: float) -> None:
        self.slider_thr.blockSignals(True)
        self.slider_thr.setValue(int(round(float(val) * 10)))
        self.slider_thr.blockSignals(False)
        self._sync_thr_ui()

    def _apply_threshold_abs(self) -> None:
        if self.radio_thr_rel.isChecked():
            return
        self._apply_threshold()

    def _apply_threshold(self) -> None:
        if self.radio_thr_abs.isChecked() and self.slider_thr.isSliderDown():
            return
        self._sync_thr_ui()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        self._push_undo()
        try:
            if self.radio_thr_rel.isChecked():
                mask = threshold_pet_mask(self._vol.pet, mode="relative", value=0.41)
            else:
                mask = threshold_pet_mask(
                    self._vol.pet, mode="absolute", value=float(self.spin_thr.value())
                )
            self._vol.native = mask
        finally:
            QApplication.restoreOverrideCursor()
        self._vol.dirty = True
        self.current_label = 1
        self.highlight_label = 0
        self._mark_masks_dirty()
        self.refresh(update_table=True)

    def _set_plane(self, plane: str) -> None:
        self.plane = plane
        self.active_view = plane
        for view in self._cells.values():
            view.view_name = plane
            view.set_laterality(plane)
        self.refresh(update_table=False)

    def _sliders_changed(self) -> None:
        self._i = self.slider_i.value()
        self._j = self.slider_j.value()
        self._k = self.slider_k.value()
        self.refresh(update_table=False)

    def _click(self, x: float, y: float) -> None:
        from display_utils import display_to_voxel

        ii, jj, kk = display_to_voxel(
            self.plane, x, y, self._i, self._j, self._k, self._vol.ct.shape
        )
        self.slider_i.setValue(ii)
        self.slider_j.setValue(jj)
        self.slider_k.setValue(kk)

    def _push_undo(self) -> None:
        self._vol._undo.append(self._vol.native.copy())
        if len(self._vol._undo) > 20:
            self._vol._undo.pop(0)

    def undo(self) -> None:
        if not self._vol._undo:
            return
        self._vol.native = self._vol._undo.pop()
        self._vol.dirty = True
        self._mark_masks_dirty()
        self.refresh(update_table=True)

    def _paint(self, x: float, y: float, erase: bool) -> None:
        if not self._stroke:
            self._push_undo()
            self._stroke = True
            self._stroke_erase = bool(erase)
        label = 0 if erase else max(int(self.current_label), 1)
        paint_disk(
            self._vol.native,
            self.plane,
            self._i,
            self._j,
            self._k,
            x,
            y,
            self.brush_radius,
            label,
        )
        self._vol.dirty = True
        self._mark_masks_dirty()
        self.refresh(update_table=False)

    def _end_stroke(self) -> None:
        if self._stroke and not self._stroke_erase and self._vol._undo:
            promote_new_islands(self._vol.native, self._vol._undo[-1])
            self._mark_masks_dirty()
        self._stroke = False
        self.refresh(update_table=True)

    def _morph(self, op: str) -> None:
        radius = int(self.spin_radius.value())
        scope = self.combo_scope.currentData()
        target = self.combo_target.currentData()
        label = int(self.current_label) if target == "current" else 0
        plane = self.plane if scope == "slice" else None
        ijk = (self._i, self._j, self._k) if plane else None
        waiting = plane is None
        if waiting:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
        self._push_undo()
        try:
            self._vol.native = morph_labels(
                self._vol.native, op, radius, label=label, plane=plane, ijk=ijk
            )
        finally:
            if waiting:
                QApplication.restoreOverrideCursor()
        self._vol.dirty = True
        self._mark_masks_dirty()
        self.refresh(update_table=True)

    def _relabel(self) -> None:
        self._push_undo()
        self._vol.native = relabel_by_volume(self._vol.native)
        self.current_label = 1
        self.highlight_label = 1
        self._vol.dirty = True
        self._mark_masks_dirty()
        self.refresh(update_table=True)

    def _on_row(self) -> None:
        items = self.table.selectedItems()
        if not items:
            self.highlight_label = 0
            self.current_label = next_label(self._vol.native)
        else:
            try:
                lid = int(items[0].text())
            except ValueError:
                lid = 0
            if lid > 0:
                self.current_label = lid
                self.highlight_label = lid
        self.refresh(update_table=False)

    def _voxel_ml(self) -> float:
        if self._vol.affine is None:
            return 0.008
        return abs(float(np.linalg.det(self._vol.affine[:3, :3]))) / 1000.0

    def _refresh_table(self) -> None:
        rows = lesion_stats(self._vol.native, self._vol.pet, self._voxel_ml())
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        selected_row = -1
        for r, row in enumerate(rows):
            vals = [row.get("id"), row.get("n_voxels"), row.get("volume_ml"), row.get("suv_max")]
            for c, val in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))
            if int(row.get("id") or 0) == self.current_label:
                selected_row = r
        if selected_row >= 0:
            self.table.selectRow(selected_row)
        self.table.blockSignals(False)

    def refresh(self, update_table: bool = True) -> None:
        vol = self._vol
        native = vol.native
        plane = self.plane
        args_base = dict(
            pet_alpha=float(self.spin_alpha.value()),
            suv_min=float(self.spin_suv_min.value()),
            suv_max=float(self.spin_suv_max.value()),
            ct_window=ct_window_from_wl(float(self.spin_wl.value()), float(self.spin_ww.value())),
            show_native=True,
            show_mapped=False,
            highlight_label=int(self.highlight_label),
            pet_cmap=self._pet_cmap(),
        )
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
        used_gpu = device == "gpu" and gpu_cache is not None
        on = self.chk_crosshair.isChecked()
        col, row = voxel_to_display(plane, self._i, self._j, self._k, vol.ct.shape)
        rgb_by_mode: dict[str, np.ndarray] | None = None
        if used_gpu:
            try:
                rgb_by_mode = {
                    mode: gpu_cache.compose_plane(
                        plane, self._i, self._j, self._k, mode=mode, **args_base
                    )
                    for mode in self._cells
                }
            except Exception:
                _LOG.exception("GPU 合成失败，回退 CPU")
                self._gpu_failed = True
                used_gpu = False
        for mode, view in self._cells.items():
            rgb = (
                rgb_by_mode[mode]
                if rgb_by_mode is not None
                else compose_plane_cpu(
                    vol, plane, self._i, self._j, self._k, mode=mode, **args_base
                )
            )
            view.set_rgb(rgb)
            view.set_crosshair_visible(on)
            if on:
                view.set_crosshair(col, row)
        suv = float(vol.pet[self._i, self._j, self._k])
        hu = float(vol.ct[self._i, self._j, self._k])
        lid = int(native[self._i, self._j, self._k])
        extra = f"  灶#{lid}" if lid else ""
        self.lbl_pos.setText(
            f"{_PLANE_LABEL[plane]}  ijk=({self._i},{self._j},{self._k})  "
            f"HU={hu:.0f}  SUV={suv:.2f}{extra}"
        )
        if update_table:
            self._refresh_table()
