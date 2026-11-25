#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""轴 / 冠 / 矢 联动三视图：显示模式、窗宽窗位、缩放、二维画笔。"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
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
from roi_draw import (
    apply_closed_paths,
    contours_from_label,
    dominant_label,
    label_on_slice,
    nearest_anchor,
    replace_slice_with_paths,
)
from render_backend import GpuVolumeCache, compose_plane_cpu, gpu_available
from volume_io import VolumeSet

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
_LOG = logging.getLogger(__name__)

_LAT_STYLE = (
    "QLabel { color: #f2f2f2; background: rgba(10, 10, 10, 165); "
    "padding: 1px 6px; border-radius: 3px; font-weight: 600; font-size: 12px; }"
)
_ZOOM_STYLE = (
    "QLabel { color: #f2f2f2; background: rgba(10, 10, 10, 165); "
    "padding: 2px 8px; border-radius: 3px; font-weight: 600; font-size: 12px; }"
)
ZOOM_MIN = 50
ZOOM_MAX = 400
DEFAULT_WHEEL_DPI = 10
WHEEL_DPI_MIN = 10
WHEEL_DPI_MAX = 400


class _ZoomBadge(QLabel):
    """视口右上角的缩放值。按住左键向右放大，向左缩小。"""

    zoom_changed = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__("100%", parent)
        self._percent = 100
        self._origin_x = 0.0
        self._origin_percent = 100
        self._dragging = False
        self.setStyleSheet(_ZOOM_STYLE)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip("按住左键向右放大，向左缩小")
        self.adjustSize()

    def percent(self) -> int:
        return self._percent

    def set_percent(self, percent: int) -> None:
        percent = int(np.clip(percent, ZOOM_MIN, ZOOM_MAX))
        if percent == self._percent and self.text() == f"{percent}%":
            return
        self._percent = percent
        self.setText(f"{percent}%")
        self.adjustSize()

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(ev)
            return
        self._dragging = True
        self._origin_x = float(ev.globalPosition().x())
        self._origin_percent = self._percent
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        if not self._dragging:
            super().mouseMoveEvent(ev)
            return
        dx = float(ev.globalPosition().x()) - self._origin_x
        percent = int(np.clip(round(self._origin_percent + dx / 2.0), ZOOM_MIN, ZOOM_MAX))
        if percent != self._percent:
            self.set_percent(percent)
            self.zoom_changed.emit(percent)
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        self._dragging = False
        ev.accept()


_PLANE_ORDER = ("axial", "coronal", "sagittal")
_PLANE_LABELS = {"axial": "轴位", "coronal": "冠状位", "sagittal": "矢状位"}


class _PlaneTitle(QLabel):
    """断面名称。单击切换下一个，按住左键左右拖动可来回选择。"""

    plane_changed = Signal(str)

    def __init__(self, parent: QWidget, plane: str) -> None:
        super().__init__(_PLANE_LABELS[plane], parent)
        self._plane = plane
        self._dragging = False
        self._moved = False
        self._origin_x = 0.0
        self._origin_index = 0
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip("单击切换断面。按住左键向右、向左拖动，可在轴位、冠状位、矢状位之间调整。")
        self.setStyleSheet(
            "QLabel { color: #f2f2f2; background: transparent; font-size: 13px; padding: 0 6px; }"
        )
        self.adjustSize()

    def set_plane(self, plane: str) -> None:
        self._plane = plane
        self.setText(_PLANE_LABELS[plane])
        self.adjustSize()

    def _emit_index(self, index: int) -> None:
        plane = _PLANE_ORDER[int(index) % len(_PLANE_ORDER)]
        if plane == self._plane:
            return
        self.set_plane(plane)
        self.plane_changed.emit(plane)

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(ev)
            return
        self._dragging = True
        self._moved = False
        self._origin_x = float(ev.globalPosition().x())
        self._origin_index = _PLANE_ORDER.index(self._plane)
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        if not self._dragging:
            super().mouseMoveEvent(ev)
            return
        dx = float(ev.globalPosition().x()) - self._origin_x
        if abs(dx) >= 8:
            self._moved = True
        self._emit_index(self._origin_index + int(np.trunc(dx / 48.0)))
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        if self._dragging and not self._moved:
            self._emit_index(self._origin_index + 1)
        self._dragging = False
        ev.accept()


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
        top = 8
        floor_left = self.height()
        floor_right = self.height()
        if hasattr(self, "_id"):
            top = self._id.geometry().bottom() + 6
            self._id.raise_()
        if hasattr(self, "_window"):
            floor_left = self._window.y() - 4
            self._window.raise_()
        if hasattr(self, "_pet"):
            floor_right = self._pet.y() - 4
            self._pet.raise_()
        if self._lat_left.y() < top:
            self._lat_left.move(self._lat_left.x(), top)
        if self._lat_left.geometry().bottom() > floor_left:
            self._lat_left.move(self._lat_left.x(), max(top, floor_left - self._lat_left.height()))
        if self._lat_right.geometry().bottom() > floor_right:
            self._lat_right.move(self._lat_right.x(), max(top, floor_right - self._lat_right.height()))
        if hasattr(self, "_zoom"):
            self._zoom.raise_()
        if getattr(self, "_title", None) is not None:
            self._place_title()


_HUD_STYLE = """
QFrame#viewHud { background: transparent; border: none; }
QFrame#viewHud QLabel { background: transparent; font-size: 11px; font-weight: 600; }
QFrame#viewHud QSpinBox, QFrame#viewHud QDoubleSpinBox, QFrame#viewHud QComboBox {
    background: transparent; border: none; font-size: 11px; font-weight: 600;
    padding: 0; min-height: 14px; margin: 0;
}
QFrame#viewHud QSpinBox::up-button, QFrame#viewHud QSpinBox::down-button,
QFrame#viewHud QDoubleSpinBox::up-button, QFrame#viewHud QDoubleSpinBox::down-button {
    width: 0; height: 0; border: none;
}
QFrame#viewHud QComboBox::drop-down { border: none; width: 10px; background: transparent; }
QFrame#viewHud QComboBox QAbstractItemView {
    background-color: #141414; color: #f4f4f4;
    selection-background-color: #2e2e2e; border: 1px solid #444;
}
"""
_HUD_COLORS = {
    "wl": "#5ec8ff",
    "ww": "#ffb020",
    "suv": "#7ddea0",
    "cmap": "#d7a6ff",
    "ratio": "#ff8d7a",
}


class _HorizontalDrag(QObject):
    """按住左键左右拖动，改变绑定数值。向右增大，向左减小。"""

    def __init__(self, spin: QSpinBox, gain: float) -> None:
        super().__init__(spin)
        self._spin = spin
        self._gain = gain
        self._active = False
        self._x = 0.0
        self._origin = 0.0
        self._watched: set[QWidget] = set()

    def watch(self, widget: QWidget | None) -> None:
        if widget is None or widget in self._watched:
            return
        self._watched.add(widget)
        widget.setCursor(Qt.CursorShape.SizeHorCursor)
        widget.setToolTip("按住左键向右增大，向左减小")
        widget.installEventFilter(self)
        line = getattr(widget, "lineEdit", None)
        editor = line() if callable(line) else None
        if editor is not None:
            self.watch(editor)

    def eventFilter(self, obj, ev) -> bool:
        etype = ev.type()
        if etype == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
            self._active = True
            self._x = float(ev.globalPosition().x())
            self._origin = float(self._spin.value())
            return True
        if etype == QEvent.Type.MouseMove and self._active:
            dx = float(ev.globalPosition().x()) - self._x
            self._spin.setValue(int(round(self._origin + dx * self._gain)))
            return True
        if etype == QEvent.Type.MouseButtonRelease and self._active:
            self._active = False
            return True
        return False


def _shadow(widget: QWidget) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(6)
    effect.setOffset(0, 0)
    effect.setColor(QColor(0, 0, 0, 230))
    widget.setGraphicsEffect(effect)


def _prepare_overlay(frame: QFrame) -> None:
    frame.setObjectName("viewHud")
    frame.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    frame.setAutoFillBackground(False)
    frame.setStyleSheet(_HUD_STYLE)
    _shadow(frame)


class _IdHud(QFrame):
    """左上角：患者号和检查日期各占一行。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        _prepare_overlay(self)
        self.lbl_id = QLabel("—")
        self.lbl_date = QLabel("")
        header = "font-weight: 600; font-size: 13px; color: #ffffff; background: transparent;"
        self.lbl_id.setStyleSheet(header)
        self.lbl_date.setStyleSheet(header)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(0)
        layout.addWidget(self.lbl_id)
        layout.addWidget(self.lbl_date)
        self.adjustSize()

    def set_header(self, patient_id: str, study_date: str) -> None:
        self.lbl_id.setText(patient_id or "—")
        self.lbl_date.setText(study_date or "")
        self.adjustSize()


class _WindowHud(QFrame):
    """左下角：WL、WW。按住左键左右拖动数值。"""

    edited = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        _prepare_overlay(self)
        self.spin_wl = QSpinBox()
        self.spin_wl.setRange(-1000, 3000)
        self.spin_wl.setValue(int(DEFAULT_WL))
        self.spin_ww = QSpinBox()
        self.spin_ww.setRange(1, 4000)
        self.spin_ww.setValue(int(DEFAULT_WW))
        for box in (self.spin_wl, self.spin_ww):
            box.setFixedWidth(46)
            box.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            box.valueChanged.connect(lambda *_: self.edited.emit())
        wl_tag = _tag("WL", "wl")
        ww_tag = _tag("WW", "ww")
        _color_value(self.spin_wl, "wl")
        _color_value(self.spin_ww, "ww")
        self._drag_wl = _install_drag(self.spin_wl, 3.0, wl_tag)
        self._drag_ww = _install_drag(self.spin_ww, 6.0, ww_tag)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(0)
        grid.addWidget(wl_tag, 0, 0)
        grid.addWidget(self.spin_wl, 0, 1)
        grid.addWidget(ww_tag, 1, 0)
        grid.addWidget(self.spin_ww, 1, 1)
        self.adjustSize()

    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        self._drag_wl.watch(self.spin_wl)
        self._drag_ww.watch(self.spin_ww)

    def wheelEvent(self, ev) -> None:
        ev.accept()

    def write_values(self, wl: int, ww: int) -> None:
        for widget in (self.spin_wl, self.spin_ww):
            widget.blockSignals(True)
        self.spin_wl.setValue(int(wl))
        self.spin_ww.setValue(int(ww))
        for widget in (self.spin_wl, self.spin_ww):
            widget.blockSignals(False)


class _PetHud(QFrame):
    """右下角：color map、SUV 范围、ratio。"""

    edited = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        _prepare_overlay(self)
        self.lbl_unit = QLabel("SUV")
        self.spin_suv_min = QDoubleSpinBox()
        self.spin_suv_min.setRange(0.0, 20.0)
        self.spin_suv_min.setDecimals(1)
        self.spin_suv_min.setSingleStep(0.5)
        self.spin_suv_min.setValue(0.0)
        self.spin_suv_max = QDoubleSpinBox()
        self.spin_suv_max.setRange(0.5, 40.0)
        self.spin_suv_max.setDecimals(1)
        self.spin_suv_max.setSingleStep(0.5)
        self.spin_suv_max.setValue(6.0)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setDecimals(2)
        self.spin_alpha.setValue(0.55)
        self.combo = QComboBox()
        for label, key in PET_CMAP_CHOICES:
            self.combo.addItem(label, key)
        for box in (self.spin_suv_min, self.spin_suv_max, self.spin_alpha):
            box.setFixedWidth(40)
            box.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            box.valueChanged.connect(lambda *_: self.edited.emit())
        self.combo.setFixedWidth(72)
        self.combo.currentIndexChanged.connect(lambda *_: self.edited.emit())
        _color_value(self.lbl_unit, "suv")
        _color_value(self.spin_suv_min, "suv")
        _color_value(self.spin_suv_max, "suv")
        _color_value(self.spin_alpha, "ratio")
        _color_value(self.combo, "cmap")
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(0)
        grid.addWidget(_tag("color map", "cmap"), 0, 0)
        grid.addWidget(self.combo, 0, 1, 1, 3)
        grid.addWidget(self.lbl_unit, 1, 0)
        grid.addWidget(self.spin_suv_min, 1, 1)
        grid.addWidget(_tag("–", "suv"), 1, 2)
        grid.addWidget(self.spin_suv_max, 1, 3)
        grid.addWidget(_tag("ratio", "ratio"), 2, 0)
        grid.addWidget(self.spin_alpha, 2, 1)
        self.adjustSize()

    def wheelEvent(self, ev) -> None:
        ev.accept()

    def set_unit(self, pet_unit: str) -> None:
        self.lbl_unit.setText(pet_unit or "SUV")
        self.adjustSize()

    def write_values(self, suv_min: float, suv_max: float, cmap_index: int, alpha: float) -> None:
        widgets = (self.spin_suv_min, self.spin_suv_max, self.spin_alpha, self.combo)
        for widget in widgets:
            widget.blockSignals(True)
        self.spin_suv_min.setValue(float(suv_min))
        self.spin_suv_max.setValue(float(suv_max))
        self.spin_alpha.setValue(float(alpha))
        if 0 <= cmap_index < self.combo.count():
            self.combo.setCurrentIndex(int(cmap_index))
        for widget in widgets:
            widget.blockSignals(False)


def _tag(text: str, key: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(
        f"color: {_HUD_COLORS[key]}; background: transparent; font-size: 11px; font-weight: 600;"
    )
    return label


def _color_value(widget: QWidget, key: str) -> None:
    color = _HUD_COLORS[key]
    if isinstance(widget, QComboBox):
        widget.setStyleSheet(
            "QComboBox { color: %s; background: transparent; border: none; "
            "font-size: 11px; font-weight: 600; padding: 0; }"
            "QComboBox::drop-down { border: none; width: 10px; background: transparent; }"
            "QComboBox QAbstractItemView { background: #141414; color: #f4f4f4; "
            "selection-background-color: #2e2e2e; }" % color
        )
        return
    widget.setStyleSheet(
        f"color: {color}; background: transparent; border: none; font-size: 11px; font-weight: 600; padding: 0;"
    )


def _install_drag(spin: QSpinBox, gain: float, *companions: QWidget) -> _HorizontalDrag:
    drag = _HorizontalDrag(spin, gain)
    drag.watch(spin)
    for widget in companions:
        drag.watch(widget)
    return drag


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
    gesture = Signal(float, float, str, bool)

    def __init__(self) -> None:
        super().__init__()
        self.edit_mode = False
        self.tool = "brush"

    @staticmethod
    def _subtracting(ev) -> bool:
        mods = ev.modifiers()
        return bool(
            mods & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.AltModifier)
        )

    def mouseClickEvent(self, ev) -> None:
        pos = ev.pos()
        if ev.button() == Qt.MouseButton.RightButton and not (
            self.edit_mode and self.tool == "brush"
        ):
            self.gesture.emit(float(pos.x()), float(pos.y()), "undo", False)
            ev.accept()
            return
        if self.tool in ("pen", "polygon"):
            if ev.button() == Qt.MouseButton.LeftButton:
                self.gesture.emit(float(pos.x()), float(pos.y()), "click", self._subtracting(ev))
            ev.accept()
            return
        if self.edit_mode:
            erase = ev.button() == Qt.MouseButton.RightButton
            self.paint_at.emit(float(pos.x()), float(pos.y()), erase)
            self.stroke_finished.emit()
            ev.accept()
            return
        self.clicked_xy.emit(float(pos.x()), float(pos.y()))
        ev.accept()

    def mouseDragEvent(self, ev) -> None:
        if self.tool in ("pen", "polygon") and ev.button() == Qt.MouseButton.LeftButton:
            if ev.isStart():
                phase = "start"
            elif ev.isFinish():
                phase = "finish"
            else:
                phase = "move"
            pos = ev.pos()
            self.gesture.emit(float(pos.x()), float(pos.y()), phase, self._subtracting(ev))
            ev.accept()
            return
        if not self.edit_mode or self.tool != "brush":
            ev.ignore()
            return
        pos = ev.pos()
        erase = ev.button() == Qt.MouseButton.RightButton
        self.paint_at.emit(float(pos.x()), float(pos.y()), erase)
        if ev.isFinish():
            self.stroke_finished.emit()
        ev.accept()


class _RgbView(_LateralityMixin, pg.GraphicsLayoutWidget):
    """一个断面视图。滚轮改变该断面的层号，不缩放。"""

    slice_scroll = Signal(int)
    plane_changed = Signal(str)

    def __init__(self, title: str, view_name: str, *, plane_switch: bool = False) -> None:
        super().__init__()
        self.view_name = view_name
        self._plane_switch = plane_switch
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
        self.roi_line = pg.PlotDataItem(pen=pg.mkPen("#7dff6a", width=1))
        self.roi_anchors = pg.ScatterPlotItem(
            size=4,
            pxMode=True,
            symbol="s",
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen("#ffe14a", width=1),
        )
        self.roi_marks = pg.ScatterPlotItem(pxMode=True, size=7)
        self.roi_line.setZValue(30)
        self.roi_anchors.setZValue(31)
        self.roi_marks.setZValue(32)
        self.roi_line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.roi_anchors.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.roi_marks.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.box.addItem(self.roi_line)
        self.box.addItem(self.roi_anchors)
        self.box.addItem(self.roi_marks)
        self._title: Optional[_PlaneTitle] = None
        if plane_switch:
            self._title = _PlaneTitle(self, view_name)
            self._title.plane_changed.connect(self._on_title_plane)
        else:
            self.addItem(pg.LabelItem(title, color="#dddddd"), row=1, col=0)
        self._init_laterality(view_name)
        self._zoom_percent = 100
        self.wheel_dpi = DEFAULT_WHEEL_DPI
        self._wheel_acc = 0.0
        self._last_slice_scroll = 0.0
        self._zoom = _ZoomBadge(self)
        self._zoom.zoom_changed.connect(self.set_zoom)
        self._place_zoom()
        self._place_title()

    def wheelEvent(self, ev) -> None:
        delta = ev.angleDelta().y()
        if delta == 0:
            delta = ev.pixelDelta().y()
        if delta == 0:
            ev.ignore()
            return
        ev.accept()
        # 高分辨率滚轮一次轻拨会连发很多事件。这些事件合成一步，只翻一层。
        # DPI 决定下一层最早何时允许再翻：10 约 0.6 秒，40 约 0.15 秒。
        gap = max(0.12, 6.0 / max(int(self.wheel_dpi), 1))
        now = time.monotonic()
        if now - self._last_slice_scroll < gap:
            return
        self._last_slice_scroll = now
        self.slice_scroll.emit(1 if delta > 0 else -1)

    def set_wheel_dpi(self, dpi: int) -> None:
        self.wheel_dpi = int(np.clip(dpi, WHEEL_DPI_MIN, WHEEL_DPI_MAX))
        self._wheel_acc = 0.0

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._place_zoom()
        self._place_hud()
        self._place_laterality()
        self._place_title()

    def _on_title_plane(self, plane: str) -> None:
        self.view_name = plane
        self.set_laterality(plane)
        self.plane_changed.emit(plane)

    def _place_title(self) -> None:
        title = getattr(self, "_title", None)
        if title is None:
            return
        title.adjustSize()
        title.move(
            max(0, (self.width() - title.width()) // 2),
            max(0, self.height() - title.height() - 2),
        )
        title.raise_()

    def _place_hud(self) -> None:
        if not hasattr(self, "_id"):
            return
        margin = 8
        title_h = 24
        zoom_w = self._zoom.width() + margin * 2 if hasattr(self, "_zoom") else 72
        self._id.setMaximumWidth(max(80, self.width() - zoom_w - margin))
        self._id.adjustSize()
        self._id.move(margin, margin)
        self._window.adjustSize()
        self._pet.adjustSize()
        self._pet.setMaximumWidth(max(140, self.width() // 2))
        self._pet.adjustSize()
        y_win = self.height() - title_h - self._window.height()
        y_pet = self.height() - title_h - self._pet.height()
        self._window.move(margin, max(margin, y_win))
        self._pet.move(max(margin, self.width() - self._pet.width() - margin), max(margin, y_pet))
        self._id.raise_()
        self._window.raise_()
        self._pet.raise_()
        if hasattr(self, "_zoom"):
            self._zoom.raise_()

    def attach_hud(self) -> None:
        self._id = _IdHud(self)
        self._window = _WindowHud(self)
        self._pet = _PetHud(self)
        self._place_hud()

    def _place_zoom(self) -> None:
        if not hasattr(self, "_zoom"):
            return
        margin = 8
        self._zoom.move(max(margin, self.width() - self._zoom.width() - margin), margin)
        self._zoom.raise_()

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
            self.set_zoom(self._zoom_percent)

    def set_zoom(self, percent: int) -> None:
        percent = int(np.clip(percent, ZOOM_MIN, ZOOM_MAX))
        self._zoom_percent = percent
        if hasattr(self, "_zoom"):
            self._zoom.set_percent(percent)
            self._place_zoom()
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
    cursor_changed = Signal(int, int, int, bool)
    display_params_changed = Signal()
    tool_hint_changed = Signal(str)

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
        self._tool_hint_sent = ""
        self._pen_points: list[tuple[float, float]] = []
        self._pen_placing = False
        self._pen_view = "axial"
        self._pen_ijk = (0, 0, 0)
        self._pen_ignore_until = 0.0
        self._poly_paths: list[np.ndarray] = []
        self._poly_plane = "axial"
        self._poly_ijk = (0, 0, 0)
        self._poly_label = 1
        self._poly_drag: Optional[tuple[int, int]] = None
        self._poly_base: Optional[np.ndarray] = None
        self._poly_changed = False
        self._poly_undo = False
        self._poly_moved = False
        self._gpu = GpuVolumeCache()
        self._masks_dirty = True
        self._gpu_failed = False
        self._shown_ijk: Optional[tuple[int, int, int]] = None
        self._hud_lock = False

        self.axial = _RgbView("轴位", "axial", plane_switch=True)
        self.coronal = _RgbView("冠状位", "coronal", plane_switch=True)
        self.sagittal = _RgbView("矢状位", "sagittal", plane_switch=True)
        self._active_host: Optional[_RgbView] = self.axial
        self._pen_host: Optional[_RgbView] = None
        self._poly_host: Optional[_RgbView] = None
        for v in (self.axial, self.coronal, self.sagittal):
            v.item.clicked_xy.connect(lambda x, y, vv=v: self._click(vv, x, y))
            v.item.paint_at.connect(lambda x, y, er, vv=v: self._paint(vv, x, y, er))
            v.item.stroke_finished.connect(self._end_stroke)
            v.item.gesture.connect(
                lambda x, y, phase, subtract, vv=v: self._gesture(vv, x, y, phase, subtract)
            )
            v.slice_scroll.connect(lambda steps, vv=v: self._scroll_slice(vv, steps))
            v.plane_changed.connect(lambda _plane, vv=v: self._on_view_plane(vv))
            v._zoom.zoom_changed.connect(self._sync_zoom)
            v.attach_hud()

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
            r.toggled.connect(lambda *_: self.refresh(True))

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
        self.chk_native.setChecked(True)
        self.chk_mapped.setChecked(True)
        self.chk_native.toggled.connect(lambda *_: self.refresh(True))
        self.chk_mapped.toggled.connect(lambda *_: self.refresh(True))
        self.chk_edit.toggled.connect(self._toggle_edit)
        self.chk_crosshair = QCheckBox("十字线", self)
        self.chk_crosshair.setChecked(True)
        self.chk_crosshair.hide()
        self.chk_crosshair.toggled.connect(lambda *_: self.refresh(False))

        self.spin_alpha = QDoubleSpinBox(self)
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setValue(0.55)
        self.spin_alpha.hide()
        self.spin_alpha.valueChanged.connect(lambda *_: self.refresh(True))

        self.spin_suv_min = QDoubleSpinBox(self)
        self.spin_suv_min.setRange(0.0, 20.0)
        self.spin_suv_min.setValue(0.0)
        self.spin_suv_min.hide()
        self.spin_suv_max = QDoubleSpinBox(self)
        self.spin_suv_max.setRange(0.5, 40.0)
        self.spin_suv_max.setValue(6.0)
        self.spin_suv_max.hide()
        self.spin_suv_min.valueChanged.connect(lambda *_: self.refresh(True))
        self.spin_suv_max.valueChanged.connect(lambda *_: self.refresh(True))
        self.spin_suv = self.spin_suv_max

        self.spin_wl = QSpinBox(self)
        self.spin_wl.setRange(-1000, 3000)
        self.spin_wl.setValue(int(DEFAULT_WL))
        self.spin_wl.hide()
        self.spin_ww = QSpinBox(self)
        self.spin_ww.setRange(1, 4000)
        self.spin_ww.setValue(int(DEFAULT_WW))
        self.spin_ww.hide()
        self.spin_wl.valueChanged.connect(lambda *_: self.refresh(True))
        self.spin_ww.valueChanged.connect(lambda *_: self.refresh(True))

        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(WHEEL_DPI_MIN, WHEEL_DPI_MAX)
        self.spin_dpi.setSingleStep(10)
        self.spin_dpi.setValue(DEFAULT_WHEEL_DPI)
        self.spin_dpi.setToolTip("每一格滚轮只翻一层。默认 10，大约 0.6 秒才能再翻一层。")
        self.spin_dpi.valueChanged.connect(self._apply_wheel_dpi)
        self._apply_wheel_dpi(self.spin_dpi.value())

        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(3, 15)
        self.spin_brush.setValue(5)
        self.spin_brush.valueChanged.connect(lambda v: setattr(self, "brush_radius", int(v)))

        self.combo_cmap = QComboBox(self)
        for label, key in PET_CMAP_CHOICES:
            self.combo_cmap.addItem(label, key)
        idx = self.combo_cmap.findData(DEFAULT_PET_CMAP)
        self.combo_cmap.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_cmap.hide()
        self.combo_cmap.currentIndexChanged.connect(lambda *_: self.refresh(True))
        for view in (self.axial, self.coronal, self.sagittal):
            self._fill_view(view)
            view._window.edited.connect(lambda v=view: self._on_view_edited(v))
            view._pet.edited.connect(lambda v=view: self._on_view_edited(v))

        self.combo_render = QComboBox()
        self.lbl_cuda = configure_render_combo(self.combo_render)
        self.combo_render.currentIndexChanged.connect(self._on_render_device)

        self.lbl_pos = QLabel("—")
        self.lbl_pet_unit = QLabel("SUV", self)
        self.lbl_pet_unit.hide()
        self.lbl_pos.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        for spin in (self.spin_dpi, self.spin_brush):
            spin.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            spin.setMaximumWidth(90)

        for slider in (self.slider_k, self.slider_j, self.slider_i):
            slider.setFixedWidth(140)

        control_content = QWidget()
        control_row = QHBoxLayout(control_content)
        control_row.setContentsMargins(4, 2, 4, 2)
        control_row.setSpacing(6)

        def add_section(*widgets: QWidget) -> None:
            if control_row.count():
                separator = QFrame()
                separator.setFrameShape(QFrame.Shape.VLine)
                separator.setFrameShadow(QFrame.Shadow.Sunken)
                control_row.addWidget(separator)
            for widget in widgets:
                control_row.addWidget(widget)

        add_section(QLabel("轴位"), self.slider_k, QLabel("冠状"), self.slider_j,
                    QLabel("矢状"), self.slider_i)
        add_section(self.radio_fusion, self.radio_ct, self.radio_pet,
                    QLabel("查看"), self.radio_bl, self.radio_in, self.radio_end)
        self.btn_pen = self._make_tool_button(
            "画笔",
            "按住左键沿边缘画一圈，松开后封闭并加入当前病灶。"
            "连续单击没有点数上限，只有点中绿色空心方块起点才封闭。"
            "按住 Shift 松开则把这块从当前病灶中去掉。"
            "没连成封闭图形会自动取消。再按一次按钮可退出。右键单击撤销上一步。",
        )
        self.btn_poly = self._make_tool_button(
            "多边形",
            "把当前层上的当前病灶转成细小的空心方块锚点，内部不再铺颜色。"
            "拖拽锚点修改形状。再按一次按钮可退出。右键单击撤销上一步。",
        )
        self.btn_pen.toggled.connect(self._on_pen_toggled)
        self.btn_poly.toggled.connect(self._on_poly_toggled)
        self.lbl_brush = QLabel("笔刷")
        add_section(self.chk_native, self.chk_mapped, self.chk_edit,
                    self.lbl_brush, self.spin_brush, self.btn_pen, self.btn_poly)
        self._set_edit_tools_enabled(self.chk_edit.isChecked())

        self._control_scroll = QScrollArea()
        self._control_scroll.setWidget(control_content)
        self._control_scroll.setWidgetResizable(False)
        self._control_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._control_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._control_scroll.setFixedHeight(
            control_content.sizeHint().height()
            + self._control_scroll.horizontalScrollBar().sizeHint().height() + 4
        )
        self.control_bar = self._control_scroll

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(views, 1)
        layout.addWidget(self.lbl_pos)

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

    def _fill_view(self, view: "_RgbView") -> None:
        view._window.write_values(self.spin_wl.value(), self.spin_ww.value())
        view._pet.write_values(
            self.spin_suv_min.value(),
            self.spin_suv_max.value(),
            self.combo_cmap.currentIndex(),
            self.spin_alpha.value(),
        )

    def _on_view_edited(self, view: "_RgbView") -> None:
        if self._hud_lock:
            return
        self._hud_lock = True
        widgets = (
            self.spin_wl, self.spin_ww, self.spin_suv_min, self.spin_suv_max,
            self.spin_alpha, self.combo_cmap,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self.spin_wl.setValue(view._window.spin_wl.value())
        self.spin_ww.setValue(view._window.spin_ww.value())
        self.spin_suv_min.setValue(view._pet.spin_suv_min.value())
        self.spin_suv_max.setValue(view._pet.spin_suv_max.value())
        self.spin_alpha.setValue(view._pet.spin_alpha.value())
        self.combo_cmap.setCurrentIndex(view._pet.combo.currentIndex())
        for other in (self.axial, self.coronal, self.sagittal):
            if other is not view:
                self._fill_view(other)
        for widget in widgets:
            widget.blockSignals(False)
        self._hud_lock = False
        self.refresh(True)
        self.display_params_changed.emit()

    def render_device(self) -> str:
        if self._gpu_failed or not gpu_available():
            return "cpu"
        data = self.combo_render.currentData()
        return str(data) if data else "cpu"

    def mark_masks_dirty(self) -> None:
        self._masks_dirty = True
        if self._vol is not None:
            self._vol.mask_revision += 1

    def _on_render_device(self, _index: int = 0) -> None:
        self._gpu_failed = False
        self._gpu.clear()
        self._masks_dirty = True
        self._bind_gpu()
        self.refresh(True)

    def _bind_gpu(self) -> None:
        vol = self._vol
        if vol is None or self.render_device() != "gpu":
            return
        if self._gpu.bind_volumes(vol):
            self._masks_dirty = False

    def display_mode(self) -> str:
        if self.radio_ct.isChecked():
            return "ct"
        if self.radio_pet.isChecked():
            return "pet"
        return "fusion"

    def _toggle_edit(self, on: bool) -> None:
        self._set_edit_tools_enabled(on)
        if not on:
            self.btn_pen.setChecked(False)
            self.btn_poly.setChecked(False)
        for view in (self.axial, self.coronal, self.sagittal):
            view.item.edit_mode = bool(on)
        self._apply_tool()
        self.refresh(False)
        self._publish_tool_hint()

    def _set_edit_tools_enabled(self, on: bool) -> None:
        for widget in (self.lbl_brush, self.spin_brush, self.btn_pen, self.btn_poly):
            widget.setEnabled(bool(on))

    def _tool_hint(self) -> str:
        if not self.chk_edit.isChecked():
            return ""
        mode = self._tool_mode()
        if mode == "pen":
            return (
                "画笔：拖动松开后封闭并加入当前病灶；连续单击须点中绿色空心方块起点，没有点数上限。"
                "按住 Shift 去掉这块。右键撤销。"
            )
        if mode == "polygon":
            return "多边形：拖拽空心方块修改当前层边界，内部不铺颜色。右键撤销。"
        return "笔刷：左键涂当前层的当前病灶，右键擦除。"

    def set_volumes(self, vol: Optional[VolumeSet]) -> None:
        self._vol = vol
        self._pen_points = []
        self._pen_placing = False
        self._pen_host = None
        self._poly_paths = []
        self._poly_host = None
        self._poly_drag = None
        self._poly_base = None
        self._poly_undo = False
        for button in (self.btn_pen, self.btn_poly):
            if button.isChecked():
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
        self._apply_tool()
        unit = "SUV" if vol is None or vol.pet_units == "SUVbw" else f"PET ({vol.pet_units})"
        self.lbl_pet_unit.setText(unit)
        for view in (self.axial, self.coronal, self.sagittal):
            view._id.set_header(
                "" if vol is None else vol.patient_id,
                "" if vol is None else vol.study_date,
            )
            view._pet.set_unit(unit)
            view._place_hud()
            view._place_laterality()
        self._gpu_failed = False
        self._masks_dirty = True
        self._shown_ijk = None
        if vol is None:
            self.lbl_pos.setText("无图像")
            for view in (self.axial, self.coronal, self.sagittal):
                view.item.clear()
                view.set_crosshair_visible(False)
            self.cursor_changed.emit(0, 0, 0, False)
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
        self.refresh(True)
        for view in (self.axial, self.coronal, self.sagittal):
            view.set_zoom(view._zoom_percent)

    def volumes(self) -> Optional[VolumeSet]:
        return self._vol

    def _set_ijk(self, i: int, j: int, k: int) -> None:
        self.slider_i.blockSignals(True)
        self.slider_j.blockSignals(True)
        self.slider_k.blockSignals(True)
        self.slider_i.setValue(int(i))
        self.slider_j.setValue(int(j))
        self.slider_k.setValue(int(k))
        self.slider_i.blockSignals(False)
        self.slider_j.blockSignals(False)
        self.slider_k.blockSignals(False)
        self._i = self.slider_i.value()
        self._j = self.slider_j.value()
        self._k = self.slider_k.value()
        self._drop_pen_if_slice_moved()
        self._drop_polygon_if_slice_moved()
        self.refresh(False)

    def _sliders_changed(self) -> None:
        self._i = self.slider_i.value()
        self._j = self.slider_j.value()
        self._k = self.slider_k.value()
        self._drop_pen_if_slice_moved()
        self._drop_polygon_if_slice_moved()
        self.refresh(False)

    def _scroll_slice(self, view: _RgbView, steps: int) -> None:
        if self._vol is None or not steps:
            return
        which = view.view_name
        self._active_host = view
        i, j, k = self._i, self._j, self._k
        if which == "axial":
            k += steps
        elif which == "coronal":
            j += steps
        else:
            i += steps
        self.active_view = which
        self._set_ijk(i, j, k)

    def _click(self, view: _RgbView, x: float, y: float) -> None:
        which = view.view_name
        self._active_host = view
        self.active_view = which
        if self._vol is None:
            return
        from display_utils import display_to_voxel

        ii, jj, kk = display_to_voxel(
            which, x, y, self._i, self._j, self._k, self._vol.ct.shape
        )
        self._set_ijk(ii, jj, kk)

    def _push_undo(self) -> None:
        if self._vol is None:
            return
        self._vol._undo.append(self._vol.native.copy())
        if len(self._vol._undo) > 20:
            self._vol._undo.pop(0)

    def undo(self) -> None:
        if self._vol is None or not self._vol._undo:
            self._pen_points = []
            self._redraw_overlay()
            return
        self._vol.native = self._vol._undo.pop()
        self._vol.dirty = True
        self._pen_points = []
        self._poly_drag = None
        self._poly_base = None
        self.mark_masks_dirty()
        self.refresh(True)
        if self.btn_poly.isChecked():
            self._load_polygon_from_mask()
        else:
            self._redraw_overlay()
        self.mask_changed.emit()

    def _paint(self, view: _RgbView, x: float, y: float, erase: bool) -> None:
        vol = self._vol
        if vol is None:
            return
        which = view.view_name
        self._active_host = view
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
        self.refresh(True)

    def _end_stroke(self) -> None:
        vol = self._vol
        if vol is not None and self._stroke and not self._stroke_erase and vol._undo:
            promote_new_islands(vol.native, vol._undo[-1])
            self.mark_masks_dirty()
            self.refresh(True)
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
        self.refresh(True)
        self.mask_changed.emit()

    def ijk(self) -> tuple[int, int, int]:
        return self._i, self._j, self._k

    def _sync_zoom(self, percent: int) -> None:
        for view in (self.axial, self.coronal, self.sagittal):
            if view._zoom_percent != int(percent):
                view.set_zoom(int(percent))

    def wheel_dpi(self) -> int:
        return int(self.spin_dpi.value())

    def _apply_wheel_dpi(self, dpi: int) -> None:
        for view in (self.axial, self.coronal, self.sagittal):
            view.set_wheel_dpi(int(dpi))

    def refresh(self, force: bool = True) -> None:
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
        prev = self._shown_ijk
        slice_index = {"axial": self._k, "coronal": self._j, "sagittal": self._i}
        prev_index = None if prev is None else {"axial": prev[2], "coronal": prev[1], "sagittal": prev[0]}
        planes = [
            (
                view,
                view.view_name,
                prev_index is None or force or prev_index[view.view_name] != slice_index[view.view_name],
            )
            for view in (self.axial, self.coronal, self.sagittal)
        ]
        need = [(view, name) for view, name, yes in planes if yes]
        if need:
            device = self.render_device()
            gpu_cache = None
            if device == "gpu":
                try:
                    if not self._gpu.bind_volumes(vol):
                        raise RuntimeError("GPU 不可用")
                    self._masks_dirty = False
                    gpu_cache = self._gpu
                except Exception:
                    _LOG.exception("GPU 体积上传失败，回退 CPU")
                    self._gpu_failed = True
                    device = "cpu"
            used_gpu = device == "gpu" and gpu_cache is not None
            if used_gpu:
                try:
                    for view, plane in need:
                        rgb = gpu_cache.compose_plane(plane, self._i, self._j, self._k, **args)
                        if self._hide_poly_fill(plane):
                            plain = gpu_cache.compose_plane(
                                plane, self._i, self._j, self._k,
                                **{**args, "show_native": False, "show_mapped": False},
                            )
                            rgb = self._clear_roi_interior(plane, rgb, plain)
                        view.set_rgb(rgb)
                except Exception:
                    _LOG.exception("GPU 合成失败，回退 CPU")
                    self._gpu_failed = True
                    used_gpu = False
            if not used_gpu:
                for view, plane in need:
                    rgb = compose_plane_cpu(vol, plane, self._i, self._j, self._k, **args)
                    if self._hide_poly_fill(plane):
                        plain = compose_plane_cpu(
                            vol, plane, self._i, self._j, self._k,
                            **{**args, "show_native": False, "show_mapped": False},
                        )
                        rgb = self._clear_roi_interior(plane, rgb, plain)
                    view.set_rgb(rgb)
        suv = float(vol.pet[self._i, self._j, self._k])
        hu = float(vol.ct[self._i, self._j, self._k])
        lid = int(native[self._i, self._j, self._k]) if native is not None else 0
        extra = f"  灶#{lid}" if lid else ""
        hint = self._tool_hint()
        hint_text = f"    {hint}" if hint else ""
        self.lbl_pos.setText(
            f"{vol.study_date}  ijk=({self._i},{self._j},{self._k})  "
            f"HU={hu:.0f}  {('SUV' if vol.pet_units == 'SUVbw' else 'PET (' + vol.pet_units + ')')}={suv:.2f}{extra}"
            f"{hint_text}"
        )
        self._shown_ijk = (self._i, self._j, self._k)
        self._update_crosshairs()
        self._redraw_overlay()
        self._publish_tool_hint()

    def _publish_tool_hint(self) -> None:
        hint = self._tool_hint()
        if hint == self._tool_hint_sent:
            return
        self._tool_hint_sent = hint
        self.tool_hint_changed.emit(hint)

    @staticmethod
    def _make_tool_button(text: str, tip: str) -> QPushButton:
        button = QPushButton(text)
        button.setCheckable(True)
        button.setToolTip(tip)
        button.setStyleSheet(
            "QPushButton {"
            " border: 1px solid #8a8a8a; border-radius: 3px; padding: 2px 8px;"
            " background: transparent; color: #f0f0f0;"
            "}"
            "QPushButton:checked {"
            " border: 2px solid #7dff6a; border-radius: 3px; padding: 1px 7px;"
            " background: #102416;"
            "}"
            "QPushButton:checked:pressed {"
            " border: 2px solid #b7ff9a; background: #0b1a10;"
            "}"
            "QPushButton:disabled {"
            " color: #777777; border: 1px solid #555555; background: transparent;"
            "}"
        )
        return button

    def _hide_poly_fill(self, plane: str) -> bool:
        vol = self._vol
        return bool(
            self.btn_poly.isChecked()
            and plane == self._poly_plane
            and self._poly_label > 0
            and vol is not None
            and vol.native is not None
        )

    def _clear_roi_interior(self, plane: str, rgb: np.ndarray, plain: np.ndarray) -> np.ndarray:
        vol = self._vol
        if vol is None or vol.native is None:
            return rgb
        region = label_on_slice(vol.native, plane, self._i, self._j, self._k, self._poly_label)
        if region.shape != rgb.shape[:2] or plain.shape != rgb.shape or not np.any(region):
            return rgb
        out = np.array(rgb, copy=True)
        out[region] = plain[region]
        return out

    def _tool_mode(self) -> str:
        if self.btn_poly.isChecked():
            return "polygon"
        if self.btn_pen.isChecked():
            return "pen"
        return "brush"

    def _apply_tool(self) -> None:
        mode = self._tool_mode()
        for view in (self.axial, self.coronal, self.sagittal):
            view.item.tool = mode

    def _view(self, name: str) -> _RgbView:
        matches = [view for view in (self.axial, self.coronal, self.sagittal) if view.view_name == name]
        for host in (self._poly_host, self._pen_host, self._active_host):
            if host in matches:
                return host
        return matches[0] if matches else self.axial

    def _on_view_plane(self, view: _RgbView) -> None:
        self._active_host = view
        self.active_view = view.view_name
        if self._pen_host is view:
            self._pen_points = []
            self._pen_host = None
        if self._poly_host is view and self.btn_poly.isChecked():
            self._load_polygon_from_mask()
        self.refresh(True)

    def _on_pen_toggled(self, on: bool) -> None:
        if on and self.btn_poly.isChecked():
            self.btn_poly.blockSignals(True)
            self.btn_poly.setChecked(False)
            self.btn_poly.blockSignals(False)
            self._poly_paths = []
        self._pen_points = []
        self._pen_host = None
        self._poly_drag = None
        self._apply_tool()
        self._redraw_overlay()
        self.refresh(True)

    def _on_poly_toggled(self, on: bool) -> None:
        if on and self.btn_pen.isChecked():
            self.btn_pen.blockSignals(True)
            self.btn_pen.setChecked(False)
            self.btn_pen.blockSignals(False)
        self._pen_points = []
        self._poly_drag = None
        self._apply_tool()
        if on:
            self._load_polygon_from_mask()
        else:
            self._poly_paths = []
            self._poly_host = None
            self._redraw_overlay()
        self.refresh(True)

    def _clear_roi_graphics(self) -> None:
        for view in (self.axial, self.coronal, self.sagittal):
            view.roi_line.setData([], [])
            view.roi_anchors.setData([], [])
            view.roi_marks.setData([], [])

    def _show_paths(self, view: _RgbView, paths: list[np.ndarray], *, anchors: bool, close: bool) -> None:
        self._clear_roi_graphics()
        xs: list[float] = []
        ys: list[float] = []
        ax: list[float] = []
        ay: list[float] = []
        for path in paths:
            pts = np.asarray(path, dtype=np.float64)
            if len(pts) == 0:
                continue
            xs.extend(pts[:, 0].tolist())
            ys.extend(pts[:, 1].tolist())
            if close and len(pts) >= 2:
                xs.append(float(pts[0, 0]))
                ys.append(float(pts[0, 1]))
            xs.append(np.nan)
            ys.append(np.nan)
            if anchors:
                ax.extend(pts[:, 0].tolist())
                ay.extend(pts[:, 1].tolist())
        view.roi_line.setPen(pg.mkPen("#7dff6a", width=1))
        view.roi_line.setData(xs, ys)
        if anchors:
            view.roi_anchors.setData(ax, ay, size=4, symbol="s")

    def _show_pen(self, view: _RgbView, points: list[tuple[float, float]]) -> None:
        self._clear_roi_graphics()
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) == 0:
            return
        view.roi_line.setPen(pg.mkPen("#7dff6a", width=1))
        view.roi_line.setData(pts[:, 0], pts[:, 1])
        xs = [float(pts[0, 0])]
        ys = [float(pts[0, 1])]
        symbols = ["s"]
        pens = [pg.mkPen("#7dff6a", width=1)]
        brushes = [pg.mkBrush(0, 0, 0, 0)]
        sizes = [8]
        same = len(pts) > 1 and float(np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])) <= 0.5
        if len(pts) > 1 and not same:
            xs.append(float(pts[-1, 0]))
            ys.append(float(pts[-1, 1]))
            symbols.append("s")
            pens.append(pg.mkPen("#ffb020", width=1))
            brushes.append(pg.mkBrush(0, 0, 0, 0))
            sizes.append(8)
        view.roi_marks.setData(x=xs, y=ys, symbol=symbols, pen=pens, brush=brushes, size=sizes)

    def _redraw_overlay(self) -> None:
        if self.btn_pen.isChecked() and self._pen_points:
            self._show_pen(self._pen_host or self._view(self._pen_view), self._pen_points)
            return
        if self.btn_poly.isChecked() and self._poly_paths:
            host = self._poly_host or self._view(self._poly_plane)
            self._show_paths(host, self._poly_paths, anchors=True, close=True)
            return
        self._clear_roi_graphics()

    def _pen_same_slice(self) -> bool:
        pi, pj, pk = self._pen_ijk
        if self._pen_view == "axial":
            return self._k == pk
        if self._pen_view == "coronal":
            return self._j == pj
        return self._i == pi

    def _cancel_pen(self) -> None:
        self._pen_points = []
        self._pen_placing = False
        self._pen_host = None
        self._redraw_overlay()

    def _hits_pen_start(self, x: float, y: float) -> bool:
        if not self._pen_points:
            return False
        sx, sy = self._pen_points[0]
        tol = 2.0
        view = self._pen_host
        if view is not None:
            try:
                px, py = view.box.viewPixelSize()
                tol = max(5.0 * float(px), 5.0 * float(py))
            except Exception:
                tol = 2.0
        tol = float(np.clip(tol, 1.5, 4.0))
        return float(np.hypot(sx - x, sy - y)) <= tol

    def _drop_pen_if_slice_moved(self) -> None:
        if not self._pen_points:
            return
        pi, pj, pk = self._pen_ijk
        moved = (
            (self._pen_view == "axial" and self._k != pk)
            or (self._pen_view == "coronal" and self._j != pj)
            or (self._pen_view == "sagittal" and self._i != pi)
        )
        if moved:
            self._pen_points = []

    def _drop_polygon_if_slice_moved(self) -> None:
        if not self._poly_paths:
            return
        pi, pj, pk = self._poly_ijk
        moved = (
            (self._poly_plane == "axial" and self._k != pk)
            or (self._poly_plane == "coronal" and self._j != pj)
            or (self._poly_plane == "sagittal" and self._i != pi)
        )
        if not moved:
            return
        self._poly_paths = []
        self._poly_drag = None
        self._poly_base = None
        if self.btn_poly.isChecked():
            self.btn_poly.blockSignals(True)
            self.btn_poly.setChecked(False)
            self.btn_poly.blockSignals(False)
            self._apply_tool()

    def _working_label(self, which: str, *, for_subtract: bool) -> int:
        vol = self._vol
        if vol is None or vol.native is None:
            return 0
        label = int(self.current_label)
        if label > 0:
            return label
        if for_subtract:
            return dominant_label(vol.native, which, self._i, self._j, self._k)
        return 1

    def _gesture(self, view: _RgbView, x: float, y: float, phase: str, subtract: bool) -> None:
        if not self.chk_edit.isChecked():
            return
        self._active_host = view
        which = view.view_name
        if phase == "undo":
            self._cancel_pen()
            self.undo()
            return
        self.active_view = which
        if self._tool_mode() == "pen":
            self._pen_host = view
            self._pen_gesture(which, x, y, phase, subtract)
        elif self._tool_mode() == "polygon":
            self._poly_gesture(view, x, y, phase)

    def _pen_gesture(self, which: str, x: float, y: float, phase: str, subtract: bool) -> None:
        if phase == "click":
            if time.monotonic() < self._pen_ignore_until:
                return
            if (
                len(self._pen_points) >= 3
                and which == self._pen_view
                and self._pen_same_slice()
                and self._hits_pen_start(x, y)
            ):
                self._pen_points.append(self._pen_points[0])
                self._pen_placing = False
                self._commit_pen(subtract)
                return
            if which != self._pen_view or not self._pen_same_slice():
                self._pen_points = []
            self._pen_view = which
            self._pen_ijk = (self._i, self._j, self._k)
            self._pen_placing = True
            self._pen_points.append((float(x), float(y)))
            self._redraw_overlay()
            return
        if phase == "start":
            if self._pen_placing and self._pen_points and which == self._pen_view and self._pen_same_slice():
                self._redraw_overlay()
                return
            self._pen_placing = False
            self._pen_view = which
            self._pen_ijk = (self._i, self._j, self._k)
            self._pen_points = [(float(x), float(y))]
            self._redraw_overlay()
            return
        if phase == "move":
            if which != self._pen_view or not self._pen_points:
                return
            lx, ly = self._pen_points[-1]
            if float(np.hypot(lx - x, ly - y)) < 1.5:
                return
            self._pen_points.append((float(x), float(y)))
            self._redraw_overlay()
            return
        if phase == "finish":
            self._pen_ignore_until = time.monotonic() + 0.25
            if self._pen_placing:
                self._redraw_overlay()
                return
            if len(self._pen_points) >= 3 and which == self._pen_view and self._pen_same_slice():
                self._commit_pen(subtract)
            else:
                self._cancel_pen()

    def _commit_pen(self, subtract: bool) -> None:
        vol = self._vol
        points = np.asarray(self._pen_points, dtype=np.float64)
        self._pen_points = []
        if vol is None or len(points) < 3:
            self._redraw_overlay()
            return
        if vol.native is None:
            vol.native = np.zeros(vol.ct.shape, dtype=np.uint16)
        label = self._working_label(self._pen_view, for_subtract=subtract)
        if label <= 0:
            self._redraw_overlay()
            return
        self._push_undo()
        changed = apply_closed_paths(
            vol.native,
            self._pen_view,
            self._i,
            self._j,
            self._k,
            [points],
            label=label,
            subtract=subtract,
        )
        if changed <= 0:
            vol._undo.pop()
            self._redraw_overlay()
            return
        if not subtract and vol._undo:
            promote_new_islands(vol.native, vol._undo[-1])
        vol.dirty = True
        self.mark_masks_dirty()
        self.refresh(True)
        self.mask_changed.emit()

    def _load_polygon_from_mask(self) -> None:
        vol = self._vol
        if vol is None or vol.native is None:
            self._reject_polygon()
            return
        host = self._active_host if self._active_host is not None else self._view(self.active_view)
        plane = host.view_name
        self.active_view = plane
        label = int(self.current_label)
        if label <= 0:
            label = dominant_label(vol.native, plane, self._i, self._j, self._k)
        paths = (
            contours_from_label(
                vol.native,
                plane,
                self._i,
                self._j,
                self._k,
                label,
                epsilon=0.5,
                max_points=200,
            )
            if label > 0
            else []
        )
        if not paths:
            self._reject_polygon()
            return
        self._poly_paths = paths
        self._poly_host = host
        self._poly_plane = plane
        self._poly_ijk = (self._i, self._j, self._k)
        self._poly_label = label
        self._poly_drag = None
        self._poly_base = None
        self._redraw_overlay()

    def _reject_polygon(self) -> None:
        self._poly_paths = []
        self._poly_host = None
        if self.btn_poly.isChecked():
            self.btn_poly.blockSignals(True)
            self.btn_poly.setChecked(False)
            self.btn_poly.blockSignals(False)
            self._apply_tool()
        self._redraw_overlay()

    def _poly_gesture(self, view: _RgbView, x: float, y: float, phase: str) -> None:
        if view is not self._poly_host or view.view_name != self._poly_plane or not self._poly_paths:
            return
        if phase in ("start", "click"):
            hit = nearest_anchor(self._poly_paths, x, y, radius=10.0)
            if hit is None or self._vol is None or self._vol.native is None:
                return
            self._poly_drag = hit
            self._poly_base = self._vol.native.copy()
            self._poly_changed = False
            self._poly_moved = False
            self._push_undo()
            self._poly_undo = True
            if phase == "click":
                self._cancel_polygon_drag()
            return
        if phase == "move":
            if self._poly_drag is None:
                return
            ox, oy = self._poly_paths[self._poly_drag[0]][self._poly_drag[1]]
            if not self._poly_moved and float(np.hypot(ox - x, oy - y)) < 0.75:
                return
            self._poly_moved = True
            self._move_polygon_anchor(x, y)
            return
        if phase == "finish":
            if not self._poly_undo:
                return
            if self._poly_moved:
                self._move_polygon_anchor(x, y)
            if not self._poly_changed:
                self._cancel_polygon_drag()
            else:
                self._poly_drag = None
                self._poly_base = None
                self._poly_undo = False
                self.mask_changed.emit()

    def _cancel_polygon_drag(self) -> None:
        vol = self._vol
        if self._poly_undo and vol is not None and vol._undo:
            vol._undo.pop()
        if vol is not None and self._poly_base is not None:
            vol.native = self._poly_base
            vol.dirty = True
        self._poly_drag = None
        self._poly_base = None
        self._poly_undo = False
        self._poly_changed = False
        if self.btn_poly.isChecked():
            self._load_polygon_from_mask()

    def _move_polygon_anchor(self, x: float, y: float) -> None:
        vol = self._vol
        if (
            vol is None
            or vol.native is None
            or self._poly_drag is None
            or self._poly_base is None
        ):
            return
        pi, idx = self._poly_drag
        paths = [np.array(path, dtype=np.float64, copy=True) for path in self._poly_paths]
        paths[pi][idx, 0] = float(x)
        paths[pi][idx, 1] = float(y)
        self._poly_paths = paths
        vol.native = self._poly_base.copy()
        changed = replace_slice_with_paths(
            vol.native,
            self._poly_plane,
            self._i,
            self._j,
            self._k,
            paths,
            self._poly_label,
        )
        self._poly_changed = changed > 0
        vol.dirty = True
        self.mark_masks_dirty()
        self.refresh(True)

    def _update_crosshairs(self) -> None:
        vol = self._vol
        views = (self.axial, self.coronal, self.sagittal)
        on = self.chk_crosshair.isChecked()
        for v in views:
            v.set_crosshair_visible(on)
        self.cursor_changed.emit(self._i, self._j, self._k, bool(vol is not None and on))
        if vol is None or not on:
            return
        shape = vol.ct.shape
        for v in views:
            col, row = voxel_to_display(v.view_name, self._i, self._j, self._k, shape)
            v.set_crosshair(col, row)
