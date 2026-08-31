#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""分割触发、形态学微调、病灶编号列表、保存 edited mask。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class EditPanel(QWidget):
    segment_requested = Signal(str)  # autopet | threshold | manual
    morph_requested = Signal(str)  # dilate/erode/open/close
    relabel_requested = Signal()
    undo_requested = Signal()
    save_requested = Signal()
    lesion_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        seg_box = QGroupBox("分割")
        seg_l = QVBoxLayout()
        self.btn_autopet = QPushButton("AutoPET 分割")
        self.btn_threshold = QPushButton("SUV 阈值分割")
        self.btn_manual = QPushButton("空白手动分割")
        self.btn_other = QPushButton("其他（载入 mask）")
        self.btn_autopet.clicked.connect(lambda: self.segment_requested.emit("autopet"))
        self.btn_threshold.clicked.connect(lambda: self.segment_requested.emit("threshold"))
        self.btn_manual.clicked.connect(lambda: self.segment_requested.emit("manual"))
        self.btn_other.clicked.connect(lambda: self.segment_requested.emit("other"))
        seg_l.addWidget(self.btn_autopet)
        seg_l.addWidget(self.btn_threshold)
        seg_l.addWidget(self.btn_manual)
        seg_l.addWidget(self.btn_other)
        self.lbl_seg = QLabel("四种入口打开编辑窗。SUV 阈值：41% SUVmax 或滑杆固定值，可在窗内再调。")
        self.lbl_seg.setWordWrap(True)
        seg_l.addWidget(self.lbl_seg)
        seg_box.setLayout(seg_l)

        morph_box = QGroupBox("形态学微调")
        morph_l = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_dilate = QPushButton("膨胀")
        self.btn_erode = QPushButton("腐蚀")
        self.btn_open = QPushButton("开运算")
        self.btn_close = QPushButton("闭运算")
        self.btn_dilate.clicked.connect(lambda: self.morph_requested.emit("dilate"))
        self.btn_erode.clicked.connect(lambda: self.morph_requested.emit("erode"))
        self.btn_open.clicked.connect(lambda: self.morph_requested.emit("open"))
        self.btn_close.clicked.connect(lambda: self.morph_requested.emit("close"))
        for b in (self.btn_dilate, self.btn_erode, self.btn_open, self.btn_close):
            row.addWidget(b)
        morph_l.addLayout(row)

        opt = QHBoxLayout()
        opt.addWidget(QLabel("半径"))
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(1, 5)
        self.spin_radius.setValue(1)
        opt.addWidget(self.spin_radius)
        self.combo_scope = QComboBox()
        self.combo_scope.addItem("当前层 2D", "slice")
        self.combo_scope.addItem("三维", "3d")
        opt.addWidget(self.combo_scope)
        self.combo_target = QComboBox()
        self.combo_target.addItem("当前病灶", "current")
        self.combo_target.addItem("全部病灶", "all")
        opt.addWidget(self.combo_target)
        morph_l.addLayout(opt)

        act = QHBoxLayout()
        self.btn_undo = QPushButton("撤销")
        self.btn_relabel = QPushButton("按体积重新编号")
        self.btn_save = QPushButton("保存调整后的 mask")
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        self.btn_relabel.clicked.connect(self.relabel_requested.emit)
        self.btn_save.clicked.connect(self.save_requested.emit)
        act.addWidget(self.btn_undo)
        act.addWidget(self.btn_relabel)
        morph_l.addLayout(act)
        morph_l.addWidget(self.btn_save)
        morph_box.setLayout(morph_l)

        list_box = QGroupBox("病灶编号")
        list_l = QVBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["编号", "体素", "体积 mL", "SUVmax"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_row)
        list_l.addWidget(self.table)
        list_box.setLayout(list_l)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(seg_box)
        layout.addWidget(morph_box)
        layout.addWidget(list_box, 1)

    def set_busy(self, busy: bool) -> None:
        for b in (
            self.btn_autopet,
            self.btn_threshold,
            self.btn_manual,
            self.btn_other,
            self.btn_dilate,
            self.btn_erode,
            self.btn_open,
            self.btn_close,
            self.btn_save,
        ):
            b.setEnabled(not busy)

    def set_lesions(self, rows: list[dict], selected: int = 0) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        selected_row = -1
        for r, row in enumerate(rows):
            vals = [row.get("id"), row.get("n_voxels"), row.get("volume_ml"), row.get("suv_max")]
            for c, val in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(val)))
            if int(row.get("id") or 0) == selected:
                selected_row = r
        if selected_row >= 0:
            self.table.selectRow(selected_row)
        self.table.blockSignals(False)

    def _on_row(self) -> None:
        items = self.table.selectedItems()
        if not items:
            self.lesion_selected.emit(0)
            return
        try:
            self.lesion_selected.emit(int(items[0].text()))
        except ValueError:
            self.lesion_selected.emit(0)
