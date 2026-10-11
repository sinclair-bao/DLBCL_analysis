#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""右侧面板：指定 baseline / interim / end。"""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from session import ROLES, LongitudinalSession

NONE_LABEL = "（未指定）"


class TimepointPanel(QWidget):
    session_changed = Signal()
    map_requested = Signal()
    features_requested = Signal()
    review_mapping_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dates: list[str] = []
        self._combos: dict[str, QComboBox] = {}
        self._suppress = False

        box = QGroupBox("时间点指定")
        form = QFormLayout()
        labels = {"baseline": "基线 Baseline", "interim": "中期 Interim", "end": "末期 End"}
        for role in ROLES:
            combo = QComboBox()
            combo.currentIndexChanged.connect(self._emit_if_ready)
            self._combos[role] = combo
            form.addRow(labels[role], combo)
        box.setLayout(form)
        self.extra = QListWidget()
        self.extra.setMaximumHeight(105)
        self.extra.itemChanged.connect(self._emit_if_ready)

        self.hint = QLabel("打开患者后，为基线/中期/末期各选一次检查。")
        self.hint.setWordWrap(True)

        self.btn_map = QPushButton("将基线病灶映射到中期/末期")
        self.btn_feat = QPushButton("计算代谢 / 组学特征")
        self.chk_per_lesion = QCheckBox("同时输出逐病灶特征（较慢）")
        self.btn_review_map = QPushButton("查看 CT 配准并记录质控")
        self.btn_map.clicked.connect(self.map_requested.emit)
        self.btn_feat.clicked.connect(self.features_requested.emit)
        self.btn_review_map.clicked.connect(self.review_mapping_requested.emit)

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(QLabel("其他随访（可多选）"))
        layout.addWidget(self.extra)
        layout.addWidget(self.hint)
        layout.addWidget(self.btn_map)
        layout.addWidget(self.btn_review_map)
        layout.addWidget(self.chk_per_lesion)
        layout.addWidget(self.btn_feat)

    def set_dates(self, dates: list[str], session: LongitudinalSession) -> None:
        self._suppress = True
        self._dates = list(dates)
        for role, combo in self._combos.items():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(NONE_LABEL, "")
            for d in dates:
                combo.addItem(d, d)
            value = getattr(session, role) or ""
            idx = combo.findData(value)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
        self.extra.blockSignals(True)
        self.extra.clear()
        for date in dates:
            item = QListWidgetItem(date)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if date in session.extra_dates else Qt.CheckState.Unchecked)
            self.extra.addItem(item)
        self.extra.blockSignals(False)
        self._suppress = False

    def current_session(self, patient_id: str) -> LongitudinalSession:
        values = {}
        for role, combo in self._combos.items():
            data = combo.currentData()
            values[role] = data or None
        values["extra_dates"] = [self.extra.item(i).text() for i in range(self.extra.count())
                                 if self.extra.item(i).checkState() == Qt.CheckState.Checked]
        return LongitudinalSession(patient_id=patient_id, **values)

    def _emit_if_ready(self) -> None:
        if not self._suppress:
            self.session_changed.emit()
