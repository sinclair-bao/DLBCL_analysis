#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""右侧面板：指定 baseline / interim / end。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
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

        self.hint = QLabel("打开患者后，为基线/中期/末期各选一次检查。")
        self.hint.setWordWrap(True)

        self.btn_map = QPushButton("计算基线 → 随访映射")
        self.btn_feat = QPushButton("计算代谢 / 组学特征")
        self.btn_map.clicked.connect(self.map_requested.emit)
        self.btn_feat.clicked.connect(self.features_requested.emit)

        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(self.hint)
        layout.addWidget(self.btn_map)
        layout.addWidget(self.btn_feat)
        layout.addStretch(1)

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
        self._suppress = False

    def current_session(self, patient_id: str) -> LongitudinalSession:
        values = {}
        for role, combo in self._combos.items():
            data = combo.currentData()
            values[role] = data or None
        return LongitudinalSession(patient_id=patient_id, **values)

    def _emit_if_ready(self) -> None:
        if not self._suppress:
            self.session_changed.emit()
