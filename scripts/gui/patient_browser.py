#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""左侧患者树：PatientID → StudyDate，状态灯表示数据完整性。"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from catalog import DataCatalog, PatientRecord

_STATUS_COLOR = {
    "ok": QColor("#3dd68c"),
    "partial": QColor("#e6b84d"),
    "missing": QColor("#e85d5d"),
}


class PatientBrowser(QTreeWidget):
    patient_selected = Signal(str)
    study_selected = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHeaderLabels(["患者 / 检查", "状态"])
        self.setColumnWidth(0, 160)
        self.setAnimated(True)
        self.itemSelectionChanged.connect(self._on_select)

    def populate(self, catalog: DataCatalog) -> None:
        self.clear()
        for pid in catalog.patient_ids():
            rec = catalog.get_patient(pid)
            if rec is None:
                continue
            self._add_patient(rec)

    def _add_patient(self, rec: PatientRecord) -> None:
        parent = QTreeWidgetItem([rec.patient_id, f"{len(rec.studies)} 次"])
        parent.setData(0, Qt.ItemDataRole.UserRole, ("patient", rec.patient_id, ""))
        worst = "ok"
        for study in rec.studies:
            status = study.completeness()
            if status == "missing":
                worst = "missing"
            elif status == "partial" and worst == "ok":
                worst = "partial"
            child = QTreeWidgetItem([study.study_date, study.status_note()])
            child.setData(0, Qt.ItemDataRole.UserRole, ("study", rec.patient_id, study.study_date))
            color = _STATUS_COLOR.get(status, _STATUS_COLOR["missing"])
            child.setForeground(1, QBrush(color))
            parent.addChild(child)
        parent.setForeground(1, QBrush(_STATUS_COLOR[worst]))
        self.addTopLevelItem(parent)

    def select_patient(self, patient_id: str) -> None:
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if item and item.text(0) == patient_id:
                self.setCurrentItem(item)
                return

    def _on_select(self) -> None:
        items = self.selectedItems()
        if not items:
            return
        kind, pid, date = items[0].data(0, Qt.ItemDataRole.UserRole)
        if kind == "patient":
            self.patient_selected.emit(pid)
        else:
            self.patient_selected.emit(pid)
            self.study_selected.emit(pid, date)
