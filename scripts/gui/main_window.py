#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""纵向 PET/CT 分析软件主窗口。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

_SCRIPTS = Path(__file__).resolve().parents[1]
_LONG = _SCRIPTS / "longitudinal"
_COMMON = _SCRIPTS / "common"
for _p in (str(_LONG), str(_COMMON), str(_SCRIPTS / "gui"), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catalog import DataCatalog  # noqa: E402
from session import load_session, save_session  # noqa: E402

from evolution_strip import EvolutionStrip  # noqa: E402
from feature_panel import FeaturePanel  # noqa: E402
from ortho_viewer import OrthoViewer  # noqa: E402
from patient_browser import PatientBrowser  # noqa: E402
from timepoint_panel import TimepointPanel  # noqa: E402
from volume_io import load_volume_set  # noqa: E402
from workers import FeatureWorker, MappingWorker  # noqa: E402

STYLESHEET = """
QMainWindow, QWidget { background: #161616; color: #e8e8e8; }
QTreeWidget { background: #1e1e1e; alternate-background-color: #252525; border: 1px solid #333; }
QGroupBox { border: 1px solid #3a3a3a; margin-top: 10px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #b8d4ff; }
QPushButton { background: #2b4c7e; border: none; padding: 6px 10px; border-radius: 3px; }
QPushButton:hover { background: #3a64a8; }
QPushButton:disabled { background: #333; color: #777; }
QComboBox, QDoubleSpinBox { background: #222; border: 1px solid #444; padding: 2px 6px; }
QHeaderView::section { background: #2a2a2a; color: #ddd; padding: 4px; border: 0; }
QStatusBar { background: #111; color: #aaa; }
QMenuBar { background: #1a1a1a; color: #ddd; }
QMenuBar::item:selected { background: #2b4c7e; }
"""


class MainWindow(QMainWindow):
    def __init__(
        self,
        interim_root: Path,
        processed_root: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("DLBCL 纵向 PET/CT 分析")
        self.resize(1600, 980)
        self.setStyleSheet(STYLESHEET)

        self.catalog = DataCatalog(interim_root, processed_root)
        self.current_patient: Optional[str] = None
        self.current_date: Optional[str] = None
        self._map_worker: Optional[MappingWorker] = None
        self._feat_worker: Optional[FeatureWorker] = None
        self._vol_cache: dict[tuple[str, str, str], object] = {}

        self.browser = PatientBrowser()
        self.timepoints = TimepointPanel()
        self.ortho = OrthoViewer()
        self.evolution = EvolutionStrip()
        self.features = FeaturePanel()

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(4, 4, 4, 4)
        left_l.addWidget(QLabel("患者"))
        left_l.addWidget(self.browser, 1)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(4, 4, 4, 4)
        right_l.addWidget(self.timepoints)
        right_l.addWidget(self.features, 1)

        center = QSplitter(Qt.Orientation.Vertical)
        center.addWidget(self.ortho)
        center.addWidget(self.evolution)
        center.setStretchFactor(0, 3)
        center.setStretchFactor(1, 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(center)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 4)
        split.setStretchFactor(2, 2)

        wrapper = QWidget()
        wrap_l = QHBoxLayout(wrapper)
        wrap_l.setContentsMargins(0, 0, 0, 0)
        wrap_l.addWidget(split)
        self.setCentralWidget(wrapper)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._build_menu()

        self.browser.patient_selected.connect(self._on_patient)
        self.browser.study_selected.connect(self._on_study)
        self.timepoints.session_changed.connect(self._on_session_edit)
        self.timepoints.map_requested.connect(self._run_mapping)
        self.timepoints.features_requested.connect(self._run_features)
        self.ortho.chk_native.toggled.connect(self._refresh_evolution)
        self.ortho.chk_mapped.toggled.connect(self._refresh_evolution)

        self.browser.populate(self.catalog)
        n = len(self.catalog.patient_ids())
        self.status.showMessage(f"已索引 {n} 名患者")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        act_refresh = QAction("刷新目录", self)
        act_refresh.triggered.connect(self._refresh)
        file_menu.addAction(act_refresh)

        act_export_csv = QAction("导出特征 CSV…", self)
        act_export_csv.setShortcut(QKeySequence.StandardKey.Save)
        act_export_csv.triggered.connect(self._export_csv)
        file_menu.addAction(act_export_csv)

        act_export_mip = QAction("导出 MIP 演变图…", self)
        act_export_mip.triggered.connect(self._export_mip)
        file_menu.addAction(act_export_mip)

        act_export_plot = QAction("导出特征折线…", self)
        act_export_plot.triggered.connect(self._export_plot)
        file_menu.addAction(act_export_plot)

        file_menu.addSeparator()
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        help_menu = self.menuBar().addMenu("帮助")
        act_about = QAction("关于", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    def _session(self):
        if not self.current_patient:
            return None
        return self.timepoints.current_session(self.current_patient)

    def _load_vol(self, patient_id: str, study_date: str, baseline_date: Optional[str]):
        key = (patient_id, study_date, baseline_date or "")
        if key not in self._vol_cache:
            assets = self.catalog.get_study(patient_id, study_date)
            if assets is None:
                self._vol_cache[key] = None
            else:
                self._vol_cache[key] = load_volume_set(assets, baseline_date)
        return self._vol_cache[key]

    def _on_patient(self, patient_id: str) -> None:
        if patient_id == self.current_patient:
            return
        self.current_patient = patient_id
        self._vol_cache.clear()
        rec = self.catalog.get_patient(patient_id)
        session = load_session(self.catalog.processed_root, patient_id)
        dates = rec.study_dates() if rec else []
        self.timepoints.set_dates(dates, session)
        feat_csv = self.catalog.processed_root / patient_id / "longitudinal_features.csv"
        self.features.load_csv(feat_csv)
        if dates:
            self._on_study(patient_id, dates[0])
        self._refresh_evolution()
        self.status.showMessage(f"患者 {patient_id}  · {len(dates)} 次检查")

    def _on_study(self, patient_id: str, study_date: str) -> None:
        self.current_patient = patient_id
        self.current_date = study_date
        session = self._session() or load_session(self.catalog.processed_root, patient_id)
        vol = self._load_vol(patient_id, study_date, session.baseline)
        if vol is None:
            self.ortho.set_volumes(None)
            self.status.showMessage(f"{patient_id}/{study_date} 缺少工作 CT 或 PET")
            return
        role = session.role_of(study_date)
        if role:
            vol.role = role
        self.ortho.set_volumes(vol)

    def _on_session_edit(self) -> None:
        if not self.current_patient:
            return
        session = self.timepoints.current_session(self.current_patient)
        rec = self.catalog.get_patient(self.current_patient)
        dates = rec.study_dates() if rec else []
        issues = session.validate(dates)
        if issues:
            self.status.showMessage("时间点冲突: " + "; ".join(issues))
            return
        save_session(self.catalog.processed_root, session)
        if self.current_date:
            self._on_study(self.current_patient, self.current_date)
        self._refresh_evolution()
        self.status.showMessage(f"已保存会话 {self.current_patient}")

    def _refresh_evolution(self) -> None:
        if not self.current_patient:
            self.evolution.clear()
            return
        session = self._session() or load_session(self.catalog.processed_root, self.current_patient)
        items = []
        labels = {"baseline": "Baseline", "interim": "Interim", "end": "End"}
        for role, date in session.dates_in_order():
            vol = self._load_vol(self.current_patient, date, session.baseline)
            if vol is None:
                continue
            vol.role = role
            items.append((f"{labels[role]}  {date}", vol))
        self.evolution.set_volumes(
            items,
            suv_max=float(self.ortho.spin_suv.value()),
            show_native=self.ortho.chk_native.isChecked(),
            show_mapped=self.ortho.chk_mapped.isChecked(),
        )

    def _run_mapping(self) -> None:
        if not self.current_patient:
            return
        session = self.timepoints.current_session(self.current_patient)
        save_session(self.catalog.processed_root, session)
        if not session.can_map():
            QMessageBox.information(self, "映射", "请先指定 baseline 以及至少一个随访时间点。")
            return
        self.timepoints.btn_map.setEnabled(False)
        self._map_worker = MappingWorker(self.catalog, self.current_patient, overwrite=False)
        self._map_worker.progress.connect(self.status.showMessage)
        self._map_worker.failed.connect(self._on_worker_fail)
        self._map_worker.finished_ok.connect(self._on_map_done)
        self._map_worker.start()

    def _on_map_done(self, msg: str) -> None:
        self.timepoints.btn_map.setEnabled(True)
        self._vol_cache.clear()
        self.catalog.refresh()
        self.browser.populate(self.catalog)
        if self.current_patient:
            self.browser.select_patient(self.current_patient)
            if self.current_date:
                self._on_study(self.current_patient, self.current_date)
            self._refresh_evolution()
        self.status.showMessage(msg)

    def _run_features(self) -> None:
        if not self.current_patient:
            return
        session = self.timepoints.current_session(self.current_patient)
        save_session(self.catalog.processed_root, session)
        if not session.assigned():
            QMessageBox.information(self, "特征", "请先指定至少一个时间点。")
            return
        self.timepoints.btn_feat.setEnabled(False)
        self._feat_worker = FeatureWorker(self.catalog, self.current_patient, include_radiomics=True)
        self._feat_worker.progress.connect(self.status.showMessage)
        self._feat_worker.failed.connect(self._on_worker_fail)
        self._feat_worker.finished_ok.connect(self._on_feat_done)
        self._feat_worker.start()

    def _on_feat_done(self, csv_path: str) -> None:
        self.timepoints.btn_feat.setEnabled(True)
        self.features.load_csv(Path(csv_path))
        self.status.showMessage(f"特征已写入 {csv_path}")

    def _on_worker_fail(self, msg: str) -> None:
        self.timepoints.btn_map.setEnabled(True)
        self.timepoints.btn_feat.setEnabled(True)
        QMessageBox.warning(self, "任务失败", msg)
        self.status.showMessage(msg)

    def _refresh(self) -> None:
        self.catalog.refresh()
        self.browser.populate(self.catalog)
        self.status.showMessage("目录已刷新")

    def _export_csv(self) -> None:
        if not self.current_patient:
            return
        src = self.catalog.processed_root / self.current_patient / "longitudinal_features.csv"
        if not src.is_file():
            QMessageBox.information(self, "导出", "尚无特征 CSV，请先计算特征。")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "导出特征 CSV", f"{self.current_patient}_features.csv", "CSV (*.csv)"
        )
        if dest:
            shutil.copy2(src, dest)
            self.status.showMessage(f"已导出 {dest}")

    def _export_mip(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(
            self, "导出 MIP", f"{self.current_patient or 'mip'}_evolution.png", "PNG (*.png)"
        )
        if dest:
            self.evolution.save_png(dest)
            self.status.showMessage(f"已导出 {dest}")

    def _export_plot(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(
            self, "导出折线", f"{self.current_patient or 'features'}_plot.png", "PNG (*.png)"
        )
        if dest:
            self.features.save_plot_png(Path(dest))
            self.status.showMessage(f"已导出 {dest}")

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "关于",
            "DLBCL 纵向 PET/CT 分析软件\n"
            "指定基线/中期/末期 → CT–CT 仿射映射基线病灶床\n"
            "→ SUVmax / MTV / TLG 与影像组学。\n"
            "红 = 本底 mask，青 = 映射的基线病灶床。",
        )
