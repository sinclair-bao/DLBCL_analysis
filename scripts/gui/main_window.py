#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""纵向 PET/CT 分析软件主窗口。"""

from __future__ import annotations

import shutil
import csv
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from weakref import ref

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

_SCRIPTS = Path(__file__).resolve().parents[1]
_LONG = _SCRIPTS / "longitudinal"
_COMMON = _SCRIPTS / "common"
for _p in (str(_LONG), str(_COMMON), str(_SCRIPTS / "gui"), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from catalog import DataCatalog, resolve_data_directory  # noqa: E402
from quality import (archive_existing_mask, fingerprint, mapping_status, mask_review_status,
                     record_lesion_link, record_mask_version, review_mapping, review_mask)  # noqa: E402
from session import LongitudinalSession, load_session, save_session  # noqa: E402

from busy import busy_progress, make_progress  # noqa: E402
from edit_panel import EditPanel  # noqa: E402
from evolution_strip import EvolutionStrip  # noqa: E402
from feature_panel import FeaturePanel  # noqa: E402
from mask_ops import lesion_stats, morph_labels, next_label, relabel_by_volume  # noqa: E402
from ortho_viewer import OrthoViewer  # noqa: E402
from patient_browser import PatientBrowser  # noqa: E402
from review_dialog import QualityReviewDialog  # noqa: E402
from timepoint_panel import TimepointPanel  # noqa: E402
from volume_io import build_mip_data, load_mask_from_path, load_volume_set, save_edited_mask  # noqa: E402
from workers import FeatureWorker, MappingWorker, MipPrefetchWorker, SegmentWorker  # noqa: E402

_VOL_CACHE_CAP = 6

STYLESHEET = """
QMainWindow, QWidget { background: #161616; color: #e8e8e8; }
QTreeWidget { background: #1e1e1e; alternate-background-color: #252525; border: 1px solid #333; }
QGroupBox { border: 1px solid #3a3a3a; margin-top: 10px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #b8d4ff; }
QPushButton { background: #2b4c7e; border: none; padding: 6px 10px; border-radius: 3px; }
QPushButton:hover { background: #3a64a8; }
QPushButton:disabled { background: #333; color: #777; }
QComboBox, QDoubleSpinBox, QSpinBox { background: #222; border: 1px solid #444; padding: 2px 6px; }
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
        manifest_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("纵向 PET/CT 分析与质控")
        self.setStyleSheet(STYLESHEET)
        self.setMinimumSize(900, 620)

        self.catalog = DataCatalog(interim_root, processed_root, manifest_path)
        self.current_patient: Optional[str] = None
        self.current_date: Optional[str] = None
        self._map_worker: Optional[MappingWorker] = None
        self._feat_worker: Optional[FeatureWorker] = None
        self._seg_worker: Optional[SegmentWorker] = None
        self._vol_cache: OrderedDict[tuple[str, str, str], object] = OrderedDict()
        self._mip_cache: OrderedDict[tuple[str, str, str], tuple[object, int, object]] = OrderedDict()
        self._prefetch_worker: Optional[MipPrefetchWorker] = None
        self._prefetch_pending: list[tuple[str, str, str]] = []
        self._prefetch_failed: set[tuple[str, str, str]] = set()
        self._prefetch_generation = 0
        self._browser_width = 0
        self._busy_dlg: Optional[QProgressDialog] = None

        self.browser = PatientBrowser()
        self.timepoints = TimepointPanel()
        self.edit_panel = EditPanel()
        self.ortho = OrthoViewer()
        self.evolution = EvolutionStrip()
        self.features = FeaturePanel()

        left = QWidget()
        self._left_panel = left
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(4, 4, 4, 4)
        left_l.addWidget(QLabel("患者"))
        left_l.addWidget(self.browser, 1)

        right_inner = QWidget()
        right_l = QVBoxLayout(right_inner)
        right_l.setContentsMargins(4, 4, 4, 4)
        right_l.addWidget(self.timepoints)
        right_l.addWidget(self.edit_panel)
        right_l.addWidget(self.features, 1)

        right = QScrollArea()
        self._right_panel = right
        right.setWidgetResizable(True)
        right.setFrameShape(QFrame.Shape.NoFrame)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right.setMinimumWidth(320)
        right.setWidget(right_inner)

        self._center_split = QSplitter(Qt.Orientation.Vertical)
        self._center_split.addWidget(self.ortho)
        self._center_split.addWidget(self.evolution)
        self._center_split.setStretchFactor(0, 3)
        self._center_split.setStretchFactor(1, 1)

        self._main_split = QSplitter(Qt.Orientation.Horizontal)
        self._main_split.addWidget(left)
        self._main_split.addWidget(self._center_split)
        self._main_split.addWidget(right)
        self._main_split.setStretchFactor(0, 1)
        self._main_split.setStretchFactor(1, 4)
        self._main_split.setStretchFactor(2, 2)

        wrapper = QWidget()
        wrap_l = QHBoxLayout(wrapper)
        wrap_l.setContentsMargins(0, 0, 0, 0)
        wrap_l.addWidget(self._main_split)
        self.setCentralWidget(wrapper)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._build_toolbar()
        self._build_menu()
        self._fit_to_screen()

        self.browser.patient_selected.connect(self._on_patient)
        self.browser.study_selected.connect(self._on_study)
        self.timepoints.session_changed.connect(self._on_session_edit)
        self.timepoints.map_requested.connect(self._run_mapping)
        self.timepoints.features_requested.connect(self._run_features)
        self.timepoints.review_mapping_requested.connect(self._review_mapping)
        self.ortho.chk_native.toggled.connect(self._refresh_evolution)
        self.ortho.chk_mapped.toggled.connect(self._refresh_evolution)
        self.ortho.combo_cmap.currentIndexChanged.connect(self._refresh_evolution)
        self.ortho.mask_changed.connect(self._refresh_lesion_table)
        self.ortho.cursor_changed.connect(self._sync_mip_cursor)
        self.ortho.role_requested.connect(self._on_role_view)
        self.edit_panel.segment_requested.connect(self._on_segment)
        self.edit_panel.morph_requested.connect(self._on_morph)
        self.edit_panel.relabel_requested.connect(self._on_relabel)
        self.edit_panel.undo_requested.connect(self.ortho.undo)
        self.edit_panel.save_requested.connect(self._save_edited)
        self.edit_panel.review_requested.connect(self._review_mask)
        self.edit_panel.link_requested.connect(self._link_lesion)
        self.edit_panel.lesion_selected.connect(self._on_lesion_selected)

        self.browser.populate(self.catalog)
        n = len(self.catalog.patient_ids())
        self.status.showMessage(f"已索引 {n} 名患者")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        act_open_data = QAction("打开数据目录…", self)
        act_open_data.setShortcut(QKeySequence.StandardKey.Open)
        act_open_data.triggered.connect(self._open_data_directory)
        file_menu.addAction(act_open_data)

        act_import_manifest = QAction("导入清单…", self)
        act_import_manifest.triggered.connect(self._import_manifest)
        file_menu.addAction(act_import_manifest)

        act_refresh = QAction("刷新目录", self)
        act_refresh.triggered.connect(self._refresh)
        file_menu.addAction(act_refresh)

        file_menu.addSeparator()

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

        view_menu = self.menuBar().addMenu("视图")
        self.act_show_browser = QAction("显示患者列表", self)
        self.act_show_browser.setCheckable(True)
        self.act_show_browser.setChecked(True)
        self.act_show_browser.setShortcut(QKeySequence("F2"))
        self.act_show_browser.toggled.connect(self._set_browser_visible)
        view_menu.addAction(self.act_show_browser)

        self.act_auto_hide = QAction("选择检查后自动隐藏患者列表", self)
        self.act_auto_hide.setCheckable(True)
        self.act_auto_hide.setChecked(True)
        view_menu.addAction(self.act_auto_hide)

        help_menu = self.menuBar().addMenu("帮助")
        act_about = QAction("关于", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("视图")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.btn_patients = QPushButton("患者")
        self.btn_patients.setCheckable(True)
        self.btn_patients.setChecked(True)
        self.btn_patients.setToolTip("显示 / 隐藏患者列表（F2）")
        self.btn_patients.clicked.connect(self._set_browser_visible)
        tb.addWidget(self.btn_patients)

    def _set_browser_visible(self, visible: bool) -> None:
        sizes = self._main_split.sizes()
        total = max(sum(sizes), 1)
        currently = sizes[0] > 12
        if bool(visible) == currently:
            self.act_show_browser.blockSignals(True)
            self.act_show_browser.setChecked(bool(visible))
            self.act_show_browser.blockSignals(False)
            self.btn_patients.blockSignals(True)
            self.btn_patients.setChecked(bool(visible))
            self.btn_patients.blockSignals(False)
            return
        if visible:
            left = self._browser_width or max(int(total * 0.18), 180)
            rest = max(total - left, 1)
            mid = int(rest * 0.70)
            right = rest - mid
            self._main_split.setSizes([left, mid, right])
        else:
            if sizes[0] > 12:
                self._browser_width = sizes[0]
            self._main_split.setSizes([0, sizes[0] + sizes[1], sizes[2]])
        self.act_show_browser.blockSignals(True)
        self.act_show_browser.setChecked(bool(visible))
        self.act_show_browser.blockSignals(False)
        self.btn_patients.blockSignals(True)
        self.btn_patients.setChecked(bool(visible))
        self.btn_patients.blockSignals(False)

    def _fit_to_screen(self) -> None:
        """按当前显示器可用区域缩放窗口，并设置分栏比例。"""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1600, 980)
            w, h = 1600, 980
        else:
            geo = screen.availableGeometry()
            if geo.height() < 900 or geo.width() < 1400:
                self.setGeometry(geo)
                self.showMaximized()
                w, h = geo.width(), geo.height()
            else:
                w = min(geo.width(), max(int(geo.width() * 0.94), 1200))
                h = min(geo.height(), max(int(geo.height() * 0.94), 720))
                self.resize(w, h)
                self.move(
                    geo.x() + (geo.width() - w) // 2,
                    geo.y() + (geo.height() - h) // 2,
                )
        right_width = max(320, min(420, int(w * 0.28)))
        left_width = 180 if w >= 1200 else 140
        self._main_split.setSizes([left_width, max(w - left_width - right_width, 1), right_width])
        self._center_split.setSizes([int(h * 0.78), int(h * 0.22)])

    def _session(self):
        if not self.current_patient:
            return None
        return self.timepoints.current_session(self.current_patient)

    def _show_progress(self, title: str, label: str) -> None:
        self._close_progress()
        self._busy_dlg = make_progress(self, title, label)
        self.status.showMessage(label)

    def _update_progress(self, label: str) -> None:
        self.status.showMessage(label)
        if self._busy_dlg is not None:
            self._busy_dlg.setLabelText(label)
            QApplication.processEvents()

    def _close_progress(self) -> None:
        if self._busy_dlg is not None:
            self._busy_dlg.close()
            self._busy_dlg = None

    def _set_work_busy(self, busy: bool) -> None:
        self.edit_panel.set_busy(busy)
        for button in (self.timepoints.btn_map, self.timepoints.btn_feat,
                       self.timepoints.btn_review_map):
            button.setEnabled(not busy)
        if busy:
            self.ortho.chk_edit.setChecked(False)
        self.ortho.chk_edit.setEnabled(not busy)

    def _load_vol(self, patient_id: str, study_date: str, baseline_date: Optional[str]):
        key = (patient_id, study_date, baseline_date or "")
        if key in self._vol_cache:
            self._vol_cache.move_to_end(key)
            return self._vol_cache[key]
        assets = self.catalog.get_study(patient_id, study_date)
        vol = None if assets is None else load_volume_set(assets, baseline_date)
        self._vol_cache[key] = vol
        self._vol_cache.move_to_end(key)
        self._evict_vol_cache()
        return vol

    def _evict_vol_cache(self) -> None:
        while len(self._vol_cache) > _VOL_CACHE_CAP:
            victim = None
            for key, cached in self._vol_cache.items():
                if cached is None or not getattr(cached, "dirty", False):
                    victim = key
                    break
            if victim is None:
                break
            del self._vol_cache[victim]

    def _drop_cached(self, *, patient_id: str, study_date: Optional[str] = None) -> None:
        self._prefetch_generation += 1
        self._prefetch_pending.clear()
        self._prefetch_failed.clear()
        drop = [
            key
            for key in self._vol_cache
            if key[0] == patient_id and (study_date is None or key[1] == study_date)
        ]
        for key in drop:
            del self._vol_cache[key]
        drop_mips = [key for key in self._mip_cache
                     if key[0] == patient_id and (study_date is None or key[1] == study_date)]
        for key in drop_mips:
            del self._mip_cache[key]

    def _on_patient(self, patient_id: str) -> None:
        if patient_id == self.current_patient:
            return
        if self.current_patient and not self._confirm_navigation():
            self.browser.select_patient(self.current_patient)
            return
        self._prefetch_generation += 1
        self._prefetch_pending.clear()
        self._prefetch_failed.clear()
        self.catalog.refresh()
        self.current_patient = patient_id
        rec = self.catalog.get_patient(patient_id)
        session = load_session(self.catalog.processed_root, patient_id)
        dates = rec.study_dates() if rec else []
        self.timepoints.set_dates(dates, session)
        feat_csv = self.catalog.processed_root / patient_id / "longitudinal_features.csv"
        if self._feature_csv_current(feat_csv, patient_id, session):
            self.features.load_csv(feat_csv)
        else:
            self.features.clear("历史特征缺少质控或输入已变化；请重新计算")
        if dates:
            self._on_study(patient_id, dates[0])
        else:
            self._refresh_evolution()
        self.status.showMessage(f"患者 {patient_id}  · {len(dates)} 次检查")

    def _feature_csv_current(self, csv_path: Path, patient_id: str, session) -> bool:
        if not csv_path.is_file():
            return False
        try:
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                return False
            represented = {row.get("study_date") for row in rows
                           if row.get("roi_type") == "native_lesion"}
            if represented != set(session.assigned().values()):
                return False
            baseline = self.catalog.get_study(patient_id, session.baseline) if session.baseline else None
            for row in rows:
                if session.role_of(row.get("study_date", "")) != row.get("role"):
                    return False
                assets = self.catalog.get_study(patient_id, row["study_date"])
                if assets is None or assets.pet_units != row.get("pet_units") or assets.tracer != row.get("tracer"):
                    return False
                pet_path = assets.working_pet
                ct_path = assets.working_ct
                if (pet_path is None or str(pet_path) != row.get("pet_path") or
                        fingerprint(pet_path)["sha256"] != row.get("pet_sha256") or
                        (str(ct_path) if ct_path else "") != row.get("ct_path") or
                        (ct_path is not None and fingerprint(ct_path)["sha256"] != row.get("ct_sha256"))):
                    return False
                mask = Path(row["mask_path"])
                if (row.get("feature_protocol_version") != "pet_suv_union_v1"
                        or not mask.is_file()
                        or fingerprint(mask)["sha256"] != row.get("mask_sha256")):
                    return False
                if row.get("roi_type", "").startswith("native"):
                    if mask_review_status(mask) != "approved":
                        return False
                elif row.get("roi_type", "").startswith("baseline_mapped"):
                    followup = self.catalog.get_study(patient_id, row["study_date"])
                    if baseline is None or followup is None or mapping_status(baseline, followup) != "approved":
                        return False
            return True
        except (OSError, KeyError, ValueError):
            return False

    def _on_study(self, patient_id: str, study_date: str) -> None:
        if patient_id != self.current_patient:
            return
        if self.current_date and study_date != self.current_date and not self._confirm_navigation():
            self.browser.select_study(patient_id, self.current_date)
            return
        self.current_patient = patient_id
        self.current_date = study_date
        session = self._session() or load_session(self.catalog.processed_root, patient_id)
        try:
            vol = self._load_vol(patient_id, study_date, session.baseline)
        except Exception as exc:  # image/header errors must not terminate the Qt event loop
            self.ortho.set_volumes(None)
            self.edit_panel.set_lesions([])
            self.status.showMessage(f"{patient_id}/{study_date} 载入失败：{exc}")
            QMessageBox.warning(self, "无法载入检查", f"{patient_id}/{study_date}\n{exc}")
            return
        if vol is None:
            self.ortho.set_volumes(None)
            self.edit_panel.set_lesions([])
            self.status.showMessage(f"{patient_id}/{study_date} 缺少工作 CT 或 PET")
            return
        role = session.role_of(study_date)
        if role:
            vol.role = role
        self.ortho.set_volumes(vol)
        self._refresh_lesion_table()
        self._sync_role_buttons()
        if self.act_auto_hide.isChecked():
            self._set_browser_visible(False)
        self._refresh_evolution()
        assets = self.catalog.get_study(patient_id, study_date)
        if assets is not None and assets.working_lesion is None:
            self.status.showMessage(
                f"{patient_id}/{study_date} 尚无病灶 mask，可用 AutoPET / 阈值 / 空白手动分割"
            )

    def _on_session_edit(self) -> None:
        if not self.current_patient:
            return
        if not self._confirm_navigation():
            self.timepoints.set_dates(
                self.catalog.get_patient(self.current_patient).study_dates(),
                load_session(self.catalog.processed_root, self.current_patient))
            return
        session = self.timepoints.current_session(self.current_patient)
        rec = self.catalog.get_patient(self.current_patient)
        dates = rec.study_dates() if rec else []
        issues = session.validate(dates)
        if issues:
            self.status.showMessage("时间点冲突: " + "; ".join(issues))
            return
        previous = load_session(self.catalog.processed_root, self.current_patient)
        save_session(self.catalog.processed_root, session)
        self.features.clear("时间点已更改；请重新计算特征")
        if previous.baseline != session.baseline:
            self._drop_cached(patient_id=self.current_patient)
        if self.current_date and previous.baseline != session.baseline:
            self._on_study(self.current_patient, self.current_date)
        else:
            self._prefetch_generation += 1
            self._prefetch_pending.clear()
            if self.ortho.volumes() is not None:
                self.ortho.volumes().role = session.role_of(self.current_date) or ""
            self._sync_role_buttons()
            self._refresh_evolution()
        self.status.showMessage(f"已保存会话 {self.current_patient}")

    def _refresh_evolution(self) -> None:
        if not self.current_patient:
            self.evolution.clear()
            return
        session = self._session() or load_session(self.catalog.processed_root, self.current_patient)
        items = []
        pending = []
        labels = {"baseline": "Baseline", "interim": "Interim", "end": "End"}
        display_dates = session.dates_in_order()
        if self.current_date and self.current_date not in {date for _, date in display_dates}:
            display_dates.append(("current", self.current_date))
        for role, date in display_dates:
            key = (self.current_patient, date, session.baseline or "")
            assets = self.catalog.get_study(self.current_patient, date)
            vol = self._vol_cache.get(key)
            cached = self._mip_cache.get(key)
            if vol is not None:
                if cached is None or cached[0] is None or cached[0]() is not vol or (
                        cached[1] != vol.mask_revision):
                    mip = build_mip_data(vol)
                    self._mip_cache[key] = (ref(vol), vol.mask_revision, mip)
                else:
                    mip = cached[2]
            else:
                mip = cached[2] if cached is not None else None
                if mip is None and key not in self._prefetch_failed and assets is not None and (
                        assets.working_ct is not None and assets.working_pet is not None):
                    pending.append(key)
            if mip is not None:
                mip.role = role if role != "current" else ""
            title = f"{labels.get(role, '当前检查' if role == 'current' else 'Follow-up')}  {date}"
            if mip is None:
                if key in self._prefetch_failed:
                    title += "  · 载入失败"
                elif assets is None or assets.working_ct is None or assets.working_pet is None:
                    title += "  · 缺少 CT/PET"
                else:
                    title += "  · 载入中"
            items.append((title, mip))
        while len(self._mip_cache) > 32:
            self._mip_cache.popitem(last=False)
        self.evolution.set_volumes(
            items,
            suv_max=float(self.ortho.spin_suv.value()),
            show_native=self.ortho.chk_native.isChecked(),
            show_mapped=self.ortho.chk_mapped.isChecked(),
            highlight_role=session.role_of(self.current_date) if self.current_date else None,
            highlight_date=self.current_date,
            pet_cmap=self.ortho.pet_cmap(),
        )
        self._sync_mip_cursor(*self.ortho.ijk(),
                              self.ortho.volumes() is not None and self.ortho.chk_crosshair.isChecked())
        self._prefetch_pending = pending
        self._start_next_prefetch()

    def _sync_mip_cursor(self, i: int, j: int, k: int, visible: bool) -> None:
        self.evolution.set_cursor(self.current_date, i, j, k, visible)

    def _start_next_prefetch(self) -> None:
        if self._prefetch_worker is not None and self._prefetch_worker.isRunning():
            return
        while self._prefetch_pending:
            key = self._prefetch_pending.pop(0)
            if key in self._mip_cache or key in self._prefetch_failed or key[0] != self.current_patient:
                continue
            assets = self.catalog.get_study(key[0], key[1])
            if assets is None:
                continue
            keep_volume = sum(k[0] == key[0] and v is not None
                              for k, v in self._vol_cache.items()) < 3
            worker = MipPrefetchWorker(assets, key[2] or None,
                                       self._prefetch_generation, keep_volume)
            self._prefetch_worker = worker
            worker.loaded.connect(self._on_prefetch_loaded)
            worker.finished.connect(self._on_prefetch_finished)
            worker.start()
            return

    def _on_prefetch_loaded(self, result: tuple) -> None:
        generation, patient_id, date, baseline_date, mip, vol, error = result
        if generation != self._prefetch_generation or patient_id != self.current_patient:
            return
        key = (patient_id, date, baseline_date)
        if error or mip is None:
            self._prefetch_failed.add(key)
        else:
            self._mip_cache[key] = (ref(vol) if vol is not None else None,
                                    vol.mask_revision if vol is not None else 0, mip)
            if vol is not None and key not in self._vol_cache:
                self._vol_cache[key] = vol
                self._vol_cache.move_to_end(key)
                self._evict_vol_cache()
        self._refresh_evolution()

    def _on_prefetch_finished(self) -> None:
        worker = self.sender()
        if worker is self._prefetch_worker:
            self._prefetch_worker = None
        if worker is not None:
            worker.deleteLater()
        self._start_next_prefetch()

    def _sync_role_buttons(self) -> None:
        session = self._session()
        if session is None and self.current_patient:
            session = load_session(self.catalog.processed_root, self.current_patient)
        if session is None:
            self.ortho.set_role_buttons(
                {"baseline": False, "interim": False, "end": False}, None
            )
            return
        current = session.role_of(self.current_date) if self.current_date else None
        self.ortho.set_role_buttons(
            {
                "baseline": bool(session.baseline),
                "interim": bool(session.interim),
                "end": bool(session.end),
            },
            current,
        )

    def _on_role_view(self, role: str) -> None:
        if not self.current_patient:
            return
        session = self._session() or load_session(self.catalog.processed_root, self.current_patient)
        date = getattr(session, role, None)
        if not date:
            return
        self._on_study(self.current_patient, date)

    def _dirty_volumes(self) -> list:
        seen: set[int] = set()
        out = []
        vol = self.ortho.volumes()
        if vol is not None and vol.dirty:
            out.append(vol)
            seen.add(id(vol))
        pid = self.current_patient
        for cached in self._vol_cache.values():
            if (
                cached is not None
                and cached.dirty
                and id(cached) not in seen
                and (not pid or cached.patient_id == pid)
            ):
                out.append(cached)
                seen.add(id(cached))
        return out

    def _confirm_navigation(self) -> bool:
        if any(worker is not None and worker.isRunning()
               for worker in (self._map_worker, self._feat_worker, self._seg_worker)):
            QMessageBox.warning(self, "任务进行中", "请等待当前后台任务完成后再切换检查。")
            return False
        dirty = self._dirty_volumes()
        if not dirty:
            return True
        answer = QMessageBox.question(
            self, "未保存的 mask", "存在未保存的 mask，是否保存后继续？",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            for vol in dirty:
                if not self._write_edited_mask(vol):
                    return False
            self.catalog.refresh()
        else:
            for vol in dirty:
                self._drop_cached(patient_id=vol.patient_id, study_date=vol.study_date)
        return True

    def closeEvent(self, event) -> None:
        if self._confirm_navigation():
            self._prefetch_generation += 1
            self._prefetch_pending.clear()
            if self._prefetch_worker is not None:
                self._prefetch_worker.wait()
            event.accept()
        else:
            event.ignore()

    def _write_edited_mask(self, vol) -> bool:
        assets = self.catalog.get_study(vol.patient_id, vol.study_date)
        if assets is None:
            return False
        out_path = assets.edited_mask_path()
        current_source = assets.working_lesion
        if (str(current_source) if current_source else None) != vol.native_source_path or (
                fingerprint(current_source)["sha256"] if current_source else None
                ) != vol.native_source_sha256:
            QMessageBox.warning(self, "版本冲突", "正在编辑的源 mask 已改变；请刷新检查后重新编辑。")
            return False
        current_sha = fingerprint(out_path)["sha256"] if out_path.is_file() else None
        if current_sha != vol.edited_sha256:
            QMessageBox.warning(self, "版本冲突", "磁盘上的 edited mask 已被其他操作修改；请刷新检查后再编辑。")
            return False
        archive_existing_mask(out_path)
        save_edited_mask(vol, out_path)
        record_mask_version(out_path, assets.lesion_auto)
        vol.edited_sha256 = fingerprint(out_path)["sha256"]
        vol.native_source_path = str(out_path)
        vol.native_source_sha256 = vol.edited_sha256
        vol.dirty = False
        return True

    def _run_mapping(self) -> None:
        if not self.current_patient:
            return
        session = self.timepoints.current_session(self.current_patient)
        rec = self.catalog.get_patient(self.current_patient)
        dates = rec.study_dates() if rec else []
        issues = session.validate(dates)
        if issues:
            QMessageBox.warning(self, "映射", "时间点无效：\n" + "\n".join(issues))
            return
        if not session.can_map():
            QMessageBox.information(self, "映射", "请先指定 baseline 以及至少一个随访时间点。")
            return
        dirty = self._dirty_volumes()
        if dirty:
            ans = QMessageBox.question(
                self,
                "映射",
                "有未保存的 mask 调整。映射使用磁盘上的基线 mask，是否先保存再映射？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
            for vol in dirty:
                if not self._write_edited_mask(vol):
                    return
            self.catalog.refresh()
        bl = self.catalog.get_study(self.current_patient, session.baseline)
        missing: list[str] = []
        if bl is None:
            missing.append(f"基线检查 {session.baseline}")
        else:
            if bl.working_ct is None:
                missing.append("基线 2 mm CT")
            if bl.working_lesion is None:
                missing.append("基线病灶 mask（请先分割或载入）")
        for role, fu_date in session.followups():
            fu = self.catalog.get_study(self.current_patient, fu_date)
            if fu is None:
                missing.append(f"{role} 检查 {fu_date}")
            elif fu.working_ct is None:
                missing.append(f"{role} 2 mm CT")
        if missing:
            QMessageBox.warning(self, "映射", "无法开始映射：\n" + "\n".join(missing))
            return
        save_session(self.catalog.processed_root, session)
        self._set_work_busy(True)
        self._show_progress("映射", "正在配准 CT（约数分钟，请勿关闭）…")
        self._map_worker = MappingWorker(self.catalog, self.current_patient, overwrite=True)
        self._map_worker.progress.connect(self._update_progress)
        self._map_worker.failed.connect(self._on_worker_fail)
        self._map_worker.finished_ok.connect(self._on_map_done)
        self._map_worker.start()

    def _on_map_done(self, msg: str) -> None:
        self._close_progress()
        self._set_work_busy(False)
        if self.current_patient:
            self._drop_cached(patient_id=self.current_patient)
        self.catalog.refresh()
        self.browser.populate(self.catalog)
        hint = "  已切换到随访，青色 = 基线病灶床"
        if self.current_patient:
            self.browser.select_patient(self.current_patient)
            session = self._session() or load_session(
                self.catalog.processed_root, self.current_patient
            )
            assigned = session.assigned()
            followups = session.followups()
            target = followups[0][1] if followups else self.current_date
            self.ortho.chk_mapped.setChecked(True)
            if target:
                self._on_study(self.current_patient, target)
            self._refresh_evolution()
        self.status.showMessage(msg + hint + "；请核对 CT 棋盘格并记录质控")
        self.features.clear("映射已更新；请完成质控并重新计算特征")
        if "失败" in msg:
            QMessageBox.information(self, "映射部分完成", msg)

    def _run_features(self) -> None:
        if not self.current_patient:
            return
        if self._dirty_volumes():
            QMessageBox.warning(self, "特征", "存在未保存的 mask，请先保存并完成质控。")
            return
        session = self.timepoints.current_session(self.current_patient)
        save_session(self.catalog.processed_root, session)
        if not session.assigned():
            QMessageBox.information(self, "特征", "请先指定至少一个时间点。")
            return
        self._set_work_busy(True)
        self._show_progress("特征", "正在提取代谢 / 组学特征…")
        self._feat_worker = FeatureWorker(
            self.catalog, self.current_patient, include_radiomics=True,
            per_lesion=self.timepoints.chk_per_lesion.isChecked())
        self._feat_worker.progress.connect(self._update_progress)
        self._feat_worker.failed.connect(self._on_worker_fail)
        self._feat_worker.finished_ok.connect(self._on_feat_done)
        self._feat_worker.start()

    def _on_feat_done(self, csv_path: str) -> None:
        self._close_progress()
        self._set_work_busy(False)
        self.features.load_csv(Path(csv_path))
        self.status.showMessage(f"特征已写入 {csv_path}")

    def _on_worker_fail(self, msg: str) -> None:
        self._close_progress()
        self._set_work_busy(False)
        QMessageBox.warning(self, "任务失败", msg)
        self.status.showMessage(msg)

    def _voxel_ml(self) -> float:
        import numpy as np

        vol = self.ortho.volumes()
        if vol is None or vol.affine is None:
            return 0.008
        return abs(float(np.linalg.det(vol.affine[:3, :3]))) / 1000.0

    def _refresh_lesion_table(self) -> None:
        vol = self.ortho.volumes()
        if vol is None:
            self.edit_panel.set_lesions([])
            return
        if vol.stats_revision != vol.mask_revision:
            vol.stats_cache = lesion_stats(vol.native, vol.pet, self._voxel_ml())
            vol.stats_revision = vol.mask_revision
        self.edit_panel.set_lesions(vol.stats_cache, selected=self.ortho.current_label)

    def _on_lesion_selected(self, lid: int) -> None:
        vol = self.ortho.volumes()
        if lid > 0:
            self.ortho.current_label = lid
            self.ortho.highlight_label = lid
        else:
            self.ortho.highlight_label = 0
            if vol is not None:
                self.ortho.current_label = next_label(vol.native)
        self.ortho.refresh()

    def _on_segment(self, method: str) -> None:
        if not self.current_patient or not self.current_date:
            return
        vol = self.ortho.volumes()
        if vol is None:
            QMessageBox.information(self, "分割", "请先打开一次检查。")
            return
        if method == "manual":
            import numpy as np

            start = np.zeros(vol.ct.shape, dtype=np.uint16)
            self._open_segment_editor(start)
            return
        if method == "other":
            self._load_other_mask()
            return
        if method == "threshold":
            if vol.pet_units != "SUVbw":
                QMessageBox.warning(self, "SUV 阈值", "当前 PET 单位不是 SUVbw，不能使用 SUV 阈值分割。")
                return
            import numpy as np

            start = np.zeros(vol.ct.shape, dtype=np.uint16)
            self._open_segment_editor(start, require_change=True)
            return
        if self.catalog.manifest_path is not None:
            QMessageBox.information(self, "自动分割", "通用清单模式未配置自动分割适配器；请导入已有 mask 或手动编辑。")
            return
        assets = self.catalog.get_study(self.current_patient, self.current_date)
        missing: list[str] = []
        if assets is None:
            missing.append("检查目录")
        else:
            if assets.ct_iso is None:
                missing.append("preprocessed/CT")
            if assets.pet_iso is None:
                missing.append("preprocessed/PET")
        if missing:
            QMessageBox.warning(
                self,
                "AutoPET",
                "AutoPET 导出需要成对的预处理 CT 与 PET，当前缺少：\n" + "\n".join(missing),
            )
            return
        if assets.lesion_auto is not None or assets.lesion_edited is not None:
            answer = QMessageBox.question(
                self, "重跑 AutoPET",
                "将覆盖当前 AutoPET 模型原稿；人工编辑稿会保留并继续优先显示。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._set_work_busy(True)
        self._show_progress("AutoPET", "正在准备 AutoPET…")
        self._seg_worker = SegmentWorker(
            method, self.current_patient, self.current_date, self.catalog
        )
        self._seg_worker.progress.connect(self._update_progress)
        self._seg_worker.failed.connect(self._on_worker_fail)
        self._seg_worker.finished_ok.connect(self._on_seg_done)
        self._seg_worker.start()

    def _load_other_mask(self) -> None:
        vol = self.ortho.volumes()
        if vol is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "载入已有 mask",
            str(self.catalog.processed_root / self.current_patient / self.current_date / "masks"),
            "NIfTI (*.nii.gz *.nii)",
        )
        if not path:
            return
        try:
            mask = load_mask_from_path(path, vol)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "载入 mask", f"无法重采样到当前 CT 网格：\n{exc}")
            return
        self._open_segment_editor(mask)

    def _open_segment_editor(self, mask=None, *, require_change: bool = False) -> None:
        from dataclasses import replace

        import numpy as np

        from segment_editor import SegmentEditorDialog

        vol = self.ortho.volumes()
        if vol is None:
            return
        native = np.asarray(mask if mask is not None else vol.native, dtype=np.uint16).copy()
        work = replace(vol, native=native, dirty=False, _undo=[])
        assets = self.catalog.get_study(vol.patient_id, vol.study_date)
        try:
            original = (load_mask_from_path(assets.lesion_auto, vol)
                        if assets is not None and assets.lesion_auto is not None else native.copy())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法打开编辑窗", f"模型原稿 mask 无效：\n{exc}")
            return
        dlg = SegmentEditorDialog(
            work,
            self,
            pet_cmap=self.ortho.pet_cmap(),
            render_device=self.ortho.render_device(),
            reference_mask=original,
            require_change=require_change,
        )
        if dlg.exec() != SegmentEditorDialog.DialogCode.Accepted:
            return
        edited = np.array(dlg.edited_mask(), copy=True, dtype=np.uint16)
        if assets is not None and assets.working_lesion is not None and np.array_equal(edited, vol.native):
            self.status.showMessage("mask 未变化，未生成新版本")
            return
        vol.native = edited
        vol.dirty = True
        self.ortho.mark_masks_dirty()
        self.ortho.refresh()
        self._save_edited()
        self._refresh_lesion_table()

    def _on_seg_done(self, msg: str) -> None:
        self._close_progress()
        self._set_work_busy(False)
        self.features.clear("自动分割已更新；请完成质控并重新计算特征")
        if self.current_patient:
            self._drop_cached(patient_id=self.current_patient, study_date=self.current_date)
        self.catalog.refresh()
        self.browser.populate(self.catalog)
        if self.current_patient:
            self.browser.select_patient(self.current_patient)
            if self.current_date:
                self._on_study(self.current_patient, self.current_date)
        self.ortho.chk_native.setChecked(True)
        self.status.showMessage(msg)
        box = QMessageBox(self)
        box.setWindowTitle("AutoPET")
        box.setText("分割已叠加在主界面。若要手动改，可用「其他（载入 mask）」打开编辑窗。")
        adj = box.addButton("手动调整", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("留在主界面", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() == adj:
            self._open_segment_editor()

    def _on_morph(self, op: str) -> None:
        vol = self.ortho.volumes()
        if vol is None:
            return
        radius = int(self.edit_panel.spin_radius.value())
        scope = self.edit_panel.combo_scope.currentData()
        target = self.edit_panel.combo_target.currentData()
        label = int(self.ortho.current_label) if target == "current" else 0
        plane = self.ortho.active_view if scope == "slice" else None
        ijk = self.ortho.ijk() if plane else None
        waiting = plane is None
        if waiting:
            with busy_progress(self, "形态学", "正在三维形态学运算…"):
                QApplication.processEvents()
                new_mask = morph_labels(
                    vol.native, op, radius, label=label, plane=plane, ijk=ijk
                )
        else:
            new_mask = morph_labels(
                vol.native, op, radius, label=label, plane=plane, ijk=ijk
            )
        self.ortho.apply_mask(new_mask)

    def _on_relabel(self) -> None:
        vol = self.ortho.volumes()
        if vol is None:
            return
        self.ortho.apply_mask(relabel_by_volume(vol.native))
        self.ortho.current_label = 1
        self.ortho.highlight_label = 1

    def _save_edited(self) -> None:
        vol = self.ortho.volumes()
        if vol is None or not self.current_patient or not self.current_date:
            return
        if not self._write_edited_mask(vol):
            return
        self.catalog.refresh()
        self.browser.populate(self.catalog)
        if self.current_patient:
            self.browser.select_patient(self.current_patient)
        assets = self.catalog.get_study(self.current_patient, self.current_date)
        name = assets.edited_mask_path().name if assets is not None else "lesion_edited.nii.gz"
        self.status.showMessage(f"已保存 {name}（未覆盖 nnU-Net 原 mask）")
        self.features.clear("mask 已更新；请完成质控并重新计算特征")

    def _review_mask(self) -> None:
        vol = self.ortho.volumes()
        if vol is None or not self.current_patient or not self.current_date:
            return
        if vol.dirty:
            QMessageBox.warning(self, "mask 质控", "请先保存当前修改。")
            return
        self.catalog.refresh()
        assets = self.catalog.get_study(self.current_patient, self.current_date)
        if assets is None or assets.working_lesion is None:
            QMessageBox.warning(self, "mask 质控", "没有可审阅的 mask。")
            return
        if (str(assets.working_lesion) != vol.native_source_path or
                fingerprint(assets.working_lesion)["sha256"] != vol.native_source_sha256):
            self._drop_cached(patient_id=self.current_patient, study_date=self.current_date)
            self._on_study(self.current_patient, self.current_date)
            QMessageBox.information(self, "mask 质控", "磁盘 mask 已变化，界面已重新载入；请重新检查后质控。")
            return
        try:
            original = (load_mask_from_path(assets.lesion_auto, vol)
                        if assets.lesion_auto is not None else vol.native.copy())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "mask 质控", f"模型原稿 mask 无效：\n{exc}")
            return
        dlg = QualityReviewDialog(ct=vol.ct, pet=vol.pet, original_mask=original,
                                  edited_mask=vol.native, parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted and dlg.verdict:
            try:
                review_mask(assets.working_lesion, dlg.reviewer.text(),
                            dlg.reason.toPlainText(), dlg.verdict)
                self.status.showMessage(f"mask 质控状态：{dlg.verdict}")
                self.features.clear("mask 质控已更新；请重新计算特征")
            except Exception as exc:
                QMessageBox.warning(self, "mask 质控", str(exc))

    def _review_mapping(self) -> None:
        import nibabel as nib
        import numpy as np
        from nibabel.processing import resample_from_to

        if not self.current_patient or not self.current_date:
            return
        session = self._session() or load_session(self.catalog.processed_root, self.current_patient)
        if not session.baseline or self.current_date == session.baseline:
            QMessageBox.information(self, "配准质控", "请先打开一项已映射的随访检查。")
            return
        baseline = self.catalog.get_study(self.current_patient, session.baseline)
        followup = self.catalog.get_study(self.current_patient, self.current_date)
        vol = self.ortho.volumes()
        if baseline is None or followup is None or vol is None:
            return
        state = mapping_status(baseline, followup)
        if state in {"stale", "invalid", "unreviewed"}:
            QMessageBox.warning(self, "配准质控", f"映射状态为 {state}，请先重新映射。")
            return
        warped_path = followup.longitudinal_dir() / "baseline_ct_warped.nii.gz"
        if not warped_path.is_file():
            QMessageBox.warning(self, "配准质控", "缺少基线 CT warped 图像。")
            return
        ref = nib.as_closest_canonical(nib.load(str(followup.working_ct)))
        warped = nib.load(str(warped_path))
        if warped.shape != ref.shape or not np.allclose(warped.affine, ref.affine, atol=1e-4):
            warped = resample_from_to(warped, (ref.shape, ref.affine), order=1)
        dlg = QualityReviewDialog(ct=vol.ct,
                                  warped_ct=np.asanyarray(warped.dataobj, dtype=np.float32),
                                  mapped_mask=vol.mapped, parent=self)
        if dlg.exec() == dlg.DialogCode.Accepted and dlg.verdict:
            try:
                review_mapping(baseline, followup, dlg.reviewer.text(),
                               dlg.reason.toPlainText(), dlg.verdict)
                self.status.showMessage(f"配准质控状态：{dlg.verdict}")
                self.features.clear("配准质控已更新；请重新计算特征")
            except Exception as exc:
                QMessageBox.warning(self, "配准质控", str(exc))

    def _link_lesion(self) -> None:
        import nibabel as nib
        import numpy as np

        if not self.current_patient or not self.current_date:
            return
        session = self._session() or load_session(self.catalog.processed_root, self.current_patient)
        if not session.baseline or self.current_date == session.baseline:
            QMessageBox.information(self, "病灶对应", "请在随访检查中选择一个病灶。")
            return
        baseline = self.catalog.get_study(self.current_patient, session.baseline)
        followup = self.catalog.get_study(self.current_patient, self.current_date)
        vol = self.ortho.volumes()
        lid = int(self.ortho.current_label)
        if vol is None or baseline is None or followup is None or followup.working_lesion is None:
            return
        if vol.dirty or not np.any(vol.native == lid):
            QMessageBox.warning(self, "病灶对应", "请先选择一个存在的病灶并保存 mask。")
            return
        number, ok = QInputDialog.getInt(self, "病灶对应", "基线病灶编号（0 = 随访新灶）", 0, 0, 65535)
        if not ok:
            return
        if number and (baseline.working_lesion is None or
                       not np.any(np.asanyarray(nib.load(str(baseline.working_lesion)).dataobj) == number)):
            QMessageBox.warning(self, "病灶对应", f"基线 mask 不含编号 {number}。")
            return
        reviewer, ok = QInputDialog.getText(self, "病灶对应", "审阅者")
        if not ok:
            return
        reason, ok = QInputDialog.getText(self, "病灶对应", "影像对应依据 / 备注")
        if not ok:
            return
        try:
            uid = record_lesion_link(self.catalog.processed_root, baseline, followup,
                                     baseline_label=number or None, followup_label=lid,
                                     reviewer=reviewer, reason=reason)
            self.status.showMessage(f"已记录病灶 UID：{uid}")
        except Exception as exc:
            QMessageBox.warning(self, "病灶对应", str(exc))

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "关于",
            "DLBCL 纵向 PET/CT 分析软件\n"
            "四种分割：AutoPET 校验 CT+PET 后在主界面展示结果；\n"
            "SUV 阈值 / 空白手动 / 载入 mask 打开 CT/PET/融合编辑窗。\n"
            "二维画笔微调 + 膨胀/腐蚀；编号病灶另存为 *_lesion_edited.nii.gz。\n"
            "显示：仅 CT / 仅 PET（灰度）/ PET-CT；轴冠 MIP 标 R/L，矢状标 P/A。\n"
            "十字线与患者列表可隐藏（F2）。\n"
            "「将基线病灶映射到中期/末期」使用 edited 优先的基线 mask。\n"
            "红 = 本底 mask，黄 = 当前选中灶，青 = 映射的基线病灶床。",
        )

    def _switch_catalog(self, catalog: DataCatalog, source: Path) -> None:
        if not self._confirm_navigation():
            return
        self._prefetch_generation += 1
        self._prefetch_pending.clear()
        self._prefetch_failed.clear()
        self._vol_cache.clear()
        self._mip_cache.clear()
        self.current_patient = None
        self.current_date = None
        self.catalog = catalog
        self.ortho.set_volumes(None)
        self.edit_panel.set_lesions([])
        self.evolution.clear()
        self.features.clear()
        self.timepoints.set_dates([], LongitudinalSession(patient_id=""))
        self.browser.populate(catalog)
        self._set_browser_visible(True)
        self.status.showMessage(
            f"已打开 {source} · {len(catalog.patient_ids())} 名患者 · 结果目录 {catalog.processed_root}"
        )

    def _open_data_directory(self) -> None:
        start = self.catalog.interim_root
        if not start.is_dir():
            start = start.parent
        selected = QFileDialog.getExistingDirectory(self, "打开数据目录", str(start))
        if not selected:
            return
        try:
            interim, processed, patient_filter = resolve_data_directory(selected)
            catalog = DataCatalog(interim, processed, patient_filter=patient_filter)
            if not catalog.patient_ids():
                raise ValueError("未发现患者检查目录；请选择 data、interim 或 interim/患者ID 目录。")
            if not any(
                any((study.ct_orig, study.pet_orig, study.ct_iso, study.pet_iso,
                     study.ct_reference, study.pet_aligned))
                for patient in catalog.patients.values() for study in patient.studies
            ):
                raise ValueError("目录中未找到 CT/PET NIfTI；请选择包含患者检查的 interim 目录。")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法打开数据目录", str(exc))
            return
        self._switch_catalog(catalog, Path(selected))

    def _import_manifest(self) -> None:
        start = self.catalog.manifest_path or self.catalog.interim_root.parent
        selected, _ = QFileDialog.getOpenFileName(
            self, "导入检查清单", str(start), "CSV 清单 (*.csv);;所有文件 (*)"
        )
        if not selected:
            return
        manifest = Path(selected).expanduser().resolve()
        processed = (self.catalog.processed_root if manifest == self.catalog.manifest_path
                     else manifest.parent / f"{manifest.stem}_review_outputs")
        try:
            catalog = DataCatalog(self.catalog.interim_root, processed, manifest)
            if not catalog.patient_ids():
                raise ValueError("清单中没有患者检查。")
        except (OSError, ValueError, KeyError, csv.Error) as exc:
            QMessageBox.warning(self, "无法导入清单", str(exc))
            return
        self._switch_catalog(catalog, manifest)

    def _refresh(self) -> None:
        if not self._confirm_navigation():
            return
        patient_id, study_date = self.current_patient, self.current_date
        self._prefetch_generation += 1
        self._prefetch_pending.clear()
        self._prefetch_failed.clear()
        self._vol_cache.clear()
        self._mip_cache.clear()
        self.catalog.refresh()
        self.browser.populate(self.catalog)
        self.current_patient = None
        self.current_date = None
        if patient_id and self.catalog.get_patient(patient_id):
            self._on_patient(patient_id)
            if study_date and study_date != self.current_date and (
                    self.catalog.get_study(patient_id, study_date) is not None):
                self._on_study(patient_id, study_date)
        else:
            self.ortho.set_volumes(None)
            self.evolution.clear()
            self.features.clear()
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
