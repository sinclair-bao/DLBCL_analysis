#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""后台线程：跨检查映射与特征提取，避免卡住 UI。"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal

_SCRIPTS = Path(__file__).resolve().parents[1]
_LONG = _SCRIPTS / "longitudinal"
_COMMON = _SCRIPTS / "common"
for _p in (_LONG, _COMMON, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class MappingWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, catalog, patient_id: str, overwrite: bool = False) -> None:
        super().__init__()
        self._catalog = catalog
        self._patient_id = patient_id
        self._overwrite = overwrite

    def run(self) -> None:
        try:
            from interscan_register import InterscanRegistrar

            self.progress.emit(f"开始跨检查映射 {self._patient_id} …")
            registrar = InterscanRegistrar(
                self._catalog, overwrite=self._overwrite
            )
            results = registrar.map_session(self._patient_id)
            errors = [r for r in results if r.status == "error"]
            if errors:
                self.failed.emit("; ".join(r.message for r in errors))
                return
            n_ok = sum(1 for r in results if r.status in ("ok", "skipped"))
            self.finished_ok.emit(f"{self._patient_id}: {n_ok} 个随访映射完成")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class FeatureWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        catalog,
        patient_id: str,
        include_radiomics: bool = True,
    ) -> None:
        super().__init__()
        self._catalog = catalog
        self._patient_id = patient_id
        self._include_radiomics = include_radiomics

    def run(self) -> None:
        try:
            from features import extract_patient_features, write_csv
            from session import load_session

            self.progress.emit(f"提取特征 {self._patient_id} …")
            session = load_session(self._catalog.processed_root, self._patient_id)
            rows = extract_patient_features(
                self._catalog,
                session,
                include_radiomics=self._include_radiomics,
            )
            if not rows:
                self.failed.emit("没有特征行：请先指定时间点并完成基线映射。")
                return
            per_patient = (
                self._catalog.processed_root
                / self._patient_id
                / "longitudinal_features.csv"
            )
            write_csv(rows, per_patient)
            self.finished_ok.emit(str(per_patient))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
