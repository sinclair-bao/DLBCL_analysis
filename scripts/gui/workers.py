#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""后台线程：跨检查映射、特征提取、AutoPET / 阈值分割。"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal

_SCRIPTS = Path(__file__).resolve().parents[1]
_LONG = _SCRIPTS / "longitudinal"
_COMMON = _SCRIPTS / "common"
_PROJECT = _SCRIPTS.parent
for _p in (_LONG, _COMMON, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

AUTOPET_PYTHON = Path("/home/sun/miniconda3/envs/autopet/bin/python")
DA_PYTHON = Path(sys.executable)


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

            self.progress.emit("正在配准 CT（约数分钟，请勿关闭）…")
            registrar = InterscanRegistrar(
                self._catalog, overwrite=self._overwrite
            )
            results = registrar.map_session(self._patient_id)
            if not results:
                self.failed.emit("没有映射任务")
                return
            ok = [r for r in results if r.status in ("ok", "skipped")]
            errors = [r for r in results if r.status == "error"]
            parts: list[str] = []
            if ok:
                parts.append(f"{len(ok)} 个随访映射完成")
            if errors:
                detail = "; ".join(
                    f"{r.study_date}: {r.message}" if r.study_date else r.message
                    for r in errors
                )
                parts.append(f"失败: {detail}")
            msg = f"{self._patient_id}: " + "；".join(parts)
            if ok:
                self.finished_ok.emit(msg)
                return
            self.failed.emit(msg)
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


class SegmentWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        method: str,
        patient_id: str,
        study_date: str,
        catalog,
    ) -> None:
        super().__init__()
        self._method = method
        self._patient_id = patient_id
        self._study_date = study_date
        self._catalog = catalog

    def run(self) -> None:
        try:
            if self._method == "autopet":
                self._run_autopet()
            elif self._method == "threshold":
                self._run_threshold()
            else:
                self.failed.emit(f"未知分割方法: {self._method}")
                return
            self.finished_ok.emit(f"{self._patient_id}/{self._study_date} 分割完成")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _run_cmd(self, cmd: list[str], label: str) -> None:
        self.progress.emit(label)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-1500:]
            raise RuntimeError(f"{label} 失败 (exit={result.returncode})\n{tail}")

    def _run_autopet(self) -> None:
        if not AUTOPET_PYTHON.is_file():
            raise RuntimeError(f"未找到 autopet Python: {AUTOPET_PYTHON}")
        export_root = _PROJECT / "data" / "nnunet_export"
        case = f"{self._patient_id}_{self._study_date}"
        if not (export_root / f"{case}_0000.nii.gz").is_file() or not (
            export_root / f"{case}_0001.nii.gz"
        ).is_file():
            self._run_cmd(
                [
                    str(DA_PYTHON),
                    str(_SCRIPTS / "processing" / "export_nnunet.py"),
                    "--patient-id",
                    self._patient_id,
                    "--study-date",
                    self._study_date,
                ],
                "导出 nnU-Net 格式（CT=_0000，PET=_0001）",
            )
        ct_nii = export_root / f"{case}_0000.nii.gz"
        pet_nii = export_root / f"{case}_0001.nii.gz"
        if not ct_nii.is_file() or not pet_nii.is_file():
            missing = []
            if not ct_nii.is_file():
                missing.append("CT（_0000）")
            if not pet_nii.is_file():
                missing.append("PET（_0001）")
            raise RuntimeError(
                "AutoPET 需要成对的 CT 与 PET 输入，缺少：" + "、".join(missing)
            )
        self._run_cmd(
            [
                str(AUTOPET_PYTHON),
                str(_SCRIPTS / "processing" / "infer_nnunet.py"),
                "--patient-id",
                self._patient_id,
                "--study-date",
                self._study_date,
            ],
            "AutoPET 推理（约 1–2 分钟）",
        )

    def _run_threshold(self) -> None:
        self._run_cmd(
            [
                str(DA_PYTHON),
                str(_SCRIPTS / "processing" / "segmentation.py"),
                "--method",
                "threshold",
                "--patient-id",
                self._patient_id,
                "--study-date",
                self._study_date,
            ],
            "SUV 阈值分割",
        )
        masks = (
            self._catalog.processed_root
            / self._patient_id
            / self._study_date
            / "masks"
        )
        dest = masks / f"{self._patient_id}_{self._study_date}_lesion_edited.nii.gz"
        cands = sorted(masks.glob("*_mask.nii.gz")) if masks.is_dir() else []
        if not cands:
            raise RuntimeError("SUV 阈值分割未写出 *_mask.nii.gz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cands[0], dest)
