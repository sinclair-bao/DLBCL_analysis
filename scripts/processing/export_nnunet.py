#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   export_nnunet.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    将 preprocess.py 产出的 preprocessed/{CT,PET} 导出为 nnU-Net 推理所需的
    扁平命名格式，供后续 `nnUNetv2_predict_from_modelfolder` 使用。

@输入 (Input)
    data/interim/<PatientID>/<StudyDate>/preprocessed/CT/*.nii.gz
    data/interim/<PatientID>/<StudyDate>/preprocessed/PET/*_SUVbw.nii.gz
    （或 *_ACT.nii.gz；要求 CT/PET 已对齐到同一网格）

@输出 (Output)
    <export-root>/<PatientID>_<StudyDate>_0000.nii.gz   # CT (HU)
    <export-root>/<PatientID>_<StudyDate>_0001.nii.gz   # PET (SUV)
    默认 export-root = data/nnunet_export/

@关键执行逻辑
    - 发现 Study → 定位参考 CT 与 PET → 校验 shape/affine 一致 →
      优先 hardlink，失败则 copy 到扁平导出目录。
    - 几何不一致时记 error（提示先重跑 preprocess --overwrite），不写出。
    - 增量执行：目标已存在则跳过，除非 overwrite=True。

@用法示例
    python scripts/processing/export_nnunet.py
    python scripts/processing/export_nnunet.py --patient-id 00857723 --overwrite
    python scripts/processing/export_nnunet.py --dry-run -v
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 nibabel，请先安装。") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from pipeline_utils import (  # noqa: E402
    StageResult,
    discover_subject_studies,
    setup_logging,
    summarize,
    write_stage_log_csv,
)

logger = logging.getLogger("export_nnunet")

STAGE_NAME = "export"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "data" / "nnunet_export"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def _find_preprocessed_ct(study_dir: Path) -> Optional[Path]:
    ct_dir = study_dir / "preprocessed" / "CT"
    if not ct_dir.is_dir():
        return None
    files = sorted(ct_dir.glob("*.nii.gz"))
    return files[0] if files else None


def _find_preprocessed_pet(study_dir: Path) -> Optional[Path]:
    pet_dir = study_dir / "preprocessed" / "PET"
    if not pet_dir.is_dir():
        return None
    candidates = sorted(pet_dir.glob("*_SUVbw.nii.gz")) or sorted(pet_dir.glob("*.nii.gz"))
    return candidates[0] if candidates else None


def _geometry_matches(ct_path: Path, pet_path: Path) -> tuple[bool, str]:
    ct = nib.load(str(ct_path))
    pet = nib.load(str(pet_path))
    if ct.shape != pet.shape:
        return False, f"shape 不一致: CT={ct.shape} PET={pet.shape}"
    if not np.allclose(ct.affine, pet.affine, atol=1e-5):
        return False, "affine 不一致"
    return True, f"shape={ct.shape}"


def _link_or_copy(src: Path, dst: Path) -> str:
    """优先 hardlink（省磁盘），失败则 copy。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


class NnuNetExporter:
    """把对齐后的 CT/PET 导出为 nnU-Net `_0000` / `_0001` 扁平命名。"""

    def __init__(
        self,
        interim_root: Path | str,
        export_root: Path | str,
        overwrite: bool = False,
    ) -> None:
        self.interim_root = Path(interim_root)
        self.export_root = Path(export_root)
        self.overwrite = overwrite

    def discover_studies(self) -> list[tuple[str, str, Path]]:
        return discover_subject_studies(self.interim_root)

    def export_study(self, patient_id: str, study_date: str, study_dir: Path) -> StageResult:
        case_id = f"{patient_id}_{study_date}"
        ct_out = self.export_root / f"{case_id}_0000.nii.gz"
        pet_out = self.export_root / f"{case_id}_0001.nii.gz"

        ct_src = _find_preprocessed_ct(study_dir)
        pet_src = _find_preprocessed_pet(study_dir)
        if ct_src is None or pet_src is None:
            missing = []
            if ct_src is None:
                missing.append("CT")
            if pet_src is None:
                missing.append("PET")
            return StageResult(
                STAGE_NAME, patient_id, study_date, "warning", "",
                f"缺少 preprocessed {'/'.join(missing)}，跳过导出。",
            )

        if ct_out.exists() and pet_out.exists() and not self.overwrite:
            return StageResult(
                STAGE_NAME, patient_id, study_date, "skipped",
                str(self.export_root / case_id),
                "输出已存在，overwrite=True 可强制重导。",
            )

        ok, geo_msg = _geometry_matches(ct_src, pet_src)
        if not ok:
            return StageResult(
                STAGE_NAME, patient_id, study_date, "error", "",
                f"CT/PET 几何不一致 ({geo_msg})；请先重跑 preprocess --overwrite。",
            )

        try:
            ct_mode = _link_or_copy(ct_src, ct_out)
            pet_mode = _link_or_copy(pet_src, pet_out)
            return StageResult(
                STAGE_NAME, patient_id, study_date, "ok",
                str(ct_out.parent / case_id),
                f"导出完成 ({geo_msg}; CT={ct_mode}, PET={pet_mode})",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("导出失败: %s", case_id)
            return StageResult(STAGE_NAME, patient_id, study_date, "error", str(ct_out), str(exc))

    def run(
        self,
        dry_run: bool = False,
        patient_id: Optional[str] = None,
        study_date: Optional[str] = None,
    ) -> list[StageResult]:
        studies = self.discover_studies()
        if patient_id:
            studies = [s for s in studies if s[0] == patient_id]
        if study_date:
            studies = [s for s in studies if s[1] == study_date]
        logger.info("共发现 %d 个 (patient, study) 待导出。", len(studies))

        if dry_run:
            for pid, sdate, study_dir in studies:
                case_id = f"{pid}_{sdate}"
                logger.info(
                    "[DRY-RUN] %s CT=%s PET=%s -> %s/{_0000,_0001}.nii.gz",
                    case_id,
                    _find_preprocessed_ct(study_dir),
                    _find_preprocessed_pet(study_dir),
                    self.export_root / case_id,
                )
            return []

        self.export_root.mkdir(parents=True, exist_ok=True)
        results: list[StageResult] = []
        for idx, (pid, sdate, study_dir) in enumerate(studies, start=1):
            logger.info("[%d/%d] 导出 patient=%s study=%s", idx, len(studies), pid, sdate)
            results.append(self.export_study(pid, sdate, study_dir))
        return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 preprocessed CT/PET 导出为 nnU-Net 推理命名格式。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--study-date", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)

    exporter = NnuNetExporter(
        interim_root=args.interim_root,
        export_root=args.export_root,
        overwrite=args.overwrite,
    )
    results = exporter.run(
        dry_run=args.dry_run, patient_id=args.patient_id, study_date=args.study_date,
    )
    if args.dry_run:
        logger.info("Dry-run 完成，未执行任何处理。")
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_stage_log_csv(results, args.log_dir / f"export_{timestamp}.csv")
    counts = summarize(results)
    logger.info("导出完成: %s", counts)
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
