#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   preprocess.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    图像预处理阶段：把 scripts/tools/pacs_dicom_to_nifti_suv.py 产出的
    data/interim/<PatientID>/<StudyDate>/{CT,PET}/*.nii.gz 重采样为统一的
    各向同性体素间距（默认 2mm），并保证同一 Study 内 PET 与 CT 落在同一
    空间网格（shape / affine 一致），作为后续 nnU-Net / 特征提取的标准化输入。

@输入 (Input)
    data/interim/<PatientID>/<StudyDate>/CT/*.nii.gz
    data/interim/<PatientID>/<StudyDate>/PET/*_SUVbw.nii.gz（优先）或
    data/interim/<PatientID>/<StudyDate>/PET/*_ACT.nii.gz（SUV 换算前提不
    满足时 pacs_dicom_to_nifti_suv.py 只会产出这一份）

@输出 (Output)
    data/interim/<PatientID>/<StudyDate>/preprocessed/{CT,PET}/<原文件名>
    - CT：先重采样到目标各向同性体素间距
    - PET：再重采样到该 Study 参考 CT 的网格（resample_from_to），保证
      shape 与 affine 与 CT 完全一致（nnU-Net 双通道推理的硬性要求）

@关键执行逻辑
    - discover_studies()：复用 scripts/common/pipeline_utils 里统一的
      <PatientID>/<StudyDate> 两级目录发现逻辑。
    - process_study()：先处理 CT，再以第一份成功写出的 CT 为参考网格，
      把 PET 对齐上去；每个文件独立 try/except。
    - 无 CT 仅有 PET：退化为独立各向同性重采样，并记 warning（后续
      nnU-Net 双通道推理不可用）。
    - 增量执行：目标文件已存在则跳过，除非 overwrite=True。注意：若只
      想强制重算 PET 对齐，需同时 overwrite（否则旧的独立重采样 PET
      会被跳过）。

@用法示例
    python scripts/processing/preprocess.py
    python scripts/processing/preprocess.py --voxel-size 1.5 --overwrite
    python scripts/processing/preprocess.py --dry-run -v
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import nibabel as nib
    import nibabel.processing as nib_processing
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("缺少 nibabel（含 processing 子模块需要 scipy），请先安装。") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from pipeline_utils import (  # noqa: E402
    StageResult,
    discover_subject_studies,
    setup_logging,
    summarize,
    write_stage_log_csv,
)

logger = logging.getLogger("preprocess")

STAGE_NAME = "preprocess"
# 各模态默认插值阶数：CT 用三次样条（更平滑），PET 用线性（避免热点振铃）。
DEFAULT_INTERP_ORDER = {"CT": 3, "PET": 1}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


class ImagePreprocessor:
    """把 data/interim 下的 CT/PET 重采样为统一体素间距，并将 PET 对齐到 CT 网格。"""

    def __init__(
        self,
        interim_root: Path | str,
        voxel_size: float = 2.0,
        overwrite: bool = False,
    ) -> None:
        self.interim_root = Path(interim_root)
        self.voxel_size = float(voxel_size)
        self.overwrite = overwrite

    def discover_studies(self) -> list[tuple[str, str, Path]]:
        return discover_subject_studies(self.interim_root)

    def _find_input_files(self, study_dir: Path, modality: str) -> list[Path]:
        modality_dir = study_dir / modality
        if not modality_dir.is_dir():
            return []
        if modality == "PET":
            # 优先使用 SUVbw 图；若当时 SUV 换算前提不满足，只会存在 ACT 图。
            suv_files = sorted(modality_dir.glob("*_SUVbw.nii.gz"))
            if suv_files:
                return suv_files
            return sorted(modality_dir.glob("*_ACT.nii.gz"))
        return sorted(modality_dir.glob("*.nii.gz"))

    def _resample_isotropic(self, src: Path, dst: Path, order: int) -> nib.Nifti1Image:
        img = nib.load(str(src))
        resampled = nib_processing.resample_to_output(
            img, voxel_sizes=self.voxel_size, order=order
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(resampled, str(dst))
        return resampled

    def _resample_to_reference(
        self, src: Path, dst: Path, reference: nib.Nifti1Image, order: int
    ) -> nib.Nifti1Image:
        """将 src 重采样到 reference 的 shape/affine（保证双通道几何一致）。"""
        img = nib.load(str(src))
        aligned = nib_processing.resample_from_to(
            img, (reference.shape, reference.affine), order=order
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(aligned, str(dst))
        return aligned

    def process_study(self, patient_id: str, study_date: str, study_dir: Path) -> list[StageResult]:
        results: list[StageResult] = []
        ct_files = self._find_input_files(study_dir, "CT")
        pet_files = self._find_input_files(study_dir, "PET")
        ct_output_dir = study_dir / "preprocessed" / "CT"
        pet_output_dir = study_dir / "preprocessed" / "PET"

        # --- CT：各向同性重采样，第一份成功写出的作为 PET 参考网格 ---
        reference_ct: Optional[nib.Nifti1Image] = None
        for src in ct_files:
            dst = ct_output_dir / src.name
            if dst.exists() and not self.overwrite:
                results.append(
                    StageResult(
                        STAGE_NAME, patient_id, study_date, "skipped",
                        str(dst), "输出已存在，overwrite=True 可强制重转。",
                    )
                )
                if reference_ct is None:
                    reference_ct = nib.load(str(dst))
                continue
            try:
                resampled = self._resample_isotropic(src, dst, DEFAULT_INTERP_ORDER["CT"])
                if reference_ct is None:
                    reference_ct = resampled
                results.append(
                    StageResult(
                        STAGE_NAME, patient_id, study_date, "ok",
                        str(dst),
                        f"CT 重采样完成 (voxel_size={self.voxel_size}mm, "
                        f"order={DEFAULT_INTERP_ORDER['CT']}, shape={resampled.shape})",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响其他文件
                logger.exception("CT 预处理失败: %s", src)
                results.append(
                    StageResult(STAGE_NAME, patient_id, study_date, "error", str(dst), str(exc))
                )

        # --- PET：优先对齐到参考 CT；无 CT 时退化为独立各向同性重采样 ---
        for src in pet_files:
            dst = pet_output_dir / src.name
            if dst.exists() and not self.overwrite:
                results.append(
                    StageResult(
                        STAGE_NAME, patient_id, study_date, "skipped",
                        str(dst), "输出已存在，overwrite=True 可强制重转。",
                    )
                )
                continue
            try:
                if reference_ct is not None:
                    aligned = self._resample_to_reference(
                        src, dst, reference_ct, DEFAULT_INTERP_ORDER["PET"]
                    )
                    results.append(
                        StageResult(
                            STAGE_NAME, patient_id, study_date, "ok",
                            str(dst),
                            f"PET 已对齐到 CT 网格 (order={DEFAULT_INTERP_ORDER['PET']}, "
                            f"shape={aligned.shape})",
                        )
                    )
                else:
                    resampled = self._resample_isotropic(src, dst, DEFAULT_INTERP_ORDER["PET"])
                    results.append(
                        StageResult(
                            STAGE_NAME, patient_id, study_date, "warning",
                            str(dst),
                            f"无参考 CT，PET 独立重采样 (voxel_size={self.voxel_size}mm, "
                            f"shape={resampled.shape})；双通道 nnU-Net 推理将不可用。",
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("PET 预处理失败: %s", src)
                results.append(
                    StageResult(STAGE_NAME, patient_id, study_date, "error", str(dst), str(exc))
                )

        return results

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
        logger.info("共发现 %d 个 (patient, study) 待预处理。", len(studies))

        if dry_run:
            for pid, sdate, study_dir in studies:
                for modality in ("CT", "PET"):
                    for src in self._find_input_files(study_dir, modality):
                        logger.info(
                            "[DRY-RUN] %s -> %s",
                            src, study_dir / "preprocessed" / modality / src.name,
                        )
            return []

        all_results: list[StageResult] = []
        for idx, (pid, sdate, study_dir) in enumerate(studies, start=1):
            logger.info("[%d/%d] 预处理 patient=%s study=%s", idx, len(studies), pid, sdate)
            all_results.extend(self.process_study(pid, sdate, study_dir))
        return all_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 data/interim 下的 CT/PET 重采样为统一体素间距，并将 PET 对齐到 CT 网格。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT, help="data/interim 根目录。")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="CSV 日志输出目录。")
    parser.add_argument("--voxel-size", type=float, default=2.0, help="目标各向同性体素间距（mm）。")
    parser.add_argument("--patient-id", default=None, help="仅处理指定 PatientID（测试用）。")
    parser.add_argument("--study-date", default=None, help="仅处理指定 StudyDate YYYYMMDD（测试用）。")
    parser.add_argument("--overwrite", action="store_true", help="已存在的输出也强制重新处理。")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要处理的文件，不实际执行。")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出 DEBUG 级别日志。")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)

    preprocessor = ImagePreprocessor(
        interim_root=args.interim_root, voxel_size=args.voxel_size, overwrite=args.overwrite,
    )
    results = preprocessor.run(
        dry_run=args.dry_run, patient_id=args.patient_id, study_date=args.study_date,
    )
    if args.dry_run:
        logger.info("Dry-run 完成，未执行任何处理。")
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_stage_log_csv(results, args.log_dir / f"preprocess_{timestamp}.csv")

    counts = summarize(results)
    logger.info("预处理完成: %s", counts)
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
