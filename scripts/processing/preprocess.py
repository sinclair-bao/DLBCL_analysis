#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   preprocess.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    图像预处理阶段：把 scripts/tools/pacs_dicom_to_nifti_suv.py 产出的
    data/interim/<PatientID>/<StudyDate>/{CT,PET}/*.nii.gz 重采样为统一的
    各向同性体素间距（默认 2mm），作为后续分割 / 特征提取的标准化输入。

@输入 (Input)
    data/interim/<PatientID>/<StudyDate>/CT/*.nii.gz
    data/interim/<PatientID>/<StudyDate>/PET/*_SUVbw.nii.gz（优先）或
    data/interim/<PatientID>/<StudyDate>/PET/*_ACT.nii.gz（SUV 换算前提不
    满足时 pacs_dicom_to_nifti_suv.py 只会产出这一份）

@输出 (Output)
    data/interim/<PatientID>/<StudyDate>/preprocessed/{CT,PET}/<原文件名>
    （与源文件同名，仅体素间距被重采样，保持可追溯）

@关键执行逻辑
    - discover_studies()：复用 scripts/common/pipeline_utils 里统一的
      <PatientID>/<StudyDate> 两级目录发现逻辑。
    - process_study()：对每个 Study，分别处理其下的 CT / PET 子目录中的
      每个 .nii.gz 文件；每个文件独立 try/except，单个文件失败不影响同一
      Study 内其他文件、也不影响其他 Study。
    - _resample()：调用 nibabel.processing.resample_to_output 做各向同性
      重采样，CT 默认三次样条插值（order=3，重采样质量更好，HU 值允许小幅
      过冲，不做额外裁剪，如需严格限制到 [-1000, 3000] HU 可在此处按需
      添加 np.clip）；PET 默认线性插值（order=1，避免样条插值在 SUV 热点
      周围引入不真实的振铃/过冲，这对定量分析更重要）。
    - 增量执行：目标文件已存在则跳过，除非 overwrite=True。

@复用方式
    与 pacs_dicom_to_nifti_suv.py 中的 Converter 同样的模式：核心逻辑是
    `ImagePreprocessor` 类，可以 import 后直接用不同的 interim_root 处理
    别的项目的数据；命令行用法见下方入口函数。

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
    """把 data/interim 下的 CT/PET NIfTI 重采样为统一的各向同性体素间距。"""

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

    def _resample_one(self, src: Path, dst: Path, order: int) -> None:
        img = nib.load(str(src))
        resampled = nib_processing.resample_to_output(
            img, voxel_sizes=self.voxel_size, order=order
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(resampled, str(dst))

    def process_study(self, patient_id: str, study_date: str, study_dir: Path) -> list[StageResult]:
        results: list[StageResult] = []
        for modality in ("CT", "PET"):
            input_files = self._find_input_files(study_dir, modality)
            if not input_files:
                continue
            output_dir = study_dir / "preprocessed" / modality
            order = DEFAULT_INTERP_ORDER[modality]
            for src in input_files:
                dst = output_dir / src.name
                if dst.exists() and not self.overwrite:
                    results.append(
                        StageResult(
                            STAGE_NAME, patient_id, study_date, "skipped",
                            str(dst), "输出已存在，overwrite=True 可强制重转。",
                        )
                    )
                    continue
                try:
                    self._resample_one(src, dst, order)
                    results.append(
                        StageResult(
                            STAGE_NAME, patient_id, study_date, "ok",
                            str(dst), f"{modality} 重采样完成 (voxel_size={self.voxel_size}mm, order={order})",
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - 单个文件失败不影响其他文件
                    logger.exception("预处理失败: %s", src)
                    results.append(
                        StageResult(STAGE_NAME, patient_id, study_date, "error", str(dst), str(exc))
                    )
        return results

    def run(self, dry_run: bool = False) -> list[StageResult]:
        studies = self.discover_studies()
        logger.info("共发现 %d 个 (patient, study) 待预处理。", len(studies))

        if dry_run:
            for patient_id, study_date, study_dir in studies:
                for modality in ("CT", "PET"):
                    for src in self._find_input_files(study_dir, modality):
                        logger.info(
                            "[DRY-RUN] %s -> %s",
                            src, study_dir / "preprocessed" / modality / src.name,
                        )
            return []

        all_results: list[StageResult] = []
        for idx, (patient_id, study_date, study_dir) in enumerate(studies, start=1):
            logger.info("[%d/%d] 预处理 patient=%s study=%s", idx, len(studies), patient_id, study_date)
            all_results.extend(self.process_study(patient_id, study_date, study_dir))
        return all_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 data/interim 下的 CT/PET 重采样为统一的各向同性体素间距。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT, help="data/interim 根目录。")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="CSV 日志输出目录。")
    parser.add_argument("--voxel-size", type=float, default=2.0, help="目标各向同性体素间距（mm）。")
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
    results = preprocessor.run(dry_run=args.dry_run)
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
