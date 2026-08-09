#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   segmentation.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    病灶分割阶段：读取 preprocess.py 产出的重采样后 CT/PET，生成二值分割
    掩码（Mask），写入 data/processed/。

    分割算法本身通过可注入的 `segmentation_fn` 解耦：
        - 默认提供一个真实可用、核医学领域标准的基线方法——固定/相对 SUV
          阈值分割（`threshold_suv_mask()`，见下方说明），适用于有 PET 的
          Study，可直接产出有意义的结果，而不是一个空占位符。
        - 如果团队后续训练了专门的分割模型（如基于 nnU-Net / TotalSegmentator
          的病灶分割），只需在构造 `LesionSegmenter` 时传入自定义
          `segmentation_fn`（签名见类文档），不需要改动本文件其余的发现 /
          跳过 / 日志基础设施。
        - 仅有 CT、没有 PET 的 Study：目前没有集成任何 CT-only 的分割算法
          （避免在没有验证依据的情况下编造分割逻辑），会记录为 "warning"，
          明确提示需要接入自定义 segmentation_fn 或专用 CT 分割模型。

@输入 (Input)
    data/interim/<PatientID>/<StudyDate>/preprocessed/PET/*.nii.gz
    data/interim/<PatientID>/<StudyDate>/preprocessed/CT/*.nii.gz（可选，
    仅用于后续可能的 CT-based 算法接入，当前默认算法不使用）

@输出 (Output)
    data/processed/<PatientID>/<StudyDate>/masks/<PET文件名去掉扩展名>_mask.nii.gz
    掩码为 uint8，1=病灶，0=背景，仿射矩阵与输入 PET 完全一致。

@默认分割算法：SUV 阈值法 (threshold_suv_mask)
    - mode="absolute"（默认）：mask = (SUV >= threshold)，默认
      threshold=2.5 g/mL，是 PET 肿瘤学文献中最常用的固定阈值之一。
    - mode="relative"：mask = (SUV >= threshold * SUVmax)，threshold 常取
      0.41（即 41% SUVmax，亦是文献中常见取值）。
    这是一个简单、可复现、有文献依据的基线，不代表最终临床级分割结果，
    仅作为流程打通与后续对比的起点。

@关键执行逻辑
    - discover_studies()：复用 scripts/common/pipeline_utils。
    - segment_study()：为每个 Study 定位其 preprocessed PET（找不到则跳过
      并给出 warning，不影响其他 Study）；调用 segmentation_fn 得到掩码
      数组；用与输入相同的 affine/header 保存，保证空间一致性；每个 Study
      独立 try/except。
    - 增量执行：目标掩码已存在则跳过，除非 overwrite=True。

@用法示例
    python scripts/processing/segmentation.py
    python scripts/processing/segmentation.py --threshold-mode relative --threshold 0.41
    python scripts/processing/segmentation.py --dry-run -v
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("缺少 nibabel，请先安装。") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from pipeline_utils import (  # noqa: E402
    StageResult,
    discover_subject_studies,
    setup_logging,
    summarize,
    write_stage_log_csv,
)

logger = logging.getLogger("segmentation")

STAGE_NAME = "segment"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

# 自定义分割函数的统一签名：接收 PET 像素数组（float），返回同形状的 0/1 掩码。
SegmentationFn = Callable[[np.ndarray], np.ndarray]


def threshold_suv_mask(suv_data: np.ndarray, mode: str = "absolute", threshold: float = 2.5) -> np.ndarray:
    """
    SUV 阈值分割基线方法，返回 uint8 掩码（1=病灶，0=背景）。

    mode="absolute": mask = suv_data >= threshold（默认 threshold=2.5 g/mL）
    mode="relative": mask = suv_data >= threshold * suv_data.max()
                      （默认建议 threshold=0.41，即 41% SUVmax）
    """
    if mode == "absolute":
        mask = suv_data >= threshold
    elif mode == "relative":
        suv_max = float(np.nanmax(suv_data)) if suv_data.size else 0.0
        mask = suv_data >= (threshold * suv_max)
    else:
        raise ValueError(f"未知的 threshold mode: {mode!r}，应为 'absolute' 或 'relative'。")
    return mask.astype(np.uint8)


class LesionSegmenter:
    """
    病灶分割阶段的编排类：负责发现 Study、定位输入、调用分割算法、落盘掩码。

    Parameters
    ----------
    interim_root, processed_root:
        分别对应 data/interim（读取 preprocessed PET/CT）与 data/processed
        （写出掩码）。
    segmentation_fn:
        可选的自定义分割函数，签名为 `f(pet_data: np.ndarray) -> np.ndarray`
        （返回与输入同形状的 0/1 掩码）。默认使用 `threshold_suv_mask`
        （固定参数见 threshold_mode / threshold）。团队后续接入真实模型时，
        只需传入形如::

            def my_model_segmentation(pet_data: np.ndarray) -> np.ndarray:
                ...  # 调用 nnU-Net / TotalSegmentator 等真实模型推理
                return mask

        并在构造时传入 `segmentation_fn=my_model_segmentation` 即可，无需
        改动本类其余逻辑。
    """

    def __init__(
        self,
        interim_root: Path | str,
        processed_root: Path | str,
        overwrite: bool = False,
        segmentation_fn: Optional[SegmentationFn] = None,
        threshold_mode: str = "absolute",
        threshold: float = 2.5,
    ) -> None:
        self.interim_root = Path(interim_root)
        self.processed_root = Path(processed_root)
        self.overwrite = overwrite
        self.threshold_mode = threshold_mode
        self.threshold = threshold
        self.segmentation_fn = segmentation_fn or self._default_segmentation_fn

    def _default_segmentation_fn(self, pet_data: np.ndarray) -> np.ndarray:
        return threshold_suv_mask(pet_data, mode=self.threshold_mode, threshold=self.threshold)

    def discover_studies(self) -> list[tuple[str, str, Path]]:
        return discover_subject_studies(self.interim_root)

    @staticmethod
    def _find_preprocessed_pet(study_dir: Path) -> Optional[Path]:
        pet_dir = study_dir / "preprocessed" / "PET"
        if not pet_dir.is_dir():
            return None
        candidates = sorted(pet_dir.glob("*_SUVbw.nii.gz")) or sorted(pet_dir.glob("*.nii.gz"))
        return candidates[0] if candidates else None

    def segment_study(self, patient_id: str, study_date: str, study_dir: Path) -> StageResult:
        pet_path = self._find_preprocessed_pet(study_dir)
        if pet_path is None:
            return StageResult(
                STAGE_NAME, patient_id, study_date, "warning", "",
                "未找到 preprocessed PET，跳过分割（当前默认算法依赖 PET/SUV）。"
                " 如需 CT-only 分割，请传入自定义 segmentation_fn。",
            )

        output_dir = self.processed_root / patient_id / study_date / "masks"
        mask_path = output_dir / f"{pet_path.stem.removesuffix('.nii')}_mask.nii.gz"
        if mask_path.exists() and not self.overwrite:
            return StageResult(
                STAGE_NAME, patient_id, study_date, "skipped", str(mask_path),
                "输出已存在，overwrite=True 可强制重转。",
            )

        try:
            img = nib.load(str(pet_path))
            pet_data = img.get_fdata(dtype=np.float64)
            mask_data = self.segmentation_fn(pet_data)
            if mask_data.shape != pet_data.shape:
                raise ValueError(
                    f"segmentation_fn 返回的掩码形状 {mask_data.shape} 与输入 {pet_data.shape} 不一致。"
                )
            mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), img.affine, img.header)
            mask_img.header.set_data_dtype(np.uint8)
            output_dir.mkdir(parents=True, exist_ok=True)
            nib.save(mask_img, str(mask_path))
            n_voxels = int(mask_data.sum())
            return StageResult(
                STAGE_NAME, patient_id, study_date, "ok", str(mask_path),
                f"分割完成，阳性体素数={n_voxels} (mode={self.threshold_mode}, threshold={self.threshold})",
            )
        except NotImplementedError as exc:
            return StageResult(
                STAGE_NAME, patient_id, study_date, "warning", str(mask_path),
                f"分割算法未实现: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - 单个 Study 失败不影响其他 Study
            logger.exception("分割失败: %s", pet_path)
            return StageResult(STAGE_NAME, patient_id, study_date, "error", str(mask_path), str(exc))

    def run(self, dry_run: bool = False) -> list[StageResult]:
        studies = self.discover_studies()
        logger.info("共发现 %d 个 (patient, study) 待分割。", len(studies))

        if dry_run:
            for patient_id, study_date, study_dir in studies:
                pet_path = self._find_preprocessed_pet(study_dir)
                logger.info(
                    "[DRY-RUN] patient=%s study=%s pet=%s",
                    patient_id, study_date, pet_path if pet_path else "(未找到)",
                )
            return []

        results: list[StageResult] = []
        for idx, (patient_id, study_date, study_dir) in enumerate(studies, start=1):
            logger.info("[%d/%d] 分割 patient=%s study=%s", idx, len(studies), patient_id, study_date)
            result = self.segment_study(patient_id, study_date, study_dir)
            results.append(result)
            if result.status == "error":
                logger.error("失败: patient=%s study=%s -> %s", patient_id, study_date, result.message)
            elif result.status == "warning":
                logger.warning("警告: patient=%s study=%s -> %s", patient_id, study_date, result.message)
        return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对 preprocessed PET 做 SUV 阈值病灶分割（基线方法，可替换为自定义模型）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT, help="data/interim 根目录。")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT, help="data/processed 根目录。")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="CSV 日志输出目录。")
    parser.add_argument(
        "--threshold-mode", choices=["absolute", "relative"], default="absolute",
        help="'absolute'=固定 SUV 阈值；'relative'=相对 SUVmax 的比例阈值。",
    )
    parser.add_argument("--threshold", type=float, default=2.5, help="阈值数值（absolute 模式单位为 SUV，relative 模式为比例）。")
    parser.add_argument("--overwrite", action="store_true", help="已存在的输出也强制重新分割。")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要处理的 Study，不实际执行。")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出 DEBUG 级别日志。")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)

    segmenter = LesionSegmenter(
        interim_root=args.interim_root,
        processed_root=args.processed_root,
        overwrite=args.overwrite,
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
    )
    results = segmenter.run(dry_run=args.dry_run)
    if args.dry_run:
        logger.info("Dry-run 完成，未执行任何处理。")
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_stage_log_csv(results, args.log_dir / f"segment_{timestamp}.csv")

    counts = summarize(results)
    logger.info("分割完成: %s", counts)
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
