#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   segmentation.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    病灶分割阶段：读取 preprocess.py 产出的重采样后 CT/PET，生成二值分割
    掩码（Mask），写入 data/processed/。

    支持两种分割后端（通过 --method 切换，默认 nnunet）：

    1. nnunet（默认）
       调用 infer_nnunet.NnuNetInferrer 对 nnunet_export 目录中已导出的
       CT/PET 对执行 AutoPET III 冠军模型推理（5-fold 集成，GPU），
       产出病灶掩码到 data/processed/<PatientID>/<StudyDate>/masks/
       文件名：{PatientID}_{StudyDate}_lesion.nii.gz

    2. threshold（SUV 阈值基线，原始方法）
       直接对 preprocessed PET SUV 图做固定/相对阈值分割，
       无需 GPU 或 nnU-Net，适用于快速验证或无 GPU 场景。
       文件名：{PET原始名}_mask.nii.gz

    两种方法产出的掩码均为 uint8（1=病灶, 0=背景），仿射矩阵与输入一致，
    因此可直接互换用于下游特征提取。

@用法示例
    # 使用 nnU-Net（默认，需在 autopet 环境运行）
    /home/sun/miniconda3/envs/autopet/bin/python scripts/processing/segmentation.py
    /home/sun/miniconda3/envs/autopet/bin/python scripts/processing/segmentation.py --patient-id 00857723

    # 使用 SUV 阈值基线（任意环境）
    python scripts/processing/segmentation.py --method threshold
    python scripts/processing/segmentation.py --method threshold --threshold-mode relative --threshold 0.41

    # 两种方法并行运行（用于对比）
    python scripts/processing/segmentation.py --method both

    # 通过 main.py（nnunet 方法需指定 autopet Python）
    python main.py --stage segment
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

logger = logging.getLogger("segmentation")

STAGE_NAME = "segment"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "data" / "nnunet_export"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_MODEL_FOLDER = (
    PROJECT_ROOT
    / "autoPET"
    / "Dataset222_AutoPETIII_2024"
    / "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3"
)

SegmentationFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# SUV 阈值基线
# ---------------------------------------------------------------------------

def threshold_suv_mask(
    suv_data: np.ndarray,
    mode: str = "absolute",
    threshold: float = 2.5,
) -> np.ndarray:
    """
    SUV 阈值分割基线（mode='absolute' 或 'relative'），返回 uint8 掩码。
    """
    if mode == "absolute":
        mask = suv_data >= threshold
    elif mode == "relative":
        suv_max = float(np.nanmax(suv_data)) if suv_data.size else 0.0
        mask = suv_data >= (threshold * suv_max)
    else:
        raise ValueError(f"未知的 threshold mode: {mode!r}")
    return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# 阈值分割后端
# ---------------------------------------------------------------------------

class ThresholdSegmenter:
    """对 preprocessed PET 执行 SUV 阈值分割（基线方法）。"""

    def __init__(
        self,
        interim_root: Path | str,
        processed_root: Path | str,
        overwrite: bool = False,
        threshold_mode: str = "absolute",
        threshold: float = 2.5,
        segmentation_fn: Optional[SegmentationFn] = None,
    ) -> None:
        self.interim_root = Path(interim_root)
        self.processed_root = Path(processed_root)
        self.overwrite = overwrite
        self.threshold_mode = threshold_mode
        self.threshold = threshold
        self.segmentation_fn = segmentation_fn or self._default_fn

    def _default_fn(self, pet_data: np.ndarray) -> np.ndarray:
        return threshold_suv_mask(pet_data, self.threshold_mode, self.threshold)

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
                "未找到 preprocessed PET，跳过阈值分割。",
            )

        output_dir = self.processed_root / patient_id / study_date / "masks"
        stem = pet_path.name.removesuffix(".nii.gz").removesuffix(".nii")
        mask_path = output_dir / f"{stem}_mask.nii.gz"

        if mask_path.exists() and not self.overwrite:
            return StageResult(STAGE_NAME, patient_id, study_date, "skipped", str(mask_path),
                               "输出已存在，overwrite=True 可强制重跑。")
        try:
            img = nib.load(str(pet_path))
            pet_data = img.get_fdata(dtype=np.float64)
            mask_data = self.segmentation_fn(pet_data)
            if mask_data.shape != pet_data.shape:
                raise ValueError(f"掩码形状 {mask_data.shape} 与输入 {pet_data.shape} 不一致。")
            mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), img.affine, img.header)
            mask_img.header.set_data_dtype(np.uint8)
            output_dir.mkdir(parents=True, exist_ok=True)
            nib.save(mask_img, str(mask_path))
            return StageResult(
                STAGE_NAME, patient_id, study_date, "ok", str(mask_path),
                f"阈值分割完成，体素数={int(mask_data.sum())} "
                f"(mode={self.threshold_mode}, threshold={self.threshold})",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("阈值分割失败: %s", pet_path)
            return StageResult(STAGE_NAME, patient_id, study_date, "error", "", str(exc))

    def run(
        self,
        dry_run: bool = False,
        patient_id: Optional[str] = None,
        study_date: Optional[str] = None,
    ) -> list[StageResult]:
        studies = discover_subject_studies(self.interim_root)
        if patient_id:
            studies = [s for s in studies if s[0] == patient_id]
        if study_date:
            studies = [s for s in studies if s[1] == study_date]
        logger.info("[threshold] 共发现 %d 个 study 待分割。", len(studies))
        if dry_run:
            for pid, sdate, study_dir in studies:
                pet = self._find_preprocessed_pet(study_dir)
                logger.info("[DRY-RUN] threshold patient=%s study=%s pet=%s", pid, sdate, pet)
            return []
        results = []
        for idx, (pid, sdate, study_dir) in enumerate(studies, start=1):
            logger.info("[%d/%d] threshold patient=%s study=%s", idx, len(studies), pid, sdate)
            results.append(self.segment_study(pid, sdate, study_dir))
        return results


# ---------------------------------------------------------------------------
# nnU-Net 后端（委托给 infer_nnunet）
# ---------------------------------------------------------------------------

class NnunetSegmenter:
    """对 nnunet_export 的 CT/PET 对执行 AutoPET nnU-Net 推理。"""

    def __init__(
        self,
        export_root: Path | str,
        processed_root: Path | str,
        model_folder: Path | str,
        overwrite: bool = False,
        folds: list[int] | None = None,
        device: str = "cuda",
        num_proc: int = 2,
    ) -> None:
        # 延迟导入，避免无 nnunet 环境时 import 阶段报错
        try:
            from infer_nnunet import NnuNetInferrer  # noqa: PLC0415
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from infer_nnunet import NnuNetInferrer  # noqa: PLC0415
        self._inferrer = NnuNetInferrer(
            export_root=export_root,
            processed_root=processed_root,
            model_folder=model_folder,
            overwrite=overwrite,
            folds=folds or [0, 1, 2, 3, 4],
            device=device,
            num_proc=num_proc,
        )

    def run(
        self,
        dry_run: bool = False,
        patient_id: Optional[str] = None,
        study_date: Optional[str] = None,
    ) -> list[StageResult]:
        return self._inferrer.run(
            dry_run=dry_run, patient_id=patient_id, study_date=study_date,
        )


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

class LesionSegmenter:
    """
    统一的分割阶段入口，支持 method='nnunet'（默认）、'threshold' 或 'both'。

    'both' 模式会同时运行两种方法，产出不同文件名，便于对比评估。
    """

    def __init__(
        self,
        interim_root: Path | str = DEFAULT_INTERIM_ROOT,
        processed_root: Path | str = DEFAULT_PROCESSED_ROOT,
        export_root: Path | str = DEFAULT_EXPORT_ROOT,
        model_folder: Path | str = DEFAULT_MODEL_FOLDER,
        overwrite: bool = False,
        method: str = "nnunet",
        # 阈值方法参数
        segmentation_fn: Optional[SegmentationFn] = None,
        threshold_mode: str = "absolute",
        threshold: float = 2.5,
        # nnU-Net 参数
        folds: list[int] | None = None,
        device: str = "cuda",
        num_proc: int = 2,
    ) -> None:
        self.method = method
        self._threshold_seg = ThresholdSegmenter(
            interim_root=interim_root,
            processed_root=processed_root,
            overwrite=overwrite,
            segmentation_fn=segmentation_fn,
            threshold_mode=threshold_mode,
            threshold=threshold,
        )
        if method in ("nnunet", "both"):
            self._nnunet_seg = NnunetSegmenter(
                export_root=export_root,
                processed_root=processed_root,
                model_folder=model_folder,
                overwrite=overwrite,
                folds=folds,
                device=device,
                num_proc=num_proc,
            )
        else:
            self._nnunet_seg = None

    def run(
        self,
        dry_run: bool = False,
        patient_id: Optional[str] = None,
        study_date: Optional[str] = None,
    ) -> list[StageResult]:
        results: list[StageResult] = []
        if self.method in ("threshold", "both"):
            results.extend(
                self._threshold_seg.run(dry_run=dry_run,
                                        patient_id=patient_id, study_date=study_date)
            )
        if self.method in ("nnunet", "both") and self._nnunet_seg is not None:
            results.extend(
                self._nnunet_seg.run(dry_run=dry_run,
                                     patient_id=patient_id, study_date=study_date)
            )
        return results


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="病灶分割：支持 nnU-Net AutoPET 模型（默认）和 SUV 阈值基线。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=["nnunet", "threshold", "both"], default="nnunet",
                        help="分割方法：nnunet=AutoPET模型，threshold=SUV阈值，both=两者并行。")
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--study-date", default=None)
    # 阈值参数
    parser.add_argument("--threshold-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--threshold", type=float, default=2.5)
    # nnU-Net 参数
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-proc", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)

    segmenter = LesionSegmenter(
        interim_root=args.interim_root,
        processed_root=args.processed_root,
        export_root=args.export_root,
        model_folder=args.model_folder,
        overwrite=args.overwrite,
        method=args.method,
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
        folds=args.folds,
        device=args.device,
        num_proc=args.num_proc,
    )
    results = segmenter.run(
        dry_run=args.dry_run, patient_id=args.patient_id, study_date=args.study_date,
    )
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
