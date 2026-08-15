#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   organ_segmentation.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    对 data/interim/<PatientID>/<StudyDate>/CT/ 中的原始 CT NIfTI 调用
    TotalSegmentator 进行器官分割，将结果重标为本研究统一的 9 类器官标签，
    写入 data/processed/<PatientID>/<StudyDate>/organs/organs.nii.gz。

    产出的器官掩码有两个用途：
        1. 直接用于 SUV 摄取分析（计算各器官背景 SUV 用于 Deauville 评分参考
           区域等）；
        2. 作为 autoPET3 双头模型微调时 organ head 的监督信号，通过
           combine_lesion_and_organs.py 注入 nnUNet 预处理目录（类别数与
           autoPET3 预训练权重完全对齐，共 10 类 0–9）。

@输入 (Input)
    data/interim/<PatientID>/<StudyDate>/CT/*.nii.gz
    ——使用 DICOM 直转的原始分辨率 CT（约 ~3mm 层厚），而非 2mm 各向同性
    重采样版本。原因：TotalSegmentator 在原始临床分辨率下精度更高，且重采样
    会引入额外的空间对应误差。

@输出 (Output)
    data/processed/<PatientID>/<StudyDate>/organs/organs.nii.gz
        10 类整数标签（uint8），与输入 CT 空间完全一致（affine/尺寸不变）；
        类别编号与 autoPET3 预训练 organ head 完全对齐：
            0  背景（Background）
            1  脾脏（Spleen）
            2  肾脏（Kidney，左+右合并）
            3  肝脏（Liver）
            4  膀胱（Urinary bladder）
            5  肺（Lung，五叶合并）
            6  脑（Brain）
            7  心脏（Heart）
            8  胃（Stomach）
            9  前列腺（Prostate；女性患者该区域自然为背景，不影响分析）
            10 头颈腺体（Head/neck glands：腮腺、颌下腺等）

    中间文件（TotalSegmentator 原始输出，整数多类 NIfTI）保留在：
    data/processed/<PatientID>/<StudyDate>/organs/_totalseg_staging/
    默认保留，便于核查；可在构造时传入 keep_staging=False 自动清理。

@标签映射说明
    TotalSegmentator 全身任务 (--ml) → 本研究标签：
        TotalSeg ID   结构               本研究 ID
        1             Spleen             1
        2             Right kidney       2
        3             Left kidney        2  (合并)
        5             Liver              3
        6             Stomach            8
        10–14         Lung lobes         5  (合并)
        21            Urinary bladder    4
        22            Prostate           9
        51            Heart              7
        90            Brain              6

    TotalSegmentator 头颈任务 (head_glands_cavities, --ml) → 本研究标签：
        TotalSeg ID   结构               本研究 ID
        6             Parotid gland R    10
        7             Parotid gland L    10 (合并)
        8             Submandibular R    10
        9             Submandibular L    10

    类别编号与 autoPET3 原始 lbl_mapping_all / lbl_mapping_head 完全对齐，
    保证微调时器官头权重可直接复用。

@前提条件
    1. TotalSegmentator 已安装（pip install TotalSegmentator 在 autopet env）：
           conda activate autopet
           pip install TotalSegmentator
       第一次运行会自动下载模型权重（~2 GB），需要网络。
    2. GPU 可用时自动使用 GPU（需要 CUDA 与 PyTorch）；无 GPU 时以 CPU
       运行，速度显著降低（每例约 30–60 分钟 vs GPU 约 2–5 分钟）。
    3. 运行本脚本前，data/interim/<PatientID>/<StudyDate>/CT/ 下必须已有
       CT NIfTI 文件（由 scripts/tools/pacs_dicom_to_nifti_suv.py 生成）。

@用法示例
    # 激活环境，确保 totalsegmentator 命令可用
    conda activate autopet

    # 全量运行
    python scripts/processing/organ_extraction/organ_segmentation.py

    # 干跑（只打印计划，不实际处理）
    python scripts/processing/organ_extraction/organ_segmentation.py --dry-run -v

    # 指定路径与并发进程数
    python scripts/processing/organ_extraction/organ_segmentation.py \
        --interim-root data/interim \
        --processed-root data/processed \
        --totalseg-device gpu \
        --overwrite
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import SimpleITK as sitk
import numpy as np

# ------------------------------------------------------------------ #
# 路径配置 & 共用工具
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "common"))
from pipeline_utils import (  # noqa: E402
    StageResult,
    discover_subject_studies,
    setup_logging,
    summarize,
    write_stage_log_csv,
)

logger = logging.getLogger("organ_seg")

STAGE_NAME = "organ_seg"
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
# TotalSegmentator v2.17.0 安装在 data-analysis 环境，使用绝对路径调用，
# 不依赖当前激活环境是否已经把该 env 加入 PATH。
DEFAULT_TOTALSEG_BIN = "/home/sun/miniconda3/envs/data-analysis/bin/TotalSegmentator"

# ------------------------------------------------------------------ #
# 标签映射常量
# ------------------------------------------------------------------ #

# TotalSegmentator 全身任务 ID → 本研究器官 ID
# 与 autoPET3 predict_and_extract_organs.py 的 lbl_mapping_all 完全对齐
TOTALSEG_WHOLEBODY_MAP: dict[int, int] = {
    0:  0,   # 背景
    1:  1,   # Spleen
    2:  2,   # Right kidney
    3:  2,   # Left kidney（合并）
    5:  3,   # Liver
    21: 4,   # Urinary bladder
    10: 5,   # Lung lobe
    11: 5,
    12: 5,
    13: 5,
    14: 5,
    90: 6,   # Brain
    51: 7,   # Heart
    6:  8,   # Stomach
    22: 9,   # Prostate
}

# TotalSegmentator 头颈腺体任务 ID → 本研究器官 ID
# 与 autoPET3 predict_and_extract_organs.py 的 lbl_mapping_head 完全对齐
TOTALSEG_HEAD_GLANDS_MAP: dict[int, int] = {
    6:  10,  # Parotid gland R
    7:  10,  # Parotid gland L
    8:  10,  # Submandibular gland R
    9:  10,  # Submandibular gland L
}

# 类别名称（用于日志可读性）
LABEL_NAMES = {
    0:  "Background",
    1:  "Spleen",
    2:  "Kidney",
    3:  "Liver",
    4:  "Bladder",
    5:  "Lung",
    6:  "Brain",
    7:  "Heart",
    8:  "Stomach",
    9:  "Prostate",
    10: "Head/neck glands",
}


# ------------------------------------------------------------------ #
# 主类
# ------------------------------------------------------------------ #

class OrganSegmentor:
    """
    对每个 (PatientID, StudyDate) 运行 TotalSegmentator 器官分割并重标标签。

    Parameters
    ----------
    interim_root : 数据的 interim 根目录（data/interim）
    processed_root : 结果写出根目录（data/processed）
    totalseg_bin : TotalSegmentator 可执行文件绝对路径（v2.x 命令名为大写
        TotalSegmentator），默认指向 data-analysis 环境
    totalseg_device : "gpu"（默认）或 "cpu"
    keep_staging : 是否保留 TotalSegmentator 原始中间输出（默认 True）
    overwrite : 已存在输出时是否强制重跑（默认 False）
    """

    def __init__(
        self,
        interim_root: Path | str,
        processed_root: Path | str,
        totalseg_bin: str = DEFAULT_TOTALSEG_BIN,
        totalseg_device: str = "gpu",
        keep_staging: bool = True,
        overwrite: bool = False,
    ) -> None:
        self.interim_root = Path(interim_root)
        self.processed_root = Path(processed_root)
        self.totalseg_bin = totalseg_bin
        self.totalseg_device = totalseg_device
        self.keep_staging = keep_staging
        self.overwrite = overwrite

        # 早失败检查：支持绝对路径（os.access）和 PATH 内命令（shutil.which）
        bin_path = Path(self.totalseg_bin)
        if not (bin_path.is_absolute() and bin_path.is_file() and os.access(bin_path, os.X_OK)) \
                and shutil.which(self.totalseg_bin) is None:
            raise RuntimeError(
                f"未找到 TotalSegmentator 可执行文件：{self.totalseg_bin!r}\n"
                "请确认 data-analysis 环境已安装：\n"
                "  conda activate data-analysis && pip install TotalSegmentator"
            )

    def discover_studies(self) -> list[tuple[str, str, Path]]:
        return discover_subject_studies(self.interim_root)

    def _find_original_ct(self, study_dir: Path) -> Optional[Path]:
        """
        在 <study_dir>/CT/ 中寻找原始分辨率 CT NIfTI（DICOM 直转产物）。
        故意不使用 preprocessed/CT/，以保留 TotalSegmentator 所需的原始体素间距。
        """
        ct_dir = study_dir / "CT"
        if not ct_dir.is_dir():
            return None
        candidates = sorted(ct_dir.glob("*.nii.gz"))
        return candidates[0] if candidates else None

    def _run_totalseg(self, input_nii: Path, output_path: Path, task: Optional[str] = None) -> None:
        """
        调用 TotalSegmentator，输出单文件多类标签（--ml）。

        Parameters
        ----------
        input_nii : 输入 CT NIfTI 路径
        output_path : 输出 NIfTI 文件路径（--ml 模式下直接指定文件名）
        task : None 代表默认全身任务；'head_glands_cavities' 等为专项任务
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.totalseg_bin,
            "-i", str(input_nii),
            "-o", str(output_path),
            "--ml",                        # 单文件整数标签输出
            "-d", self.totalseg_device,    # v2.x 用 -d，旧版为 --device
        ]
        if task:
            cmd += ["-ta", task]

        logger.debug("TotalSegmentator 命令: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            logger.debug("TotalSegmentator stdout:\n%s", result.stdout[-2000:])
        if result.returncode != 0:
            # stderr 完整写入日志，方便回溯具体失败原因
            logger.debug("TotalSegmentator stderr:\n%s", result.stderr)
            raise RuntimeError(
                f"TotalSegmentator 失败 (exit={result.returncode})\n"
                f"stderr (末尾 2000 字符):\n{result.stderr[-2000:]}"
            )

    @staticmethod
    def _remap_labels(
        src_array: np.ndarray,
        label_map: dict[int, int],
        out_array: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        按 label_map 映射整数标签数组，在现有 out_array 上叠加写入
        （后调用的映射覆盖先前写入，因此头颈腺体任务的结果会覆盖全身任务中
        编号相同的背景区域，这是期望行为）。
        """
        if out_array is None:
            out_array = np.zeros_like(src_array, dtype=np.uint8)
        for src_id, dst_id in label_map.items():
            out_array[src_array == src_id] = dst_id
        return out_array

    def segment_study(self, patient_id: str, study_date: str, study_dir: Path) -> StageResult:
        ct_path = self._find_original_ct(study_dir)
        if ct_path is None:
            return StageResult(
                STAGE_NAME, patient_id, study_date, "warning", "",
                "未找到原始 CT NIfTI（data/interim/.../CT/），跳过。",
            )

        output_dir = self.processed_root / patient_id / study_date / "organs"
        final_output = output_dir / "organs.nii.gz"

        if final_output.exists() and not self.overwrite:
            return StageResult(
                STAGE_NAME, patient_id, study_date, "skipped", str(final_output),
                "器官掩码已存在，使用 --overwrite 可强制重跑。",
            )

        staging_dir = output_dir / "_totalseg_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ── Step 1: 全身器官分割 ──────────────────────────────────
            wb_output = staging_dir / "totalseg_wholebody.nii.gz"
            if not wb_output.exists() or self.overwrite:
                logger.info("  TotalSegmentator 全身任务: %s", ct_path.name)
                self._run_totalseg(ct_path, wb_output, task=None)

            # ── Step 2: 头颈腺体分割 ──────────────────────────────────
            head_output = staging_dir / "totalseg_head_glands.nii.gz"
            if not head_output.exists() or self.overwrite:
                logger.info("  TotalSegmentator 头颈腺体任务: %s", ct_path.name)
                self._run_totalseg(ct_path, head_output, task="head_glands_cavities")

            # ── Step 3: 读取 → 重标 → 合并 ───────────────────────────
            wb_img = sitk.ReadImage(str(wb_output))
            wb_arr = sitk.GetArrayFromImage(wb_img).astype(np.int32)

            head_img = sitk.ReadImage(str(head_output))
            head_arr = sitk.GetArrayFromImage(head_img).astype(np.int32)

            # 合并规则：先用全身任务填充，再用头颈任务覆盖头颈腺体区域
            # （头颈任务中 ID 6-9 对应四个腺体结构，空间范围小于全身任务中
            # 相同 ID 对应的胃（ID 6）等结构，不会互相干扰）
            combined = self._remap_labels(wb_arr, TOTALSEG_WHOLEBODY_MAP)
            combined = self._remap_labels(head_arr, TOTALSEG_HEAD_GLANDS_MAP, combined)

            # ── Step 4: 保存（复用全身任务 affine/spacing/origin）────
            out_img = sitk.GetImageFromArray(combined.astype(np.uint8))
            out_img.CopyInformation(wb_img)
            sitk.WriteImage(out_img, str(final_output))

            # ── Step 5: 可选清理 staging 目录 ────────────────────────
            if not self.keep_staging:
                shutil.rmtree(staging_dir, ignore_errors=True)

            # 统计各类体素数，写入日志便于快速核查
            counts = {LABEL_NAMES[i]: int((combined == i).sum()) for i in LABEL_NAMES if i > 0}
            return StageResult(
                STAGE_NAME, patient_id, study_date, "ok", str(final_output),
                f"器官分割完成 | 体素计数: {counts}",
            )

        except Exception as exc:  # noqa: BLE001 - 单 Study 失败不影响其他
            logger.exception("器官分割失败: patient=%s study=%s", patient_id, study_date)
            return StageResult(
                STAGE_NAME, patient_id, study_date, "error", str(final_output), str(exc)
            )

    def run(self, dry_run: bool = False) -> list[StageResult]:
        studies = self.discover_studies()
        logger.info("共发现 %d 个 (patient, study) 待器官分割。", len(studies))

        if dry_run:
            for patient_id, study_date, study_dir in studies:
                ct = self._find_original_ct(study_dir)
                output = self.processed_root / patient_id / study_date / "organs" / "organs.nii.gz"
                logger.info(
                    "[DRY-RUN] patient=%s study=%s  CT=%s  -> %s",
                    patient_id, study_date,
                    ct.name if ct else "(未找到)",
                    output,
                )
            return []

        results: list[StageResult] = []
        for idx, (patient_id, study_date, study_dir) in enumerate(studies, start=1):
            logger.info(
                "[%d/%d] 器官分割  patient=%s  study=%s",
                idx, len(studies), patient_id, study_date,
            )
            result = self.segment_study(patient_id, study_date, study_dir)
            results.append(result)
            if result.status == "error":
                logger.error("失败: %s", result.message)
            elif result.status == "warning":
                logger.warning("跳过: %s", result.message)
        return results


# ------------------------------------------------------------------ #
# CLI 入口
# ------------------------------------------------------------------ #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对每个 Study 的原始 CT 运行 TotalSegmentator，生成 9 类器官掩码。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--totalseg-bin", default=DEFAULT_TOTALSEG_BIN,
        help="TotalSegmentator 可执行文件绝对路径。",
    )
    parser.add_argument(
        "--totalseg-device", choices=["gpu", "cpu"], default="gpu",
        help="推理设备。GPU 下每例约 2–5 min，CPU 下约 30–60 min。",
    )
    parser.add_argument(
        "--no-keep-staging", action="store_true",
        help="运行完成后删除 TotalSegmentator 中间输出（默认保留以便核查）。",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)

    try:
        segmentor = OrganSegmentor(
            interim_root=args.interim_root,
            processed_root=args.processed_root,
            totalseg_bin=args.totalseg_bin,
            totalseg_device=args.totalseg_device,
            keep_staging=not args.no_keep_staging,
            overwrite=args.overwrite,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    results = segmentor.run(dry_run=args.dry_run)
    if args.dry_run:
        logger.info("Dry-run 完成，未执行任何实际处理。")
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_stage_log_csv(results, args.log_dir / f"organ_seg_{timestamp}.csv")

    counts = summarize(results)
    logger.info("器官分割完成: %s", counts)
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
