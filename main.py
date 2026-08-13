#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   main.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    DLBCL 影像分析流程的统一入口。本文件本身不实现任何算法，只负责按顺序
    调用各阶段模块（真正的逻辑都在对应模块里，既可以被这里 import 调用，
    也可以单独用命令行跑，便于调试单个阶段）：

        convert    -> scripts/tools/pacs_dicom_to_nifti_suv.py
                      （PACS 归档 DICOM -> NIfTI，PET 换算为 SUVbw）
        preprocess -> scripts/processing/preprocess.py
                      （CT/PET 重采样为统一各向同性体素间距，
                        PET 对齐到 CT 网格，保证双通道 shape 一致）
        export     -> scripts/processing/export_nnunet.py
                      （将 preprocessed CT/PET 导出为 nnU-Net 推理命名格式
                        data/nnunet_export/{patient}_{date}_0000/0001.nii.gz）
        segment    -> scripts/processing/segmentation.py
                      （病灶分割：默认使用 AutoPET III nnU-Net 模型，
                        也可用 --segment-method threshold 切换到 SUV 阈值基线，
                        或 --segment-method both 并行运行两种方法对比）
        analyze    -> scripts/analysis/plot_results.py (+ stats_analysis.R)
                      （特征统计与绘图；当前为占位，等特征提取步骤补齐后接入）

    每个阶段内部都做了"输出已存在则跳过"的增量式处理（除非 --overwrite），
    单个病例/序列失败不会中断整批任务，因此：
        - 中断后重跑 `python main.py --stage all` 只会补跑未完成的部分；
        - 只想调试某一步时，可以只跑 `--stage preprocess` 等单一阶段，或
          直接运行对应模块自己的 CLI（见各模块文件头部的用法示例）。

@用法示例
    # 完整跑一遍（需在 autopet 环境，包含 GPU nnU-Net 推理）
    /home/sun/miniconda3/envs/autopet/bin/python main.py --stage all

    # 只跑某一阶段
    /home/sun/miniconda3/envs/autopet/bin/python main.py --stage segment
    /home/sun/miniconda3/envs/autopet/bin/python main.py --stage export

    # 使用 SUV 阈值基线（任意环境均可）
    python main.py --stage segment --segment-method threshold

    # 干跑，只看计划
    python main.py --stage all --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent

for _subdir in ("tools", "processing", "analysis", "common"):
    sys.path.insert(0, str(ROOT / "scripts" / _subdir))

from pacs_dicom_to_nifti_suv import PacsDicomToNiftiSuvConverter  # noqa: E402
from preprocess import ImagePreprocessor  # noqa: E402
from export_nnunet import NnuNetExporter  # noqa: E402
from segmentation import LesionSegmenter  # noqa: E402
from pipeline_utils import (  # noqa: E402
    StageResult,
    setup_logging,
    summarize,
    write_stage_log_csv,
)

import plot_results  # noqa: E402

logger = logging.getLogger("main")

DEFAULT_MODEL_FOLDER = (
    ROOT
    / "autoPET"
    / "Dataset222_AutoPETIII_2024"
    / "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DLBCL imaging analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["convert", "preprocess", "export", "segment", "analyze", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data",
                        help="Data root containing raw/interim/processed")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results",
                        help="Results output root")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs",
                        help="Stage log (CSV) output directory")

    parser.add_argument(
        "--source", action="append", dest="sources",
        help="[convert] 源 DICOM 路径（glob 或目录），可重复传入；默认 <data-root>/raw/DICOM*",
    )
    parser.add_argument("--dcm2niix-bin", default="dcm2niix",
                        help="[convert] dcm2niix 可执行文件路径。")

    parser.add_argument("--voxel-size", type=float, default=2.0,
                        help="[preprocess] 目标各向同性体素间距（mm）。")

    parser.add_argument(
        "--segment-method", choices=["nnunet", "threshold", "both"], default="nnunet",
        help="[segment] nnunet=AutoPET模型（默认），threshold=SUV阈值基线，both=两者并行对比。",
    )
    parser.add_argument(
        "--threshold-mode", choices=["absolute", "relative"], default="absolute",
        help="[segment/threshold] 'absolute'=固定 SUV 阈值；'relative'=相对 SUVmax 比例阈值。",
    )
    parser.add_argument("--threshold", type=float, default=2.5,
                        help="[segment/threshold] 阈值数值。")
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER,
                        help="[segment/nnunet] AutoPET nnU-Net 模型权重目录（含 fold_0..4）。")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="[segment/nnunet] 使用的 fold 编号。")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="[segment/nnunet] 推理设备。")
    parser.add_argument("--num-proc", type=int, default=2,
                        help="[segment/nnunet] nnU-Net 预处理/后处理并行进程数。")

    parser.add_argument("--overwrite", action="store_true",
                        help="已存在的输出也强制重新处理（适用于所有阶段）。")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印各阶段将要处理的对象，不实际执行。")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出 DEBUG 级别日志。")
    return parser


def run_convert(args: argparse.Namespace) -> list[StageResult]:
    converter = PacsDicomToNiftiSuvConverter(
        output_root=args.data_root / "interim",
        dcm2niix_bin=args.dcm2niix_bin,
        overwrite=args.overwrite,
    )
    sources = args.sources or [str(args.data_root / "raw" / "DICOM*")]
    conversion_results = converter.run(sources, dry_run=args.dry_run)
    return [
        StageResult("convert", r.patient_id, r.study_date, r.status, r.output_dir, r.message)
        for r in conversion_results
    ]


def run_preprocess(args: argparse.Namespace) -> list[StageResult]:
    preprocessor = ImagePreprocessor(
        interim_root=args.data_root / "interim",
        voxel_size=args.voxel_size,
        overwrite=args.overwrite,
    )
    return preprocessor.run(dry_run=args.dry_run)


def run_export(args: argparse.Namespace) -> list[StageResult]:
    exporter = NnuNetExporter(
        interim_root=args.data_root / "interim",
        export_root=args.data_root / "nnunet_export",
        overwrite=args.overwrite,
    )
    return exporter.run(dry_run=args.dry_run)


def run_segment(args: argparse.Namespace) -> list[StageResult]:
    segmenter = LesionSegmenter(
        interim_root=args.data_root / "interim",
        processed_root=args.data_root / "processed",
        export_root=args.data_root / "nnunet_export",
        model_folder=args.model_folder,
        overwrite=args.overwrite,
        method=args.segment_method,
        threshold_mode=args.threshold_mode,
        threshold=args.threshold,
        folds=args.folds,
        device=args.device,
        num_proc=args.num_proc,
    )
    return segmenter.run(dry_run=args.dry_run)


def run_analyze(args: argparse.Namespace) -> list[StageResult]:
    """
    分析阶段当前为占位：特征提取步骤（从 data/processed 的掩码/影像汇总出
    results/tables/features.csv）尚未实现，因此这里只是尝试调用绘图函数
    并把 NotImplementedError 记录为 warning，而不是让整个流程崩溃。
    """
    if args.dry_run:
        logger.info("[DRY-RUN] analyze: 将读取 %s 并输出图表到 %s",
                    args.results_root / "tables", args.results_root / "figures")
        return []
    try:
        plot_results.plot_feature_distributions(
            args.results_root / "tables" / "features.csv",
            args.results_root / "figures" / "feature_distributions.png",
        )
        return [StageResult("analyze", "-", "-", "ok",
                            str(args.results_root / "figures"), "分析完成。")]
    except NotImplementedError as exc:
        logger.warning("analyze 阶段尚未实现: %s", exc)
        return [StageResult("analyze", "-", "-", "warning", "", f"尚未实现: {exc}")]


STAGE_RUNNERS = {
    "convert": run_convert,
    "preprocess": run_preprocess,
    "export": run_export,
    "segment": run_segment,
    "analyze": run_analyze,
}


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    stages = list(STAGE_RUNNERS) if args.stage == "all" else [args.stage]

    all_results: list[StageResult] = []
    for stage in stages:
        logger.info("===== 开始阶段: %s =====", stage)
        results = STAGE_RUNNERS[stage](args)
        all_results.extend(results)
        if not args.dry_run:
            write_stage_log_csv(results, args.log_dir / f"{stage}_summary.csv")
        logger.info("===== 阶段 %s 完成: %s =====", stage, summarize(results))

    if args.dry_run:
        logger.info("Dry-run 完成，未执行任何实际处理。")
        return 0

    overall = summarize(all_results)
    logger.info("流程结束，总计: %s", overall)
    return 1 if overall.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
