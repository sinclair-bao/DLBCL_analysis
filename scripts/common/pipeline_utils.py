#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   pipeline_utils.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    各处理阶段（DICOM 转换 / 预处理 / 分割 / 分析）共用的小工具集合：
        - StageResult：统一的“单个被试 / 单个 Study 处理结果”数据结构，
          所有阶段都用同一种记录格式写日志，方便 main.py 汇总打印。
        - discover_subject_studies()：按本项目约定的目录结构
          data/interim/<PatientID>/<StudyDate>/ 发现所有 (patient_id,
          study_date, study_dir) 三元组，供 preprocess / segmentation 等
          阶段复用，避免每个阶段各写一套目录遍历逻辑。
        - write_stage_log_csv() / setup_logging()：日志落盘与初始化。

    本模块只服务于 DLBCL 项目自身的目录约定（因此放在 scripts/common/ 而不是
    scripts/tools/），不追求跨项目可移植；跨项目可复用的通用能力（如 PACS
    DICOM 发现/转换）见 scripts/tools/pacs_dicom_to_nifti_suv.py。
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from pathlib import Path
from typing import Iterable


@dataclasses.dataclass
class StageResult:
    """记录某个处理阶段对某个 (患者, 检查日期) 的处理结果。"""

    stage: str
    patient_id: str
    study_date: str
    status: str  # "ok" / "skipped" / "warning" / "error"
    output_path: str
    message: str


def discover_subject_studies(interim_root: Path) -> list[tuple[str, str, Path]]:
    """
    遍历 <interim_root>/<PatientID>/<StudyDate>/ 两级目录，返回三元组列表。

    这是本项目 data/interim 的统一组织约定（与
    scripts/tools/pacs_dicom_to_nifti_suv.py 的输出结构一致），后续每个
    处理阶段都基于这一层级发现待处理的对象，不必各自重复实现目录遍历。
    """
    interim_root = Path(interim_root)
    if not interim_root.is_dir():
        return []
    results: list[tuple[str, str, Path]] = []
    for patient_dir in sorted(p for p in interim_root.iterdir() if p.is_dir()):
        for study_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
            results.append((patient_dir.name, study_dir.name, study_dir))
    return results


def write_stage_log_csv(results: Iterable[StageResult], log_path: Path) -> None:
    """把某个阶段的处理结果写入 CSV，便于事后核查 / 补跑失败项。"""
    results = list(results)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in dataclasses.fields(StageResult)]
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))


def summarize(results: Iterable[StageResult]) -> dict[str, int]:
    """统计各状态数量，供阶段结束时打印摘要。"""
    counts = {"ok": 0, "skipped": 0, "warning": 0, "error": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
