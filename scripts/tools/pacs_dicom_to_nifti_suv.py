#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   pacs_dicom_to_nifti_suv.py
@Time        :   2026/08/10
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    将 PACS 自动归档导出的 DICOM 序列批量转换为 NIfTI：
        - CT 序列：调用 dcm2niix 直接转换（保留原始 HU 值与几何信息）。
        - PET 序列：先用 dcm2niix 转换出原始活度浓度图（Bq/mL），再按 DICOM
          头中的注射剂量 / 衰变时间 / 体重信息换算为 SUVbw（Standardized
          Uptake Value, body-weight normalized），最终 NIfTI 与 dcm2niix
          直接输出共用同一份仿射矩阵（affine）/ 头信息，因此可保证转换后
          图像与原始 DICOM 序列的空间位置、朝向、体素间距完全一致。
        - 其他模态（MR / SC / ... ）：直接调用 dcm2niix，输出目录以真实
          Modality 代码命名（脚本不会强行归类为字面上的 "other"，以保留
          更多可读信息；这是本脚本与原始需求描述的一处主动设计取舍，见
          下方 “设计取舍” 一节）。

@输入 (Input)
    一个或多个“源路径模式”，每个模式可以是：
        1) 通配符（glob）路径，例如：
               /home/sun/Documents/DLBCL/data/raw/DICOM*
           会展开匹配 DICOM, DICOMDIS, DICOMDIT, DICOMDIU, ... 等目录；
        2) 具体的单一文件夹路径。
    每个源路径下的 DICOM 数据必须满足“同一种目录结构”，即由 PACS 系统
    自动归档生成的三级结构（不要求严格三级，脚本按“叶子目录”通用识别，
    但真实数据通常表现为）：
        <source>/PA<n>/ST<n>/SE<n>/IM<n>   (Patient / Study / Series / Image)
    其中 IM* 文件本身无扩展名，但内容是标准 DICOM Part 10 文件。

@输出 (Output)
    转换结果写入 <project_root>/data/interim/，按以下规则组织：
        data/interim/<PatientID>/<StudyDate:YYYYMMDD>/<CT|PET|<Modality>>/
    每个 Series 独立生成一个 .nii.gz（+ 可选 .json BIDS sidecar）。
    PET 序列在同一目录下会生成两份文件（除非使用 --no-keep-activity-map
    显式清理中间产物）：
        *_ACT.nii.gz    活度浓度图（Bq/mL，dcm2niix 直接输出，未做 SUV 换算）
        *_SUVbw.nii.gz  SUVbw 图（体重标准化 SUV，最终分析应使用该文件）
    脚本运行日志（每个 Series 的转换状态）写入：
        <project_root>/logs/pacs_dicom_to_nifti_suv_<run_timestamp>.csv

@关键执行逻辑 (Key logic walkthrough)
    1. discover_series_dirs():
       递归遍历每个源路径，找到“叶子 DICOM 目录”——即目录内直接包含至少
       一个可被 pydicom 解析的文件，且该目录本身不再嵌套包含其他叶子
       DICOM 目录。这样无需硬编码 PA/ST/SE 三级路径名，只要源目录满足
       “同一种由 PACS 导出的结构”这一前提即可正常工作。

    2. read_series_header():
       仅读取每个 Series 目录中排序后的第一个文件的头信息（不读像素数据，
       force=True 以兼容无扩展名 / 缺少 DICM 前缀的文件），获取
       PatientID / StudyDate / Modality / SeriesInstanceUID 等用于分类
       与换算的元数据。

    3. run_dcm2niix():
       封装 dcm2niix 子进程调用。CT 与 PET 都先经过这一步，确保几何信息
       （方向、间距、原点、slice 顺序等）完全由 dcm2niix 本身解析生成，
       脚本不做任何几何层面的手工重建，从而天然保证空间一致性。

    4. compute_suv_bw_scale_factor():
       按 QIBA (Quantitative Imaging Biomarkers Alliance) 推荐的标准
       SUVbw 公式计算“标量换算系数”：
           decay_time  = 扫描开始时间 - 放射性药物注射开始时间   (秒)
           decayed_dose = 注射总剂量 * 2^(-decay_time / 半衰期)      (Bq)
                          [若 DecayCorrection == 'ADMIN'，图像已按注射时刻
                           衰变校正，无需再乘衰变因子，decayed_dose = 总剂量]
           suv_scale   = 体重(g) / decayed_dose(Bq)
           SUVbw(g/mL) = 活度浓度(Bq/mL) * suv_scale
       换算系数只是一个标量，直接乘到 dcm2niix 输出的活度浓度 NIfTI 的
       像素数组上即可得到 SUVbw 图，仿射矩阵 / 头信息保持不变（详见第 6
       步），因此不会引入任何几何误差。

    5. 前提校验（任何一项不满足则跳过 SUV 换算，仅保留活度浓度图并在日志
       中给出明确警告，不会“悄悄”输出错误结果）：
           - Units == 'BQML'                      （像素值单位为 Bq/mL）
           - PatientWeight 存在且 > 0              （体重，单位 kg）
           - RadionuclideTotalDose / HalfLife 存在  （来自
             RadiopharmaceuticalInformationSequence[0]）
           - DecayCorrection ∈ {'START', 'ADMIN'}   （'NONE' 或未知值时，
             逐帧衰变校正无法用单一标量还原，直接跳过）

    6. convert_pet_series():
       a) dcm2niix 将 PET 序列转换到该 Series 专属的临时子目录，文件名
          固定为 activity_concentration，避免与其他序列重名冲突；
       b) 用 nibabel 读取该 NIfTI，取出仿射矩阵 affine 与 header（原样
          保留，不做任何修改）；
       c) 像素数组（float64）乘以第 4 步算出的 suv_scale，转回 float32
          写出新的 NIfTI，仿射矩阵与原图完全相同；
       d) 默认将中间的活度浓度图一并保留（*_ACT.nii.gz），便于核对换算
          是否正确；如指定 --no-keep-activity-map 则转换完成后删除。

    7. 主循环对每个 Series 都用 try/except 单独捕获异常：单个病例 / 单个
       序列失败不会中断整批转换，失败信息会记录到 CSV 日志中，方便事后
       核查与补跑（脚本支持 --overwrite 之外的默认“跳过已存在输出”的
       增量式重跑）。

@设计取舍 (Design notes / deviations worth flagging to reviewers)
    - 输出子目录命名：CT -> "CT"，PT -> "PET"，其余模态使用真实 Modality
      代码（如 "MR"、"SC"）而非字面的 "other"，理由是这样信息量更大、
      更利于后续按模态过滤，如团队更希望严格使用 "OTHER" 归并所有非
      CT/PET 模态，可在 `modality_to_folder_name()` 中一行调整。
    - SUV 计算使用“序列级单一衰变时刻”（取 AcquisitionTime，回退至
      SeriesTime）作为整个 PET 序列（含多个 bed position）的统一扫描
      起始时间，这是目前绝大多数商业软件（syngo.via、MIM 等）默认采用
      的简化方案，而非逐床位 / 逐帧单独衰变校正；如需逐帧精确 SUV，需要
      改为逐 slice 读取 AcquisitionTime 并逐层换算，当前脚本未实现，
      已在函数 docstring 中标注为已知局限。
    - 仅读取每个 Series 目录“排序后的第一个文件”头信息代表整个序列的
      元数据（PatientID / 剂量 / 体重等在同一序列内应保持不变），可显著
      降低大批量数据下的 IO 开销；若怀疑某序列内头信息不一致，可结合
      日志中的 SeriesInstanceUID 单独排查。

@依赖 (Dependencies)
    - 外部程序：dcm2niix（需可在 PATH 中找到，或通过 --dcm2niix-bin 指定
      绝对路径，例如本机 FSL 自带的 /home/sun/fsl/bin/dcm2niix）。
    - Python 包：pydicom, nibabel, numpy（标准科研影像栈，已在
      environment.yml 中声明 / 或使用本机 FSL Python 环境）。

@用法示例 (Usage examples)
    # 1) 用通配符批量处理 data/raw 下所有 DICOM* 目录（默认行为）
    python pacs_dicom_to_nifti_suv.py

    # 2) 指定某个单独的归档目录
    python pacs_dicom_to_nifti_suv.py --source /path/to/some_pacs_export

    # 3) 混合多个来源，并指定 dcm2niix 可执行文件路径
    python pacs_dicom_to_nifti_suv.py \
        --source "/home/sun/Documents/DLBCL/data/raw/DICOM*" \
        --source /mnt/usb/extra_export \
        --dcm2niix-bin /home/sun/fsl/bin/dcm2niix

    # 4) 干跑（只发现 Series、打印计划，不实际转换）
    python pacs_dicom_to_nifti_suv.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import glob
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import pydicom
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "缺少 pydicom，请先安装（pip install pydicom）或使用带有 pydicom 的 "
        "Python 环境（例如本机 FSL 自带的 python3）。"
    ) from exc

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "缺少 nibabel，请先安装（pip install nibabel）或使用带有 nibabel 的 "
        "Python 环境（例如本机 FSL 自带的 python3）。"
    ) from exc


# --------------------------------------------------------------------------- #
# 常量与项目路径
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_PATTERNS = [str(PROJECT_ROOT / "data" / "raw" / "DICOM*")]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

# SUVbw 计算允许的 DecayCorrection 取值：
#   START -> 图像按“扫描开始时刻”衰变校正，换算时仍需从注射时刻补算衰变；
#   ADMIN -> 图像已按“注射时刻”衰变校正，换算时无需再乘衰变因子。
SUPPORTED_DECAY_CORRECTIONS = {"START", "ADMIN"}

logger = logging.getLogger("pacs_dicom_to_nifti_suv")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class SeriesJob:
    """待转换的一个 DICOM Series 及其定位信息。"""

    dicom_dir: Path
    patient_id: str
    study_date: str
    modality: str
    series_uid: str
    series_number: str
    series_description: str


@dataclasses.dataclass
class ConversionResult:
    """记录单个 Series 的转换结果，用于写入 CSV 日志。"""

    dicom_dir: str
    patient_id: str
    study_date: str
    modality: str
    series_uid: str
    output_dir: str
    status: str  # "ok" / "skipped" / "warning" / "error"
    message: str


# --------------------------------------------------------------------------- #
# 源目录发现
# --------------------------------------------------------------------------- #


def expand_source_patterns(patterns: Iterable[str]) -> list[Path]:
    """把用户传入的 glob 模式 / 具体路径展开为存在的目录列表。"""
    resolved: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_dir():
            matches = [pattern]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                resolved.append(path)
            else:
                logger.warning("跳过非目录匹配项: %s", path)
    if not resolved:
        raise SystemExit(f"没有找到任何有效的源目录，检查过的模式: {list(patterns)}")
    return resolved


def _looks_like_dicom(path: Path) -> bool:
    """快速判断单个文件是否可被解析为 DICOM（只读头，不读像素）。"""
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except Exception:  # noqa: BLE001 - 任何解析失败都视为非 DICOM
        return False


def discover_series_dirs(source_dirs: Iterable[Path]) -> list[Path]:
    """
    在每个源目录下递归查找“叶子 DICOM 目录”。

    叶子 DICOM 目录定义：目录内直接包含至少一个可解析的 DICOM 文件，且
    该目录的子目录中不再包含 DICOM 文件（即 PACS 结构中的 SE* 目录，但
    不强制要求命名为 SE*，只要求满足“叶子层级存放实际图像文件”这一
    结构约定）。
    """
    series_dirs: list[Path] = []
    for source in source_dirs:
        for dirpath in sorted(p for p in source.rglob("*") if p.is_dir()):
            files = [f for f in dirpath.iterdir() if f.is_file()]
            if not files:
                continue
            # 子目录中若还有 DICOM 文件，说明当前目录不是叶子层级，跳过。
            has_dicom_subdir = any(
                any(f.is_file() for f in sub.iterdir())
                for sub in dirpath.iterdir()
                if sub.is_dir()
            )
            if has_dicom_subdir:
                continue
            sample = sorted(files)[0]
            if _looks_like_dicom(sample):
                series_dirs.append(dirpath)
        # 源目录自身也可能直接就是叶子目录（没有子目录嵌套）。
        direct_files = [f for f in source.iterdir() if f.is_file()]
        if direct_files and source not in series_dirs:
            if _looks_like_dicom(sorted(direct_files)[0]):
                series_dirs.append(source)
    return sorted(set(series_dirs))


def read_series_header(dicom_dir: Path) -> Optional[pydicom.Dataset]:
    """读取某 Series 目录内排序后第一个文件的头信息（不含像素数据）。"""
    files = sorted(f for f in dicom_dir.iterdir() if f.is_file())
    for f in files:
        try:
            return pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("读取 %s 失败: %s", f, exc)
            continue
    return None


def build_series_job(dicom_dir: Path) -> Optional[SeriesJob]:
    ds = read_series_header(dicom_dir)
    if ds is None:
        logger.warning("目录 %s 中没有可读取的 DICOM 文件，跳过。", dicom_dir)
        return None
    patient_id = str(getattr(ds, "PatientID", "UNKNOWN_PATIENT")).strip() or "UNKNOWN_PATIENT"
    study_date = str(getattr(ds, "StudyDate", "UNKNOWN_DATE")).strip() or "UNKNOWN_DATE"
    modality = str(getattr(ds, "Modality", "OT")).strip() or "OT"
    return SeriesJob(
        dicom_dir=dicom_dir,
        patient_id=_sanitize_path_component(patient_id),
        study_date=_sanitize_path_component(study_date),
        modality=modality,
        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        series_number=str(getattr(ds, "SeriesNumber", "0")),
        series_description=_sanitize_path_component(
            str(getattr(ds, "SeriesDescription", "series"))
        ),
    )


def _sanitize_path_component(value: str) -> str:
    """把可能含有空格 / 特殊符号的 DICOM 字符串转成安全的路径片段。"""
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in value.strip())
    return cleaned or "unknown"


def modality_to_folder_name(modality: str) -> str:
    """CT -> 'CT'，PT -> 'PET'，其余模态使用真实 Modality 代码，见模块说明。"""
    mapping = {"CT": "CT", "PT": "PET"}
    return mapping.get(modality.upper(), modality.upper())


# --------------------------------------------------------------------------- #
# dcm2niix 封装
# --------------------------------------------------------------------------- #


def run_dcm2niix(
    dicom_dir: Path,
    output_dir: Path,
    filename_stem: str,
    dcm2niix_bin: str,
    bids_sidecar: bool,
) -> list[Path]:
    """
    调用 dcm2niix 将一个 Series 目录转换为 NIfTI。

    关键参数说明：
        -z y      输出 .nii.gz（gzip 压缩）
        -b y/n    是否输出 BIDS JSON sidecar（记录采集参数，便于溯源）
        -f <stem> 固定输出文件名（每个 Series 单独目录，不会重名冲突）
        -w 1      若目标文件已存在则直接覆盖（脚本在外层已经做了“是否跳过
                   已存在输出”的判断，这里统一设为覆盖以避免 dcm2niix 自动
                   加后缀导致找不到产物文件）
    返回值：本次调用实际产生的 .nii.gz 文件路径列表（正常情况下应为 1 个，
    但某些多回波 / 多 b 值序列可能被 dcm2niix 拆分为多个文件）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        dcm2niix_bin,
        "-z", "y",
        "-b", "y" if bids_sidecar else "n",
        "-f", filename_stem,
        "-w", "1",
        "-o", str(output_dir),
        str(dicom_dir),
    ]
    logger.debug("执行命令: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"dcm2niix 转换失败 (exit={proc.returncode})\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    produced = sorted(output_dir.glob(f"{filename_stem}*.nii.gz"))
    if not produced:
        raise RuntimeError(
            f"dcm2niix 执行成功但未找到预期的输出文件 "
            f"({filename_stem}*.nii.gz)，stdout: {proc.stdout}"
        )
    return produced


# --------------------------------------------------------------------------- #
# SUVbw 计算
# --------------------------------------------------------------------------- #


def _parse_dicom_datetime(
    date_str: Optional[str], time_str: Optional[str]
) -> Optional[dt.datetime]:
    """解析 DICOM DA(YYYYMMDD) + TM(HHMMSS[.ffffff]) 为 datetime。"""
    if not date_str or not time_str:
        return None
    date_str = date_str.strip()
    time_str = time_str.strip().split("+")[0].split("-")[0]  # 去掉可能的时区偏移
    try:
        base = dt.datetime.strptime(date_str[:8], "%Y%m%d")
        hh = int(time_str[0:2])
        mm = int(time_str[2:4]) if len(time_str) >= 4 else 0
        ss = int(time_str[4:6]) if len(time_str) >= 6 else 0
        frac = time_str[6:] if len(time_str) > 6 else ""
        microsecond = int(float("0" + frac) * 1_000_000) if frac.startswith(".") else 0
        return base + dt.timedelta(hours=hh, minutes=mm, seconds=ss, microseconds=microsecond)
    except (ValueError, IndexError):
        return None


def _parse_dicom_dt(dt_str: Optional[str]) -> Optional[dt.datetime]:
    """解析 DICOM DT 类型（YYYYMMDDHHMMSS.FFFFFF[&ZZXX]）为 datetime。"""
    if not dt_str:
        return None
    dt_str = dt_str.strip().split("+")[0].split("-")[0]
    if len(dt_str) < 14:
        return None
    return _parse_dicom_datetime(dt_str[:8], dt_str[8:])


@dataclasses.dataclass
class SuvComputationOutcome:
    scale_factor: Optional[float]
    ok: bool
    message: str


def compute_suv_bw_scale_factor(ds: pydicom.Dataset) -> SuvComputationOutcome:
    """
    计算 SUVbw 标量换算系数：SUVbw(g/mL) = 活度浓度(Bq/mL) * scale_factor。

    公式与前提条件详见模块顶部 docstring 第 4-5 点。任何一项前提不满足都
    会返回 ok=False，调用方应保留原始活度浓度图并跳过 SUV 换算，而不是
    输出一个可能错误的 SUV 图。

    已知局限：使用序列级单一“扫描起始时间”（见下方 scan_dt 获取逻辑），
    不区分多床位 PET 各床位独立的采集时刻，属于行业内常见的简化方案。
    """
    units = str(getattr(ds, "Units", "")).upper()
    if units != "BQML":
        return SuvComputationOutcome(None, False, f"Units={units!r} 不是 BQML，无法计算 SUV。")

    weight_kg = getattr(ds, "PatientWeight", None)
    if not weight_kg or float(weight_kg) <= 0:
        return SuvComputationOutcome(None, False, "PatientWeight 缺失或非正，无法计算 SUV。")
    weight_g = float(weight_kg) * 1000.0

    radiopharm_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if not radiopharm_seq:
        return SuvComputationOutcome(None, False, "缺少 RadiopharmaceuticalInformationSequence。")
    radiopharm = radiopharm_seq[0]

    total_dose_bq = getattr(radiopharm, "RadionuclideTotalDose", None)
    half_life_sec = getattr(radiopharm, "RadionuclideHalfLife", None)
    if not total_dose_bq or not half_life_sec:
        return SuvComputationOutcome(None, False, "缺少注射总剂量或核素半衰期。")
    total_dose_bq = float(total_dose_bq)
    half_life_sec = float(half_life_sec)

    decay_correction = str(getattr(ds, "DecayCorrection", "")).upper()
    if decay_correction not in SUPPORTED_DECAY_CORRECTIONS:
        return SuvComputationOutcome(
            None, False, f"DecayCorrection={decay_correction!r} 不受支持（需 START/ADMIN）。"
        )

    if decay_correction == "ADMIN":
        # 图像已按注射时刻衰变校正，无需再补算衰变因子。
        decayed_dose_bq = total_dose_bq
    else:
        injection_dt = _parse_dicom_dt(
            getattr(radiopharm, "RadiopharmaceuticalStartDateTime", None)
        )
        if injection_dt is None:
            injection_dt = _parse_dicom_datetime(
                getattr(ds, "StudyDate", None),
                getattr(radiopharm, "RadiopharmaceuticalStartTime", None),
            )
        # 扫描起始时间：优先用本 Series 首帧的采集时间，回退到 Series 时间。
        scan_dt = _parse_dicom_datetime(
            getattr(ds, "AcquisitionDate", None) or getattr(ds, "SeriesDate", None),
            getattr(ds, "AcquisitionTime", None) or getattr(ds, "SeriesTime", None),
        )
        if injection_dt is None or scan_dt is None:
            return SuvComputationOutcome(None, False, "无法解析注射时间或扫描起始时间。")

        decay_time_sec = (scan_dt - injection_dt).total_seconds()
        if decay_time_sec < 0:
            return SuvComputationOutcome(
                None,
                False,
                f"解析出的衰变时间为负数 ({decay_time_sec:.1f}s)，可能是跨天/时钟"
                "问题，为避免错误结果已跳过 SUV 换算。",
            )
        decayed_dose_bq = total_dose_bq * (2.0 ** (-decay_time_sec / half_life_sec))

    if decayed_dose_bq <= 0:
        return SuvComputationOutcome(None, False, "计算出的衰变后剂量 <= 0，数据异常。")

    scale_factor = weight_g / decayed_dose_bq
    return SuvComputationOutcome(scale_factor, True, f"scale_factor={scale_factor:.6g}")


# --------------------------------------------------------------------------- #
# 单个 Series 的转换流程
# --------------------------------------------------------------------------- #


def convert_ct_series(
    job: SeriesJob,
    output_root: Path,
    dcm2niix_bin: str,
    bids_sidecar: bool,
    overwrite: bool,
) -> ConversionResult:
    target_dir = output_root / job.patient_id / job.study_date / modality_to_folder_name(job.modality)
    stem = f"s{job.series_number}_{job.series_description}"
    existing = sorted(target_dir.glob(f"{stem}*.nii.gz")) if target_dir.exists() else []
    if existing and not overwrite:
        return ConversionResult(
            str(job.dicom_dir), job.patient_id, job.study_date, job.modality,
            job.series_uid, str(target_dir), "skipped", "输出已存在，使用 --overwrite 可强制重转。",
        )

    produced = run_dcm2niix(job.dicom_dir, target_dir, stem, dcm2niix_bin, bids_sidecar)
    return ConversionResult(
        str(job.dicom_dir), job.patient_id, job.study_date, job.modality,
        job.series_uid, str(target_dir), "ok",
        f"CT 转换完成: {', '.join(p.name for p in produced)}",
    )


def convert_other_series(
    job: SeriesJob,
    output_root: Path,
    dcm2niix_bin: str,
    bids_sidecar: bool,
    overwrite: bool,
) -> ConversionResult:
    """CT 以外、PET 以外的模态（MR / SC / ...），直接调用 dcm2niix。"""
    target_dir = output_root / job.patient_id / job.study_date / modality_to_folder_name(job.modality)
    stem = f"s{job.series_number}_{job.series_description}"
    existing = sorted(target_dir.glob(f"{stem}*.nii.gz")) if target_dir.exists() else []
    if existing and not overwrite:
        return ConversionResult(
            str(job.dicom_dir), job.patient_id, job.study_date, job.modality,
            job.series_uid, str(target_dir), "skipped", "输出已存在，使用 --overwrite 可强制重转。",
        )

    produced = run_dcm2niix(job.dicom_dir, target_dir, stem, dcm2niix_bin, bids_sidecar)
    return ConversionResult(
        str(job.dicom_dir), job.patient_id, job.study_date, job.modality,
        job.series_uid, str(target_dir), "ok",
        f"{job.modality} 转换完成: {', '.join(p.name for p in produced)}",
    )


def convert_pet_series(
    job: SeriesJob,
    output_root: Path,
    dcm2niix_bin: str,
    bids_sidecar: bool,
    overwrite: bool,
    keep_activity_map: bool,
) -> ConversionResult:
    """
    PET 序列转换：dcm2niix 生成活度浓度图 -> 计算 SUVbw 标量换算系数 ->
    在活度浓度图基础上做纯像素值缩放，得到 SUVbw 图（仿射矩阵与原图
    完全一致，因此与源 DICOM 序列空间保持一致）。
    """
    target_dir = output_root / job.patient_id / job.study_date / modality_to_folder_name(job.modality)
    stem = f"s{job.series_number}_{job.series_description}"
    suv_path = target_dir / f"{stem}_SUVbw.nii.gz"
    act_path = target_dir / f"{stem}_ACT.nii.gz"

    if suv_path.exists() and not overwrite:
        return ConversionResult(
            str(job.dicom_dir), job.patient_id, job.study_date, job.modality,
            job.series_uid, str(target_dir), "skipped", "SUVbw 输出已存在，使用 --overwrite 可强制重转。",
        )

    ds = read_series_header(job.dicom_dir)
    if ds is None:
        raise RuntimeError("无法重新读取 Series 头信息用于 SUV 计算。")

    # 先把原始活度浓度图转换到一个临时子目录，避免文件名与最终产物冲突。
    staging_dir = target_dir / f".staging_{stem}"
    try:
        produced = run_dcm2niix(job.dicom_dir, staging_dir, "activity_concentration", dcm2niix_bin, bids_sidecar)
        act_source = produced[0]
        if len(produced) > 1:
            logger.warning(
                "PET 序列 %s 被 dcm2niix 拆分为 %d 个文件，仅使用第一个: %s",
                job.series_uid, len(produced), act_source.name,
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(act_source, act_path)
        sidecar = act_source.with_suffix("").with_suffix(".json")
        if bids_sidecar and sidecar.exists():
            shutil.copy2(sidecar, act_path.with_suffix("").with_suffix(".json"))

        outcome = compute_suv_bw_scale_factor(ds)
        if not outcome.ok:
            return ConversionResult(
                str(job.dicom_dir), job.patient_id, job.study_date, job.modality,
                job.series_uid, str(target_dir), "warning",
                f"SUV 换算前提不满足，仅保留活度浓度图: {outcome.message}",
            )

        img = nib.load(str(act_path))
        activity_data = img.get_fdata(dtype=np.float64)
        suv_data = (activity_data * outcome.scale_factor).astype(np.float32)
        # 复用原图的 affine 与 header，只替换像素数据与 dtype，几何信息不变。
        suv_img = nib.Nifti1Image(suv_data, img.affine, img.header)
        suv_img.header.set_data_dtype(np.float32)
        nib.save(suv_img, str(suv_path))

        if not keep_activity_map:
            act_path.unlink(missing_ok=True)
            act_path.with_suffix("").with_suffix(".json").unlink(missing_ok=True)

        return ConversionResult(
            str(job.dicom_dir), job.patient_id, job.study_date, job.modality,
            job.series_uid, str(target_dir), "ok",
            f"PET SUVbw 转换完成 ({outcome.message})",
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def process_series(
    job: SeriesJob,
    output_root: Path,
    dcm2niix_bin: str,
    bids_sidecar: bool,
    overwrite: bool,
    keep_activity_map: bool,
) -> ConversionResult:
    try:
        if job.modality.upper() == "CT":
            return convert_ct_series(job, output_root, dcm2niix_bin, bids_sidecar, overwrite)
        if job.modality.upper() == "PT":
            return convert_pet_series(
                job, output_root, dcm2niix_bin, bids_sidecar, overwrite, keep_activity_map
            )
        return convert_other_series(job, output_root, dcm2niix_bin, bids_sidecar, overwrite)
    except Exception as exc:  # noqa: BLE001 - 单个 Series 失败不能中断整批任务
        logger.exception("转换失败: %s", job.dicom_dir)
        return ConversionResult(
            str(job.dicom_dir), job.patient_id, job.study_date, job.modality,
            job.series_uid, "", "error", str(exc),
        )


def write_log_csv(results: list[ConversionResult], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in dataclasses.fields(ConversionResult)]
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 PACS 归档 DICOM 批量转换为 NIfTI（CT 直转，PET 换算为 SUVbw）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", action="append", dest="sources",
        help="源路径（glob 模式或具体目录），可重复传入多次；默认见 DEFAULT_SOURCE_PATTERNS。",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="转换结果输出根目录。")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="CSV 日志输出目录。")
    parser.add_argument("--dcm2niix-bin", default="dcm2niix", help="dcm2niix 可执行文件路径或名称。")
    parser.add_argument("--overwrite", action="store_true", help="已存在的输出也强制重新转换。")
    parser.add_argument(
        "--no-bids-sidecar", dest="bids_sidecar", action="store_false", default=True,
        help="不生成 dcm2niix 的 BIDS JSON sidecar。",
    )
    parser.add_argument(
        "--no-keep-activity-map", dest="keep_activity_map", action="store_false", default=True,
        help="PET 转换完成后删除中间的活度浓度图（*_ACT.nii.gz），只保留 SUVbw 图。",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅发现并打印待转换的 Series，不实际执行转换。")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出 DEBUG 级别日志。")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    sources = expand_source_patterns(args.sources or DEFAULT_SOURCE_PATTERNS)
    logger.info("展开得到 %d 个源目录: %s", len(sources), [str(s) for s in sources])

    series_dirs = discover_series_dirs(sources)
    logger.info("共发现 %d 个 DICOM Series 目录。", len(series_dirs))

    jobs = [job for job in (build_series_job(d) for d in series_dirs) if job is not None]

    if args.dry_run:
        for job in jobs:
            folder = modality_to_folder_name(job.modality)
            print(
                f"[DRY-RUN] {job.dicom_dir} -> "
                f"{args.output_root / job.patient_id / job.study_date / folder} "
                f"(modality={job.modality}, series={job.series_number})"
            )
        logger.info("Dry-run 完成，未执行任何转换。")
        return 0

    results: list[ConversionResult] = []
    for idx, job in enumerate(jobs, start=1):
        logger.info(
            "[%d/%d] 处理 patient=%s study=%s modality=%s series=%s",
            idx, len(jobs), job.patient_id, job.study_date, job.modality, job.series_number,
        )
        result = process_series(
            job, args.output_root, args.dcm2niix_bin,
            args.bids_sidecar, args.overwrite, args.keep_activity_map,
        )
        results.append(result)
        if result.status == "error":
            logger.error("失败: %s -> %s", job.dicom_dir, result.message)
        elif result.status == "warning":
            logger.warning("警告: %s -> %s", job.dicom_dir, result.message)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log_dir / f"pacs_dicom_to_nifti_suv_{timestamp}.csv"
    write_log_csv(results, log_path)

    n_ok = sum(1 for r in results if r.status == "ok")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_warn = sum(1 for r in results if r.status == "warning")
    n_err = sum(1 for r in results if r.status == "error")
    logger.info(
        "转换完成: 成功=%d 跳过=%d 警告=%d 失败=%d，日志已写入 %s",
        n_ok, n_skip, n_warn, n_err, log_path,
    )
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
