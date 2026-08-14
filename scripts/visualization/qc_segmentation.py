#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   qc_segmentation.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    为每个完成 nnU-Net 分割的病例生成质控（QC）图。
    每张图含三列：
        1. PET 全身 MIP（冠状面最大强度投影）
        2. nnU-Net 病灶 mask MIP
        3. 病灶 mask 叠加在 PET MIP 上

    同时生成矢状面（sagittal）三列图，保存为第二行，最终输出 6-panel PNG。

@输入
    data/interim/<PatientID>/<StudyDate>/preprocessed/PET/*_SUVbw.nii.gz
    data/processed/<PatientID>/<StudyDate>/masks/<PatientID>_<StudyDate>_lesion.nii.gz

@输出
    data/qc/<PatientID>/<StudyDate>_qc.png

@用法
    # 全部病例
    python scripts/visualization/qc_segmentation.py

    # 指定病例
    python scripts/visualization/qc_segmentation.py --patient-id 00136597 --study-date 20220425

    # 覆盖已有图
    python scripts/visualization/qc_segmentation.py --overwrite

    # 输出目录自定义
    python scripts/visualization/qc_segmentation.py --qc-root data/qc
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
except ImportError as exc:
    raise SystemExit("缺少 matplotlib，请在 data-analysis 环境中运行。") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from pipeline_utils import setup_logging  # noqa: E402

logger = logging.getLogger("qc_segmentation")

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_QC_ROOT = PROJECT_ROOT / "data" / "qc"

# SUV 显示范围（裁剪膀胱等高摄取器官的极值，保留病灶对比度）
SUV_DISPLAY_MAX = 6.0
SUV_DISPLAY_MIN = 0.0

# 病灶 overlay 颜色（RGBA：亮红色，半透明）
LESION_COLOR = (1.0, 0.15, 0.15)  # RGB


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _load_pet_mask(
    interim_root: Path, processed_root: Path, patient_id: str, study_date: str
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    加载预处理 PET 和 lesion mask，返回 (pet_data, mask_data)。
    PET shape 应与 mask shape 一致（均为 preprocessed 后的 RAS 图像）。
    """
    pet_dir = interim_root / patient_id / study_date / "preprocessed" / "PET"
    mask_path = (
        processed_root / patient_id / study_date / "masks"
        / f"{patient_id}_{study_date}_lesion.nii.gz"
    )

    # 找 PET（优先 SUVbw）
    pet_files = sorted(pet_dir.glob("*_SUVbw.nii.gz")) if pet_dir.is_dir() else []
    if not pet_files:
        pet_files = sorted(pet_dir.glob("*.nii.gz")) if pet_dir.is_dir() else []
    if not pet_files:
        logger.warning("未找到预处理 PET：%s/%s", patient_id, study_date)
        return None, None
    if not mask_path.exists():
        logger.warning("未找到 lesion mask：%s/%s", patient_id, study_date)
        return None, None

    pet_img = nib.load(str(pet_files[0]))
    mask_img = nib.load(str(mask_path))

    pet_data = np.asarray(pet_img.dataobj, dtype=np.float32)
    mask_data = np.asarray(mask_img.dataobj, dtype=np.uint8)

    if pet_data.shape != mask_data.shape:
        logger.warning(
            "%s/%s shape 不一致：PET=%s mask=%s，跳过。",
            patient_id, study_date, pet_data.shape, mask_data.shape,
        )
        return None, None

    return pet_data, mask_data


def _compute_mip(volume: np.ndarray, axis: int) -> np.ndarray:
    """沿指定轴计算最大强度投影（MIP）。"""
    return np.max(volume, axis=axis)


def _orient_for_display(mip: np.ndarray, view: str) -> np.ndarray:
    """
    将 MIP 转换为标准放射学显示方向：头在上，冠状面患者右侧在图像左侧。
    图像已是 RAS 方向：
        coronal MIP (axis=1, A→P):  输出形状 (R, S)
            转置→(S, R)，flip S 使头朝上，flip R 使患者右侧在图像左侧（放射学惯例）
        sagittal MIP (axis=0, R→L): 输出形状 (A, S)
            转置→(S, A)，flip S 使头朝上，保持 A 轴（前方在右）
    """
    if view == "coronal":
        # 转置 + 头朝上 + 水平镜像（放射学惯例：患者右在图像左）
        img = mip.T[::-1, ::-1]
    else:  # sagittal
        # 转置 + 头朝上，前方保持在右侧
        img = mip.T[::-1, :]
    return img


def _add_lesion_overlay(ax: plt.Axes, pet_mip: np.ndarray, mask_mip: np.ndarray) -> None:
    """在已显示的 PET MIP 上叠加病灶轮廓/填充。"""
    if mask_mip.max() == 0:
        return
    # 半透明红色填充
    rgba = np.zeros((*mask_mip.shape, 4), dtype=np.float32)
    rgba[..., 0] = LESION_COLOR[0]
    rgba[..., 1] = LESION_COLOR[1]
    rgba[..., 2] = LESION_COLOR[2]
    rgba[..., 3] = np.where(mask_mip > 0, 0.55, 0.0).astype(np.float32)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")


def _pet_colormap() -> matplotlib.colors.Colormap:
    """
    模拟临床 PET 伪彩色（黑→紫→蓝→绿→黄→红→白）。
    """
    colors = [
        (0.0,  (0.00, 0.00, 0.00)),
        (0.12, (0.25, 0.00, 0.50)),
        (0.25, (0.00, 0.00, 1.00)),
        (0.45, (0.00, 0.75, 0.75)),
        (0.60, (0.00, 1.00, 0.00)),
        (0.75, (1.00, 1.00, 0.00)),
        (0.90, (1.00, 0.40, 0.00)),
        (1.00, (1.00, 1.00, 1.00)),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pet_clinical", [(v, c) for v, c in colors]
    )
    return cmap


PET_CMAP = _pet_colormap()
MASK_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "mask_mono", [(0, (0, 0, 0)), (1, (1, 1, 1))]
)


# ---------------------------------------------------------------------------
# 主绘图函数
# ---------------------------------------------------------------------------

def make_qc_figure(
    pet_data: np.ndarray,
    mask_data: np.ndarray,
    patient_id: str,
    study_date: str,
    out_path: Path,
) -> None:
    """
    生成 2×3 QC 图（冠状面 + 矢状面各三列）并保存。
    列：① PET MIP  ② Lesion mask MIP  ③ 叠加图
    """
    lesion_voxels = int(mask_data.sum())

    views = [
        ("coronal",  1, "Coronal (AP)"),
        ("sagittal", 0, "Sagittal (RL)"),
    ]

    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        figsize=(15, 20),
        facecolor="black",
        gridspec_kw={"hspace": 0.04, "wspace": 0.03},
    )

    col_titles = ["PET MIP", "Lesion Mask", "Overlay"]
    pet_clip = np.clip(pet_data, SUV_DISPLAY_MIN, SUV_DISPLAY_MAX)

    for row_idx, (view, axis, view_label) in enumerate(views):
        pet_mip_raw = _compute_mip(pet_clip, axis=axis)
        mask_mip_raw = _compute_mip(mask_data.astype(np.float32), axis=axis)

        pet_mip = _orient_for_display(pet_mip_raw, view)
        mask_mip = _orient_for_display(mask_mip_raw, view)

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col_idx == 0:
                # PET MIP
                ax.imshow(pet_mip, cmap=PET_CMAP, vmin=SUV_DISPLAY_MIN,
                          vmax=SUV_DISPLAY_MAX, aspect="auto", interpolation="bilinear")
                if row_idx == 0:
                    ax.set_title(col_titles[0], color="white", fontsize=12, pad=4)
                ax.set_ylabel(view_label, color="white", fontsize=10, labelpad=4)

            elif col_idx == 1:
                # Mask MIP（白色病灶，黑底）
                ax.imshow(pet_mip, cmap="gray",
                          vmin=SUV_DISPLAY_MIN, vmax=SUV_DISPLAY_MAX,
                          aspect="auto", interpolation="bilinear", alpha=0.35)
                ax.imshow(mask_mip, cmap=MASK_CMAP, vmin=0, vmax=1,
                          aspect="auto", interpolation="nearest")
                if row_idx == 0:
                    ax.set_title(col_titles[1], color="white", fontsize=12, pad=4)

            else:
                # Overlay
                ax.imshow(pet_mip, cmap=PET_CMAP, vmin=SUV_DISPLAY_MIN,
                          vmax=SUV_DISPLAY_MAX, aspect="auto", interpolation="bilinear")
                _add_lesion_overlay(ax, pet_mip, mask_mip)
                if row_idx == 0:
                    ax.set_title(col_titles[2], color="white", fontsize=12, pad=4)

    # 标题与元信息
    suv_max = float(pet_data.max())
    title_str = (
        f"Patient: {patient_id}   Study: {study_date}\n"
        f"Lesion voxels: {lesion_voxels:,}   "
        f"SUV max (raw): {suv_max:.1f}   "
        f"Display: {SUV_DISPLAY_MIN}–{SUV_DISPLAY_MAX} SUV"
    )
    fig.suptitle(title_str, color="white", fontsize=11, y=0.995,
                 fontfamily="monospace")

    # 颜色条
    sm = plt.cm.ScalarMappable(
        cmap=PET_CMAP,
        norm=mcolors.Normalize(vmin=SUV_DISPLAY_MIN, vmax=SUV_DISPLAY_MAX),
    )
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.75])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("SUVbw", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # 图例
    legend_elements = [
        Patch(facecolor=LESION_COLOR, alpha=0.8, label="nnU-Net lesion"),
    ]
    axes[0, 2].legend(
        handles=legend_elements, loc="lower right",
        facecolor="black", edgecolor="gray",
        labelcolor="white", fontsize=8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor="black", pad_inches=0.05)
    plt.close(fig)
    logger.info("已保存 QC 图：%s  (lesion=%d voxels)", out_path.name, lesion_voxels)


# ---------------------------------------------------------------------------
# 发现待处理病例
# ---------------------------------------------------------------------------

def discover_cases(
    processed_root: Path,
    patient_id: Optional[str] = None,
    study_date: Optional[str] = None,
) -> list[tuple[str, str]]:
    """扫描 processed_root，找到含有 _lesion.nii.gz 的病例。"""
    cases: list[tuple[str, str]] = []
    if patient_id and study_date:
        mask = (
            processed_root / patient_id / study_date / "masks"
            / f"{patient_id}_{study_date}_lesion.nii.gz"
        )
        if mask.exists():
            return [(patient_id, study_date)]
        else:
            logger.error("找不到 lesion mask：%s/%s", patient_id, study_date)
            return []

    for pid_dir in sorted(processed_root.iterdir()):
        if not pid_dir.is_dir():
            continue
        for date_dir in sorted(pid_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            mask = date_dir / "masks" / f"{pid_dir.name}_{date_dir.name}_lesion.nii.gz"
            if mask.exists():
                cases.append((pid_dir.name, date_dir.name))
    return cases


# ---------------------------------------------------------------------------
# 批量运行
# ---------------------------------------------------------------------------

def run(
    interim_root: Path,
    processed_root: Path,
    qc_root: Path,
    overwrite: bool = False,
    patient_id: Optional[str] = None,
    study_date: Optional[str] = None,
) -> dict[str, int]:
    """生成所有病例的 QC 图，返回统计字典。"""
    cases = discover_cases(processed_root, patient_id, study_date)
    total = len(cases)
    stats = {"ok": 0, "skipped": 0, "error": 0}

    logger.info("发现 %d 个待生成 QC 图的病例", total)

    for idx, (pid, date) in enumerate(cases, 1):
        out_path = qc_root / pid / f"{pid}_{date}_qc.png"
        if out_path.exists() and not overwrite:
            logger.info("[%d/%d] 跳过（已存在）：%s/%s", idx, total, pid, date)
            stats["skipped"] += 1
            continue

        logger.info("[%d/%d] 生成 QC：%s/%s", idx, total, pid, date)
        try:
            pet_data, mask_data = _load_pet_mask(
                interim_root, processed_root, pid, date
            )
            if pet_data is None:
                stats["error"] += 1
                continue
            make_qc_figure(pet_data, mask_data, pid, date, out_path)
            stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("生成失败 %s/%s: %s", pid, date, exc, exc_info=True)
            stats["error"] += 1

    logger.info("QC 图生成完成: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="为 nnU-Net 分割结果生成 PET MIP 质控图"
    )
    p.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT,
                   help="预处理图像根目录（含 preprocessed/PET）")
    p.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT,
                   help="分割结果根目录（含 masks/*_lesion.nii.gz）")
    p.add_argument("--qc-root", type=Path, default=DEFAULT_QC_ROOT,
                   help="QC 图输出根目录")
    p.add_argument("--patient-id", default=None, help="仅处理指定患者 ID")
    p.add_argument("--study-date", default=None, help="仅处理指定检查日期")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有 QC 图")
    p.add_argument("--suv-max", type=float, default=SUV_DISPLAY_MAX,
                   help=f"PET 显示上限 SUV（默认 {SUV_DISPLAY_MAX}）")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    setup_logging(args.log_level)
    SUV_DISPLAY_MAX = args.suv_max
    run(
        interim_root=args.interim_root,
        processed_root=args.processed_root,
        qc_root=args.qc_root,
        overwrite=args.overwrite,
        patient_id=args.patient_id,
        study_date=args.study_date,
    )
