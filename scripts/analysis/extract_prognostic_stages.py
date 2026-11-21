#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
提取技术路线第五至第八阶段的量化指标。

对照 SZR/scripts/image-preprocess/processing/metabolic_metrics.py 与
suvmax41_segmentation.py，按 DLBCL 数据目录计算。指标定义以
data/DLBCL预后评估与多维分析技术路线升级版.rtf 的第五至第八阶段为准。

第五阶段：TMTV、TLG、最大病灶 MBV、BVG。AutoPET 掩码优先；掩码缺失或
过滤后为空时，才用肝脏自适应阈值加每个连通灶 41% SUVmax。不用 SUV≥2.5。
连通灶少于 10 个体素的不计入，并记录被部分容积效应影响的小灶。

第六阶段：Dmax、病灶数 NOL、脾脏离散度 Spread-Spleen，以及 TMTV/MBV 的
全套 IBSI 特征。只在原始图像上计算，特征族为 shape、firstorder、glcm、
glrlm、glszm、gldm、ngtdm。PET 固定 bin width 0.25 SUV，CT 固定 25 HU，
不用固定阶数。Dmax 同时给出除以身高的值。

第七阶段：只在 L3 椎体中点层面计算 SMA、SMI、VATI、SATI 和肌肉 SUV。
层面主导椎体不是 L3 时不计算体成分。肌肉 SUV 去掉病灶重叠和灶内高摄取。

第八阶段：最早一次检查为基线，第二次为中期。
ΔTLG、ΔTMTV = (基线 − 中期) / 基线 × 100%。纵向 SUV 比较使用
SUVmax / 肝 SUVmean。中期病灶消失时，只有已有跨检查配准掩码才回测原灶区。

日志：
    logs/extract_stages_5_8_<时间>.log
    results/tables/stages_5_8/run_log.csv
    results/tables/stages_5_8/stage5_metabolic.csv
    results/tables/stages_5_8/stage6_spatial_radiomics.csv
    results/tables/stages_5_8/stage7_body_composition.csv
    results/tables/stages_5_8/stage8_longitudinal.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
import logging
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.spatial.distance import pdist

_SCRIPTS = Path(__file__).resolve().parents[1]
for _p in (_SCRIPTS / "common", _SCRIPTS / "longitudinal"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline_utils import discover_subject_studies  # noqa: E402
from quant import parse_height_m  # noqa: E402

logger = logging.getLogger("stages_5_8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "stages_5_8_v1"
IBSI_PROTOCOL = "ibsi_original_v1"
IBSI_FILTERED_PROTOCOL = "ibsi_wavelet_log_v1"
IBSI_CLASSES = ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
# PyRadiomics 默认：一层 coif1 小波得到 8 个子带；LoG 的 sigma 为毫米。
# 形状特征只来自掩码，不在滤波图像上重复计算。
# 每个图像因此是 107（原始）+ 8×93（小波）+ 5×93（LoG）= 1316。
WAVELET_NAME = "coif1"
LOG_SIGMA_MM = (1.0, 2.0, 3.0, 4.0, 5.0)
IBSI_BASE_FIELDS = [
    "患者ID", "姓名", "检查日期", "时间点", "状态", "说明",
    "掩码来源", "特征数", "单图像特征数", "图像类型", "特征类", "小波", "LoG尺度_mm",
    "PET分箱宽度", "CT分箱宽度", "协议",
]
MIN_VOXELS = 10
PVE_VOLUME_CM3 = 4.2
SUV_FRACTION = 0.41
LIVER_MEAN_FACTOR = 1.5
CONN26 = np.ones((3, 3, 3), dtype=np.int8)

# organs.nii.gz（autoPET 对齐后的标签）
ORGAN_SPLEEN = 1
ORGAN_KIDNEY = 2
ORGAN_LIVER = 3
ORGAN_BLADDER = 4
ORGAN_BRAIN = 6
ORGAN_HEART = 7
ORGAN_PHYSIO = (ORGAN_KIDNEY, ORGAN_BLADDER, ORGAN_BRAIN, ORGAN_HEART)

# TotalSegmentator v2 total 任务
TS_LIVER = 5
TS_SPLEEN = 1
TS_L3 = 29
TS_L2 = 30
TS_L4 = 28
TS_LUMBAR = (26, 27, 28, 29, 30, 31)  # S1, L5, L4, L3, L2, L1
TS_PSOAS = (88, 89)
TS_BONE = frozenset(range(25, 51)) | frozenset(range(69, 79)) | frozenset(range(91, 118))
TS_VISCERA = frozenset({
    1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    51, 52, 53, 62, 63, 64, 79, 90,
})
TS_PHYSIO = frozenset({2, 3, 21, 51, 90})  # 肾、膀胱、心、脑

HU_MUSCLE = (-29.0, 150.0)
HU_FAT = (-190.0, -30.0)
BODY_HU = -500.0

STAGE5_FIELDS = [
    "患者ID", "姓名", "检查日期", "时间点", "状态", "说明",
    "掩码来源", "过滤前体素数", "过滤后体素数", "排除小于10体素灶数", "小于4.2ml灶数",
    "TMTV_cm3", "SUVmax", "SUVmean", "TLG", "MBV_cm3", "巨块SUVmean", "BVG",
    "肝SUVmean", "肝SUV_Q3", "肝归一化SUVmax", "肝SUVmean高于3",
    "备用阈值", "备用阈值依据", "协议",
]
STAGE6_FIELDS = [
    "患者ID", "姓名", "检查日期", "时间点", "状态", "说明",
    "NOL", "Dmax_cm", "身高_m", "Dmax除以身高", "SpreadSpleen_cm", "脾脏状态",
    "TMTV_球形度", "TMTV_PET_GLCM相关", "TMTV_PET_GLSZM灰度方差",
    "TMTV_CT_GLCM相关", "TMTV_CT_GLSZM灰度方差",
    "MBV_球形度", "MBV_PET_GLCM相关", "MBV_PET_GLSZM灰度方差",
    "MBV_CT_GLCM相关", "MBV_CT_GLSZM灰度方差",
    "PET分箱宽度", "CT分箱宽度", "协议",
]
STAGE7_FIELDS = [
    "患者ID", "姓名", "检查日期", "时间点", "状态", "说明",
    "L3层号", "L3世界坐标_mm", "层面主导椎体", "层面L3体素数", "层面L2体素数", "层面L4体素数",
    "SMA_cm2", "SMI", "腰大肌面积_cm2", "VAT面积_cm2", "SAT面积_cm2", "VATI", "SATI",
    "骨骼肌SUVmax", "骨骼肌SUVmean", "腰大肌SUVmax", "腰大肌SUVmean", "剔除高摄取体素数",
    "身高_m", "体重_kg", "BMI", "性别", "肌少症切点", "肌少症_Martin", "协议",
]
STAGE8_FIELDS = [
    "患者ID", "姓名", "状态", "说明",
    "基线日期", "中期日期", "时间点定义",
    "基线TMTV_cm3", "中期TMTV_cm3", "delta_TMTV_pct",
    "基线TLG", "中期TLG", "delta_TLG_pct",
    "基线肝归一化SUVmax", "中期肝归一化SUVmax", "delta_肝归一化SUVmax_pct",
    "中期病灶消失", "CMR映射状态", "原灶区中期SUVmax", "原灶区中期SUVmean", "原灶区肝归一化SUVmax",
    "协议",
]
LOG_FIELDS = ["阶段", "患者ID", "检查日期", "状态", "说明"]


def _blank(fields: list[str]) -> dict[str, object]:
    return {name: "" for name in fields}


def _num(value: float, digits: int = 4) -> float | str:
    if value is None or not np.isfinite(value):
        return ""
    return round(float(value), digits)


def _voxel_ml(affine: np.ndarray) -> float:
    return abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0


def _same_grid(left: nib.Nifti1Image, right: nib.Nifti1Image) -> bool:
    return left.shape == right.shape and np.allclose(left.affine, right.affine, atol=1e-3)


def _load(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def _finite(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=np.float32)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def _labels_on(reference: nib.Nifti1Image, path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    src = _load(path)
    if _same_grid(src, reference):
        return np.rint(np.asanyarray(src.dataobj)).astype(np.int16)
    warped = resample_from_to(src, (reference.shape, reference.affine), order=0)
    return np.rint(np.asanyarray(warped.dataobj)).astype(np.int16)


def _first_match(directory: Path, predicate) -> Path | None:
    if not directory.is_dir():
        return None
    files = [path for path in sorted(directory.glob("*.nii.gz")) if predicate(path.name)]
    return files[0] if files else None


def _is_brain(name: str) -> bool:
    upper = name.upper()
    return "BRAIN" in upper or "HEAD" in upper


def _wb_ct(study_dir: Path, preprocessed: bool) -> Path | None:
    base = study_dir / "preprocessed" / "CT" if preprocessed else study_dir / "CT"
    preferred = _first_match(base, lambda name: "WB_Standard" in name and not _is_brain(name))
    if preferred is not None:
        return preferred
    return _first_match(base, lambda name: not _is_brain(name))


def _wb_suv(study_dir: Path) -> Path | None:
    base = study_dir / "preprocessed" / "PET"
    preferred = _first_match(
        base, lambda name: "SUVbw" in name and "WB" in name.upper() and not _is_brain(name)
    )
    if preferred is not None:
        return preferred
    return _first_match(base, lambda name: "SUVbw" in name and not _is_brain(name))


def _lesion_path(processed: Path, patient_id: str, study_date: str) -> tuple[Path | None, str]:
    mask_dir = processed / patient_id / study_date / "masks"
    edited = mask_dir / f"{patient_id}_{study_date}_lesion_edited.nii.gz"
    auto = mask_dir / f"{patient_id}_{study_date}_lesion.nii.gz"
    if edited.is_file():
        return edited, "edited"
    if auto.is_file():
        return auto, "autopet"
    return None, ""


def _components(mask: np.ndarray, min_voxels: int) -> tuple[np.ndarray, int, int]:
    """去掉小于 min_voxels 的 26 连通灶。返回掩码、保留灶数、剔除灶数。"""
    labeled, n = ndimage.label(mask, structure=CONN26)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool), 0, 0
    counts = np.bincount(labeled.ravel())
    keep_ids = [idx for idx in range(1, n + 1) if counts[idx] >= min_voxels]
    drop = n - len(keep_ids)
    if not keep_ids:
        return np.zeros(mask.shape, dtype=bool), 0, drop
    kept = np.isin(labeled, keep_ids)
    relabeled, n_kept = ndimage.label(kept, structure=CONN26)
    return relabeled > 0, int(n_kept), int(drop)


def _centroids(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask, structure=CONN26)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    centers = ndimage.center_of_mass(mask.astype(np.float32), labeled, index=np.arange(1, n + 1))
    return nib.affines.apply_affine(affine, np.asarray(centers, dtype=np.float64))


def _bulk_mask(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask, structure=CONN26)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    counts = np.bincount(labeled.ravel())
    bulk_id = int(np.argmax(counts[1:]) + 1)
    return labeled == bulk_id


def _mask_stats(suv: np.ndarray, mask: np.ndarray, voxel_ml: float) -> dict[str, float]:
    values = suv[mask]
    values = values[np.isfinite(values)]
    n_vox = int(values.size)
    volume = n_vox * voxel_ml
    if n_vox == 0:
        return {
            "n_voxels": 0,
            "volume_cm3": 0.0,
            "suv_max": float("nan"),
            "suv_mean": float("nan"),
            "tlg": 0.0,
        }
    mean = float(np.mean(values))
    return {
        "n_voxels": n_vox,
        "volume_cm3": float(volume),
        "suv_max": float(np.max(values)),
        "suv_mean": mean,
        "tlg": float(np.sum(values)) * voxel_ml,
    }


def _liver_stats(suv: np.ndarray, liver: np.ndarray | None, lesion: np.ndarray) -> tuple[float, float]:
    if liver is None or not np.any(liver):
        return float("nan"), float("nan")
    ref = liver & ~lesion
    if not np.any(ref):
        ref = liver
    values = suv[ref]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.quantile(values, 0.75))


def _fallback_mask(
    suv: np.ndarray,
    liver_mean: float,
    liver_q3: float,
    physio: np.ndarray | None,
) -> tuple[np.ndarray, float, str]:
    """肝脏阈值找出候选灶，再对每个灶做 41% SUVmax。没有肝脏参考时不分割。"""
    candidates: list[tuple[float, str]] = []
    if np.isfinite(liver_mean) and liver_mean > 0:
        candidates.append((LIVER_MEAN_FACTOR * liver_mean, "1.5倍肝SUVmean"))
    if np.isfinite(liver_q3) and liver_q3 > 0:
        candidates.append((liver_q3, "肝SUV_Q3"))
    if not candidates:
        return np.zeros(suv.shape, dtype=bool), float("nan"), "无肝脏参考，未使用绝对阈值"
    threshold = max(value for value, _name in candidates)
    basis = "+".join(name for value, name in candidates if np.isclose(value, threshold))
    if len(candidates) == 2:
        basis = f"取较大者({basis})"
    candidate = suv >= threshold
    if physio is not None:
        candidate &= ~physio
    labeled, n = ndimage.label(candidate, structure=CONN26)
    refined = np.zeros(suv.shape, dtype=bool)
    for idx in range(1, n + 1):
        region = labeled == idx
        local_max = float(suv[region].max())
        if local_max <= 0:
            continue
        refined |= region & (suv >= SUV_FRACTION * local_max)
    return refined, float(threshold), basis


def _pve_count(mask: np.ndarray, voxel_ml: float) -> int:
    labeled, n = ndimage.label(mask, structure=CONN26)
    if n == 0:
        return 0
    counts = np.bincount(labeled.ravel())
    return int(sum(1 for idx in range(1, n + 1) if counts[idx] * voxel_ml < PVE_VOLUME_CM3))


def _organ_arrays(pet_img: nib.Nifti1Image, organs_path: Path | None, totalseg_path: Path | None):
    organs = _labels_on(pet_img, organs_path)
    total = _labels_on(pet_img, totalseg_path)
    liver = spleen = physio = None
    if organs is not None and np.any(organs == ORGAN_LIVER):
        liver = organs == ORGAN_LIVER
        spleen = organs == ORGAN_SPLEEN
        physio = np.isin(organs, ORGAN_PHYSIO)
    elif total is not None:
        liver = total == TS_LIVER
        spleen = total == TS_SPLEEN
        physio = np.isin(total, list(TS_PHYSIO))
    return liver, spleen, physio


def _radiomics(
    image_path: Path,
    mask: np.ndarray,
    reference: nib.Nifti1Image,
    bin_width: float,
    *,
    filtered: bool = False,
) -> dict[str, float]:
    """IBSI 特征族。分箱宽度固定，不重采样，不归一化。filtered 时加上小波和 LoG。"""
    if int(mask.sum()) < MIN_VOXELS:
        return {}
    try:
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError:
        return {}
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    import tempfile

    extractor = RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.disableAllImageTypes()
    for feature_class in IBSI_CLASSES:
        extractor.enableFeatureClassByName(feature_class)
    if filtered:
        extractor.enableImageTypes(
            Original={},
            Wavelet={"wavelet": WAVELET_NAME, "start_level": 0, "level": 1},
            LoG={"sigma": list(LOG_SIGMA_MM)},
        )
    else:
        extractor.enableImageTypeByName("Original")
    extractor.settings["binWidth"] = float(bin_width)
    extractor.settings["normalize"] = False
    extractor.settings["resampledPixelSpacing"] = None
    extractor.settings["additionalInfo"] = False
    extractor.settings["correctMask"] = True
    if filtered:
        extractor.settings["padDistance"] = 10
    try:
        with tempfile.TemporaryDirectory(prefix="stage6_") as tmp:
            image_for_extract = image_path
            mask_for_extract = mask
            affine = reference.affine
            if filtered:
                image_for_extract, mask_for_extract, affine = _crop_for_filters(image_path, mask, reference, Path(tmp), pad=10)
            mask_path = Path(tmp) / "mask.nii.gz"
            nib.save(nib.Nifti1Image(mask_for_extract.astype(np.uint8), affine), str(mask_path))
            result = extractor.execute(str(image_for_extract), str(mask_path), label=1)
    except Exception as exc:  # noqa: BLE001 - 单组特征失败时留空并继续
        logger.warning("组学提取失败 %s: %s", image_path, exc)
        return {}
    features: dict[str, float] = {}
    for key, value in result.items():
        name = str(key)
        if not (name.startswith("original_") or name.startswith("wavelet-") or name.startswith("log-sigma-")):
            continue
        try:
            features[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return features


def _crop_for_filters(image_path: Path, mask: np.ndarray, reference: nib.Nifti1Image, tmp: Path, pad: int) -> tuple[Path, np.ndarray, np.ndarray]:
    """把图像裁到病灶外接盒加 pad，避免小波和 LoG 在全身体积上滤波。"""
    image = nib.load(str(image_path))
    if image.shape != mask.shape:
        raise ValueError(f"图像与掩码尺寸不一致 {image.shape} {mask.shape}")
    coords = np.argwhere(mask > 0)
    low = np.maximum(coords.min(axis=0) - pad, 0)
    high = np.minimum(coords.max(axis=0) + pad + 1, np.array(mask.shape))
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(low, high))
    cropped = np.asanyarray(image.dataobj[slices])
    affine = image.affine.copy()
    origin = image.affine @ np.array([low[0], low[1], low[2], 1.0], dtype=float)
    affine[:3, 3] = origin[:3]
    out = tmp / "image.nii.gz"
    nib.save(nib.Nifti1Image(cropped, affine), str(out))
    return out, np.asanyarray(mask[slices]), affine


def _slice_axis(affine: np.ndarray) -> int:
    return int(np.argmax(np.abs(affine[2, :3])))


def _take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def _largest_and_fill(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labeled.ravel())
    keep = labeled == int(np.argmax(counts[1:]) + 1)
    return ndimage.binary_fill_holes(keep)


def _body_composition(ct_img: nib.Nifti1Image, seg: np.ndarray) -> dict[str, float | str]:
    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    axis = _slice_axis(ct_img.affine)
    l3 = seg == TS_L3
    counts = [int(np.take(l3, index, axis=axis).sum()) for index in range(l3.shape[axis])]
    present = [index for index, count in enumerate(counts) if count > 0]
    out: dict[str, float | str] = {
        "status": "warning",
        "message": "",
        "slice": "",
        "world_mm": "",
        "dominant": "",
        "n_l3": "",
        "n_l2": "",
        "n_l4": "",
    }
    if not present:
        out["message"] = "TotalSegmentator 中没有 L3"
        return out
    weights = np.asarray([counts[index] for index in present], dtype=np.float64)
    centroid = float(np.average(np.asarray(present, dtype=np.float64), weights=weights))

    def dominant(index: int) -> tuple[int, int, int, int]:
        slab = _take_slice(seg, axis, index)
        lumbar = slab[np.isin(slab, TS_LUMBAR)]
        if lumbar.size == 0:
            return 0, 0, 0, 0
        values, freq = np.unique(lumbar, return_counts=True)
        winner = int(values[int(np.argmax(freq))])
        return winner, int(np.sum(slab == TS_L3)), int(np.sum(slab == TS_L2)), int(np.sum(slab == TS_L4))

    candidates = []
    for index in present:
        winner, n_l3, n_l2, n_l4 = dominant(index)
        if winner == TS_L3 and n_l3 > n_l2 and n_l3 > n_l4:
            candidates.append((abs(index - centroid), -n_l3, index, n_l3, n_l2, n_l4))
    if not candidates:
        out["message"] = "L3 中点附近的主导椎体不是 L3，未计算 SMI，避免混用 L2/L4"
        return out
    _distance, _neg, index, n_l3, n_l2, n_l4 = min(candidates)
    ijk = np.zeros(3, dtype=np.float64)
    ijk[axis] = index
    world_z = float(nib.affines.apply_affine(ct_img.affine, ijk)[2])
    ct_slice = _take_slice(ct, axis, index)
    seg_slice = _take_slice(seg, axis, index)
    in_plane = [dim for dim in range(3) if dim != axis]
    zooms = np.sqrt(np.sum(ct_img.affine[:3, :3] ** 2, axis=0))
    pixel_cm2 = float(zooms[in_plane[0]] * zooms[in_plane[1]] / 100.0)

    body = _largest_and_fill(ct_slice > BODY_HU)
    excluded = np.isin(seg_slice, list(TS_BONE | TS_VISCERA))
    muscle = body & (ct_slice >= HU_MUSCLE[0]) & (ct_slice <= HU_MUSCLE[1]) & ~excluded
    psoas = body & np.isin(seg_slice, TS_PSOAS) & (ct_slice >= HU_MUSCLE[0]) & (ct_slice <= HU_MUSCLE[1])
    fat = body & (ct_slice >= HU_FAT[0]) & (ct_slice <= HU_FAT[1])
    wall = ndimage.binary_closing(muscle | np.isin(seg_slice, list(TS_BONE)), structure=np.ones((5, 5)))
    cavity = ndimage.binary_fill_holes(wall)
    leaked = int(body.sum()) > 0 and int(cavity.sum()) > 0.95 * int(body.sum())
    sma = float(muscle.sum()) * pixel_cm2
    psoas_area = float(psoas.sum()) * pixel_cm2
    if leaked:
        vat = sat = float("nan")
        fat_note = "腹壁未闭合，VATI/SATI 留空"
    else:
        vat = float(np.sum(fat & cavity & ~muscle)) * pixel_cm2
        sat = float(np.sum(fat & ~cavity)) * pixel_cm2
        fat_note = ""
    out.update({
        "status": "ok" if not leaked else "warning",
        "message": fat_note,
        "slice": index,
        "world_mm": round(world_z, 2),
        "dominant": "L3",
        "n_l3": n_l3,
        "n_l2": n_l2,
        "n_l4": n_l4,
        "sma_cm2": sma,
        "psoas_cm2": psoas_area,
        "vat_cm2": vat,
        "sat_cm2": sat,
        "pixel_cm2": pixel_cm2,
        "axis": axis,
        "leaked": leaked,
    })
    return out


def _index_on_pet(ct_img: nib.Nifti1Image, pet_affine: np.ndarray, pet_shape, axis: int, index: int, slice_mask: np.ndarray) -> np.ndarray:
    """把 CT 单个横断层面上的掩码映射到 PET 体素。"""
    coords = np.argwhere(slice_mask)
    if coords.size == 0:
        return np.zeros(0, dtype=np.int64)
    # np.take 去掉了层面轴，argwhere 的列按剩余轴的顺序排列。
    remaining = [dim for dim in range(3) if dim != axis]
    ijk = np.zeros((coords.shape[0], 3), dtype=np.float64)
    for column, dim in enumerate(remaining):
        ijk[:, dim] = coords[:, column]
    ijk[:, axis] = index
    world = nib.affines.apply_affine(ct_img.affine, ijk)
    pet_ijk = nib.affines.apply_affine(np.linalg.inv(pet_affine), world)
    pet_ijk = np.rint(pet_ijk).astype(np.int64)
    inside = np.all((pet_ijk >= 0) & (pet_ijk < np.asarray(pet_shape)), axis=1)
    pet_ijk = pet_ijk[inside]
    if pet_ijk.size == 0:
        return np.zeros(0, dtype=np.int64)
    flat = np.ravel_multi_index(pet_ijk.T, pet_shape)
    return np.unique(flat)


def _suv_in_flat(suv: np.ndarray, lesion: np.ndarray, flat: np.ndarray) -> tuple[float, float, int]:
    if flat.size == 0:
        return float("nan"), float("nan"), 0
    values = suv.ravel()[flat]
    tumor = lesion.ravel()[flat]
    keep = np.isfinite(values) & ~tumor
    values = values[keep]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    q1, q3 = np.quantile(values, [0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    focal = values > max(float(fence), 2.5)
    # 2.5 只用来识别肌肉内部的灶性高摄取，不作为肿瘤分割阈值。
    cleaned = values[~focal]
    if cleaned.size == 0:
        return float("nan"), float("nan"), int(focal.sum())
    return float(np.max(cleaned)), float(np.mean(cleaned)), int(focal.sum())


def _read_excel(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = book.active
    rows = sheet.iter_rows(values_only=True)
    header = [str(cell).strip() if cell is not None else "" for cell in next(rows)]
    id_col = header.index("ID")
    name_col = header.index("姓名")
    sex_col = header.index("性别")
    names: dict[str, str] = {}
    sexes: dict[str, str] = {}
    for row in rows:
        raw_id = "" if row[id_col] is None else str(row[id_col]).strip()
        if not raw_id:
            continue
        key = raw_id.lstrip("0") or "0"
        names[key] = "" if row[name_col] is None else str(row[name_col]).strip()
        label = "" if row[sex_col] is None else str(row[sex_col]).strip()
        sexes[key] = {"男": "M", "女": "F"}.get(label, label[:1].upper())
    book.close()
    return names, sexes


def _dicom_demographics(data_root: Path) -> dict[tuple[str, str], dict[str, float | str | None]]:
    import pydicom

    roots = [path for path in sorted(data_root.glob("raw/DICOM*")) if path.is_dir()]
    roots.extend(path for path in sorted(data_root.glob("01-*")) if path.is_dir())
    found: dict[tuple[str, str], dict[str, float | str | None]] = {}
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            names = [name for name in filenames if not name.startswith(".") and name not in {"DICOMDIR", "DIR_INFO.TXT"}]
            if not names:
                continue
            try:
                dataset = pydicom.dcmread(str(Path(dirpath) / names[0]), stop_before_pixels=True, force=True)
            except Exception:
                continue
            patient_id = str(getattr(dataset, "PatientID", "") or "").strip()
            study_date = str(getattr(dataset, "StudyDate", "") or "").strip()
            if not patient_id or not study_date:
                continue
            sex = str(getattr(dataset, "PatientSex", "") or "").strip().upper()
            height = parse_height_m(getattr(dataset, "PatientSize", None))
            weight = getattr(dataset, "PatientWeight", None)
            try:
                weight_kg = float(weight) if weight is not None and float(weight) > 0 else None
            except (TypeError, ValueError):
                weight_kg = None
            record = {
                "sex": sex if sex in {"M", "F"} else "",
                "height_m": height,
                "weight_kg": weight_kg,
            }
            key = (patient_id.lstrip("0") or "0", study_date)
            current = found.get(key)
            if current is None or (current.get("height_m") is None and height is not None):
                found[key] = record
    return found


def _lookup(table: dict, patient_id: str, study_date: str | None = None):
    key_id = patient_id.lstrip("0") or "0"
    if study_date is None:
        return table.get(key_id)
    return table.get((key_id, study_date))


def _percent_drop(baseline: float, follow: float) -> float:
    if not np.isfinite(baseline) or not np.isfinite(follow) or baseline == 0:
        return float("nan")
    return (baseline - follow) / baseline * 100.0


def process_study(payload: dict) -> dict:
    """计算一例检查的第五、六、七阶段。异常写入该例的状态，不中断整批。"""
    if payload.get("skip_radiomics"):
        os.environ["DLBCL_SKIP_RADIOMICS"] = "1"
    patient_id = payload["patient_id"]
    study_date = payload["study_date"]
    study_dir = Path(payload["study_dir"])
    processed = Path(payload["processed_root"])
    name = payload["name"]
    demo = payload["demo"] or {}
    notes: list[str] = []
    stage5 = _blank(STAGE5_FIELDS)
    stage6 = _blank(STAGE6_FIELDS)
    stage7 = _blank(STAGE7_FIELDS)
    for row in (stage5, stage6, stage7):
        row["患者ID"] = patient_id
        row["姓名"] = name
        row["检查日期"] = study_date
        row["协议"] = PROTOCOL
    try:
        _fill_study(stage5, stage6, stage7, study_dir, processed, patient_id, study_date, demo, notes)
    except Exception as exc:  # noqa: BLE001 - 单例失败继续
        message = f"{type(exc).__name__}: {exc}"
        logger.error("%s %s 失败: %s\n%s", patient_id, study_date, message, traceback.format_exc())
        for row in (stage5, stage6, stage7):
            if row["状态"] == "":
                row["状态"] = "error"
                row["说明"] = message
    for row in (stage5, stage6, stage7):
        if notes and row["说明"] == "":
            row["说明"] = "；".join(notes)
    return {"stage5": stage5, "stage6": stage6, "stage7": stage7}


def _fill_study(stage5, stage6, stage7, study_dir, processed, patient_id, study_date, demo, notes: list[str]) -> None:
    suv_path = _wb_suv(study_dir)
    ct_pre = _wb_ct(study_dir, preprocessed=True)
    ct_orig = _wb_ct(study_dir, preprocessed=False)
    mask_path, mask_source = _lesion_path(processed, patient_id, study_date)
    organs_path = processed / patient_id / study_date / "organs" / "organs.nii.gz"
    total_path = processed / patient_id / study_date / "organs" / "_totalseg_staging" / "totalseg_wholebody.nii.gz"
    if suv_path is None:
        stage5["状态"] = stage6["状态"] = "warning"
        stage5["说明"] = stage6["说明"] = "缺少体部 SUVbw"
        notes.append("缺少体部 SUVbw")
        _fill_body(stage7, ct_orig, total_path, None, None, None, demo, notes)
        return

    pet_img = _load(suv_path)
    suv = _finite(np.asanyarray(pet_img.dataobj))
    voxel_ml = _voxel_ml(np.asarray(pet_img.affine, dtype=np.float64))
    lesion = np.zeros(pet_img.shape, dtype=bool)
    raw_voxels = 0
    if mask_path is not None:
        mask_img = _load(mask_path)
        if not _same_grid(mask_img, pet_img):
            mask_img = resample_from_to(mask_img, (pet_img.shape, pet_img.affine), order=0)
            notes.append("病灶掩码已重采样到 SUV 网格")
        lesion = np.asanyarray(mask_img.dataobj) > 0
        raw_voxels = int(lesion.sum())
    liver, spleen, physio = _organ_arrays(
        pet_img,
        organs_path if organs_path.is_file() else None,
        total_path if total_path.is_file() else None,
    )
    liver_mean, liver_q3 = _liver_stats(suv, liver, lesion)
    threshold = float("nan")
    threshold_basis = ""
    if raw_voxels == 0:
        lesion, threshold, threshold_basis = _fallback_mask(suv, liver_mean, liver_q3, physio)
        mask_source = "fallback_41pct_liver" if int(lesion.sum()) else "fallback_empty"
        if mask_source == "fallback_empty":
            notes.append(threshold_basis or "降级分割未得到病灶")
        else:
            notes.append(f"AutoPET 不可用，已降级：{threshold_basis}，阈值 {threshold:.4f}")
    filtered, n_kept, n_dropped = _components(lesion, MIN_VOXELS)
    if n_dropped:
        notes.append(f"剔除小于 {MIN_VOXELS} 体素的连通灶 {n_dropped} 个")
    stats = _mask_stats(suv, filtered, voxel_ml)
    bulk = _bulk_mask(filtered)
    bulk_stats = _mask_stats(suv, bulk, voxel_ml)
    ratio = float("nan")
    if np.isfinite(stats["suv_max"]) and np.isfinite(liver_mean) and liver_mean > 0:
        ratio = stats["suv_max"] / liver_mean
    if np.isfinite(liver_mean) and liver_mean > 3.0:
        notes.append("肝 SUVmean>3，未使用 SUV≥2.5")
    stage5.update({
        "状态": "ok" if mask_source != "fallback_empty" else "warning",
        "掩码来源": mask_source,
        "过滤前体素数": raw_voxels if mask_source in {"edited", "autopet"} else int(lesion.sum()),
        "过滤后体素数": stats["n_voxels"],
        "排除小于10体素灶数": n_dropped,
        "小于4.2ml灶数": _pve_count(filtered, voxel_ml),
        "TMTV_cm3": _num(stats["volume_cm3"], 3),
        "SUVmax": _num(stats["suv_max"]),
        "SUVmean": _num(stats["suv_mean"]),
        "TLG": _num(stats["tlg"], 3),
        "MBV_cm3": _num(bulk_stats["volume_cm3"], 3),
        "巨块SUVmean": _num(bulk_stats["suv_mean"]),
        "BVG": _num(bulk_stats["tlg"], 3),
        "肝SUVmean": _num(liver_mean),
        "肝SUV_Q3": _num(liver_q3),
        "肝归一化SUVmax": _num(ratio),
        "肝SUVmean高于3": "是" if np.isfinite(liver_mean) and liver_mean > 3.0 else "否",
        "备用阈值": _num(threshold),
        "备用阈值依据": threshold_basis,
    })
    if stats["n_voxels"] == 0:
        stage5["状态"] = "warning"
        notes.append("过滤后 TMTV 为 0")

    centers = _centroids(filtered, np.asarray(pet_img.affine, dtype=np.float64))
    dmax_cm = float("nan")
    if len(centers) >= 2:
        dmax_cm = float(pdist(centers).max()) / 10.0
    height = demo.get("height_m")
    height_f = float(height) if height not in (None, "") else float("nan")
    spread = float("nan")
    spleen_state = "无脾脏掩码"
    if spleen is not None and np.any(spleen) and len(centers):
        spleen_center = nib.affines.apply_affine(
            pet_img.affine,
            np.asarray(ndimage.center_of_mass(spleen.astype(np.float32)), dtype=np.float64),
        )
        distance = np.linalg.norm(centers - spleen_center, axis=1)
        spread = float(np.std(distance, ddof=0)) / 10.0
        spleen_state = "已计算"
    elif len(centers) == 0:
        spleen_state = "无病灶"
    stage6.update({
        "状态": "ok",
        "NOL": int(len(centers)),
        "Dmax_cm": _num(dmax_cm, 3),
        "身高_m": _num(height_f, 3),
        "Dmax除以身高": _num(dmax_cm / height_f, 3) if np.isfinite(dmax_cm) and np.isfinite(height_f) and height_f > 0 else "",
        "SpreadSpleen_cm": _num(spread, 3),
        "脾脏状态": spleen_state,
        "PET分箱宽度": 0.25,
        "CT分箱宽度": 25,
    })
    if not np.isfinite(height_f):
        notes.append("DICOM 无身高，Dmax/身高留空")
    if len(centers) < 2:
        notes.append("不足两个病灶，Dmax 留空")
    if os.environ.get("DLBCL_SKIP_RADIOMICS") == "1":
        notes.append("本次运行跳过组学")
    else:
        try:
            _fill_radiomics(stage6, suv_path, ct_pre, pet_img, filtered, bulk, notes)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"组学失败: {exc}")
            stage6["状态"] = "warning"
    _fill_body(stage7, ct_orig, total_path if total_path.is_file() else None, pet_img, suv, filtered, demo, notes, stage5)


def _is_feature_column(key: str) -> bool:
    return "_original_" in key or "_wavelet-" in key or "_log-sigma-" in key


def _stamp_features(row: dict[str, object], prefix: str, features: dict[str, float]) -> None:
    for key, value in features.items():
        row[f"{prefix}_{key}"] = _num(value, 6)


def _fill_radiomics(stage6, suv_path, ct_pre, pet_img, filtered, bulk, notes: list[str], *, use_filters: bool = False) -> None:
    if int(filtered.sum()) < MIN_VOXELS:
        notes.append("无足够病灶体素，未提取组学")
        return
    pet_all = _radiomics(suv_path, filtered, pet_img, 0.25, filtered=use_filters)
    pet_bulk = _radiomics(suv_path, bulk, pet_img, 0.25, filtered=use_filters)
    ct_all: dict[str, float] = {}
    ct_bulk: dict[str, float] = {}
    if ct_pre is not None and ct_pre.is_file():
        ct_img = _load(ct_pre)
        if not _same_grid(ct_img, pet_img):
            notes.append("预处理 CT 与 SUV 网格不一致，CT 组学留空")
        else:
            ct_all = _radiomics(ct_pre, filtered, ct_img, 25.0, filtered=use_filters)
            ct_bulk = _radiomics(ct_pre, bulk, ct_img, 25.0, filtered=use_filters)
    else:
        notes.append("缺少预处理体部 CT，CT 组学留空")
    _stamp_features(stage6, "TMTV_PET", pet_all)
    _stamp_features(stage6, "MBV_PET", pet_bulk)
    _stamp_features(stage6, "TMTV_CT", ct_all)
    _stamp_features(stage6, "MBV_CT", ct_bulk)
    stage6["TMTV_球形度"] = stage6.get("TMTV_PET_original_shape_Sphericity", "")
    stage6["TMTV_PET_GLCM相关"] = stage6.get("TMTV_PET_original_glcm_Correlation", "")
    stage6["TMTV_PET_GLSZM灰度方差"] = stage6.get("TMTV_PET_original_glszm_GrayLevelVariance", "")
    stage6["TMTV_CT_GLCM相关"] = stage6.get("TMTV_CT_original_glcm_Correlation", "")
    stage6["TMTV_CT_GLSZM灰度方差"] = stage6.get("TMTV_CT_original_glszm_GrayLevelVariance", "")
    stage6["MBV_球形度"] = stage6.get("MBV_PET_original_shape_Sphericity", "")
    stage6["MBV_PET_GLCM相关"] = stage6.get("MBV_PET_original_glcm_Correlation", "")
    stage6["MBV_PET_GLSZM灰度方差"] = stage6.get("MBV_PET_original_glszm_GrayLevelVariance", "")
    stage6["MBV_CT_GLCM相关"] = stage6.get("MBV_CT_original_glcm_Correlation", "")
    stage6["MBV_CT_GLSZM灰度方差"] = stage6.get("MBV_CT_original_glszm_GrayLevelVariance", "")
    stage6["特征数"] = sum(1 for key, value in stage6.items() if _is_feature_column(key) and value != "")
    stage6["单图像特征数"] = sum(
        1 for key, value in stage6.items() if key.startswith("TMTV_PET_") and _is_feature_column(key) and value != ""
    )
    stage6["图像类型"] = "Original,Wavelet,LoG" if use_filters else "Original"
    stage6["特征类"] = ",".join(IBSI_CLASSES)
    if use_filters:
        stage6["小波"] = f"{WAVELET_NAME} level 1，8 个子带"
        stage6["LoG尺度_mm"] = ",".join(str(sigma) for sigma in LOG_SIGMA_MM)
        stage6["协议"] = IBSI_FILTERED_PROTOCOL


def _fill_body(stage7, ct_orig, total_path, pet_img, suv, lesion, demo, notes: list[str], stage5=None) -> None:
    stage7["状态"] = "warning"
    if ct_orig is None or total_path is None:
        stage7["说明"] = "缺少原始体部 CT 或 totalseg_wholebody，未计算 L3 体成分"
        return
    ct_img = _load(ct_orig)
    seg = _labels_on(ct_img, total_path)
    if seg is None:
        stage7["说明"] = "无法读取 TotalSegmentator 标签"
        return
    comp = _body_composition(ct_img, seg)
    height = demo.get("height_m")
    weight = demo.get("weight_kg")
    height_f = float(height) if height not in (None, "") else float("nan")
    weight_f = float(weight) if weight not in (None, "") else float("nan")
    bmi = weight_f / height_f ** 2 if np.isfinite(weight_f) and np.isfinite(height_f) and height_f > 0 else float("nan")
    sex = demo.get("sex") or ""
    sma = comp.get("sma_cm2", float("nan"))
    smi = float(sma) / height_f ** 2 if isinstance(sma, float) and np.isfinite(sma) and np.isfinite(height_f) and height_f > 0 else float("nan")
    vat = comp.get("vat_cm2", float("nan"))
    sat = comp.get("sat_cm2", float("nan"))
    vati = float(vat) / height_f ** 2 if isinstance(vat, float) and np.isfinite(vat) and np.isfinite(height_f) and height_f > 0 else float("nan")
    sati = float(sat) / height_f ** 2 if isinstance(sat, float) and np.isfinite(sat) and np.isfinite(height_f) and height_f > 0 else float("nan")
    cutoff = ""
    sarcopenia = ""
    if np.isfinite(smi) and sex == "M" and np.isfinite(bmi):
        cutoff = 43 if bmi < 25 else 53
        sarcopenia = "是" if smi < cutoff else "否"
    elif np.isfinite(smi) and sex == "F":
        cutoff = 41
        sarcopenia = "是" if smi < cutoff else "否"
    elif comp.get("status") == "ok":
        notes.append("缺少身高、体重或性别，SMI 或肌少症留空")
    muscle_max = muscle_mean = psoas_max = psoas_mean = float("nan")
    n_focal = ""
    if pet_img is not None and comp.get("status") in {"ok", "warning"} and comp.get("slice") != "":
        axis = int(comp["axis"])
        index = int(comp["slice"])
        ct = np.asarray(ct_img.dataobj, dtype=np.float32)
        seg_slice = _take_slice(seg, axis, index)
        ct_slice = _take_slice(ct, axis, index)
        body = _largest_and_fill(ct_slice > BODY_HU)
        excluded = np.isin(seg_slice, list(TS_BONE | TS_VISCERA))
        muscle = body & (ct_slice >= HU_MUSCLE[0]) & (ct_slice <= HU_MUSCLE[1]) & ~excluded
        psoas = body & np.isin(seg_slice, TS_PSOAS)
        muscle_flat = _index_on_pet(ct_img, pet_img.affine, pet_img.shape, axis, index, muscle)
        psoas_flat = _index_on_pet(ct_img, pet_img.affine, pet_img.shape, axis, index, psoas)
        lesion_mask = lesion if lesion is not None else np.zeros(pet_img.shape, dtype=bool)
        muscle_max, muscle_mean, n_muscle_focal = _suv_in_flat(suv, lesion_mask, muscle_flat)
        psoas_max, psoas_mean, _n_psoas = _suv_in_flat(suv, lesion_mask, psoas_flat)
        n_focal = n_muscle_focal
    if not np.isfinite(height_f):
        notes.append("DICOM 无身高，SMI/VATI/SATI 留空")
    message = str(comp.get("message") or "")
    stage7.update({
        "状态": comp.get("status", "warning"),
        "说明": message,
        "L3层号": comp.get("slice", ""),
        "L3世界坐标_mm": comp.get("world_mm", ""),
        "层面主导椎体": comp.get("dominant", ""),
        "层面L3体素数": comp.get("n_l3", ""),
        "层面L2体素数": comp.get("n_l2", ""),
        "层面L4体素数": comp.get("n_l4", ""),
        "SMA_cm2": _num(sma, 3) if isinstance(sma, float) else "",
        "SMI": _num(smi, 3),
        "腰大肌面积_cm2": _num(comp.get("psoas_cm2", float("nan")), 3) if comp.get("psoas_cm2") not in (None, "") else "",
        "VAT面积_cm2": _num(vat, 3) if isinstance(vat, float) else "",
        "SAT面积_cm2": _num(sat, 3) if isinstance(sat, float) else "",
        "VATI": _num(vati, 3),
        "SATI": _num(sati, 3),
        "骨骼肌SUVmax": _num(muscle_max),
        "骨骼肌SUVmean": _num(muscle_mean),
        "腰大肌SUVmax": _num(psoas_max),
        "腰大肌SUVmean": _num(psoas_mean),
        "剔除高摄取体素数": n_focal,
        "身高_m": _num(height_f, 3),
        "体重_kg": _num(weight_f, 2),
        "BMI": _num(bmi, 2),
        "性别": sex,
        "肌少症切点": cutoff,
        "肌少症_Martin": sarcopenia,
    })
    if stage5 is not None and stage7["说明"] and stage5["说明"] == "":
        pass


def _as_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def build_longitudinal(rows: list[dict], names: dict[str, str]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if not str(row["检查日期"]).isdigit():
            continue
        grouped.setdefault(row["患者ID"], []).append(row)
    output: list[dict] = []
    definition = "最早日期为基线，第二早日期为中期；UNKNOWN_DATE 不计入"
    for patient_id, studies in sorted(grouped.items()):
        studies = sorted(studies, key=lambda item: item["检查日期"])
        row = _blank(STAGE8_FIELDS)
        row["患者ID"] = patient_id
        row["姓名"] = names.get(patient_id.lstrip("0") or "0", studies[0].get("姓名", ""))
        row["时间点定义"] = definition
        row["协议"] = PROTOCOL
        baseline = studies[0]
        row["基线日期"] = baseline["检查日期"]
        row["基线TMTV_cm3"] = baseline["TMTV_cm3"]
        row["基线TLG"] = baseline["TLG"]
        row["基线肝归一化SUVmax"] = baseline["肝归一化SUVmax"]
        if len(studies) < 2:
            row["状态"] = "warning"
            row["说明"] = "只有一次可定日期的检查，没有中期"
            output.append(row)
            continue
        interim = studies[1]
        row["中期日期"] = interim["检查日期"]
        row["中期TMTV_cm3"] = interim["TMTV_cm3"]
        row["中期TLG"] = interim["TLG"]
        row["中期肝归一化SUVmax"] = interim["肝归一化SUVmax"]
        base_tmtv = _as_float(baseline["TMTV_cm3"])
        next_tmtv = _as_float(interim["TMTV_cm3"])
        row["delta_TMTV_pct"] = _num(_percent_drop(base_tmtv, next_tmtv), 2)
        row["delta_TLG_pct"] = _num(_percent_drop(_as_float(baseline["TLG"]), _as_float(interim["TLG"])), 2)
        row["delta_肝归一化SUVmax_pct"] = _num(
            _percent_drop(_as_float(baseline["肝归一化SUVmax"]), _as_float(interim["肝归一化SUVmax"])),
            2,
        )
        disappeared = np.isfinite(next_tmtv) and next_tmtv == 0 and np.isfinite(base_tmtv) and base_tmtv > 0
        row["中期病灶消失"] = "是" if disappeared else "否"
        mapped = _map_baseline_roi(patient_id, baseline["检查日期"], interim["检查日期"], row)
        if disappeared and not mapped:
            row["CMR映射状态"] = "无跨检查配准，未把基线坐标硬套到中期"
            row["状态"] = "warning"
            row["说明"] = "中期分割未见病灶，但没有配准掩码，未测量原灶区背景"
        elif mapped:
            row["状态"] = "ok"
        else:
            row["CMR映射状态"] = "中期仍有病灶，未做原灶区回测"
            row["状态"] = "ok" if baseline["状态"] == "ok" and interim["状态"] == "ok" else "warning"
            if row["状态"] == "warning":
                row["说明"] = "基线或中期代谢指标不是 ok"
        output.append(row)
    return output


def _map_baseline_roi(patient_id: str, baseline_date: str, interim_date: str, row: dict) -> bool:
    """仅当中期目录已有该基线日期的配准病灶掩码时，测量原灶区 SUV。"""
    processed = PROJECT_ROOT / "data" / "processed"
    interim_root = PROJECT_ROOT / "data" / "interim"
    warped = processed / patient_id / interim_date / "longitudinal" / "baseline_lesion_warped.nii.gz"
    manifest = processed / patient_id / interim_date / "longitudinal" / "mapping_qc.json"
    if not warped.is_file():
        return False
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("inputs", {}).get("baseline_date") not in {None, baseline_date}:
                return False
        except (OSError, ValueError):
            return False
    suv_path = _wb_suv(interim_root / patient_id / interim_date)
    if suv_path is None:
        row["CMR映射状态"] = "有配准掩码但中期缺少体部 SUV"
        return True
    pet_img = _load(suv_path)
    suv = _finite(np.asanyarray(pet_img.dataobj))
    mask_img = _load(warped)
    if not _same_grid(mask_img, pet_img):
        mask_img = resample_from_to(mask_img, (pet_img.shape, pet_img.affine), order=0)
    mask = np.asanyarray(mask_img.dataobj) > 0
    values = suv[mask]
    values = values[np.isfinite(values)]
    organs = _labels_on(pet_img, processed / patient_id / interim_date / "organs" / "organs.nii.gz")
    liver = None if organs is None else organs == ORGAN_LIVER
    liver_mean, _q3 = _liver_stats(suv, liver, np.zeros(suv.shape, dtype=bool))
    suv_max = float(np.max(values)) if values.size else float("nan")
    row["CMR映射状态"] = "已用跨检查配准掩码"
    row["原灶区中期SUVmax"] = _num(suv_max)
    row["原灶区中期SUVmean"] = _num(float(np.mean(values)) if values.size else float("nan"))
    if np.isfinite(suv_max) and np.isfinite(liver_mean) and liver_mean > 0:
        row["原灶区肝归一化SUVmax"] = _num(suv_max / liver_mean)
    return True


def _write_table(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras: list[str] = []
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                extras.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields) + sorted(extras), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _log_rows(studies: list[dict]) -> list[dict]:
    rows = []
    for item in studies:
        for key, stage in (("stage5", "5代谢负荷"), ("stage6", "6空间与组学"), ("stage7", "7体成分")):
            record = item[key]
            rows.append({
                "阶段": stage,
                "患者ID": record["患者ID"],
                "检查日期": record["检查日期"],
                "状态": record["状态"],
                "说明": record["说明"],
            })
    return rows


def _assign_roles(studies: list[dict]) -> None:
    grouped: dict[str, list[dict]] = {}
    for item in studies:
        grouped.setdefault(item["stage5"]["患者ID"], []).append(item)
    for group in grouped.values():
        dated = [item for item in group if str(item["stage5"]["检查日期"]).isdigit()]
        dated.sort(key=lambda item: item["stage5"]["检查日期"])
        roles = {}
        if dated:
            roles[dated[0]["stage5"]["检查日期"]] = "基线"
        if len(dated) > 1:
            roles[dated[1]["stage5"]["检查日期"]] = "中期"
        for item in group:
            date = item["stage5"]["检查日期"]
            role = roles.get(date, "未纳入时间点")
            for key in ("stage5", "stage6", "stage7"):
                item[key]["时间点"] = role


def _timepoint_lookup(output_dir: Path) -> dict[tuple[str, str], str]:
    path = output_dir / "stage5_metabolic.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row["患者ID"], row["检查日期"]): row.get("时间点", "")
            for row in csv.DictReader(handle)
        }


def ibsi_for_study(payload: dict) -> dict[str, object]:
    """只重算第六阶段的全套 IBSI 特征，不改代谢负荷和体成分。"""
    patient_id = payload["patient_id"]
    study_date = payload["study_date"]
    study_dir = Path(payload["study_dir"])
    processed = Path(payload["processed_root"])
    row = _blank(IBSI_BASE_FIELDS)
    row.update({
        "患者ID": patient_id,
        "姓名": payload["name"],
        "检查日期": study_date,
        "时间点": payload.get("timepoint", ""),
        "图像类型": "Original",
        "特征类": ",".join(IBSI_CLASSES),
        "PET分箱宽度": 0.25,
        "CT分箱宽度": 25,
        "协议": IBSI_FILTERED_PROTOCOL if payload.get("use_filters") else IBSI_PROTOCOL,
        "状态": "ok",
    })
    notes: list[str] = []
    try:
        suv_path = _wb_suv(study_dir)
        ct_pre = _wb_ct(study_dir, preprocessed=True)
        mask_path, mask_source = _lesion_path(processed, patient_id, study_date)
        organs_path = processed / patient_id / study_date / "organs" / "organs.nii.gz"
        total_path = processed / patient_id / study_date / "organs" / "_totalseg_staging" / "totalseg_wholebody.nii.gz"
        row["掩码来源"] = mask_source
        if suv_path is None:
            row["状态"] = "warning"
            row["说明"] = "缺少体部 SUVbw"
            return row
        pet_img = _load(suv_path)
        suv = _finite(np.asanyarray(pet_img.dataobj))
        lesion = np.zeros(pet_img.shape, dtype=bool)
        raw_voxels = 0
        if mask_path is not None:
            mask_img = _load(mask_path)
            if not _same_grid(mask_img, pet_img):
                mask_img = resample_from_to(mask_img, (pet_img.shape, pet_img.affine), order=0)
                notes.append("病灶掩码已重采样到 SUV 网格")
            lesion = np.asanyarray(mask_img.dataobj) > 0
            raw_voxels = int(lesion.sum())
        if raw_voxels == 0:
            liver, _spleen, physio = _organ_arrays(
                pet_img,
                organs_path if organs_path.is_file() else None,
                total_path if total_path.is_file() else None,
            )
            liver_mean, liver_q3 = _liver_stats(suv, liver, lesion)
            lesion, _threshold, basis = _fallback_mask(suv, liver_mean, liver_q3, physio)
            row["掩码来源"] = "fallback_41pct_liver" if int(lesion.sum()) else "fallback_empty"
            if row["掩码来源"] == "fallback_empty":
                notes.append(basis or "降级分割未得到病灶")
        filtered, _kept, dropped = _components(lesion, MIN_VOXELS)
        if drop := dropped:
            notes.append(f"剔除小于 {MIN_VOXELS} 体素的连通灶 {drop} 个")
        bulk = _bulk_mask(filtered)
        _fill_radiomics(row, suv_path, ct_pre, pet_img, filtered, bulk, notes, use_filters=bool(payload.get("use_filters")))
        if int(row.get("特征数") or 0) == 0:
            row["状态"] = "warning"
    except Exception as exc:  # noqa: BLE001 - 单例失败继续
        logger.error("%s %s IBSI 失败: %s", patient_id, study_date, exc)
        row["状态"] = "error"
        notes.append(f"{type(exc).__name__}: {exc}")
    if notes:
        row["说明"] = "；".join(notes)
    return row


def run_ibsi(args: argparse.Namespace, names: dict[str, str], *, filtered: bool = False) -> int:
    protocol = IBSI_FILTERED_PROTOCOL if filtered else IBSI_PROTOCOL
    image_types = "Original、Wavelet(coif1 八个子带)、LoG sigma 1–5 mm" if filtered else "原始图像"
    logger.info("IBSI 协议 %s，特征类 %s，图像 %s", protocol, ",".join(IBSI_CLASSES), image_types)
    studies = discover_subject_studies(args.interim_root)
    if args.patient_id:
        studies = [item for item in studies if item[0] == args.patient_id]
    stem = "stage6_ibsi_filtered" if filtered else "stage6_ibsi"
    jsonl_path = args.output_dir / f"{stem}.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if args.resume and jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["患者ID"], record["检查日期"])
            if key in done:
                continue
            done.add(key)
            rows.append(record)
        logger.info("续跑，已有 IBSI %d 次检查", len(done))
    elif jsonl_path.exists() and not args.patient_id:
        jsonl_path.unlink()
    roles = _timepoint_lookup(args.output_dir)
    studies = [item for item in studies if (item[0], item[1]) not in done]
    logger.info("待提取 IBSI %d", len(studies))
    payloads = []
    for patient_id, study_date, study_dir in studies:
        payloads.append({
            "patient_id": patient_id,
            "study_date": study_date,
            "study_dir": str(study_dir),
            "processed_root": str(args.processed_root),
            "name": names.get(patient_id.lstrip("0") or "0", ""),
            "timepoint": roles.get((patient_id, study_date), ""),
            "use_filters": filtered,
        })

    def store(record: dict) -> None:
        rows.append(record)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(ibsi_for_study, payload) for payload in payloads]
            for finished, future in enumerate(as_completed(futures), start=1):
                store(future.result())
                if finished % 10 == 0 or finished == len(futures):
                    logger.info("IBSI 已完成 %d/%d", finished, len(futures))
    else:
        for index, payload in enumerate(payloads, start=1):
            store(ibsi_for_study(payload))
            logger.info("IBSI 已完成 %d/%d %s %s", index, len(payloads), payload["patient_id"], payload["study_date"])
    rows.sort(key=lambda item: (item["患者ID"], item["检查日期"]))
    csv_path = args.output_dir / f"{stem}_radiomics.csv"
    _write_table(csv_path, IBSI_BASE_FIELDS, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["状态"])] = counts.get(str(row["状态"]), 0) + 1
    logger.info("IBSI %s，写出 %s", counts, csv_path)
    return 1 if counts.get("error") else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="提取第五至第八阶段量化指标。")
    parser.add_argument("--interim-root", type=Path, default=PROJECT_ROOT / "data" / "interim")
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--excel", type=Path, default=PROJECT_ROOT / "data" / "弥漫大B细胞性淋巴瘤 终版.xlsx")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "tables" / "stages_5_8")
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-radiomics", action="store_true")
    parser.add_argument("--resume", action="store_true", help="跳过已经完成的检查")
    parser.add_argument("--radiomics-only", action="store_true", help="只重算第六阶段的全套 IBSI 原始特征")
    parser.add_argument("--radiomics-filtered", action="store_true", help="提取原始、小波和 LoG 的高通量 IBSI 特征")
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT_ROOT / "logs" / f"extract_stages_5_8_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    if args.skip_radiomics:
        os.environ["DLBCL_SKIP_RADIOMICS"] = "1"
    logger.info("协议 %s", PROTOCOL)
    logger.info("日志 %s", log_path)
    logger.info("最小连通灶 %s 体素；部分容积参考体积 %s cm3；PET bin width 0.25；CT bin width 25", MIN_VOXELS, PVE_VOLUME_CM3)
    logger.info("不用 SUV≥2.5 做肿瘤分割。体成分只接受主导椎体为 L3 的层面。")

    names, sexes = _read_excel(args.excel)
    logger.info("Excel 姓名 %d 条", len(names))
    if args.radiomics_filtered or args.radiomics_only:
        return run_ibsi(args, names, filtered=args.radiomics_filtered)
    demographics = _dicom_demographics(args.data_root)
    logger.info("DICOM 人口学 %d 次检查", len(demographics))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dicom_demographics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {f"{pid}|{date}": record for (pid, date), record in demographics.items()},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    studies = discover_subject_studies(args.interim_root)
    if args.patient_id:
        studies = [item for item in studies if item[0] == args.patient_id]
    jsonl_path = args.output_dir / "studies.jsonl"
    results: list[dict] = []
    done: set[tuple[str, str]] = set()
    if args.resume and jsonl_path.is_file():
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["stage5"]["患者ID"], record["stage5"]["检查日期"])
                if key in done:
                    continue
                done.add(key)
                results.append(record)
        logger.info("续跑，已有 %d 次检查", len(done))
    elif jsonl_path.exists():
        jsonl_path.unlink()
    studies = [item for item in studies if (item[0], item[1]) not in done]
    logger.info("待处理检查 %d", len(studies))
    payloads = []
    for patient_id, study_date, study_dir in studies:
        demo = dict(_lookup(demographics, patient_id, study_date) or {})
        excel_sex = sexes.get(patient_id.lstrip("0") or "0", "")
        if not demo.get("sex"):
            demo["sex"] = excel_sex
        payloads.append({
            "patient_id": patient_id,
            "study_date": study_date,
            "study_dir": str(study_dir),
            "processed_root": str(args.processed_root),
            "name": names.get(patient_id.lstrip("0") or "0", ""),
            "demo": demo,
            "skip_radiomics": args.skip_radiomics,
        })

    def store(record: dict) -> None:
        results.append(record)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_study, payload) for payload in payloads]
            for finished, future in enumerate(as_completed(futures), start=1):
                store(future.result())
                if finished % 10 == 0 or finished == len(futures):
                    logger.info("已完成 %d/%d", finished, len(futures))
    else:
        for index, payload in enumerate(payloads, start=1):
            store(process_study(payload))
            logger.info(
                "已完成 %d/%d %s %s 状态 %s",
                index,
                len(payloads),
                payload["patient_id"],
                payload["study_date"],
                results[-1]["stage5"]["状态"],
            )

    _assign_roles(results)
    results.sort(key=lambda item: (item["stage5"]["患者ID"], item["stage5"]["检查日期"]))
    stage5_rows = [item["stage5"] for item in results]
    stage6_rows = [item["stage6"] for item in results]
    stage7_rows = [item["stage7"] for item in results]
    stage8_rows = build_longitudinal(stage5_rows, names)
    log_rows = _log_rows(results)
    for row in stage8_rows:
        log_rows.append({
            "阶段": "8纵向",
            "患者ID": row["患者ID"],
            "检查日期": row["中期日期"] or row["基线日期"],
            "状态": row["状态"],
            "说明": row["说明"],
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_table(args.output_dir / "stage5_metabolic.csv", STAGE5_FIELDS, stage5_rows)
    _write_table(args.output_dir / "stage6_spatial_radiomics.csv", STAGE6_FIELDS, stage6_rows)
    _write_table(args.output_dir / "stage7_body_composition.csv", STAGE7_FIELDS, stage7_rows)
    _write_table(args.output_dir / "stage8_longitudinal.csv", STAGE8_FIELDS, stage8_rows)
    _write_table(args.output_dir / "run_log.csv", LOG_FIELDS, log_rows)
    for label, rows in (
        ("第五阶段", stage5_rows),
        ("第六阶段", stage6_rows),
        ("第七阶段", stage7_rows),
        ("第八阶段", stage8_rows),
    ):
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["状态"]] = counts.get(row["状态"], 0) + 1
        logger.info("%s %s", label, counts)
    logger.info("写出 %s", args.output_dir)
    errors = sum(1 for row in log_rows if row["状态"] == "error")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
