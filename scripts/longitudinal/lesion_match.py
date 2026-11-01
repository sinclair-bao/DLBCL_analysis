#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
基线 CT → 中期 CT 的刚体、仿射和 SyN 配准，再用 AutoPET mask 做病灶匹配。

固定图像是中期 2 mm CT，移动图像是基线 2 mm CT。形变不使用 PET。
人工勾画的 mask 不参与。过大或 SUV 不可能的连通域记为 invalid。
同一或相邻站点、20 mm 内记为 matched；20–40 mm 记为 review。
大灶按表面距离可对应多块小灶。残余 SUV 取病灶核心，并排除脑、肾、膀胱和心脏；
只有接近肝脏本底才记为 resolved，其余未配上的基线灶记为 unmatched。
中期未能配对的病灶记为 new。

产物：
    data/processed/<ID>/<中期日期>/longitudinal/match/
    results/tables/lesion_match_pilot.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.spatial import cKDTree

_SCRIPTS = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _SCRIPTS / "common"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline_utils import setup_logging  # noqa: E402

from ants_runner import resolve_ants_bin  # noqa: E402
from catalog import ORGAN_LIVER, DataCatalog, StudyAssets  # noqa: E402
from stations import ST_BONE, build_station_mask, station_name, stations_compatible  # noqa: E402

logger = logging.getLogger("lesion_match")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "tables" / "lesion_match_pilot.csv"

PILOT_PAIRS = (
    ("00136597", "20220425", "20220728"),
    ("00598808", "20190930", "20200103"),
    ("00673130", "20220117", "20220616"),
)

MIN_COMPONENT_VOXELS = 10
MAX_DISTANCE_MM = 20.0
REVIEW_DISTANCE_MM = 40.0
W_DISTANCE = 1.0
W_IOU = 20.0
INELIGIBLE_COST = 1.0e12
LARGE_COMPONENT_ML = 20.0
INVALID_COMPONENT_ML = 200.0
INVALID_SUV_MAX = 100.0
RESIDUAL_LIVER_RATIO = 1.5
RESIDUAL_SUV_FALLBACK = 2.5
CORE_RADIUS_MM = 15.0
# organs.nii.gz：肾、膀胱、脑、心脏。残余 SUV 不计这些生理性高摄取。
PHYSIOLOGIC_ORGANS = (2, 4, 6, 7)

MATCH_FIELDS = [
    "patient_id",
    "baseline_date",
    "interim_date",
    "match_status",
    "baseline_lesion_id",
    "interim_lesion_id",
    "station",
    "centroid_distance_mm",
    "iou",
    "cost",
    "baseline_mtv_ml",
    "baseline_suv_max",
    "baseline_suv_mean",
    "interim_mtv_ml",
    "interim_suv_max",
    "interim_suv_mean",
    "residual_suv_max",
    "residual_suv_mean",
    "liver_suv_mean",
    "suv_max_liver_ratio",
]


def label_lesions(mask: np.ndarray, min_voxels: int = MIN_COMPONENT_VOXELS) -> np.ndarray:
    """26 连通域。小于 min_voxels 的碎点丢弃，其余从 1 连续编号。"""
    binary = np.asarray(mask) > 0
    labeled, n_components = ndimage.label(binary, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_components == 0:
        return np.zeros(mask.shape, dtype=np.uint16)
    counts = np.bincount(labeled.ravel())
    out = np.zeros(labeled.shape, dtype=np.uint16)
    next_id = 1
    for label_id in range(1, n_components + 1):
        if int(counts[label_id]) < min_voxels:
            continue
        out[labeled == label_id] = next_id
        next_id += 1
    return out


def measure_components(labels: np.ndarray, pet: np.ndarray, affine: np.ndarray) -> dict[int, dict]:
    pet = np.asarray(pet, dtype=np.float64)
    voxel_ml = abs(float(np.linalg.det(np.asarray(affine)[:3, :3]))) / 1000.0
    measured: dict[int, dict] = {}
    for label_id in [int(v) for v in np.unique(labels) if int(v) > 0]:
        selected = labels == label_id
        values = pet[selected]
        values = values[np.isfinite(values)]
        n_voxels = int(selected.sum())
        measured[label_id] = {
            "n_voxels": n_voxels,
            "mtv_ml": voxel_ml * n_voxels,
            "suv_max": float(values.max()) if values.size else float("nan"),
            "suv_mean": float(values.mean()) if values.size else float("nan"),
        }
    return measured


def pair_cost(
    distance_mm: float,
    iou: float,
    compatible: bool,
    *,
    max_distance_mm: float = REVIEW_DISTANCE_MM,
    w_distance: float = W_DISTANCE,
    w_iou: float = W_IOU,
) -> Optional[float]:
    """站点不兼容或距离超过复核半径时不可配对。"""
    if not compatible or distance_mm > max_distance_mm:
        return None
    return w_distance * float(distance_mm) + w_iou * (1.0 - float(iou))


def build_match_rows(
    baseline_labels: np.ndarray,
    interim_labels: np.ndarray,
    baseline_meta: dict[int, dict],
    interim_meta: dict[int, dict],
    interim_pet: np.ndarray,
    stations: np.ndarray,
    affine: np.ndarray,
    liver_suv_mean: float,
    organs: np.ndarray | None = None,
    *,
    max_distance_mm: float = MAX_DISTANCE_MM,
    review_distance_mm: float = REVIEW_DISTANCE_MM,
    w_distance: float = W_DISTANCE,
    w_iou: float = W_IOU,
) -> list[dict]:
    """baseline_labels 已经在中期网格上。baseline_meta 来自形变前的基线 PET。"""
    interim_pet = np.asarray(interim_pet, dtype=np.float64)
    baseline_geom = _geometries(baseline_labels, stations, affine)
    interim_geom = _geometries(interim_labels, stations, affine)
    valid_baseline = {
        label_id: geom for label_id, geom in baseline_geom.items()
        if not _component_is_invalid(baseline_meta.get(label_id, {}))
    }
    valid_interim = {
        label_id: geom for label_id, geom in interim_geom.items()
        if not _component_is_invalid(interim_meta.get(label_id, {}))
    }
    matched, review = _assign_pairs(
        valid_baseline, valid_interim, baseline_meta, interim_meta,
        baseline_labels, interim_labels, affine,
        max_distance_mm=max_distance_mm, review_distance_mm=review_distance_mm,
        w_distance=w_distance, w_iou=w_iou,
    )
    used_baseline = {item[0] for item in matched} | {item[0] for item in review}
    used_interim = {item[1] for item in matched} | {item[1] for item in review}
    rows: list[dict] = []

    for status, pairs in (("matched", matched), ("review", review)):
        for baseline_id, interim_id, distance_mm, iou, cost in pairs:
            base = baseline_meta.get(baseline_id, {})
            follow = interim_meta.get(interim_id, {})
            rows.append(_row(
                status,
                baseline_id,
                interim_id,
                baseline_geom[baseline_id]["station"],
                distance_mm,
                iou,
                cost,
                base,
                follow,
                float("nan"),
                float("nan"),
                liver_suv_mean,
                _ratio(follow.get("suv_max", float("nan")), liver_suv_mean),
            ))

    for baseline_id, geom in baseline_geom.items():
        if baseline_id in used_baseline:
            continue
        base = baseline_meta.get(baseline_id, {})
        if _component_is_invalid(base):
            rows.append(_row(
                "invalid", baseline_id, None, geom["station"], None, None, None,
                base, {}, float("nan"), float("nan"), liver_suv_mean, float("nan"),
            ))
            continue
        residual_max, residual_mean = _core_residual(
            interim_pet, geom["coords"], geom["centroid"], affine, organs,
        )
        status = "resolved" if _near_background(residual_max, liver_suv_mean) else "unmatched"
        rows.append(_row(
            status,
            baseline_id,
            None,
            geom["station"],
            None,
            None,
            None,
            base,
            {"mtv_ml": 0.0} if status == "resolved" else {},
            residual_max,
            residual_mean,
            liver_suv_mean,
            _ratio(residual_max, liver_suv_mean),
        ))

    for interim_id, geom in interim_geom.items():
        if interim_id in used_interim:
            continue
        follow = interim_meta.get(interim_id, {})
        status = "invalid" if _component_is_invalid(follow) else "new"
        rows.append(_row(
            status,
            None,
            interim_id,
            geom["station"],
            None,
            None,
            None,
            {},
            follow,
            float("nan"),
            float("nan"),
            liver_suv_mean,
            _ratio(follow.get("suv_max", float("nan")), liver_suv_mean),
        ))
    return rows


def _assign_pairs(
    baseline_geom: dict[int, dict],
    interim_geom: dict[int, dict],
    baseline_meta: dict[int, dict],
    interim_meta: dict[int, dict],
    baseline_labels: np.ndarray,
    interim_labels: np.ndarray,
    affine: np.ndarray,
    *,
    max_distance_mm: float,
    review_distance_mm: float,
    w_distance: float,
    w_iou: float,
) -> tuple[list[tuple], list[tuple]]:
    """小灶一一配对；大灶可以用表面距离接受多块对侧小灶。"""
    if not baseline_geom or not interim_geom:
        return [], []
    candidates: list[tuple] = []
    for baseline_id, baseline in baseline_geom.items():
        for interim_id, interim in interim_geom.items():
            compatible = stations_compatible(baseline["station"], interim["station"])
            distance = _lesion_distance(
                baseline, interim,
                _is_large(baseline_meta.get(baseline_id, {})),
                _is_large(interim_meta.get(interim_id, {})),
                affine,
            )
            overlap = 0.0
            if compatible and distance <= review_distance_mm:
                overlap = _iou(baseline_labels, baseline_id, interim_labels, interim_id)
            value = pair_cost(
                distance, overlap, compatible,
                max_distance_mm=review_distance_mm, w_distance=w_distance, w_iou=w_iou,
            )
            if value is None:
                continue
            candidates.append((
                value, distance, overlap, baseline_id, interim_id,
                _is_large(baseline_meta.get(baseline_id, {})),
            ))
    matched = _take_pairs(candidates, max_distance_mm)
    used_baseline = {item[0] for item in matched if not _is_large(baseline_meta.get(item[0], {}))}
    used_interim = {item[1] for item in matched}
    review = _take_pairs(
        [item for item in candidates if item[1] > max_distance_mm],
        review_distance_mm,
        used_baseline=used_baseline,
        used_interim=used_interim,
    )
    return matched, review


def _take_pairs(
    candidates: list[tuple],
    max_distance_mm: float,
    used_baseline: set[int] | None = None,
    used_interim: set[int] | None = None,
) -> list[tuple]:
    used_baseline = set(used_baseline or ())
    used_interim = set(used_interim or ())
    chosen: list[tuple] = []
    for cost, distance, overlap, baseline_id, interim_id, baseline_large in sorted(candidates, key=lambda item: item[0]):
        if distance > max_distance_mm or interim_id in used_interim or baseline_id in used_baseline:
            continue
        chosen.append((baseline_id, interim_id, distance, overlap, cost))
        used_interim.add(interim_id)
        if not baseline_large:
            used_baseline.add(baseline_id)
    return chosen


def _lesion_distance(
    baseline: dict,
    interim: dict,
    baseline_large: bool,
    interim_large: bool,
    affine: np.ndarray,
) -> float:
    if baseline_large and not interim_large:
        return _point_to_mask_mm(interim["centroid"], baseline["coords"], affine)
    if interim_large and not baseline_large:
        return _point_to_mask_mm(baseline["centroid"], interim["coords"], affine)
    if baseline_large and interim_large:
        return min(
            _point_to_mask_mm(baseline["centroid"], interim["coords"], affine),
            _point_to_mask_mm(interim["centroid"], baseline["coords"], affine),
        )
    return float(np.linalg.norm(baseline["centroid"] - interim["centroid"]))


def _point_to_mask_mm(point: np.ndarray, coords: np.ndarray, affine: np.ndarray) -> float:
    step = max(1, len(coords) // 30000)
    sample = coords[::step]
    world = nib.affines.apply_affine(affine, sample.astype(np.float64))
    return float(cKDTree(world).query(np.asarray(point, dtype=np.float64), k=1)[0])


def _geometries(labels: np.ndarray, stations: np.ndarray, affine: np.ndarray) -> dict[int, dict]:
    geometries: dict[int, dict] = {}
    for label_id in [int(v) for v in np.unique(labels) if int(v) > 0]:
        coords = np.argwhere(labels == label_id)
        if len(coords) == 0:
            continue
        centroid = nib.affines.apply_affine(affine, coords.mean(axis=0))
        values = stations[coords[:, 0], coords[:, 1], coords[:, 2]]
        geometries[label_id] = {
            "centroid": np.asarray(centroid, dtype=np.float64),
            "coords": coords,
            "station": _majority_station(values),
        }
    return geometries


def _majority_station(values: np.ndarray) -> int:
    """贴着椎体的软组织灶不因骨骼体素占优而被标成 bone。"""
    nonzero = values[values > 0]
    if nonzero.size == 0:
        return 0
    soft = nonzero[nonzero != ST_BONE]
    chosen = soft if soft.size else nonzero
    return int(np.bincount(chosen.astype(np.int64)).argmax())


def _component_is_invalid(meta: dict) -> bool:
    mtv = meta.get("mtv_ml")
    suv = meta.get("suv_max")
    if mtv is not None and math.isfinite(float(mtv)) and float(mtv) > INVALID_COMPONENT_ML:
        return True
    return suv is not None and math.isfinite(float(suv)) and float(suv) > INVALID_SUV_MAX


def _is_large(meta: dict) -> bool:
    mtv = meta.get("mtv_ml")
    return mtv is not None and math.isfinite(float(mtv)) and float(mtv) >= LARGE_COMPONENT_ML


def _near_background(suv_max: float, liver_suv_mean: float) -> bool:
    if not math.isfinite(suv_max):
        return False
    if math.isfinite(liver_suv_mean) and liver_suv_mean > 0:
        return suv_max <= RESIDUAL_LIVER_RATIO * liver_suv_mean
    return suv_max <= RESIDUAL_SUV_FALLBACK


def _iou(baseline_labels: np.ndarray, baseline_id: int, interim_labels: np.ndarray, interim_id: int) -> float:
    baseline_coords = np.argwhere(baseline_labels == baseline_id)
    interim_coords = np.argwhere(interim_labels == interim_id)
    if len(baseline_coords) == 0 or len(interim_coords) == 0:
        return 0.0
    origin = np.minimum(baseline_coords.min(axis=0), interim_coords.min(axis=0))
    stop = np.maximum(baseline_coords.max(axis=0), interim_coords.max(axis=0)) + 1
    shape = tuple(int(v) for v in (stop - origin))
    baseline_crop = np.zeros(shape, dtype=bool)
    interim_crop = np.zeros(shape, dtype=bool)
    baseline_local = baseline_coords - origin
    interim_local = interim_coords - origin
    baseline_crop[baseline_local[:, 0], baseline_local[:, 1], baseline_local[:, 2]] = True
    interim_crop[interim_local[:, 0], interim_local[:, 1], interim_local[:, 2]] = True
    intersection = int(np.count_nonzero(baseline_crop & interim_crop))
    union = int(np.count_nonzero(baseline_crop | interim_crop))
    if union == 0:
        return 0.0
    return intersection / union


def _core_residual(
    pet: np.ndarray,
    coords: np.ndarray,
    centroid: np.ndarray,
    affine: np.ndarray,
    organs: np.ndarray | None,
) -> tuple[float, float]:
    """在病灶核心内取残余 SUV，并去掉脑、肾、膀胱和心脏。"""
    world = nib.affines.apply_affine(affine, coords.astype(np.float64))
    core = np.linalg.norm(world - centroid, axis=1) <= CORE_RADIUS_MM
    keep = core
    if organs is not None:
        organ_values = organs[coords[:, 0], coords[:, 1], coords[:, 2]]
        physiologic = np.isin(organ_values, PHYSIOLOGIC_ORGANS)
        keep = core & ~physiologic
        if not np.any(keep):
            keep = ~physiologic
    if not np.any(keep):
        keep = np.ones(len(coords), dtype=bool)
    selected = coords[keep]
    values = np.asarray(pet, dtype=np.float64)[selected[:, 0], selected[:, 1], selected[:, 2]]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.max()), float(values.mean())


def _ratio(suv_max: float, liver_suv_mean: float) -> float:
    if not math.isfinite(suv_max) or not math.isfinite(liver_suv_mean) or liver_suv_mean <= 0:
        return float("nan")
    return float(suv_max) / float(liver_suv_mean)


def _row(
    status: str,
    baseline_id: int | None,
    interim_id: int | None,
    station: int,
    distance_mm: float | None,
    iou: float | None,
    cost: float | None,
    baseline_meta: dict,
    interim_meta: dict,
    residual_max: float,
    residual_mean: float,
    liver_suv_mean: float,
    ratio: float,
) -> dict:
    interim_mtv = interim_meta.get("mtv_ml", float("nan"))
    if status == "resolved":
        interim_mtv = 0.0
    return {
        "match_status": status,
        "baseline_lesion_id": "" if baseline_id is None else int(baseline_id),
        "interim_lesion_id": "" if interim_id is None else int(interim_id),
        "station": station_name(station),
        "centroid_distance_mm": _round(distance_mm),
        "iou": _round(iou),
        "cost": _round(cost),
        "baseline_mtv_ml": _round(baseline_meta.get("mtv_ml")),
        "baseline_suv_max": _round(baseline_meta.get("suv_max")),
        "baseline_suv_mean": _round(baseline_meta.get("suv_mean")),
        "interim_mtv_ml": _round(interim_mtv),
        "interim_suv_max": _round(interim_meta.get("suv_max")),
        "interim_suv_mean": _round(interim_meta.get("suv_mean")),
        "residual_suv_max": _round(residual_max if status in ("resolved", "unmatched") else None),
        "residual_suv_mean": _round(residual_mean if status in ("resolved", "unmatched") else None),
        "liver_suv_mean": _round(liver_suv_mean),
        "suv_max_liver_ratio": _round(ratio),
    }


def _round(value: object, digits: int = 4) -> str | float:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return round(number, digits)


def _prefix(baseline_date: str) -> str:
    return f"baseline_{baseline_date}_to_interim_"


def _match_dir(interim: StudyAssets) -> Path:
    return interim.processed_dir / "longitudinal" / "match"


def _affine_path(out_dir: Path, baseline_date: str) -> Path:
    return out_dir / f"{_prefix(baseline_date)}0GenericAffine.mat"


def _warp_path(out_dir: Path, baseline_date: str) -> Path:
    return out_dir / f"{_prefix(baseline_date)}1Warp.nii.gz"


def register_and_warp(
    baseline: StudyAssets,
    interim: StudyAssets,
    ants_bin: Path,
    *,
    threads: int = 8,
    overwrite: bool = False,
) -> Path:
    """CT→CT Rigid+Affine+SyN，并把基线 CT、PET、病灶 mask 变到中期网格。"""
    if baseline.ct_iso is None or interim.ct_iso is None:
        raise FileNotFoundError("基线或中期缺少 2 mm CT")
    if baseline.pet_iso is None or interim.pet_iso is None:
        raise FileNotFoundError("基线或中期缺少 2 mm PET")
    if baseline.lesion_auto is None:
        raise FileNotFoundError("基线缺少 AutoPET 病灶 mask")

    out_dir = _match_dir(interim)
    out_dir.mkdir(parents=True, exist_ok=True)
    affine = _affine_path(out_dir, baseline.study_date)
    warp = _warp_path(out_dir, baseline.study_date)
    warped_ct = out_dir / "baseline_ct_syn.nii.gz"
    warped_pet = out_dir / "baseline_pet_syn.nii.gz"
    warped_lesion = out_dir / "baseline_lesion_syn.nii.gz"
    registration_ready = all(path.is_file() for path in (affine, warp, warped_ct, warped_pet))
    registration, apply_transforms, _convert = resolve_ants_bin(ants_bin)
    if registration_ready and not overwrite:
        logger.info("已有 SyN 变换，跳过配准: %s", out_dir)
    else:
        _run_syn(
            registration, interim.ct_iso, baseline.ct_iso, out_dir, baseline.study_date, warped_ct, threads,
        )
        _apply(apply_transforms, baseline.pet_iso, interim.ct_iso, warped_pet, warp, affine, "Linear")
    _apply(apply_transforms, baseline.lesion_auto, interim.ct_iso, warped_lesion, warp, affine, "GenericLabel")
    return out_dir


def warp_labeled_mask(
    baseline_labels_path: Path,
    reference_ct: Path,
    out_path: Path,
    warp: Path,
    affine: Path,
    ants_bin: Path,
) -> None:
    _registration, apply_transforms, _convert = resolve_ants_bin(ants_bin)
    _apply(apply_transforms, baseline_labels_path, reference_ct, out_path, warp, affine, "GenericLabel")


def _run_syn(
    registration: Path,
    fixed_ct: Path,
    moving_ct: Path,
    out_dir: Path,
    baseline_date: str,
    warped_ct: Path,
    threads: int,
) -> None:
    prefix = str(out_dir / _prefix(baseline_date))
    metric_mi = f"MI[{fixed_ct},{moving_ct},1,32,Regular,0.25]"
    metric_cc = f"CC[{fixed_ct},{moving_ct},1,4]"
    command = [
        str(registration),
        "--dimensionality", "3",
        "--float", "1",
        "--output", f"[{prefix},{warped_ct}]",
        "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "1",
        "--initial-moving-transform", f"[{fixed_ct},{moving_ct},1]",
        "--transform", "Rigid[0.1]",
        "--metric", metric_mi,
        "--convergence", "[1000x500x250x0,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--transform", "Affine[0.1]",
        "--metric", metric_mi,
        "--convergence", "[1000x500x250x0,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--transform", "SyN[0.1,3,0]",
        "--metric", metric_cc,
        "--convergence", "[100x70x50x0,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--collapse-output-transforms", "1",
        "--verbose", "1",
    ]
    try:
        _run_syn_command(command, out_dir, baseline_date, fixed_ct, moving_ct, threads)
    except RuntimeError as exc:
        # 这台机器上 ANTs 多线程会偶发 SIGFPE（exit 为负数）。单线程是现有跨检查配准的稳定设置。
        if threads > 1 and "exit=-" in str(exc):
            logger.warning("多线程 antsRegistration 被信号终止，改用单线程重试。")
            _run_syn_command(command, out_dir, baseline_date, fixed_ct, moving_ct, 1)
        else:
            raise


def _run_syn_command(
    command: list[str],
    out_dir: Path,
    baseline_date: str,
    fixed_ct: Path,
    moving_ct: Path,
    threads: int,
    env: dict[str, str] | None = None,
) -> None:
    if env is None:
        env = os.environ.copy()
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = "1"
    log_path = out_dir / "ants_syn.log"
    logger.info("SyN 配准 fixed=%s moving=%s threads=%d", fixed_ct.name, moving_ct.name, threads)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False)
    if result.returncode != 0:
        tail = _tail(log_path)
        raise RuntimeError(f"antsRegistration 失败 (exit={result.returncode})\n{tail}")
    warp = _warp_path(out_dir, baseline_date)
    affine = _affine_path(out_dir, baseline_date)
    if not warp.is_file() or not affine.is_file():
        raise RuntimeError(f"SyN 结束但缺少变换文件: {affine.name} / {warp.name}")


def _apply(
    apply_transforms: Path,
    moving: Path,
    reference: Path,
    output: Path,
    warp: Path,
    affine: Path,
    interpolation: str,
) -> None:
    def command(interp: str) -> list[str]:
        return [
            str(apply_transforms),
            "-d", "3",
            "-i", str(moving),
            "-r", str(reference),
            "-o", str(output),
            "-t", str(warp),
            "-t", str(affine),
            "-n", interp,
            "-v", "1",
        ]

    try:
        _run_logged(command(interpolation), apply_transforms.parent, f"apply {output.name}")
    except RuntimeError:
        if interpolation != "GenericLabel":
            raise
        logger.warning("GenericLabel 不可用，改用 NearestNeighbor: %s", output.name)
        _run_logged(command("NearestNeighbor"), apply_transforms.parent, f"apply {output.name}")


def _run_logged(command: list[str], ants_bin: Path, label: str) -> None:
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = env.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "8")
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "8")
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} 失败 (exit={result.returncode})\n{(result.stderr or result.stdout)[-2000:]}")


def _tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def resample_nearest(src: nib.Nifti1Image, reference: nib.Nifti1Image) -> np.ndarray:
    if src.shape == reference.shape and np.allclose(src.affine, reference.affine, atol=1e-3):
        data = np.asanyarray(src.dataobj)
    else:
        data = np.asanyarray(resample_from_to(src, (reference.shape, reference.affine), order=0).dataobj)
    return np.rint(data).astype(np.int16)


def _load_array(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    return image, np.asanyarray(image.dataobj)


def _save_image(data: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    image = nib.Nifti1Image(data, reference.affine, reference.header)
    image.set_data_dtype(data.dtype)
    nib.save(image, str(path))


def _liver_mean(interim: StudyAssets, pet_image: nib.Nifti1Image, pet: np.ndarray) -> float:
    if interim.organs is None or not interim.organs.is_file():
        return float("nan")
    organs = resample_nearest(nib.load(str(interim.organs)), pet_image)
    selected = organs == ORGAN_LIVER
    if not np.any(selected):
        return float("nan")
    values = np.asarray(pet, dtype=np.float64)[selected]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def _on_reference(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    return resample_nearest(nib.load(str(path)), reference)


def match_pair(
    baseline: StudyAssets,
    interim: StudyAssets,
    ants_bin: Path,
    *,
    threads: int = 8,
    overwrite: bool = False,
) -> list[dict]:
    if baseline.lesion_auto is None or interim.lesion_auto is None:
        raise FileNotFoundError("基线或中期缺少 AutoPET 病灶 mask")
    out_dir = register_and_warp(baseline, interim, ants_bin, threads=threads, overwrite=overwrite)
    matches_path = out_dir / "matches.csv"
    reference = nib.load(str(interim.ct_iso))
    _pet_image, interim_pet = _load_array(interim.pet_iso)
    if interim_pet.shape != reference.shape or not np.allclose(_pet_image.affine, reference.affine, atol=1e-3):
        interim_pet = resample_nearest(_pet_image, reference).astype(np.float32)
        _pet_image = reference
    baseline_pet_image, baseline_pet = _load_array(baseline.pet_iso)
    baseline_mask = _on_reference(baseline.lesion_auto, baseline_pet_image)
    baseline_labels = label_lesions(baseline_mask)
    baseline_meta = measure_components(baseline_labels, baseline_pet, baseline_pet_image.affine)
    labels_path = out_dir / "baseline_components.nii.gz"
    _save_image(baseline_labels, baseline_pet_image, labels_path)

    warped_labels_path = out_dir / "components_baseline.nii.gz"
    warp_labeled_mask(
        labels_path, interim.ct_iso, warped_labels_path,
        _warp_path(out_dir, baseline.study_date),
        _affine_path(out_dir, baseline.study_date),
        ants_bin,
    )
    warped = resample_nearest(nib.load(str(warped_labels_path)), reference)
    warped = np.clip(warped, 0, None).astype(np.uint16)
    _save_image(warped, reference, warped_labels_path)

    interim_mask = _on_reference(interim.lesion_auto, reference)
    interim_labels = label_lesions(interim_mask)
    _save_image(interim_labels, reference, out_dir / "components_interim.nii.gz")
    interim_meta = measure_components(interim_labels, interim_pet, reference.affine)

    totalseg_path = interim.processed_dir / "organs" / "_totalseg_staging" / "totalseg_wholebody.nii.gz"
    if not totalseg_path.is_file():
        raise FileNotFoundError(f"缺少 TotalSegmentator 全身标签: {totalseg_path}")
    totalseg = resample_nearest(nib.load(str(totalseg_path)), reference)
    stations = build_station_mask(totalseg, reference.affine)
    _save_image(stations, reference, out_dir / "stations.nii.gz")

    liver = _liver_mean(interim, reference, interim_pet)
    organs = None
    if interim.organs is not None and interim.organs.is_file():
        organs = resample_nearest(nib.load(str(interim.organs)), reference)
    rows = build_match_rows(
        warped, interim_labels, baseline_meta, interim_meta, interim_pet, stations,
        reference.affine, liver, organs,
    )
    for row in rows:
        row["patient_id"] = interim.patient_id
        row["baseline_date"] = baseline.study_date
        row["interim_date"] = interim.study_date
    _write_rows(rows, matches_path)
    logger.info(
        "%s %s→%s 匹配 %d，复核 %d，缓解 %d，未配上 %d，新发 %d，无效 %d",
        interim.patient_id,
        baseline.study_date,
        interim.study_date,
        sum(row["match_status"] == "matched" for row in rows),
        sum(row["match_status"] == "review" for row in rows),
        sum(row["match_status"] == "resolved" for row in rows),
        sum(row["match_status"] == "unmatched" for row in rows),
        sum(row["match_status"] == "new" for row in rows),
        sum(row["match_status"] == "invalid" for row in rows),
    )
    return rows


def _write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MATCH_FIELDS})


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基线到中期的 SyN 配准和解剖站点约束病灶匹配。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--ants-bin", type=Path, default=None)
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--interim", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _pairs_from_args(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    specified = [args.patient_id, args.baseline, args.interim]
    if any(specified):
        if not all(specified):
            raise SystemExit("指定患者时需要同时给出 --patient-id、--baseline 和 --interim")
        return [(args.patient_id, args.baseline, args.interim)]
    return list(PILOT_PAIRS)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)
    pairs = _pairs_from_args(args)
    catalog = DataCatalog(args.interim_root, args.processed_root)
    ants_bin = args.ants_bin
    if ants_bin is None:
        from ants_runner import DEFAULT_ANTS_BIN
        ants_bin = DEFAULT_ANTS_BIN

    all_rows: list[dict] = []
    failed = False
    for patient_id, baseline_date, interim_date in pairs:
        baseline = catalog.get_study(patient_id, baseline_date)
        interim = catalog.get_study(patient_id, interim_date)
        if baseline is None or interim is None:
            logger.error("找不到检查 %s %s→%s", patient_id, baseline_date, interim_date)
            failed = True
            continue
        if args.dry_run:
            logger.info(
                "[DRY-RUN] %s %s→%s ct=%s mask=%s",
                patient_id, baseline_date, interim_date, interim.ct_iso, baseline.lesion_auto,
            )
            continue
        try:
            all_rows.extend(match_pair(
                baseline, interim, ants_bin, threads=args.threads, overwrite=args.overwrite,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.exception("匹配失败 %s %s→%s: %s", patient_id, baseline_date, interim_date, exc)
            failed = True
    if args.dry_run:
        return 0
    if all_rows:
        _write_rows(all_rows, args.results_csv)
        logger.info("汇总 %d 行 -> %s", len(all_rows), args.results_csv)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
