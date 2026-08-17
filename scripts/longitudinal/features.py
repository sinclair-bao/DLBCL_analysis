#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   features.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    对指定时间点的 ROI 计算 PET 代谢参数与影像组学特征。

    ROI 类型：
        native_lesion     该时间点 nnU-Net 病灶 mask
        baseline_mapped   基线病灶床映射到该时间点（仅随访）

    工作图像：2 mm CT + 同机对齐 SUV（缺则回退 preprocess PET）。
    器官 mask 在原始 CT 空间，计算肝/脾参考 SUV 前重采样到 PET 网格。
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

_SCRIPTS = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _SCRIPTS / "common"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline_utils import setup_logging  # noqa: E402

from catalog import (  # noqa: E402
    ORGAN_LIVER,
    ORGAN_SPLEEN,
    DataCatalog,
    StudyAssets,
)
from session import LongitudinalSession, load_session  # noqa: E402

logger = logging.getLogger("longitudinal.features")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "tables" / "longitudinal_features.csv"
RADIOMICS_YAML = Path(__file__).resolve().parent / "radiomics_params.yaml"

# 1 cm³ 球体半径 (mm)：V = 4/3 π r³ = 1000 mm³
SUV_PEAK_RADIUS_MM = (3.0 * 1000.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

CORE_FIELDS = [
    "patient_id",
    "study_date",
    "role",
    "roi_type",
    "n_voxels",
    "volume_ml",
    "suv_max",
    "suv_mean",
    "suv_peak",
    "mtv_ml",
    "tlg",
    "liver_suv_mean",
    "spleen_suv_mean",
    "suv_max_liver_ratio",
    "pet_path",
    "mask_path",
]


def _load_nifti(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return img, data


def _voxel_volume_ml(affine: np.ndarray) -> float:
    vol_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    return vol_mm3 / 1000.0


def _binary_mask(data: np.ndarray) -> np.ndarray:
    return np.asarray(data > 0, dtype=bool)


def _resample_label_to(src_path: Path, ref_img: nib.Nifti1Image) -> np.ndarray:
    src = nib.load(str(src_path))
    warped = resample_from_to(src, (ref_img.shape, ref_img.affine), order=0)
    return np.asanyarray(warped.dataobj)


def _organ_mean(pet: np.ndarray, organs: np.ndarray, label: int) -> float:
    sel = organs == label
    if not np.any(sel):
        return float("nan")
    vals = pet[sel]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def _suv_peak(pet: np.ndarray, affine: np.ndarray, mask: np.ndarray) -> float:
    """以 SUVmax 体素为球心、1 cm³ 球内均值（近似 SUVpeak）。"""
    if not np.any(mask):
        return float("nan")
    masked = np.where(mask, pet, -np.inf)
    max_idx = np.unravel_index(int(np.argmax(masked)), pet.shape)
    zooms = np.array(nib.affines.voxel_sizes(affine)[:3], dtype=np.float64)
    radius = SUV_PEAK_RADIUS_MM
    r_vox = np.ceil(radius / np.maximum(zooms, 1e-6)).astype(int)
    i0, j0, k0 = (int(max_idx[0]), int(max_idx[1]), int(max_idx[2]))
    i_lo, i_hi = max(0, i0 - r_vox[0]), min(pet.shape[0], i0 + r_vox[0] + 1)
    j_lo, j_hi = max(0, j0 - r_vox[1]), min(pet.shape[1], j0 + r_vox[1] + 1)
    k_lo, k_hi = max(0, k0 - r_vox[2]), min(pet.shape[2], k0 + r_vox[2] + 1)

    ii, jj, kk = np.mgrid[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi]
    dist2 = (
        ((ii - i0) * zooms[0]) ** 2
        + ((jj - j0) * zooms[1]) ** 2
        + ((kk - k0) * zooms[2]) ** 2
    )
    ball = dist2 <= radius ** 2
    vals = pet[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi][ball]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def _pyradiomics_available() -> bool:
    try:
        import radiomics  # noqa: F401
        import SimpleITK  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_radiomics(
    pet_path: Path,
    ct_path: Optional[Path],
    mask_path: Path,
) -> dict[str, float]:
    """PET（及可选 CT）上的 shape / firstorder / GLCM。失败时返回空 dict。"""
    if not _pyradiomics_available():
        return {}
    try:
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError:
        return {}

    out: dict[str, float] = {}
    try:
        extractor_pet = RadiomicsFeatureExtractor(str(RADIOMICS_YAML))
        extractor_pet.settings["binWidth"] = 0.25
        pet_feats = extractor_pet.execute(str(pet_path), str(mask_path), label=1)
        for key, val in pet_feats.items():
            if key.startswith("diagnostics_"):
                continue
            try:
                out[f"pet_{key}"] = float(val)
            except (TypeError, ValueError):
                continue
    except Exception as exc:
        logger.warning("PET 组学提取失败 (%s): %s", pet_path.name, exc)

    if ct_path is not None and ct_path.is_file():
        try:
            extractor_ct = RadiomicsFeatureExtractor(str(RADIOMICS_YAML))
            extractor_ct.settings["binWidth"] = 25.0
            ct_feats = extractor_ct.execute(str(ct_path), str(mask_path), label=1)
            for key, val in ct_feats.items():
                if key.startswith("diagnostics_"):
                    continue
                try:
                    out[f"ct_{key}"] = float(val)
                except (TypeError, ValueError):
                    continue
        except Exception as exc:
            logger.warning("CT 组学提取失败 (%s): %s", ct_path.name, exc)
    return out


def compute_roi_features(
    assets: StudyAssets,
    mask_path: Path,
    roi_type: str,
    role: str,
    *,
    include_radiomics: bool = True,
) -> dict[str, Any]:
    pet_path = assets.working_pet
    ct_path = assets.working_ct
    if pet_path is None:
        raise FileNotFoundError(f"{assets.patient_id}/{assets.study_date} 缺少工作 PET")
    if not mask_path.is_file():
        raise FileNotFoundError(f"缺少 mask: {mask_path}")

    pet_img, pet = _load_nifti(pet_path)
    pet = np.asarray(pet, dtype=np.float64)
    _, mask_raw = _load_nifti(mask_path)
    if mask_raw.shape != pet.shape:
        mask_raw = _resample_label_to(mask_path, pet_img)
    mask = _binary_mask(mask_raw)

    n_voxels = int(mask.sum())
    vol_ml = _voxel_volume_ml(np.asarray(pet_img.affine)) * n_voxels
    if n_voxels == 0:
        suv_max = suv_mean = suv_peak = tlg = float("nan")
        mtv = 0.0
    else:
        vals = pet[mask]
        vals = vals[np.isfinite(vals)]
        suv_max = float(np.max(vals)) if vals.size else float("nan")
        suv_mean = float(np.mean(vals)) if vals.size else float("nan")
        suv_peak = _suv_peak(pet, np.asarray(pet_img.affine), mask)
        mtv = vol_ml
        tlg = suv_mean * mtv if math.isfinite(suv_mean) else float("nan")

    liver = spleen = ratio = float("nan")
    if assets.organs is not None:
        try:
            organs = _resample_label_to(assets.organs, pet_img)
            if organs.shape == pet.shape:
                liver = _organ_mean(pet, organs, ORGAN_LIVER)
                spleen = _organ_mean(pet, organs, ORGAN_SPLEEN)
                if math.isfinite(suv_max) and math.isfinite(liver) and liver > 0:
                    ratio = suv_max / liver
        except Exception as exc:
            logger.warning("器官参考 SUV 失败 %s/%s: %s", assets.patient_id, assets.study_date, exc)

    row: dict[str, Any] = {
        "patient_id": assets.patient_id,
        "study_date": assets.study_date,
        "role": role,
        "roi_type": roi_type,
        "n_voxels": n_voxels,
        "volume_ml": round(vol_ml, 4) if math.isfinite(vol_ml) else "",
        "suv_max": round(suv_max, 4) if math.isfinite(suv_max) else "",
        "suv_mean": round(suv_mean, 4) if math.isfinite(suv_mean) else "",
        "suv_peak": round(suv_peak, 4) if math.isfinite(suv_peak) else "",
        "mtv_ml": round(mtv, 4) if math.isfinite(mtv) else "",
        "tlg": round(tlg, 4) if math.isfinite(tlg) else "",
        "liver_suv_mean": round(liver, 4) if math.isfinite(liver) else "",
        "spleen_suv_mean": round(spleen, 4) if math.isfinite(spleen) else "",
        "suv_max_liver_ratio": round(ratio, 4) if math.isfinite(ratio) else "",
        "pet_path": str(pet_path),
        "mask_path": str(mask_path),
    }
    if include_radiomics and n_voxels > 0:
        rad = _extract_radiomics(pet_path, ct_path, mask_path)
        row.update(rad)
    return row


def extract_patient_features(
    catalog: DataCatalog,
    session: LongitudinalSession,
    *,
    include_radiomics: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rec = catalog.get_patient(session.patient_id)
    if rec is None:
        logger.warning("目录中无患者 %s", session.patient_id)
        return rows
    issues = session.validate(rec.study_dates())
    if issues:
        logger.warning("会话无效 %s: %s", session.patient_id, "; ".join(issues))
        return rows

    baseline_date = session.baseline
    for role, date in session.dates_in_order():
        assets = rec.get(date)
        if assets is None:
            continue
        if assets.lesion_mask is not None:
            try:
                rows.append(
                    compute_roi_features(
                        assets, assets.lesion_mask, "native_lesion", role,
                        include_radiomics=include_radiomics,
                    )
                )
            except Exception as exc:
                logger.error("native 特征失败 %s/%s: %s", session.patient_id, date, exc)
        if role != "baseline" and baseline_date:
            mapped = assets.warped_baseline_mask(baseline_date)
            if mapped is not None:
                try:
                    rows.append(
                        compute_roi_features(
                            assets, mapped, "baseline_mapped", role,
                            include_radiomics=include_radiomics,
                        )
                    )
                except Exception as exc:
                    logger.error("mapped 特征失败 %s/%s: %s", session.patient_id, date, exc)
            else:
                logger.info("尚无基线映射 mask：%s/%s（请先跑 interscan_register）", session.patient_id, date)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra: list[str] = []
    seen = set(CORE_FIELDS)
    for row in rows:
        for key in row:
            if key not in seen:
                extra.append(key)
                seen.add(key)
    fieldnames = CORE_FIELDS + sorted(extra)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    logger.info("写出 %d 行 → %s", len(rows), path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提取纵向 PET 代谢参数与影像组学特征。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--no-radiomics", action="store_true", help="只算代谢参数，跳过 pyradiomics")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)
    catalog = DataCatalog(args.interim_root, args.processed_root)
    include_rad = not args.no_radiomics
    if include_rad and not _pyradiomics_available():
        logger.warning("未安装 pyradiomics，仅计算代谢参数。pip install pyradiomics")
        include_rad = False

    ids = [args.patient_id] if args.patient_id else catalog.patient_ids()
    all_rows: list[dict[str, Any]] = []
    for pid in ids:
        session = load_session(catalog.processed_root, pid)
        if not session.assigned():
            if args.patient_id:
                logger.error("患者 %s 尚未指定时间点（请先在 GUI 中保存会话）。", pid)
                return 2
            continue
        rows = extract_patient_features(catalog, session, include_radiomics=include_rad)
        if rows:
            per_patient = catalog.processed_root / pid / "longitudinal_features.csv"
            write_csv(rows, per_patient)
            all_rows.extend(rows)

    if not all_rows:
        logger.info("没有可写的特征行。")
        return 0
    write_csv(all_rows, args.out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
