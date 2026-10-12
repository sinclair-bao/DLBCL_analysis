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
from quality import fingerprint, lesion_uid, mapping_status, mask_review_status  # noqa: E402

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
    "pet_units",
    "tracer",
    "lesion_id",
    "lesion_uid",
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
    "pet_sha256",
    "ct_path",
    "ct_sha256",
    "mask_path",
    "mask_sha256",
    "mask_qc_status",
    "mapping_qc_status",
    "feature_protocol_version",
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
    masked = np.where(mask & np.isfinite(pet), pet, -np.inf)
    if not np.isfinite(masked).any():
        return float("nan")
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
    label: int = 1,
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
        pet_feats = extractor_pet.execute(str(pet_path), str(mask_path), label=label)
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
            ct_img = nib.load(str(ct_path))
            mask_img = nib.load(str(mask_path))
            if ct_img.shape != mask_img.shape or not np.allclose(ct_img.affine, mask_img.affine, atol=1e-4):
                import tempfile

                with tempfile.TemporaryDirectory(prefix="ct_roi_") as tmp:
                    ct_mask_path = Path(tmp) / "ct_mask.nii.gz"
                    aligned = resample_from_to(mask_img, (ct_img.shape, ct_img.affine), order=0)
                    nib.save(aligned, str(ct_mask_path))
                    ct_feats = extractor_ct.execute(str(ct_path), str(ct_mask_path), label=label)
            else:
                ct_feats = extractor_ct.execute(str(ct_path), str(mask_path), label=label)
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
    lesion_id: int | None = None,
    mapping_qc_status: str = "",
) -> dict[str, Any]:
    pet_path = assets.working_pet
    ct_path = assets.working_ct
    if pet_path is None:
        raise FileNotFoundError(f"{assets.patient_id}/{assets.study_date} 缺少工作 PET")
    if assets.pet_units != "SUVbw":
        raise ValueError(f"当前特征协议仅支持 SUVbw，收到 {assets.pet_units!r}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"缺少 mask: {mask_path}")
    mask_signature = fingerprint(mask_path)
    pet_signature = fingerprint(pet_path)
    ct_signature = fingerprint(ct_path) if ct_path is not None else None

    pet_img, pet = _load_nifti(pet_path)
    pet = np.asarray(pet, dtype=np.float64)
    _, mask_raw = _load_nifti(mask_path)
    mask_img = nib.load(str(mask_path))
    if mask_raw.shape != pet.shape or not np.allclose(mask_img.affine, pet_img.affine, atol=1e-4):
        mask_raw = _resample_label_to(mask_path, pet_img)
        if not np.any(mask_raw) and np.any(np.asanyarray(mask_img.dataobj)):
            raise ValueError(f"mask 与 PET 空间不重叠: {mask_path}")
    mask = _binary_mask(mask_raw) if lesion_id is None else np.asarray(mask_raw == lesion_id, dtype=bool)

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
        "pet_units": assets.pet_units,
        "tracer": assets.tracer,
        "lesion_id": "all" if lesion_id is None else lesion_id,
        "lesion_uid": (lesion_uid(assets.processed_dir.parents[1], assets.patient_id,
                                  assets.study_date, lesion_id, mask_path)
                       if lesion_id is not None and roi_type.startswith("native") else ""),
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
        "pet_sha256": pet_signature["sha256"],
        "ct_path": str(ct_path) if ct_path is not None else "",
        "ct_sha256": ct_signature["sha256"] if ct_signature else "",
        "mask_path": str(mask_path),
        "mask_sha256": mask_signature["sha256"],
        "mask_qc_status": mask_review_status(mask_path) if roi_type.startswith("native") else "",
        "mapping_qc_status": mapping_qc_status,
        "feature_protocol_version": "pet_suv_union_v1",
    }
    if include_radiomics and n_voxels > 0:
        # The metabolic ROI is the union of all positive labels; radiomics must
        # use the same ROI rather than silently extracting label 1 only.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="pet_roi_") as tmp:
            union_path = Path(tmp) / "union.nii.gz"
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), pet_img.affine), str(union_path))
            rad = _extract_radiomics(pet_path, ct_path, union_path)
        row.update(rad)
    if fingerprint(mask_path) != mask_signature:
        raise RuntimeError(f"计算过程中 mask 已变化: {mask_path}")
    if fingerprint(pet_path) != pet_signature or (ct_path is not None and
                                                  fingerprint(ct_path) != ct_signature):
        raise RuntimeError("计算过程中 CT/PET 影像已变化")
    return row


def extract_patient_features(
    catalog: DataCatalog,
    session: LongitudinalSession,
    *,
    include_radiomics: bool = True,
    require_qc: bool = False,
    per_lesion: bool = False,
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
        if require_qc and assets.lesion_mask is None:
            raise ValueError(f"{date} 缺少当前检查病灶 mask")
        if assets.lesion_mask is not None:
            if require_qc and mask_review_status(assets.lesion_mask) != "approved":
                raise ValueError(f"{date} mask 尚未通过质控")
            try:
                rows.append(
                    compute_roi_features(
                        assets, assets.lesion_mask, "native_lesion", role,
                        include_radiomics=include_radiomics,
                    )
                )
                if per_lesion:
                    labels = np.unique(np.asanyarray(nib.load(str(assets.lesion_mask)).dataobj))
                    for lid in labels[labels > 0]:
                        rows.append(compute_roi_features(
                            assets, assets.lesion_mask, "native_lesion_component", role,
                            include_radiomics=include_radiomics, lesion_id=int(lid),
                        ))
            except Exception as exc:
                if require_qc:
                    raise
                logger.error("native 特征失败 %s/%s: %s", session.patient_id, date, exc)
        if role != "baseline" and baseline_date:
            mapped = assets.warped_baseline_mask(baseline_date)
            if mapped is not None:
                baseline = rec.get(baseline_date)
                qc_status = mapping_status(baseline, assets) if baseline else "invalid"
                if require_qc and qc_status != "approved":
                    raise ValueError(f"{date} 基线映射尚未通过质控或已过期: {qc_status}")
                try:
                    rows.append(
                        compute_roi_features(
                            assets, mapped, "baseline_mapped", role,
                            include_radiomics=include_radiomics,
                            mapping_qc_status=qc_status,
                        )
                    )
                except Exception as exc:
                    if require_qc:
                        raise
                    logger.error("mapped 特征失败 %s/%s: %s", session.patient_id, date, exc)
            else:
                if require_qc:
                    raise ValueError(f"{date} 缺少基线病灶床映射")
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
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--require-qc", action="store_true", help="只计算已通过 mask 和映射质控的结果")
    parser.add_argument("--per-lesion", action="store_true", help="另输出每个编号病灶的特征行")
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--no-radiomics", action="store_true", help="只算代谢参数，跳过 pyradiomics")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)
    catalog = DataCatalog(args.interim_root, args.processed_root, args.manifest)
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
        rows = extract_patient_features(catalog, session, include_radiomics=include_rad,
                                        require_qc=args.require_qc, per_lesion=args.per_lesion)
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
