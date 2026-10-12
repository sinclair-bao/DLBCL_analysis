#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""把一次检查的 CT/PET/mask 读到同一 2 mm 网格上的 numpy 数组。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from catalog import StudyAssets
from display_utils import coronal_mip
from mask_ops import ensure_labeled
from quality import fingerprint


@dataclass
class VolumeSet:
    ct: np.ndarray
    pet: np.ndarray
    native: np.ndarray
    mapped: Optional[np.ndarray]
    study_date: str
    patient_id: str = ""
    affine: Optional[np.ndarray] = None
    role: str = ""
    pet_units: str = "SUVbw"
    dirty: bool = False
    edited_sha256: Optional[str] = None
    native_source_path: Optional[str] = None
    native_source_sha256: Optional[str] = None
    mask_revision: int = 0
    stats_revision: int = -1
    stats_cache: list[dict] = field(default_factory=list, repr=False)
    _undo: list = field(default_factory=list, repr=False)


@dataclass
class MipData:
    pet: np.ndarray
    native: np.ndarray
    mapped: Optional[np.ndarray]
    role: str = ""
    study_date: str = ""
    volume_shape: tuple[int, int, int] = (0, 0, 0)


def build_mip_data(vol: VolumeSet) -> MipData:
    """Keep only 2D projections so inactive studies need little resident memory."""
    return MipData(
        pet=coronal_mip(vol.pet),
        native=coronal_mip(vol.native),
        mapped=coronal_mip(vol.mapped) if vol.mapped is not None else None,
        role=vol.role,
        study_date=vol.study_date,
        volume_shape=vol.ct.shape,
    )


def _to_ref(
    src_path,
    ref_img: nib.Nifti1Image,
    *,
    order: int,
    dtype,
) -> np.ndarray:
    src = nib.load(str(src_path))
    if tuple(src.shape) == tuple(ref_img.shape) and np.allclose(src.affine, ref_img.affine, atol=1e-4):
        return np.asanyarray(src.dataobj, dtype=dtype)
    warped = resample_from_to(src, (ref_img.shape, ref_img.affine), order=order)
    return np.asanyarray(warped.dataobj, dtype=dtype)


def _label_to_ref(src_path, ref_img: nib.Nifti1Image) -> np.ndarray:
    """Resample a label image without silently truncating invalid values."""
    src = nib.load(str(src_path))
    if tuple(src.shape) == tuple(ref_img.shape) and np.allclose(src.affine, ref_img.affine, atol=1e-4):
        raw = np.asanyarray(src.dataobj)
    else:
        warped = resample_from_to(src, (ref_img.shape, ref_img.affine), order=0)
        raw = np.asanyarray(warped.dataobj)
    fractional = (np.any(raw != np.floor(raw))
                  if np.issubdtype(raw.dtype, np.floating) else False)
    if (not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > np.iinfo(np.uint16).max)
            or fractional):
        raise ValueError(f"mask 必须是有限、非负的整数标签（0–65535）: {src_path}")
    return np.asarray(raw, dtype=np.uint16)


def load_volume_set(
    assets: StudyAssets,
    baseline_date: Optional[str] = None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Optional[VolumeSet]:
    if assets.working_ct is None or assets.working_pet is None:
        return None
    if progress:
        progress(0, "读取 CT")
    ref = nib.as_closest_canonical(nib.load(str(assets.working_ct)))
    if not np.isfinite(ref.affine).all() or abs(np.linalg.det(ref.affine[:3, :3])) < 1e-8:
        raise ValueError(f"CT 几何信息无效: {assets.working_ct}")
    ct = np.asanyarray(ref.dataobj, dtype=np.float32)
    if progress:
        progress(1, "CT 已读取，读取 PET")
    pet = _to_ref(assets.working_pet, ref, order=1, dtype=np.float32)
    if progress:
        progress(2, "PET 已读取，读取病灶 mask")
    native = np.zeros(ct.shape, dtype=np.uint16)
    if assets.working_lesion is not None:
        raw = _label_to_ref(assets.working_lesion, ref)
        if not np.any(raw) and np.any(np.asanyarray(nib.load(str(assets.working_lesion)).dataobj)):
            raise ValueError(f"mask 与 CT 空间不重叠: {assets.working_lesion}")
        native = ensure_labeled(raw)
    if progress:
        progress(3, "病灶 mask 已检查，读取映射 mask")
    mapped = None
    if baseline_date:
        wp = assets.warped_baseline_mask(baseline_date)
        if wp is not None:
            mapped = _label_to_ref(wp, ref)
    volume = VolumeSet(
        ct=ct,
        pet=pet,
        native=native,
        mapped=mapped,
        study_date=assets.study_date,
        patient_id=assets.patient_id,
        pet_units=assets.pet_units,
        affine=np.asarray(ref.affine),
        edited_sha256=(fingerprint(assets.lesion_edited)["sha256"]
                       if assets.lesion_edited is not None else None),
        native_source_path=str(assets.working_lesion) if assets.working_lesion else None,
        native_source_sha256=(fingerprint(assets.working_lesion)["sha256"]
                              if assets.working_lesion else None),
    )
    if progress:
        progress(4, "检查影像读取完成")
    return volume


def load_mask_from_path(path, vol: VolumeSet) -> np.ndarray:
    """把任意 NIfTI mask 最近邻重采样到工作 CT 网格并编号。"""
    if vol.affine is None:
        raise ValueError("体积缺少仿射，无法重采样 mask")
    ref = nib.Nifti1Image(np.zeros(vol.ct.shape, dtype=np.uint16), vol.affine)
    raw = _label_to_ref(path, ref)
    if not np.any(raw) and np.any(np.asanyarray(nib.load(str(path)).dataobj)):
        raise ValueError(f"导入 mask 与当前 CT 空间不重叠: {path}")
    return ensure_labeled(raw)


def save_edited_mask(vol: VolumeSet, out_path) -> None:
    """把当前编号 mask 写成 uint16 NIfTI，仿射与工作 CT 一致。"""
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    affine = vol.affine if vol.affine is not None else np.eye(4)
    img = nib.Nifti1Image(np.asarray(vol.native, dtype=np.uint16), affine)
    img.header.set_data_dtype(np.uint16)
    temp = out_path.with_name(out_path.name + ".tmp.nii.gz")
    nib.save(img, str(temp))
    temp.replace(out_path)
