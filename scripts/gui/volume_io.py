#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""把一次检查的 CT/PET/mask 读到同一 2 mm 网格上的 numpy 数组。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from catalog import StudyAssets
from mask_ops import ensure_labeled


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
    dirty: bool = False
    _undo: list = field(default_factory=list, repr=False)


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


def load_volume_set(assets: StudyAssets, baseline_date: Optional[str] = None) -> Optional[VolumeSet]:
    if assets.working_ct is None or assets.working_pet is None:
        return None
    ref = nib.load(str(assets.working_ct))
    ct = np.asanyarray(ref.dataobj, dtype=np.float32)
    pet = _to_ref(assets.working_pet, ref, order=1, dtype=np.float32)
    native = np.zeros(ct.shape, dtype=np.uint16)
    if assets.working_lesion is not None:
        raw = _to_ref(assets.working_lesion, ref, order=0, dtype=np.uint16)
        native = ensure_labeled(raw)
    mapped = None
    if baseline_date:
        wp = assets.warped_baseline_mask(baseline_date)
        if wp is not None:
            mapped = _to_ref(wp, ref, order=0, dtype=np.uint16)
    return VolumeSet(
        ct=ct,
        pet=pet,
        native=native,
        mapped=mapped,
        study_date=assets.study_date,
        patient_id=assets.patient_id,
        affine=np.asarray(ref.affine),
    )


def load_mask_from_path(path, vol: VolumeSet) -> np.ndarray:
    """把任意 NIfTI mask 最近邻重采样到工作 CT 网格并编号。"""
    if vol.affine is None:
        raise ValueError("体积缺少仿射，无法重采样 mask")
    ref = nib.Nifti1Image(np.zeros(vol.ct.shape, dtype=np.uint16), vol.affine)
    raw = _to_ref(path, ref, order=0, dtype=np.uint16)
    return ensure_labeled(raw)


def save_edited_mask(vol: VolumeSet, out_path) -> None:
    """把当前编号 mask 写成 uint16 NIfTI，仿射与工作 CT 一致。"""
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    affine = vol.affine if vol.affine is not None else np.eye(4)
    img = nib.Nifti1Image(np.asarray(vol.native, dtype=np.uint16), affine)
    img.header.set_data_dtype(np.uint16)
    nib.save(img, str(out_path))
