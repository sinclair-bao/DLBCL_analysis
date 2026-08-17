#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""把一次检查的 CT/PET/mask 读到同一 2 mm 网格上的 numpy 数组。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from catalog import StudyAssets


@dataclass
class VolumeSet:
    ct: np.ndarray
    pet: np.ndarray
    native: Optional[np.ndarray]
    mapped: Optional[np.ndarray]
    study_date: str
    role: str = ""


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
    native = None
    if assets.lesion_mask is not None:
        native = _to_ref(assets.lesion_mask, ref, order=0, dtype=np.uint8)
    mapped = None
    if baseline_date:
        wp = assets.warped_baseline_mask(baseline_date)
        if wp is not None:
            mapped = _to_ref(wp, ref, order=0, dtype=np.uint8)
    return VolumeSet(
        ct=ct,
        pet=pet,
        native=native,
        mapped=mapped,
        study_date=assets.study_date,
    )
