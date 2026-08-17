#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""切片方向、PET 伪彩与 MIP，供正交浏览与演变条共用。"""

from __future__ import annotations

import numpy as np

# 临床 PET 伪彩（与 qc_segmentation.py 一致）
_PET_STOPS = np.array(
    [
        [0.00, 0.00, 0.00, 0.00],
        [0.12, 0.25, 0.00, 0.50],
        [0.25, 0.00, 0.00, 1.00],
        [0.45, 0.00, 0.75, 0.75],
        [0.60, 0.00, 1.00, 0.00],
        [0.75, 1.00, 1.00, 0.00],
        [0.90, 1.00, 0.40, 0.00],
        [1.00, 1.00, 1.00, 1.00],
    ],
    dtype=np.float32,
)

NATIVE_RGB = np.array([1.00, 0.18, 0.18], dtype=np.float32)
MAPPED_RGB = np.array([0.15, 0.85, 1.00], dtype=np.float32)

CT_WINDOW = (-160.0, 240.0)  # 软组织窗


def pet_lut(n: int = 256) -> np.ndarray:
    """返回 (n, 3) float32 LUT。"""
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    t, r, g, b = _PET_STOPS.T
    lut = np.stack(
        [np.interp(x, t, r), np.interp(x, t, g), np.interp(x, t, b)],
        axis=1,
    ).astype(np.float32)
    return lut


PET_LUT = pet_lut()


def apply_pet_cmap(values: np.ndarray, vmin: float = 0.0, vmax: float = 6.0) -> np.ndarray:
    denom = max(vmax - vmin, 1e-6)
    idx = np.clip((values - vmin) / denom, 0.0, 1.0)
    return PET_LUT[(idx * (len(PET_LUT) - 1)).astype(np.int32)]


def normalize_ct(ct: np.ndarray, window: tuple[float, float] = CT_WINDOW) -> np.ndarray:
    lo, hi = window
    return np.clip((ct - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def orient_axial(sl: np.ndarray) -> np.ndarray:
    """RAS 轴位 → 头足方向朝向观察者：患者右在左，前在上。"""
    return np.rot90(sl, 1)


def orient_coronal(sl: np.ndarray) -> np.ndarray:
    """RAS 冠状 (R, S) → 头上、患者右在左。"""
    return sl.T[::-1, ::-1]


def orient_sagittal(sl: np.ndarray) -> np.ndarray:
    """RAS 矢状 (A, S) → 头上、前方在右。"""
    return sl.T[::-1, :]


def slice_axial(vol: np.ndarray, k: int) -> np.ndarray:
    k = int(np.clip(k, 0, vol.shape[2] - 1))
    return orient_axial(vol[:, :, k])


def slice_coronal(vol: np.ndarray, j: int) -> np.ndarray:
    j = int(np.clip(j, 0, vol.shape[1] - 1))
    return orient_coronal(vol[:, j, :])


def slice_sagittal(vol: np.ndarray, i: int) -> np.ndarray:
    i = int(np.clip(i, 0, vol.shape[0] - 1))
    return orient_sagittal(vol[i, :, :])


def coronal_mip(vol: np.ndarray) -> np.ndarray:
    mip = np.max(vol, axis=1)
    return orient_coronal(mip)


def compose_rgb(
    ct_sl: np.ndarray,
    pet_sl: np.ndarray,
    native_sl: np.ndarray | None,
    mapped_sl: np.ndarray | None,
    *,
    pet_alpha: float = 0.55,
    suv_max: float = 6.0,
    show_native: bool = True,
    show_mapped: bool = True,
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """CT 灰阶 + PET 伪彩融合，再叠本底/映射 mask。返回 (H, W, 3) float32。"""
    ct_n = normalize_ct(ct_sl)
    gray = np.stack([ct_n, ct_n, ct_n], axis=-1)
    pet_rgb = apply_pet_cmap(pet_sl, 0.0, suv_max)
    rgb = gray * (1.0 - pet_alpha) + pet_rgb * pet_alpha
    if show_native and native_sl is not None:
        hit = native_sl > 0
        if np.any(hit):
            rgb[hit] = rgb[hit] * (1.0 - mask_alpha) + NATIVE_RGB * mask_alpha
    if show_mapped and mapped_sl is not None:
        hit = mapped_sl > 0
        if np.any(hit):
            rgb[hit] = rgb[hit] * (1.0 - mask_alpha) + MAPPED_RGB * mask_alpha
    return np.clip(rgb, 0.0, 1.0)
