#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""切片方向、PET 伪彩、窗宽窗位与显示坐标 ↔ 体素。"""

from __future__ import annotations

import numpy as np

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
HIGHLIGHT_RGB = np.array([1.00, 0.85, 0.15], dtype=np.float32)
DIM_RGB = np.array([0.70, 0.20, 0.20], dtype=np.float32)

CT_WINDOW = (-160.0, 240.0)
DEFAULT_WL = 40.0
DEFAULT_WW = 400.0


def pet_lut(n: int = 256) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    t, r, g, b = _PET_STOPS.T
    return np.stack(
        [np.interp(x, t, r), np.interp(x, t, g), np.interp(x, t, b)],
        axis=1,
    ).astype(np.float32)


PET_LUT = pet_lut()


def ct_window_from_wl(wl: float, ww: float) -> tuple[float, float]:
    half = max(ww, 1.0) / 2.0
    return wl - half, wl + half


def apply_pet_cmap(values: np.ndarray, vmin: float = 0.0, vmax: float = 6.0) -> np.ndarray:
    denom = max(vmax - vmin, 1e-6)
    idx = np.clip((values - vmin) / denom, 0.0, 1.0)
    return PET_LUT[(idx * (len(PET_LUT) - 1)).astype(np.int32)]


def normalize_ct(ct: np.ndarray, window: tuple[float, float] = CT_WINDOW) -> np.ndarray:
    lo, hi = window
    return np.clip((ct - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def orient_axial(sl: np.ndarray) -> np.ndarray:
    return np.rot90(sl, 1)


def orient_coronal(sl: np.ndarray) -> np.ndarray:
    return sl.T[::-1, ::-1]


def orient_sagittal(sl: np.ndarray) -> np.ndarray:
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
    return orient_coronal(np.max(vol, axis=1))


def display_to_voxel(
    view: str,
    col: float,
    row: float,
    i: int,
    j: int,
    k: int,
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    """把显示图 (col, row) 映回 RAS 体素 (i, j, k)。"""
    nx, ny, nz = shape
    col_i = int(round(col))
    row_i = int(round(row))
    if view == "axial":
        ii, jj, kk = nx - 1 - col_i, row_i, k
    elif view == "coronal":
        ii, jj, kk = nx - 1 - col_i, j, nz - 1 - row_i
    else:
        ii, jj, kk = i, col_i, nz - 1 - row_i
    return (
        int(np.clip(ii, 0, nx - 1)),
        int(np.clip(jj, 0, ny - 1)),
        int(np.clip(kk, 0, nz - 1)),
    )


def _blend(rgb: np.ndarray, hit: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if not np.any(hit):
        return
    rgb[hit] = rgb[hit] * (1.0 - alpha) + color * alpha


def compose_rgb(
    ct_sl: np.ndarray,
    pet_sl: np.ndarray,
    native_sl: np.ndarray | None,
    mapped_sl: np.ndarray | None,
    *,
    mode: str = "fusion",
    pet_alpha: float = 0.55,
    suv_min: float = 0.0,
    suv_max: float = 6.0,
    ct_window: tuple[float, float] = CT_WINDOW,
    show_native: bool = True,
    show_mapped: bool = True,
    mask_alpha: float = 0.45,
    highlight_label: int = 0,
) -> np.ndarray:
    """mode: ct / pet / fusion。返回 (H, W, 3) float32。"""
    ct_n = normalize_ct(ct_sl, ct_window)
    gray = np.stack([ct_n, ct_n, ct_n], axis=-1)
    pet_rgb = apply_pet_cmap(pet_sl, suv_min, suv_max)
    if mode == "ct":
        rgb = gray
    elif mode == "pet":
        rgb = pet_rgb
    else:
        rgb = gray * (1.0 - pet_alpha) + pet_rgb * pet_alpha

    if show_native and native_sl is not None:
        if highlight_label > 0:
            _blend(rgb, (native_sl > 0) & (native_sl != highlight_label), DIM_RGB, mask_alpha * 0.55)
            _blend(rgb, native_sl == highlight_label, HIGHLIGHT_RGB, mask_alpha + 0.15)
        else:
            _blend(rgb, native_sl > 0, NATIVE_RGB, mask_alpha)
    if show_mapped and mapped_sl is not None:
        _blend(rgb, mapped_sl > 0, MAPPED_RGB, mask_alpha)
    return np.clip(rgb, 0.0, 1.0)
