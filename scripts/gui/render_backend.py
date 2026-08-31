#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""显示合成：CPU（NumPy）或 GPU（CuPy，整本上传后切片）。无 CUDA 时回退 CPU。"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from display_utils import (
    CT_WINDOW,
    DEFAULT_PET_CMAP,
    DIM_RGB,
    HIGHLIGHT_RGB,
    MAPPED_RGB,
    NATIVE_RGB,
    PET_CMAPS,
    compose_rgb,
    slice_axial,
    slice_coronal,
    slice_sagittal,
)

_LOG = logging.getLogger(__name__)
_GPU_OK: Optional[bool] = None
_GPU_LUTS: dict[str, object] = {}
_SLICE_FN = {
    "axial": slice_axial,
    "coronal": slice_coronal,
    "sagittal": slice_sagittal,
}


def gpu_available() -> bool:
    global _GPU_OK
    if _GPU_OK is not None:
        return _GPU_OK
    try:
        import cupy as cp

        _GPU_OK = int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        _GPU_OK = False
    return _GPU_OK


def compose_plane_cpu(vol, plane: str, i: int, j: int, k: int, **kwargs) -> np.ndarray:
    fn = _SLICE_FN[plane]
    sl = {"axial": k, "coronal": j, "sagittal": i}[plane]
    native = vol.native
    mapped = vol.mapped
    return compose_rgb(
        fn(vol.ct, sl),
        fn(vol.pet, sl),
        fn(native, sl) if native is not None else None,
        fn(mapped, sl) if mapped is not None else None,
        **kwargs,
    )


class GpuVolumeCache:
    """把 CT/PET/mask 留在设备上，滚层只改切片下标。"""

    def __init__(self) -> None:
        self._cp = None
        self._ct = None
        self._pet = None
        self._native = None
        self._mapped = None
        self._ct_id = 0
        self._pet_id = 0

    def clear(self) -> None:
        self._ct = self._pet = self._native = self._mapped = None
        self._ct_id = self._pet_id = 0
        self._cp = None

    def bind_volumes(self, vol) -> bool:
        if not gpu_available():
            return False
        import cupy as cp

        self._cp = cp
        if self._ct is None or self._ct_id != id(vol.ct) or self._pet_id != id(vol.pet):
            self._ct = cp.asarray(np.ascontiguousarray(vol.ct, dtype=np.float32))
            self._pet = cp.asarray(np.ascontiguousarray(vol.pet, dtype=np.float32))
            self._ct_id = id(vol.ct)
            self._pet_id = id(vol.pet)
        return True

    def sync_masks(self, vol) -> None:
        if self._cp is None:
            return
        cp = self._cp
        native = vol.native
        mapped = vol.mapped
        self._native = (
            cp.asarray(np.ascontiguousarray(native, dtype=np.uint16))
            if native is not None
            else None
        )
        self._mapped = (
            cp.asarray(np.ascontiguousarray(mapped, dtype=np.uint16))
            if mapped is not None
            else None
        )

    def compose_plane(self, plane: str, i: int, j: int, k: int, **kwargs) -> np.ndarray:
        if self._cp is None or self._ct is None or self._pet is None:
            raise RuntimeError("GPU 体积尚未上传")
        sl_ct = _gpu_slice(self._cp, self._ct, plane, i, j, k)
        sl_pet = _gpu_slice(self._cp, self._pet, plane, i, j, k)
        sl_nat = (
            _gpu_slice(self._cp, self._native, plane, i, j, k)
            if self._native is not None
            else None
        )
        sl_map = (
            _gpu_slice(self._cp, self._mapped, plane, i, j, k)
            if self._mapped is not None
            else None
        )
        rgb = _compose_gpu(self._cp, sl_ct, sl_pet, sl_nat, sl_map, **kwargs)
        return np.asarray(rgb.get(), dtype=np.float32)


def compose_plane(
    vol,
    plane: str,
    i: int,
    j: int,
    k: int,
    *,
    device: str = "cpu",
    gpu_cache: Optional[GpuVolumeCache] = None,
    **kwargs,
) -> np.ndarray:
    if device == "gpu" and gpu_cache is not None:
        try:
            return gpu_cache.compose_plane(plane, i, j, k, **kwargs)
        except Exception:
            _LOG.exception("GPU 合成失败，回退 CPU")
    return compose_plane_cpu(vol, plane, i, j, k, **kwargs)


def _gpu_slice(cp, vol, plane: str, i: int, j: int, k: int):
    nx, ny, nz = vol.shape
    if plane == "axial":
        kk = int(np.clip(k, 0, nz - 1))
        return cp.rot90(vol[:, :, kk], 1)
    if plane == "coronal":
        jj = int(np.clip(j, 0, ny - 1))
        return vol[:, jj, :].T[::-1, ::-1]
    ii = int(np.clip(i, 0, nx - 1))
    return vol[ii, :, :].T[::-1, :]


def _compose_gpu(
    cp,
    ct_sl,
    pet_sl,
    native_sl,
    mapped_sl,
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
    pet_cmap: str = DEFAULT_PET_CMAP,
):
    lo, hi = ct_window
    ct_n = cp.clip((ct_sl - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(cp.float32)
    gray = cp.stack([ct_n, ct_n, ct_n], axis=-1)
    pet_rgb = _gpu_pet_cmap(cp, pet_sl, suv_min, suv_max, pet_cmap)
    if mode == "ct":
        rgb = gray
    elif mode == "pet":
        rgb = pet_rgb
    else:
        rgb = gray * (1.0 - pet_alpha) + pet_rgb * pet_alpha

    if show_native and native_sl is not None:
        if highlight_label > 0:
            _gpu_blend(
                cp,
                rgb,
                (native_sl > 0) & (native_sl != highlight_label),
                DIM_RGB,
                mask_alpha * 0.55,
            )
            _gpu_blend(
                cp, rgb, native_sl == highlight_label, HIGHLIGHT_RGB, mask_alpha + 0.15
            )
        else:
            _gpu_blend(cp, rgb, native_sl > 0, NATIVE_RGB, mask_alpha)
    if show_mapped and mapped_sl is not None:
        _gpu_blend(cp, rgb, mapped_sl > 0, MAPPED_RGB, mask_alpha)
    return cp.clip(rgb, 0.0, 1.0)


def _gpu_pet_cmap(cp, values, vmin: float, vmax: float, name: str):
    denom = max(vmax - vmin, 1e-6)
    n = cp.clip((values.astype(cp.float32) - vmin) / denom, 0.0, 1.0)
    lut_np = PET_CMAPS.get(name)
    if name == "gray" or lut_np is None:
        return cp.stack([n, n, n], axis=-1)
    lut = _GPU_LUTS.get(name)
    if lut is None:
        lut = cp.asarray(lut_np, dtype=cp.float32)
        _GPU_LUTS[name] = lut
    idx = (n * (lut.shape[0] - 1)).astype(cp.int32)
    return lut[idx]


def _gpu_blend(cp, rgb, hit, color: np.ndarray, alpha: float) -> None:
    if not bool(hit.any()):
        return
    col = cp.asarray(color, dtype=rgb.dtype)
    rgb[hit] = rgb[hit] * (1.0 - alpha) + col * alpha
