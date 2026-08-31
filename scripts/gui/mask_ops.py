#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""编号病灶：连通域、画笔、形态学。"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

from display_utils import display_to_voxel


def ensure_labeled(mask: np.ndarray) -> np.ndarray:
    """二值或已编号 mask → uint16 连通域编号（1..N）。"""
    arr = np.asarray(mask)
    if arr.size == 0:
        return arr.astype(np.uint16)
    uniq = np.unique(arr[arr > 0])
    if uniq.size == 0:
        return np.zeros(arr.shape, dtype=np.uint16)
    if uniq.size == 1 and int(uniq[0]) == 1:
        labeled, _ = ndimage.label(arr > 0)
        return labeled.astype(np.uint16)
    return arr.astype(np.uint16)


def relabel_by_volume(mask: np.ndarray) -> np.ndarray:
    """按体素数从大到小重新编号为 1..N。"""
    binary = np.asarray(mask) > 0
    labeled, n = ndimage.label(binary)
    if n == 0:
        return np.zeros(mask.shape, dtype=np.uint16)
    counts = ndimage.sum(binary, labeled, index=np.arange(1, n + 1))
    order = np.argsort(-np.asarray(counts))
    out = np.zeros(mask.shape, dtype=np.uint16)
    for new_id, old_idx in enumerate(order, start=1):
        out[labeled == (old_idx + 1)] = new_id
    return out


def next_label(mask: np.ndarray) -> int:
    mx = int(np.max(mask)) if mask.size else 0
    return max(mx, 0) + 1


def promote_new_islands(current: np.ndarray, previous: np.ndarray) -> None:
    """把本笔新画、且不与旧灶相连的孤立区域改成下一编号。"""
    prev = np.asarray(previous)
    cur = np.asarray(current)
    if cur.shape != prev.shape:
        return
    nxt = next_label(np.maximum(cur, prev))
    for lid in [int(v) for v in np.unique(cur) if v > 0]:
        cc, ncc = ndimage.label(cur == lid)
        if ncc <= 1:
            continue
        for c in range(1, ncc + 1):
            sel = cc == c
            if np.any(prev[sel] == lid):
                continue
            cur[sel] = np.uint16(nxt)
            nxt += 1


def lesion_stats(
    mask: np.ndarray,
    pet: Optional[np.ndarray],
    voxel_ml: float,
) -> list[dict]:
    labeled = np.asarray(mask, dtype=np.uint16)
    ids = [int(v) for v in np.unique(labeled) if v > 0]
    rows: list[dict] = []
    for lid in ids:
        sel = labeled == lid
        n = int(sel.sum())
        suv_max = ""
        if pet is not None and n:
            vals = pet[sel]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                suv_max = round(float(np.max(vals)), 3)
        rows.append(
            {
                "id": lid,
                "n_voxels": n,
                "volume_ml": round(n * voxel_ml, 3),
                "suv_max": suv_max,
            }
        )
    return rows


def paint_disk(
    mask: np.ndarray,
    view: str,
    i: int,
    j: int,
    k: int,
    col: float,
    row: float,
    radius: int,
    label: int,
) -> None:
    """在当前切片平面画圆盘。label=0 为擦除。"""
    ci, cj, ck = display_to_voxel(view, col, row, i, j, k, mask.shape)
    r = max(int(radius), 1)
    nx, ny, nz = mask.shape
    if view == "axial":
        i0, i1 = max(0, ci - r), min(nx, ci + r + 1)
        j0, j1 = max(0, cj - r), min(ny, cj + r + 1)
        ii, jj = np.ogrid[i0:i1, j0:j1]
        disk = (ii - ci) ** 2 + (jj - cj) ** 2 <= r * r
        sl = mask[i0:i1, j0:j1, ck]
        sl[disk] = label
    elif view == "coronal":
        i0, i1 = max(0, ci - r), min(nx, ci + r + 1)
        k0, k1 = max(0, ck - r), min(nz, ck + r + 1)
        ii, kk = np.ogrid[i0:i1, k0:k1]
        disk = (ii - ci) ** 2 + (kk - ck) ** 2 <= r * r
        sl = mask[i0:i1, cj, k0:k1]
        sl[disk] = label
    else:
        j0, j1 = max(0, cj - r), min(ny, cj + r + 1)
        k0, k1 = max(0, ck - r), min(nz, ck + r + 1)
        jj, kk = np.ogrid[j0:j1, k0:k1]
        disk = (jj - cj) ** 2 + (kk - ck) ** 2 <= r * r
        sl = mask[ci, j0:j1, k0:k1]
        sl[disk] = label


def morph_labels(
    mask: np.ndarray,
    op: str,
    radius: int,
    *,
    label: int = 0,
    plane: Optional[str] = None,
    ijk: Optional[tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    op: dilate / erode / open / close
    label=0 表示全部病灶（膨胀出的新体素归该编号，不覆盖其它灶）。
    plane 为 axial/coronal/sagittal 时只改当前层。
    """
    out = np.array(mask, copy=True, dtype=np.uint16)
    r = max(int(radius), 1)
    if plane and ijk is not None:
        _morph_labels_2d(out, op, r, label, plane, ijk)
    else:
        _morph_labels_3d(out, op, r, label)
    return out


def _slice_view(
    mask: np.ndarray, plane: str, ijk: tuple[int, int, int]
) -> np.ndarray:
    i, j, k = ijk
    if plane == "axial":
        return mask[:, :, k]
    if plane == "coronal":
        return mask[:, j, :]
    return mask[i, :, :]


def _morph_labels_2d(
    out: np.ndarray,
    op: str,
    radius: int,
    label: int,
    plane: str,
    ijk: tuple[int, int, int],
) -> None:
    sl = _slice_view(out, plane, ijk)
    ids = [int(v) for v in np.unique(sl) if v > 0]
    targets = [label] if label > 0 else ids
    struct = ndimage.generate_binary_structure(2, 1)
    work = sl.copy()
    for lid in targets:
        binary = sl == lid
        if not np.any(binary):
            continue
        morphed = _morph_2d_slice(binary, op, radius, struct)
        work[work == lid] = 0
        grow = morphed & (work == 0)
        work[grow] = np.uint16(lid)
    sl[:] = work


def _bbox_slices(binary: np.ndarray, pad: int) -> Optional[tuple[slice, ...]]:
    coords = np.nonzero(binary)
    if coords[0].size == 0:
        return None
    box = []
    for axis, idx in enumerate(coords):
        lo = max(int(idx.min()) - pad, 0)
        hi = min(int(idx.max()) + pad + 1, binary.shape[axis])
        box.append(slice(lo, hi))
    return tuple(box)


def _morph_labels_3d(out: np.ndarray, op: str, radius: int, label: int) -> None:
    struct = ndimage.generate_binary_structure(3, 1)
    ids = [int(v) for v in np.unique(out) if v > 0]
    targets = [label] if label > 0 else ids
    for lid in targets:
        binary = out == lid
        if not np.any(binary):
            continue
        box = _bbox_slices(binary, radius)
        if box is None:
            continue
        crop = binary[box]
        morphed = _morph_3d(crop, op, radius, struct)
        region = out[box]
        region[region == lid] = 0
        grow = morphed & (region == 0)
        region[grow] = np.uint16(lid)


def _morph_3d(binary: np.ndarray, op: str, radius: int, struct) -> np.ndarray:
    iters = radius
    if op == "dilate":
        return ndimage.binary_dilation(binary, structure=struct, iterations=iters)
    if op == "erode":
        return ndimage.binary_erosion(binary, structure=struct, iterations=iters)
    if op == "open":
        return ndimage.binary_opening(binary, structure=struct, iterations=iters)
    return ndimage.binary_closing(binary, structure=struct, iterations=iters)


def _morph_2d_slice(sl: np.ndarray, op: str, radius: int, struct) -> np.ndarray:
    if op == "dilate":
        return ndimage.binary_dilation(sl, structure=struct, iterations=radius)
    if op == "erode":
        return ndimage.binary_erosion(sl, structure=struct, iterations=radius)
    if op == "open":
        return ndimage.binary_opening(sl, structure=struct, iterations=radius)
    return ndimage.binary_closing(sl, structure=struct, iterations=radius)
