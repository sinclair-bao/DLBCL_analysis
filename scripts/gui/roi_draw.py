"""当前层上的封闭画笔和多边形锚点。坐标是显示坐标 (列, 行)。"""

from __future__ import annotations

from typing import Optional

import numpy as np
from skimage.draw import polygon as draw_polygon
from skimage.measure import find_contours

from display_utils import orient_axial, orient_coronal, orient_sagittal

_ORIENT = {"axial": orient_axial, "coronal": orient_coronal, "sagittal": orient_sagittal}


def display_shape(mask: np.ndarray, plane: str) -> tuple[int, int]:
    nx, ny, nz = mask.shape
    if plane == "axial":
        return ny, nx
    if plane == "coronal":
        return nz, nx
    if plane == "sagittal":
        return nz, ny
    raise ValueError(plane)


def label_on_slice(mask: np.ndarray, plane: str, i: int, j: int, k: int, label: int) -> np.ndarray:
    if plane == "axial":
        raw = mask[:, :, int(k)] == int(label)
    elif plane == "coronal":
        raw = mask[:, int(j), :] == int(label)
    elif plane == "sagittal":
        raw = mask[int(i), :, :] == int(label)
    else:
        raise ValueError(plane)
    return _ORIENT[plane](raw.astype(np.uint8)).astype(bool)


def dominant_label(mask: np.ndarray, plane: str, i: int, j: int, k: int) -> int:
    if plane == "axial":
        sl = mask[:, :, int(k)]
    elif plane == "coronal":
        sl = mask[:, int(j), :]
    elif plane == "sagittal":
        sl = mask[int(i), :, :]
    else:
        raise ValueError(plane)
    vals = sl[sl > 0]
    if vals.size == 0:
        return 0
    uniq, counts = np.unique(vals, return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def display_points_to_voxels(
    plane: str,
    rows: np.ndarray,
    cols: np.ndarray,
    shape: tuple[int, int, int],
    i: int,
    j: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = shape
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if plane == "axial":
        ii = nx - 1 - cols
        jj = ny - 1 - rows
        kk = np.full(ii.shape, int(k), dtype=np.int64)
    elif plane == "coronal":
        ii = nx - 1 - cols
        jj = np.full(rows.shape, int(j), dtype=np.int64)
        kk = nz - 1 - rows
    elif plane == "sagittal":
        ii = np.full(rows.shape, int(i), dtype=np.int64)
        jj = cols
        kk = nz - 1 - rows
    else:
        raise ValueError(plane)
    ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
    return ii[ok], jj[ok], kk[ok]


def rasterize(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """显示坐标多边形（列, 行）填成布尔图。"""
    return rasterize_many([points], shape)


def rasterize_many(paths: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """同一组轮廓一起填充。方向相反的内环保持为孔洞。"""
    height, width = shape
    out = np.zeros((height, width), dtype=bool)
    usable = [np.asarray(path, dtype=np.float64) for path in paths if len(path) >= 3]
    if not usable:
        return out
    areas = [_shoelace(pts) for pts in usable]
    outer_sign = float(np.sign(areas[int(np.argmax(np.abs(areas)))])) or 1.0
    for pts, area in zip(usable, areas):
        rows, cols = draw_polygon(pts[:, 1], pts[:, 0], shape=(height, width))
        if float(np.sign(area) or outer_sign) == outer_sign:
            out[rows, cols] = True
        else:
            out[rows, cols] = False
    return out


def apply_closed_paths(
    mask: np.ndarray,
    plane: str,
    i: int,
    j: int,
    k: int,
    paths: list[np.ndarray],
    *,
    label: int,
    subtract: bool,
) -> int:
    """把封闭图形画进当前层。减去时只清掉当前编号。返回改动的体素数。"""
    if label <= 0 or not paths:
        return 0
    filled = rasterize_many(paths, display_shape(mask, plane))
    rows, cols = np.nonzero(filled)
    if rows.size == 0:
        return 0
    ii, jj, kk = display_points_to_voxels(plane, rows, cols, mask.shape, i, j, k)
    if ii.size == 0:
        return 0
    current = mask[ii, jj, kk]
    if subtract:
        hit = current == int(label)
        if not np.any(hit):
            return 0
        mask[ii[hit], jj[hit], kk[hit]] = 0
        return int(np.count_nonzero(hit))
    hit = current != int(label)
    if not np.any(hit):
        return 0
    mask[ii[hit], jj[hit], kk[hit]] = int(label)
    return int(np.count_nonzero(hit))


def replace_slice_with_paths(
    mask: np.ndarray,
    plane: str,
    i: int,
    j: int,
    k: int,
    paths: list[np.ndarray],
    label: int,
) -> int:
    """用多边形重画当前层上这一个编号，其它编号不动。"""
    if label <= 0:
        return 0
    before = label_on_slice(mask, plane, i, j, k, label).copy()
    rows, cols = np.nonzero(before)
    if rows.size:
        ii, jj, kk = display_points_to_voxels(plane, rows, cols, mask.shape, i, j, k)
        mask[ii, jj, kk] = 0
    changed = apply_closed_paths(mask, plane, i, j, k, paths, label=label, subtract=False)
    after = label_on_slice(mask, plane, i, j, k, label)
    if before.shape == after.shape and np.array_equal(before, after) and changed == 0:
        return 0
    return int(np.count_nonzero(before != after))


def contours_from_label(
    mask: np.ndarray,
    plane: str,
    i: int,
    j: int,
    k: int,
    label: int,
    *,
    epsilon: float = 0.5,
    max_points: int = 200,
) -> list[np.ndarray]:
    """每个连通块取外轮廓，并简化成锚点。坐标为 (列, 行)。"""
    region = label_on_slice(mask, plane, i, j, k, label)
    if not np.any(region):
        return []
    raw = find_contours(region.astype(np.float64), 0.5)
    if not raw:
        return []
    kept: list[np.ndarray] = []
    for contour in raw:
        if len(contour) < 4:
            continue
        # find_contours 给出 (行, 列)，锚点按画面 (列, 行) 存放。
        # 内孔的绕向与外壳相反，两条都保留，填充时孔洞才不会被涂上。
        pts = np.column_stack([contour[:, 1], contour[:, 0]])
        if abs(_shoelace(pts)) < 4.0:
            continue
        simple = simplify_closed(pts, epsilon=epsilon, max_points=max_points)
        if len(simple) >= 3:
            kept.append(simple)
    kept.sort(key=lambda pts: abs(_shoelace(pts)), reverse=True)
    return kept


def _shoelace(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


def simplify_closed(points: np.ndarray, epsilon: float = 0.5, max_points: int = 200) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) >= 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        pts = pts[:-1]
    if len(pts) < 3:
        return pts
    tol = max(float(epsilon), 0.4)
    out = _rdp_closed(pts, tol)
    while len(out) > max_points and tol < 64:
        tol *= 1.35
        out = _rdp_closed(pts, tol)
    return out


def _rdp_closed(points: np.ndarray, epsilon: float) -> np.ndarray:
    ring = np.vstack([points, points[0]])
    simplified = _rdp(ring, epsilon)
    if len(simplified) >= 2 and np.linalg.norm(simplified[0] - simplified[-1]) < 1e-6:
        simplified = simplified[:-1]
    return simplified if len(simplified) >= 3 else points.copy()


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 3:
        return points.copy()
    start = points[0]
    end = points[-1]
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length < 1e-9:
        dist = np.linalg.norm(points - start, axis=1)
    else:
        dist = np.abs(segment[0] * (start[1] - points[:, 1]) - segment[1] * (start[0] - points[:, 0])) / length
    index = int(np.argmax(dist))
    if dist[index] > epsilon:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def nearest_anchor(
    paths: list[np.ndarray],
    x: float,
    y: float,
    radius: float = 12.0,
) -> Optional[tuple[int, int]]:
    best: Optional[tuple[int, int]] = None
    best_d = float(radius) ** 2
    for pi, path in enumerate(paths):
        delta = path - np.array([x, y], dtype=np.float64)
        d2 = np.sum(delta * delta, axis=1)
        idx = int(np.argmin(d2))
        if float(d2[idx]) <= best_d:
            best_d = float(d2[idx])
            best = (pi, idx)
    return best
