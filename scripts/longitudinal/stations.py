#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""由 TotalSegmentator v2 total 标签推导淋巴引流区与结外器官站点。

这是椎体、锁骨、肱骨和髂血管上的几何分区，不是临床淋巴结图谱。
实质器官和骨骼体素优先于按层面划分的软组织站点。
"""

from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np

# TotalSegmentator v2 task "total"
LIVER_ID = 5
SPLEEN_ID = 1
BRAIN_ID = 90
HUMERUS_LEFT, HUMERUS_RIGHT = 69, 70
SCAPULA_LEFT, SCAPULA_RIGHT = 71, 72
CLAVICLE_LEFT, CLAVICLE_RIGHT = 73, 74
FEMUR_LEFT, FEMUR_RIGHT = 75, 76
HIP_LEFT, HIP_RIGHT = 77, 78
ILIAC_ARTERY_LEFT, ILIAC_ARTERY_RIGHT = 65, 66
ILIAC_VEIN_LEFT, ILIAC_VEIN_RIGHT = 67, 68
# sacrum=25 ... C1=50
VERTEBRA_IDS = tuple(range(25, 51))
C1_ID, C7_ID = 50, 44
T1_ID, T2_ID, T6_ID, T12_ID = 43, 42, 38, 32
L5_ID, S1_ID = 27, 26
BONE_IDS = frozenset(VERTEBRA_IDS) | frozenset(range(69, 79)) | frozenset(range(92, 118))

ST_LIVER = 1
ST_SPLEEN = 2
ST_BRAIN = 3
ST_BONE = 4
ST_NECK_LEFT = 5
ST_NECK_RIGHT = 6
ST_SUPRACLAV_LEFT = 7
ST_SUPRACLAV_RIGHT = 8
ST_AXILLA_LEFT = 9
ST_AXILLA_RIGHT = 10
ST_MEDIASTINUM = 11
ST_RETROPERITONEUM = 12
ST_ILIAC_LEFT = 13
ST_ILIAC_RIGHT = 14
ST_INGUINAL_LEFT = 15
ST_INGUINAL_RIGHT = 16

STATION_NAMES = {
    0: "unassigned",
    ST_LIVER: "liver",
    ST_SPLEEN: "spleen",
    ST_BRAIN: "brain",
    ST_BONE: "bone",
    ST_NECK_LEFT: "neck_left",
    ST_NECK_RIGHT: "neck_right",
    ST_SUPRACLAV_LEFT: "supraclavicular_left",
    ST_SUPRACLAV_RIGHT: "supraclavicular_right",
    ST_AXILLA_LEFT: "axilla_left",
    ST_AXILLA_RIGHT: "axilla_right",
    ST_MEDIASTINUM: "mediastinum",
    ST_RETROPERITONEUM: "retroperitoneum",
    ST_ILIAC_LEFT: "iliac_left",
    ST_ILIAC_RIGHT: "iliac_right",
    ST_INGUINAL_LEFT: "inguinal_left",
    ST_INGUINAL_RIGHT: "inguinal_right",
}

MEDIASTINUM_HALF_WIDTH_MM = 70.0
RETRO_HALF_WIDTH_MM = 80.0
ILIAC_HALF_WIDTH_MM = 90.0
INGUINAL_OFFSET_MM = 40.0
AXILLA_HUMERUS_FRACTION = 0.45
AXILLA_OFFSET_FALLBACK_MM = 90.0


@dataclass(frozen=True)
class BodyFrame:
    """头足方向、左右方向，以及各分区用的解剖标志高度（世界坐标 mm）。"""

    midline_x: float
    left_sign: float
    superior_sign: float
    z_c7: float
    z_t1: float
    z_t2: float
    z_t6: float
    z_t12: float
    z_clavicle: float
    z_iliac: float
    z_hip: float
    axilla_offset: float


def station_name(station_id: int) -> str:
    return STATION_NAMES.get(int(station_id), "unassigned")


# 左右不互通。中线站点可以和相邻的单侧站点互配，骨与贴邻的纵隔/腹膜后也互通。
_ADJACENT = {
    ST_NECK_LEFT: {ST_SUPRACLAV_LEFT, ST_BONE},
    ST_NECK_RIGHT: {ST_SUPRACLAV_RIGHT, ST_BONE},
    ST_SUPRACLAV_LEFT: {ST_NECK_LEFT, ST_AXILLA_LEFT, ST_MEDIASTINUM},
    ST_SUPRACLAV_RIGHT: {ST_NECK_RIGHT, ST_AXILLA_RIGHT, ST_MEDIASTINUM},
    ST_AXILLA_LEFT: {ST_SUPRACLAV_LEFT},
    ST_AXILLA_RIGHT: {ST_SUPRACLAV_RIGHT},
    ST_MEDIASTINUM: {ST_SUPRACLAV_LEFT, ST_SUPRACLAV_RIGHT, ST_RETROPERITONEUM, ST_BONE},
    ST_RETROPERITONEUM: {ST_MEDIASTINUM, ST_ILIAC_LEFT, ST_ILIAC_RIGHT, ST_BONE},
    ST_ILIAC_LEFT: {ST_RETROPERITONEUM, ST_INGUINAL_LEFT, ST_BONE},
    ST_ILIAC_RIGHT: {ST_RETROPERITONEUM, ST_INGUINAL_RIGHT, ST_BONE},
    ST_INGUINAL_LEFT: {ST_ILIAC_LEFT},
    ST_INGUINAL_RIGHT: {ST_ILIAC_RIGHT},
    ST_BONE: {
        ST_NECK_LEFT, ST_NECK_RIGHT, ST_MEDIASTINUM, ST_RETROPERITONEUM,
        ST_ILIAC_LEFT, ST_ILIAC_RIGHT,
    },
}


def stations_compatible(left: int, right: int) -> bool:
    """同一站点，或共享边界的站点。左右成对的站点不能跨侧。"""
    if int(left) == int(right):
        return True
    return int(right) in _ADJACENT.get(int(left), ())


def build_station_mask(totalseg: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """把已重采样到目标网格的 TotalSegmentator 标签写成站点图。"""
    labels = np.rint(np.asarray(totalseg)).astype(np.int16)
    frame = build_body_frame(labels, affine)
    stations = np.zeros(labels.shape, dtype=np.uint8)
    stations[np.isin(labels, tuple(BONE_IDS))] = ST_BONE
    stations[labels == SPLEEN_ID] = ST_SPLEEN
    stations[labels == LIVER_ID] = ST_LIVER
    stations[labels == BRAIN_ID] = ST_BRAIN

    for k in range(labels.shape[2]):
        free = stations[:, :, k] == 0
        if not np.any(free):
            continue
        x, z = _slice_xz(k, labels.shape, affine)
        nodal = _assign_nodal(x, z, frame)
        slab = stations[:, :, k]
        slab[free] = nodal[free]
        stations[:, :, k] = slab
    return stations


def build_body_frame(labels: np.ndarray, affine: np.ndarray) -> BodyFrame:
    centroids = {
        int(label_id): _centroid_world(labels == label_id, affine)
        for label_id in (
            *VERTEBRA_IDS,
            HUMERUS_LEFT, HUMERUS_RIGHT,
            CLAVICLE_LEFT, CLAVICLE_RIGHT,
            HIP_LEFT, HIP_RIGHT,
            FEMUR_LEFT, FEMUR_RIGHT,
            ILIAC_ARTERY_LEFT, ILIAC_ARTERY_RIGHT,
            ILIAC_VEIN_LEFT, ILIAC_VEIN_RIGHT,
        )
    }
    centroids = {key: value for key, value in centroids.items() if value is not None}
    vertebra_coords = np.argwhere(np.isin(labels, VERTEBRA_IDS))
    if len(vertebra_coords) == 0:
        raise ValueError("TotalSegmentator 中没有椎体，无法建立解剖站点")
    vertebra_world = nib.affines.apply_affine(affine, vertebra_coords.astype(np.float64))
    midline_x = float(np.median(vertebra_world[:, 0]))

    left = _first_point(centroids, (HUMERUS_LEFT, CLAVICLE_LEFT, HIP_LEFT, FEMUR_LEFT))
    right = _first_point(centroids, (HUMERUS_RIGHT, CLAVICLE_RIGHT, HIP_RIGHT, FEMUR_RIGHT))
    if left is None or right is None:
        raise ValueError("无法判断左右：缺少成对的肱骨、锁骨或髋骨")
    left_sign = float(np.sign(left[0] - right[0])) or 1.0

    top = _first_point(centroids, (C1_ID, 49, 48, 47, C7_ID))
    bottom = _first_point(centroids, (L5_ID, 28, S1_ID, 25))
    if top is None or bottom is None:
        raise ValueError("椎体太少，无法确定头足方向")
    superior_sign = float(np.sign(top[2] - bottom[2])) or 1.0

    z_c7 = _z_of(centroids, (C7_ID, 45, 46, T1_ID), "C7")
    z_t1 = _z_of(centroids, (T1_ID, T2_ID, C7_ID), "T1")
    z_t2 = _z_of(centroids, (T2_ID, T1_ID, 41), "T2")
    z_t6 = _z_of(centroids, (T6_ID, 39, 37), "T6")
    z_t12 = _z_of(centroids, (T12_ID, 33, 31), "T12")
    clavicles = [centroids[item][2] for item in (CLAVICLE_LEFT, CLAVICLE_RIGHT) if item in centroids]
    z_clavicle = float(np.mean(clavicles)) if clavicles else z_t1
    iliacs = [
        centroids[item][2]
        for item in (ILIAC_ARTERY_LEFT, ILIAC_ARTERY_RIGHT, ILIAC_VEIN_LEFT, ILIAC_VEIN_RIGHT)
        if item in centroids
    ]
    z_iliac = float(np.mean(iliacs)) if iliacs else _z_of(centroids, (L5_ID, S1_ID, 25), "髂血管")
    hips = [centroids[item][2] for item in (HIP_LEFT, HIP_RIGHT) if item in centroids]
    z_hip = float(np.mean(hips)) if hips else z_iliac - superior_sign * 40.0

    humerus_dx = [
        abs(float(centroids[item][0]) - midline_x)
        for item in (HUMERUS_LEFT, HUMERUS_RIGHT)
        if item in centroids
    ]
    axilla_offset = (
        AXILLA_HUMERUS_FRACTION * float(np.mean(humerus_dx))
        if humerus_dx else AXILLA_OFFSET_FALLBACK_MM
    )
    return BodyFrame(
        midline_x=midline_x,
        left_sign=left_sign,
        superior_sign=superior_sign,
        z_c7=z_c7,
        z_t1=z_t1,
        z_t2=z_t2,
        z_t6=z_t6,
        z_t12=z_t12,
        z_clavicle=z_clavicle,
        z_iliac=z_iliac,
        z_hip=z_hip,
        axilla_offset=axilla_offset,
    )


def _assign_nodal(x: np.ndarray, z: np.ndarray, frame: BodyFrame) -> np.ndarray:
    """低优先级先写，高优先级后覆盖。返回与 x 同形的站点编号。"""
    left = (x - frame.midline_x) * frame.left_sign >= 0
    dx = np.abs(x - frame.midline_x)
    out = np.zeros(x.shape, dtype=np.uint8)

    retro = _between(z, frame.z_t12, frame.z_iliac, frame.superior_sign) & (dx <= RETRO_HALF_WIDTH_MM)
    out[retro] = ST_RETROPERITONEUM

    iliac = _between(z, frame.z_iliac, frame.z_hip, frame.superior_sign) & (dx <= ILIAC_HALF_WIDTH_MM)
    _paint_sides(out, iliac, left, ST_ILIAC_LEFT, ST_ILIAC_RIGHT)

    inguinal = _inferior_to(z, frame.z_hip, frame.superior_sign) & (dx >= INGUINAL_OFFSET_MM)
    _paint_sides(out, inguinal, left, ST_INGUINAL_LEFT, ST_INGUINAL_RIGHT)

    mediastinum = (
        _between(z, frame.z_t1, frame.z_t12, frame.superior_sign)
        & (dx <= MEDIASTINUM_HALF_WIDTH_MM)
    )
    out[mediastinum] = ST_MEDIASTINUM

    supraclav = (
        _between(z, frame.z_c7, frame.z_t2, frame.superior_sign)
        & (dx <= frame.axilla_offset)
    )
    _paint_sides(out, supraclav, left, ST_SUPRACLAV_LEFT, ST_SUPRACLAV_RIGHT)

    axilla = (
        _between(z, frame.z_clavicle, frame.z_t6, frame.superior_sign)
        & (dx >= frame.axilla_offset)
    )
    _paint_sides(out, axilla, left, ST_AXILLA_LEFT, ST_AXILLA_RIGHT)

    neck = _superior_to(z, frame.z_c7, frame.superior_sign)
    _paint_sides(out, neck, left, ST_NECK_LEFT, ST_NECK_RIGHT)
    return out


def _paint_sides(out: np.ndarray, mask: np.ndarray, left: np.ndarray, left_id: int, right_id: int) -> None:
    out[mask & left] = left_id
    out[mask & ~left] = right_id


def _superior_to(z: np.ndarray, landmark: float, superior_sign: float) -> np.ndarray:
    return (z - landmark) * superior_sign > 0


def _inferior_to(z: np.ndarray, landmark: float, superior_sign: float) -> np.ndarray:
    return (landmark - z) * superior_sign > 0


def _between(z: np.ndarray, z_superior: float, z_inferior: float, superior_sign: float) -> np.ndarray:
    return ((z_superior - z) * superior_sign >= 0) & ((z - z_inferior) * superior_sign >= 0)


def _slice_xz(k: int, shape: tuple[int, ...], affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(shape[0], dtype=np.float64)[:, None]
    j = np.arange(shape[1], dtype=np.float64)[None, :]
    x = affine[0, 0] * i + affine[0, 1] * j + affine[0, 2] * k + affine[0, 3]
    z = affine[2, 0] * i + affine[2, 1] * j + affine[2, 2] * k + affine[2, 3]
    return x, z


def _centroid_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    return nib.affines.apply_affine(affine, coords.mean(axis=0))


def _first_point(centroids: dict[int, np.ndarray], ids: tuple[int, ...]) -> np.ndarray | None:
    for label_id in ids:
        if label_id in centroids:
            return centroids[label_id]
    return None


def _z_of(centroids: dict[int, np.ndarray], ids: tuple[int, ...], name: str) -> float:
    point = _first_point(centroids, ids)
    if point is None:
        raise ValueError(f"缺少{name}附近的椎体，无法划分站点")
    return float(point[2])
