#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""Lecture figures for the DLBCL PET/CT longitudinal pipeline.

Writes 16:9 PNGs to results/lecture/. Images use the three pilot pairs and the
AutoPET masks that entered lesion matching.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from nibabel.processing import resample_from_to
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "lecture"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
CSV_PATH = ROOT / "results" / "tables" / "lesion_match_pilot.csv"

PAIRS = (
    ("00136597", "20220425", "20220728"),
    ("00598808", "20190930", "20200103"),
    ("00673130", "20220117", "20220616"),
)

INK = "#1B1F24"
MUTED = "#5C6570"
PAPER = "#F7F6F3"
LINE = "#D5D1C8"
BLUE = "#1F4E79"
TEAL = "#1A5F7A"
MATCH = "#1D4E89"
REVIEW = "#A15C07"
RESOLVED = "#2E7D4F"
UNMATCHED = "#6B4C7A"
NEW = "#A33B3B"
INVALID = "#5C6570"
STATUS_COLOR = {
    "matched": MATCH,
    "review": REVIEW,
    "resolved": RESOLVED,
    "unmatched": UNMATCHED,
    "new": NEW,
    "invalid": INVALID,
}
STATUS_ZH = {
    "matched": "匹配",
    "review": "复核",
    "resolved": "缓解",
    "unmatched": "未配上",
    "new": "新发",
    "invalid": "无效",
}
STATION_ZH = {
    "mediastinum": "纵隔",
    "retroperitoneum": "腹膜后",
    "bone": "骨",
    "neck_left": "左颈",
    "neck_right": "右颈",
    "unassigned": "未分区",
    "liver": "肝",
    "spleen": "脾",
}
STATION_COLOR = {
    1: "#8C6A43",  # liver
    2: "#7A5A78",  # spleen
    3: "#8D8D8D",  # brain
    4: "#C4B49A",  # bone
    5: "#2F6F8F",  # neck L
    6: "#C47B4A",  # neck R
    7: "#3E8C8C",
    8: "#D0894C",
    9: "#4C8C6A",
    10: "#C46B6B",
    11: "#1F4E79",  # mediastinum
    12: "#6B4C7A",  # retroperitoneum
    13: "#3D7A5A",
    14: "#A85A4A",
    15: "#3C4F9A",  # inguinal left, distinct from axilla green
    16: "#C4A35A",  # inguinal right, distinct from axilla pink
}
STATION_LEGEND = (
    (4, "骨"),
    (11, "纵隔"),
    (12, "腹膜后"),
    (5, "左颈"),
    (6, "右颈"),
    (9, "左腋"),
    (10, "右腋"),
    (15, "左腹股沟"),
    (16, "右腹股沟"),
    (1, "肝"),
    (2, "脾"),
)


def _font(size: float, weight: str = "regular") -> font_manager.FontProperties:
    path = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        if weight == "regular"
        else "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    )
    return font_manager.FontProperties(fname=path, size=size)


def _load(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def _arr(image: nib.Nifti1Image) -> np.ndarray:
    return np.nan_to_num(np.asanyarray(image.dataobj), nan=0.0).astype(np.float32)


def _pre_ct(patient: str, date: str) -> Path:
    return INTERIM / patient / date / "preprocessed" / "CT" / "s3_WB_Standard.nii.gz"


def _pre_pet(patient: str, date: str) -> Path:
    return INTERIM / patient / date / "preprocessed" / "PET" / "s12_WB_3D_MAC_SUVbw.nii.gz"


def _lesion(patient: str, date: str) -> Path:
    return PROCESSED / patient / date / "masks" / f"{patient}_{date}_lesion.nii.gz"


def _match_dir(patient: str, interim: str) -> Path:
    return PROCESSED / patient / interim / "longitudinal" / "match"


def _coronal(vol: np.ndarray) -> np.ndarray:
    """RAS coronal MIP. Image left is the patient's right; top is superior."""
    projected = np.max(vol, axis=1)
    return projected.T[::-1, ::-1]


def _axial(vol: np.ndarray, index: int) -> np.ndarray:
    """Radiological axial. Image left is the patient's right; top is anterior."""
    return vol[:, :, index].T[::-1, ::-1]


def _coronal_slice(vol: np.ndarray, index: int) -> np.ndarray:
    return vol[:, index, :].T[::-1, ::-1]


def _body_crop(ct: np.ndarray, margin: int = 6) -> tuple[slice, slice, slice]:
    body = ndimage.binary_fill_holes(ct > -450)
    coords = np.argwhere(body)
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin, np.array(ct.shape) - 1)
    return tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))


def _crop(vol: np.ndarray, bounds: tuple[slice, slice, slice]) -> np.ndarray:
    return vol[bounds]


def _window(ct: np.ndarray, level: float, width: float) -> np.ndarray:
    lo = level - width / 2.0
    hi = level + width / 2.0
    return np.clip((ct - lo) / (hi - lo), 0.0, 1.0)


def _style_ax(ax: plt.Axes, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#E4E0D8")
        spine.set_linewidth(0.6)
    ax.set_title(title, fontproperties=_font(13), color=INK, pad=8, loc="left")


def _rl(ax: plt.Axes) -> None:
    ax.text(0.02, 0.04, "R", transform=ax.transAxes, color="white", fontproperties=_font(11, "bold"))
    ax.text(0.94, 0.04, "L", transform=ax.transAxes, color="white", fontproperties=_font(11, "bold"))


def _scale_bar(ax: plt.Axes, voxels: int = 50) -> None:
    """100 mm bar. Images are 2 mm isotropic and displayed in voxel units."""
    ax.plot([8, 8 + voxels], [12, 12], color="white", lw=2.0, solid_capstyle="butt")
    ax.text(8, 18, "10 cm", color="white", fontproperties=_font(9))


def _new_fig() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=PAPER)
    return fig, fig.add_axes([0, 0, 1, 1])


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(path)


def _panel(fig: plt.Figure, rect: tuple[float, float, float, float]) -> plt.Axes:
    ax = fig.add_axes(rect)
    ax.set_facecolor("black")
    return ax


def _dates(a: str, b: str) -> str:
    def one(value: str) -> str:
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return f"{one(a)} → {one(b)}"


def _rows() -> list[dict]:
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, lines: list[str], face: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        linewidth=0.8, edgecolor=face, facecolor="white",
        mutation_aspect=0.6,
    )
    ax.add_patch(patch)
    ax.add_patch(plt.Rectangle((x, y), 0.08, h, color=face, linewidth=0))
    for i, line in enumerate(lines):
        weight = "bold" if i == 0 else "regular"
        color = INK if i == 0 else MUTED
        ax.text(
            x + 0.18, y + h - 0.28 - i * 0.34,
            line, fontproperties=_font(12 if i == 0 else 10.5, weight),
            color=color, va="top",
        )


def _arrow(ax: plt.Axes, x1: float, x2: float, y: float) -> None:
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.0, color=LINE, shrinkA=0, shrinkB=0,
    ))


def figure_roadmap() -> None:
    fig, ax = _new_fig()
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.55, 8.45, "DLBCL PET/CT 纵向分析技术路线", fontproperties=_font(26, "bold"), color=INK)
    ax.text(0.55, 7.95, "单次检查先得到 SUV、病灶和解剖站点，再把基线变到中期网格上做病灶对应", fontproperties=_font(13), color=MUTED)

    ax.text(0.55, 7.45, "单次检查", fontproperties=_font(12, "bold"), color=BLUE)
    tops = [
        (["1  SUV 转换", "DICOM 归档", "生成 SUVbw"], BLUE),
        (["2  统一网格", "CT 重采样 2 mm", "PET 放到同一网格"], TEAL),
        (["3  同机配准", "PET 刚体到 CT", "互信息，不变形"], TEAL),
        (["4  病灶分割", "AutoPET III", "nnU-Net 双通道"], BLUE),
        (["5  解剖站点", "TotalSegmentator", "器官与淋巴结区"], BLUE),
    ]
    width, gap, y, height = 2.85, 0.22, 5.55, 1.7
    x0 = 0.55
    for index, (lines, face) in enumerate(tops):
        x = x0 + index * (width + gap)
        _box(ax, x, y, width, height, lines, face)
        if index < len(tops) - 1:
            _arrow(ax, x + width + 0.02, x + width + gap - 0.02, y + height / 2)

    ax.text(0.55, 5.1, "基线 → 中期", fontproperties=_font(12, "bold"), color=TEAL)
    bottoms = [
        (["6  跨期配准", "基线 CT → 中期 CT", "刚体 + 仿射 + SyN"], TEAL),
        (["7  形变场", "同一变换作用于", "基线 PET 与 AutoPET"], TEAL),
        (["8  病灶对应", "相邻站点可配对", "大灶用表面距离"], BLUE),
        (["9  读出结局", "匹配、复核、缓解", "未配上、新发、无效"], BLUE),
    ]
    width2 = 3.6
    for index, (lines, face) in enumerate(bottoms):
        x = x0 + index * (width2 + gap)
        _box(ax, x, 3.15, width2, height, lines, face)
        if index < len(bottoms) - 1:
            _arrow(ax, x + width2 + 0.02, x + width2 + gap - 0.02, 3.15 + height / 2)

    ax.text(0.55, 2.7, "判定", fontproperties=_font(12, "bold"), color=INK)
    rules = [
        (MATCH, "匹配", "同站或邻站，≤20 mm"),
        (REVIEW, "复核", "同站或邻站，20–40 mm"),
        (RESOLVED, "缓解", "残余 ≤1.5×肝均，MTV=0"),
        (UNMATCHED, "未配上", "没有配对，残余仍高"),
        (NEW, "新发", "中期病灶没有前身"),
        (INVALID, "无效", ">200 ml 或 SUVmax>100"),
    ]
    for index, (color, name, detail) in enumerate(rules):
        x = 0.55 + index * 2.55
        ax.add_patch(plt.Rectangle((x, 1.35), 2.4, 1.15, facecolor="white", edgecolor=LINE, linewidth=0.8))
        ax.add_patch(plt.Rectangle((x, 1.35), 0.08, 1.15, color=color, linewidth=0))
        ax.text(x + 0.22, 2.22, name, fontproperties=_font(13, "bold"), color=color, va="top")
        ax.text(x + 0.22, 1.82, detail, fontproperties=_font(9.5), color=MUTED, va="top")

    ax.text(
        0.55, 0.7,
        "试点：00136597（2022-04-25→07-28）、00598808（2019-09-30→2020-01-03）、00673130（2022-01-17→06-16）。病灶只用 AutoPET，不用 GUI 测试勾画。",
        fontproperties=_font(11), color=MUTED,
    )
    ax.text(
        0.55, 0.32,
        "大灶（≥20 ml）按表面距离，一块基线灶可以对应多块中期小灶。残余 SUV 取病灶核心，并排除脑、肾、膀胱和心肌。",
        fontproperties=_font(11), color=MUTED,
    )
    _save(fig, "01_roadmap.png")


def figure_grid() -> None:
    patient, date = "00598808", "20190930"
    ct_img = _load(_pre_ct(patient, date))
    pet_img = _load(_pre_pet(patient, date))
    ct = _arr(ct_img)
    pet = _arr(pet_img)
    bounds = _body_crop(ct)
    ct_c = _crop(ct, bounds)
    pet_c = _crop(pet, bounds)
    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=PAPER)
    fig.text(0.04, 0.93, "统一到 2 mm 网格", fontproperties=_font(22, "bold"), color=INK)
    fig.text(0.04, 0.885, "00598808  基线 2019-09-30    CT 与 PET 同为 351×351×430，体素 2×2×2 mm", fontproperties=_font(12), color=MUTED)
    ax1 = _panel(fig, (0.04, 0.08, 0.44, 0.76))
    ax2 = _panel(fig, (0.52, 0.08, 0.44, 0.76))
    bone = np.clip(ct_c, 150, 1200)
    ax1.imshow(_coronal(bone), cmap="gray", vmin=150, vmax=1200, aspect="equal")
    ax2.imshow(_coronal(pet_c), cmap="hot", vmin=0, vmax=10, aspect="equal")
    _style_ax(ax1, "CT  骨窗最大密度投影")
    _style_ax(ax2, "PET  冠状最大密度投影  SUV 0–10")
    _rl(ax1)
    _rl(ax2)
    _scale_bar(ax1)
    _scale_bar(ax2)
    fig.text(0.04, 0.03, "这一步只按图像头把 PET 重采样到 CT 网格，还不是解剖配准。SUV 窗固定为 0–10，高于 10 的体素饱和。", fontproperties=_font(11), color=MUTED)
    _save(fig, "02_suv_grid.png")


def _fusion(ct_view: np.ndarray, pet_view: np.ndarray) -> np.ndarray:
    gray = plt.get_cmap("gray")(_window(ct_view, 40, 400))[..., :3]
    hot = plt.get_cmap("hot")(np.clip(pet_view / 10.0, 0, 1))[..., :3]
    alpha = np.clip(pet_view / 6.0, 0, 0.72)[..., None]
    return np.clip(gray * (1 - alpha) + hot * alpha, 0, 1)


def figure_rigid() -> None:
    patient, date = "00598808", "20190930"
    ct_img = _load(PROCESSED / patient / date / "registration" / "ct_iso_reference.nii.gz")
    ct = _arr(ct_img)
    aligned = _arr(_load(PROCESSED / patient / date / "registration" / "pet_iso_aligned.nii.gz"))
    # Heart centroid from the original-resolution organ mask, converted into the 2 mm grid.
    heart_world = np.array([1.4, 26.8, -296.3])
    axial = int(np.clip(round(nib.affines.apply_affine(np.linalg.inv(ct_img.affine), heart_world)[2]), 0, ct.shape[2] - 1))
    bounds = _body_crop(ct)
    i0, i1 = bounds[0].start, bounds[0].stop
    j0, j1 = bounds[1].start, bounds[1].stop
    axial_ct = ct[i0:i1, j0:j1, axial].T[::-1, ::-1]
    axial_pet = aligned[i0:i1, j0:j1, axial].T[::-1, ::-1]

    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=PAPER)
    fig.text(0.04, 0.93, "同机 PET 刚体配准到 CT", fontproperties=_font(22, "bold"), color=INK)
    fig.text(0.04, 0.885, "00598808  基线 2019-09-30    心脏层面。刚体、互信息；本例图像头已经基本对齐，这一步修正很小。", fontproperties=_font(12), color=MUTED)
    panels = (
        (0.04, "CT  窗位 40 / 窗宽 400", _window(axial_ct, 40, 400), "gray", 0, 1),
        (0.28, "刚体配准后的 PET", axial_pet, "hot", 0, 10),
        (0.52, "融合：心肌应落在心脏内", _fusion(axial_ct, axial_pet), None, None, None),
    )
    for x, title, image, cmap, vmin, vmax in panels:
        ax = _panel(fig, (x, 0.18, 0.22, 0.66))
        if cmap is None:
            ax.imshow(image, aspect="equal")
        else:
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        _style_ax(ax, title)
        _rl(ax)
    axm = _panel(fig, (0.76, 0.18, 0.20, 0.66))
    ct_c = _crop(ct, bounds)
    pet_c = _crop(aligned, bounds)
    axm.imshow(_coronal(np.clip(ct_c, 150, 1200)), cmap="gray", vmin=150, vmax=1200, aspect="equal")
    hot = np.clip(_coronal(pet_c) / 10.0, 0, 1)
    color = plt.get_cmap("hot")(hot)
    color[..., 3] = np.clip(hot * 1.4, 0, 0.85)
    axm.imshow(color, aspect="equal")
    _style_ax(axm, "全身融合")
    _rl(axm)
    _scale_bar(axm)
    fig.text(
        0.04, 0.06,
        "病灶分割用的是按图像头放上 CT 网格的 PET。刚体结果用于同机融合和影像组学，不进入 AutoPET。",
        fontproperties=_font(12), color=MUTED,
    )
    _save(fig, "03_pet_ct_rigid.png")


def _mask_on_pet(ax: plt.Axes, pet: np.ndarray, mask: np.ndarray, title: str) -> None:
    ax.imshow(_coronal(pet), cmap="hot", vmin=0, vmax=10, aspect="equal")
    silhouette = _coronal(mask.astype(np.float32))
    if np.any(silhouette > 0):
        color = np.zeros(silhouette.shape + (4,), dtype=np.float32)
        color[silhouette > 0] = (0.45, 0.95, 1.0, 0.38)
        ax.imshow(color, aspect="equal")
        ax.contour(silhouette, levels=[0.5], colors=["white"], linewidths=0.8)
    _style_ax(ax, title)
    _rl(ax)
    _scale_bar(ax)


def figure_autopet() -> None:
    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=PAPER)
    fig.text(0.04, 0.93, "AutoPET 病灶", fontproperties=_font(22, "bold"), color=INK)
    fig.text(0.04, 0.885, "青色覆盖是 AutoPET III 的病灶掩码。冠状最大密度投影，SUV 0–10。", fontproperties=_font(12), color=MUTED)
    specs = (
        ("00598808", "20190930", "基线"),
        ("00598808", "20200103", "中期"),
        ("00673130", "20220117", "基线"),
        ("00673130", "20220616", "中期"),
    )
    for index, (patient, date, role) in enumerate(specs):
        ct = _arr(_load(_pre_ct(patient, date)))
        pet = _arr(_load(_pre_pet(patient, date)))
        mask = _arr(_load(_lesion(patient, date))) > 0
        bounds = _body_crop(ct)
        ax = _panel(fig, (0.04 + index * 0.24, 0.10, 0.22, 0.74))
        ml = float(mask.sum()) * 0.008
        _mask_on_pet(ax, _crop(pet, bounds), _crop(mask, bounds), f"{patient}  {role}  {date[4:6]}-{date[6:]}\n{ml:.1f} ml")
    fig.text(0.04, 0.035, "00598808 由 1294 ml 降到 65 ml；00673130 由 29.4 ml 降到 2.0 ml。体积按 2 mm 体素、0.008 ml/体素计算。", fontproperties=_font(11), color=MUTED)
    _save(fig, "04_autopet.png")


def figure_edited_mask() -> None:
    patient, date = "00136597", "20220425"
    ct = _arr(_load(_pre_ct(patient, date)))
    pet = _arr(_load(_pre_pet(patient, date)))
    auto = _arr(_load(_lesion(patient, date))) > 0
    edited_img = _load(PROCESSED / patient / date / "masks" / f"{patient}_{date}_lesion_edited.nii.gz")
    reference = _load(_pre_ct(patient, date))
    edited = np.rint(np.asanyarray(resample_from_to(edited_img, reference, order=0).dataobj)) > 0
    bounds = _body_crop(ct)
    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=PAPER)
    fig.text(0.04, 0.93, "病灶掩码只用 AutoPET", fontproperties=_font(22, "bold"), color=INK)
    fig.text(0.04, 0.885, "00136597  基线 2022-04-25。右侧是 GUI 测试勾画，不进入纵向匹配。", fontproperties=_font(12), color=MUTED)
    ax1 = _panel(fig, (0.06, 0.12, 0.40, 0.72))
    ax2 = _panel(fig, (0.52, 0.12, 0.40, 0.72))
    _mask_on_pet(ax1, _crop(pet, bounds), _crop(auto, bounds), f"AutoPET    {auto.sum() * 0.008:.1f} ml")
    _mask_on_pet(ax2, _crop(pet, bounds), _crop(edited, bounds), f"GUI 测试勾画    {edited.sum() * 0.008:.0f} ml")
    fig.text(0.06, 0.045, "测试勾画把大片正常组织包了进去。若用它做匹配，会得到两百多个假缓解。", fontproperties=_font(12), color=MUTED)
    _save(fig, "05_ignore_edited_mask.png")


def figure_stations() -> None:
    patient, interim = "00598808", "20200103"
    ct = _arr(_load(_pre_ct(patient, interim)))
    stations = np.rint(_arr(_load(_match_dir(patient, interim) / "stations.nii.gz"))).astype(np.int16)
    body = ndimage.binary_fill_holes(ct > -450)
    stations = np.where(body, stations, 0)
    # Midline in RAS: x = 0.
    affine = _load(_pre_ct(patient, interim)).affine
    midline = int(np.clip(round((0.0 - affine[0, 3]) / affine[0, 0]), 0, ct.shape[0] - 1))
    ct_view = _axial(_window(ct, 40, 500), midline) if False else _coronal_slice(_window(ct, 40, 500), int(ct.shape[1] * 0.48))
    # Use a coronal slice through the vertebral column rather than the geometric midline of the array.
    bone = stations == 4
    bone_i = np.argwhere(bone)
    coronal_j = int(np.median(bone_i[:, 1])) if len(bone_i) else ct.shape[1] // 2
    ct_view = _coronal_slice(_window(ct, 40, 500), coronal_j)
    st_view = _coronal_slice(stations, coronal_j)
    color = np.zeros(st_view.shape + (4,), dtype=np.float32)
    for label, hex_color in STATION_COLOR.items():
        rgb = matplotlib.colors.to_rgba(hex_color)
        color[st_view == label] = (*rgb[:3], 0.62)
    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=PAPER)
    fig.text(0.04, 0.93, "解剖站点", fontproperties=_font(22, "bold"), color=INK)
    fig.text(0.04, 0.885, "00598808  中期 2020-01-03    正中冠状面。器官和骨优先，其余按椎体高度分成淋巴结区。左右不跨侧配对。", fontproperties=_font(12), color=MUTED)
    ax = _panel(fig, (0.18, 0.10, 0.46, 0.74))
    ax.imshow(ct_view, cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax.imshow(color, aspect="equal")
    _style_ax(ax, "站点盖在 CT 上，体外空气不着色")
    _rl(ax)
    for index, (label, name) in enumerate(STATION_LEGEND):
        y = 0.80 - index * 0.055
        fig.patches.append(plt.Rectangle((0.70, y), 0.025, 0.028, transform=fig.transFigure, color=STATION_COLOR[label], linewidth=0))
        fig.text(0.74, y + 0.004, name, fontproperties=_font(12), color=INK)
    fig.text(0.70, 0.12, "腹股沟在髋臼以下，\n颜色不与腋窝共用。\n贴着椎体的软组织灶\n若同时含有非骨站点，\n不把整灶记成骨。", fontproperties=_font(11), color=MUTED)
    _save(fig, "06_stations.png")


def figure_syn() -> None:
    patient, baseline, interim = "00598808", "20190930", "20200103"
    fixed = _arr(_load(_pre_ct(patient, interim)))
    moving = _arr(_load(_pre_ct(patient, baseline)))
    warped = _arr(_load(_match_dir(patient, interim) / "baseline_ct_syn.nii.gz"))
    bounds = _body_crop(fixed)
    images = (
        ("基线 CT", _crop(moving, bounds)),
        ("SyN 之后的基线 CT", _crop(warped, bounds)),
        ("中期 CT", _crop(fixed, bounds)),
    )
    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=PAPER)
    fig.text(0.04, 0.93, "基线 CT 可变形到中期 CT", fontproperties=_font(22, "bold"), color=INK)
    fig.text(0.04, 0.885, "00598808    刚体、仿射，然后 SyN。度量先用互信息，变形用局部互相关。冠状骨性最大密度投影。", fontproperties=_font(12), color=MUTED)
    for index, (title, vol) in enumerate(images):
        ax = _panel(fig, (0.04 + index * 0.24, 0.16, 0.22, 0.68))
        bone = np.clip(vol, 150, 1200)
        ax.imshow(_coronal(bone), cmap="gray", vmin=150, vmax=1200, aspect="equal")
        _style_ax(ax, title)
        _rl(ax)
    warped_c = _crop(warped, bounds)
    fixed_c = _crop(fixed, bounds)
    bone_i = np.argwhere(fixed_c > 300)
    coronal_j = int(np.median(bone_i[:, 1])) if len(bone_i) else warped_c.shape[1] // 2
    a = _coronal_slice(_window(warped_c, 40, 400), coronal_j)
    b = _coronal_slice(_window(fixed_c, 40, 400), coronal_j)
    tile = 18
    yy, xx = np.indices(a.shape)
    board = ((yy // tile + xx // tile) % 2).astype(bool)
    mix = np.where(board, a, b)
    ax = _panel(fig, (0.76, 0.16, 0.20, 0.68))
    ax.imshow(mix, cmap="gray", vmin=0, vmax=1, aspect="equal")
    _style_ax(ax, "棋盘格：形变基线 / 中期")
    _rl(ax)
    fig.text(0.04, 0.05, "变形只在 CT 上估计。PET 和病灶掩码随后用同一个形变场搬到中期网格，不参与配准。", fontproperties=_font(12), color=MUTED)
    _save(fig, "07_syn_registration.png")


def _status_ids(rows: list[dict], patient: str) -> dict[str, dict[str, set[int]]]:
    chosen: dict[str, dict[str, set[int]]] = {"baseline": {}, "interim": {}}
    rank = {"matched": 5, "review": 4, "unmatched": 3, "resolved": 2, "new": 3, "invalid": 1}
    best: dict[tuple[str, int], tuple[int, str]] = {}
    for row in rows:
        if row["patient_id"] != patient:
            continue
        for side, key in (("baseline", "baseline_lesion_id"), ("interim", "interim_lesion_id")):
            if not row[key]:
                continue
            label = int(row[key])
            score = rank[row["match_status"]]
            previous = best.get((side, label))
            if previous is None or score > previous[0]:
                best[(side, label)] = (score, row["match_status"])
    for (side, label), (_score, status) in best.items():
        chosen[side].setdefault(status, set()).add(label)
    return chosen


def _display_xy(labels: np.ndarray, lesion_id: int) -> tuple[float, float] | None:
    coords = np.argwhere(labels == lesion_id)
    if len(coords) == 0:
        return None
    i = float(coords[:, 0].mean())
    k = float(coords[:, 2].mean())
    col = labels.shape[0] - 1 - i
    row = labels.shape[2] - 1 - k
    return col, row


def _ring(ax: plt.Axes, labels: np.ndarray, lesion_id: int, color: str, tag: str) -> None:
    point = _display_xy(labels, lesion_id)
    if point is None:
        return
    ax.scatter(
        [point[0]], [point[1]], s=160, facecolors="none", edgecolors="white", linewidths=2.4, zorder=4,
    )
    ax.scatter(
        [point[0]], [point[1]], s=160, facecolors="none", edgecolors=color, linewidths=1.4, zorder=5,
    )
    ax.text(
        point[0] + 8, point[1] - 8, tag, color="white", fontproperties=_font(11, "bold"),
        zorder=6, bbox={"boxstyle": "round,pad=0.15", "facecolor": color, "edgecolor": "none", "alpha": 0.92},
    )


def _overlay_status(ax: plt.Axes, labels: np.ndarray, groups: dict[str, set[int]], linewidth: float) -> None:
    for status, ids in groups.items():
        if not ids:
            continue
        selected = np.isin(labels, list(ids)).astype(np.float32)
        contour = _coronal(selected)
        if not np.any(contour > 0):
            continue
        ax.contour(contour, levels=[0.5], colors=["white"], linewidths=linewidth + 1.1)
        ax.contour(contour, levels=[0.5], colors=[STATUS_COLOR[status]], linewidths=linewidth)


def _legend(fig: plt.Figure, y: float = 0.045) -> None:
    items = (("matched", "匹配"), ("review", "复核"), ("resolved", "缓解"), ("unmatched", "未配上"), ("new", "新发"), ("invalid", "无效"))
    for index, (status, name) in enumerate(items):
        x = 0.04 + index * 0.15
        fig.patches.append(plt.Rectangle((x, y), 0.018, 0.016, transform=fig.transFigure, color=STATUS_COLOR[status], linewidth=0))
        fig.text(x + 0.025, y, name, fontproperties=_font(11), color=INK)


def figure_correspondence(
    patient: str,
    baseline: str,
    interim: str,
    filename: str,
    headline: str,
    note: str,
    focus: list[tuple[str, int | None, int | None]],
    hide: set[str] | None = None,
    show_hidden_above_ml: float | None = None,
) -> None:
    rows = _rows()
    pet = _arr(_load(_pre_pet(patient, interim)))
    ct = _arr(_load(_pre_ct(patient, interim)))
    warped_pet = _arr(_load(_match_dir(patient, interim) / "baseline_pet_syn.nii.gz"))
    base_labels = np.rint(_arr(_load(_match_dir(patient, interim) / "components_baseline.nii.gz"))).astype(np.int16)
    follow_labels = np.rint(_arr(_load(_match_dir(patient, interim) / "components_interim.nii.gz"))).astype(np.int16)
    bounds = _body_crop(ct)
    groups = _status_ids(rows, patient)
    if hide:
        kept: dict[str, set[int]] = {"baseline": set(), "interim": set()}
        if show_hidden_above_ml is not None:
            for row in rows:
                if row["patient_id"] != patient or row["match_status"] not in hide:
                    continue
                if row["baseline_lesion_id"] and float(row["baseline_mtv_ml"] or 0) >= show_hidden_above_ml:
                    kept["baseline"].add(int(row["baseline_lesion_id"]))
                if row["interim_lesion_id"] and float(row["interim_mtv_ml"] or 0) >= show_hidden_above_ml:
                    kept["interim"].add(int(row["interim_lesion_id"]))
        for side, hidden in groups.items():
            for status in hide:
                ids = hidden.get(status)
                if not ids:
                    continue
                remain = ids & kept[side]
                if remain:
                    hidden[status] = remain
                else:
                    hidden.pop(status, None)
    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=PAPER)
    fig.text(0.04, 0.935, headline, fontproperties=_font(20, "bold"), color=INK)
    fig.text(0.04, 0.895, f"{patient}    {_dates(baseline, interim)}", fontproperties=_font(12), color=MUTED)
    base_crop = _crop(base_labels, bounds)
    follow_crop = _crop(follow_labels, bounds)
    panels = (
        (0.03, "形变到中期网格的基线 PET", _crop(warped_pet, bounds), base_crop, groups["baseline"], "baseline"),
        (0.34, "中期 PET", _crop(pet, bounds), follow_crop, groups["interim"], "interim"),
    )
    for x, title, image, labels, group, side in panels:
        ax = _panel(fig, (x, 0.16, 0.30, 0.70))
        ax.imshow(_coronal(image), cmap="hot", vmin=0, vmax=10, aspect="equal")
        _overlay_status(ax, labels, group, 0.8)
        for number, (_text, base_id, follow_id) in enumerate(focus, start=1):
            status = _status_of(rows, patient, base_id, follow_id)
            color = STATUS_COLOR.get(status, "white")
            if side == "baseline" and base_id is not None:
                _ring(ax, base_crop, int(base_id), color, str(number))
            elif side == "interim" and follow_id is not None:
                _ring(ax, follow_crop, int(follow_id), color, str(number))
            elif side == "interim" and base_id is not None:
                _ring(ax, base_crop, int(base_id), color, str(number))
        _style_ax(ax, title)
        _rl(ax)
        _scale_bar(ax)
    ax = fig.add_axes([0.67, 0.16, 0.30, 0.70])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(PAPER)
    y = 0.96
    for number, (text, label_id, other) in enumerate(focus, start=1):
        ax.text(0.0, y, f"{number}  {text}", fontproperties=_font(12, "bold"), color=INK, va="top")
        y -= 0.08
        detail = _detail(rows, patient, label_id, other)
        ax.text(0.0, y, detail, fontproperties=_font(10.5), color=MUTED, va="top")
        y -= 0.16
    _legend(fig)
    fig.text(0.04, 0.085, note, fontproperties=_font(11), color=MUTED)
    _save(fig, filename)


def _status_of(rows: list[dict], patient: str, baseline_id: int | None, interim_id: int | None) -> str:
    for row in rows:
        if row["patient_id"] != patient:
            continue
        if baseline_id is not None and row["baseline_lesion_id"] != str(baseline_id):
            continue
        if interim_id is not None and row["interim_lesion_id"] != str(interim_id):
            continue
        return row["match_status"]
    return "new"


def _detail(rows: list[dict], patient: str, baseline_id: int | None, interim_id: int | None) -> str:
    for row in rows:
        if row["patient_id"] != patient:
            continue
        if baseline_id is not None and row["baseline_lesion_id"] != str(baseline_id):
            continue
        if interim_id is not None and row["interim_lesion_id"] != str(interim_id):
            continue
        status = STATUS_ZH[row["match_status"]]
        station = STATION_ZH.get(row["station"], row["station"])
        bits = [status, station]
        if row["centroid_distance_mm"]:
            bits.append(f"距离 {float(row['centroid_distance_mm']):.1f} mm")
        if row["baseline_mtv_ml"]:
            bits.append(f"基线 {float(row['baseline_mtv_ml']):.1f} ml")
        if row["interim_mtv_ml"]:
            bits.append(f"中期 {float(row['interim_mtv_ml']):.1f} ml")
        if row["residual_suv_max"]:
            bits.append(f"残余 SUVmax {float(row['residual_suv_max']):.2f}")
        if row["suv_max_liver_ratio"]:
            bits.append(f"肝比 {float(row['suv_max_liver_ratio']):.2f}")
        return "\n".join(bits)
    return ""


def figure_zoom() -> None:
    """Local slabs for the two mediastinal matches and one review on 00598808."""
    patient, interim = "00598808", "20200103"
    pet = _arr(_load(_pre_pet(patient, interim)))
    warped = _arr(_load(_match_dir(patient, interim) / "baseline_pet_syn.nii.gz"))
    base_labels = np.rint(_arr(_load(_match_dir(patient, interim) / "components_baseline.nii.gz"))).astype(np.int16)
    follow_labels = np.rint(_arr(_load(_match_dir(patient, interim) / "components_interim.nii.gz"))).astype(np.int16)
    cases = (
        ("匹配  0.98 mm", 38, 9, MATCH),
        ("匹配  12.4 mm", 19, 7, MATCH),
        ("复核  24.7 mm", 30, 3, REVIEW),
    )
    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=PAPER)
    fig.text(0.04, 0.93, "纵隔病灶的局部对应", fontproperties=_font(22, "bold"), color=INK)
    fig.text(0.04, 0.885, "00598808    只投影病灶周围的冠状厚层，避免全身其他高摄取叠进来。蓝线是形变后的基线灶，红线是中期灶。", fontproperties=_font(12), color=MUTED)
    for index, (title, base_id, follow_id, _color) in enumerate(cases):
        selected = (base_labels == base_id) | (follow_labels == follow_id)
        coords = np.argwhere(selected)
        lo = np.maximum(coords.min(axis=0) - 18, 0)
        hi = np.minimum(coords.max(axis=0) + 18, np.array(pet.shape) - 1)
        slices = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))
        for column, (name, vol) in enumerate((("形变后的基线 PET", warped), ("中期 PET", pet))):
            ax = _panel(fig, (0.04 + column * 0.16 + index * 0.32, 0.14, 0.15, 0.68))
            ax.imshow(_coronal(vol[slices]), cmap="hot", vmin=0, vmax=10, aspect="equal")
            base_view = _coronal((base_labels[slices] == base_id).astype(np.float32))
            follow_view = _coronal((follow_labels[slices] == follow_id).astype(np.float32))
            if np.any(base_view > 0):
                ax.contour(base_view, levels=[0.5], colors=["white"], linewidths=2.4)
                ax.contour(base_view, levels=[0.5], colors=[MATCH], linewidths=1.3)
            if np.any(follow_view > 0):
                ax.contour(follow_view, levels=[0.5], colors=["white"], linewidths=2.4)
                ax.contour(follow_view, levels=[0.5], colors=[NEW], linewidths=1.3)
            _style_ax(ax, f"{title}\n{name}" if column == 0 else name)
            _rl(ax)
    fig.text(0.04, 0.045, "前两对大灶用表面距离而不是质心距离，所以 90 ml 的基线灶可以贴住 3 ml 的中期残留。第三对超过 20 mm，留在复核，不记成匹配。", fontproperties=_font(12), color=MUTED)
    _save(fig, "09_match_zoom_00598808.png")


def figure_landmarks() -> None:
    """SyN 之后，椎体和髋臼质心在同一 2 mm 网格上的距离。"""
    labels = ["C1", "C7", "T1", "T2", "T6", "T12", "L1", "L5", "S1", "骶骨", "左髋", "右髋"]
    series = (
        ("00136597", [2.7, 2.5, 3.5, 2.3, 2.0, 1.2, 2.9, 0.8, 0.7, 2.5, 1.6, 1.1], "20.7 → 2.1", "25.9 → 3.5", BLUE),
        ("00598808", [7.4, 4.1, 4.8, 5.7, 7.5, 2.7, 1.8, 2.1, 1.7, 4.2, 2.7, 1.1], "17.7 → 3.4", "21.3 → 7.5", TEAL),
        ("00673130", [0.9, 0.9, 1.6, 1.3, 6.0, 2.2, 4.2, 2.4, 0.9, 6.7, 5.1, 1.2], "21.0 → 1.9", "25.3 → 6.7", "#8C6A43"),
    )
    fig = plt.figure(figsize=(16, 9), dpi=140, facecolor=PAPER)
    fig.text(0.05, 0.93, "骨性标志配准后还剩多少距离", fontproperties=_font(22, "bold"), color=INK)
    fig.text(0.05, 0.885, "TotalSegmentator 质心。配准后把基线标签变到中期 2 mm 网格，再与重采样到同一网格的中期标签比较。", fontproperties=_font(12), color=MUTED)
    ax = fig.add_axes([0.07, 0.28, 0.88, 0.56])
    ax.set_facecolor("white")
    x = np.arange(len(labels))
    width = 0.24
    for index, (patient, values, _median, _maximum, color) in enumerate(series):
        offset = (index - 1) * width
        ax.bar(x + offset, values, width=width, color=color, label=patient)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=_font(11))
    ax.set_ylabel("配准后质心距离（mm）", fontproperties=_font(12), color=INK)
    ax.set_ylim(0, 9)
    ax.axhline(5, color=LINE, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.legend(frameon=False, prop=_font(11), loc="upper right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=LINE, linewidth=0.6)
    for index, (patient, _values, median_change, maximum_change, color) in enumerate(series):
        x0 = 0.07 + index * 0.30
        fig.text(x0, 0.16, patient, fontproperties=_font(13, "bold"), color=color)
        fig.text(x0, 0.11, f"中位数  {median_change} mm", fontproperties=_font(12), color=INK)
        fig.text(x0, 0.07, f"最大    {maximum_change} mm", fontproperties=_font(12), color=MUTED)
    fig.text(0.05, 0.025, "箭头前是配准前两次扫描的世界坐标距离，箭头后是 SyN 之后。最大残余在 00598808 的 T6（7.5 mm）和 00673130 的骶骨（6.7 mm）。", fontproperties=_font(11), color=MUTED)
    _save(fig, "07b_landmark_distances.png")


def figure_summary() -> None:
    rows = _rows()
    fig, ax = _new_fig()
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.55, 8.35, "三例试点的对应结果", fontproperties=_font(24, "bold"), color=INK)
    ax.text(0.55, 7.85, "同一套规则。SyN 没有重跑，只换了 AutoPET 掩码和对应规则。", fontproperties=_font(13), color=MUTED)
    columns = (
        ("00136597", "2022-04-25 → 07-28", "两块基线灶都没有 20 mm 内的中期灶。\n3.7 ml 腹膜后灶残余接近肝本底，记缓解。\n44.6 ml 灶残余仍高，记未配上。\n中期骨灶距基线 186 mm，记新发。"),
        ("00598808", "2019-09-30 → 2020-01-03", "纵隔两对匹配，距离 1.0 mm 和 12.4 mm。\n另有两对落在 20–40 mm，记复核。\n776 ml 的腹膜后连通域记无效。\n66 个小灶残余接近肝本底。"),
        ("00673130", "2022-01-17 → 06-16", "左颈 29.4 ml 灶残余 SUVmax 2.06，\n与肝均几乎相同，记缓解。\n两块中期纵隔灶最近 178 mm，\n是新发，不是原来的颈部灶。"),
    )
    for index, (patient, when, text) in enumerate(columns):
        x = 0.5 + index * 5.15
        ax.add_patch(plt.Rectangle((x, 1.3), 4.9, 6.1, facecolor="white", edgecolor=LINE, linewidth=0.8))
        ax.text(x + 0.3, 6.95, patient, fontproperties=_font(16, "bold"), color=INK)
        ax.text(x + 0.3, 6.5, when, fontproperties=_font(11), color=MUTED)
        counts = {}
        for row in rows:
            if row["patient_id"] == patient:
                counts[row["match_status"]] = counts.get(row["match_status"], 0) + 1
        y = 5.9
        for status in ("matched", "review", "resolved", "unmatched", "new", "invalid"):
            if status not in counts:
                continue
            ax.add_patch(plt.Rectangle((x + 0.3, y - 0.18), 0.16, 0.28, color=STATUS_COLOR[status], linewidth=0))
            ax.text(x + 0.6, y, f"{STATUS_ZH[status]}  {counts[status]}", fontproperties=_font(13), color=INK, va="center")
            y -= 0.48
        ax.text(x + 0.3, 2.7, text, fontproperties=_font(11), color=MUTED, va="top", linespacing=1.45)
    _save(fig, "11_pilot_summary.png")


def main() -> None:
    figure_roadmap()
    figure_grid()
    figure_rigid()
    figure_autopet()
    figure_edited_mask()
    figure_stations()
    figure_syn()
    figure_correspondence(
        "00136597", "20220425", "20220728", "08_correspondence_00136597.png",
        "没有近距离对应时，不把消失和残留混为一谈",
        "缓解要求残余摄取接近肝本底。44.6 ml 这灶没有配对，但核心残余仍高于 1.5 倍肝均，所以保留为未配上。",
        [
            ("腹膜后  3.7 ml", 1, None),
            ("未分区  44.6 ml", 2, None),
            ("中期骨灶  0.27 ml", None, 1),
        ],
    )
    figure_correspondence(
        "00598808", "20190930", "20200103", "09_correspondence_00598808.png",
        "纵隔两对匹配，另外两对留在复核",
        "绿色是已识别、并判为缓解的较大病灶：左颈 146 ml、右颈 35 ml、左腋 54 ml、右腋 50 ml、左腹股沟 8 ml、右腹股沟 3–4 ml。残余摄取已接近肝本底。更小的缓解灶不逐一描边。",
        [
            ("纵隔匹配  0.98 mm", 38, 9),
            ("纵隔匹配  12.4 mm", 19, 7),
            ("纵隔复核  24.7 mm", 30, 3),
            ("腹膜后无效灶", 17, None),
        ],
        hide={"resolved"},
        show_hidden_above_ml=3.0,
    )
    figure_zoom()
    figure_correspondence(
        "00673130", "20220117", "20220616", "10_correspondence_00673130.png",
        "颈部灶消退，纵隔灶是新的",
        "两块中期灶最近仍有 178 mm，站点和距离都不支持配成同一灶。缓解来自残余 SUV，不是因为“没配上就算消失”。",
        [
            ("左颈基线灶", 1, None),
            ("中期纵隔灶 1", None, 1),
            ("中期纵隔灶 2", None, 2),
        ],
    )
    figure_summary()


if __name__ == "__main__":
    main()
