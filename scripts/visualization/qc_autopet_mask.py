#!/usr/bin/env python3
"""核对每个患者基线 AutoPET 掩码，并保存冠状面质控图。

图像里不超过 100 的最大体素常常落在膀胱、脑或肾，不把这种情况当成漏分割。
漏分割：随访表 SUVmax 不低于 8，图像里也能达到它的 75%，但掩码内最大值低于表内值的 70%。
过分割：掩码体积达到随访表 MTV 的 5 倍；表内 MTV 小于 1 ml 而掩码达到 100 ml 时同样记入。

输出目录：results/autopet_mask_qc/
"""

from __future__ import annotations

import argparse
import html
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import font_manager, patheffects
from nibabel.processing import resample_from_to

logger = logging.getLogger("qc_autopet_mask")

PROJECT = Path("/home/sun/Documents/DLBCL")
OUT = PROJECT / "results" / "autopet_mask_qc"
INTERIM = PROJECT / "data" / "interim"
PROCESSED = PROJECT / "data" / "processed"
STAGE5 = PROJECT / "results" / "tables" / "stages_5_8" / "stage5_metabolic.csv"
SHEET = PROJECT / "data" / "DLBCL随访表_与PET库匹配_最终ID_全量csv解析.csv"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SUV_ARTIFACT = 100.0
MISS_RATIO = 1.3
OVER_RATIO = 5.0


def key_of(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.lstrip("0") or "0"


def find_suv(study_dir: Path) -> Path | None:
    pet = study_dir / "preprocessed" / "PET"
    if not pet.is_dir():
        return None
    files = [path for path in pet.glob("*.nii.gz") if "BRAIN" not in path.name.upper() and "HEAD" not in path.name.upper()]
    suv = [path for path in files if "SUVbw" in path.name]
    whole = [path for path in suv if "WB" in path.name.upper()]
    chosen = whole or suv or files
    return sorted(chosen)[0] if chosen else None


def find_mask(patient_id: str, study_date: str) -> tuple[Path | None, str]:
    mask_dir = PROCESSED / patient_id / study_date / "masks"
    edited = mask_dir / f"{patient_id}_{study_date}_lesion_edited.nii.gz"
    auto = mask_dir / f"{patient_id}_{study_date}_lesion.nii.gz"
    if edited.is_file():
        return edited, "edited"
    if auto.is_file():
        return auto, "autopet"
    return None, "无掩码"


def orient_coronal(slab: np.ndarray) -> np.ndarray:
    return np.asarray(slab).T[::-1, ::-1]


def orient_axial(slab: np.ndarray) -> np.ndarray:
    return np.asarray(slab).T[::-1, ::-1]


def pet_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.00, (0.00, 0.00, 0.00)),
        (0.15, (0.35, 0.00, 0.55)),
        (0.35, (0.00, 0.15, 0.95)),
        (0.55, (0.00, 0.85, 0.75)),
        (0.75, (0.95, 0.95, 0.00)),
        (1.00, (1.00, 1.00, 1.00)),
    ]
    return LinearSegmentedColormap.from_list("pet_qc", colors)


def mark(ax: plt.Axes, x: int, y: int, z: int, kind: str, nx: int, ny: int, nz: int) -> None:
    if kind == "coronal":
        column, row = nx - 1 - x, nz - 1 - z
    else:
        column, row = nx - 1 - x, ny - 1 - y
    ax.scatter([column], [row], s=36, c="#FF4D4D", marker="x", linewidths=1.4)


def classify(row: dict) -> str:
    flags = []
    if row["掩码来源"] == "无掩码" or row["掩码体积ml"] <= 0:
        flags.append("无掩码")
    if row["图像SUVmax"] > SUV_ARTIFACT:
        flags.append("极高体素")
    usable = row["可用图像SUVmax"]
    mask_max = row["掩码SUVmax"]
    sheet_suv = row["表SUVmax"]
    if (
        "无掩码" not in flags
        and np.isfinite(sheet_suv)
        and sheet_suv >= 8
        and usable >= sheet_suv * 0.75
        and mask_max < sheet_suv * 0.70
    ):
        flags.append("漏分割")
    elif np.isfinite(sheet_suv) and sheet_suv >= 8 and usable < sheet_suv * 0.75:
        flags.append("图像达不到表内SUVmax")
    sheet_mtv = row["表MTV"]
    volume = row["掩码体积ml"]
    if (sheet_mtv >= 1 and volume >= sheet_mtv * OVER_RATIO) or (0 < sheet_mtv < 1 and volume >= 100):
        flags.append("过分割")
    if 0 < sheet_mtv < 1 and row["表SUVmax"] >= 10 and volume >= 20:
        flags.append("随访表体积异常")
    if not flags:
        flags.append("通过")
    return "|".join(flags)


def review_one(payload: dict) -> dict:
    patient_id = payload["患者ID"]
    study_date = payload["检查日期"]
    out_path = Path(payload["out_path"])
    row = {
        "患者ID": patient_id,
        "姓名": payload["姓名"],
        "检查日期": study_date,
        "掩码来源": payload["掩码来源"],
        "图像SUVmax": np.nan,
        "可用图像SUVmax": np.nan,
        "掩码SUVmax": np.nan,
        "热点在掩码内": False,
        "掩码体积ml": 0.0,
        "表SUVmax": payload["表SUVmax"],
        "表MTV": payload["表MTV"],
        "体积比": np.nan,
        "判定": "",
        "图": out_path.name,
        "说明": "",
    }
    try:
        suv_path = find_suv(INTERIM / patient_id / study_date)
        mask_path, source = find_mask(patient_id, study_date)
        row["掩码来源"] = source
        if suv_path is None:
            row["判定"] = "缺少PET"
            row["说明"] = "没有体部 SUVbw"
            return row
        image = nib.load(str(suv_path))
        suv = np.asanyarray(image.dataobj, dtype=np.float32)
        suv[~np.isfinite(suv)] = 0
        mask = np.zeros(suv.shape, dtype=bool)
        if mask_path is not None:
            mask_img = nib.load(str(mask_path))
            if mask_img.shape != image.shape or not np.allclose(mask_img.affine, image.affine, atol=0.1):
                mask_img = resample_from_to(mask_img, (image.shape, image.affine), order=0)
            mask = np.asanyarray(mask_img.dataobj) > 0
        voxel_ml = abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000.0
        row["掩码体积ml"] = float(mask.sum() * voxel_ml)
        positive = suv[suv > 0]
        row["图像SUVmax"] = float(suv.max()) if suv.size else np.nan
        usable = np.where(suv <= SUV_ARTIFACT, suv, 0)
        hot = np.unravel_index(int(np.argmax(usable)), usable.shape)
        row["可用图像SUVmax"] = float(usable[hot])
        if mask.any():
            row["掩码SUVmax"] = float(suv[mask].max())
            row["热点在掩码内"] = bool(mask[hot])
        if payload["表MTV"] > 0:
            row["体积比"] = row["掩码体积ml"] / payload["表MTV"]
        row["判定"] = classify(row)
        _draw(suv, mask, hot, positive, row, out_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("核对失败 %s", patient_id)
        row["判定"] = "错误"
        row["说明"] = f"{type(exc).__name__}: {exc}"
    return row


def _draw(suv: np.ndarray, mask: np.ndarray, hot: tuple[int, int, int], positive: np.ndarray, row: dict, out_path: Path) -> None:
    nx, ny, nz = suv.shape
    pet_mip = orient_coronal(np.max(suv, axis=1))
    mask_mip = orient_coronal(np.max(mask, axis=1))
    z = int(np.clip(hot[2], 0, nz - 1))
    axial = orient_axial(suv[:, :, z])
    axial_mask = orient_axial(mask[:, :, z])
    ceiling = float(np.clip(np.percentile(positive, 99.5), 6, 15)) if positive.size else 12.0
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 7.2), facecolor="black")
    panels = [
        (pet_mip, None, "PET 冠状最大密度投影"),
        (pet_mip, mask_mip, "掩码覆盖"),
        (axial, axial_mask, "热点所在横断面"),
    ]
    for ax, (image, overlay, title) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=ceiling, interpolation="bilinear")
        if overlay is not None and np.any(overlay):
            colored = np.zeros((*overlay.shape, 4), dtype=np.float32)
            colored[overlay] = (1.0, 0.25, 0.2, 0.55)
            ax.imshow(colored, interpolation="nearest")
        mark(ax, hot[0], hot[1], hot[2], "axial" if image is axial else "coronal", nx, ny, nz)
        ax.set_title(title, color="white", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")
    sheet_suv = row["表SUVmax"]
    sheet_mtv = row["表MTV"]
    mtv_text = f"{sheet_mtv:.1f}" if sheet_mtv >= 1 else f"{sheet_mtv:.3f}"
    sheet_text = "随访表无记录" if not np.isfinite(sheet_suv) else f"表 SUVmax {sheet_suv:.1f}，表 MTV {mtv_text} ml"
    ratio = row["体积比"]
    ratio_text = "体积比 —" if not np.isfinite(ratio) else f"体积比 {ratio:.2f}"
    inside = "在掩码内" if row["热点在掩码内"] else "不在掩码内"
    fig.suptitle(
        f"{row['患者ID']}  {row['姓名']}  {row['检查日期']}  {row['掩码来源']}    {row['判定']}",
        color="white", fontsize=14,
    )
    fig.text(
        0.5, 0.02,
        f"图像 SUVmax {row['图像SUVmax']:.1f}，可用热点 {row['可用图像SUVmax']:.1f}（{inside}），"
        f"掩码 SUVmax {row['掩码SUVmax']:.1f}，掩码体积 {row['掩码体积ml']:.1f} ml。{sheet_text}。{ratio_text}。"
        f" 红叉是不超过 100 的图像最高体素，可落在膀胱或脑。显示上限 {ceiling:.0f} SUV。头在上，患者右在图像左。",
        ha="center", color="#D0D0D0", fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def load_jobs() -> list[dict]:
    stage = pd.read_csv(STAGE5, dtype=str, encoding="utf-8-sig")
    base = stage[stage["时间点"] == "基线"].drop_duplicates("患者ID", keep="first")
    sheet = pd.read_csv(SHEET, dtype=str, encoding="utf-8-sig")
    sheet["匹配键"] = sheet["PET患者ID"].map(key_of)
    lookup = sheet.drop_duplicates("匹配键").set_index("匹配键")
    jobs = []
    for _, item in base.iterrows():
        patient_id = str(item["患者ID"])
        matched = lookup.loc[key_of(patient_id)] if key_of(patient_id) in lookup.index else None
        jobs.append({
            "患者ID": patient_id,
            "姓名": item["姓名"],
            "检查日期": str(item["检查日期"]),
            "掩码来源": item.get("掩码来源", ""),
            "表SUVmax": float(matched["SUVmax"]) if matched is not None else np.nan,
            "表MTV": float(matched["MTV"]) if matched is not None else np.nan,
            "out_path": str(OUT / f"{patient_id}_{item['检查日期']}.png"),
        })
    return jobs


def write_index(frame: pd.DataFrame) -> None:
    order = {"漏分割": 0, "过分割": 1, "图像达不到表内SUVmax": 2, "极高体素": 3, "无掩码": 4, "随访表体积异常": 5, "通过": 6, "错误": 7}
    frame = frame.copy()
    frame["排序"] = frame["判定"].map(lambda text: min(order.get(part, 9) for part in str(text).split("|")))
    frame = frame.sort_values(["排序", "体积比"], ascending=[True, False])
    cards = []
    for _, row in frame.iterrows():
        verdict = html.escape(str(row["判定"]))
        color = "#F25F5C" if "漏分割" in verdict or "过分割" in verdict else "#8BD17C"
        ratio = "—" if not np.isfinite(row["体积比"]) else f"{row['体积比']:.2f}"
        cards.append(
            f"<figure><a href='{html.escape(row['图'])}'><img src='{html.escape(row['图'])}'></a>"
            f"<figcaption><b>{html.escape(row['患者ID'])} {html.escape(str(row['姓名']))}</b> {row['检查日期']}"
            f"<br><span style='color:{color}'>{verdict}</span>"
            f"<br>掩码 {row['掩码体积ml']:.1f} ml，体积比 {ratio}</figcaption></figure>"
        )
    counts = frame["判定"].value_counts().to_dict()
    document = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>AutoPET 掩码核对</title>
<style>
body {{ font-family: "Noto Sans CJK SC", sans-serif; background:#111; color:#eee; margin:24px; }}
figure {{ display:inline-block; width:420px; margin:8px; vertical-align:top; }}
img {{ width:420px; background:#000; }}
figcaption {{ font-size:13px; line-height:1.45; }}
</style></head><body>
<h1>基线 AutoPET 掩码核对</h1>
<p>红叉是图像中不超过 100 SUV 的最高体素，可以落在膀胱、脑或肾。漏分割只在随访表 SUVmax 能在图像中找到、但掩码内最大值低于表内值 70% 时成立。过分割：掩码体积达到随访表 MTV 的 5 倍。判定计数：{html.escape(str(counts))}</p>
{''.join(cards)}
</body></html>
"""
    (OUT / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs()
    logger.info("待核对基线 %d", len(jobs))
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(review_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done % 25 == 0 or done == len(jobs):
                logger.info("已完成 %d/%d", done, len(jobs))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "mask_review.csv", index=False)
    write_index(frame)
    logger.info("判定\n%s", frame["判定"].value_counts().to_string())


if __name__ == "__main__":
    main()
