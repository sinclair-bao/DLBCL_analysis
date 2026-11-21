#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""基线—中期配准的冠状面质量控制图。

固定空间是中期 2 mm 网格。基线 CT、PET 用已经算好的 SyN 变过来，
和中期 CT、PET 并排。体积达到 1 ml 的连通灶标上与 lesion_match.csv 相同的编号。
头在上，患者右侧在图像左侧。

输出：
    results/figures/lesion_station_qc/<患者ID>_<基线>_<中期>.png
    results/figures/lesion_station_qc/index.html
"""

from __future__ import annotations

import argparse
import csv
import html
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import font_manager, patheffects

_SCRIPTS = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _SCRIPTS / "longitudinal", _SCRIPTS / "common"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ants_runner import resolve_ants_bin, run_ants  # noqa: E402

logger = logging.getLogger("qc_lesion_match")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QC = PROJECT_ROOT / "results" / "tables" / "lesion_station_v1" / "pair_qc.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "figures" / "lesion_station_qc"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
TRACK_ML = 1.0
CT_WINDOW = (-160.0, 240.0)
PET_FLOOR = 0.0
PET_CEILING = 12.0

font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()


def render_pair(payload: dict) -> dict:
    """生成一对检查的四联图。已有 PNG 且不覆盖时直接跳过。"""
    out_path = Path(payload["out_path"])
    if out_path.is_file() and not payload.get("overwrite"):
        return {"patient_id": payload["patient_id"], "status": "skipped", "path": str(out_path)}
    try:
        _render(payload, out_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("作图失败 %s", payload["patient_id"])
        return {"patient_id": payload["patient_id"], "status": "error", "path": "", "note": str(exc)}
    return {"patient_id": payload["patient_id"], "status": "ok", "path": str(out_path)}


def _render(payload: dict, out_path: Path) -> None:
    interim_ct = nib.load(payload["interim_ct"])
    interim_pet_img = nib.load(payload["interim_pet"])
    baseline_ct = _on_reference(payload["baseline_ct"], interim_ct)
    interim_pet = _on_reference(payload["interim_pet"], interim_ct)
    baseline_pet_path = _ensure_baseline_pet(payload, interim_ct)
    baseline_pet = _on_reference(str(baseline_pet_path), interim_ct)
    baseline_labels = _labels(payload["baseline_labels"], interim_ct)
    interim_labels = _labels(payload["interim_labels"], interim_ct)

    interim_ct_data = np.asanyarray(interim_ct.dataobj, dtype=np.float32)
    coronal_index = _coronal_index(interim_ct_data)
    pet_max = _shared_pet_max(baseline_pet, interim_pet)
    panels = [
        (_ct_slice(baseline_ct, coronal_index), "gray", CT_WINDOW, baseline_labels, "基线 CT（配准到中期）", "#7FDBFF"),
        (_ct_slice(interim_ct_data, coronal_index), "gray", CT_WINDOW, interim_labels, "中期 CT", "#FFD166"),
        (_pet_mip(baseline_pet), "pet", (PET_FLOOR, pet_max), baseline_labels, "基线 PET（配准到中期）", "#7FDBFF"),
        (_pet_mip(interim_pet), "pet", (PET_FLOOR, pet_max), interim_labels, "中期 PET", "#FFD166"),
    ]
    _ = interim_pet_img
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 16), facecolor="black")
    for ax, (image, cmap_name, window, labels, title, color) in zip(axes.ravel(), panels):
        display = _orient_coronal(image)
        cmap = _pet_cmap() if cmap_name == "pet" else "gray"
        ax.imshow(display, cmap=cmap, vmin=window[0], vmax=window[1], interpolation="bilinear")
        _draw_labels(ax, labels, color)
        ax.set_title(title, color="white", fontsize=13, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_color("#333333")
    dice = payload.get("dice") or "—"
    passed = payload.get("passed") or ""
    fig.suptitle(
        f"{payload['patient_id']}  {payload.get('name', '')}    "
        f"{payload['baseline_date']} → {payload['interim_date']}    "
        f"配准{passed}    Dice {dice}",
        color="white", fontsize=14, y=0.995,
    )
    fig.text(
        0.5, 0.012,
        "冠状面。头在上，患者右侧在图像左侧。两幅 CT 用中期躯干最厚处的同一层，PET 是最大密度投影。"
        "编号是体积 ≥ 1 ml 的连通灶，与病灶匹配表的灶号一致。PET 显示 0–"
        f"{pet_max:.0f} SUV。",
        ha="center", color="#C8C8C8", fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def _ensure_baseline_pet(payload: dict, reference: nib.Nifti1Image) -> Path:
    output = Path(payload["baseline_pet_out"])
    if output.is_file() and output.stat().st_size > 0:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    _registration, apply_transforms, _convert = resolve_ants_bin(payload["ants_bin"])
    command = [
        str(apply_transforms), "-d", "3",
        "-i", payload["baseline_pet"],
        "-r", payload["interim_ct"],
        "-o", str(output),
        "-t", payload["warp"],
        "-t", payload["affine"],
        "-n", "Linear",
    ]
    run_ants(command, f"warp PET {payload['patient_id']}")
    if not output.is_file():
        raise FileNotFoundError(f"未生成配准后的基线 PET: {output}")
    _ = reference
    return output


def _coronal_index(ct: np.ndarray) -> int:
    """躯干最厚的冠状层。配准填出来的 0 不算出身体。"""
    values = np.nan_to_num(np.asarray(ct, dtype=np.float32), nan=-1000.0, posinf=-1000.0, neginf=-1000.0)
    body = (values > -180) & ((values < -1) | (values > 1))
    counts = body.sum(axis=(0, 2))
    if int(counts.max()) == 0:
        return values.shape[1] // 2
    return int(np.argmax(counts))


def _ct_slice(ct: np.ndarray, index: int) -> np.ndarray:
    values = np.nan_to_num(np.asarray(ct, dtype=np.float32), nan=-1000.0, posinf=-1000.0, neginf=-1000.0)
    index = int(np.clip(index, 0, values.shape[1] - 1))
    return values[:, index, :]


def _pet_mip(pet: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(pet, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.max(values, axis=1)


def _shared_pet_max(baseline: np.ndarray, interim: np.ndarray) -> float:
    samples = []
    for volume in (baseline, interim):
        values = np.asarray(volume, dtype=np.float32)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            samples.append(float(np.percentile(values, 99.5)))
    if not samples:
        return PET_CEILING
    return float(np.clip(max(samples), 6.0, PET_CEILING))


def _orient_coronal(slab: np.ndarray) -> np.ndarray:
    """(x, z) → 头在上、患者右在图像左。"""
    return np.asarray(slab).T[::-1, ::-1]


def _draw_labels(ax: plt.Axes, labels: np.ndarray, color: str) -> None:
    voxel_ml = 0.008
    nx, _ny, nz = labels.shape
    for label_id in [int(value) for value in np.unique(labels) if int(value) > 0]:
        coords = np.argwhere(labels == label_id)
        if len(coords) * voxel_ml < TRACK_ML:
            continue
        centroid = coords.mean(axis=0)
        column = nx - 1 - float(centroid[0])
        row = nz - 1 - float(centroid[2])
        ax.text(
            column, row, str(label_id),
            color=color, fontsize=7, ha="center", va="center",
            path_effects=[patheffects.withStroke(linewidth=2.0, foreground="black")],
        )


def _on_reference(path: str, reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(path)
    if image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-2):
        return np.asanyarray(image.dataobj, dtype=np.float32)
    from nibabel.processing import resample_from_to

    warped = resample_from_to(image, (reference.shape, reference.affine), order=1)
    return np.asanyarray(warped.dataobj, dtype=np.float32)


def _labels(path: str, reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(path)
    data = np.rint(np.asanyarray(image.dataobj)).astype(np.int16)
    if image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-2):
        return data
    from nibabel.processing import resample_from_to

    warped = resample_from_to(image, (reference.shape, reference.affine), order=0)
    return np.rint(np.asanyarray(warped.dataobj)).astype(np.int16)


def _pet_cmap():
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


def _wb_file(directory: Path, kind: str) -> Path:
    files = sorted(path for path in directory.glob("*.nii.gz") if not path.name.startswith("._"))
    files = [path for path in files if "BRAIN" not in path.name.upper() and "HEAD" not in path.name.upper()]
    if kind == "ct":
        preferred = [path for path in files if "WB_Standard" in path.name]
    else:
        files = [path for path in files if "SUVbw" in path.name]
        preferred = [path for path in files if "WB" in path.name]
    chosen = preferred or files
    if not chosen:
        raise FileNotFoundError(f"缺少{kind}: {directory}")
    return chosen[0]


def _payloads(rows: list[dict], interim_root: Path, processed_root: Path, out_dir: Path, ants_bin: Path, overwrite: bool) -> list[dict]:
    payloads = []
    for row in rows:
        patient_id = row["患者ID"]
        baseline_date = row["基线日期"]
        interim_date = row["中期日期"]
        match = processed_root / patient_id / interim_date / "longitudinal" / "match"
        station = processed_root / patient_id / interim_date / "longitudinal" / "match_station"
        warp = next(match.glob(f"baseline_{baseline_date}_to_interim_1Warp.nii.gz"), None)
        affine = next(match.glob(f"baseline_{baseline_date}_to_interim_0GenericAffine.mat"), None)
        baseline_ct = match / "baseline_ct_syn.nii.gz"
        interim_ct = _wb_file(interim_root / patient_id / interim_date / "preprocessed" / "CT", "ct")
        interim_pet = _wb_file(interim_root / patient_id / interim_date / "preprocessed" / "PET", "pet")
        baseline_pet = _wb_file(interim_root / patient_id / baseline_date / "preprocessed" / "PET", "pet")
        if warp is None or affine is None or not baseline_ct.is_file():
            logger.error("缺少配准结果 %s %s→%s", patient_id, baseline_date, interim_date)
            continue
        payloads.append({
            "patient_id": patient_id,
            "name": row.get("姓名", ""),
            "baseline_date": baseline_date,
            "interim_date": interim_date,
            "dice": row.get("Dice均值", ""),
            "passed": row.get("配准通过", ""),
            "status": row.get("状态", ""),
            "interim_ct": str(interim_ct),
            "interim_pet": str(interim_pet),
            "baseline_pet": str(baseline_pet),
            "baseline_ct": str(baseline_ct),
            "baseline_pet_out": str(match / "baseline_pet_syn.nii.gz"),
            "baseline_labels": str(station / "baseline_labels_syn.nii.gz"),
            "interim_labels": str(station / "interim_labels.nii.gz"),
            "warp": str(warp),
            "affine": str(affine),
            "ants_bin": str(ants_bin),
            "out_path": str(out_dir / f"{patient_id}_{baseline_date}_{interim_date}.png"),
            "overwrite": overwrite,
        })
    return payloads


def _write_index(out_dir: Path, rows: list[dict], results: dict[str, dict]) -> None:
    cards = []
    for row in rows:
        patient_id = row["患者ID"]
        result = results.get(patient_id, {})
        name = html.escape(row.get("姓名", ""))
        image_name = f"{patient_id}_{row['基线日期']}_{row['中期日期']}.png"
        status = html.escape(row.get("状态", ""))
        dice = html.escape(row.get("Dice均值", "") or "—")
        note = html.escape(result.get("note", "") or row.get("说明", ""))
        color = "#1B7F4E" if row.get("配准通过") == "是" else "#A15C07"
        cards.append(
            f"<figure><a href='{image_name}'><img src='{image_name}' alt='{patient_id}'></a>"
            f"<figcaption><b>{patient_id} {name}</b> {row['基线日期']} → {row['中期日期']}"
            f"<br><span style='color:{color}'>配准{html.escape(row.get('配准通过', ''))}</span>"
            f" Dice {dice} {status}<br>{note}</figcaption></figure>"
        )
    document = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>基线—中期配准质控</title>
<style>
body {{ font-family: "Noto Sans CJK SC", sans-serif; background:#111; color:#eee; margin:24px; }}
figure {{ display:inline-block; width:460px; margin:8px; vertical-align:top; }}
img {{ width:460px; background:#000; }}
figcaption {{ font-size:13px; line-height:1.45; }}
</style></head><body>
<h1>基线—中期配准质控</h1>
<p>左列为配准到中期空间的基线，右列为中期。编号与病灶匹配表一致。</p>
{''.join(cards)}
</body></html>
"""
    (out_dir / "index.html").write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="绘制基线与中期的配准质控图。")
    parser.add_argument("--qc-csv", type=Path, default=DEFAULT_QC)
    parser.add_argument("--interim-root", type=Path, default=PROJECT_ROOT / "data" / "interim")
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--ants-bin", type=Path, default=Path(os.environ.get("ANTS_BIN", "/home/sun/ants-2.6.5/bin")))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with args.qc_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.patient_id:
        rows = [row for row in rows if row["患者ID"] == args.patient_id]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = _payloads(rows, args.interim_root, args.processed_root, args.output_dir, args.ants_bin, args.overwrite)
    logger.info("待绘制 %d 对", len(payloads))
    results: dict[str, dict] = {}
    if args.workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(render_pair, payload) for payload in payloads]
            for finished, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results[result["patient_id"]] = result
                if finished % 10 == 0 or finished == len(futures):
                    logger.info("已完成 %d/%d", finished, len(futures))
    else:
        for index, payload in enumerate(payloads, start=1):
            result = render_pair(payload)
            results[result["patient_id"]] = result
            logger.info("已完成 %d/%d %s %s", index, len(payloads), payload["patient_id"], result["status"])
    _write_index(args.output_dir, rows, results)
    failed = sum(result.get("status") == "error" for result in results.values())
    logger.info("写出 %s，失败 %d", args.output_dir, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
