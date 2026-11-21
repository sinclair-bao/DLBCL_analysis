#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""站点优先的纵向病灶匹配。

患者级负担直接沿用第五至第八阶段已经算好的 ΔTMTV、ΔTLG 和肝归一化 SUVmax。
本脚本不改那些表，只新增三张表：

    results/tables/lesion_station_v1/pair_qc.csv
    results/tables/lesion_station_v1/station_summary.csv
    results/tables/lesion_station_v1/lesion_match.csv

逐灶规则：
    同一解剖站点，并且形变后有重叠或质心距离不超过 30 mm，才允许配对。
    代价只用距离。落在基线灶膨胀 10 mm 内的中期灶按包含关系挂接，一块基线灶可以对应多块。
    没有候选人时测形变后核心的残余 SUV。不超过 1.5 倍肝脏均值记缓解，否则复核。
    中期灶在所有基线灶膨胀范围之外、且该站点基线没有病灶，才记新发。
    骨骼灶用仿射变换，其余用 SyN。肝、脾、肾 Dice 均值低于 0.50 时，逐灶身份改为复核。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.optimize import linear_sum_assignment

_SCRIPTS = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _SCRIPTS / "common"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline_utils import setup_logging  # noqa: E402

from ants_runner import resolve_ants_bin, run_ants  # noqa: E402
from catalog import DataCatalog, StudyAssets  # noqa: E402
from lesion_match import (  # noqa: E402
    _core_residual,
    _iou,
    _is_large,
    _lesion_distance,
    _liver_mean,
    _near_background,
    _on_reference,
    _point_to_mask_mm,
    _save_image,
    label_lesions,
    measure_components,
)
from stations import ST_BONE, build_station_mask, station_name  # noqa: E402

logger = logging.getLogger("lesion_match_station")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "lesion_station_v1"
DEFAULT_STAGE8 = PROJECT_ROOT / "results" / "tables" / "stages_5_8" / "stage8_longitudinal.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "tables" / "lesion_station_v1"

MATCH_MM = 30.0
CONTAIN_MM = 10.0
TRACK_ML = 1.0
DICE_MIN = 0.50
MIN_ORGAN_VOXELS = 50
DUMMY_COST = 1.0e5
BIG_COST = 1.0e6
QC_FIELDS = [
    "患者ID", "姓名", "基线日期", "中期日期", "状态", "说明",
    "Dice肝", "Dice脾", "Dice肾", "Dice均值", "配准通过",
    "匹配数", "分裂基线数", "缓解数", "新发数", "复核数", "低于阈值数", "无效数",
    "基线TMTV_cm3", "中期TMTV_cm3", "delta_TMTV_pct",
    "基线TLG", "中期TLG", "delta_TLG_pct",
    "基线肝归一化SUVmax", "中期肝归一化SUVmax", "delta_肝归一化SUVmax_pct",
    "协议",
]
STATION_FIELDS = [
    "患者ID", "姓名", "基线日期", "中期日期", "站点",
    "基线灶数", "中期灶数", "基线MTV_ml", "中期MTV_ml", "delta_MTV_pct",
    "基线TLG", "中期TLG", "delta_TLG_pct", "基线SUVmax", "中期SUVmax",
    "站点状态", "协议",
]
LESION_FIELDS = [
    "患者ID", "姓名", "基线日期", "中期日期", "配准通过", "Dice均值", "形变",
    "判定", "几何判定", "生命周期", "基线灶号", "中期灶号", "站点", "形变后站点",
    "质心距离_mm", "IoU", "包含", "代价",
    "基线MTV_ml", "基线SUVmax", "基线SUVmean", "基线SUVpeak", "基线TLG",
    "中期MTV_ml", "中期SUVmax", "中期SUVmean", "中期SUVpeak", "中期TLG",
    "残余SUVmax", "残余SUVmean", "残余SUVpeak", "肝SUVmean", "肝归一化SUVmax",
    "delta_MTV_pct", "delta_TLG_pct", "delta_SUVmax_pct", "说明", "协议",
]

_STATUS_CN = {
    "matched": "匹配",
    "review": "复核",
    "resolved": "缓解",
    "new": "新发",
    "below": "低于阈值",
    "invalid": "无效",
    "unassigned": "复核",
}
_LIFE_CN = {
    "persist": "持续",
    "split": "分裂",
    "resolved": "缓解",
    "new": "新发",
    "review": "复核",
    "below": "低于阈值",
    "invalid": "无效",
}


def drop_pct(baseline: float, follow: float) -> float:
    """(基线 − 随访) / 基线 × 100。正值表示下降，与第八阶段一致。"""
    if not math.isfinite(baseline) or not math.isfinite(follow) or baseline == 0:
        return float("nan")
    return (baseline - follow) / baseline * 100.0


def classify_lesions(
    baseline: list[dict],
    interim: list[dict],
    pair_metrics: dict[tuple[int, int], dict],
    residuals: dict[int, dict],
    liver_suv_mean: float,
    *,
    qc_pass: bool,
) -> list[dict]:
    """按站点、包含关系和残余摄取给每个灶一个身份。qc_pass 为假时身份改为复核。"""
    rows: list[dict] = []
    track_baseline = [item for item in baseline if item["kind"] == "track"]
    track_interim = [item for item in interim if item["kind"] == "track"]
    valid_baseline = [item for item in baseline if item["kind"] != "invalid"]
    by_baseline = {item["id"]: item for item in baseline}
    by_interim = {item["id"]: item for item in interim}

    contained: list[tuple[int, int, dict]] = []
    used_interim: set[int] = set()
    for follow in track_interim:
        choices = []
        for base in track_baseline:
            metrics = pair_metrics.get((base["id"], follow["id"]))
            if metrics is None or base["station"] != follow["station"] or base["station"] <= 0:
                continue
            contain = metrics.get("contain_mm", float("inf"))
            if contain <= CONTAIN_MM:
                distance = metrics.get("distance_mm", float("inf"))
                choices.append((contain, distance, base["id"], metrics))
        if not choices:
            continue
        choices.sort()
        _contain, _distance, baseline_id, metrics = choices[0]
        contained.append((baseline_id, follow["id"], metrics))
        used_interim.add(follow["id"])
    used_baseline = {baseline_id for baseline_id, _interim_id, _metrics in contained}

    remaining_baseline = [item for item in track_baseline if item["id"] not in used_baseline]
    remaining_interim = [item for item in track_interim if item["id"] not in used_interim]
    assigned = _hungarian(remaining_baseline, remaining_interim, pair_metrics)
    used_baseline.update(baseline_id for baseline_id, _interim_id, _metrics in assigned)
    used_interim.update(interim_id for _baseline_id, interim_id, _metrics in assigned)

    split_counts: dict[int, int] = {}
    for baseline_id, _interim_id, _metrics in contained:
        split_counts[baseline_id] = split_counts.get(baseline_id, 0) + 1
    contained_keys = {(baseline_id, interim_id) for baseline_id, interim_id, _metrics in contained}

    for baseline_id, interim_id, metrics in contained + assigned:
        inside = (baseline_id, interim_id) in contained_keys
        lifecycle = "split" if inside and split_counts.get(baseline_id, 0) > 1 else "persist"
        rows.append(_pair_row(
            "matched", lifecycle, by_baseline[baseline_id], by_interim[interim_id], metrics, liver_suv_mean,
            contained=inside,
        ))

    for base in track_baseline:
        if base["id"] in used_baseline:
            continue
        residual = residuals.get(base["id"], {})
        residual_max = float(residual.get("suv_max", float("nan")))
        status = "resolved" if _near_background(residual_max, liver_suv_mean) else "review"
        note = "" if status == "resolved" else "残余摄取高于肝本底，未判缓解"
        if not math.isfinite(residual_max):
            status = "review"
            note = "形变后没有可测的残余区域"
        rows.append(_unpaired_baseline(status, base, residual, liver_suv_mean, note))

    baseline_counts: dict[int, int] = {}
    unmatched_stations: set[int] = set()
    for base in valid_baseline:
        baseline_counts[base["station"]] = baseline_counts.get(base["station"], 0) + 1
    for base in track_baseline:
        if base["id"] not in used_baseline:
            unmatched_stations.add(base["station"])

    for follow in track_interim:
        if follow["id"] in used_interim:
            continue
        outside = _outside_all(follow["id"], valid_baseline, pair_metrics)
        occupied = baseline_counts.get(follow["station"], 0) > 0
        if not outside:
            status, note = "review", "落在基线灶膨胀范围内但未形成配对"
        elif not occupied:
            status, note = "new", ""
        elif follow["station"] in unmatched_stations:
            status, note = "review", "同站点仍有未配上的基线灶"
        else:
            status, note = "review", "同站点基线有病灶，未判为新发"
        rows.append(_unpaired_interim(status, follow, liver_suv_mean, note))

    for source in (baseline, interim):
        for item in source:
            if item["kind"] == "track":
                continue
            side = "baseline" if source is baseline else "interim"
            rows.append(_passthrough(item, side))

    if not qc_pass:
        for row in rows:
            if row["geometry_status"] in ("matched", "resolved", "new"):
                row["note"] = _join_note("配准器官 Dice 未达 0.50，逐灶身份改为复核", row["note"])
                row["status"] = "review"
                row["lifecycle"] = "review"
    return rows


def summarize_stations(baseline: list[dict], interim: list[dict]) -> list[dict]:
    """同一站点的灶数、MTV 和 TLG。无效灶不进入合计。"""
    stations: dict[int, dict] = {}

    def bucket(station: int) -> dict:
        row = stations.get(station)
        if row is None:
            row = {
                "station": station_name(station) if station > 0 else "未分配",
                "baseline_count": 0,
                "interim_count": 0,
                "baseline_mtv": 0.0,
                "interim_mtv": 0.0,
                "baseline_tlg": 0.0,
                "interim_tlg": 0.0,
                "baseline_suv_max": float("nan"),
                "interim_suv_max": float("nan"),
            }
            stations[station] = row
        return row

    for item in baseline:
        if item["kind"] == "invalid":
            continue
        row = bucket(int(item["station"]))
        row["baseline_count"] += 1
        row["baseline_mtv"] += _finite(item.get("mtv_ml"))
        row["baseline_tlg"] += _finite(item.get("tlg"))
        row["baseline_suv_max"] = _nanmax(row["baseline_suv_max"], item.get("suv_max"))
    for item in interim:
        if item["kind"] == "invalid":
            continue
        row = bucket(int(item["station"]))
        row["interim_count"] += 1
        row["interim_mtv"] += _finite(item.get("mtv_ml"))
        row["interim_tlg"] += _finite(item.get("tlg"))
        row["interim_suv_max"] = _nanmax(row["interim_suv_max"], item.get("suv_max"))

    summaries = []
    for row in stations.values():
        if row["baseline_count"] == 0 and row["interim_count"] > 0:
            state = "新发"
        elif row["interim_count"] == 0 and row["baseline_count"] > 0:
            state = "清除"
        else:
            state = "持续"
        row["status"] = state
        row["delta_mtv"] = drop_pct(row["baseline_mtv"], row["interim_mtv"]) if row["baseline_mtv"] > 0 else float("nan")
        row["delta_tlg"] = drop_pct(row["baseline_tlg"], row["interim_tlg"]) if row["baseline_tlg"] > 0 else float("nan")
        summaries.append(row)
    summaries.sort(key=lambda item: item["station"])
    return summaries


def _hungarian(
    baseline: list[dict],
    interim: list[dict],
    pair_metrics: dict[tuple[int, int], dict],
) -> list[tuple[int, int, dict]]:
    if not baseline or not interim:
        return []
    base_index = {item["id"]: index for index, item in enumerate(baseline)}
    follow_index = {item["id"]: index for index, item in enumerate(interim)}
    n_base = len(baseline)
    n_follow = len(interim)
    cost = np.full((n_base + n_follow, n_base + n_follow), BIG_COST, dtype=np.float64)
    info: dict[tuple[int, int], tuple[int, int, dict]] = {}
    for base in baseline:
        for follow in interim:
            if base["station"] != follow["station"] or base["station"] <= 0:
                continue
            metrics = pair_metrics.get((base["id"], follow["id"]))
            if metrics is None or not _pair_allowed(metrics):
                continue
            distance = float(metrics.get("distance_mm", BIG_COST))
            row = base_index[base["id"]]
            column = follow_index[follow["id"]]
            cost[row, column] = distance
            info[(row, column)] = (base["id"], follow["id"], metrics)
    for row in range(n_base):
        cost[row, n_follow + row] = DUMMY_COST
    for column in range(n_follow):
        cost[n_base + column, column] = DUMMY_COST
    cost[n_base:, n_follow:] = 0.0
    chosen = []
    for row, column in zip(*linear_sum_assignment(cost)):
        record = info.get((int(row), int(column)))
        if record is not None:
            chosen.append(record)
    return chosen


def _pair_allowed(metrics: dict) -> bool:
    distance = metrics.get("distance_mm", float("inf"))
    overlap = metrics.get("iou", 0.0) or 0.0
    if overlap > 0:
        return True
    return distance is not None and math.isfinite(float(distance)) and float(distance) <= MATCH_MM


def _outside_all(interim_id: int, baseline: list[dict], pair_metrics: dict[tuple[int, int], dict]) -> bool:
    if not baseline:
        return True
    for base in baseline:
        metrics = pair_metrics.get((base["id"], interim_id))
        contain = float("inf") if metrics is None else metrics.get("contain_mm", float("inf"))
        if contain <= CONTAIN_MM:
            return False
    return True


def _pair_row(status: str, lifecycle: str, base: dict, follow: dict, metrics: dict, liver: float, *, contained: bool) -> dict:
    distance = metrics.get("distance_mm", float("nan"))
    row = _blank_row(status, lifecycle, base, follow)
    row["distance_mm"] = distance
    row["iou"] = metrics.get("iou", float("nan"))
    row["contained"] = contained
    row["cost"] = distance
    row["delta_mtv"] = drop_pct(_num(base.get("mtv_ml")), _num(follow.get("mtv_ml")))
    row["delta_tlg"] = drop_pct(_num(base.get("tlg")), _num(follow.get("tlg")))
    row["delta_suv"] = drop_pct(_num(base.get("suv_max")), _num(follow.get("suv_max")))
    row["liver_suv_mean"] = liver
    follow_max = _num(follow.get("suv_max"))
    row["liver_ratio"] = follow_max / liver if math.isfinite(follow_max) and math.isfinite(liver) and liver > 0 else float("nan")
    return row


def _unpaired_baseline(status: str, base: dict, residual: dict, liver: float, note: str) -> dict:
    follow = {"id": "", "mtv_ml": 0.0 if status == "resolved" else float("nan"), "tlg": 0.0 if status == "resolved" else float("nan")}
    row = _blank_row(status, status, base, follow)
    row["residual_suv_max"] = residual.get("suv_max", float("nan"))
    row["residual_suv_mean"] = residual.get("suv_mean", float("nan"))
    row["residual_suv_peak"] = residual.get("suv_peak", float("nan"))
    row["liver_suv_mean"] = liver
    residual_max = _num(residual.get("suv_max"))
    row["liver_ratio"] = residual_max / liver if math.isfinite(residual_max) and math.isfinite(liver) and liver > 0 else float("nan")
    if status == "resolved":
        row["delta_mtv"] = drop_pct(_num(base.get("mtv_ml")), 0.0)
        row["delta_tlg"] = drop_pct(_num(base.get("tlg")), 0.0)
    row["note"] = note
    return row


def _unpaired_interim(status: str, follow: dict, liver: float, note: str) -> dict:
    row = _blank_row(status, status, {"id": ""}, follow)
    row["liver_suv_mean"] = liver
    follow_max = _num(follow.get("suv_max"))
    row["liver_ratio"] = follow_max / liver if math.isfinite(follow_max) and math.isfinite(liver) and liver > 0 else float("nan")
    row["note"] = note
    return row


def _passthrough(item: dict, side: str) -> dict:
    status = {"below": "below", "invalid": "invalid"}.get(item["kind"], "review")
    note = ""
    if item["kind"] == "unassigned":
        note = "缺少解剖站点，未做逐灶配对"
    base = item if side == "baseline" else {"id": ""}
    follow = item if side == "interim" else {"id": ""}
    row = _blank_row(status, status, base, follow)
    row["note"] = note
    return row


def _blank_row(status: str, lifecycle: str, base: dict, follow: dict) -> dict:
    return {
        "geometry_status": status,
        "status": status,
        "lifecycle": lifecycle,
        "split": lifecycle == "split",
        "baseline_id": base.get("id", ""),
        "interim_id": follow.get("id", ""),
        "station": _station_label(base, follow),
        "warped_station": base.get("warped_station_name", ""),
        "warp": base.get("warp", ""),
        "distance_mm": float("nan"),
        "iou": float("nan"),
        "contained": False,
        "cost": float("nan"),
        "baseline_mtv": base.get("mtv_ml", float("nan")),
        "baseline_suv_max": base.get("suv_max", float("nan")),
        "baseline_suv_mean": base.get("suv_mean", float("nan")),
        "baseline_suv_peak": base.get("suv_peak", float("nan")),
        "baseline_tlg": base.get("tlg", float("nan")),
        "interim_mtv": follow.get("mtv_ml", float("nan")),
        "interim_suv_max": follow.get("suv_max", float("nan")),
        "interim_suv_mean": follow.get("suv_mean", float("nan")),
        "interim_suv_peak": follow.get("suv_peak", float("nan")),
        "interim_tlg": follow.get("tlg", float("nan")),
        "residual_suv_max": float("nan"),
        "residual_suv_mean": float("nan"),
        "residual_suv_peak": float("nan"),
        "liver_suv_mean": float("nan"),
        "liver_ratio": float("nan"),
        "delta_mtv": float("nan"),
        "delta_tlg": float("nan"),
        "delta_suv": float("nan"),
        "note": "",
    }


def process_pair(payload: dict) -> dict:
    """配准一对基线—中期，返回质控、站点和逐灶行。失败时仍返回质控行。"""
    identity = {
        "患者ID": payload["patient_id"],
        "姓名": payload.get("name", ""),
        "基线日期": payload["baseline_date"],
        "中期日期": payload["interim_date"],
    }
    try:
        return _process_pair(payload, identity)
    except Exception as exc:  # noqa: BLE001
        logger.exception("匹配失败 %s %s→%s", payload["patient_id"], payload["baseline_date"], payload["interim_date"])
        qc = _qc_shell(identity, payload.get("stage8", {}))
        qc["状态"] = "error"
        qc["说明"] = str(exc)
        return {"qc": qc, "stations": [], "lesions": []}


def _process_pair(payload: dict, identity: dict) -> dict:
    baseline = _study_from_payload(payload["baseline"], payload["patient_id"], payload["baseline_date"])
    interim = _study_from_payload(payload["interim"], payload["patient_id"], payload["interim_date"])
    ants_bin = Path(payload["ants_bin"])
    out_dir = _ensure_registration(baseline, interim, ants_bin, int(payload.get("threads", 1)))
    station_dir = interim.processed_dir / "longitudinal" / "match_station"
    station_dir.mkdir(parents=True, exist_ok=True)

    reference = nib.load(str(interim.ct_iso))
    interim_pet = _pet_on(interim.pet_iso, reference)
    baseline_pet_image = nib.load(str(baseline.pet_iso))
    baseline_pet = np.asanyarray(baseline_pet_image.dataobj, dtype=np.float32)
    baseline_mask = _on_reference(baseline.lesion_auto, baseline_pet_image)
    interim_mask = _on_reference(interim.lesion_auto, reference)
    baseline_labels = label_lesions(baseline_mask)
    interim_labels = label_lesions(interim_mask)
    _save_image(baseline_labels, baseline_pet_image, station_dir / "baseline_labels.nii.gz")
    _save_image(interim_labels, reference, station_dir / "interim_labels.nii.gz")

    labels_path = station_dir / "baseline_labels.nii.gz"
    syn_path = station_dir / "baseline_labels_syn.nii.gz"
    affine_path = station_dir / "baseline_labels_affine.nii.gz"
    _warp_syn_labels(labels_path, interim.ct_iso, syn_path, out_dir, baseline.study_date, ants_bin)
    _warp_affine_labels(labels_path, interim.ct_iso, affine_path, out_dir, baseline.study_date, ants_bin)
    labels_syn = _labels_on(syn_path, reference)
    labels_affine = _labels_on(affine_path, reference)

    interim_stations = _station_volume(interim, reference)
    baseline_stations = _station_volume(baseline, baseline_pet_image)
    dice = _organ_dice(baseline, interim, reference, out_dir, baseline.study_date, ants_bin, station_dir)
    finite_dice = [value for value in dice.values() if math.isfinite(value)]
    dice_mean = float(np.mean(finite_dice)) if finite_dice else float("nan")
    qc_pass = bool(finite_dice) and dice_mean >= DICE_MIN

    liver = _liver_mean(interim, reference, interim_pet)
    baseline_meta = measure_components(baseline_labels, baseline_pet, baseline_pet_image.affine)
    interim_meta = measure_components(interim_labels, interim_pet, reference.affine)
    baseline_items, baseline_space = _baseline_items(
        baseline_labels, labels_syn, labels_affine, baseline_stations, interim_stations,
        baseline_meta, reference.affine,
    )
    interim_items, interim_space = _interim_items(interim_labels, interim_stations, interim_meta, reference.affine)
    metrics = _pair_metrics(baseline_space, interim_space, labels_syn, labels_affine, interim_labels, reference.affine)
    residuals = _residuals(baseline_space, interim_pet, reference.affine, _organs_on(interim, reference))
    lesion_rows = classify_lesions(baseline_items, interim_items, metrics, residuals, liver, qc_pass=qc_pass)
    station_rows = summarize_stations(baseline_items, interim_items)

    qc = _qc_shell(identity, payload.get("stage8", {}))
    qc["状态"] = "ok" if qc_pass else "warning"
    qc["说明"] = "" if qc_pass else "肝、脾、肾 Dice 均值低于 0.50 或缺器官，逐灶身份改为复核"
    qc["Dice肝"] = _cell(dice.get("肝"))
    qc["Dice脾"] = _cell(dice.get("脾"))
    qc["Dice肾"] = _cell(dice.get("肾"))
    qc["Dice均值"] = _cell(dice_mean)
    qc["配准通过"] = "是" if qc_pass else "否"
    _fill_counts(qc, lesion_rows)
    return {
        "qc": qc,
        "stations": [_station_csv(identity, row) for row in station_rows],
        "lesions": [_lesion_csv(identity, row, qc_pass, dice_mean) for row in lesion_rows],
    }


def _ensure_registration(baseline: StudyAssets, interim: StudyAssets, ants_bin: Path, threads: int) -> Path:
    """刚体、仿射，以及停在 8 mm 的互信息 SyN。30 mm 的配对半径不需要全身 2 mm 互相关。"""
    out_dir = interim.processed_dir / "longitudinal" / "match"
    affine = out_dir / f"baseline_{baseline.study_date}_to_interim_0GenericAffine.mat"
    warp = out_dir / f"baseline_{baseline.study_date}_to_interim_1Warp.nii.gz"
    if affine.is_file() and warp.is_file():
        logger.info("复用已有变换 %s", out_dir)
        return out_dir
    if baseline.ct_iso is None or interim.ct_iso is None:
        raise FileNotFoundError("基线或中期缺少 2 mm CT")
    out_dir.mkdir(parents=True, exist_ok=True)
    from lesion_match import _run_syn_command

    registration, _apply_bin, _convert = resolve_ants_bin(ants_bin)
    fixed = interim.ct_iso
    moving = baseline.ct_iso
    prefix = str(out_dir / f"baseline_{baseline.study_date}_to_interim_")
    warped_ct = out_dir / "baseline_ct_syn.nii.gz"
    metric_mi = f"MI[{fixed},{moving},1,32,Regular,0.25]"
    command = [
        str(registration), "--dimensionality", "3", "--float", "1",
        "--output", f"[{prefix},{warped_ct}]",
        "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "1",
        "--initial-moving-transform", f"[{fixed},{moving},1]",
        "--transform", "Rigid[0.1]", "--metric", metric_mi,
        "--convergence", "[500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2", "--smoothing-sigmas", "3x2x1vox",
        "--transform", "Affine[0.1]", "--metric", metric_mi,
        "--convergence", "[500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2", "--smoothing-sigmas", "3x2x1vox",
        "--transform", "SyN[0.1,3,0]", "--metric", metric_mi,
        "--convergence", "[30x20,1e-6,10]",
        "--shrink-factors", "8x4", "--smoothing-sigmas", "2x1vox",
        "--collapse-output-transforms", "1", "--verbose", "1",
    ]
    logger.info("计算 SyN %s %s→%s", baseline.patient_id, baseline.study_date, interim.study_date)
    try:
        _run_syn_command(command, out_dir, baseline.study_date, fixed, moving, threads)
    except RuntimeError as exc:
        if threads > 1 and "exit=-" in str(exc):
            logger.warning("多线程 antsRegistration 被信号终止，改用单线程重试。")
            _run_syn_command(command, out_dir, baseline.study_date, fixed, moving, 1)
        else:
            raise
    return out_dir


def _totalseg_path(study: StudyAssets) -> Path:
    return study.processed_dir / "organs" / "_totalseg_staging" / "totalseg_wholebody.nii.gz"


def _warp_syn_labels(labels: Path, reference: Path, output: Path, match_dir: Path, baseline_date: str, ants_bin: Path) -> None:
    if output.is_file() and output.stat().st_size > 0:
        return
    _registration, apply_transforms, _convert = resolve_ants_bin(ants_bin)
    warp = match_dir / f"baseline_{baseline_date}_to_interim_1Warp.nii.gz"
    affine = match_dir / f"baseline_{baseline_date}_to_interim_0GenericAffine.mat"
    _apply_transforms(apply_transforms, labels, reference, output, [warp, affine])


def _warp_affine_labels(labels: Path, reference: Path, output: Path, match_dir: Path, baseline_date: str, ants_bin: Path) -> None:
    if output.is_file() and output.stat().st_size > 0:
        return
    _registration, apply_transforms, _convert = resolve_ants_bin(ants_bin)
    affine = match_dir / f"baseline_{baseline_date}_to_interim_0GenericAffine.mat"
    _apply_transforms(apply_transforms, labels, reference, output, [affine])


def _apply_transforms(apply_transforms: Path, moving: Path, reference: Path, output: Path, transforms: list[Path]) -> None:
    def command(interpolation: str) -> list[str]:
        cmd = [str(apply_transforms), "-d", "3", "-i", str(moving), "-r", str(reference), "-o", str(output)]
        for transform in transforms:
            cmd.extend(["-t", str(transform)])
        cmd.extend(["-n", interpolation])
        return cmd

    try:
        run_ants(command("GenericLabel"), f"apply {output.name}")
    except RuntimeError:
        logger.warning("GenericLabel 不可用，改用 NearestNeighbor: %s", output.name)
        run_ants(command("NearestNeighbor"), f"apply {output.name}")


def _organ_dice(
    baseline: StudyAssets,
    interim: StudyAssets,
    reference: nib.Nifti1Image,
    match_dir: Path,
    baseline_date: str,
    ants_bin: Path,
    station_dir: Path,
) -> dict[str, float]:
    """用 TotalSegmentator 的肝、脾、肾做配准质量。器官合成掩码并不是每次检查都有。"""
    empty = {name: float("nan") for name in ("肝", "脾", "肾")}
    moving = _totalseg_path(baseline)
    fixed_path = _totalseg_path(interim)
    if not moving.is_file() or not fixed_path.is_file():
        return empty
    warped_path = station_dir / "baseline_totalseg_syn.nii.gz"
    _warp_syn_labels(moving, interim.ct_iso, warped_path, match_dir, baseline_date, ants_bin)
    warped = _labels_on(warped_path, reference)
    fixed = _labels_on(fixed_path, reference)
    return {
        "肝": _dice(warped == 5, fixed == 5),
        "脾": _dice(warped == 1, fixed == 1),
        "肾": _dice(np.isin(warped, (2, 3)), np.isin(fixed, (2, 3))),
    }


def _dice(left: np.ndarray, right: np.ndarray) -> float:
    left_count = int(np.count_nonzero(left))
    right_count = int(np.count_nonzero(right))
    if left_count < MIN_ORGAN_VOXELS or right_count < MIN_ORGAN_VOXELS:
        return float("nan")
    overlap = int(np.count_nonzero(left & right))
    return 2.0 * overlap / (left_count + right_count)


def _baseline_items(native_labels, labels_syn, labels_affine, native_stations, interim_stations, meta, affine) -> tuple[list[dict], dict[int, dict]]:
    items = []
    space = {}
    label_ids = [int(value) for value in np.unique(native_labels) if int(value) > 0]
    for label_id in label_ids:
        native_coords = np.argwhere(native_labels == label_id)
        native_station = _majority_station(native_stations, native_coords)
        use_affine = _mostly_bone(native_stations, native_coords)
        warped_labels = labels_affine if use_affine else labels_syn
        coords = np.argwhere(warped_labels == label_id)
        warped_station = _majority_station(interim_stations, coords) if len(coords) else 0
        record = meta.get(label_id, {})
        kind = _kind(record, native_station)
        item = {
            "id": label_id,
            "kind": kind,
            "station": int(native_station),
            "warped_station_name": station_name(warped_station) if warped_station else "",
            "warp": "仿射" if use_affine else "SyN",
            "mtv_ml": record.get("mtv_ml", float("nan")),
            "suv_max": record.get("suv_max", float("nan")),
            "suv_mean": record.get("suv_mean", float("nan")),
            "suv_peak": record.get("suv_peak", float("nan")),
            "tlg": record.get("tlg", float("nan")),
        }
        items.append(item)
        if len(coords):
            centroid = nib.affines.apply_affine(affine, coords.mean(axis=0))
            space[label_id] = {
                "coords": coords,
                "centroid": np.asarray(centroid, dtype=np.float64),
                "large": _is_large(record),
                "label_id": label_id,
                "affine_warp": use_affine,
            }
    return items, space


def _interim_items(labels, stations, meta, affine) -> tuple[list[dict], dict[int, dict]]:
    items = []
    space = {}
    for label_id in [int(value) for value in np.unique(labels) if int(value) > 0]:
        coords = np.argwhere(labels == label_id)
        station = _majority_station(stations, coords)
        record = meta.get(label_id, {})
        item = {
            "id": label_id,
            "kind": _kind(record, station),
            "station": int(station),
            "warp": "",
            "mtv_ml": record.get("mtv_ml", float("nan")),
            "suv_max": record.get("suv_max", float("nan")),
            "suv_mean": record.get("suv_mean", float("nan")),
            "suv_peak": record.get("suv_peak", float("nan")),
            "tlg": record.get("tlg", float("nan")),
        }
        items.append(item)
        centroid = nib.affines.apply_affine(affine, coords.mean(axis=0))
        space[label_id] = {
            "coords": coords,
            "centroid": np.asarray(centroid, dtype=np.float64),
            "large": _is_large(record),
        }
    return items, space


def _pair_metrics(baseline_space, interim_space, labels_syn, labels_affine, interim_labels, affine) -> dict[tuple[int, int], dict]:
    metrics: dict[tuple[int, int], dict] = {}
    for baseline_id, base in baseline_space.items():
        warped_labels = labels_affine if base["affine_warp"] else labels_syn
        for interim_id, follow in interim_space.items():
            contain = _point_to_mask_mm(follow["centroid"], base["coords"], affine)
            distance = _lesion_distance(base, follow, base["large"], follow["large"], affine)
            overlap = 0.0
            if _boxes_overlap(base["coords"], follow["coords"]) or distance <= MATCH_MM or contain <= CONTAIN_MM:
                overlap = _iou(warped_labels, baseline_id, interim_labels, interim_id)
            metrics[(baseline_id, interim_id)] = {
                "distance_mm": distance,
                "contain_mm": contain,
                "iou": overlap,
            }
    return metrics


def _residuals(baseline_space, interim_pet, affine, organs) -> dict[int, dict]:
    residuals = {}
    for label_id, base in baseline_space.items():
        suv_max, suv_mean, suv_peak = _core_residual(interim_pet, base["coords"], base["centroid"], affine, organs)
        residuals[label_id] = {"suv_max": suv_max, "suv_mean": suv_mean, "suv_peak": suv_peak}
    return residuals


def _kind(record: dict, station: int) -> str:
    suv = _num(record.get("suv_max"))
    if math.isfinite(suv) and suv > 100.0:
        return "invalid"
    mtv = _num(record.get("mtv_ml"))
    if not math.isfinite(mtv) or mtv < TRACK_ML:
        return "below"
    if int(station) <= 0:
        return "unassigned"
    return "track"


def _inside(coords: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(coords) == 0:
        return coords
    keep = np.all((coords >= 0) & (coords < np.array(shape)), axis=1)
    return coords[keep]


def _majority_station(stations: np.ndarray | None, coords: np.ndarray) -> int:
    """贴着椎体的软组织灶不因骨骼体素占优而被标成 bone。"""
    if stations is None or len(coords) == 0:
        return 0
    coords = _inside(coords, stations.shape)
    if len(coords) == 0:
        return 0
    values = stations[coords[:, 0], coords[:, 1], coords[:, 2]]
    nonzero = values[values > 0]
    if nonzero.size == 0:
        return 0
    soft = nonzero[nonzero != ST_BONE]
    chosen = soft if soft.size else nonzero
    return int(np.bincount(chosen.astype(np.int64)).argmax())


def _mostly_bone(stations: np.ndarray | None, coords: np.ndarray) -> bool:
    """体素多数落在骨骼上时，用仿射而不是 SyN。"""
    if stations is None or len(coords) == 0:
        return False
    coords = _inside(coords, stations.shape)
    if len(coords) == 0:
        return False
    values = stations[coords[:, 0], coords[:, 1], coords[:, 2]]
    return int(np.count_nonzero(values == ST_BONE)) * 2 > len(values)


def _station_volume(study: StudyAssets, reference: nib.Nifti1Image) -> np.ndarray | None:
    path = study.processed_dir / "organs" / "_totalseg_staging" / "totalseg_wholebody.nii.gz"
    if not path.is_file():
        return None
    labels = _labels_on(path, reference)
    try:
        return build_station_mask(labels, reference.affine)
    except ValueError as exc:
        logger.warning("站点无法建立 %s %s: %s", study.patient_id, study.study_date, exc)
        return None


def _organs_on(study: StudyAssets, reference: nib.Nifti1Image) -> np.ndarray | None:
    if study.organs is None or not study.organs.is_file():
        return None
    return _labels_on(study.organs, reference)


def _labels_on(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    return _on_reference(Path(path), reference).astype(np.int16)


def _boxes_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    if len(left) == 0 or len(right) == 0:
        return False
    return bool(np.all(left.min(axis=0) <= right.max(axis=0)) and np.all(right.min(axis=0) <= left.max(axis=0)))


def _pet_on(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-3):
        return np.asanyarray(image.dataobj, dtype=np.float32)
    warped = resample_from_to(image, (reference.shape, reference.affine), order=1)
    return np.asanyarray(warped.dataobj, dtype=np.float32)


def _study_from_payload(info: dict, patient_id: str, study_date: str) -> StudyAssets:
    def optional(key: str) -> Path | None:
        value = info.get(key) or ""
        return Path(value) if value else None

    return StudyAssets(
        patient_id=patient_id,
        study_date=study_date,
        interim_dir=Path(info["interim_dir"]),
        processed_dir=Path(info["processed_dir"]),
        ct_iso=optional("ct_iso"),
        pet_iso=optional("pet_iso"),
        lesion_auto=optional("lesion"),
        organs=optional("organs"),
    )


def _qc_shell(identity: dict, stage8: dict) -> dict:
    row = {field: "" for field in QC_FIELDS}
    row.update(identity)
    for source, target in (
        ("基线TMTV_cm3", "基线TMTV_cm3"),
        ("中期TMTV_cm3", "中期TMTV_cm3"),
        ("delta_TMTV_pct", "delta_TMTV_pct"),
        ("基线TLG", "基线TLG"),
        ("中期TLG", "中期TLG"),
        ("delta_TLG_pct", "delta_TLG_pct"),
        ("基线肝归一化SUVmax", "基线肝归一化SUVmax"),
        ("中期肝归一化SUVmax", "中期肝归一化SUVmax"),
        ("delta_肝归一化SUVmax_pct", "delta_肝归一化SUVmax_pct"),
    ):
        row[target] = stage8.get(source, "")
    row["协议"] = PROTOCOL
    return row


def _fill_counts(qc: dict, rows: list[dict]) -> None:
    qc["匹配数"] = sum(row["geometry_status"] == "matched" and not row["split"] for row in rows)
    qc["分裂基线数"] = len({row["baseline_id"] for row in rows if row["split"]})
    qc["缓解数"] = sum(row["geometry_status"] == "resolved" for row in rows)
    qc["新发数"] = sum(row["geometry_status"] == "new" for row in rows)
    qc["复核数"] = sum(row["status"] == "review" for row in rows)
    qc["低于阈值数"] = sum(row["geometry_status"] == "below" for row in rows)
    qc["无效数"] = sum(row["geometry_status"] == "invalid" for row in rows)


def _station_csv(identity: dict, row: dict) -> dict:
    out = {field: "" for field in STATION_FIELDS}
    out.update(identity)
    out["站点"] = row["station"]
    out["基线灶数"] = row["baseline_count"]
    out["中期灶数"] = row["interim_count"]
    out["基线MTV_ml"] = _cell(row["baseline_mtv"])
    out["中期MTV_ml"] = _cell(row["interim_mtv"])
    out["delta_MTV_pct"] = _cell(row["delta_mtv"], 2)
    out["基线TLG"] = _cell(row["baseline_tlg"])
    out["中期TLG"] = _cell(row["interim_tlg"])
    out["delta_TLG_pct"] = _cell(row["delta_tlg"], 2)
    out["基线SUVmax"] = _cell(row["baseline_suv_max"])
    out["中期SUVmax"] = _cell(row["interim_suv_max"])
    out["站点状态"] = row["status"]
    out["协议"] = PROTOCOL
    return out


def _lesion_csv(identity: dict, row: dict, qc_pass: bool, dice_mean: float) -> dict:
    out = {field: "" for field in LESION_FIELDS}
    out.update(identity)
    out["配准通过"] = "是" if qc_pass else "否"
    out["Dice均值"] = _cell(dice_mean)
    out["形变"] = row["warp"]
    out["判定"] = _STATUS_CN[row["status"]]
    out["几何判定"] = _STATUS_CN[row["geometry_status"]]
    out["生命周期"] = _LIFE_CN[row["lifecycle"]]
    out["基线灶号"] = row["baseline_id"]
    out["中期灶号"] = row["interim_id"]
    out["站点"] = row["station"]
    out["形变后站点"] = row["warped_station"]
    out["质心距离_mm"] = _cell(row["distance_mm"], 2)
    out["IoU"] = _cell(row["iou"], 4)
    out["包含"] = "是" if row["contained"] else "否"
    out["代价"] = _cell(row["cost"], 2)
    for source, target, digits in (
        ("baseline_mtv", "基线MTV_ml", 4),
        ("baseline_suv_max", "基线SUVmax", 4),
        ("baseline_suv_mean", "基线SUVmean", 4),
        ("baseline_suv_peak", "基线SUVpeak", 4),
        ("baseline_tlg", "基线TLG", 4),
        ("interim_mtv", "中期MTV_ml", 4),
        ("interim_suv_max", "中期SUVmax", 4),
        ("interim_suv_mean", "中期SUVmean", 4),
        ("interim_suv_peak", "中期SUVpeak", 4),
        ("interim_tlg", "中期TLG", 4),
        ("residual_suv_max", "残余SUVmax", 4),
        ("residual_suv_mean", "残余SUVmean", 4),
        ("residual_suv_peak", "残余SUVpeak", 4),
        ("liver_suv_mean", "肝SUVmean", 4),
        ("liver_ratio", "肝归一化SUVmax", 4),
        ("delta_mtv", "delta_MTV_pct", 2),
        ("delta_tlg", "delta_TLG_pct", 2),
        ("delta_suv", "delta_SUVmax_pct", 2),
    ):
        out[target] = _cell(row[source], digits)
    out["说明"] = row["note"]
    out["协议"] = PROTOCOL
    return out


def _station_label(base: dict, follow: dict) -> str:
    if "station" in base and base.get("station") not in ("", None):
        station = int(base["station"])
    elif "station" in follow and follow.get("station") not in ("", None):
        station = int(follow["station"])
    else:
        station = 0
    return station_name(station) if station > 0 else "未分配"


def _join_note(*parts: str) -> str:
    return "；".join(part for part in parts if part)


def _finite(value: object) -> float:
    number = _num(value)
    return number if math.isfinite(number) else 0.0


def _nanmax(current: float, value: object) -> float:
    number = _num(value)
    if not math.isfinite(number):
        return current
    if not math.isfinite(current):
        return number
    return max(current, number)


def _num(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _cell(value: object, digits: int = 4) -> str | float:
    number = _num(value)
    if not math.isfinite(number):
        return ""
    return round(number, digits)


def load_pairs(stage8_path: Path, patient_id: str | None) -> list[dict]:
    with stage8_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    pairs = []
    for row in rows:
        if row.get("状态") != "ok" or not row.get("基线日期") or not row.get("中期日期"):
            continue
        if patient_id and row["患者ID"] != patient_id:
            continue
        pairs.append(row)
    return pairs


def build_payload(catalog: DataCatalog, row: dict, ants_bin: Path, threads: int) -> dict | None:
    baseline = catalog.get_study(row["患者ID"], row["基线日期"])
    interim = catalog.get_study(row["患者ID"], row["中期日期"])
    if baseline is None or interim is None:
        return None
    return {
        "patient_id": row["患者ID"],
        "name": row.get("姓名", ""),
        "baseline_date": row["基线日期"],
        "interim_date": row["中期日期"],
        "baseline": _asset_dict(baseline),
        "interim": _asset_dict(interim),
        "stage8": row,
        "ants_bin": str(ants_bin),
        "threads": threads,
    }


def _asset_dict(study: StudyAssets) -> dict:
    lesion = study.working_lesion or study.lesion_auto
    return {
        "interim_dir": str(study.interim_dir),
        "processed_dir": str(study.processed_dir),
        "ct_iso": str(study.ct_iso or ""),
        "pet_iso": str(study.pet_iso or ""),
        "lesion": str(lesion or ""),
        "organs": str(study.organs or ""),
    }


def _write_table(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _key(record: dict) -> tuple[str, str, str]:
    qc = record["qc"]
    return qc["患者ID"], qc["基线日期"], qc["中期日期"]


def run(args: argparse.Namespace) -> int:
    pairs = load_pairs(args.stage8, args.patient_id)
    catalog = DataCatalog(args.interim_root, args.processed_root, patient_filter=args.patient_id)
    ants_bin = args.ants_bin or Path(os.environ.get("ANTS_BIN", "/home/sun/ants-2.6.5/bin"))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "pairs.jsonl"
    done: dict[tuple[str, str, str], dict] = {}
    if args.resume and jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            done[_key(record)] = record
        logger.info("续跑，已有 %d 对", len(done))
        retry = [key for key, record in done.items() if record["qc"].get("状态") == "error"]
        for key in retry:
            del done[key]
        if retry:
            logger.info("重试上次失败的 %d 对", len(retry))
        _publish(output, done)
    elif jsonl_path.exists() and not args.patient_id:
        jsonl_path.unlink()

    payloads = []
    missing = 0
    for row in pairs:
        key = (row["患者ID"], row["基线日期"], row["中期日期"])
        if key in done:
            continue
        payload = build_payload(catalog, row, ants_bin, args.threads)
        needed = ("ct_iso", "pet_iso", "lesion")
        if payload is None or any(not payload[side][field] for side in ("baseline", "interim") for field in needed):
            missing += 1
            qc = _qc_shell(
                {"患者ID": row["患者ID"], "姓名": row.get("姓名", ""), "基线日期": row["基线日期"], "中期日期": row["中期日期"]},
                row,
            )
            qc["状态"] = "error"
            qc["说明"] = "缺少检查目录、CT、PET 或病灶掩码"
            done[key] = {"qc": qc, "stations": [], "lesions": []}
            continue
        payloads.append(payload)
    logger.info("待匹配 %d 对，缺文件 %d", len(payloads), missing)

    def store(record: dict) -> None:
        done[_key(record)] = record
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.workers > 1 and payloads:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_pair, payload) for payload in payloads]
            for finished, future in enumerate(as_completed(futures), start=1):
                store(future.result())
                _publish(output, done)
                if finished % 5 == 0 or finished == len(futures):
                    logger.info("已完成 %d/%d", finished, len(futures))
    else:
        for index, payload in enumerate(payloads, start=1):
            store(process_pair(payload))
            _publish(output, done)
            logger.info("已完成 %d/%d %s", index, len(payloads), payload["patient_id"])
    _publish(output, done)
    counts: dict[str, int] = {}
    for record in done.values():
        status = record["qc"]["状态"]
        counts[status] = counts.get(status, 0) + 1
    logger.info("写出 %s，%s", output, counts)
    return 1 if counts.get("error") else 0


def _publish(output: Path, done: dict[tuple[str, str, str], dict]) -> None:
    records = [done[key] for key in sorted(done)]
    _write_table(output / "pair_qc.csv", QC_FIELDS, [record["qc"] for record in records])
    _write_table(output / "station_summary.csv", STATION_FIELDS, [row for record in records for row in record["stations"]])
    _write_table(output / "lesion_match.csv", LESION_FIELDS, [row for record in records for row in record["lesions"]])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按解剖站点匹配基线和中期病灶，并写出新的量化表。")
    parser.add_argument("--interim-root", type=Path, default=PROJECT_ROOT / "data" / "interim")
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--stage8", type=Path, default=DEFAULT_STAGE8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--ants-bin", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
