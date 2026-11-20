#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
单病灶定量：SUL、PERCIST、qPET 百分变化和 Deauville 规则近似。

瘦体重用 Janmahasatian（2005），身高以米、体重以千克：
    男  9270 × 体重 / (6680 + 216 × BMI)
    女  9270 × 体重 / (8780 + 244 × BMI)
SUL = SUVbw × LBM / 体重。基线和中期各用自己的体重。

Deauville 分界是本流程的规则近似，不是阅片评分：≤血池为 2，≤肝为 3，
≤2×肝均值为 4，更高为 5。血池取 TotalSegmentator total 任务的主动脉标签 52。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger("lesion_quant")

LBM_FORMULA = "janmahasatian2005"
AORTA_LABEL = 52
DEAUVILLE_LIVER_MULTIPLE = 2.0
PERCIST_MAX_TARGETS = 5
PERCIST_MAX_PER_STATION = 2

RESPONSE_FIELDS = [
    "patient_id",
    "baseline_date",
    "interim_date",
    "pairing",
    "sul_evaluable",
    "lbm_formula",
    "baseline_sul_scale",
    "interim_sul_scale",
    "n_measurable",
    "n_targets",
    "baseline_sul_sum",
    "interim_sul_sum",
    "pct_delta_sul_sum",
    "abs_delta_sul",
    "new_measurable",
    "percist",
    "blood_pool_suv",
]


def parse_height_m(patient_size: object) -> float | None:
    """DICOM PatientSize。大于 3 视为厘米。"""
    value = _as_float(patient_size)
    if value is None or value <= 0:
        return None
    if value > 3.0:
        value = value / 100.0
    if value <= 0 or value > 3.0:
        return None
    return value


def lean_body_mass_kg(sex: str, height_m: float | None, weight_kg: float | None) -> float | None:
    """Janmahasatian 2005。性别不是 M/F，或身高、体重缺失时返回 None。"""
    label = (sex or "").strip().upper()
    if label not in {"M", "F"} or height_m is None or weight_kg is None:
        return None
    if height_m <= 0 or weight_kg <= 0:
        return None
    bmi = weight_kg / (height_m ** 2)
    if label == "M":
        return 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
    return 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)


def sul_scale(sex: str, height_m: float | None, weight_kg: float | None) -> float | None:
    lbm = lean_body_mass_kg(sex, height_m, weight_kg)
    if lbm is None or weight_kg is None or weight_kg <= 0:
        return None
    return lbm / weight_kg


def percent_change(baseline: float, follow: float) -> float:
    """(随访 − 基线) / 基线 × 100。基线为 0 或非数时返回 nan。"""
    if not math.isfinite(baseline) or not math.isfinite(follow) or baseline == 0:
        return float("nan")
    return (follow - baseline) / baseline * 100.0


def deauville_surrogate(suv: float, blood: float, liver: float) -> int | None:
    """规则近似。血池或肝均值缺失时不评分。"""
    if not math.isfinite(suv) or not math.isfinite(blood) or not math.isfinite(liver):
        return None
    if blood <= 0 or liver <= 0:
        return None
    if suv <= blood:
        return 2
    if suv <= liver:
        return 3
    if suv <= DEAUVILLE_LIVER_MULTIPLE * liver:
        return 4
    return 5


def masked_mean_sd(values_image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if mask is None or not np.any(mask):
        return float("nan"), float("nan")
    values = np.asarray(values_image, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std())


def liver_reference(
    pet: np.ndarray,
    organs: np.ndarray | None,
    lesion_labels: np.ndarray | None,
    liver_label: int,
) -> tuple[float, float]:
    """全肝均值和标准差，优先去掉病灶体素。"""
    if organs is None:
        return float("nan"), float("nan")
    mask = np.asarray(organs) == int(liver_label)
    if lesion_labels is not None:
        kept = mask & (np.asarray(lesion_labels) <= 0)
        if np.any(kept):
            mask = kept
    return masked_mean_sd(pet, mask)


def blood_pool_suv(
    pet: np.ndarray,
    totalseg: np.ndarray | None,
    lesion_labels: np.ndarray | None,
    aorta_label: int = AORTA_LABEL,
) -> float:
    """主动脉平均 SUV。没有主动脉标签时返回 nan。"""
    if totalseg is None:
        return float("nan")
    mask = np.asarray(totalseg) == int(aorta_label)
    if lesion_labels is not None:
        kept = mask & (np.asarray(lesion_labels) <= 0)
        if np.any(kept):
            mask = kept
    mean, _sd = masked_mean_sd(pet, mask)
    return mean


def apply_study_metrics(
    rows: list[dict],
    *,
    baseline_scale: float,
    interim_scale: float,
    blood_pool: float,
    deauville_liver: float,
    baseline_liver_mean: float,
    baseline_liver_sd: float,
) -> None:
    """把 SUL、可测量标记和 Deauville 近似写进已有匹配行。"""
    threshold = _measurable_threshold(baseline_scale, baseline_liver_mean, baseline_liver_sd)
    for row in rows:
        baseline_peak = _num(row.get("baseline_suv_peak"))
        interim_peak = _num(row.get("interim_suv_peak"))
        baseline_sul = baseline_peak * baseline_scale if _both_finite(baseline_peak, baseline_scale) else float("nan")
        interim_sul = interim_peak * interim_scale if _both_finite(interim_peak, interim_scale) else float("nan")
        row["baseline_sul_peak"] = _cell(baseline_sul)
        row["interim_sul_peak"] = _cell(interim_sul)
        if row["match_status"] in ("matched", "review"):
            row["pct_delta_sul_peak"] = _cell(percent_change(baseline_sul, interim_sul))
        else:
            row["pct_delta_sul_peak"] = ""
        if row.get("baseline_lesion_id") == "" or not math.isfinite(baseline_sul) or not math.isfinite(threshold):
            row["baseline_measurable"] = ""
        else:
            row["baseline_measurable"] = 1 if baseline_sul >= threshold else 0
        score_suv = _deauville_suv(row)
        score = deauville_surrogate(score_suv, blood_pool, deauville_liver)
        row["deauville_surrogate"] = "" if score is None else int(score)


def summarize_percist(
    rows: list[dict],
    *,
    patient_id: str,
    baseline_date: str,
    interim_date: str,
    pairing: str,
    baseline_scale: float,
    interim_scale: float,
    baseline_liver_mean: float,
    baseline_liver_sd: float,
    interim_liver_mean: float,
    interim_liver_sd: float,
    blood_pool: float,
) -> dict:
    """最多 5 个最热可测量灶，同一站点最多 2 个，比较 SULpeak 之和。"""
    evaluable = _both_finite(baseline_scale, interim_scale) and all(
        math.isfinite(value)
        for value in (baseline_liver_mean, baseline_liver_sd, interim_liver_mean, interim_liver_sd)
    )
    targets = _percist_targets(rows) if evaluable else []
    baseline_sum, interim_sum = _target_sums(rows, targets, interim_scale)
    new_measurable = _has_new_measurable(
        rows, _measurable_threshold(interim_scale, interim_liver_mean, interim_liver_sd),
    ) if evaluable else False
    hot_new = any(row.get("match_status") == "new" and _flag(row.get("quantitative_outlier")) == 1 for row in rows)
    measurable = _unique_measurable(rows) if evaluable else []
    category = _percist_category(
        evaluable=evaluable,
        measurable=measurable,
        rows=rows,
        baseline_sum=baseline_sum,
        interim_sum=interim_sum,
        new_measurable=new_measurable,
        hot_new=hot_new,
    )
    return {
        "patient_id": patient_id,
        "baseline_date": baseline_date,
        "interim_date": interim_date,
        "pairing": pairing,
        "sul_evaluable": 1 if evaluable else 0,
        "lbm_formula": LBM_FORMULA,
        "baseline_sul_scale": _cell(baseline_scale),
        "interim_sul_scale": _cell(interim_scale),
        "n_measurable": len(measurable),
        "n_targets": len(targets),
        "baseline_sul_sum": _cell(baseline_sum),
        "interim_sul_sum": _cell(interim_sum),
        "pct_delta_sul_sum": _cell(percent_change(baseline_sum, interim_sum)),
        "abs_delta_sul": _cell(interim_sum - baseline_sum if _both_finite(baseline_sum, interim_sum) else float("nan")),
        "new_measurable": 1 if new_measurable else 0,
        "percist": category,
        "blood_pool_suv": _cell(blood_pool),
    }


def load_pet_demographics(
    patient_id: str,
    study_date: str,
    processed_dir: Path,
    project_root: Path,
) -> dict:
    """读缓存；没有则从转换日志或 data/raw/DICOM* 的 PET 头里取性别、身高、体重。"""
    cache = Path(processed_dir) / "pet_demographics.json"
    if cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        if isinstance(cached, dict) and cached.get("patient_id") == patient_id and cached.get("study_date") == study_date:
            return cached
    record = _demographics_from_dicom(patient_id, study_date, project_root)
    record["patient_id"] = patient_id
    record["study_date"] = study_date
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("无法写入人口学缓存 %s: %s", cache, exc)
    return record


def scale_from_demographics(record: dict) -> float:
    value = record.get("sul_scale")
    number = _as_float(value)
    return float("nan") if number is None else number


def _measurable_threshold(scale: float, liver_mean: float, liver_sd: float) -> float:
    if not _both_finite(scale, liver_mean) or not math.isfinite(liver_sd):
        return float("nan")
    return scale * (1.5 * liver_mean + 2.0 * liver_sd)


def _deauville_suv(row: dict) -> float:
    status = row.get("match_status")
    if status in ("matched", "review", "new"):
        return _num(row.get("interim_suv_max"))
    if status in ("resolved", "unmatched"):
        return _num(row.get("residual_suv_max"))
    return float("nan")


def _unique_measurable(rows: list[dict]) -> dict[object, dict]:
    chosen: dict[object, dict] = {}
    for row in rows:
        if _flag(row.get("baseline_measurable")) != 1:
            continue
        if row.get("match_status") == "invalid" or row.get("baseline_lesion_id") == "":
            continue
        lesion_id = row["baseline_lesion_id"]
        peak = _num(row.get("baseline_sul_peak"))
        current = chosen.get(lesion_id)
        if current is None or peak > _num(current.get("baseline_sul_peak")):
            chosen[lesion_id] = row
    return chosen


def _percist_targets(rows: list[dict]) -> list[dict]:
    ordered = sorted(
        _unique_measurable(rows).values(),
        key=lambda row: _num(row.get("baseline_sul_peak")),
        reverse=True,
    )
    chosen: list[dict] = []
    per_station: dict[str, int] = {}
    for row in ordered:
        if not math.isfinite(_num(row.get("baseline_sul_peak"))):
            continue
        station = str(row.get("station", ""))
        if per_station.get(station, 0) >= PERCIST_MAX_PER_STATION:
            continue
        chosen.append(row)
        per_station[station] = per_station.get(station, 0) + 1
        if len(chosen) >= PERCIST_MAX_TARGETS:
            break
    return chosen


def _target_sums(rows: list[dict], targets: list[dict], interim_scale: float) -> tuple[float, float]:
    if not targets:
        return float("nan"), float("nan")
    baseline_sum = 0.0
    interim_sum = 0.0
    for target in targets:
        baseline_peak = _num(target.get("baseline_sul_peak"))
        if not math.isfinite(baseline_peak):
            return float("nan"), float("nan")
        follow = _followup_sul(rows, target["baseline_lesion_id"], interim_scale)
        if not math.isfinite(follow):
            return float("nan"), float("nan")
        baseline_sum += baseline_peak
        interim_sum += follow
    return baseline_sum, interim_sum


def _followup_sul(rows: list[dict], baseline_id: object, interim_scale: float) -> float:
    total = 0.0
    found = False
    for row in rows:
        if row.get("baseline_lesion_id") != baseline_id:
            continue
        status = row.get("match_status")
        if status in ("matched", "review"):
            peak = _num(row.get("interim_sul_peak"))
        elif status in ("resolved", "unmatched"):
            residual = _num(row.get("_residual_suv_peak"))
            peak = residual * interim_scale if _both_finite(residual, interim_scale) else float("nan")
        else:
            continue
        if not math.isfinite(peak):
            return float("nan")
        total += peak
        found = True
    return total if found else float("nan")


def _has_new_measurable(rows: list[dict], threshold: float) -> bool:
    if not math.isfinite(threshold):
        return False
    for row in rows:
        if row.get("match_status") != "new":
            continue
        if _num(row.get("interim_sul_peak")) >= threshold:
            return True
    return False


def _percist_category(
    *,
    evaluable: bool,
    measurable: dict,
    rows: list[dict],
    baseline_sum: float,
    interim_sum: float,
    new_measurable: bool,
    hot_new: bool,
) -> str:
    if not evaluable or not measurable:
        return "not_evaluable"
    if new_measurable or hot_new:
        return "PMD"
    all_resolved = all(
        any(
            row.get("baseline_lesion_id") == lesion_id and row.get("match_status") == "resolved"
            for row in rows
        )
        for lesion_id in measurable
    )
    if all_resolved:
        return "CMR"
    if not _both_finite(baseline_sum, interim_sum) or baseline_sum == 0:
        return "not_evaluable"
    change = interim_sum - baseline_sum
    percent = change / baseline_sum * 100.0
    if percent <= -30.0 and change <= -0.8:
        return "PMR"
    if percent >= 30.0 and change >= 0.8:
        return "PMD"
    return "SMD"


def _demographics_from_dicom(patient_id: str, study_date: str, project_root: Path) -> dict:
    found = _dataset_from_logs(patient_id, study_date, project_root)
    if found is None:
        found = _lookup_raw_index(patient_id, study_date, project_root)
    if found is None:
        logger.warning("未找到 %s %s 的 PET DICOM，SUL 留空", patient_id, study_date)
        return _empty_demographics("")
    return found


_PET_INDEX: dict[tuple[str, str], dict] | None = None


def _dataset_from_logs(patient_id: str, study_date: str, project_root: Path) -> dict | None:
    log_dir = project_root / "logs"
    if not log_dir.is_dir():
        return None
    for path in sorted(log_dir.glob("pacs_dicom_to_nifti_suv_*.csv")):
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("patient_id") != patient_id or row.get("study_date") != study_date:
                        continue
                    if str(row.get("modality", "")).upper() != "PT":
                        continue
                    series = Path(row.get("dicom_dir", ""))
                    dataset = _read_header(_first_dicom_file(series))
                    if dataset is None or str(getattr(dataset, "Modality", "")).upper() != "PT":
                        continue
                    logger.info("PET DICOM %s %s -> %s", patient_id, study_date, series)
                    return _demographics_from_dataset(dataset, str(series))
        except OSError:
            continue
    return None


def _lookup_raw_index(patient_id: str, study_date: str, project_root: Path) -> dict | None:
    """整库只扫一遍 PET 头，供同一次运行里的其余检查复用。"""
    global _PET_INDEX
    if _PET_INDEX is None:
        _PET_INDEX = _build_pet_index(project_root)
    found = _PET_INDEX.get((patient_id, study_date))
    if found is not None:
        return found
    normalized = patient_id.strip().lstrip("0")
    for (found_id, found_date), record in _PET_INDEX.items():
        if found_date == study_date and found_id.strip().lstrip("0") == normalized:
            return record
    return None


def _build_pet_index(project_root: Path) -> dict[tuple[str, str], dict]:
    raw = project_root / "data" / "raw"
    index: dict[tuple[str, str], dict] = {}
    if not raw.is_dir():
        return index
    checked = 0
    for pacs in sorted(path for path in raw.glob("DICOM*") if path.is_dir()):
        for dirpath, _dirnames, filenames in os.walk(pacs):
            files = [
                name for name in filenames
                if name not in {"DICOMDIR", "DICOMDIR.OLD"} and not name.startswith(".")
            ]
            if not files:
                continue
            checked += 1
            if checked % 400 == 0:
                logger.info("正在索引 PET DICOM 头，已查看 %d 个序列", checked)
            dataset = _read_header(Path(dirpath) / files[0])
            if dataset is None or str(getattr(dataset, "Modality", "")).upper() != "PT":
                continue
            patient_id = str(getattr(dataset, "PatientID", "")).strip()
            study_date = str(getattr(dataset, "StudyDate", "")).strip()
            if not patient_id or not study_date:
                continue
            record = _demographics_from_dataset(dataset, dirpath)
            key = (patient_id, study_date)
            current = index.get(key)
            if current is None or (current.get("sul_scale") is None and record.get("sul_scale") is not None):
                index[key] = record
    logger.info("PET DICOM 索引完成，%d 个序列中有 %d 次 PT 检查", checked, len(index))
    return index


def _demographics_from_dataset(dataset, dicom_dir: str) -> dict:
    sex = str(getattr(dataset, "PatientSex", "") or "").strip().upper()
    height = parse_height_m(getattr(dataset, "PatientSize", None))
    weight = _as_float(getattr(dataset, "PatientWeight", None))
    if weight is not None and weight <= 0:
        weight = None
    lbm = lean_body_mass_kg(sex, height, weight)
    scale = None if lbm is None or weight is None else lbm / weight
    return {
        "sex": sex if sex in {"M", "F"} else "",
        "height_m": height,
        "weight_kg": weight,
        "lbm_kg": lbm,
        "sul_scale": scale,
        "lbm_formula": LBM_FORMULA,
        "dicom_dir": dicom_dir,
    }


def _empty_demographics(dicom_dir: str) -> dict:
    return {
        "sex": "",
        "height_m": None,
        "weight_kg": None,
        "lbm_kg": None,
        "sul_scale": None,
        "lbm_formula": LBM_FORMULA,
        "dicom_dir": dicom_dir,
    }


def _first_dicom_file(directory: Path) -> Path | None:
    if directory is None or not directory.is_dir():
        return None
    files = [
        path for path in directory.iterdir()
        if path.is_file() and path.name not in {"DICOMDIR", "DICOMDIR.OLD"} and not path.name.startswith(".")
    ]
    if not files:
        return None
    return min(files, key=lambda path: path.name)


def _read_header(path: Path | None):
    if path is None or not path.is_file():
        return None
    try:
        import pydicom
    except ImportError:
        logger.warning("未安装 pydicom，无法读取 PET 头信息")
        return None
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _num(value: object) -> float:
    number = _as_float(value)
    return float("nan") if number is None else number


def _both_finite(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isfinite(right)


def _flag(value: object) -> int:
    number = _as_float(value)
    if number is None:
        return 0
    return 1 if number == 1 else 0


def _cell(value: object, digits: int = 4):
    number = _as_float(value)
    if number is None:
        return ""
    return round(number, digits)
