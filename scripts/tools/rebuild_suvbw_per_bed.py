#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""用 SZR 的按床位 SUVbw 公式，重算 DLBCL interim 里的 PET SUVbw。

活度图（*_ACT.nii.gz）保持不变。原生 SUVbw 与 preprocessed/PET 里对齐到 CT
网格的 SUVbw 一起更新；不改 CT、掩码和已有定量表。
"""

from __future__ import annotations

import csv
import importlib.util
import os
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from nibabel.processing import resample_from_to

PROJECT = Path(__file__).resolve().parents[2]
INTERIM = PROJECT / "data" / "interim"
LOG_PATH = PROJECT / "logs" / "rebuild_suvbw_per_bed.csv"
SZR_CONVERTER = Path("/home/sun/Documents/SZR/scripts/tools/pacs_dicom_to_nifti_suv.py")
STEM_RE = re.compile(r"^s(\d+)_(.+)_ACT\.nii\.gz$")


def load_szr():
    spec = importlib.util.spec_from_file_location("szr_suv", SZR_CONVERTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["szr_suv"] = module
    spec.loader.exec_module(module)
    return module


def sanitize(value: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in value.strip())
    return re.sub(r"_{2,}", "_", cleaned) or "unknown"


def read_header(path: Path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def first_file(directory: Path) -> Path | None:
    try:
        names = [
            name for name in os.listdir(directory)
            if name not in {"DICOMDIR", "DICOMDIR.OLD"} and not name.startswith(".")
        ]
    except OSError:
        return None
    files = [directory / name for name in names if (directory / name).is_file()]
    if not files:
        return None
    return min(files, key=lambda path: path.name)


def index_series(roots: list[Path]) -> dict[tuple[str, str], list[dict]]:
    """(PatientID, StudyDate) -> PET series directories."""
    index: dict[tuple[str, str], list[dict]] = {}
    seen: set[Path] = set()
    checked = 0
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            files = [
                name for name in filenames
                if name not in {"DICOMDIR", "DICOMDIR.OLD"} and not name.startswith(".")
            ]
            if not files:
                continue
            directory = Path(dirpath)
            if directory in seen:
                continue
            seen.add(directory)
            checked += 1
            if checked % 500 == 0:
                print(f"索引 DICOM {checked}", flush=True)
            header = read_header(directory / files[0])
            if header is None or str(getattr(header, "Modality", "")).upper() != "PT":
                continue
            patient_id = str(getattr(header, "PatientID", "")).strip()
            study_date = str(getattr(header, "StudyDate", "")).strip()
            try:
                series_number = int(getattr(header, "SeriesNumber", -1))
            except (TypeError, ValueError):
                series_number = -1
            description = sanitize(str(getattr(header, "SeriesDescription", "")))
            if not patient_id or not study_date:
                continue
            index.setdefault((patient_id, study_date), []).append({
                "dir": directory,
                "number": series_number,
                "description": description,
            })
    print(f"DICOM 索引完成，查看 {checked} 个目录，PET 检查 {len(index)}", flush=True)
    return index


def match_series(index, patient_id: str, study_date: str, series_number: int, description: str):
    keys = [(patient_id, study_date)]
    stripped = patient_id.lstrip("0")
    if stripped != patient_id:
        keys.append((stripped, study_date))
    candidates = []
    for key, rows in index.items():
        if (key[0], key[1]) in keys or (key[0].lstrip("0"), key[1]) == (stripped, study_date):
            candidates.extend(rows)
    numbered = [row for row in candidates if row["number"] == series_number]
    exact = [row for row in numbered if row["description"] == description]
    if len(exact) >= 1:
        note = "" if len(exact) == 1 else "重复归档，已使用其中一份"
        return exact[0]["dir"], note
    if len(numbered) == 1:
        return numbered[0]["dir"], ""
    return None, "未找到 DICOM"


def rewrite_suv(szr, act_path: Path, dicom_dir: Path) -> dict:
    header = read_header(first_file(dicom_dir))
    prereq = szr.extract_suv_prerequisites(header)
    if not prereq.ok:
        raise RuntimeError(prereq.message)
    act_img = nib.load(str(act_path))
    activity = np.asanyarray(act_img.dataobj, dtype=np.float64)
    instances = szr._read_pet_instance_meta(dicom_dir)
    scales, outcome = szr.compute_suv_bw_slice_scales(
        prereq, instances, act_img.affine, activity.shape[:3],
    )
    if not outcome.ok or scales is None:
        raise RuntimeError(outcome.message)
    axis = szr._nifti_slice_axis(act_img.affine)
    broadcast = [1] * activity.ndim
    broadcast[axis] = scales.shape[0]
    suv = (activity * scales.reshape(broadcast)).astype(np.float32)
    suv_path = act_path.with_name(act_path.name.replace("_ACT.nii.gz", "_SUVbw.nii.gz"))
    old = np.asanyarray(nib.load(str(suv_path)).dataobj, dtype=np.float32) if suv_path.exists() else None
    out = nib.Nifti1Image(suv, act_img.affine, act_img.header)
    out.header.set_data_dtype(np.float32)
    out.header.set_slope_inter(1.0, 0.0)
    nib.save(out, str(suv_path))

    pre_path = act_path.parents[1] / "preprocessed" / "PET" / suv_path.name
    pre_status = "无预处理图"
    if pre_path.is_file():
        reference = nib.load(str(pre_path))
        aligned = resample_from_to(out, reference, order=1)
        aligned.set_data_dtype(np.float32)
        aligned.header.set_slope_inter(1.0, 0.0)
        nib.save(aligned, str(pre_path))
        pre_status = "已更新"

    changed = ""
    if old is not None and old.shape == suv.shape:
        mask = old > 0.5
        if int(mask.sum()) > 0:
            changed = f"{float(np.median(np.abs(suv[mask] - old[mask]) / old[mask])):.4f}"
    return {
        "状态": "ok",
        "说明": outcome.message,
        "预处理": pre_status,
        "SUV大于0.5的中位相对变化": changed,
        "输出": str(suv_path),
    }


def main() -> int:
    szr = load_szr()
    roots = [path for path in (PROJECT / "data" / "raw").glob("DICOM*") if path.is_dir()]
    roots.extend(path for path in (PROJECT / "data").glob("01-*") if path.is_dir())
    log_dirs = set()
    for path in (PROJECT / "logs").glob("pacs_dicom_to_nifti_suv_*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("modality", "")).upper() == "PT" and row.get("dicom_dir"):
                    log_dirs.add(Path(row["dicom_dir"]))
    roots.extend(path for path in sorted(log_dirs) if path.is_dir())
    index = index_series(roots)

    rows = []
    act_files = sorted(INTERIM.glob("*/*/PET/*_ACT.nii.gz"))
    for number, act_path in enumerate(act_files, start=1):
        patient_id = act_path.parents[2].name
        study_date = act_path.parents[1].name
        matched = STEM_RE.match(act_path.name)
        record = {
            "患者ID": patient_id,
            "日期": study_date,
            "文件": act_path.name,
            "状态": "",
            "说明": "",
            "预处理": "",
            "SUV大于0.5的中位相对变化": "",
            "输出": "",
        }
        if matched is None:
            record["状态"] = "error"
            record["说明"] = "文件名不是 s序列号_描述_ACT"
        else:
            dicom_dir, reason = match_series(
                index, patient_id, study_date, int(matched.group(1)), matched.group(2),
            )
            if dicom_dir is None:
                record["状态"] = "missing"
                record["说明"] = reason
            else:
                try:
                    record.update(rewrite_suv(szr, act_path, dicom_dir))
                except Exception as exc:  # noqa: BLE001
                    record["状态"] = "error"
                    record["说明"] = str(exc)
        rows.append(record)
        if number % 20 == 0 or record["状态"] != "ok":
            print(f"{number}/{len(act_files)} {patient_id} {study_date} {record['状态']} {record['说明']}", flush=True)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["状态"]] = counts.get(row["状态"], 0) + 1
    print("完成", counts, "日志", LOG_PATH, flush=True)
    return 0 if counts.get("ok", 0) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
