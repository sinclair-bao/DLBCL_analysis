#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   catalog.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    扫描 data/interim 与 data/processed，为每个 (PatientID, StudyDate)
    解析工作空间路径：2 mm CT/PET、同机配准产物、病灶/器官 mask。
    GUI 与跨检查配准/特征提取共用这一索引，缺文件时用状态灯表示，不抛异常。
"""

from __future__ import annotations

import sys
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS / "common") not in sys.path:
    sys.path.insert(0, str(_SCRIPTS / "common"))

from pipeline_utils import discover_subject_studies  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def resolve_data_directory(directory: Path | str) -> tuple[Path, Path, str | None]:
    """Resolve a data/, interim/, or single-patient directory for the GUI."""
    selected = Path(directory).expanduser().resolve()
    if not selected.is_dir():
        raise ValueError(f"数据目录不存在：{selected}")
    if (selected / "interim").is_dir():
        return selected / "interim", selected / "processed", None
    if selected.name == "interim":
        return selected, selected.parent / "processed", None
    study_dirs = [child for child in selected.iterdir() if child.is_dir()]
    is_patient = selected.parent.name == "interim" or any(
        any((study / name).is_dir() for name in ("CT", "PET", "preprocessed"))
        for study in study_dirs
    )
    if is_patient:
        processed = (selected.parent.parent / "processed" if selected.parent.name == "interim"
                     else selected.parent / "processed")
        return selected.parent, processed, selected.name
    return selected, selected.parent / "processed", None

# 器官标签（与 organ_segmentation.py 一致）
ORGAN_SPLEEN = 1
ORGAN_LIVER = 3


def _first_nifti(directory: Path, preferred_substr: Optional[str] = None) -> Optional[Path]:
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("*.nii.gz"))
    if not files:
        return None
    if preferred_substr:
        hit = [p for p in files if preferred_substr in p.name]
        if hit:
            return hit[0]
    return files[0]


def _first_glob(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.is_dir():
        return None
    files = sorted(directory.glob(pattern))
    return files[0] if files else None


@dataclass
class StudyAssets:
    """单次 PET/CT 检查在磁盘上的可用文件。"""

    patient_id: str
    study_date: str
    interim_dir: Path
    processed_dir: Path
    ct_orig: Optional[Path] = None
    pet_orig: Optional[Path] = None
    ct_iso: Optional[Path] = None
    pet_iso: Optional[Path] = None
    pet_aligned: Optional[Path] = None
    ct_reference: Optional[Path] = None
    lesion_auto: Optional[Path] = None
    lesion_edited: Optional[Path] = None
    organs: Optional[Path] = None
    pet_to_ct_mat: Optional[Path] = None
    pet_units: str = "SUVbw"
    tracer: str = ""

    @property
    def working_ct(self) -> Optional[Path]:
        """影像组学 / 显示用 CT：优先同机配准副本，否则 2 mm 预处理 CT。"""
        return self.ct_reference or self.ct_iso

    @property
    def working_pet(self) -> Optional[Path]:
        """影像组学 / 显示用 PET：优先同机刚体对齐后的 SUV，否则 2 mm PET。"""
        return self.pet_aligned or self.pet_iso

    @property
    def working_lesion(self) -> Optional[Path]:
        """显示 / 映射 / 特征用 mask：人工调整副本优先。"""
        return self.lesion_edited or self.lesion_auto

    @property
    def lesion_mask(self) -> Optional[Path]:
        """兼容旧调用，等同 working_lesion。"""
        return self.working_lesion

    def edited_mask_path(self) -> Path:
        return (
            self.processed_dir
            / "masks"
            / f"{self.patient_id}_{self.study_date}_lesion_edited.nii.gz"
        )

    def completeness(self) -> str:
        """
        ok      : 工作 CT + 工作 PET + 病灶 mask 齐全
        partial : 有 CT/PET，但缺 mask 或同机配准
        missing : 缺 CT 或 PET，无法浏览
        """
        if self.working_ct is None or self.working_pet is None:
            return "missing"
        if self.working_lesion is None:
            return "partial"
        if self.pet_aligned is None:
            return "partial"
        return "ok"

    def status_note(self) -> str:
        missing: list[str] = []
        if self.working_ct is None:
            missing.append("CT")
        if self.working_pet is None:
            missing.append("PET")
        if self.working_lesion is None:
            missing.append("lesion")
        elif self.lesion_edited is not None:
            missing.append("edited")
        if self.pet_aligned is None:
            missing.append("intra-scan")
        if self.organs is None:
            missing.append("organs")
        return ",".join(missing) if missing else "complete"

    def warped_baseline_mask(self, baseline_date: str) -> Optional[Path]:
        """随访目录中、由指定基线日期映射来的病灶床 mask。"""
        out = self.processed_dir / "longitudinal" / "baseline_lesion_warped.nii.gz"
        mat = (
            self.processed_dir
            / "longitudinal"
            / f"baseline_{baseline_date}_to_this_0GenericAffine.mat"
        )
        if out.is_file() and mat.is_file():
            manifest = self.processed_dir / "longitudinal" / "mapping_qc.json"
            if manifest.is_file():
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    if data.get("inputs", {}).get("baseline_date") != baseline_date:
                        return None
                except (OSError, ValueError):
                    return None
            return out
        return None

    def longitudinal_dir(self) -> Path:
        return self.processed_dir / "longitudinal"


@dataclass
class PatientRecord:
    patient_id: str
    studies: list[StudyAssets] = field(default_factory=list)

    def study_dates(self) -> list[str]:
        return [s.study_date for s in self.studies]

    def get(self, study_date: str) -> Optional[StudyAssets]:
        for s in self.studies:
            if s.study_date == study_date:
                return s
        return None


class DataCatalog:
    """data/interim + data/processed 的只读索引。"""

    def __init__(
        self,
        interim_root: Path | str = DEFAULT_INTERIM_ROOT,
        processed_root: Path | str = DEFAULT_PROCESSED_ROOT,
        manifest_path: Path | str | None = None,
        *,
        patient_filter: str | None = None,
    ) -> None:
        self.interim_root = Path(interim_root)
        self.processed_root = Path(processed_root)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.patient_filter = patient_filter
        self.patients: dict[str, PatientRecord] = {}
        self.refresh()

    def refresh(self) -> None:
        self.patients = {}
        if self.manifest_path is not None:
            self._scan_manifest()
            return
        for patient_id, study_date, study_dir in discover_subject_studies(self.interim_root):
            if self.patient_filter is not None and patient_id != self.patient_filter:
                continue
            assets = self._scan_study(patient_id, study_date, study_dir)
            rec = self.patients.setdefault(patient_id, PatientRecord(patient_id))
            rec.studies.append(assets)

    def _scan_manifest(self) -> None:
        """Index an arbitrary longitudinal PET cohort from explicit paths."""
        with self.manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"patient_id", "study_date", "ct_path", "pet_path", "pet_units"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"manifest requires columns: {sorted(required)}")
            for row in reader:
                patient_id = row["patient_id"].strip()
                study_date = row["study_date"].strip()
                if not patient_id or not study_date:
                    raise ValueError("manifest patient_id and study_date cannot be empty")
                if any(v in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]+", v) is None
                       for v in (patient_id, study_date)):
                    raise ValueError("manifest IDs must be safe path segments")
                pet_units = (row.get("pet_units") or "").strip()
                if not pet_units:
                    raise ValueError(f"manifest pet_units cannot be empty: {patient_id}/{study_date}")
                def path(key: str) -> Optional[Path]:
                    value = (row.get(key) or "").strip()
                    if not value:
                        return None
                    candidate = Path(value).expanduser()
                    if not candidate.is_absolute():
                        candidate = self.manifest_path.parent / candidate
                    if not candidate.is_file():
                        raise FileNotFoundError(f"manifest {key}: {candidate}")
                    return candidate
                processed = self.processed_root / patient_id / study_date
                edited = processed / "masks" / f"{patient_id}_{study_date}_lesion_edited.nii.gz"
                assets = StudyAssets(
                    patient_id=patient_id, study_date=study_date,
                    interim_dir=self.manifest_path.parent, processed_dir=processed,
                    ct_reference=path("ct_path"), pet_aligned=path("pet_path"),
                    lesion_auto=path("lesion_mask_path"),
                    lesion_edited=edited if edited.is_file() else None,
                    organs=path("organ_mask_path"),
                    pet_units=pet_units,
                    tracer=(row.get("tracer") or "").strip(),
                )
                rec = self.patients.setdefault(patient_id, PatientRecord(patient_id))
                if rec.get(study_date) is not None:
                    raise ValueError(f"duplicate patient/study: {patient_id}/{study_date}")
                rec.studies.append(assets)

    def patient_ids(self) -> list[str]:
        return list(self.patients.keys())

    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        return self.patients.get(patient_id)

    def get_study(self, patient_id: str, study_date: str) -> Optional[StudyAssets]:
        rec = self.patients.get(patient_id)
        return rec.get(study_date) if rec else None

    def _scan_study(self, patient_id: str, study_date: str, study_dir: Path) -> StudyAssets:
        processed_dir = self.processed_root / patient_id / study_date
        reg_dir = processed_dir / "registration"
        return StudyAssets(
            patient_id=patient_id,
            study_date=study_date,
            interim_dir=study_dir,
            processed_dir=processed_dir,
            ct_orig=_first_nifti(study_dir / "CT", "WB_Standard"),
            pet_orig=_first_glob(study_dir / "PET", "*_SUVbw.nii.gz"),
            ct_iso=_first_nifti(study_dir / "preprocessed" / "CT", "WB_Standard"),
            pet_iso=_first_glob(study_dir / "preprocessed" / "PET", "*_SUVbw.nii.gz"),
            pet_aligned=_first_glob(reg_dir, "pet_iso_aligned.nii.gz"),
            ct_reference=_first_glob(reg_dir, "ct_iso_reference.nii.gz"),
            lesion_auto=_first_glob(
                processed_dir / "masks", f"{patient_id}_{study_date}_lesion.nii.gz"
            ),
            lesion_edited=_first_glob(
                processed_dir / "masks", f"{patient_id}_{study_date}_lesion_edited.nii.gz"
            ),
            organs=_first_glob(processed_dir / "organs", "organs.nii.gz"),
            pet_to_ct_mat=_first_glob(reg_dir, "pet_to_ct_0GenericAffine.mat"),
        )
