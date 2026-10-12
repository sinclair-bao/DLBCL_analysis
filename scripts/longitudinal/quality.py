"""Versioned mask review and provenance for longitudinal PET studies."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def fingerprint(path: Path) -> dict:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def review_path(mask_path: Path) -> Path:
    mask_path = Path(mask_path)
    return mask_path.parent / f"{mask_path.name}.review.json"


def _mask_events_path(mask_path: Path) -> Path:
    return Path(mask_path).parent / f"{Path(mask_path).name}.events.jsonl"


def record_mask_version(mask_path: Path, source_path: Path | None = None) -> Path:
    """Preserve every edited revision; set its review state to pending."""
    mask_path = Path(mask_path)
    versions = mask_path.parent / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    version = versions / f"{mask_path.name.removesuffix('.nii.gz')}_{stamp}.nii.gz"
    shutil.copy2(mask_path, version)
    record = {
        "schema_version": 1,
        "mask": fingerprint(mask_path),
        "version_path": str(version),
        "source": fingerprint(source_path) if source_path and Path(source_path).is_file() else None,
        "status": "pending",
        "reviewer": "",
        "reason": "",
        "updated_at": _now(),
    }
    _write_json(review_path(mask_path), record)
    _append_event(_mask_events_path(mask_path), {"event": "version_saved", **record})
    return version


def archive_existing_mask(mask_path: Path) -> Path | None:
    """Preserve a pre-existing working mask before it is replaced."""
    mask_path = Path(mask_path)
    if not mask_path.is_file():
        return None
    versions = mask_path.parent / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = versions / f"{mask_path.name.removesuffix('.nii.gz')}_{stamp}_before.nii.gz"
    shutil.copy2(mask_path, backup)
    _append_event(_mask_events_path(mask_path), {
        "event": "preedit_archived", "mask": fingerprint(mask_path),
        "version_path": str(backup), "updated_at": _now(),
    })
    return backup


def review_mask(mask_path: Path, reviewer: str, reason: str, status: str) -> None:
    if status not in {"approved", "rejected"}:
        raise ValueError("review status must be approved or rejected")
    if not reviewer.strip() or not reason.strip():
        raise ValueError("reviewer and reason are required")
    mask_path = Path(mask_path)
    path = review_path(mask_path)
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    source = record.get("source")
    if source and (not Path(source["path"]).is_file() or
                   source != fingerprint(Path(source["path"]))):
        raise ValueError("source mask has changed; reopen and save a new edited version")
    if record.get("mask") and record["mask"] != fingerprint(mask_path):
        raise ValueError("mask has changed since its last version; save a new version first")
    record.update({
        "schema_version": 1,
        "mask": fingerprint(mask_path),
        "status": status,
        "reviewer": reviewer.strip(),
        "reason": reason.strip(),
        "updated_at": _now(),
    })
    _write_json(path, record)
    _append_event(_mask_events_path(mask_path), {"event": "review", **record})


def mask_review_status(mask_path: Path) -> str:
    path = review_path(mask_path)
    if not path.is_file():
        return "unreviewed"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("mask") != fingerprint(mask_path):
            return "stale"
        source = record.get("source")
        if source and (not Path(source["path"]).is_file() or
                       source != fingerprint(Path(source["path"]))):
            return "stale"
        return record.get("status", "unreviewed")
    except (OSError, ValueError, KeyError):
        return "invalid"


def mapping_manifest_path(followup) -> Path:
    return followup.longitudinal_dir() / "mapping_qc.json"


def mapping_inputs(baseline, followup) -> dict:
    if baseline.working_ct is None or baseline.working_lesion is None or followup.working_ct is None:
        raise ValueError("mapping inputs are incomplete")
    return {
        "baseline_date": baseline.study_date,
        "followup_date": followup.study_date,
        "baseline_ct": fingerprint(baseline.working_ct),
        "followup_ct": fingerprint(followup.working_ct),
        "baseline_mask": fingerprint(baseline.working_lesion),
        "method": "ANTs-Rigid-Affine-GenericLabel",
        "registration_protocol_version": "rigid_affine_mi_v1",
    }


def write_mapping_manifest(baseline, followup, result_path: Path) -> None:
    warped_ct = followup.longitudinal_dir() / "baseline_ct_warped.nii.gz"
    record = {
        "schema_version": 1,
        "inputs": mapping_inputs(baseline, followup),
        "result": fingerprint(result_path),
        "warped_ct": fingerprint(warped_ct),
        "status": "pending",
        "reviewer": "",
        "reason": "",
        "updated_at": _now(),
    }
    _write_json(mapping_manifest_path(followup), record)
    _append_event(followup.longitudinal_dir() / "mapping_events.jsonl",
                  {"event": "mapping_created", **record})


def mapping_status(baseline, followup) -> str:
    path = mapping_manifest_path(followup)
    result = followup.warped_baseline_mask(baseline.study_date)
    if not path.is_file():
        return "unreviewed"
    if result is None:
        return "stale"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("inputs") != mapping_inputs(baseline, followup):
            return "stale"
        if record.get("result") != fingerprint(result):
            return "stale"
        warped_ct = followup.longitudinal_dir() / "baseline_ct_warped.nii.gz"
        if not warped_ct.is_file() or record.get("warped_ct") != fingerprint(warped_ct):
            return "stale"
        return record.get("status", "unreviewed")
    except (OSError, ValueError, KeyError):
        return "invalid"


def review_mapping(baseline, followup, reviewer: str, reason: str, status: str) -> None:
    if status not in {"approved", "rejected"}:
        raise ValueError("review status must be approved or rejected")
    if not reviewer.strip() or not reason.strip():
        raise ValueError("reviewer and reason are required")
    if mapping_status(baseline, followup) in {"stale", "invalid", "unreviewed"}:
        raise ValueError("mapping is missing, stale or invalid; rerun mapping first")
    path = mapping_manifest_path(followup)
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update({"status": status, "reviewer": reviewer.strip(),
                   "reason": reason.strip(), "updated_at": _now()})
    _write_json(path, record)
    _append_event(followup.longitudinal_dir() / "mapping_events.jsonl",
                  {"event": "review", **record})


def lesion_links_path(processed_root: Path, patient_id: str) -> Path:
    return Path(processed_root) / patient_id / "lesion_links.json"


def record_lesion_link(processed_root: Path, baseline, followup, *,
                       baseline_label: int | None, followup_label: int,
                       reviewer: str, reason: str) -> str:
    """Record a reviewed cross-time identity independent of component numbering."""
    if not reviewer.strip() or not reason.strip():
        raise ValueError("reviewer and reason are required")
    if followup_label <= 0:
        raise ValueError("followup label must be positive")
    if followup.working_lesion is None:
        raise ValueError("followup mask is missing")
    if baseline_label is not None and baseline_label <= 0:
        raise ValueError("baseline label must be positive")
    if baseline_label is not None and baseline.working_lesion is None:
        raise ValueError("baseline mask is missing")
    uid = (f"baseline:{baseline.study_date}:{baseline_label}" if baseline_label is not None
           else f"new:{followup.study_date}:{followup_label}")
    path = lesion_links_path(processed_root, followup.patient_id)
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"schema_version": 1, "links": []}
    entry = {
        "lesion_uid": uid, "baseline_date": baseline.study_date,
        "baseline_label": baseline_label, "followup_date": followup.study_date,
        "followup_label": followup_label, "baseline_mask":
            fingerprint(baseline.working_lesion) if baseline.working_lesion else None,
        "followup_mask": fingerprint(followup.working_lesion),
        "reviewer": reviewer.strip(), "reason": reason.strip(), "updated_at": _now(),
    }
    data["links"] = [v for v in data["links"] if not (
        v["followup_date"] == followup.study_date and v["followup_label"] == followup_label)]
    data["links"].append(entry)
    _write_json(path, data)
    return uid


def lesion_uid(processed_root: Path, patient_id: str, study_date: str,
               label: int, mask_path: Path) -> str:
    path = lesion_links_path(processed_root, patient_id)
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        current = fingerprint(mask_path)
        for row in data.get("links", []):
            if (row.get("followup_date") == study_date and row.get("followup_label") == label
                    and row.get("followup_mask") == current):
                return row["lesion_uid"]
    except (OSError, ValueError, KeyError):
        pass
    return ""
