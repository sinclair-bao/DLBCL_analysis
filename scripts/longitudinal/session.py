#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   session.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    每个患者一份纵向会话 JSON：人工指定的 baseline / interim / end 检查日期。
    三个角色均可为空；跨检查映射至少需要 baseline + 一个随访日期。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

ROLES = ("baseline", "interim", "end")
SESSION_FILENAME = "longitudinal_session.json"


@dataclass
class LongitudinalSession:
    patient_id: str
    baseline: Optional[str] = None
    interim: Optional[str] = None
    end: Optional[str] = None

    def as_dict(self) -> dict[str, Optional[str]]:
        return asdict(self)

    def role_of(self, study_date: str) -> Optional[str]:
        for role in ROLES:
            if getattr(self, role) == study_date:
                return role
        return None

    def assigned(self) -> dict[str, str]:
        """角色 → 日期，仅含已指定项。"""
        out: dict[str, str] = {}
        for role in ROLES:
            value = getattr(self, role)
            if value:
                out[role] = value
        return out

    def followups(self) -> list[tuple[str, str]]:
        """(role, study_date) 中非 baseline 的已指定时间点，按日期排序。"""
        items = [(r, d) for r, d in self.assigned().items() if r != "baseline"]
        items.sort(key=lambda x: x[1])
        return items

    def dates_in_order(self) -> list[tuple[str, str]]:
        """按临床角色顺序：baseline → interim → end。"""
        return [(r, self.assigned()[r]) for r in ROLES if r in self.assigned()]

    def can_map(self) -> bool:
        return bool(self.baseline) and bool(self.followups())

    def validate(self, available_dates: list[str]) -> list[str]:
        """返回问题描述列表；空列表表示可用。"""
        issues: list[str] = []
        used: dict[str, str] = {}
        for role, date in self.assigned().items():
            if date not in available_dates:
                issues.append(f"{role}={date} 不在该患者的检查列表中")
            if date in used:
                issues.append(f"{role} 与 {used[date]} 指向同一日期 {date}")
            used[date] = role
        return issues


def session_path(processed_root: Path, patient_id: str) -> Path:
    return Path(processed_root) / patient_id / SESSION_FILENAME


def load_session(processed_root: Path, patient_id: str) -> LongitudinalSession:
    path = session_path(processed_root, patient_id)
    if not path.is_file():
        return LongitudinalSession(patient_id=patient_id)
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return LongitudinalSession(
        patient_id=str(raw.get("patient_id") or patient_id),
        baseline=_empty_to_none(raw.get("baseline")),
        interim=_empty_to_none(raw.get("interim")),
        end=_empty_to_none(raw.get("end")),
    )


def save_session(processed_root: Path, session: LongitudinalSession) -> Path:
    path = session_path(processed_root, session.patient_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(session.as_dict(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def _empty_to_none(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
