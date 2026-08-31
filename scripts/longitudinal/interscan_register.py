#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   interscan_register.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    跨检查 CT→CT 刚体+仿射配准：以随访 2 mm CT 为 fixed、基线 2 mm CT 为
    moving，把基线病灶 mask 用 GenericLabel 拉到随访网格，得到“原始病灶床”。

    不做 SyN。DLBCL 病灶会消退/进展，强变形容易把病灶床拉碎。
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path
from typing import Optional

_SCRIPTS = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _SCRIPTS / "common"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline_utils import (  # noqa: E402
    StageResult,
    setup_logging,
    summarize,
    write_stage_log_csv,
)

from ants_runner import (  # noqa: E402
    DEFAULT_ANTS_BIN,
    is_signal_kill,
    resolve_ants_bin,
    run_ants,
)
from catalog import DataCatalog, StudyAssets  # noqa: E402
from session import load_session  # noqa: E402

logger = logging.getLogger("interscan_register")

STAGE_NAME = "interscan_register"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"


def _mat_name(baseline_date: str) -> str:
    return f"baseline_{baseline_date}_to_this_0GenericAffine.mat"


def _prefix_stem(baseline_date: str) -> str:
    return f"baseline_{baseline_date}_to_this_"


class InterscanRegistrar:
    """基线 CT → 随访 CT 仿射，并映射基线 lesion mask。"""

    def __init__(
        self,
        catalog: DataCatalog,
        ants_bin: Path | str = DEFAULT_ANTS_BIN,
        overwrite: bool = False,
    ) -> None:
        self.catalog = catalog
        self.overwrite = overwrite
        self.ants_registration, self.ants_apply, self.convert_transform = resolve_ants_bin(
            ants_bin
        )

    def map_pair(
        self,
        patient_id: str,
        baseline_date: str,
        followup_date: str,
    ) -> StageResult:
        bl = self.catalog.get_study(patient_id, baseline_date)
        fu = self.catalog.get_study(patient_id, followup_date)
        if bl is None:
            return StageResult(
                STAGE_NAME, patient_id, followup_date, "error", "",
                f"未找到基线检查 {baseline_date}",
            )
        if fu is None:
            return StageResult(
                STAGE_NAME, patient_id, followup_date, "error", "",
                f"未找到随访检查 {followup_date}",
            )
        return self._register_and_warp(bl, fu)

    def map_session(self, patient_id: str) -> list[StageResult]:
        session = load_session(self.catalog.processed_root, patient_id)
        rec = self.catalog.get_patient(patient_id)
        dates = rec.study_dates() if rec else []
        issues = session.validate(dates)
        if issues:
            return [
                StageResult(
                    STAGE_NAME, patient_id, session.baseline or "", "error", "",
                    "; ".join(issues),
                )
            ]
        if not session.can_map():
            return [
                StageResult(
                    STAGE_NAME, patient_id, session.baseline or "", "warning", "",
                    "需要指定 baseline 以及至少一个随访（interim 或 end）",
                )
            ]
        results: list[StageResult] = []
        for _role, fu_date in session.followups():
            results.append(self.map_pair(patient_id, session.baseline, fu_date))
        return results

    def _register_and_warp(self, baseline: StudyAssets, followup: StudyAssets) -> StageResult:
        patient_id = followup.patient_id
        followup_date = followup.study_date
        baseline_date = baseline.study_date
        out_dir = followup.longitudinal_dir()
        mat = out_dir / _mat_name(baseline_date)
        warped_mask = out_dir / "baseline_lesion_warped.nii.gz"

        if (
            not self.overwrite
            and mat.is_file()
            and warped_mask.is_file()
        ):
            return StageResult(
                STAGE_NAME, patient_id, followup_date, "skipped", str(warped_mask),
                "已存在映射结果",
            )

        fixed_ct = followup.working_ct
        moving_ct = baseline.working_ct
        moving_mask = baseline.working_lesion
        if fixed_ct is None:
            return StageResult(
                STAGE_NAME, patient_id, followup_date, "error", str(out_dir),
                "随访缺少 2 mm CT",
            )
        if moving_ct is None:
            return StageResult(
                STAGE_NAME, patient_id, followup_date, "error", str(out_dir),
                "基线缺少 2 mm CT",
            )
        if moving_mask is None:
            return StageResult(
                STAGE_NAME, patient_id, followup_date, "error", str(out_dir),
                "基线缺少 lesion mask",
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._register_ct_ct(fixed_ct, moving_ct, out_dir, baseline_date)
            self._warp_mask(moving_mask, fixed_ct, mat, warped_mask)
        except Exception as exc:
            logger.exception("跨检查配准失败: %s %s → %s", patient_id, baseline_date, followup_date)
            return StageResult(
                STAGE_NAME, patient_id, followup_date, "error", str(out_dir), str(exc)
            )

        txt = out_dir / f"baseline_{baseline_date}_to_this_affine.txt"
        self._write_transform_txt(mat, txt)
        return StageResult(
            STAGE_NAME, patient_id, followup_date, "ok", str(warped_mask),
            f"baseline {baseline_date} → {followup_date}",
        )

    def _build_cmd(
        self,
        fixed: Path,
        moving: Path,
        prefix: str,
        warped: Path,
        *,
        fallback: bool = False,
    ) -> list[str]:
        if fallback:
            conv, shrink, smooth = "[100x50x20,1e-6,10]", "4x2x1", "2x1x0vox"
        else:
            conv, shrink, smooth = "[200x100x50x20,1e-6,10]", "8x4x2x1", "3x2x1x0vox"

        metric = f"MI[{fixed},{moving},1,32,Regular,0.25]"
        return [
            str(self.ants_registration),
            "--dimensionality", "3",
            "--float", "1",
            "--output", f"[{prefix},{warped}]",
            "--interpolation", "Linear",
            "--winsorize-image-intensities", "[0.005,0.995]",
            "--use-histogram-matching", "1",
            "--initial-moving-transform", f"[{fixed},{moving},1]",
            "--transform", "Rigid[0.1]",
            "--metric", metric,
            "--convergence", conv,
            "--shrink-factors", shrink,
            "--smoothing-sigmas", smooth,
            "--transform", "Affine[0.1]",
            "--metric", metric,
            "--convergence", conv,
            "--shrink-factors", shrink,
            "--smoothing-sigmas", smooth,
            "--collapse-output-transforms", "1",
            "--verbose", "1",
        ]

    def _register_ct_ct(
        self, fixed: Path, moving: Path, out_dir: Path, baseline_date: str
    ) -> Path:
        stem = _prefix_stem(baseline_date)
        prefix = str(out_dir / stem)
        warped = out_dir / "baseline_ct_warped.nii.gz"
        mat = out_dir / _mat_name(baseline_date)

        for attempt, fallback in enumerate([False, True], start=1):
            if attempt > 1:
                logger.warning("  antsRegistration 第 %d 次重试（保守参数）…", attempt)
                for f in out_dir.glob(f"{stem}*"):
                    f.unlink(missing_ok=True)
                warped.unlink(missing_ok=True)

            cmd = self._build_cmd(fixed, moving, prefix, warped, fallback=fallback)
            try:
                run_ants(cmd, "antsRegistration")
            except RuntimeError as exc:
                if is_signal_kill(exc) and attempt == 1:
                    logger.warning("  antsRegistration 被信号终止，切换保守参数重试。")
                    continue
                raise
            if mat.is_file():
                return mat
            raise RuntimeError(f"配准结束但未找到变换文件: {mat}")

        raise RuntimeError("antsRegistration 两次均失败，已放弃。")

    def _warp_mask(self, mask: Path, reference: Path, transform: Path, out_mask: Path) -> None:
        cmd = [
            str(self.ants_apply),
            "-d", "3",
            "-i", str(mask),
            "-r", str(reference),
            "-t", str(transform),
            "-o", str(out_mask),
            "-n", "GenericLabel",
            "-v", "1",
        ]
        try:
            run_ants(cmd, "antsApplyTransforms")
        except RuntimeError:
            cmd[-3] = "NearestNeighbor"
            logger.warning("GenericLabel 不可用，改用 NearestNeighbor。")
            run_ants(cmd, "antsApplyTransforms")

    def _write_transform_txt(self, mat: Path, txt: Path) -> None:
        if not self.convert_transform.is_file():
            return
        try:
            run_ants(
                [str(self.convert_transform), "3", str(mat), str(txt)],
                "ConvertTransformFile",
            )
        except RuntimeError as exc:
            logger.warning("写出可读变换失败: %s", exc)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基线 CT → 随访 CT 仿射配准，并映射基线病灶 mask。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--ants-bin", type=Path, default=DEFAULT_ANTS_BIN)
    parser.add_argument("--patient-id", default=None, help="只处理该患者")
    parser.add_argument("--baseline", default=None, help="基线 StudyDate（YYYYMMDD）")
    parser.add_argument("--followup", default=None, help="随访 StudyDate；缺省则用会话 JSON")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)
    catalog = DataCatalog(args.interim_root, args.processed_root)
    try:
        registrar = InterscanRegistrar(catalog, ants_bin=args.ants_bin, overwrite=args.overwrite)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    results: list[StageResult] = []
    if args.patient_id and args.baseline and args.followup:
        if args.dry_run:
            logger.info(
                "[DRY-RUN] %s  %s → %s", args.patient_id, args.baseline, args.followup
            )
            return 0
        results.append(registrar.map_pair(args.patient_id, args.baseline, args.followup))
    elif args.patient_id:
        session = load_session(catalog.processed_root, args.patient_id)
        if args.dry_run:
            logger.info("[DRY-RUN] session %s", session.as_dict())
            return 0
        results.extend(registrar.map_session(args.patient_id))
    else:
        ids = catalog.patient_ids()
        logger.info("扫描 %d 名患者的会话 JSON…", len(ids))
        for pid in ids:
            session = load_session(catalog.processed_root, pid)
            if not session.can_map():
                continue
            if args.dry_run:
                logger.info("[DRY-RUN] %s %s", pid, session.as_dict())
                continue
            results.extend(registrar.map_session(pid))
        if args.dry_run:
            return 0

    if not results:
        logger.info("没有需要处理的映射任务（请先在 GUI 中指定时间点）。")
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_stage_log_csv(results, args.log_dir / f"interscan_register_{timestamp}.csv")
    counts = summarize(results)
    logger.info("跨检查映射完成: %s", counts)
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
