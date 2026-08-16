#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   register_pet_ct.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    在原始分辨率 SUV map 与 CT 上用 ANTs 做刚体配准（PET → CT），保存变换
    矩阵，再把同一变换应用到各向同性（2 mm）重采样后的 PET/CT。

    配准在物理空间（mm）中求解，与体素大小无关。原始图细节更丰富，适合
    估计变换；各向同性图才是后续影像组学的工作空间，因此变换只在原始图
    上估计一次，再复用到 2 mm 网格，避免“先重采样再配准”的双重插值。

    注意：preprocess.py 里 PET 对齐到 CT 网格只用了 NIfTI header（仿射
    重采样），并不校正呼吸运动。本脚本补的是强度驱动的残余刚体偏差。

@输入 (Input)
    data/interim/<PatientID>/<StudyDate>/CT/*.nii.gz          原始 CT
    data/interim/<PatientID>/<StudyDate>/PET/*_SUVbw.nii.gz   原始 SUV
    data/interim/<PatientID>/<StudyDate>/preprocessed/CT/*.nii.gz
    data/interim/<PatientID>/<StudyDate>/preprocessed/PET/*.nii.gz

@输出 (Output)
    data/processed/<PatientID>/<StudyDate>/registration/
        pet_to_ct_0GenericAffine.mat   ANTs 刚体变换（PET → CT）
        pet_to_ct_affine.txt           可读文本形式，便于核查平移/旋转
        pet_orig_warped.nii.gz         原始 SUV 变到原始 CT 空间（质控）
        pet_iso_aligned.nii.gz         2 mm PET 变到 2 mm CT 网格
        ct_iso_reference.nii.gz        2 mm CT 的副本（固定图像，不变换）

@用法示例
    # 三例患者试点（每人取最早一个完整 study）
    python scripts/processing/register_pet_ct.py \\
        --patient-ids 00136597,00500538,00555598 --first-study-only

    python scripts/processing/register_pet_ct.py --dry-run -v
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from pipeline_utils import (  # noqa: E402
    StageResult,
    discover_subject_studies,
    setup_logging,
    summarize,
    write_stage_log_csv,
)

logger = logging.getLogger("register_pet_ct")

STAGE_NAME = "register_pet_ct"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_ANTS_BIN = Path("/home/sun/ants-2.6.5/bin")


class PetCtRegistrar:
    """
    原始 SUV → CT 刚体配准，并把变换应用到 2 mm 各向同性数据。

    Parameters
    ----------
    interim_root : data/interim
    processed_root : data/processed
    ants_bin : ANTs 可执行文件目录（需含 antsRegistration / antsApplyTransforms）
    overwrite : 已有输出时是否重跑
    """

    def __init__(
        self,
        interim_root: Path | str,
        processed_root: Path | str,
        ants_bin: Path | str = DEFAULT_ANTS_BIN,
        overwrite: bool = False,
    ) -> None:
        self.interim_root = Path(interim_root)
        self.processed_root = Path(processed_root)
        self.ants_bin = Path(ants_bin)
        self.overwrite = overwrite

        self.ants_registration = self.ants_bin / "antsRegistration"
        self.ants_apply = self.ants_bin / "antsApplyTransforms"
        self.convert_transform = self.ants_bin / "ConvertTransformFile"
        missing = [
            p.name for p in (self.ants_registration, self.ants_apply) if not p.is_file()
        ]
        if missing:
            raise RuntimeError(
                f"未找到 ANTs 可执行文件 {missing}，请检查 --ants-bin={self.ants_bin}"
            )

    def discover_studies(self) -> list[tuple[str, str, Path]]:
        return discover_subject_studies(self.interim_root)

    @staticmethod
    def _find_original_ct(study_dir: Path) -> Optional[Path]:
        ct_dir = study_dir / "CT"
        if not ct_dir.is_dir():
            return None
        files = sorted(ct_dir.glob("*.nii.gz"))
        if not files:
            return None
        preferred = [p for p in files if "WB_Standard" in p.name]
        return preferred[0] if preferred else files[0]

    @staticmethod
    def _find_original_pet(study_dir: Path) -> Optional[Path]:
        pet_dir = study_dir / "PET"
        if not pet_dir.is_dir():
            return None
        suv = sorted(pet_dir.glob("*_SUVbw.nii.gz"))
        return suv[0] if suv else None

    @staticmethod
    def _find_iso_ct(study_dir: Path) -> Optional[Path]:
        d = study_dir / "preprocessed" / "CT"
        if not d.is_dir():
            return None
        files = sorted(d.glob("*.nii.gz"))
        preferred = [p for p in files if "WB_Standard" in p.name]
        return preferred[0] if preferred else (files[0] if files else None)

    @staticmethod
    def _find_iso_pet(study_dir: Path) -> Optional[Path]:
        d = study_dir / "preprocessed" / "PET"
        if not d.is_dir():
            return None
        suv = sorted(d.glob("*_SUVbw.nii.gz"))
        return suv[0] if suv else None

    # 传给所有 ANTs 子进程的安全环境变量：单线程避免与 NumPy/OpenMP 竞争
    _ANTS_ENV: dict[str, str] = {
        **os.environ,
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }

    def _run(self, cmd: list[str], label: str) -> None:
        logger.debug("%s: %s", label, " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=self._ANTS_ENV
        )
        if result.stdout:
            logger.debug("%s stdout (末尾):\n%s", label, result.stdout[-2000:])
        if result.returncode != 0:
            logger.debug("%s stderr:\n%s", label, result.stderr)
            raise RuntimeError(
                f"{label} 失败 (exit={result.returncode})\n"
                f"stderr (末尾 2000 字符):\n{result.stderr[-2000:]}"
            )

    def _build_registration_cmd(
        self,
        ct: Path,
        pet: Path,
        prefix: str,
        warped: Path,
        *,
        fallback: bool = False,
    ) -> list[str]:
        """
        构造 antsRegistration 命令列表。

        fallback=True 时使用更保守的参数（3 尺度 + float32），专门用于
        在 SIGFPE / 内存崩溃后重试，可规避与 NumPy OpenMP 的线程冲突。
        """
        if fallback:
            convergence   = "[100x50x20,1e-6,10]"
            shrink        = "4x2x1"
            smoothing     = "2x1x0vox"
        else:
            convergence   = "[200x100x50x20,1e-6,10]"
            shrink        = "8x4x2x1"
            smoothing     = "3x2x1x0vox"

        return [
            str(self.ants_registration),
            "--dimensionality",           "3",
            "--float",                    "1",        # float32：省内存、避免极端值溢出
            "--output",                   f"[{prefix},{warped}]",
            "--interpolation",            "Linear",
            "--winsorize-image-intensities", "[0.005,0.995]",
            "--use-histogram-matching",   "0",
            "--initial-moving-transform", f"[{ct},{pet},1]",
            "--transform",                "Rigid[0.1]",
            "--metric",                   f"MI[{ct},{pet},1,32,Regular,0.25]",
            "--convergence",              convergence,
            "--shrink-factors",           shrink,
            "--smoothing-sigmas",         smoothing,
            "--collapse-output-transforms", "1",
            "--verbose",                  "1",
        ]

    def _register_original(self, ct_orig: Path, pet_orig: Path, out_dir: Path) -> Path:
        """
        以原始 CT 为 fixed、原始 SUV 为 moving，估计刚体变换。

        互信息（MI）适合跨模态；关闭 histogram matching。
        初始变换用几何中心，同机 PET/CT 已大致对齐，只需修正残余偏差。

        若 ANTs 因 SIGFPE（NumPy/ITK OpenMP 竞争）崩溃（exit < 0），
        自动以保守参数重试一次。
        """
        prefix = str(out_dir / "pet_to_ct_")
        warped = out_dir / "pet_orig_warped.nii.gz"
        mat    = out_dir / "pet_to_ct_0GenericAffine.mat"

        for attempt, fallback in enumerate([False, True], start=1):
            if attempt > 1:
                logger.warning("  antsRegistration 第 %d 次重试（保守参数）…", attempt)
                # 清理上次可能的残留
                for f in out_dir.glob("pet_to_ct_*"):
                    f.unlink(missing_ok=True)
                warped.unlink(missing_ok=True)

            cmd = self._build_registration_cmd(
                ct_orig, pet_orig, prefix, warped, fallback=fallback
            )
            try:
                self._run(cmd, "antsRegistration")
            except RuntimeError as exc:
                msg = str(exc)
                # 负退出码 = 被信号杀死（SIGFPE=-8, SIGABRT=-6 等）
                m = re.search(r"exit=(-\d+)", msg)
                if m and attempt == 1:
                    logger.warning(
                        "  antsRegistration 被信号终止 (exit=%s)，可能是 OMP 竞争；切换保守参数重试。",
                        m.group(1),
                    )
                    continue
                raise          # 第 2 次仍失败，或非信号错误，直接抛出

            if mat.is_file():
                return mat
            raise RuntimeError(f"配准结束但未找到变换文件: {mat}")

        raise RuntimeError("antsRegistration 两次均失败，已放弃。")

    def _write_transform_txt(self, mat: Path, txt: Path) -> None:
        if not self.convert_transform.is_file():
            logger.warning("未找到 ConvertTransformFile，跳过文本变换写出。")
            return
        self._run(
            [str(self.convert_transform), "3", str(mat), str(txt)],
            "ConvertTransformFile",
        )

    def _apply_to_iso(
        self, pet_iso: Path, ct_iso: Path, transform: Path, out_pet: Path
    ) -> None:
        """把原始空间求得的刚体变换应用到 2 mm PET，参考网格为 2 mm CT。"""
        cmd = [
            str(self.ants_apply),
            "-d", "3",
            "-i", str(pet_iso),
            "-r", str(ct_iso),
            "-t", str(transform),
            "-o", str(out_pet),
            "-n", "Linear",
            "-v", "1",
        ]
        self._run(cmd, "antsApplyTransforms")

    def process_study(self, patient_id: str, study_date: str, study_dir: Path) -> StageResult:
        ct_orig = self._find_original_ct(study_dir)
        pet_orig = self._find_original_pet(study_dir)
        ct_iso = self._find_iso_ct(study_dir)
        pet_iso = self._find_iso_pet(study_dir)

        missing = []
        if ct_orig is None:
            missing.append("原始 CT")
        if pet_orig is None:
            missing.append("原始 SUVbw")
        if ct_iso is None:
            missing.append("2mm CT (preprocessed)")
        if pet_iso is None:
            missing.append("2mm SUVbw (preprocessed)")
        if missing:
            return StageResult(
                STAGE_NAME, patient_id, study_date, "warning", "",
                f"缺少输入，跳过: {', '.join(missing)}",
            )

        out_dir = self.processed_root / patient_id / study_date / "registration"
        mat = out_dir / "pet_to_ct_0GenericAffine.mat"
        pet_iso_out = out_dir / "pet_iso_aligned.nii.gz"
        ct_iso_out = out_dir / "ct_iso_reference.nii.gz"

        if pet_iso_out.exists() and mat.exists() and not self.overwrite:
            return StageResult(
                STAGE_NAME, patient_id, study_date, "skipped", str(out_dir),
                "配准产物已存在，使用 --overwrite 可强制重跑。",
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("  原始配准  PET=%s  ->  CT=%s", pet_orig.name, ct_orig.name)
            mat = self._register_original(ct_orig, pet_orig, out_dir)
            self._write_transform_txt(mat, out_dir / "pet_to_ct_affine.txt")

            logger.info("  应用变换到 2 mm 网格  PET=%s  ref=%s", pet_iso.name, ct_iso.name)
            self._apply_to_iso(pet_iso, ct_iso, mat, pet_iso_out)
            # CT 是固定图像，各向同性 CT 不施加变换，只复制作为参考空间。
            shutil.copy2(ct_iso, ct_iso_out)

            return StageResult(
                STAGE_NAME, patient_id, study_date, "ok", str(out_dir),
                f"刚体配准完成 | transform={mat.name} | iso_pet={pet_iso_out.name}",
            )
        except Exception as exc:  # noqa: BLE001 - 单 Study 失败不影响其他
            logger.exception("配准失败: patient=%s study=%s", patient_id, study_date)
            return StageResult(
                STAGE_NAME, patient_id, study_date, "error", str(out_dir), str(exc)
            )

    def run(
        self,
        dry_run: bool = False,
        patient_ids: Optional[list[str]] = None,
        first_study_only: bool = False,
    ) -> list[StageResult]:
        studies = self.discover_studies()
        if patient_ids:
            wanted = set(patient_ids)
            studies = [s for s in studies if s[0] in wanted]
        if first_study_only:
            seen: set[str] = set()
            kept: list[tuple[str, str, Path]] = []
            for item in studies:
                if item[0] in seen:
                    continue
                seen.add(item[0])
                kept.append(item)
            studies = kept

        logger.info("共 %d 个 (patient, study) 待配准。", len(studies))
        if dry_run:
            for pid, sdate, study_dir in studies:
                logger.info(
                    "[DRY-RUN] patient=%s study=%s  CT=%s  PET=%s  isoCT=%s  isoPET=%s",
                    pid, sdate,
                    self._find_original_ct(study_dir).name if self._find_original_ct(study_dir) else "(无)",
                    self._find_original_pet(study_dir).name if self._find_original_pet(study_dir) else "(无)",
                    self._find_iso_ct(study_dir).name if self._find_iso_ct(study_dir) else "(无)",
                    self._find_iso_pet(study_dir).name if self._find_iso_pet(study_dir) else "(无)",
                )
            return []

        results: list[StageResult] = []
        for idx, (pid, sdate, study_dir) in enumerate(studies, start=1):
            logger.info("[%d/%d] 配准  patient=%s  study=%s", idx, len(studies), pid, sdate)
            result = self.process_study(pid, sdate, study_dir)
            results.append(result)
            if result.status == "error":
                logger.error("失败: %s", result.message)
            elif result.status == "warning":
                logger.warning("跳过: %s", result.message)
        return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在原始 SUV/CT 上用 ANTs 刚体配准，再将变换应用到 2 mm 各向同性数据。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--ants-bin", type=Path, default=DEFAULT_ANTS_BIN)
    parser.add_argument(
        "--patient-ids",
        default=None,
        help="逗号分隔的 PatientID 列表；缺省则处理全部。",
    )
    parser.add_argument(
        "--first-study-only",
        action="store_true",
        help="每个患者只处理时间最早的一个完整 Study（试点用）。",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)

    patient_ids = None
    if args.patient_ids:
        patient_ids = [p.strip() for p in args.patient_ids.split(",") if p.strip()]

    try:
        registrar = PetCtRegistrar(
            interim_root=args.interim_root,
            processed_root=args.processed_root,
            ants_bin=args.ants_bin,
            overwrite=args.overwrite,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    results = registrar.run(
        dry_run=args.dry_run,
        patient_ids=patient_ids,
        first_study_only=args.first_study_only,
    )
    if args.dry_run:
        logger.info("Dry-run 完成，未执行任何实际处理。")
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_stage_log_csv(results, args.log_dir / f"register_pet_ct_{timestamp}.csv")
    counts = summarize(results)
    logger.info("配准完成: %s", counts)
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
