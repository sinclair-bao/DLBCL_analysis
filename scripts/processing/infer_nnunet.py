#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   infer_nnunet.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    AutoPET nnU-Net 推理阶段：对 export_nnunet.py 产出的扁平命名
    CT/PET 对调用 nnUNetv2_predict_from_modelfolder，将病灶分割
    mask 写入 data/processed/<PatientID>/<StudyDate>/masks/。

    模型：autoPET3_Trainer（LesionTracer，MICCAI 2024 AutoPET III 冠军）
    权重：Dataset222_AutoPETIII_2024/autoPET3_Trainer__...（5-fold 集成）

    Python 解释器须使用 autopet conda 环境：
        /home/sun/miniconda3/envs/autopet/bin/python

@输入 (Input)
    data/nnunet_export/{PatientID}_{StudyDate}_0000.nii.gz   # CT
    data/nnunet_export/{PatientID}_{StudyDate}_0001.nii.gz   # PET

@输出 (Output)
    data/processed/<PatientID>/<StudyDate>/masks/
        {PatientID}_{StudyDate}.nii.gz          # nnU-Net 原始输出（含病灶+器官双头）
        {PatientID}_{StudyDate}_lesion.nii.gz   # 提取的病灶通道（label=1），uint8

@关键执行逻辑
    - discover_cases()：扫描 nnunet_export 目录，找配对的 _0000/_0001。
    - infer_batch()：调用 nnUNetv2_predict_from_modelfolder（子进程方式），
      利用 autopet 环境的 Python/CLI，不依赖当前调用环境。
    - extract_lesion_mask()：从 nnU-Net 输出中提取 label=1 的二值掩码，
      写成 uint8 NIfTI（与 segmentation.py 产出格式兼容）。
    - 增量执行：目标已存在则跳过，除非 overwrite=True。

@用法示例
    # 直接用 autopet 环境 Python 运行
    /home/sun/miniconda3/envs/autopet/bin/python scripts/processing/infer_nnunet.py

    # 或在 autopet 环境激活时
    python scripts/processing/infer_nnunet.py --patient-id 00857723

    # 批量推理全部 export 过的病例
    python scripts/processing/infer_nnunet.py --all

    # 通过 main.py（将自动使用 autopet 环境）
    python main.py --stage infer
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    import nibabel as nib
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 nibabel/numpy，请在 autopet 环境中运行。") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from pipeline_utils import (  # noqa: E402
    StageResult,
    setup_logging,
    summarize,
    write_stage_log_csv,
)

logger = logging.getLogger("infer_nnunet")

STAGE_NAME = "infer"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "data" / "nnunet_export"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

# 模型权重目录（5 个 fold 的上级目录）
DEFAULT_MODEL_FOLDER = (
    PROJECT_ROOT
    / "autoPET"
    / "Dataset222_AutoPETIII_2024"
    / "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3"
)

# autopet 环境中的 nnUNetv2_predict_from_modelfolder 路径
NNUNET_CLI = Path("/home/sun/miniconda3/envs/autopet/bin/nnUNetv2_predict_from_modelfolder")


# ---------------------------------------------------------------------------
# 发现待推理的病例
# ---------------------------------------------------------------------------

def discover_cases(export_root: Path) -> list[tuple[str, str]]:
    """
    扫描 export_root 目录，找到配对的 _0000/_0001 文件对。
    返回 (patient_id, study_date) 列表。
    """
    cases: list[tuple[str, str]] = []
    if not export_root.is_dir():
        return cases
    ct_files = sorted(export_root.glob("*_0000.nii.gz"))
    for ct in ct_files:
        stem = ct.name[: -len("_0000.nii.gz")]       # e.g. 00857723_20180905
        pet = export_root / f"{stem}_0001.nii.gz"
        if not pet.exists():
            logger.warning("CT 有但 PET 缺失，跳过: %s", stem)
            continue
        parts = stem.split("_", 1)
        if len(parts) != 2:
            logger.warning("无法解析 case ID: %s，跳过。", stem)
            continue
        cases.append((parts[0], parts[1]))
    return cases


# ---------------------------------------------------------------------------
# nnU-Net 推理（子进程，隔离 Python 环境）
# ---------------------------------------------------------------------------

def _run_nnunet_predict(
    input_folder: Path,
    output_folder: Path,
    model_folder: Path,
    folds: list[int],
    device: str = "cuda",
    num_proc: int = 2,
) -> tuple[bool, str]:
    """
    调用 nnUNetv2_predict_from_modelfolder。
    返回 (success, message)。
    """
    if not NNUNET_CLI.exists():
        return False, f"CLI 不存在: {NNUNET_CLI}"

    fold_args = " ".join(str(f) for f in folds)
    cmd = (
        f"{NNUNET_CLI} "
        f"-i {input_folder} "
        f"-o {output_folder} "
        f"-m {model_folder} "
        f"-f {fold_args} "
        f"-device {device} "
        f"-npp {num_proc} "
        f"-nps {num_proc} "
        f"--disable_progress_bar"
    )
    logger.debug("运行: %s", cmd)
    try:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "")[-800:]
            return False, f"CLI 退出码 {result.returncode}: {err}"
        return True, "推理完成"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ---------------------------------------------------------------------------
# 提取病灶掩码
# ---------------------------------------------------------------------------

def extract_lesion_mask(seg_path: Path, lesion_path: Path, label_id: int = 1) -> None:
    """
    从 nnU-Net 输出的多标签分割图中提取 label=1（病灶），
    写成 uint8 二值 NIfTI，与 segmentation.py 产出格式一致。
    """
    seg = nib.load(str(seg_path))
    data = np.asarray(seg.dataobj, dtype=np.uint8)
    lesion = (data == label_id).astype(np.uint8)
    out_img = nib.Nifti1Image(lesion, seg.affine, seg.header)
    out_img.header.set_data_dtype(np.uint8)
    lesion_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(lesion_path))


# ---------------------------------------------------------------------------
# 主推理类
# ---------------------------------------------------------------------------

class NnuNetInferrer:
    """对 nnunet_export 中的每个 case 执行 AutoPET nnU-Net 推理。"""

    def __init__(
        self,
        export_root: Path | str,
        processed_root: Path | str,
        model_folder: Path | str,
        overwrite: bool = False,
        folds: list[int] | None = None,
        device: str = "cuda",
        num_proc: int = 2,
    ) -> None:
        self.export_root = Path(export_root)
        self.processed_root = Path(processed_root)
        self.model_folder = Path(model_folder)
        self.overwrite = overwrite
        self.folds = folds or [0, 1, 2, 3, 4]
        self.device = device
        self.num_proc = num_proc

    def infer_case(self, patient_id: str, study_date: str) -> StageResult:
        case_id = f"{patient_id}_{study_date}"
        ct_path = self.export_root / f"{case_id}_0000.nii.gz"
        pet_path = self.export_root / f"{case_id}_0001.nii.gz"

        if not ct_path.exists() or not pet_path.exists():
            return StageResult(
                STAGE_NAME, patient_id, study_date, "error", "",
                f"输入文件不存在: {ct_path} / {pet_path}；请先运行 export_nnunet.py。",
            )

        # 目标路径
        mask_dir = self.processed_root / patient_id / study_date / "masks"
        raw_seg = mask_dir / f"{case_id}.nii.gz"
        lesion_mask = mask_dir / f"{case_id}_lesion.nii.gz"

        if lesion_mask.exists() and not self.overwrite:
            return StageResult(
                STAGE_NAME, patient_id, study_date, "skipped",
                str(lesion_mask), "病灶掩码已存在，overwrite=True 可强制重跑。",
            )

        # nnU-Net 需要单独的输入目录（一次只处理一个 case）
        with tempfile.TemporaryDirectory(prefix=f"nnunet_in_{case_id}_") as tmp_in:
            tmp_in_path = Path(tmp_in)
            # 软链接避免复制大文件
            try:
                (tmp_in_path / ct_path.name).symlink_to(ct_path)
                (tmp_in_path / pet_path.name).symlink_to(pet_path)
            except OSError:
                import shutil
                shutil.copy2(ct_path, tmp_in_path / ct_path.name)
                shutil.copy2(pet_path, tmp_in_path / pet_path.name)

            tmp_out = mask_dir / "_nnunet_raw_out"
            tmp_out.mkdir(parents=True, exist_ok=True)

            ok, msg = _run_nnunet_predict(
                tmp_in_path, tmp_out, self.model_folder,
                self.folds, self.device, self.num_proc,
            )

        if not ok:
            return StageResult(STAGE_NAME, patient_id, study_date, "error", str(mask_dir), msg)

        # 将原始输出（case_id.nii.gz）移到 mask_dir，再提取病灶通道
        raw_out_file = tmp_out / f"{case_id}.nii.gz"
        if not raw_out_file.exists():
            # 尝试找 nnU-Net 实际输出的名称（不带通道后缀）
            candidates = list(tmp_out.glob("*.nii.gz"))
            if not candidates:
                return StageResult(
                    STAGE_NAME, patient_id, study_date, "error", str(mask_dir),
                    f"nnU-Net 输出目录为空: {tmp_out}",
                )
            raw_out_file = candidates[0]

        raw_seg.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.move(str(raw_out_file), str(raw_seg))

        # 清理临时输出目录
        try:
            shutil.rmtree(tmp_out)
        except Exception:  # noqa: BLE001
            pass

        # 提取病灶通道
        try:
            extract_lesion_mask(raw_seg, lesion_mask)
            voxels = int(np.sum(nib.load(str(lesion_mask)).get_fdata() > 0))
        except Exception as exc:  # noqa: BLE001
            return StageResult(
                STAGE_NAME, patient_id, study_date, "error", str(lesion_mask),
                f"病灶掩码提取失败: {exc}",
            )

        return StageResult(
            STAGE_NAME, patient_id, study_date, "ok",
            str(lesion_mask),
            f"推理完成，病灶体素数={voxels}",
        )

    def run(
        self,
        dry_run: bool = False,
        patient_id: Optional[str] = None,
        study_date: Optional[str] = None,
    ) -> list[StageResult]:
        cases = discover_cases(self.export_root)
        if patient_id:
            cases = [c for c in cases if c[0] == patient_id]
        if study_date:
            cases = [c for c in cases if c[1] == study_date]
        logger.info("共发现 %d 个 case 待推理。", len(cases))

        if dry_run:
            for pid, sdate in cases:
                case_id = f"{pid}_{sdate}"
                mask_dir = self.processed_root / pid / sdate / "masks"
                logger.info("[DRY-RUN] %s -> %s", case_id, mask_dir)
            return []

        results: list[StageResult] = []
        for idx, (pid, sdate) in enumerate(cases, start=1):
            logger.info("[%d/%d] 推理 patient=%s study=%s", idx, len(cases), pid, sdate)
            result = self.infer_case(pid, sdate)
            results.append(result)
            if result.status == "error":
                logger.error("失败 patient=%s study=%s: %s", pid, sdate, result.message)
            elif result.status == "warning":
                logger.warning("警告 patient=%s study=%s: %s", pid, sdate, result.message)
        return results


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对 nnunet_export 中的 CT/PET 对执行 AutoPET nnU-Net 推理。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--study-date", default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="使用的 fold 编号（默认全 5-fold 集成）。")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="推理设备。")
    parser.add_argument("--num-proc", type=int, default=2,
                        help="预处理/后处理并行进程数（npp/nps）。")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.verbose)

    inferrer = NnuNetInferrer(
        export_root=args.export_root,
        processed_root=args.processed_root,
        model_folder=args.model_folder,
        overwrite=args.overwrite,
        folds=args.folds,
        device=args.device,
        num_proc=args.num_proc,
    )
    results = inferrer.run(
        dry_run=args.dry_run,
        patient_id=args.patient_id,
        study_date=args.study_date,
    )
    if args.dry_run:
        logger.info("Dry-run 完成，未执行任何推理。")
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_stage_log_csv(results, args.log_dir / f"infer_{timestamp}.csv")
    counts = summarize(results)
    logger.info("推理完成: %s", counts)
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
