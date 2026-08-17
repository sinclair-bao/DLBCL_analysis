#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""ANTs 子进程封装：单线程环境，避免与 NumPy OpenMP 竞争导致 SIGFPE。"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger("longitudinal.ants")

DEFAULT_ANTS_BIN = Path("/home/sun/ants-2.6.5/bin")

ANTS_ENV: dict[str, str] = {
    **os.environ,
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def resolve_ants_bin(ants_bin: Path | str = DEFAULT_ANTS_BIN) -> tuple[Path, Path, Path]:
    ants_bin = Path(ants_bin)
    registration = ants_bin / "antsRegistration"
    apply_xfm = ants_bin / "antsApplyTransforms"
    convert = ants_bin / "ConvertTransformFile"
    missing = [p.name for p in (registration, apply_xfm) if not p.is_file()]
    if missing:
        raise RuntimeError(f"未找到 ANTs 可执行文件 {missing}，请检查 ants-bin={ants_bin}")
    return registration, apply_xfm, convert


def run_ants(cmd: list[str], label: str) -> None:
    logger.debug("%s: %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=ANTS_ENV)
    if result.stdout:
        logger.debug("%s stdout (末尾):\n%s", label, result.stdout[-2000:])
    if result.returncode != 0:
        logger.debug("%s stderr:\n%s", label, result.stderr)
        raise RuntimeError(
            f"{label} 失败 (exit={result.returncode})\n"
            f"stderr (末尾 2000 字符):\n{result.stderr[-2000:]}"
        )


def is_signal_kill(exc: BaseException) -> bool:
    return bool(re.search(r"exit=(-\d+)", str(exc)))
