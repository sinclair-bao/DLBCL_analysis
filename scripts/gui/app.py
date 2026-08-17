#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File        :   app.py
@Author      :   Sinclair
@Email       :   slbao@ntu.edu.cn

@description
    DLBCL 纵向 PET/CT 分析桌面软件入口。

    conda activate data-analysis
    python scripts/gui/app.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 必须在导入 pyplot / FigureCanvas 之前
import matplotlib

matplotlib.use("QtAgg")

_GUI_DIR = Path(__file__).resolve().parent
_SCRIPTS = _GUI_DIR.parent
_LONG = _SCRIPTS / "longitudinal"
_COMMON = _SCRIPTS / "common"
for _p in (_GUI_DIR, _LONG, _COMMON, _SCRIPTS):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

PROJECT_ROOT = _SCRIPTS.parent
DEFAULT_INTERIM = PROJECT_ROOT / "data" / "interim"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"


def _check_qt() -> None:
    try:
        import PySide6  # noqa: F401
        import pyqtgraph  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "缺少 GUI 依赖。请在 data-analysis 环境中安装：\n"
            "  pip install PySide6 pyqtgraph\n"
            f"原始错误: {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    _check_qt()
    from PySide6.QtWidgets import QApplication

    from main_window import MainWindow

    parser = argparse.ArgumentParser(description="DLBCL 纵向 PET/CT 分析软件")
    parser.add_argument("--interim-root", type=Path, default=DEFAULT_INTERIM)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED)
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    app.setApplicationName("DLBCL Longitudinal")
    win = MainWindow(args.interim_root, args.processed_root)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
