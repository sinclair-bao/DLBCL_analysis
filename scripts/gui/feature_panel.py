#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""特征表 + SUVmax / MTV / TLG 随时间折线。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_CJK_FONTS = ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "Microsoft YaHei",
              "PingFang SC", "SimHei", "Noto Sans CJK JP")
_available_fonts = {font.name for font in font_manager.fontManager.ttflist}
for _font in _CJK_FONTS:
    if _font in _available_fonts:
        rcParams["font.sans-serif"] = [_font, *rcParams["font.sans-serif"]]
        rcParams["font.family"] = "sans-serif"
        rcParams["axes.unicode_minus"] = False
        break

DISPLAY_COLS = [
    "study_date",
    "role",
    "roi_type",
    "suv_max",
    "suv_mean",
    "suv_peak",
    "mtv_ml",
    "tlg",
    "liver_suv_mean",
    "suv_max_liver_ratio",
]


class FeaturePanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.table = QTableWidget(0, len(DISPLAY_COLS))
        self.table.setHorizontalHeaderLabels(DISPLAY_COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        self.fig = Figure(figsize=(4.5, 2.8), facecolor="#1a1a1a")
        self.canvas = FigureCanvasQTAgg(self.fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel("尚无特征")
        layout.addWidget(self.status)
        layout.addWidget(self.table, 2)
        layout.addWidget(self.canvas, 2)
        self._rows: list[dict] = []
        self.clear()

    def clear(self, message: str = "尚无特征") -> None:
        self._rows = []
        self.status.setText(message)
        self.table.setRowCount(0)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#aaaaaa")
        ax.set_title("尚无特征", color="#888888", fontsize=10)
        self.canvas.draw_idle()

    def load_csv(self, path: Path) -> None:
        if not path.is_file():
            self.clear()
            return
        with path.open("r", encoding="utf-8") as fh:
            self._rows = list(csv.DictReader(fh))
        self.status.setText("特征结果：已核对 mask 与映射版本")
        self.table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            for c, key in enumerate(DISPLAY_COLS):
                self.table.setItem(r, c, QTableWidgetItem(str(row.get(key, ""))))
        self._draw_lines()

    def _draw_lines(self) -> None:
        self.fig.clear()
        dates = sorted({row.get("study_date", "") for row in self._rows
                        if row.get("study_date")})
        if not dates:
            self.clear("无数值可绘")
            return
        positions = {date: i for i, date in enumerate(dates)}
        metrics = (("suv_max", "SUVmax"), ("mtv_ml", "MTV (mL)"), ("tlg", "TLG"))
        axes = self.fig.subplots(3, 1, sharex=True)
        for ax, (field, title) in zip(axes, metrics):
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#cccccc", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#555555")
            ax.set_ylabel(title, color="#cccccc", fontsize=8)
            for roi, label, color, marker in (
                ("native_lesion", "当前病灶", "#e85d5d", "o"),
                ("baseline_mapped", "基线映射", "#7ec8e3", "s"),
            ):
                points = []
                for row in self._rows:
                    if row.get("roi_type") != roi or row.get("study_date") not in positions:
                        continue
                    try:
                        value = float(row.get(field) or "nan")
                    except ValueError:
                        continue
                    if np.isfinite(value):
                        points.append((positions[row["study_date"]], value))
                if points:
                    points.sort()
                    ax.plot([p[0] for p in points], [p[1] for p in points],
                            marker=marker, color=color, label=label, linewidth=1.3)
            if ax.lines:
                ax.legend(fontsize=7, facecolor="#222222", edgecolor="#444444",
                          labelcolor="#dddddd", loc="upper right")
        axes[-1].set_xticks(range(len(dates)))
        axes[-1].set_xticklabels(dates, rotation=30, ha="right", color="#cccccc")
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def save_plot_png(self, path: Path) -> None:
        self.fig.savefig(str(path), dpi=120, facecolor=self.fig.get_facecolor())
