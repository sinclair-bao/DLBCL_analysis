#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""特征表 + SUVmax / MTV / TLG 随时间折线。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

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
        layout.addWidget(self.table, 2)
        layout.addWidget(self.canvas, 2)
        self._rows: list[dict] = []
        self.clear()

    def clear(self) -> None:
        self._rows = []
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
        self.table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            for c, key in enumerate(DISPLAY_COLS):
                self.table.setItem(r, c, QTableWidgetItem(str(row.get(key, ""))))
        self._draw_lines()

    def _draw_lines(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#cccccc", labelsize=8)
        ax.spines["bottom"].set_color("#555555")
        ax.spines["left"].set_color("#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("value", color="#cccccc", fontsize=8)

        series = {
            ("native_lesion", "suv_max"): ("native SUVmax", "#e85d5d", "o"),
            ("native_lesion", "mtv_ml"): ("native MTV", "#e6b84d", "s"),
            ("native_lesion", "tlg"): ("native TLG", "#7ec8e3", "^"),
            ("baseline_mapped", "suv_max"): ("mapped SUVmax", "#ff9aa2", "o"),
            ("baseline_mapped", "mtv_ml"): ("mapped MTV", "#ffdd9a", "s"),
            ("baseline_mapped", "tlg"): ("mapped TLG", "#b5e2fa", "^"),
        }
        plotted = False
        for (roi, field), (label, color, marker) in series.items():
            pts = []
            for row in self._rows:
                if row.get("roi_type") != roi:
                    continue
                date = row.get("study_date") or ""
                try:
                    val = float(row.get(field) or "nan")
                except ValueError:
                    continue
                if not np.isfinite(val):
                    continue
                pts.append((date, val))
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            xs = np.arange(len(pts))
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker=marker, color=color, label=label, linewidth=1.4)
            ax.set_xticks(xs)
            ax.set_xticklabels([p[0] for p in pts], rotation=30, ha="right", color="#cccccc")
            plotted = True
        if plotted:
            ax.legend(fontsize=7, facecolor="#222222", edgecolor="#444444", labelcolor="#dddddd")
        else:
            ax.set_title("无数值可绘", color="#888888", fontsize=10)
        self.fig.tight_layout(pad=0.4)
        self.canvas.draw_idle()

    def save_plot_png(self, path: Path) -> None:
        self.fig.savefig(str(path), dpi=120, facecolor=self.fig.get_facecolor())
