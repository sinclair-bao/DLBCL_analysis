#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""不确定进度的等待对话框。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressDialog, QWidget


def make_progress(parent: Optional[QWidget], title: str, label: str) -> QProgressDialog:
    dlg = QProgressDialog(label, None, 0, 0, parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setCancelButton(None)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    dlg.show()
    QApplication.processEvents()
    return dlg


@contextmanager
def busy_progress(parent: Optional[QWidget], title: str, label: str) -> Iterator[QProgressDialog]:
    dlg = make_progress(parent, title, label)
    try:
        yield dlg
    finally:
        dlg.close()
