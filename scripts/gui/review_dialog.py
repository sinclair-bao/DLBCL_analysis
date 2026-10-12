"""Visual mask and registration review before quantitative analysis."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QSlider, QVBoxLayout,
)

from display_utils import apply_pet_cmap, normalize_ct, slice_axial, slice_coronal, slice_sagittal

SLICES = {"axial": slice_axial, "coronal": slice_coronal, "sagittal": slice_sagittal}
AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}


class QualityReviewDialog(QDialog):
    def __init__(self, *, ct: np.ndarray, pet: np.ndarray | None = None,
                 original_mask: np.ndarray | None = None,
                 edited_mask: np.ndarray | None = None,
                 warped_ct: np.ndarray | None = None,
                 mapped_mask: np.ndarray | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("分割质控" if warped_ct is None else "跨检查配准质控")
        self.resize(1100, 620)
        self.ct, self.pet = ct, pet
        self.original_mask, self.edited_mask = original_mask, edited_mask
        self.warped_ct, self.mapped_mask = warped_ct, mapped_mask
        self.verdict: str | None = None

        self.plane = QComboBox()
        for title, key in (("轴位", "axial"), ("冠状", "coronal"), ("矢状", "sagittal")):
            self.plane.addItem(title, key)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.plane.currentIndexChanged.connect(self._set_plane)
        self.slider.valueChanged.connect(self._draw)

        names = ("当前 CT", "基线 CT 已配准", "CT 棋盘格") if warped_ct is not None else (
            "原始分割", "当前分割", "增删差异")
        images = QHBoxLayout()
        self.views: list[QLabel] = []
        for name in names:
            column = QVBoxLayout()
            column.addWidget(QLabel(name))
            view = QLabel()
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setMinimumSize(300, 300)
            column.addWidget(view, 1)
            images.addLayout(column, 1)
            self.views.append(view)

        self.reviewer = QLineEdit()
        self.reviewer.setPlaceholderText("审阅者姓名或研究编号")
        self.reason = QPlainTextEdit()
        self.reason.setPlaceholderText("记录通过依据，或失败位置与原因")
        self.reason.setMaximumHeight(72)
        approve = QPushButton("质控通过并锁定")
        reject = QPushButton("标记未通过")
        approve.clicked.connect(lambda: self._finish("approved"))
        reject.clicked.connect(lambda: self._finish("rejected"))
        actions = QHBoxLayout()
        actions.addWidget(approve)
        actions.addWidget(reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.plane)
        layout.addWidget(self.slider)
        layout.addLayout(images, 1)
        layout.addWidget(self.reviewer)
        layout.addWidget(self.reason)
        layout.addLayout(actions)
        self._set_plane()

    def _set_plane(self) -> None:
        axis = AXIS[self.plane.currentData()]
        self.slider.setRange(0, self.ct.shape[axis] - 1)
        self.slider.setValue(self.ct.shape[axis] // 2)
        self._draw()

    def _draw(self) -> None:
        plane = self.plane.currentData()
        fn = SLICES[plane]
        index = self.slider.value()
        ct = normalize_ct(fn(self.ct, index), (-160, 240))
        base = np.repeat(ct[..., None], 3, axis=2)
        if self.warped_ct is not None:
            warped = normalize_ct(fn(self.warped_ct, index), (-160, 240))
            other = np.repeat(warped[..., None], 3, axis=2)
            yy, xx = np.indices(ct.shape)
            checker = np.where((((yy // 24) + (xx // 24)) % 2)[..., None] == 0, base, other)
            if self.mapped_mask is not None:
                hit = fn(self.mapped_mask > 0, index)
                checker[hit] = 0.55 * checker[hit] + 0.45 * np.array([0.0, 1.0, 1.0])
            pictures = (base, other, checker)
        else:
            if self.pet is not None:
                pet = fn(self.pet, index)
                base = 0.5 * base + 0.5 * apply_pet_cmap(pet, 0.0, 6.0)
            original = fn(self.original_mask > 0, index) if self.original_mask is not None else np.zeros(ct.shape, bool)
            edited = fn(self.edited_mask > 0, index) if self.edited_mask is not None else np.zeros(ct.shape, bool)
            first, second, difference = (base.copy() for _ in range(3))
            first[original] = 0.5 * first[original] + 0.5 * np.array([1.0, 0.15, 0.15])
            second[edited] = 0.5 * second[edited] + 0.5 * np.array([1.0, 0.15, 0.15])
            add, remove = edited & ~original, original & ~edited
            difference[add] = 0.35 * difference[add] + 0.65 * np.array([0.15, 1.0, 0.2])
            difference[remove] = 0.35 * difference[remove] + 0.65 * np.array([0.85, 0.2, 1.0])
            pictures = (first, second, difference)
        for view, rgb in zip(self.views, pictures):
            pixels = np.ascontiguousarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))
            img = QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                         pixels.strides[0], QImage.Format.Format_RGB888).copy()
            view.setPixmap(QPixmap.fromImage(img).scaled(
                view.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def _finish(self, status: str) -> None:
        if not self.reviewer.text().strip() or not self.reason.toPlainText().strip():
            QMessageBox.warning(self, "质控记录", "请填写审阅者和质控理由。")
            return
        self.verdict = status
        self.accept()
