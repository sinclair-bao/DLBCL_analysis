"""Offscreen construction check; run with QT_QPA_PLATFORM=offscreen."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "gui", ROOT / "longitudinal", ROOT / "common", ROOT):
    sys.path.insert(0, str(folder))

from main_window import MainWindow  # noqa: E402
from review_dialog import QualityReviewDialog  # noqa: E402
from segment_editor import SegmentEditorDialog  # noqa: E402
from features import compute_roi_features, write_csv  # noqa: E402
from quality import review_mask  # noqa: E402
from session import LongitudinalSession  # noqa: E402


def main() -> None:
    app = QApplication([])
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        ct = np.zeros((16, 16, 16), dtype=np.float32)
        pet = np.zeros_like(ct)
        pet[7, 7, 7] = 8.0
        mask = np.zeros_like(ct, dtype=np.uint8)
        mask[7, 7, 7] = 1
        for name, data in (("ct.nii.gz", ct), ("pet.nii.gz", pet), ("mask.nii.gz", mask)):
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(root / name))
        manifest = root / "studies.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["patient_id", "study_date", "ct_path",
                                                       "pet_path", "pet_units", "lesion_mask_path"])
            writer.writeheader()
            writer.writerow({"patient_id": "P1", "study_date": "2026-01-01",
                             "ct_path": "ct.nii.gz", "pet_path": "pet.nii.gz",
                             "pet_units": "SUVbw", "lesion_mask_path": "mask.nii.gz"})
        win = MainWindow(root, root / "outputs", manifest_path=manifest)
        win._on_patient("P1")
        vol = win.ortho.volumes()
        assert vol is not None and vol.ct.shape == ct.shape
        editor = SegmentEditorDialog(vol, reference_mask=vol.native)
        editor.combo_compare.setCurrentIndex(2)
        editor.refresh()
        mask_review = QualityReviewDialog(ct=vol.ct, pet=vol.pet,
                                          original_mask=vol.native, edited_mask=vol.native)
        reg_review = QualityReviewDialog(ct=vol.ct, warped_ct=vol.ct, mapped_mask=vol.native)
        assert mask_review.views[0].pixmap() is not None
        assert reg_review.views[2].pixmap() is not None
        assets = win.catalog.get_study("P1", "2026-01-01")
        review_mask(assets.working_lesion, "R1", "All planes checked", "approved")
        row = compute_roi_features(assets, assets.working_lesion, "native_lesion",
                                   "baseline", include_radiomics=False)
        feature_csv = root / "outputs" / "P1" / "longitudinal_features.csv"
        write_csv([row], feature_csv)
        session = LongitudinalSession("P1", baseline="2026-01-01")
        assert win._feature_csv_current(feature_csv, "P1", session)
        nib.save(nib.Nifti1Image(pet + 1, np.eye(4)), str(root / "pet.nii.gz"))
        assert not win._feature_csv_current(feature_csv, "P1", session)
        editor.close()
        mask_review.close()
        reg_review.close()
        win.close()
    app.quit()


if __name__ == "__main__":
    main()
