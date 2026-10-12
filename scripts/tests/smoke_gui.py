"""Offscreen construction check; run with QT_QPA_PLATFORM=offscreen."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

import nibabel as nib
import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "gui", ROOT / "longitudinal", ROOT / "common", ROOT):
    sys.path.insert(0, str(folder))

from main_window import MainWindow  # noqa: E402
from review_dialog import QualityReviewDialog  # noqa: E402
from segment_editor import SegmentEditorDialog  # noqa: E402
from features import compute_roi_features, write_csv  # noqa: E402
from quality import fingerprint, mask_review_status, review_mask  # noqa: E402
from render_backend import gpu_available  # noqa: E402
from session import LongitudinalSession, save_session  # noqa: E402
from workers import SegmentWorker  # noqa: E402


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
            writer.writerow({"patient_id": "P1", "study_date": "2026-06-01",
                             "ct_path": "ct.nii.gz", "pet_path": "pet.nii.gz",
                             "pet_units": "SUVbw", "lesion_mask_path": "mask.nii.gz"})
        save_session(root / "outputs", LongitudinalSession(
            "P1", baseline="2026-01-01", interim="2026-06-01"))
        win = MainWindow(root, root / "outputs", manifest_path=manifest)
        win.showNormal()
        for width in (1024, 1366):
            win.resize(width, 768)
            app.processEvents()
            assert win._right_panel.width() >= 320
            viewport = win._right_panel.viewport()
            buttons = (win.timepoints.btn_map, win.timepoints.btn_feat,
                       win.timepoints.btn_review_map, win.edit_panel.btn_autopet,
                       win.edit_panel.btn_save, win.edit_panel.btn_review,
                       win.edit_panel.btn_link, win.edit_panel.btn_dilate,
                       win.edit_panel.btn_close)
            for button in buttons:
                edge = button.mapTo(viewport, QPoint(button.width(), 0)).x()
                assert edge <= viewport.width(), (width, button.text(), edge,
                                                  viewport.width())
            control_scroll = win.ortho._control_scroll
            content = control_scroll.widget()
            controls = (win.ortho.slider_k, win.ortho.radio_fusion,
                        win.ortho.chk_native, win.ortho.spin_wl,
                        win.ortho.combo_render)
            centers = [widget.mapTo(content, QPoint(0, widget.height() // 2)).y()
                       for widget in controls]
            assert max(centers) - min(centers) <= 2, centers
            assert all(0 <= widget.mapTo(control_scroll.viewport(),
                                         QPoint(0, widget.height() // 2)).y()
                       < control_scroll.viewport().height() for widget in controls)
            assert control_scroll.horizontalScrollBar().maximum() > 0
            control_scroll.ensureWidgetVisible(win.ortho.combo_render)
            app.processEvents()
            render_x = win.ortho.combo_render.mapTo(control_scroll.viewport(), QPoint(0, 0)).x()
            assert 0 <= render_x < control_scroll.viewport().width()
        win._set_browser_visible(False)
        app.processEvents()
        assert win._right_panel.width() >= 320
        tall = win.evolution._views[0]
        tall.set_rgb(np.zeros((100, 20, 3), dtype=np.float32))
        app.processEvents()
        x_range, y_range = tall.box.viewRange()
        assert x_range[0] <= 1e-6 and x_range[1] >= 20 - 1e-6, x_range
        assert y_range[0] <= 1e-6 and y_range[1] >= 100 - 1e-6, y_range
        win._on_patient("P1")
        deadline = monotonic() + 5
        while (win._prefetch_worker is not None or win._prefetch_pending) and monotonic() < deadline:
            app.processEvents()
            sleep(0.01)
        app.processEvents()
        assert win.evolution._views[1].item.image is not None
        current_mip = win.evolution._views[0]
        other_mip = win.evolution._views[1]
        assert current_mip.vline.isVisible() and current_mip.hline.isVisible()
        assert not other_mip.vline.isVisible()
        win.ortho._set_ijk(3, 4, 5)
        assert current_mip.vline.value() == 12
        assert current_mip.hline.value() == 10
        win.ortho.slider_i.setValue(4)
        assert current_mip.vline.value() == 11
        win.ortho.chk_crosshair.setChecked(False)
        assert not current_mip.vline.isVisible()
        win.ortho.chk_crosshair.setChecked(True)
        assert current_mip.vline.isVisible()
        win._on_study("P1", "2026-06-01")
        assert not current_mip.vline.isVisible() and other_mip.vline.isVisible()
        win._on_study("P1", "2026-01-01")
        if gpu_available():
            win.ortho.combo_render.setCurrentIndex(win.ortho.combo_render.findData("gpu"))
            first_gpu_ct = win.ortho._gpu._ct
            win._on_study("P1", "2026-06-01")
            win._on_study("P1", "2026-01-01")
            assert win.ortho._gpu._ct is first_gpu_ct
            win.ortho.volumes().native[8, 8, 8] = 2
            win.ortho.mark_masks_dirty()
            win.ortho.refresh(True)
            assert win.ortho._gpu._native.get()[8, 8, 8] == 2
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
        win.features.load_csv(feature_csv)
        assert len(win.features.fig.axes) == 3
        session = LongitudinalSession("P1", baseline="2026-01-01")
        assert win._feature_csv_current(feature_csv, "P1", session)
        nib.save(nib.Nifti1Image(pet + 1, np.eye(4)), str(root / "pet.nii.gz"))
        assert not win._feature_csv_current(feature_csv, "P1", session)
        win._refresh()
        assert win.ortho.volumes() is not vol
        assert win.ortho.volumes().pet[7, 7, 7] == 9.0
        original_sha = fingerprint(root / "mask.nii.gz")["sha256"]
        edited_path = root / "outputs" / "P1" / "2026-01-01" / "masks" / "P1_2026-01-01_lesion_edited.nii.gz"

        def paint_and_accept(dialog):
            from display_utils import voxel_to_display

            x, y = voxel_to_display("axial", 11, 11, dialog._k, dialog._vol.ct.shape)
            dialog._paint(x, y, False)
            dialog._end_stroke()
            return dialog.DialogCode.Accepted

        with patch.object(SegmentEditorDialog, "exec", paint_and_accept):
            win._open_segment_editor()
        assert edited_path.is_file()
        assert fingerprint(root / "mask.nii.gz")["sha256"] == original_sha
        assert win.catalog.get_study("P1", "2026-01-01").working_lesion == edited_path
        assert mask_review_status(edited_path) == "pending"
        first_saved = np.asanyarray(nib.load(str(edited_path)).dataobj)
        assert np.array_equal(first_saved, win.ortho.volumes().native)
        versions = edited_path.parent / "versions"
        assert len(list(versions.glob("*.nii.gz"))) == 1
        review_mask(edited_path, "R1", "Checked all planes", "approved")
        with patch.object(SegmentEditorDialog, "exec", lambda dialog: dialog.DialogCode.Accepted):
            win._open_segment_editor()
        assert len(list(versions.glob("*.nii.gz"))) == 1
        assert mask_review_status(edited_path) == "approved"
        changed = first_saved.copy()
        changed[11, 11, 8] = 0
        win.ortho.apply_mask(changed)
        win._save_edited()
        assert len(list(versions.glob("*.nii.gz"))) == 3
        assert mask_review_status(edited_path) == "pending"
        win._drop_cached(patient_id="P1", study_date="2026-01-01")
        win.catalog.refresh()
        win._on_study("P1", "2026-01-01")
        assert np.array_equal(win.ortho.volumes().native,
                              np.asanyarray(nib.load(str(edited_path)).dataobj))

        threshold_editor = SegmentEditorDialog(win.ortho.volumes(), require_change=True)
        with patch("segment_editor.QMessageBox.warning") as warning:
            threshold_editor._accept_edit()
        assert warning.called and threshold_editor.result() != threshold_editor.DialogCode.Accepted
        threshold_editor.radio_thr_rel.setChecked(True)
        assert threshold_editor._vol.dirty
        threshold_editor._accept_edit()
        assert threshold_editor.result() == threshold_editor.DialogCode.Accepted
        threshold_editor.close()
        interim = root / "data" / "interim"
        patient_dir = interim / "Q1"
        (patient_dir / "20260101" / "preprocessed" / "CT").mkdir(parents=True)
        (patient_dir / "20260101" / "preprocessed" / "CT" / "ct.nii.gz").touch()
        (interim / "Q2" / "20260101" / "preprocessed" / "CT").mkdir(parents=True)
        with patch("main_window.QFileDialog.getExistingDirectory", return_value=str(patient_dir)):
            win._open_data_directory()
        assert win.catalog.patient_ids() == ["Q1"]
        assert win.catalog.processed_root == root / "data" / "processed"
        assert win.current_patient is None and win.ortho.volumes() is None
        assert win.browser.topLevelItemCount() == 1
        fake_python = root / "autopet python"
        fake_python.touch()
        calls = []

        def fake_run(cmd, label):
            calls.append(cmd)
            if "export_nnunet.py" in cmd[1]:
                export = root / "data" / "nnunet_export"
                export.mkdir()
                for channel in ("0000", "0001"):
                    (export / f"Q1_20260101_{channel}.nii.gz").touch()

        worker = SegmentWorker("autopet", "Q1", "20260101", win.catalog)
        with patch("workers.AUTOPET_PYTHON", fake_python), patch.object(worker, "_run_cmd", fake_run):
            worker._run_autopet()
        assert len(calls) == 2
        assert str(interim) in calls[0] and "--overwrite" in calls[0]
        assert str(root / "data" / "processed") in calls[1] and "--overwrite" in calls[1]
        with patch("main_window.QFileDialog.getOpenFileName", return_value=(str(manifest), "CSV")):
            win._import_manifest()
        assert win.catalog.patient_ids() == ["P1"]
        assert win.catalog.processed_root == root / "studies_review_outputs"
        assert win.browser.topLevelItemCount() == 1
        non_suv = root / "non_suv.csv"
        with non_suv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["patient_id", "study_date", "ct_path",
                                                       "pet_path", "pet_units"])
            writer.writeheader()
            writer.writerow({"patient_id": "P2", "study_date": "2026-01-01",
                             "ct_path": "ct.nii.gz", "pet_path": "pet.nii.gz",
                             "pet_units": "Bq/mL"})
        with patch("main_window.QFileDialog.getOpenFileName", return_value=(str(non_suv), "CSV")):
            win._import_manifest()
        win._on_patient("P2")
        assert win.ortho.volumes().pet_units == "Bq/mL"
        assert win.ortho.lbl_pet_unit.text() == "PET (Bq/mL)"
        deadline = monotonic() + 5
        while (win._prefetch_worker is not None or win._prefetch_pending) and monotonic() < deadline:
            app.processEvents()
            sleep(0.01)
        assert win.evolution._views[0].item.image is not None
        assert win.evolution._views[0].vline.isVisible()
        with (patch("main_window.QMessageBox.warning") as warning,
              patch.object(win, "_open_segment_editor") as open_editor):
            win._on_segment("threshold")
        assert warning.called and not open_editor.called
        non_suv_editor = SegmentEditorDialog(win.ortho.volumes())
        assert not non_suv_editor.radio_thr_rel.isEnabled()
        assert not non_suv_editor.radio_thr_abs.isEnabled()
        non_suv_editor.close()
        empty = root / "empty"
        empty.mkdir()
        with (patch("main_window.QFileDialog.getExistingDirectory", return_value=str(empty)),
              patch("main_window.QMessageBox.warning") as warning):
            win._open_data_directory()
        assert warning.called
        assert win.catalog.manifest_path == non_suv
        editor.close()
        mask_review.close()
        reg_review.close()
        win.close()
    app.quit()


if __name__ == "__main__":
    main()
