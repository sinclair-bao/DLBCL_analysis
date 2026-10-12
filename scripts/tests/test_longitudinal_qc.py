"""Synthetic checks for spatial alignment, version review and mapping invalidation."""

from __future__ import annotations

import sys
import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "longitudinal"))
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "processing"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "visualization"))
sys.path.insert(0, str(ROOT / "processing" / "organ_extraction"))

from catalog import DataCatalog, StudyAssets, resolve_data_directory  # noqa: E402
from display_utils import apply_pet_cmap, coronal_mip, normalize_ct  # noqa: E402
from export_nnunet import _find_preprocessed_ct, _find_preprocessed_pet  # noqa: E402
from features import _suv_peak, compute_roi_features  # noqa: E402
from infer_nnunet import NnuNetInferrer, _run_nnunet_predict  # noqa: E402
from preprocess import ImagePreprocessor  # noqa: E402
from pacs_dicom_to_nifti_suv import PacsDicomToNiftiSuvConverter, SeriesJob  # noqa: E402
from qc_segmentation import _load_pet_mask  # noqa: E402
from organ_segmentation import OrganSegmentor  # noqa: E402
from mask_ops import ensure_labeled, lesion_stats, threshold_pet_mask  # noqa: E402
from segmentation import ThresholdSegmenter, threshold_suv_mask  # noqa: E402
from volume_io import VolumeSet, load_mask_from_path  # noqa: E402
from quality import (archive_existing_mask, mapping_status, mask_review_status, record_mask_version,
                     review_mapping, review_mask, write_mapping_manifest)  # noqa: E402
from session import LongitudinalSession  # noqa: E402


class LongitudinalQcTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_data_directory_selection_and_patient_filter(self) -> None:
        interim = self.root / "data" / "interim"
        for patient in ("P1", "P2"):
            (interim / patient / "20260101" / "preprocessed" / "CT").mkdir(parents=True)
        processed = self.root / "data" / "processed"
        for selected in (self.root / "data", interim):
            self.assertEqual(resolve_data_directory(selected), (interim, processed, None))
        self.assertEqual(
            resolve_data_directory(interim / "P1"), (interim, processed, "P1")
        )
        custom_patient = self.root / "custom" / "P3"
        (custom_patient / "T1" / "PET").mkdir(parents=True)
        self.assertEqual(resolve_data_directory(custom_patient),
                         (custom_patient.parent, custom_patient.parent / "processed", "P3"))
        catalog = DataCatalog(interim, processed, patient_filter="P1")
        self.assertEqual(catalog.patient_ids(), ["P1"])
        self.assertEqual(catalog.get_patient("P1").study_dates(), ["20260101"])
        self.assertIsNone(catalog.get_patient("P2"))

    def test_binary_255_mask_splits_disconnected_lesions(self) -> None:
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[1, 1, 1] = 255
        mask[6, 6, 6] = 255
        labeled = ensure_labeled(mask)
        self.assertEqual(set(np.unique(labeled)), {0, 1, 2})
        stable = np.zeros_like(labeled)
        stable[1, 1, 1] = 2
        self.assertEqual(int(ensure_labeled(stable)[1, 1, 1]), 2)

    def test_import_rejects_fractional_and_negative_mask_labels(self) -> None:
        zeros = np.zeros((4, 4, 4), dtype=np.float32)
        vol = VolumeSet(zeros, zeros, np.zeros_like(zeros, dtype=np.uint16),
                        None, "T1", affine=np.eye(4))
        for value in (0.5, -1.0, np.nan):
            mask = zeros.copy()
            mask[1, 1, 1] = value
            path = self._image("invalid_mask.nii.gz", mask, np.eye(4))
            with self.assertRaises(ValueError):
                load_mask_from_path(path, vol)

    def test_offline_threshold_and_suv_peak_handle_nonfinite_pet(self) -> None:
        self.assertEqual(int(threshold_suv_mask(np.zeros((4, 4, 4)), "relative", 0.41).sum()), 0)
        pet = np.zeros((16, 16, 16), dtype=np.float32)
        pet[8, 8, 8] = np.nan
        pet[9, 8, 8] = 9.0
        mask = np.zeros_like(pet, dtype=bool)
        mask[8:10, 8, 8] = True
        self.assertTrue(np.isfinite(_suv_peak(pet, np.eye(4), mask)))

    def test_nonfinite_image_values_remain_displayable(self) -> None:
        image = np.array([[np.nan, np.inf], [-np.inf, 3.0]], dtype=np.float32)
        self.assertTrue(np.isfinite(apply_pet_cmap(image)).all())
        self.assertTrue(np.isfinite(normalize_ct(image)).all())
        self.assertTrue(np.isfinite(coronal_mip(image[:, :, None])).all())

    def test_nnunet_command_preserves_paths_with_spaces(self) -> None:
        cli = self.root / "model cli"
        cli.touch()
        with patch("infer_nnunet.NNUNET_CLI", cli), patch("infer_nnunet.subprocess.run") as run:
            run.return_value.returncode = 0
            ok, _ = _run_nnunet_predict(self.root / "input scans",
                                        self.root / "output masks", self.root / "model weights",
                                        [0, 1])
        self.assertTrue(ok)
        cmd = run.call_args.args[0]
        self.assertIn(str(self.root / "input scans"), cmd)
        self.assertIn(str(self.root / "model weights"), cmd)

    def test_model_export_and_threshold_require_suvbw(self) -> None:
        study = self.root / "P1" / "T1"
        ct_dir = study / "preprocessed" / "CT"
        pet_dir = study / "preprocessed" / "PET"
        ct_dir.mkdir(parents=True)
        pet_dir.mkdir(parents=True)
        (ct_dir / "a_other.nii.gz").touch()
        preferred = ct_dir / "z_WB_Standard.nii.gz"
        preferred.touch()
        (pet_dir / "scan_ACT.nii.gz").touch()
        self.assertEqual(_find_preprocessed_ct(study), preferred)
        self.assertIsNone(_find_preprocessed_pet(study))
        self.assertIsNone(ThresholdSegmenter._find_preprocessed_pet(study))
        suv = pet_dir / "scan_SUVbw.nii.gz"
        suv.touch()
        self.assertEqual(_find_preprocessed_pet(study), suv)
        self.assertEqual(ThresholdSegmenter._find_preprocessed_pet(study), suv)

    def test_preprocess_prefers_same_ct_series_as_export_and_gui(self) -> None:
        study = self.root / "study"
        ct_dir = study / "CT"
        ct_dir.mkdir(parents=True)
        other = ct_dir / "a_soft.nii.gz"
        preferred = ct_dir / "z_WB_Standard.nii.gz"
        other.touch()
        preferred.touch()
        processor = ImagePreprocessor(self.root)
        self.assertEqual(processor._find_input_files(study, "CT"), [preferred, other])
        self.assertEqual(OrganSegmentor.__new__(OrganSegmentor)._find_original_ct(study), preferred)

    def test_split_pet_series_cannot_silently_lose_coverage(self) -> None:
        converter = PacsDicomToNiftiSuvConverter(self.root / "interim")
        job = SeriesJob(self.root / "dicom", "P1", "20260101", "PET", "uid", "1", "PET")
        with (patch.object(converter, "read_series_header", return_value=object()),
              patch.object(converter, "_run_dcm2niix", return_value=[Path("a.nii.gz"), Path("b.nii.gz")])):
            with self.assertRaisesRegex(RuntimeError, "拆分为 2 个文件"):
                converter._convert_pet_series(job)
        self.assertFalse((self.root / "interim" / "P1" / "20260101" / "PET" / "s1_PET_SUVbw.nii.gz").exists())

    def test_qc_requires_suv_and_matching_mask_geometry(self) -> None:
        interim = self.root / "interim"
        processed = self.root / "processed"
        pet_dir = interim / "P1" / "T1" / "preprocessed" / "PET"
        mask_dir = processed / "P1" / "T1" / "masks"
        pet_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        data = np.ones((4, 4, 4), dtype=np.float32)
        mask_path = mask_dir / "P1_T1_lesion.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(mask_path))
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(pet_dir / "scan_ACT.nii.gz"))
        self.assertEqual(_load_pet_mask(interim, processed, "P1", "T1"), (None, None))
        shifted = np.eye(4)
        shifted[0, 3] = 2.0
        suv_path = pet_dir / "scan_SUVbw.nii.gz"
        nib.save(nib.Nifti1Image(data, shifted), str(suv_path))
        self.assertEqual(_load_pet_mask(interim, processed, "P1", "T1"), (None, None))

    def test_nnunet_rerun_ignores_stale_raw_output(self) -> None:
        export = self.root / "exports"
        export.mkdir()
        for channel in ("0000", "0001"):
            (export / f"P1_T1_{channel}.nii.gz").touch()
        masks = self.root / "processed" / "P1" / "T1" / "masks"
        stale = masks / "_nnunet_raw_out"
        stale.mkdir(parents=True)
        (stale / "old.nii.gz").write_bytes(b"stale")

        def predict(_input, output, *_args):
            array = np.zeros((4, 4, 4), dtype=np.uint8)
            array[1, 1, 1] = 1
            nib.save(nib.Nifti1Image(array, np.eye(4)), str(output / "P1_T1.nii.gz"))
            return True, "ok"

        inferrer = NnuNetInferrer(export, self.root / "processed", self.root / "model",
                                 overwrite=True)
        with patch("infer_nnunet._run_nnunet_predict", side_effect=predict):
            result = inferrer.infer_case("P1", "T1")
        self.assertEqual(result.status, "ok")
        self.assertEqual(int(np.asanyarray(nib.load(str(masks / "P1_T1_lesion.nii.gz")).dataobj).sum()), 1)
        self.assertEqual(len(list(masks.glob("_nnunet_raw_*"))), 1)  # only the pre-existing legacy folder

    def test_nnunet_invalid_output_preserves_prior_lesion_mask(self) -> None:
        export = self.root / "exports"
        export.mkdir()
        for channel in ("0000", "0001"):
            (export / f"P1_T1_{channel}.nii.gz").touch()
        masks = self.root / "processed" / "P1" / "T1" / "masks"
        masks.mkdir(parents=True)
        prior = masks / "P1_T1_lesion.nii.gz"
        prior.write_bytes(b"prior model mask")

        def predict(_input, output, *_args):
            invalid = np.zeros((4, 4, 4), dtype=np.float32)
            invalid[1, 1, 1] = np.nan
            nib.save(nib.Nifti1Image(invalid, np.eye(4)), str(output / "P1_T1.nii.gz"))
            return True, "ok"

        inferrer = NnuNetInferrer(export, self.root / "processed", self.root / "model",
                                 overwrite=True)
        with patch("infer_nnunet._run_nnunet_predict", side_effect=predict):
            result = inferrer.infer_case("P1", "T1")
        self.assertEqual(result.status, "error")
        self.assertEqual(prior.read_bytes(), b"prior model mask")

    def _image(self, name: str, data: np.ndarray, affine: np.ndarray) -> Path:
        path = self.root / name
        nib.save(nib.Nifti1Image(data, affine), str(path))
        return path

    def test_same_shape_different_affine_resamples_mask(self) -> None:
        pet = np.zeros((4, 4, 4), dtype=np.float32)
        pet[2, 1, 1] = 10.0
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[1, 1, 1] = 1
        shifted = np.eye(4)
        shifted[0, 3] = 1.0
        ct_path = self._image("ct.nii.gz", np.zeros_like(pet), np.eye(4))
        pet_path = self._image("pet.nii.gz", pet, np.eye(4))
        mask_path = self._image("mask.nii.gz", mask, shifted)
        study = StudyAssets("P1", "T1", self.root, self.root / "processed" / "P1" / "T1",
                            ct_reference=ct_path, pet_aligned=pet_path, lesion_auto=mask_path)
        row = compute_roi_features(study, mask_path, "native_lesion", "baseline",
                                   include_radiomics=False)
        self.assertEqual(row["suv_max"], 10.0)
        self.assertEqual(row["n_voxels"], 1)

    def test_radiomics_receives_union_of_all_labels(self) -> None:
        pet = np.zeros((4, 4, 4), dtype=np.float32)
        pet[1, 1, 1] = 3.0
        pet[2, 2, 2] = 7.0
        mask = np.zeros_like(pet, dtype=np.uint8)
        mask[1, 1, 1], mask[2, 2, 2] = 1, 2
        ct_path = self._image("ct.nii.gz", np.zeros_like(pet), np.eye(4))
        pet_path = self._image("pet.nii.gz", pet, np.eye(4))
        mask_path = self._image("mask.nii.gz", mask, np.eye(4))
        study = StudyAssets("P1", "T1", self.root, self.root / "processed" / "P1" / "T1",
                            ct_reference=ct_path, pet_aligned=pet_path, lesion_auto=mask_path)

        def inspect_union(_pet, _ct, binary_path):
            return {"test_union_voxels": int(np.asanyarray(nib.load(str(binary_path)).dataobj).sum())}

        with patch("features._extract_radiomics", side_effect=inspect_union):
            all_row = compute_roi_features(study, mask_path, "native_lesion", "baseline")
            second = compute_roi_features(study, mask_path, "native_lesion_component",
                                          "baseline", lesion_id=2)
        self.assertEqual(all_row["test_union_voxels"], 2)
        self.assertEqual(second["test_union_voxels"], 1)

    def test_mask_review_invalidates_when_file_changes(self) -> None:
        mask = self.root / "mask.nii.gz"
        mask.write_bytes(b"first")
        previous = archive_existing_mask(mask)
        self.assertEqual(previous.read_bytes(), b"first")
        version = record_mask_version(mask)
        self.assertTrue(version.is_file())
        review_mask(mask, "R1", "Checked all planes", "approved")
        self.assertEqual(mask_review_status(mask), "approved")
        mask.write_bytes(b"second")
        self.assertEqual(mask_review_status(mask), "stale")
        with self.assertRaises(ValueError):
            review_mask(mask, "R1", "Changed without version", "approved")

    def test_mapping_invalidates_after_baseline_mask_edit(self) -> None:
        ct = self.root / "ct.nii.gz"
        followup_ct = self.root / "fu_ct.nii.gz"
        mask = self.root / "mask.nii.gz"
        for path in (ct, followup_ct, mask):
            path.write_bytes(path.name.encode())
        baseline = StudyAssets("P1", "T1", self.root, self.root / "P1" / "T1",
                               ct_reference=ct, lesion_auto=mask)
        followup = StudyAssets("P1", "T2", self.root, self.root / "P1" / "T2",
                               ct_reference=followup_ct)
        out = followup.longitudinal_dir()
        out.mkdir(parents=True)
        result = out / "baseline_lesion_warped.nii.gz"
        warped_ct = out / "baseline_ct_warped.nii.gz"
        mat = out / "baseline_T1_to_this_0GenericAffine.mat"
        result.write_bytes(b"mapped")
        warped_ct.write_bytes(b"warped ct")
        mat.write_bytes(b"transform")
        write_mapping_manifest(baseline, followup, result)
        self.assertEqual(mapping_status(baseline, followup), "pending")
        review_mapping(baseline, followup, "R1", "Anatomic landmarks align", "approved")
        self.assertEqual(mapping_status(baseline, followup), "approved")
        warped_ct.write_bytes(b"altered warped ct")
        self.assertEqual(mapping_status(baseline, followup), "stale")
        warped_ct.write_bytes(b"warped ct")
        self.assertEqual(mapping_status(baseline, followup), "approved")
        mask.write_bytes(b"revised")
        self.assertEqual(mapping_status(baseline, followup), "stale")

    def test_more_than_three_timepoints(self) -> None:
        session = LongitudinalSession("P1", baseline="T1", interim="T2", end="T3",
                                      extra_dates=["T4", "T5"])
        self.assertEqual(len(session.dates_in_order()), 5)
        self.assertEqual(len(session.followups()), 4)
        self.assertEqual(session.validate(["T1", "T2", "T3", "T4", "T5"]), [])

    def test_sparse_lesion_stats_and_nonfinite_pet(self) -> None:
        mask = np.zeros((20, 20, 20), dtype=np.uint16)
        pet = np.zeros(mask.shape, dtype=np.float32)
        mask[1, 1, 1] = mask[1, 1, 2] = 1
        mask[4, 4, 4] = 3
        pet[1, 1, 1], pet[1, 1, 2], pet[4, 4, 4] = 5.0, np.nan, 7.0
        rows = lesion_stats(mask, pet, 0.008)
        self.assertEqual([r["id"] for r in rows], [1, 3])
        self.assertEqual([r["n_voxels"] for r in rows], [2, 1])
        self.assertEqual([r["suv_max"] for r in rows], [5.0, 7.0])

    def test_manifest_and_local_threshold(self) -> None:
        image = np.zeros((24, 24, 24), dtype=np.float32)
        image[4, 4, 4] = 8.0
        image[20, 20, 20] = 12.0
        ct = self._image("ct.nii.gz", np.zeros_like(image), np.eye(4))
        pet = self._image("pet.nii.gz", image, np.eye(4))
        manifest = self.root / "studies.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["patient_id", "study_date", "ct_path",
                                                       "pet_path", "pet_units"])
            writer.writeheader()
            writer.writerow({"patient_id": "P1", "study_date": "2026-01-01",
                             "ct_path": ct.name, "pet_path": pet.name, "pet_units": "SUVbw"})
        catalog = DataCatalog(self.root, self.root / "outputs", manifest)
        self.assertEqual(catalog.get_study("P1", "2026-01-01").working_pet, pet)
        selected = threshold_pet_mask(image, mode="relative", value=0.41,
                                      roi_center=(4, 4, 4), roi_radius_mm=5,
                                      affine=np.eye(4))
        self.assertGreater(selected[4, 4, 4], 0)
        self.assertEqual(selected[20, 20, 20], 0)
        empty = threshold_pet_mask(np.zeros_like(image), mode="relative", value=0.41,
                                   roi_center=(4, 4, 4), roi_radius_mm=5,
                                   affine=np.eye(4))
        self.assertEqual(int(np.count_nonzero(empty)), 0)

        unsafe = self.root / "unsafe.csv"
        with unsafe.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["patient_id", "study_date", "ct_path",
                                                       "pet_path", "pet_units"])
            writer.writeheader()
            writer.writerow({"patient_id": "../escape", "study_date": "2026-01-01",
                             "ct_path": ct.name, "pet_path": pet.name, "pet_units": "SUVbw"})
        with self.assertRaises(ValueError):
            DataCatalog(self.root, self.root / "outputs", unsafe)


if __name__ == "__main__":
    unittest.main()
