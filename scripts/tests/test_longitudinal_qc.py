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

from catalog import DataCatalog, StudyAssets  # noqa: E402
from features import compute_roi_features  # noqa: E402
from mask_ops import lesion_stats, threshold_pet_mask  # noqa: E402
from quality import (archive_existing_mask, mapping_status, mask_review_status, record_mask_version,
                     review_mapping, review_mask, write_mapping_manifest)  # noqa: E402
from session import LongitudinalSession  # noqa: E402


class LongitudinalQcTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

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
