"""Synthetic checks for station masks and Hungarian lesion matching."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "longitudinal"))

from lesion_match import (  # noqa: E402
    build_match_rows,
    label_lesions,
    measure_components,
    pair_cost,
)
from stations import (  # noqa: E402
    ST_AXILLA_LEFT,
    ST_BONE,
    ST_LIVER,
    ST_MEDIASTINUM,
    ST_NECK_LEFT,
    ST_NECK_RIGHT,
    ST_SPLEEN,
    build_station_mask,
    stations_compatible,
)


def _affine() -> np.ndarray:
    return np.diag([2.0, 2.0, 2.0, 1.0])


def _put(labels: np.ndarray, label_id: int, i: int, j: int, k: int) -> None:
    labels[i, j, k] = label_id


class StationMaskTests(unittest.TestCase):
    def test_organs_override_nodal_slabs_and_laterality(self) -> None:
        labels = np.zeros((48, 8, 40), dtype=np.int16)
        # Vertebrae on the midline. Larger k is superior because z = 2*k.
        for label_id, k in (
            (50, 36),  # C1
            (44, 30),  # C7
            (43, 28),  # T1
            (42, 27),  # T2
            (38, 22),  # T6
            (32, 14),  # T12
            (27, 8),   # L5
        ):
            _put(labels, label_id, 24, 4, k)
        _put(labels, 73, 30, 4, 28)  # clavicle left
        _put(labels, 74, 18, 4, 28)  # clavicle right
        _put(labels, 69, 40, 4, 24)  # humerus left
        _put(labels, 70, 8, 4, 24)   # humerus right
        _put(labels, 5, 24, 4, 33)   # liver sitting in the neck slab

        stations = build_station_mask(labels, _affine())

        self.assertEqual(int(stations[24, 4, 30]), ST_BONE)
        self.assertEqual(int(stations[24, 4, 33]), ST_LIVER)
        self.assertEqual(int(stations[28, 4, 33]), ST_NECK_LEFT)
        self.assertEqual(int(stations[34, 4, 24]), ST_AXILLA_LEFT)
        self.assertEqual(int(stations[24, 4, 20]), ST_MEDIASTINUM)


class LesionMatchTests(unittest.TestCase):
    def test_cost_uses_distance_and_iou_and_rejects_station_or_radius(self) -> None:
        self.assertEqual(pair_cost(10.0, 0.5, True), 20.0)
        self.assertEqual(pair_cost(21.0, 1.0, True), 21.0)
        self.assertEqual(pair_cost(21.0, 0.0, True), 41.0)
        self.assertIsNone(pair_cost(4.0, 1.0, False))
        self.assertIsNone(pair_cost(41.0, 1.0, True))

    def test_adjacent_stations_can_match_but_sides_stay_separate(self) -> None:
        self.assertTrue(stations_compatible(ST_BONE, ST_MEDIASTINUM))
        self.assertFalse(stations_compatible(ST_NECK_LEFT, ST_NECK_RIGHT))
        self.assertFalse(stations_compatible(ST_LIVER, ST_SPLEEN))

    def test_small_components_are_dropped(self) -> None:
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[0, 0, 0:2] = 1
        mask[4:7, 4:7, 4:7] = 1
        labeled = label_lesions(mask)
        self.assertEqual(int(labeled.max()), 1)
        self.assertEqual(int(labeled[0, 0, 0]), 0)
        self.assertEqual(int(labeled[5, 5, 5]), 1)

    def test_same_station_near_overlap_matches(self) -> None:
        shape = (16, 16, 16)
        baseline = np.zeros(shape, dtype=np.uint16)
        interim = np.zeros(shape, dtype=np.uint16)
        baseline[2:6, 2:6, 2:6] = 1
        interim[3:7, 3:7, 3:7] = 1
        stations = np.full(shape, ST_MEDIASTINUM, dtype=np.uint8)
        affine = _affine()
        base_pet = np.full(shape, 4.0, dtype=np.float32)
        follow_pet = np.full(shape, 6.0, dtype=np.float32)
        rows = build_match_rows(
            baseline, interim,
            measure_components(baseline, base_pet, affine),
            measure_components(interim, follow_pet, affine),
            follow_pet, stations, affine, liver_suv_mean=2.0,
        )
        self.assertEqual([row["match_status"] for row in rows], ["matched"])
        self.assertEqual(rows[0]["baseline_lesion_id"], 1)
        self.assertEqual(rows[0]["interim_lesion_id"], 1)
        self.assertEqual(rows[0]["station"], "mediastinum")
        self.assertLess(rows[0]["centroid_distance_mm"], 20.0)
        self.assertGreater(rows[0]["iou"], 0.0)

    def test_different_station_is_not_matched(self) -> None:
        shape = (16, 16, 16)
        baseline = np.zeros(shape, dtype=np.uint16)
        interim = np.zeros(shape, dtype=np.uint16)
        baseline[2:6, 8:12, 8:12] = 1
        interim[6:10, 8:12, 8:12] = 1
        stations = np.zeros(shape, dtype=np.uint8)
        stations[2:6, 8:12, 8:12] = ST_LIVER
        stations[6:10, 8:12, 8:12] = ST_SPLEEN
        affine = _affine()
        pet = np.ones(shape, dtype=np.float32)
        rows = build_match_rows(
            baseline, interim,
            measure_components(baseline, pet, affine),
            measure_components(interim, pet, affine),
            pet, stations, affine, liver_suv_mean=1.0,
        )
        self.assertEqual({row["match_status"] for row in rows}, {"resolved", "new"})
        self.assertFalse(any(row["match_status"] == "matched" for row in rows))

    def test_distance_beyond_20mm_is_not_matched(self) -> None:
        shape = (12, 12, 28)
        baseline = np.zeros(shape, dtype=np.uint16)
        interim = np.zeros(shape, dtype=np.uint16)
        baseline[2:6, 2:6, 2:6] = 1
        interim[2:6, 2:6, 16:20] = 1
        stations = np.full(shape, ST_MEDIASTINUM, dtype=np.uint8)
        affine = _affine()
        pet = np.ones(shape, dtype=np.float32)
        rows = build_match_rows(
            baseline, interim,
            measure_components(baseline, pet, affine),
            measure_components(interim, pet, affine),
            pet, stations, affine, liver_suv_mean=1.0,
        )
        self.assertEqual([row["match_status"] for row in rows], ["review"])
        self.assertEqual(rows[0]["baseline_lesion_id"], 1)
        self.assertEqual(rows[0]["interim_lesion_id"], 1)
        self.assertGreater(rows[0]["centroid_distance_mm"], 20.0)
        self.assertLess(rows[0]["centroid_distance_mm"], 40.0)

    def test_bone_and_mediastinum_within_20mm_match(self) -> None:
        shape = (16, 16, 16)
        baseline = np.zeros(shape, dtype=np.uint16)
        interim = np.zeros(shape, dtype=np.uint16)
        baseline[2:6, 8:12, 8:12] = 1
        interim[6:10, 8:12, 8:12] = 1
        stations = np.zeros(shape, dtype=np.uint8)
        stations[2:6, 8:12, 8:12] = ST_MEDIASTINUM
        stations[6:10, 8:12, 8:12] = ST_BONE
        affine = _affine()
        pet = np.full(shape, 4.0, dtype=np.float32)
        rows = build_match_rows(
            baseline, interim,
            measure_components(baseline, pet, affine),
            measure_components(interim, pet, affine),
            pet, stations, affine, liver_suv_mean=2.0,
        )
        self.assertEqual([row["match_status"] for row in rows], ["matched"])

    def test_large_baseline_accepts_two_small_lesions_by_surface_distance(self) -> None:
        shape = (40, 40, 40)
        baseline = np.zeros(shape, dtype=np.uint16)
        interim = np.zeros(shape, dtype=np.uint16)
        baseline[10:24, 10:24, 10:24] = 1
        interim[1:4, 10:13, 10:13] = 1
        interim[30:33, 10:13, 10:13] = 2
        stations = np.full(shape, ST_MEDIASTINUM, dtype=np.uint8)
        affine = _affine()
        pet = np.full(shape, 4.0, dtype=np.float32)
        rows = build_match_rows(
            baseline, interim,
            measure_components(baseline, pet, affine),
            measure_components(interim, pet, affine),
            pet, stations, affine, liver_suv_mean=2.0,
        )
        matched = [row for row in rows if row["match_status"] == "matched"]
        self.assertEqual(len(matched), 2)
        self.assertEqual({row["baseline_lesion_id"] for row in matched}, {1})
        self.assertEqual({row["interim_lesion_id"] for row in matched}, {1, 2})

    def test_extreme_suv_component_is_invalid(self) -> None:
        shape = (12, 12, 12)
        baseline = np.zeros(shape, dtype=np.uint16)
        interim = np.zeros(shape, dtype=np.uint16)
        baseline[2:6, 2:6, 2:6] = 1
        interim[3:7, 3:7, 3:7] = 1
        stations = np.full(shape, ST_MEDIASTINUM, dtype=np.uint8)
        affine = _affine()
        baseline_pet = np.full(shape, 150.0, dtype=np.float32)
        interim_pet = np.full(shape, 4.0, dtype=np.float32)
        rows = build_match_rows(
            baseline, interim,
            measure_components(baseline, baseline_pet, affine),
            measure_components(interim, interim_pet, affine),
            interim_pet, stations, affine, liver_suv_mean=2.0,
        )
        self.assertEqual({row["match_status"] for row in rows}, {"invalid", "new"})

    def test_resolved_keeps_residual_suv_and_zero_mtv(self) -> None:
        shape = (12, 12, 48)
        baseline = np.zeros(shape, dtype=np.uint16)
        interim = np.zeros(shape, dtype=np.uint16)
        baseline[2:6, 2:6, 2:6] = 1
        interim[2:6, 2:6, 40:44] = 2
        stations = np.full(shape, ST_NECK_LEFT, dtype=np.uint8)
        organs = np.zeros(shape, dtype=np.uint8)
        organs[3, 3, 3] = 4
        affine = _affine()
        baseline_pet = np.full(shape, 12.0, dtype=np.float32)
        interim_pet = np.zeros(shape, dtype=np.float32)
        interim_pet[2:6, 2:6, 2:6] = 3.0
        interim_pet[3, 3, 3] = 9.0
        interim_pet[2:6, 2:6, 40:44] = 5.0
        rows = build_match_rows(
            baseline, interim,
            measure_components(baseline, baseline_pet, affine),
            measure_components(interim, interim_pet, affine),
            interim_pet, stations, affine, liver_suv_mean=3.0, organs=organs,
        )
        by_status = {row["match_status"]: row for row in rows}
        self.assertEqual(set(by_status), {"resolved", "new"})
        resolved = by_status["resolved"]
        self.assertEqual(resolved["interim_mtv_ml"], 0.0)
        self.assertEqual(resolved["residual_suv_max"], 3.0)
        self.assertEqual(resolved["baseline_suv_max"], 12.0)
        self.assertEqual(resolved["suv_max_liver_ratio"], 1.0)
        self.assertEqual(resolved["interim_lesion_id"], "")
        new = by_status["new"]
        self.assertEqual(new["baseline_lesion_id"], "")
        self.assertEqual(new["interim_suv_max"], 5.0)
        self.assertEqual(new["residual_suv_max"], "")


if __name__ == "__main__":
    unittest.main()
