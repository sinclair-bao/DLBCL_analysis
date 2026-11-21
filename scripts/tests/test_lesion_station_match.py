"""站点优先配对：缩小、消失、分裂和新发。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "longitudinal"))

from lesion_match_station import classify_lesions, summarize_stations  # noqa: E402
from stations import ST_AXILLA_LEFT, ST_MEDIASTINUM  # noqa: E402


def _lesion(lesion_id: int, station: int, mtv: float = 8.0, kind: str = "track") -> dict:
    return {
        "id": lesion_id,
        "kind": kind,
        "station": station,
        "mtv_ml": mtv,
        "suv_max": 12.0,
        "suv_mean": 6.0,
        "suv_peak": 9.0,
        "tlg": mtv * 6.0,
        "warp": "SyN",
    }


class StationMatchTests(unittest.TestCase):
    def test_shrink_matches_on_distance_without_overlap(self) -> None:
        baseline = [_lesion(1, ST_MEDIASTINUM)]
        interim = [_lesion(4, ST_MEDIASTINUM, mtv=1.5)]
        metrics = {(1, 4): {"distance_mm": 12.0, "contain_mm": 12.0, "iou": 0.0}}
        rows = classify_lesions(baseline, interim, metrics, {}, liver_suv_mean=2.0, qc_pass=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "matched")
        self.assertEqual(rows[0]["cost"], 12.0)
        self.assertFalse(rows[0]["contained"])
        self.assertAlmostEqual(rows[0]["delta_mtv"], (8.0 - 1.5) / 8.0 * 100.0)

    def test_overlap_beyond_30_mm_still_matches(self) -> None:
        baseline = [_lesion(1, ST_MEDIASTINUM, mtv=40.0)]
        interim = [_lesion(2, ST_MEDIASTINUM, mtv=30.0)]
        metrics = {(1, 2): {"distance_mm": 45.0, "contain_mm": 20.0, "iou": 0.15}}
        rows = classify_lesions(baseline, interim, metrics, {}, liver_suv_mean=2.0, qc_pass=True)
        self.assertEqual(rows[0]["status"], "matched")

    def test_disappearance_uses_residual_and_empty_station_is_new(self) -> None:
        baseline = [_lesion(1, ST_MEDIASTINUM)]
        interim = [_lesion(7, ST_AXILLA_LEFT)]
        metrics = {(1, 7): {"distance_mm": 180.0, "contain_mm": 180.0, "iou": 0.0}}
        residuals = {1: {"suv_max": 1.2, "suv_mean": 0.8, "suv_peak": 1.0}}
        rows = classify_lesions(baseline, interim, metrics, residuals, liver_suv_mean=2.0, qc_pass=True)
        by_status = {row["geometry_status"]: row for row in rows}
        self.assertEqual(by_status["resolved"]["baseline_id"], 1)
        self.assertEqual(by_status["resolved"]["interim_mtv"], 0.0)
        self.assertEqual(by_status["new"]["interim_id"], 7)

    def test_same_station_without_partner_is_review_not_new(self) -> None:
        baseline = [_lesion(1, ST_MEDIASTINUM)]
        interim = [_lesion(2, ST_MEDIASTINUM)]
        metrics = {(1, 2): {"distance_mm": 80.0, "contain_mm": 80.0, "iou": 0.0}}
        residuals = {1: {"suv_max": 8.0, "suv_mean": 4.0, "suv_peak": 6.0}}
        rows = classify_lesions(baseline, interim, metrics, residuals, liver_suv_mean=2.0, qc_pass=True)
        statuses = {row["geometry_status"] for row in rows}
        self.assertEqual(statuses, {"review"})
        self.assertTrue(all("新发" not in row["note"] for row in rows))

    def test_containment_splits_one_baseline_into_two(self) -> None:
        baseline = [_lesion(1, ST_MEDIASTINUM, mtv=30.0)]
        interim = [_lesion(2, ST_MEDIASTINUM, mtv=4.0), _lesion(3, ST_MEDIASTINUM, mtv=5.0)]
        metrics = {
            (1, 2): {"distance_mm": 6.0, "contain_mm": 4.0, "iou": 0.1},
            (1, 3): {"distance_mm": 9.0, "contain_mm": 8.0, "iou": 0.05},
        }
        rows = classify_lesions(baseline, interim, metrics, {}, liver_suv_mean=2.0, qc_pass=True)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "matched" and row["split"] for row in rows))

    def test_failed_registration_downgrades_identity(self) -> None:
        baseline = [_lesion(1, ST_MEDIASTINUM)]
        interim = [_lesion(2, ST_MEDIASTINUM)]
        metrics = {(1, 2): {"distance_mm": 8.0, "contain_mm": 8.0, "iou": 0.4}}
        rows = classify_lesions(baseline, interim, metrics, {}, liver_suv_mean=2.0, qc_pass=False)
        self.assertEqual(rows[0]["geometry_status"], "matched")
        self.assertEqual(rows[0]["status"], "review")

    def test_station_summary_marks_cleared_and_new(self) -> None:
        baseline = [_lesion(1, ST_MEDIASTINUM, mtv=10.0), _lesion(2, ST_MEDIASTINUM, mtv=0.4, kind="below")]
        interim = [_lesion(3, ST_AXILLA_LEFT, mtv=2.0)]
        rows = summarize_stations(baseline, interim)
        by_name = {row["station"]: row for row in rows}
        self.assertEqual(by_name["mediastinum"]["status"], "清除")
        self.assertEqual(by_name["mediastinum"]["baseline_count"], 2)
        self.assertEqual(by_name["axilla_left"]["status"], "新发")
        self.assertAlmostEqual(by_name["mediastinum"]["delta_mtv"], 100.0)


if __name__ == "__main__":
    unittest.main()
