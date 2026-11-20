"""SUL、百分变化、Deauville 近似和 PERCIST 分类。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "longitudinal"))

from quant import (  # noqa: E402
    deauville_surrogate,
    lean_body_mass_kg,
    parse_height_m,
    percent_change,
    summarize_percist,
)


class QuantTests(unittest.TestCase):
    def test_janmahasatian_and_height(self) -> None:
        self.assertAlmostEqual(lean_body_mass_kg("M", 1.75, 70), 55.8571, places=4)
        self.assertAlmostEqual(lean_body_mass_kg("f", 1.55, 60), 37.3950, places=4)
        self.assertIsNone(lean_body_mass_kg("O", 1.70, 70))
        self.assertIsNone(lean_body_mass_kg("M", None, 70))
        self.assertAlmostEqual(parse_height_m(170), 1.70)
        self.assertAlmostEqual(parse_height_m(1.70), 1.70)
        self.assertIsNone(parse_height_m(0))
        self.assertIsNone(parse_height_m(None))

    def test_percent_change(self) -> None:
        self.assertAlmostEqual(percent_change(10, 5), -50.0)
        self.assertTrue(math.isnan(percent_change(0, 5)))

    def test_deauville_cuts(self) -> None:
        self.assertEqual(deauville_surrogate(2.0, blood=2.0, liver=4.0), 2)
        self.assertEqual(deauville_surrogate(3.0, blood=2.0, liver=4.0), 3)
        self.assertEqual(deauville_surrogate(8.0, blood=2.0, liver=4.0), 4)
        self.assertEqual(deauville_surrogate(8.1, blood=2.0, liver=4.0), 5)
        self.assertIsNone(deauville_surrogate(3.0, blood=float("nan"), liver=4.0))

    def test_percist_partial_response(self) -> None:
        rows = [_lesion(baseline_sul_peak=10.0, interim_sul_peak=4.0)]
        summary = _summarize(rows)
        self.assertEqual(summary["percist"], "PMR")
        self.assertEqual(summary["n_targets"], 1)

    def test_percist_new_measurable_is_progression(self) -> None:
        rows = [
            _lesion(baseline_sul_peak=10.0, interim_sul_peak=4.0),
            _lesion(
                match_status="new", baseline_lesion_id="", interim_lesion_id=2,
                baseline_sul_peak="", interim_sul_peak=6.0, baseline_measurable="",
                quantitative_outlier=1,
            ),
        ]
        self.assertEqual(_summarize(rows)["percist"], "PMD")

    def test_percist_complete_response_when_all_measurable_resolve(self) -> None:
        rows = [_lesion(
            match_status="resolved", interim_lesion_id="", interim_sul_peak="",
            lifecycle="resolved", _residual_suv_peak=1.0,
        )]
        self.assertEqual(_summarize(rows)["percist"], "CMR")

    def test_percist_missing_scale_is_not_evaluable(self) -> None:
        summary = _summarize([_lesion()], baseline_scale=float("nan"))
        self.assertEqual(summary["percist"], "not_evaluable")

    def test_percist_keeps_two_targets_per_station(self) -> None:
        rows = [
            _lesion(baseline_lesion_id=index, interim_lesion_id=index, baseline_sul_peak=10 - index)
            for index in range(1, 4)
        ]
        summary = _summarize(rows)
        self.assertEqual(summary["n_measurable"], 3)
        self.assertEqual(summary["n_targets"], 2)
        self.assertEqual(summary["percist"], "SMD")


def _lesion(**overrides) -> dict:
    row = {
        "match_status": "matched",
        "baseline_lesion_id": 1,
        "interim_lesion_id": 1,
        "station": "mediastinum",
        "baseline_sul_peak": 10.0,
        "interim_sul_peak": 10.0,
        "baseline_measurable": 1,
        "quantitative_outlier": 0,
        "lifecycle": "persist",
        "_residual_suv_peak": float("nan"),
    }
    row.update(overrides)
    return row


def _summarize(rows: list[dict], *, baseline_scale: float = 1.0) -> dict:
    return summarize_percist(
        rows,
        patient_id="P",
        baseline_date="20200101",
        interim_date="20200401",
        pairing="greedy",
        baseline_scale=baseline_scale,
        interim_scale=1.0,
        baseline_liver_mean=2.0,
        baseline_liver_sd=0.5,
        interim_liver_mean=2.0,
        interim_liver_sd=0.5,
        blood_pool=1.5,
    )


if __name__ == "__main__":
    unittest.main()
