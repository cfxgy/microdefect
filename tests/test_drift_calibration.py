#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2 bootstrap 阈值校准测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift.calibration import CalibrationError, bootstrap_calibrate  # noqa: E402
from src.drift.image_statistics import compute_image_stats  # noqa: E402

DIM = 512


def make_reference(n: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    images = [rng.integers(0, 256, size=(64, 64), dtype=np.uint8) for _ in range(n)]
    stats = [compute_image_stats(image) for image in images]
    features = rng.normal(size=(n, DIM))
    return stats, features


class CalibrationTest(unittest.TestCase):
    def test_01_reproducible(self) -> None:
        stats, features = make_reference(25, 1)
        first = bootstrap_calibrate(
            stats, features, seed=42, class_name="Cr",
            window_size=16, iterations=50, random_seed=123,
        )
        second = bootstrap_calibrate(
            stats, features, seed=42, class_name="Cr",
            window_size=16, iterations=50, random_seed=123,
        )
        self.assertEqual(first.threshold_A, second.threshold_A)
        self.assertEqual(first.threshold_B, second.threshold_B)
        self.assertEqual(first.threshold_fusion, second.threshold_fusion)
        np.testing.assert_array_equal(first.null_scores_A, second.null_scores_A)

    def test_02_different_seed_differs(self) -> None:
        stats, features = make_reference(25, 2)
        first = bootstrap_calibrate(
            stats, features, seed=42, class_name="Cr",
            window_size=16, iterations=50, random_seed=123,
        )
        second = bootstrap_calibrate(
            stats, features, seed=42, class_name="Cr",
            window_size=16, iterations=50, random_seed=124,
        )
        self.assertFalse(np.array_equal(first.null_scores_A, second.null_scores_A))

    def test_03_threshold_is_quantile_of_null(self) -> None:
        stats, features = make_reference(25, 3)
        result = bootstrap_calibrate(
            stats, features, seed=42, class_name="Cr",
            window_size=16, iterations=200, random_seed=7, quantile=0.95,
        )
        self.assertAlmostEqual(result.threshold_A, float(np.quantile(result.null_scores_A, 0.95)), places=12)
        self.assertAlmostEqual(result.threshold_B, float(np.quantile(result.null_scores_B, 0.95)), places=12)
        expected_fusion = float(
            np.quantile(
                np.maximum(
                    result.null_scores_A / max(result.threshold_A, 1e-12),
                    result.null_scores_B / max(result.threshold_B, 1e-12),
                ),
                0.95,
            )
        )
        self.assertAlmostEqual(result.threshold_fusion, expected_fusion, places=12)

    def test_04_class_isolation(self) -> None:
        stats_a, features_a = make_reference(25, 4)
        stats_b, features_b = make_reference(25, 5)
        a = bootstrap_calibrate(stats_a, features_a, seed=42, class_name="Cr",
                                window_size=16, iterations=50, random_seed=9)
        b = bootstrap_calibrate(stats_b, features_b, seed=42, class_name="In",
                                window_size=16, iterations=50, random_seed=9)
        self.assertNotEqual(a.threshold_A, b.threshold_A)
        self.assertNotEqual(a.threshold_B, b.threshold_B)

    def test_05_invalid_arguments(self) -> None:
        stats, features = make_reference(10, 6)
        with self.assertRaises(CalibrationError):
            bootstrap_calibrate(stats, features, seed=1, class_name="Cr",
                                window_size=1, iterations=50, random_seed=1)
        with self.assertRaises(CalibrationError):
            bootstrap_calibrate(stats, features, seed=1, class_name="Cr",
                                window_size=8, iterations=5, random_seed=1)
        with self.assertRaises(CalibrationError):
            bootstrap_calibrate(stats, features, seed=1, class_name="Cr",
                                window_size=8, iterations=50, random_seed=1, quantile=0.4)

    def test_06_reference_size_mismatch(self) -> None:
        stats, features = make_reference(10, 7)
        with self.assertRaises(CalibrationError):
            bootstrap_calibrate(stats, features[:5], seed=1, class_name="Cr",
                                window_size=4, iterations=20, random_seed=1)

    def test_07_summary_fields(self) -> None:
        stats, features = make_reference(25, 8)
        result = bootstrap_calibrate(stats, features, seed=42, class_name="PS",
                                     window_size=16, iterations=30, random_seed=3)
        summary = result.summary()
        self.assertEqual(summary["seed"], 42)
        self.assertEqual(summary["class_name"], "PS")
        self.assertEqual(summary["iterations"], 30)
        self.assertGreater(summary["threshold_A"], 0.0)
        self.assertGreater(summary["threshold_B"], 0.0)
        self.assertEqual(summary["reference_count"], 25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
