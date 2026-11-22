#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-2.3.3 阈值校准模块测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.anomaly.mvtec_calibration import (  # noqa: E402
    CALIBRATION_IMAGE_COUNT,
    CalibrationError,
    calibrate_image_scores,
    calibrate_pixel_heatmaps,
    calibration_metadata,
    is_anomalous,
    linear_quantile,
)


class LinearQuantileTest(unittest.TestCase):
    def test_01_matches_numpy(self) -> None:
        values = np.arange(1, 26, dtype=np.float64)  # 1..25
        self.assertAlmostEqual(linear_quantile(values, 0.95), float(np.quantile(values, 0.95, method="linear")))
        self.assertAlmostEqual(linear_quantile(values, 0.95), 23.8)
        self.assertAlmostEqual(linear_quantile(values, 0.99), 24.76)
        self.assertAlmostEqual(linear_quantile(values, 0.5), 13.0)

    def test_02_invalid_quantile(self) -> None:
        values = np.arange(5, dtype=np.float64)
        for q in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(CalibrationError):
                linear_quantile(values, q)

    def test_03_empty_and_nonfinite(self) -> None:
        with self.assertRaises(CalibrationError):
            linear_quantile(np.array([]), 0.95)
        with self.assertRaises(CalibrationError):
            linear_quantile(np.array([1.0, np.nan]), 0.95)


class StrictGreaterTest(unittest.TestCase):
    def test_04_strict_greater(self) -> None:
        self.assertFalse(is_anomalous(5.0, 5.0))
        self.assertTrue(is_anomalous(5.0001, 5.0))
        self.assertFalse(is_anomalous(4.999, 5.0))

    def test_05_nonfinite_rejected(self) -> None:
        with self.assertRaises(CalibrationError):
            is_anomalous(np.nan, 1.0)
        with self.assertRaises(CalibrationError):
            is_anomalous(1.0, np.inf)


class ImageCalibrationTest(unittest.TestCase):
    def test_06_threshold_and_exceed(self) -> None:
        scores = np.arange(1, 26, dtype=np.float64)  # 25 张
        result = calibrate_image_scores(scores)
        self.assertEqual(result.n_scores, 25)
        self.assertAlmostEqual(result.threshold, 23.8)
        self.assertEqual(result.quantile, 0.95)
        self.assertEqual(result.method, "linear")
        # 严格大于 23.8 的只有 24、25 → 2 个
        self.assertEqual(result.exceed_count, 2)
        self.assertEqual(result.stats["min"], 1.0)
        self.assertEqual(result.stats["max"], 25.0)

    def test_07_requires_25_scores(self) -> None:
        with self.assertRaises(CalibrationError):
            calibrate_image_scores(np.arange(24, dtype=np.float64))
        with self.assertRaises(CalibrationError):
            calibrate_image_scores(np.arange(26, dtype=np.float64))

    def test_08_nonfinite_rejected(self) -> None:
        scores = np.arange(1, 26, dtype=np.float64)
        scores[3] = np.nan
        with self.assertRaises(CalibrationError):
            calibrate_image_scores(scores)


class PixelCalibrationTest(unittest.TestCase):
    def test_09_pooling_matches_manual(self) -> None:
        rng = np.random.default_rng(0)
        heatmaps = [rng.random((4, 4)).astype(np.float32) for _ in range(CALIBRATION_IMAGE_COUNT)]
        result = calibrate_pixel_heatmaps(heatmaps)
        pooled = np.concatenate([h.ravel() for h in heatmaps])
        self.assertEqual(result.n_pixels, 25 * 16)
        self.assertAlmostEqual(result.threshold, float(np.quantile(pooled, 0.99, method="linear")))
        self.assertEqual(result.exceed_count, int(np.count_nonzero(pooled > result.threshold)))
        self.assertEqual(result.quantile, 0.99)

    def test_10_empty_rejected(self) -> None:
        with self.assertRaises(CalibrationError):
            calibrate_pixel_heatmaps([])

    def test_11_nonfinite_rejected(self) -> None:
        bad = np.zeros((4, 4), dtype=np.float32)
        bad[0, 0] = np.nan
        with self.assertRaises(CalibrationError):
            calibrate_pixel_heatmaps([bad])

    def test_12_threshold_strict(self) -> None:
        heatmaps = [np.full((2, 2), 1.0, dtype=np.float32) for _ in range(CALIBRATION_IMAGE_COUNT)]
        result = calibrate_pixel_heatmaps(heatmaps)
        self.assertEqual(result.threshold, 1.0)
        self.assertEqual(result.exceed_count, 0)
        self.assertFalse(is_anomalous(1.0, result.threshold))
        self.assertTrue(is_anomalous(1.0001, result.threshold))


class MetadataTest(unittest.TestCase):
    def test_13_metadata(self) -> None:
        meta = calibration_metadata()
        self.assertEqual(meta["image_quantile"], 0.95)
        self.assertEqual(meta["pixel_quantile"], 0.99)
        self.assertEqual(meta["quantile_method"], "linear")
        self.assertEqual(meta["calibration_images"], 25)
        self.assertFalse(meta["is_conformal"])
        self.assertEqual(meta["fpr_guarantee"], "none")


if __name__ == "__main__":
    unittest.main(verbosity=2)
