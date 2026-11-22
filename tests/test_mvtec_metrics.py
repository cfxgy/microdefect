#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-2.4 指标模块测试（手工可验证的小样本）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.anomaly.mvtec_metrics import (  # noqa: E402
    MetricError,
    auroc_rank_average,
    average_precision,
    image_metrics,
    macro_average,
    mean_std,
    pixel_average_precision,
    pixel_metrics,
    pixel_auroc,
    threshold_metrics,
)


class ImageMetricsTest(unittest.TestCase):
    def test_01_perfect_separation(self) -> None:
        scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        labels = np.array([1, 1, 0, 0], dtype=bool)
        self.assertAlmostEqual(auroc_rank_average(scores, labels), 1.0)
        self.assertAlmostEqual(average_precision(scores, labels), 1.0)

    def test_02_ties_average_rank(self) -> None:
        scores = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        labels = np.array([1, 0, 0, 0], dtype=bool)
        # 正样本位于并列组，秩 2,3,4 的平均 = 3 → (3-1)/3
        self.assertAlmostEqual(auroc_rank_average(scores, labels), 2.0 / 3.0, places=6)

    def test_03_ap_hand(self) -> None:
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        labels = np.array([1, 0, 1], dtype=bool)
        # 降序：pos@1 (P=1), neg@2, pos@3 (P=2/3)
        self.assertAlmostEqual(average_precision(scores, labels), (1.0 + 2.0 / 3.0) / 2.0, places=6)

    def test_04_threshold_strict(self) -> None:
        scores = np.array([0.5, 0.6], dtype=np.float32)
        labels = np.array([1, 1], dtype=bool)
        result = threshold_metrics(scores, labels, 0.5)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 2.0 / 3.0)

    def test_05_equal_threshold_is_normal(self) -> None:
        scores = np.array([0.5], dtype=np.float32)
        labels = np.array([1], dtype=bool)
        result = threshold_metrics(scores, labels, 0.5)
        self.assertEqual(result["predicted_positive"], 0)
        self.assertEqual(result["fn"], 1)

    def test_06_none_on_zero_denominator(self) -> None:
        scores = np.array([0.1, 0.2], dtype=np.float32)
        labels = np.array([1, 1], dtype=bool)
        result = threshold_metrics(scores, labels, 1.0)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["f1"])
        self.assertEqual(result["recall"], 0.0)
        all_neg = np.zeros(4, dtype=bool)
        self.assertIsNone(auroc_rank_average(scores[:2], all_neg[:2]))
        self.assertIsNone(average_precision(scores[:2], all_neg[:2]))
        all_pos = np.ones(4, dtype=bool)
        self.assertIsNone(auroc_rank_average(scores, all_pos[:2]))

    def test_07_image_metrics_bundle(self) -> None:
        scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        labels = np.array([1, 1, 0, 0], dtype=bool)
        result = image_metrics(scores, labels, 0.5)
        self.assertEqual(result["n_pos"], 2)
        self.assertEqual(result["n_neg"], 2)
        self.assertEqual(result["tp"], 2)
        self.assertAlmostEqual(result["f1"], 1.0)


class PixelMetricsTest(unittest.TestCase):
    def test_08_pooled_equivalence(self) -> None:
        rng = np.random.default_rng(0)
        a_scores = rng.random(500).astype(np.float32)
        a_labels = rng.random(500) < 0.2
        b_scores = rng.random(700).astype(np.float32)
        b_labels = rng.random(700) < 0.2
        pooled = pixel_metrics(np.concatenate([a_scores, b_scores]),
                               np.concatenate([a_labels, b_labels]))
        self.assertEqual(pooled["n_pixels"], 1200)

    def test_09_pixel_equals_bruteforce(self) -> None:
        rng = np.random.default_rng(1)
        scores = rng.random(2000).astype(np.float32)
        scores[::17] = 0.5  # 制造并列
        labels = rng.random(2000) < 0.15
        # 手工：对每个正样本统计负样本中更小/相等的数量
        neg = np.sort(scores[~labels])
        pos = scores[labels]
        less = np.searchsorted(neg, pos, side="left")
        leq = np.searchsorted(neg, pos, side="right")
        manual = (less + 0.5 * (leq - less)).sum() / (labels.sum() * (~labels).sum())
        self.assertAlmostEqual(pixel_auroc(scores, labels), float(manual), places=6)
        # AP 与稳定降序 argsort 结果一致
        order = np.argsort(-scores, kind="stable")
        positions = np.flatnonzero(labels[order]) + 1
        manual_ap = (np.arange(1, labels.sum() + 1) / positions).mean()
        self.assertAlmostEqual(pixel_average_precision(scores, labels), float(manual_ap), places=6)

    def test_10_pixel_none_cases(self) -> None:
        scores = np.array([0.1, 0.2], dtype=np.float32)
        self.assertIsNone(pixel_auroc(scores, np.array([1, 1], dtype=bool)))
        self.assertIsNone(pixel_average_precision(scores, np.zeros(2, dtype=bool)))


class AggregationTest(unittest.TestCase):
    def test_11_macro(self) -> None:
        self.assertAlmostEqual(macro_average([0.5, None, 0.7]), 0.6)
        self.assertIsNone(macro_average([None, None]))

    def test_12_mean_std(self) -> None:
        result = mean_std([1.0, 2.0, 3.0])
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["std"], 1.0)
        self.assertEqual(result["n"], 3)
        self.assertIsNone(mean_std([None])["mean"])

    def test_13_input_errors(self) -> None:
        with self.assertRaises(MetricError):
            auroc_rank_average(np.array([], dtype=np.float32), np.array([], dtype=bool))
        with self.assertRaises(MetricError):
            auroc_rank_average(np.array([np.nan], dtype=np.float32), np.array([True]))
        with self.assertRaises(MetricError):
            threshold_metrics(np.array([1.0], dtype=np.float32), np.array([True, False]), 0.5)
        with self.assertRaises(MetricError):
            threshold_metrics(np.array([1.0], dtype=np.float32), np.array([True]), np.inf)


if __name__ == "__main__":
    unittest.main(verbosity=2)
