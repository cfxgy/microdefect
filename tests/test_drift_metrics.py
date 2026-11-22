#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2 指标模块测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift.metrics import (  # noqa: E402
    MetricsError,
    auroc,
    binary_metrics,
    binary_metrics_from_predictions,
    macro_average,
    mean_std,
)


class BinaryMetricsTest(unittest.TestCase):
    def test_01_precision_recall_f1(self) -> None:
        # 预测: 1,1,1,0,0 ; 真实: 1,1,0,0,1 → tp=2, fp=1, fn=1, tn=1
        scores = [0.9, 0.8, 0.7, 0.2, 0.1]
        labels = [1, 1, 0, 0, 1]
        result = binary_metrics(scores, labels, threshold=0.5)
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (2, 1, 1, 1))
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["fpr"], 1 / 2)
        self.assertAlmostEqual(result["fnr"], 1 / 3)

    def test_02_predictions_variant(self) -> None:
        scores = [0.9, 0.8, 0.7, 0.2]
        labels = [1, 1, 0, 0]
        predictions = [1, 0, 0, 0]
        result = binary_metrics_from_predictions(scores, labels, predictions, threshold=0.5)
        self.assertEqual((result["tp"], result["fn"], result["tn"], result["fp"]), (1, 1, 2, 0))
        self.assertAlmostEqual(result["recall"], 0.5)

    def test_03_auroc_continuous(self) -> None:
        labels = [0, 0, 1, 1]
        perfect = [0.1, 0.2, 0.8, 0.9]
        reversed_scores = [0.9, 0.8, 0.2, 0.1]
        self.assertAlmostEqual(auroc(perfect, labels), 1.0)
        self.assertAlmostEqual(auroc(reversed_scores, labels), 0.0)

    def test_04_auroc_ignores_threshold(self) -> None:
        labels = [0, 1, 0, 1]
        scores = [0.1, 0.4, 0.35, 0.8]
        low = binary_metrics(scores, labels, threshold=0.3)["auroc"]
        high = binary_metrics(scores, labels, threshold=0.7)["auroc"]
        self.assertEqual(low, high)

    def test_05_single_class_returns_none(self) -> None:
        self.assertIsNone(auroc([0.1, 0.2], [1, 1]))
        result = binary_metrics([0.1, 0.2], [0, 0], 0.5)
        self.assertIsNone(result["auroc"])

    def test_06_invalid_inputs(self) -> None:
        with self.assertRaises(MetricsError):
            binary_metrics([0.1], [0, 1], 0.5)
        with self.assertRaises(MetricsError):
            binary_metrics([np.nan, 0.2], [0, 1], 0.5)
        with self.assertRaises(MetricsError):
            binary_metrics([0.1, 0.2], [0, 2], 0.5)


class AggregationTest(unittest.TestCase):
    def test_07_mean_std_ddof1(self) -> None:
        result = mean_std([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["std"], float(np.std([1, 2, 3, 4], ddof=1)))

    def test_08_mean_std_single_sample(self) -> None:
        result = mean_std([5.0])
        self.assertEqual(result["count"], 1)
        self.assertAlmostEqual(result["mean"], 5.0)
        self.assertIsNone(result["std"])

    def test_09_mean_std_empty(self) -> None:
        result = mean_std([])
        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["std"])

    def test_10_macro_average_ignores_none(self) -> None:
        records = [
            {"precision": 1.0, "recall": 0.5, "f1": 0.6, "auroc": None},
            {"precision": 0.5, "recall": 0.5, "f1": 0.5, "auroc": 0.9},
        ]
        result = macro_average(records)
        self.assertAlmostEqual(result["precision"], 0.75)
        self.assertAlmostEqual(result["auroc"], 0.9)
        self.assertIsNone(macro_average([{"precision": None}])["precision"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
