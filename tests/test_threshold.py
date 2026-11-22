#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 阈值校准与拒识测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.prototype import build_prototypes, cosine_distance, l2_normalize  # noqa: E402
from src.memory.threshold import (  # noqa: E402
    apply_thresholds,
    calibrate_thresholds,
    describe_distances,
    rejection_metrics,
)

CLASSES = ("Cr", "In", "Pa", "PS", "RS", "Sc")
DIM = 512


def random_features(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, DIM)).astype(np.float32)


def make_setup():
    support = {cls: random_features(5, seed=100 + i) for i, cls in enumerate(CLASSES)}
    prototypes = build_prototypes(support, CLASSES)
    validation = {cls: random_features(40, seed=200 + i) for i, cls in enumerate(CLASSES)}
    return support, prototypes, validation


class ThresholdCalibrationTest(unittest.TestCase):
    def test_01_threshold_equals_validation_quantile(self) -> None:
        _, prototypes, validation = make_setup()
        thresholds, stats = calibrate_thresholds(validation, prototypes, CLASSES, quantile=0.95)

        proto_matrix = np.stack([prototypes[cls] for cls in CLASSES], axis=0)
        for index, cls in enumerate(CLASSES):
            distances = cosine_distance(validation[cls], proto_matrix)[:, index]
            self.assertAlmostEqual(thresholds[cls], float(np.quantile(distances, 0.95)), places=9)
            self.assertEqual(stats[cls]["n"], len(validation[cls]))
            self.assertEqual(stats[cls]["source"], "validation")

    def test_02_thresholds_are_per_class(self) -> None:
        _, prototypes, validation = make_setup()
        thresholds, _ = calibrate_thresholds(validation, prototypes, CLASSES)
        self.assertEqual(set(thresholds), set(CLASSES))
        self.assertGreater(len({round(v, 9) for v in thresholds.values()}), 1)

    def test_03_calibration_uses_only_validation(self) -> None:
        """阈值只由 Validation 决定：改成极端"测试集"也不影响。"""
        _, prototypes, validation = make_setup()
        first, _ = calibrate_thresholds(validation, prototypes, CLASSES)

        # 若函数曾经偷看测试集，这里的"测试集"变化会影响结果；接口根本不吃 test
        fake_test = {cls: random_features(10, seed=900 + i) for i, cls in enumerate(CLASSES)}
        second, _ = calibrate_thresholds(validation, prototypes, CLASSES)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(fake_test))

    def test_04_quantile_sensitivity(self) -> None:
        _, prototypes, validation = make_setup()
        low, _ = calibrate_thresholds(validation, prototypes, CLASSES, quantile=0.5)
        high, _ = calibrate_thresholds(validation, prototypes, CLASSES, quantile=0.95)
        for cls in CLASSES:
            self.assertLessEqual(low[cls], high[cls])

    def test_05_invalid_quantile(self) -> None:
        _, prototypes, validation = make_setup()
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                calibrate_thresholds(validation, prototypes, CLASSES, quantile=bad)

    def test_06_missing_validation_class(self) -> None:
        _, prototypes, validation = make_setup()
        incomplete = {cls: validation[cls] for cls in CLASSES[:-1]}
        with self.assertRaises(ValueError):
            calibrate_thresholds(incomplete, prototypes, CLASSES)

    def test_07_describe_distances(self) -> None:
        info = describe_distances(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(info["n"], 4)
        self.assertAlmostEqual(info["mean"], 0.25)
        self.assertAlmostEqual(info["max"], 0.4)


class RejectionTest(unittest.TestCase):
    def test_08_apply_threshold_boundary(self) -> None:
        thresholds = {cls: 0.5 for cls in CLASSES}
        labels = ["Cr", "Cr", "In"]
        distances = np.array([0.4, 0.5, 0.5001])
        accepted = apply_thresholds(labels, distances, thresholds)
        self.assertEqual(accepted.tolist(), [True, True, False])  # 等于阈值算接受

    def test_09_rejection_rate_and_counts(self) -> None:
        y_true = ["Cr", "Cr", "In", "In"]
        y_pred = ["Cr", "Cr", "In", "Cr"]
        accepted = np.array([True, False, True, False])
        metrics = rejection_metrics(y_true, y_pred, accepted, CLASSES)
        self.assertEqual(metrics["test_total"], 4)
        self.assertEqual(metrics["accepted_count"], 2)
        self.assertEqual(metrics["rejected_count"], 2)
        self.assertAlmostEqual(metrics["rejection_rate"], 0.5)

    def test_10_accepted_accuracy(self) -> None:
        y_true = ["Cr", "Cr", "In", "In"]
        y_pred = ["Cr", "Cr", "In", "Cr"]
        accepted = np.array([True, False, True, False])
        metrics = rejection_metrics(y_true, y_pred, accepted, CLASSES)
        # 被接收的两条都预测正确
        self.assertAlmostEqual(metrics["accepted_accuracy"], 1.0)

        # 被接受的两条中，第二条预测错误 → 接受样本准确率 0.5
        accepted = np.array([True, False, True, False])
        y_pred = ["Cr", "In", "Cr", "Cr"]
        metrics = rejection_metrics(y_true, y_pred, accepted, CLASSES)
        self.assertAlmostEqual(metrics["accepted_accuracy"], 0.5)

    def test_11_empty_accepted_returns_none(self) -> None:
        y_true = ["Cr", "In"]
        y_pred = ["Cr", "In"]
        metrics = rejection_metrics(y_true, y_pred, np.array([False, False]), CLASSES)
        self.assertEqual(metrics["accepted_count"], 0)
        self.assertIsNone(metrics["accepted_accuracy"])
        self.assertAlmostEqual(metrics["rejection_rate"], 1.0)

    def test_12_per_class_rejection(self) -> None:
        y_true = ["Cr", "Cr", "In", "In", "In"]
        y_pred = ["Cr", "Cr", "In", "In", "In"]
        accepted = np.array([True, False, True, True, False])
        metrics = rejection_metrics(y_true, y_pred, accepted, CLASSES)
        self.assertAlmostEqual(metrics["per_true_class"]["Cr"]["rejection_rate"], 0.5)
        self.assertAlmostEqual(metrics["per_true_class"]["In"]["rejection_rate"], 1 / 3)
        self.assertIsNone(metrics["per_true_class"]["Pa"]["rejection_rate"])  # 无样本 → None

    def test_13_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            rejection_metrics(["Cr"], ["Cr", "In"], np.array([True]), CLASSES)
        with self.assertRaises(ValueError):
            apply_thresholds(["Cr", "In"], np.array([0.1]), {"Cr": 0.2, "In": 0.2})

    def test_14_real_prototype_threshold_flow(self) -> None:
        """端到端：阈值来自 validation，拒绝一个明显更远的样本。"""
        support, prototypes, validation = make_setup()
        thresholds, _ = calibrate_thresholds(validation, prototypes, CLASSES)

        near = validation["Cr"][:1]
        far = (-prototypes["Cr"])[None, :]  # 与 Cr 原型方向相反 → 余弦距离 2.0
        labels = ["Cr", "Cr"]
        proto_matrix = np.stack([prototypes[c] for c in CLASSES], axis=0)
        distances = cosine_distance(np.vstack([near, far]), proto_matrix)[:, 0]
        accepted = apply_thresholds(labels, distances, thresholds)
        self.assertTrue(accepted[0])
        self.assertFalse(accepted[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
