#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 原型与最近原型识别测试（纯 NumPy，不依赖模型/权重）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.prototype import (  # noqa: E402
    PrototypeError,
    build_prototype,
    build_prototypes,
    cosine_distance,
    l2_normalize,
    predict_nearest,
    stack_prototypes,
)

CLASSES = ("Cr", "In", "Pa", "PS", "RS", "Sc")
DIM = 512


def random_features(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


class PrototypeTest(unittest.TestCase):
    def test_01_prototype_shape_and_norm(self) -> None:
        features = random_features(5)
        prototype = build_prototype(features)
        self.assertEqual(prototype.shape, (DIM,))
        self.assertAlmostEqual(float(np.linalg.norm(prototype)), 1.0, places=5)

    def test_02_prototype_finite(self) -> None:
        prototype = build_prototype(random_features(3, seed=2))
        self.assertTrue(np.isfinite(prototype).all())

    def test_03_prototype_matches_formula(self) -> None:
        """p_c = L2Norm(mean(L2Norm(f_i)))。"""
        features = random_features(5, seed=3)
        manual = l2_normalize(l2_normalize(features).mean(axis=0))
        np.testing.assert_allclose(build_prototype(features), manual, rtol=1e-5, atol=1e-6)

    def test_04_shot_subsets_use_correct_support(self) -> None:
        """1/3/5-shot 原型必须分别来自前 1/3/5 张 support，且三者不同。"""
        features = random_features(5, seed=4)
        p1 = build_prototype(features[:1])
        p3 = build_prototype(features[:3])
        p5 = build_prototype(features[:5])
        np.testing.assert_allclose(p1, l2_normalize(features[0]), rtol=1e-5)
        self.assertFalse(np.allclose(p1, p3))
        self.assertFalse(np.allclose(p3, p5))
        np.testing.assert_allclose(p5, build_prototype(features), rtol=1e-6)

    def test_05_same_input_same_prototype(self) -> None:
        features = random_features(5, seed=5)
        np.testing.assert_array_equal(build_prototype(features), build_prototype(features))

    def test_06_build_prototypes_for_six_classes(self) -> None:
        data = {cls: random_features(5, seed=10 + i) for i, cls in enumerate(CLASSES)}
        prototypes = build_prototypes(data, CLASSES)
        self.assertEqual(list(prototypes), list(CLASSES))
        matrix = stack_prototypes(prototypes, CLASSES)
        self.assertEqual(matrix.shape, (6, DIM))
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-5)

    def test_07_missing_class_rejected(self) -> None:
        data = {cls: random_features(2) for cls in CLASSES[:-1]}
        with self.assertRaises(PrototypeError):
            build_prototypes(data, CLASSES)

    def test_08_nan_features_rejected(self) -> None:
        features = random_features(2)
        features[0, 0] = np.nan
        with self.assertRaises(PrototypeError):
            build_prototype(features)

    def test_09_empty_support_rejected(self) -> None:
        with self.assertRaises(PrototypeError):
            build_prototype(np.zeros((0, DIM), dtype=np.float32))


class DistanceAndPredictionTest(unittest.TestCase):
    def test_10_cosine_distance_formula(self) -> None:
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        distances = cosine_distance(a, b)
        np.testing.assert_allclose(distances, [[1.0, 0.0]], atol=1e-6)

    def test_11_distance_range(self) -> None:
        features = random_features(20, seed=6)
        prototypes = stack_prototypes(
            {cls: build_prototype(random_features(3, seed=20 + i)) for i, cls in enumerate(CLASSES)},
            CLASSES,
        )
        distances = cosine_distance(features, prototypes)
        self.assertGreaterEqual(float(distances.min()), 0.0)
        self.assertLessEqual(float(distances.max()), 2.0)

    def test_12_predict_nearest_six_classes(self) -> None:
        prototypes = {cls: l2_normalize(np.eye(DIM)[i]) for i, cls in enumerate(CLASSES)}
        features = np.stack([np.eye(DIM)[i] for i in range(6)], axis=0)
        result = predict_nearest(features, prototypes, CLASSES)
        self.assertEqual(result.labels, list(CLASSES))
        self.assertEqual(result.distances.shape, (6, 6))
        np.testing.assert_allclose(result.min_distance, np.diag(result.distances), atol=1e-6)
        self.assertTrue((result.min_distance < 1e-6).all())

    def test_13_prediction_picks_minimum_distance(self) -> None:
        prototypes = {
            cls: l2_normalize(np.eye(DIM)[i]) for i, cls in enumerate(CLASSES)
        }
        sample = np.zeros(DIM)
        sample[0], sample[1] = 1.0, 0.5
        result = predict_nearest(sample, prototypes, CLASSES)
        self.assertEqual(result.labels[0], "Cr")
        self.assertEqual(int(np.argmin(result.distances[0])), 0)
        self.assertAlmostEqual(float(result.min_distance[0]), float(result.distances[0, 0]), places=6)

    def test_14_threshold_accepts_and_rejects(self) -> None:
        prototypes = {cls: l2_normalize(np.eye(DIM)[i]) for i, cls in enumerate(CLASSES)}
        close = np.zeros(DIM)
        close[0] = 1.0
        far = np.zeros(DIM)
        far[0], far[2] = 1.0, 1.0  # 与 Cr 夹角 45° → 距离约 0.293
        thresholds = {cls: 0.2 for cls in CLASSES}
        result = predict_nearest(np.stack([close, far]), prototypes, CLASSES, thresholds=thresholds)
        self.assertTrue(result.accepted[0])
        self.assertFalse(result.accepted[1])

    def test_15_empty_prediction_rejected(self) -> None:
        prototypes = {cls: l2_normalize(np.eye(DIM)[i]) for i, cls in enumerate(CLASSES)}
        with self.assertRaises(PrototypeError):
            predict_nearest(np.zeros((0, DIM)), prototypes, CLASSES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
