#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2 特征通道（RBF-MMD）测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift.feature_distribution import (  # noqa: E402
    FeatureDistributionError,
    compute_mmd_score,
    median_heuristic_gamma,
    rbf_kernel,
)

DIM = 512


def sample(n: int, seed: int, shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, DIM)) + shift


class MMDTest(unittest.TestCase):
    def test_01_identical_distributions_near_zero(self) -> None:
        data = sample(30, 1)
        gamma = median_heuristic_gamma(data)
        result = compute_mmd_score(data, data.copy(), gamma)
        self.assertAlmostEqual(result["mmd_squared"], 0.0, places=10)
        self.assertEqual(result["feature_dimension"], DIM)

    def test_02_different_distributions_larger(self) -> None:
        reference = sample(30, 2)
        gamma = median_heuristic_gamma(reference)
        shifted = compute_mmd_score(reference, sample(16, 3, shift=3.0), gamma)["mmd_squared"]
        same = compute_mmd_score(reference, sample(16, 4), gamma)["mmd_squared"]
        self.assertGreater(shifted, same)

    def test_03_dimension_mismatch(self) -> None:
        with self.assertRaises(FeatureDistributionError):
            compute_mmd_score(sample(10, 5), np.random.default_rng(0).normal(size=(8, 128)), 0.5)

    def test_04_empty_arrays(self) -> None:
        with self.assertRaises(FeatureDistributionError):
            compute_mmd_score(np.zeros((0, DIM)), sample(4, 6), 0.5)
        with self.assertRaises(FeatureDistributionError):
            compute_mmd_score(sample(4, 7), np.zeros((0, DIM)), 0.5)

    def test_05_nan_rejected(self) -> None:
        bad = sample(4, 8)
        bad[0, 0] = np.nan
        with self.assertRaises(FeatureDistributionError):
            compute_mmd_score(bad, sample(4, 9), 0.5)
        bad = sample(4, 10)
        bad[0, 0] = np.inf
        with self.assertRaises(FeatureDistributionError):
            compute_mmd_score(sample(4, 11), bad, 0.5)

    def test_06_invalid_gamma(self) -> None:
        for gamma in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(FeatureDistributionError):
                compute_mmd_score(sample(4, 12), sample(4, 13), gamma)

    def test_07_gamma_fixed_by_reference(self) -> None:
        reference = sample(25, 14)
        gamma_a = median_heuristic_gamma(reference)
        candidate_a = sample(16, 15)
        candidate_b = sample(16, 16, shift=5.0)
        result_a = compute_mmd_score(reference, candidate_a, gamma_a)
        result_b = compute_mmd_score(reference, candidate_b, gamma_a)
        self.assertEqual(result_a["gamma"], result_b["gamma"])
        self.assertEqual(result_a["gamma"], gamma_a)
        # 同一 gamma 下，候选变化只影响 MMD，不影响 gamma
        self.assertNotAlmostEqual(result_a["mmd_squared"], result_b["mmd_squared"])

    def test_08_gamma_positive(self) -> None:
        gamma = median_heuristic_gamma(sample(10, 17))
        self.assertGreater(gamma, 0.0)
        self.assertTrue(np.isfinite(gamma))

    def test_09_rbf_kernel_properties(self) -> None:
        a = sample(5, 18)
        kernel = rbf_kernel(a, a, 0.1)
        self.assertEqual(kernel.shape, (5, 5))
        np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-12)
        self.assertTrue(np.all(kernel <= 1.0 + 1e-12))
        self.assertTrue(np.all(kernel >= 0.0))

    def test_10_counts_reported(self) -> None:
        reference = sample(25, 19)
        candidate = sample(16, 20)
        result = compute_mmd_score(reference, candidate, median_heuristic_gamma(reference))
        self.assertEqual(result["reference_count"], 25)
        self.assertEqual(result["candidate_count"], 16)
        self.assertGreaterEqual(result["score_B"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
