#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-2.3.2 异常评分模块测试（人工构造数据，不依赖完整数据集）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.anomaly.mvtec_scoring import (  # noqa: E402
    ScoringError,
    pairwise_euclidean,
    score_image,
    score_spatial,
)


class ImageScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.memory = rng.normal(size=(7, 512)).astype(np.float32)
        self.query = rng.normal(size=(512,)).astype(np.float32)

    def test_01_image_nn_matches_manual(self) -> None:
        result = score_image(self.query, self.memory)
        manual = float(np.linalg.norm(self.memory - self.query, axis=1).min())
        self.assertAlmostEqual(float(result.distances[0]), manual, places=4)
        nearest = int(np.argmin(np.linalg.norm(self.memory - self.query, axis=1)))
        self.assertEqual(int(result.nearest_indices[0]), nearest)

    def test_02_non_negative(self) -> None:
        result = score_image(self.query, self.memory)
        self.assertGreaterEqual(float(result.distances[0]), 0.0)

    def test_03_identical_feature_zero(self) -> None:
        result = score_image(self.memory[3], self.memory)
        self.assertEqual(float(result.distances[0]), 0.0)
        self.assertEqual(int(result.nearest_indices[0]), 3)

    def test_04_batch(self) -> None:
        queries = np.stack([self.query, self.memory[0], self.memory[5]])
        result = score_image(queries, self.memory)
        self.assertEqual(result.distances.shape, (3,))
        self.assertEqual(float(result.distances[1]), 0.0)
        self.assertEqual(float(result.distances[2]), 0.0)
        manual = np.linalg.norm(self.memory[None, :, :] - queries[:, None, :], axis=2).min(axis=1)
        np.testing.assert_allclose(result.distances, manual, atol=1e-4)

    def test_05_chunked_equals_direct(self) -> None:
        a = score_image(self.query, self.memory, memory_chunk=1)
        b = score_image(self.query, self.memory, memory_chunk=8192)
        self.assertEqual(float(a.distances[0]), float(b.distances[0]))

    def test_06_empty_memory_rejected(self) -> None:
        with self.assertRaises(ScoringError):
            score_image(self.query, np.zeros((0, 512), dtype=np.float32))

    def test_07_shape_errors(self) -> None:
        with self.assertRaises(ScoringError):
            score_image(self.query, np.zeros((4, 256), dtype=np.float32))
        with self.assertRaises(ScoringError):
            score_image(np.zeros((2, 3), dtype=np.float32), self.memory)

    def test_08_nonfinite_rejected(self) -> None:
        bad = self.query.copy()
        bad[0] = np.nan
        with self.assertRaises(ScoringError):
            score_image(bad, self.memory)
        bad_mem = self.memory.copy()
        bad_mem[0, 0] = np.inf
        with self.assertRaises(ScoringError):
            score_image(self.query, bad_mem)

    def test_09_pairwise_matrix(self) -> None:
        matrix = pairwise_euclidean(self.memory[:3], self.memory[:3])
        manual = np.linalg.norm(self.memory[None, :3, :] - self.memory[:3, None, :], axis=2)
        np.testing.assert_allclose(matrix, manual, atol=1e-4)


class SpatialScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.memory = rng.normal(size=(4, 256, 14, 14)).astype(np.float32)
        self.feature = rng.normal(size=(256, 14, 14)).astype(np.float32)

    def test_10_spatial_nn_matches_manual(self) -> None:
        result = score_spatial(self.feature, self.memory, (64, 64))
        flat_memory = self.memory.transpose(0, 2, 3, 1).reshape(-1, 256)
        flat_feature = self.feature.reshape(256, 196).T  # [196,256]
        manual = np.linalg.norm(flat_feature[:, None, :] - flat_memory[None, :, :], axis=2).min(axis=1)
        np.testing.assert_allclose(
            result.patch_distances[0].ravel(), manual.reshape(14, 14).ravel(), atol=1e-4
        )

    def test_11_non_negative_and_finite(self) -> None:
        result = score_spatial(self.feature, self.memory, (32, 32))
        self.assertTrue(np.isfinite(result.patch_distances).all())
        self.assertTrue((result.patch_distances >= 0).all())
        self.assertTrue(np.isfinite(result.heatmaps).all())

    def test_12_heatmap_size(self) -> None:
        for size in ((64, 64), (100, 80)):
            result = score_spatial(self.feature, self.memory, size)
            self.assertEqual(result.heatmaps.shape, (1, size[0], size[1]))
            self.assertEqual(result.patch_distances.shape, (1, 14, 14))

    def test_13_bilinear_align_corners_false(self) -> None:
        result = score_spatial(self.feature, self.memory, (56, 56))
        patch = torch.as_tensor(result.patch_distances[0:1, None])
        expected = F.interpolate(patch, size=(56, 56), mode="bilinear", align_corners=False)
        np.testing.assert_allclose(
            result.heatmaps[0], expected.numpy()[0, 0], atol=1e-6
        )
        wrong = F.interpolate(patch, size=(56, 56), mode="bilinear", align_corners=True)
        self.assertFalse(np.allclose(result.heatmaps[0], wrong.numpy()[0, 0], atol=1e-6))

    def test_14_identical_patch_zero(self) -> None:
        feature = self.memory[2]
        result = score_spatial(feature, self.memory, (28, 28))
        np.testing.assert_allclose(result.patch_distances[0], 0.0, atol=1e-5)

    def test_15_batch_and_chunk(self) -> None:
        features = np.stack([self.feature, self.memory[0], self.memory[3]])
        result = score_spatial(features, self.memory, (32, 32), memory_chunk=1)
        self.assertEqual(result.patch_distances.shape, (3, 14, 14))
        np.testing.assert_allclose(result.patch_distances[1], 0.0, atol=1e-5)
        np.testing.assert_allclose(result.patch_distances[2], 0.0, atol=1e-5)
        reference = score_spatial(features, self.memory, (32, 32), memory_chunk=4096)
        np.testing.assert_allclose(result.heatmaps, reference.heatmaps, atol=1e-6)

    def test_16_flat_memory_equivalent(self) -> None:
        flat = self.memory.transpose(0, 2, 3, 1).reshape(-1, 256)
        a = score_spatial(self.feature, self.memory, (32, 32))
        b = score_spatial(self.feature, flat, (32, 32))
        np.testing.assert_allclose(a.patch_distances, b.patch_distances, atol=1e-6)

    def test_17_empty_and_shape_errors(self) -> None:
        with self.assertRaises(ScoringError):
            score_spatial(self.feature, np.zeros((0, 256), dtype=np.float32), (32, 32))
        with self.assertRaises(ScoringError):
            score_spatial(self.feature, np.zeros((4, 128, 14, 14), dtype=np.float32), (32, 32))
        with self.assertRaises(ScoringError):
            score_spatial(np.zeros((128, 14, 14), dtype=np.float32), self.memory, (32, 32))

    def test_18_nonfinite_rejected(self) -> None:
        bad = self.feature.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ScoringError):
            score_spatial(bad, self.memory, (32, 32))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 不可用")
    def test_19_cpu_gpu_consistency(self) -> None:
        queries = np.random.default_rng(3).normal(size=(2, 512)).astype(np.float32)
        memory = np.random.default_rng(4).normal(size=(11, 512)).astype(np.float32)
        a = score_image(queries, memory, device="cpu")
        b = score_image(queries, memory, device="cuda")
        np.testing.assert_allclose(a.distances, b.distances, atol=1e-5)
        c = score_spatial(self.feature, self.memory, (32, 32), device="cpu")
        d = score_spatial(self.feature, self.memory, (32, 32), device="cuda")
        np.testing.assert_allclose(c.patch_distances, d.patch_distances, atol=1e-5)
        np.testing.assert_allclose(c.heatmaps, d.heatmaps, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
