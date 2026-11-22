#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-2.3.2.1 协议余弦评分测试。"""

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

from src.anomaly.mvtec_protocol_scoring import (  # noqa: E402
    ProtocolScoringError,
    score_image_cosine,
    score_spatial_cosine,
)


def cosine_distance_matrix(query: np.ndarray, memory: np.ndarray) -> np.ndarray:
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    m = memory / np.linalg.norm(memory, axis=1, keepdims=True)
    return 1.0 - q @ m.T


class ImageCosineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = np.random.default_rng(0).normal(size=(7, 512)).astype(np.float32)
        self.query = np.random.default_rng(1).normal(size=(512,)).astype(np.float32)

    def test_01_matches_manual(self) -> None:
        result = score_image_cosine(self.query, self.memory)
        manual = float(cosine_distance_matrix(self.query[None, :], self.memory).min())
        self.assertAlmostEqual(float(result.distances[0]), manual, places=5)
        self.assertEqual(int(result.nearest_indices[0]),
                         int(np.argmax((self.memory @ self.query) /
                                       (np.linalg.norm(self.memory, axis=1) * np.linalg.norm(self.query)))))

    def test_02_same_direction_zero(self) -> None:
        result = score_image_cosine(self.memory[3] * 5.0, self.memory)
        self.assertAlmostEqual(float(result.distances[0]), 0.0, places=5)

    def test_03_orthogonal_one(self) -> None:
        memory = np.eye(512, dtype=np.float32)
        query = np.zeros(512, dtype=np.float32)
        query[0] = 1.0
        result = score_image_cosine(query, memory)
        self.assertAlmostEqual(float(result.distances[0]), 0.0, places=6)
        query[0], query[1] = 0.0, 1.0
        result = score_image_cosine(query, memory[:1])
        self.assertAlmostEqual(float(result.distances[0]), 1.0, places=6)

    def test_04_opposite_two(self) -> None:
        memory = np.eye(512, dtype=np.float32)
        # 仅保留一个记忆向量，避免正交基向量（cos=0）成为最近邻
        result = score_image_cosine(-memory[0], memory[:1])
        self.assertAlmostEqual(float(result.distances[0]), 2.0, places=6)

    def test_05_batch_and_range(self) -> None:
        queries = np.stack([self.query, self.memory[0], -self.memory[1]])
        result = score_image_cosine(queries, self.memory)
        self.assertEqual(result.distances.shape, (3,))
        self.assertTrue(np.all(result.distances >= 0) and np.all(result.distances <= 2))
        self.assertTrue(np.all(result.max_cosine >= -1) and np.all(result.max_cosine <= 1))
        self.assertAlmostEqual(float(result.distances[1]), 0.0, places=6)

    def test_06_chunk_equivalence(self) -> None:
        a = score_image_cosine(self.query, self.memory, memory_chunk=1)
        b = score_image_cosine(self.query, self.memory, memory_chunk=8192)
        self.assertEqual(float(a.distances[0]), float(b.distances[0]))

    def test_07_zero_norm_rejected(self) -> None:
        with self.assertRaises(ProtocolScoringError):
            score_image_cosine(np.zeros(512, dtype=np.float32), self.memory)
        bad_memory = self.memory.copy()
        bad_memory[2] = 0.0
        with self.assertRaises(ProtocolScoringError):
            score_image_cosine(self.query, bad_memory)

    def test_08_empty_and_shape_and_finite(self) -> None:
        with self.assertRaises(ProtocolScoringError):
            score_image_cosine(self.query, np.zeros((0, 512), dtype=np.float32))
        with self.assertRaises(ProtocolScoringError):
            score_image_cosine(self.query, np.zeros((4, 256), dtype=np.float32))
        bad = self.query.copy()
        bad[0] = np.nan
        with self.assertRaises(ProtocolScoringError):
            score_image_cosine(bad, self.memory)
        bad_memory = self.memory.copy()
        bad_memory[0, 0] = np.inf
        with self.assertRaises(ProtocolScoringError):
            score_image_cosine(self.query, bad_memory)


class SpatialCosineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = np.random.default_rng(2).normal(size=(4, 256, 14, 14)).astype(np.float32)
        self.feature = np.random.default_rng(3).normal(size=(256, 14, 14)).astype(np.float32)

    def test_09_matches_manual(self) -> None:
        result = score_spatial_cosine(self.feature, self.memory, (64, 64))
        memory = self.memory.transpose(0, 2, 3, 1).reshape(-1, 256)
        query = self.feature.reshape(256, 196).T
        manual = cosine_distance_matrix(query, memory).min(axis=1).reshape(14, 14)
        np.testing.assert_allclose(result.patch_distances[0], manual, atol=1e-5)

    def test_10_scale_invariance(self) -> None:
        """patch 逐向量 L2 归一化后，整体缩放不应改变距离。"""
        a = score_spatial_cosine(self.feature, self.memory, (32, 32))
        b = score_spatial_cosine(self.feature * 7.5, self.memory * 3.0, (32, 32))
        np.testing.assert_allclose(a.patch_distances, b.patch_distances, atol=1e-5)

    def test_11_same_patch_zero(self) -> None:
        result = score_spatial_cosine(self.memory[2], self.memory, (32, 32))
        np.testing.assert_allclose(result.patch_distances[0], 0.0, atol=1e-5)

    def test_12_range_and_finite(self) -> None:
        result = score_spatial_cosine(self.feature, self.memory, (40, 40))
        self.assertTrue(np.all(result.patch_distances >= 0))
        self.assertTrue(np.all(result.patch_distances <= 2))
        self.assertTrue(np.isfinite(result.patch_distances).all())
        self.assertTrue(np.isfinite(result.heatmaps).all())

    def test_13_heatmap_size_and_interpolation(self) -> None:
        result = score_spatial_cosine(self.feature, self.memory, (56, 56))
        self.assertEqual(result.heatmaps.shape, (1, 56, 56))
        patch = torch.as_tensor(result.patch_distances[0:1, None])
        expected = F.interpolate(patch, size=(56, 56), mode="bilinear", align_corners=False)
        np.testing.assert_allclose(result.heatmaps[0], expected.numpy()[0, 0], atol=1e-6)
        wrong = F.interpolate(patch, size=(56, 56), mode="bilinear", align_corners=True)
        self.assertFalse(np.allclose(result.heatmaps[0], wrong.numpy()[0, 0], atol=1e-6))

    def test_14_batch_and_chunk(self) -> None:
        features = np.stack([self.feature, self.memory[0], self.memory[3]])
        a = score_spatial_cosine(features, self.memory, (32, 32), memory_chunk=1)
        b = score_spatial_cosine(features, self.memory, (32, 32), memory_chunk=4096)
        self.assertEqual(a.patch_distances.shape, (3, 14, 14))
        np.testing.assert_allclose(a.heatmaps, b.heatmaps, atol=1e-6)
        np.testing.assert_allclose(a.patch_distances[1], 0.0, atol=1e-5)

    def test_15_zero_norm_patch_rejected(self) -> None:
        bad = self.feature.copy()
        bad[:, 0, 0] = 0.0  # 某个 patch 全零
        with self.assertRaises(ProtocolScoringError):
            score_spatial_cosine(bad, self.memory, (32, 32))
        bad_memory = self.memory.copy()
        bad_memory[1, :, 3, 3] = 0.0
        with self.assertRaises(ProtocolScoringError):
            score_spatial_cosine(self.feature, bad_memory, (32, 32))

    def test_16_empty_shape_finite(self) -> None:
        with self.assertRaises(ProtocolScoringError):
            score_spatial_cosine(self.feature, np.zeros((0, 256), dtype=np.float32), (32, 32))
        with self.assertRaises(ProtocolScoringError):
            score_spatial_cosine(np.zeros((128, 14, 14), dtype=np.float32), self.memory, (32, 32))
        bad = self.feature.copy()
        bad[0, 0, 0] = np.inf
        with self.assertRaises(ProtocolScoringError):
            score_spatial_cosine(bad, self.memory, (32, 32))

    def test_17_flat_memory_equivalent(self) -> None:
        flat = self.memory.transpose(0, 2, 3, 1).reshape(-1, 256)
        a = score_spatial_cosine(self.feature, self.memory, (32, 32))
        b = score_spatial_cosine(self.feature, flat, (32, 32))
        np.testing.assert_allclose(a.patch_distances, b.patch_distances, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 不可用")
    def test_18_cpu_gpu_consistency(self) -> None:
        memory = np.random.default_rng(4).normal(size=(9, 512)).astype(np.float32)
        query = np.random.default_rng(5).normal(size=(2, 512)).astype(np.float32)
        a = score_image_cosine(query, memory, device="cpu")
        b = score_image_cosine(query, memory, device="cuda")
        np.testing.assert_allclose(a.distances, b.distances, atol=1e-6)
        c = score_spatial_cosine(self.feature, self.memory, (32, 32), device="cpu")
        d = score_spatial_cosine(self.feature, self.memory, (32, 32), device="cuda")
        np.testing.assert_allclose(c.patch_distances, d.patch_distances, atol=1e-6)
        np.testing.assert_allclose(c.heatmaps, d.heatmaps, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
