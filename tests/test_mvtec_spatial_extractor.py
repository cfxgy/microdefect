#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-2.2 空间特征提取器测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.features.extractor import FeatureExtractor  # noqa: E402
from src.features.mvtec_spatial_extractor import (  # noqa: E402
    SPATIAL_CHANNELS,
    SPATIAL_FEATURE_VERSION,
    SPATIAL_GRID,
    MvtecSpatialExtractor,
    load_image_rgb_mvtec,
    weights_sha256,
)

DATA = PROJECT_ROOT / "datasets" / "mvtec_ad"
RGB_NORMAL = DATA / "bottle/train/good/000.png"
GRAY_NORMAL = DATA / "grid/train/good/000.png"
RGB_DEFECT = DATA / "bottle/test/broken_large/000.png"
GRAY_DEFECT = DATA / "zipper/test/broken_teeth/000.png"


class SpatialExtractorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = MvtecSpatialExtractor(device="cpu", batch_size=2)
        cls.has_cuda = torch.cuda.is_available()

    def test_01_single_shapes(self) -> None:
        g = self.extractor.extract_global_path(RGB_NORMAL)
        s = self.extractor.extract_spatial_path(RGB_NORMAL)
        self.assertEqual(g.shape, (512,))
        self.assertEqual(s.shape, (SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID))
        self.assertEqual(g.dtype, np.float32)
        self.assertEqual(s.dtype, np.float32)
        self.assertTrue(np.isfinite(g).all())
        self.assertTrue(np.isfinite(s).all())

    def test_02_batch_shapes(self) -> None:
        paths = [RGB_NORMAL, GRAY_NORMAL, RGB_DEFECT, GRAY_DEFECT]
        g = self.extractor.extract_global_paths(paths)
        s = self.extractor.extract_spatial_paths(paths)
        self.assertEqual(g.shape, (4, 512))
        self.assertEqual(s.shape, (4, SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID))
        self.assertTrue(np.isfinite(g).all() and np.isfinite(s).all())

    def test_03_eval_and_frozen(self) -> None:
        self.assertFalse(self.extractor.model.training)
        self.assertFalse(self.extractor._trunk.training)
        self.assertTrue(all(not p.requires_grad for p in self.extractor.model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.extractor._trunk.parameters()))

    def test_04_color_preserved(self) -> None:
        rgb = np.asarray(load_image_rgb_mvtec(RGB_NORMAL))
        gray = np.asarray(load_image_rgb_mvtec(GRAY_NORMAL))
        self.assertEqual(rgb.shape[2], 3)
        self.assertGreater(int(rgb.std(axis=2).max()), 0)  # 彩色通道确实不同
        self.assertEqual(int(np.abs(gray[:, :, 0].astype(int) - gray[:, :, 1]).max()), 0)

    def test_05_repeat_deterministic(self) -> None:
        a_g = self.extractor.extract_global_path(RGB_NORMAL)
        b_g = self.extractor.extract_global_path(RGB_NORMAL)
        a_s = self.extractor.extract_spatial_path(RGB_NORMAL)
        b_s = self.extractor.extract_spatial_path(RGB_NORMAL)
        self.assertTrue(np.array_equal(a_g, b_g))
        self.assertTrue(np.array_equal(a_s, b_s))

    def test_06_both_matches_single(self) -> None:
        paths = [RGB_NORMAL, GRAY_NORMAL]
        g_both, s_both = self.extractor.extract_paths_both(paths)
        for index, path in enumerate(paths):
            self.assertTrue(np.array_equal(g_both[index], self.extractor.extract_global_path(path)))
            self.assertTrue(np.array_equal(s_both[index], self.extractor.extract_spatial_path(path)))

    def test_07_batch_equals_single(self) -> None:
        paths = [RGB_NORMAL, GRAY_NORMAL, RGB_DEFECT]
        g_batch = self.extractor.extract_global_paths(paths)
        s_batch = self.extractor.extract_spatial_paths(paths)
        for index, path in enumerate(paths):
            np.testing.assert_allclose(
                g_batch[index], self.extractor.extract_global_path(path), atol=1e-6, rtol=0
            )
            np.testing.assert_allclose(
                s_batch[index], self.extractor.extract_spatial_path(path), atol=1e-6, rtol=0
            )

    def test_08_p2_parity_on_grayscale(self) -> None:
        p2_extractor = FeatureExtractor(device="cpu", batch_size=1)
        expected = p2_extractor.extract_path(GRAY_NORMAL)
        actual = self.extractor.extract_global_path(GRAY_NORMAL)
        self.assertTrue(np.array_equal(expected, actual))

    def test_09_metadata(self) -> None:
        info = self.extractor.describe()
        self.assertEqual(info["layer"], "layer3")
        self.assertEqual(info["spatial_shape"], [256, 14, 14])
        self.assertEqual(info["global_dim"], 512)
        self.assertEqual(info["weights_sha256"], weights_sha256())
        self.assertNotEqual(info["global_feature_version"], SPATIAL_FEATURE_VERSION)
        self.assertEqual(info["dtype"], "float32")

    def test_10_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            self.extractor.extract_spatial_tensor(torch.zeros(3, 224, 224))
        with self.assertRaises(ValueError):
            self.extractor.extract_spatial_tensor(torch.zeros(1, 1, 224, 224))
        with self.assertRaises(ValueError):
            self.extractor.extract_spatial_tensor(torch.zeros(1, 3, 128, 128))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 不可用")
    def test_11_cpu_gpu_consistency(self) -> None:
        gpu = MvtecSpatialExtractor(device="cuda", batch_size=2)
        paths = [RGB_NORMAL, GRAY_NORMAL]
        g_cpu, s_cpu = self.extractor.extract_paths_both(paths)
        g_gpu, s_gpu = gpu.extract_paths_both(paths)
        self.assertEqual(g_cpu.shape, g_gpu.shape)
        self.assertEqual(s_cpu.shape, s_gpu.shape)
        self.assertEqual(g_gpu.dtype, np.float32)
        gp = np.abs(g_cpu - g_gpu)
        sp = np.abs(s_cpu - s_gpu)
        self.assertLess(float(gp.max()), 1e-3)
        self.assertLess(float(sp.max()), 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
