#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2 图像统计通道测试。"""

from __future__ import annotations

import sys
import unittest
import uuid
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift.image_statistics import (  # noqa: E402
    MIN_SCALE,
    ReferenceModel,
    StatisticsError,
    compute_image_stats,
    load_gray,
    stats_matrix,
    window_stat_means,
)


def base_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(200, 200), dtype=np.uint8)


class ImageStatsTest(unittest.TestCase):
    def test_01_normal_gray(self) -> None:
        image = base_image(1)
        stats = compute_image_stats(image)
        self.assertAlmostEqual(stats["mean_intensity"], float(image.mean()), places=6)
        self.assertAlmostEqual(stats["contrast"], float(image.std()), places=6)
        self.assertGreater(stats["laplacian_variance"], 0.0)

    def test_02_brightness_changes_mean(self) -> None:
        image = base_image(2)
        bright = np.clip(image.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        self.assertGreater(
            compute_image_stats(bright)["mean_intensity"],
            compute_image_stats(image)["mean_intensity"],
        )

    def test_03_low_contrast_reduces_std(self) -> None:
        image = base_image(3)
        low = np.clip(0.5 * (image.astype(np.float32) - image.mean()) + image.mean(), 0, 255).astype(np.uint8)
        self.assertLess(compute_image_stats(low)["contrast"], compute_image_stats(image)["contrast"])

    def test_04_blur_reduces_laplacian(self) -> None:
        import cv2

        image = base_image(4)
        blurred = cv2.GaussianBlur(image, (7, 7), 1.5)
        self.assertLess(
            compute_image_stats(blurred)["laplacian_variance"],
            compute_image_stats(image)["laplacian_variance"],
        )

    def test_05_empty_and_bad_dims(self) -> None:
        with self.assertRaises(StatisticsError):
            compute_image_stats(np.zeros((0, 0), dtype=np.uint8))
        with self.assertRaises(StatisticsError):
            compute_image_stats(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_06_nan_inf_rejected(self) -> None:
        image = base_image(5).astype(np.float32)
        image[0, 0] = np.nan
        with self.assertRaises(StatisticsError):
            compute_image_stats(image)
        image[0, 0] = np.inf
        with self.assertRaises(StatisticsError):
            compute_image_stats(image)

    def test_07_corrupt_file(self) -> None:
        tmp = PROJECT_ROOT / f"tmp_p42_stats_{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            bad = tmp / "bad.bmp"
            bad.write_bytes(b"not-an-image")
            with self.assertRaises(StatisticsError):
                load_gray(bad)
            with self.assertRaises(StatisticsError):
                load_gray(tmp / "missing.bmp")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_08_mad_zero_falls_back(self) -> None:
        constant = [compute_image_stats(np.full((200, 200), 100, dtype=np.uint8)) for _ in range(5)]
        model = ReferenceModel.fit(constant, "Cr")
        self.assertTrue(np.all(model.mad == 0))
        self.assertTrue(np.all(model.scale >= MIN_SCALE))
        self.assertTrue(all(s in {"std_fallback", "epsilon"} for s in model.scale_source))

    def test_09_reference_model_scores(self) -> None:
        stats = [compute_image_stats(base_image(10 + i)) for i in range(20)]
        model = ReferenceModel.fit(stats, "In")
        self.assertEqual(model.n_samples, 20)
        matrix = stats_matrix(stats)
        inside = model.score(matrix.mean(axis=0))
        self.assertGreaterEqual(inside["score_A"], 0.0)
        shifted = model.score(matrix.mean(axis=0) + model.scale * 5)
        self.assertGreater(shifted["score_A"], inside["score_A"])

    def test_10_window_stat_means(self) -> None:
        stats = [compute_image_stats(base_image(20 + i)) for i in range(4)]
        mean = window_stat_means(stats)
        self.assertEqual(mean.shape, (3,))
        self.assertAlmostEqual(mean[0], np.mean([s["mean_intensity"] for s in stats]))

    def test_11_real_bmp_roundtrip(self) -> None:
        tmp = PROJECT_ROOT / f"tmp_p42_stats_{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            path = tmp / "x.bmp"
            Image.fromarray(base_image(30), mode="L").save(path, format="BMP")
            image = load_gray(path)
            self.assertEqual(image.shape, (200, 200))
            self.assertEqual(image.dtype, np.uint8)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
