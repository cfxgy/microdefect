#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-1 扰动算子测试（小型合成图像，不依赖真实数据集）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.perturbations import (  # noqa: E402
    BORDER_TYPE,
    IMAGE_SIZE,
    PerturbationError,
    apply_brightness,
    apply_contrast,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_perturbation,
    apply_translation,
    build_specs,
    decode_bmp,
    derive_noise_seed,
    encode_bmp,
    gaussian_kernel_size,
    image_statistics,
    specs_config_signature,
    validate_image,
)


def sample_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
    # 加一点结构，避免纯噪声导致统计不稳定
    base[:100, :] = np.clip(base[:100, :].astype(np.int16) + 20, 0, 255)
    return base.astype(np.uint8)


class RangeAndShapeTest(unittest.TestCase):
    def test_01_brightness_range_and_formula(self) -> None:
        image = sample_image(1)
        out = apply_brightness(image, 1.30)
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLessEqual(int(out.max()), 255)
        expected = np.floor(np.clip(image.astype(np.float32) * 1.3, 0, 255) + 0.5).astype(np.uint8)
        np.testing.assert_array_equal(out, expected)
        self.assertGreater(float(out.mean()), float(image.mean()))

    def test_02_contrast_range_and_formula(self) -> None:
        image = sample_image(2)
        out = apply_contrast(image, 0.70)
        expected = np.floor(
            np.clip(0.7 * (image.astype(np.float32) - image.mean()) + image.mean(), 0, 255) + 0.5
        ).astype(np.uint8)
        np.testing.assert_array_equal(out, expected)
        self.assertLess(float(out.std()), float(image.std()))

    def test_03_noise_reproducible_same_seed(self) -> None:
        image = sample_image(3)
        spec = [s for s in build_specs() if s.name == "noise_low"][0]
        first = apply_perturbation(image, spec, seed=42, source_md5="a" * 32)
        second = apply_perturbation(image, spec, seed=42, source_md5="a" * 32)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, image))

    def test_04_noise_differs_for_different_seed(self) -> None:
        image = sample_image(4)
        spec = [s for s in build_specs() if s.name == "noise_low"][0]
        a = apply_perturbation(image, spec, seed=42, source_md5="a" * 32)
        b = apply_perturbation(image, spec, seed=123, source_md5="a" * 32)
        self.assertFalse(np.array_equal(a, b))

    def test_05_blur_keeps_size(self) -> None:
        image = sample_image(5)
        out = apply_gaussian_blur(image, 1.5)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(gaussian_kernel_size(0.8) % 2, 1)
        self.assertEqual(gaussian_kernel_size(1.5) % 2, 1)

    def test_06_translation_keeps_size(self) -> None:
        image = sample_image(6)
        for tx, ty in ((2, 2), (5, 5)):
            out = apply_translation(image, tx, ty)
            self.assertEqual(out.shape, image.shape)
        # 整数平移 2 像素：内部区域应精确等于源图移位结果
        out = apply_translation(image, 2, 2)
        np.testing.assert_array_equal(out[2:, 2:], image[:-2, :-2])

    def test_07_all_specs_grayscale_and_size(self) -> None:
        image = sample_image(7)
        image_md5 = "b" * 32
        for spec in build_specs():
            out = apply_perturbation(image, spec, seed=7, source_md5=image_md5)
            self.assertEqual(out.ndim, 2, spec.name)
            self.assertEqual(out.dtype, np.uint8, spec.name)
            self.assertEqual(out.shape, IMAGE_SIZE, spec.name)
            self.assertGreaterEqual(int(out.min()), 0)
            self.assertLessEqual(int(out.max()), 255)

    def test_08_invalid_parameters(self) -> None:
        image = sample_image(8)
        with self.assertRaises(PerturbationError):
            apply_brightness(image, 0.0)
        with self.assertRaises(PerturbationError):
            apply_contrast(image, -1.0)
        with self.assertRaises(PerturbationError):
            apply_gaussian_blur(image, 0.0)
        with self.assertRaises(PerturbationError):
            apply_translation(image, 1.5, 2)  # type: ignore[arg-type]
        with self.assertRaises(PerturbationError):
            apply_gaussian_noise(image, -1.0, np.random.default_rng(0))
        with self.assertRaises(PerturbationError):
            validate_image(image.astype(np.float32))
        with self.assertRaises(PerturbationError):
            validate_image(image[:, :, None])

    def test_09_seed_derivation_stable(self) -> None:
        a = derive_noise_seed(42, "c" * 32, "noise_low")
        b = derive_noise_seed(42, "c" * 32, "noise_low")
        self.assertEqual(a, b)
        self.assertNotEqual(a, derive_noise_seed(43, "c" * 32, "noise_low"))
        self.assertNotEqual(a, derive_noise_seed(42, "d" * 32, "noise_low"))
        self.assertNotEqual(a, derive_noise_seed(42, "c" * 32, "noise_high"))

    def test_10_bmp_roundtrip_and_stats(self) -> None:
        image = sample_image(9)
        encoded = encode_bmp(image)
        decoded = decode_bmp(encoded)
        np.testing.assert_array_equal(decoded, image)

        stats = image_statistics(image)
        self.assertAlmostEqual(stats["brightness"], float(image.mean()), places=6)
        self.assertAlmostEqual(stats["contrast"], float(image.std()), places=6)
        self.assertGreaterEqual(stats["sharpness"], 0.0)

    def test_11_repeat_generation_identical_bytes(self) -> None:
        image = sample_image(10)
        a = encode_bmp(apply_brightness(image, 0.7))
        b = encode_bmp(apply_brightness(image, 0.7))
        self.assertEqual(a, b)

    def test_12_source_not_mutated(self) -> None:
        image = sample_image(11)
        snapshot = image.copy()
        for spec in build_specs():
            apply_perturbation(image, spec, seed=1, source_md5="e" * 32)
        np.testing.assert_array_equal(image, snapshot)

    def test_13_config_signature_stable(self) -> None:
        self.assertEqual(specs_config_signature(), specs_config_signature())
        self.assertEqual(len(build_specs()), 10)
        names = [s.name for s in build_specs()]
        self.assertEqual(len(set(names)), 10)
        self.assertEqual(BORDER_TYPE, 4)  # cv2.BORDER_REFLECT_101


if __name__ == "__main__":
    unittest.main(verbosity=2)
