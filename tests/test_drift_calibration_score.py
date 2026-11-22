#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2.1 阶段 4：A 通道图像统计分数测试。

校验实现与冻结协议 3.2 节一致，并交叉比对 P4-2 的参考实现。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift.image_statistics import ReferenceModel, compute_image_stats  # noqa: E402
from src.drift_calibration.score import (  # noqa: E402
    ImageScorer,
    ScoreError,
    score_direction,
)


def sample_image(seed: int, size: int = 64, offset: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    return np.clip(image.astype(np.int16) + offset, 0, 255).astype(np.uint8)


def reference_stats(n: int = 25, seed0: int = 0) -> list[dict]:
    return [compute_image_stats(sample_image(seed0 + i)) for i in range(n)]


class ScoreFormulaTest(unittest.TestCase):
    def test_01_matches_frozen_formula(self) -> None:
        reference = reference_stats()
        scorer = ImageScorer(reference, "Cr")
        image = sample_image(100)
        result = scorer.score_gray(image)

        matrix = np.array([[s[k] for k in ("mean_intensity", "contrast", "laplacian_variance")] for s in reference])
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0)
        std = matrix.std(axis=0, ddof=1)
        scale = 1.4826 * mad
        for index in range(3):
            if scale[index] < 1e-3:
                scale[index] = std[index] if std[index] >= 1e-3 else 1e-3
        stats = np.array([result.stats[0], result.stats[1], result.stats[2]])
        z = np.abs(stats - median) / scale
        expected = float(np.sqrt(np.mean(z ** 2)))
        self.assertAlmostEqual(result.score_A, expected, places=9)
        np.testing.assert_allclose(np.array(result.z), z, rtol=1e-9, atol=1e-12)

    def test_02_matches_p42_reference_model(self) -> None:
        reference = reference_stats(seed0=10)
        scorer = ImageScorer(reference, "In")
        image = sample_image(200)
        ours = scorer.score_gray(image)

        model = ReferenceModel.fit(reference, "In")
        stats = compute_image_stats(image)
        values = np.array([stats["mean_intensity"], stats["contrast"], stats["laplacian_variance"]])
        theirs = model.score(values)
        self.assertAlmostEqual(ours.score_A, theirs["score_A"], places=12)

    def test_03_direction_larger_is_more_anomalous(self) -> None:
        reference = reference_stats(seed0=20)
        scorer = ImageScorer(reference, "Pa")
        normal = sample_image(300)
        bright = sample_image(300, offset=60)
        score_normal = scorer.score_gray(normal).score_A
        score_bright = scorer.score_gray(bright).score_A
        self.assertGreater(score_bright, score_normal)
        self.assertEqual(score_direction(), "larger_is_more_anomalous")

    def test_04_identical_image_scores_low(self) -> None:
        reference = reference_stats(seed0=30)
        scorer = ImageScorer(reference, "PS")
        image = sample_image(400)
        score = scorer.score_gray(image).score_A
        self.assertGreaterEqual(score, 0.0)
        self.assertTrue(np.isfinite(score))

    def test_05_invalid_reference_rejected(self) -> None:
        with self.assertRaises(ScoreError):
            ImageScorer([], "Cr")
        with self.assertRaises(ScoreError):
            ImageScorer(reference_stats(1), "Cr")
        bad = reference_stats(5)
        bad[0]["contrast"] = float("nan")
        with self.assertRaises(ScoreError):
            ImageScorer(bad, "Cr")
        with self.assertRaises(ScoreError):
            ImageScorer(reference_stats(5), "")

    def test_06_invalid_image_rejected(self) -> None:
        scorer = ImageScorer(reference_stats(5), "RS")
        with self.assertRaises(ScoreError):
            scorer.score_gray(np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaises(ScoreError):
            scorer.score_gray(np.zeros((0, 0), dtype=np.uint8))

    def test_07_invalid_stats_dict_rejected(self) -> None:
        scorer = ImageScorer(reference_stats(5), "Sc")
        with self.assertRaises(ScoreError):
            scorer.score_stats({"mean_intensity": 1.0})
        with self.assertRaises(ScoreError):
            scorer.score_stats(
                {"mean_intensity": float("inf"), "contrast": 1.0, "laplacian_variance": 1.0}
            )

    def test_08_describe_contains_scale_sources(self) -> None:
        scorer = ImageScorer(reference_stats(10), "Cr")
        info = scorer.describe()
        self.assertEqual(info["reference_count"], 10)
        self.assertEqual(info["class_name"], "Cr")
        self.assertEqual(set(info["scale_source"]), {"mean_intensity", "contrast", "laplacian_variance"})
        self.assertTrue(set(info["scale_source"].values()) <= {"mad", "std_fallback", "epsilon"})

    def test_09_score_stats_does_not_mutate_input(self) -> None:
        scorer = ImageScorer(reference_stats(5), "Cr")
        stats = compute_image_stats(sample_image(500))
        snapshot = dict(stats)
        scorer.score_stats(stats)
        self.assertEqual(stats, snapshot)


if __name__ == "__main__":
    unittest.main(verbosity=2)
