#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2 检测器（窗口构建 + 三通道打分）测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift.detector import (  # noqa: E402
    DetectorError,
    Window,
    build_clean_windows,
    build_perturbed_windows,
    chunk_sources,
    fuse_scores,
    numeric_key,
    score_window,
)
from src.drift.feature_distribution import median_heuristic_gamma  # noqa: E402
from src.drift.image_statistics import ReferenceModel, compute_image_stats  # noqa: E402

DIM = 512


def source_rows(n: int, cls: str = "Cr") -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "source_file": f"{cls}_{i}.bmp",
                "source_class": cls,
                "output_md5": f"md5{cls}{i}",
                "split": "test",
            }
        )
    return rows


def make_context(n: int = 40, cls: str = "Cr", seed: int = 0):
    rng = np.random.default_rng(seed)
    stats_by_md5 = {}
    features_by_md5 = {}
    rows_by_file = {}
    for i in range(1, n + 1):
        md5 = f"md5{cls}{i}"
        image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        stats_by_md5[md5] = compute_image_stats(image)
        features_by_md5[md5] = rng.normal(size=DIM)
        rows_by_file[f"{cls}_{i}.bmp"] = {
            "source_file": f"{cls}_{i}.bmp",
            "source_class": cls,
            "output_md5": md5,
            "split": "test",
        }
    reference_stats = list(stats_by_md5.values())[:10]
    reference_features = np.stack(list(features_by_md5.values())[:10], axis=0)
    model = ReferenceModel.fit(reference_stats, cls)
    gamma = median_heuristic_gamma(reference_features)
    return {
        "stats": stats_by_md5,
        "features": features_by_md5,
        "rows": rows_by_file,
        "model": model,
        "reference_features": reference_features,
        "gamma": gamma,
    }


class WindowTest(unittest.TestCase):
    def test_01_numeric_key(self) -> None:
        names = ["Cr_10.bmp", "Cr_2.bmp", "Cr_1.bmp"]
        self.assertEqual(sorted(names, key=numeric_key), ["Cr_1.bmp", "Cr_2.bmp", "Cr_10.bmp"])

    def test_02_chunk_no_overlap_and_tail(self) -> None:
        rows = source_rows(35)
        windows, tail = chunk_sources(rows, 16)
        self.assertEqual(len(windows), 2)
        self.assertEqual(len(tail), 3)
        ids = [row["source_file"] for window in windows for row in window]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids) + len(tail), 35)

    def test_03_clean_windows_labels(self) -> None:
        windows, _ = build_clean_windows(source_rows(32), seed=42, class_name="Cr", split="test", size=16)
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(w.is_drift == 0 and w.perturbation_type == "clean" for w in windows))
        self.assertEqual(len(windows[0].source_ids), 16)

    def test_04_perturbed_pairing(self) -> None:
        clean, _ = build_clean_windows(source_rows(16), seed=42, class_name="Cr", split="test", size=16)
        mapping = {
            ("brightness", "low"): {row["source_file"]: dict(row) for row in source_rows(16)}
        }
        perturbed = build_perturbed_windows(clean, mapping)
        self.assertEqual(len(perturbed), 1)
        self.assertEqual(perturbed[0].source_ids, clean[0].source_ids)
        self.assertEqual(perturbed[0].is_drift, 1)
        self.assertTrue(perturbed[0].window_id.startswith(clean[0].window_id))

    def test_05_missing_source_raises(self) -> None:
        clean, _ = build_clean_windows(source_rows(16), seed=42, class_name="Cr", split="test", size=16)
        with self.assertRaises(DetectorError):
            build_perturbed_windows(clean, {("brightness", "low"): {}})

    def test_06_fusion_formula(self) -> None:
        self.assertAlmostEqual(fuse_scores(2.0, 1.0, 1.0, 1.0), 2.0)
        self.assertAlmostEqual(fuse_scores(0.5, 4.0, 1.0, 2.0), 2.0)


class ScoringTest(unittest.TestCase):
    def test_07_stat_channel(self) -> None:
        ctx = make_context()
        clean, _ = build_clean_windows(source_rows(16), seed=42, class_name="Cr", split="test", size=16)
        result = score_window(
            clean[0],
            rows_by_file=ctx["rows"],
            stats_by_md5=ctx["stats"],
            features_by_md5=ctx["features"],
            model=ctx["model"],
            reference_features=ctx["reference_features"],
            gamma=ctx["gamma"],
            threshold_a=1.0,
            threshold_b=0.01,
            threshold_fusion=1.0,
        )
        self.assertGreaterEqual(result.score_A, 0.0)
        self.assertTrue(np.isfinite(result.score_A))
        self.assertEqual(result.is_drift_A, int(result.score_A > 1.0))

    def test_08_feature_and_fusion(self) -> None:
        ctx = make_context()
        clean, _ = build_clean_windows(source_rows(16), seed=42, class_name="Cr", split="test", size=16)
        result = score_window(
            clean[0],
            rows_by_file=ctx["rows"],
            stats_by_md5=ctx["stats"],
            features_by_md5=ctx["features"],
            model=ctx["model"],
            reference_features=ctx["reference_features"],
            gamma=ctx["gamma"],
            threshold_a=1.0,
            threshold_b=1.0,
            threshold_fusion=1.0,
        )
        self.assertGreaterEqual(result.score_B, 0.0)
        expected = max(result.score_A / 1.0, result.score_B / 1.0)
        self.assertAlmostEqual(result.score_fusion, expected, places=8)

    def test_09_missing_feature_raises(self) -> None:
        ctx = make_context()
        clean, _ = build_clean_windows(source_rows(4), seed=42, class_name="Cr", split="test", size=4)
        broken = Window(
            seed=42, class_name="Cr", split="test", window_id="w",
            source_ids=["Cr_999.bmp", "Cr_1.bmp"], perturbation_type="clean",
            perturbation_level="none", is_drift=0,
        )
        with self.assertRaises(DetectorError):
            score_window(
                broken,
                rows_by_file=ctx["rows"],
                stats_by_md5=ctx["stats"],
                features_by_md5=ctx["features"],
                model=ctx["model"],
                reference_features=ctx["reference_features"],
                gamma=ctx["gamma"],
                threshold_a=1.0, threshold_b=1.0, threshold_fusion=1.0,
            )

    def test_10_classes_not_mixed(self) -> None:
        windows_cr, _ = build_clean_windows(source_rows(16, "Cr"), seed=1, class_name="Cr", split="test", size=16)
        windows_in, _ = build_clean_windows(source_rows(16, "In"), seed=1, class_name="In", split="test", size=16)
        self.assertTrue(all(w.class_name == "Cr" for w in windows_cr))
        self.assertTrue(all(w.class_name == "In" for w in windows_in))
        self.assertFalse(set(windows_cr[0].source_ids) & set(windows_in[0].source_ids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
