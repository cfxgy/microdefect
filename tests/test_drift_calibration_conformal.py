#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2.1 阶段 4：conformal p 值与告警判定测试。"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift_calibration.conformal import (  # noqa: E402
    CalibrationSet,
    ConformalError,
    conformal_p_value,
    decide,
    decide_raw,
    p_min,
)
from src.drift_calibration.protocol import ALPHA, N_CALIBRATION  # noqa: E402

SCORES = [round(0.10 + 0.01 * i, 4) for i in range(N_CALIBRATION)]  # 0.10 .. 0.34


class ConformalMathTest(unittest.TestCase):
    def test_01_below_all_scores_no_alert(self) -> None:
        result = decide_raw(SCORES, 0.05)
        self.assertEqual(result["count_ge"], N_CALIBRATION)
        self.assertAlmostEqual(result["p_value"], 1.0)
        self.assertFalse(result["alert"])

    def test_02_ties_counted_with_ge(self) -> None:
        # 23 个低于阈值 + 2 个恰好等于 test_score
        scores = [0.0] * 23 + [0.2, 0.2]
        result = decide_raw(scores, 0.2)
        self.assertEqual(result["count_ge"], 2)  # 若误用 > 会得到 0
        self.assertAlmostEqual(result["p_value"], 3 / 26)
        self.assertFalse(result["alert"])

    def test_03_equal_to_max_no_alert(self) -> None:
        result = decide_raw(SCORES, max(SCORES))
        self.assertEqual(result["count_ge"], 1)
        self.assertAlmostEqual(result["p_value"], 2 / 26)
        self.assertFalse(result["alert"])

    def test_04_strictly_greater_alerts(self) -> None:
        result = decide_raw(SCORES, max(SCORES) + 1e-9)
        self.assertEqual(result["count_ge"], 0)
        self.assertAlmostEqual(result["p_value"], 1 / 26)
        self.assertTrue(result["alert"])

    def test_05_manual_formula_matches(self) -> None:
        test_score = 0.235
        expected_count = sum(1 for s in SCORES if s >= test_score)
        expected_p = (1 + expected_count) / 26
        result = conformal_p_value(SCORES, test_score)
        self.assertEqual(result["count_ge"], expected_count)
        self.assertAlmostEqual(result["p_value"], expected_p, places=12)

    def test_06_wrong_calibration_size_rejected(self) -> None:
        for size in (0, 1, 24, 26):
            with self.assertRaises(ConformalError):
                conformal_p_value([0.1] * size, 0.2)
            with self.assertRaises(ConformalError):
                CalibrationSet(seed=42, class_name="Cr", scores=tuple([0.1] * size))

    def test_07_nan_inf_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            scores = SCORES[:-1] + [bad]
            with self.assertRaises(ConformalError):
                conformal_p_value(scores, 0.2)
            with self.assertRaises(ConformalError):
                CalibrationSet(seed=42, class_name="Cr", scores=tuple(scores))
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ConformalError):
                conformal_p_value(SCORES, bad)

    def test_08_invalid_alpha_rejected(self) -> None:
        calibration = CalibrationSet(seed=42, class_name="Cr", scores=tuple(SCORES))
        for alpha in (0.1, 0.049, 0.5, -0.1, float("nan")):
            with self.assertRaises(ConformalError):
                decide(calibration, 0.5, alpha=alpha)
            with self.assertRaises(ConformalError):
                decide_raw(SCORES, 0.5, alpha=alpha)

    def test_09_order_invariance(self) -> None:
        shuffled = SCORES.copy()
        random.Random(7).shuffle(shuffled)
        self.assertEqual(
            conformal_p_value(SCORES, 0.215)["p_value"],
            conformal_p_value(shuffled, 0.215)["p_value"],
        )

    def test_10_seed_and_class_mismatch_rejected(self) -> None:
        calibration = CalibrationSet(seed=42, class_name="Cr", scores=tuple(SCORES))
        with self.assertRaises(ConformalError):
            decide(calibration, 0.5, seed=123)
        with self.assertRaises(ConformalError):
            decide(calibration, 0.5, class_name="In")
        ok = decide(calibration, 0.5, seed=42, class_name="Cr")
        self.assertTrue(ok["alert"])

    def test_11_p_min_and_fusion_boundary(self) -> None:
        self.assertAlmostEqual(p_min(), 1 / 26)
        self.assertLessEqual(p_min(), ALPHA)
        self.assertGreater(2 / 26, ALPHA)  # Bonferroni 融合不可用

    def test_12_result_fields(self) -> None:
        calibration = CalibrationSet(seed=7, class_name="Pa", scores=tuple(SCORES))
        result = decide(calibration, 0.9, seed=7, class_name="Pa")
        for key in (
            "seed",
            "class_name",
            "test_score",
            "p_value",
            "alpha",
            "alert",
            "rule",
            "equivalent_rule",
            "count_ge",
            "n_calibration",
            "p_min",
            "tie_rule",
            "channel",
        ):
            self.assertIn(key, result)
        self.assertEqual(result["n_calibration"], 25)
        self.assertEqual(result["channel"], "A")
        self.assertEqual(result["tie_rule"], ">=")
        self.assertTrue(math.isclose(result["p_value"], 1 / 26))


if __name__ == "__main__":
    unittest.main(verbosity=2)
