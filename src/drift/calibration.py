#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阈值校准：仅使用 validation 参考池的 bootstrap 零假设（P4-2）。

对每个 (seed, 缺陷类别)：

1. 固定该类的参考池（来自 validation 原图）；
2. 有放回抽样 ``window_size`` 张，构成"干净窗口"；
3. 计算 score_A / score_B，重复 ``iterations`` 次；
4. 阈值取零假设分布的 ``quantile`` 分位（默认 95%）；
5. 融合分数 ``max(A/tA, B/tB)`` 的阈值同样由该零假设分布给出。

该阈值是**经验校准**：零假设来自有限参考样本的重采样，不构成严格的
统计覆盖保证；接口上不接收任何 test 数据。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.drift.feature_distribution import FeatureDistributionError, compute_mmd_score
from src.drift.image_statistics import STAT_KEYS, ReferenceModel, stats_matrix


class CalibrationError(ValueError):
    """校准输入非法。"""


@dataclass
class CalibrationResult:
    seed: int
    class_name: str
    window_size: int
    iterations: int
    random_seed: int
    quantile: float
    threshold_A: float
    threshold_B: float
    threshold_fusion: float
    null_scores_A: np.ndarray
    null_scores_B: np.ndarray
    null_scores_fusion: np.ndarray
    reference_count: int
    gamma: float
    epsilon: float

    def summary(self) -> dict:
        def describe(values: np.ndarray) -> dict:
            return {
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q95": float(np.quantile(values, 0.95)),
            }

        return {
            "seed": self.seed,
            "class_name": self.class_name,
            "window_size": self.window_size,
            "iterations": self.iterations,
            "random_seed": self.random_seed,
            "quantile": self.quantile,
            "reference_count": self.reference_count,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "threshold_A": self.threshold_A,
            "threshold_B": self.threshold_B,
            "threshold_fusion": self.threshold_fusion,
            "null_A": describe(self.null_scores_A),
            "null_B": describe(self.null_scores_B),
            "null_fusion": describe(self.null_scores_fusion),
        }


def bootstrap_calibrate(
    reference_stats: list[dict],
    reference_features: np.ndarray,
    *,
    seed: int,
    class_name: str,
    window_size: int,
    iterations: int,
    random_seed: int,
    quantile: float = 0.95,
    epsilon: float = 1e-12,
) -> CalibrationResult:
    """用参考池的 bootstrap 零假设给出三个通道的阈值。"""
    if window_size < 2:
        raise CalibrationError(f"window_size 至少为 2，实际 {window_size}")
    if iterations < 10:
        raise CalibrationError(f"iterations 过少：{iterations}")
    if not 0.5 < quantile < 1.0:
        raise CalibrationError(f"quantile 非法：{quantile}")

    matrix = stats_matrix(reference_stats)
    n_reference = matrix.shape[0]
    if n_reference < 2:
        raise CalibrationError(f"参考池样本不足：{n_reference}")
    features = np.asarray(reference_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != n_reference:
        raise CalibrationError(
            f"参考特征与统计量数量不一致：{features.shape} vs {n_reference}"
        )

    model = ReferenceModel.fit(reference_stats, class_name)
    # gamma 只由参考池决定，整个 calibration 期间固定
    from src.drift.feature_distribution import median_heuristic_gamma

    gamma = median_heuristic_gamma(features)

    rng = np.random.default_rng(random_seed)
    null_a = np.empty(iterations, dtype=np.float64)
    null_b = np.empty(iterations, dtype=np.float64)

    for i in range(iterations):
        index = rng.integers(0, n_reference, size=window_size)
        window_stats = matrix[index].mean(axis=0)
        null_a[i] = model.score(window_stats)["score_A"]
        null_b[i] = compute_mmd_score(features, features[index], gamma)["score_B"]

    threshold_a = float(np.quantile(null_a, quantile))
    threshold_b = float(np.quantile(null_b, quantile))
    null_fusion = np.maximum(
        null_a / max(threshold_a, epsilon), null_b / max(threshold_b, epsilon)
    )
    threshold_fusion = float(np.quantile(null_fusion, quantile))

    return CalibrationResult(
        seed=seed,
        class_name=class_name,
        window_size=window_size,
        iterations=iterations,
        random_seed=random_seed,
        quantile=quantile,
        threshold_A=threshold_a,
        threshold_B=threshold_b,
        threshold_fusion=threshold_fusion,
        null_scores_A=null_a,
        null_scores_B=null_b,
        null_scores_fusion=null_fusion,
        reference_count=int(n_reference),
        gamma=gamma,
        epsilon=epsilon,
    )


__all__ = [
    "STAT_KEYS",
    "CalibrationError",
    "CalibrationResult",
    "bootstrap_calibrate",
    "FeatureDistributionError",
]
