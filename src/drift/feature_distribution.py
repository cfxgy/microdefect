#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通道 B：ResNet18 深度特征的 RBF-MMD 漂移分数（P4-2）。

复用 P2 的冻结特征（512 维），不重新训练、不改预处理、不改特征版本。

采用 biased MMD² 估计::

    MMD²(X, Y) = mean(K(X,X)) + mean(K(Y,Y)) - 2 * mean(K(X,Y))
    K(x, y)    = exp(-gamma * ||x - y||²)

``gamma`` 由**参考特征池**按 median heuristic 固定::

    gamma = 1 / (2 * median(成对平方距离))

不随候选窗口变化，避免用 test 数据调参。
"""

from __future__ import annotations

import numpy as np

DEFAULT_DIM = 512


class FeatureDistributionError(ValueError):
    """特征输入非法。"""


def validate_features(features: np.ndarray, *, name: str = "features") -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise FeatureDistributionError(f"{name} 必须是二维 [N, D]，实际 {array.shape}")
    if array.shape[0] == 0:
        raise FeatureDistributionError(f"{name} 为空")
    if array.shape[1] == 0:
        raise FeatureDistributionError(f"{name} 维度为 0")
    if not np.isfinite(array).all():
        raise FeatureDistributionError(f"{name} 含 NaN/Inf")
    return array


def pairwise_squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[1] != b.shape[1]:
        raise FeatureDistributionError(f"维度不一致：{a.shape[1]} vs {b.shape[1]}")
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    squared = aa + bb - 2.0 * (a @ b.T)
    np.maximum(squared, 0.0, out=squared)
    return squared


def median_heuristic_gamma(reference_features: np.ndarray, *, floor: float = 1e-12) -> float:
    """由参考池估计并固定 gamma（median heuristic）。"""
    reference = validate_features(reference_features, name="reference_features")
    squared = pairwise_squared_distances(reference, reference)
    n = squared.shape[0]
    # 排除对角线的自距离
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        values = squared[mask]
    else:
        values = squared.reshape(-1)
    median = float(np.median(values))
    if median <= floor:
        # 参考池几乎重合：退化为按维度缩放，并明确记录
        median = 1.0
    return float(1.0 / (2.0 * median))


def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    if not np.isfinite(gamma) or gamma <= 0:
        raise FeatureDistributionError(f"gamma 非法：{gamma}")
    squared = pairwise_squared_distances(a, b)
    return np.exp(-gamma * squared)


def compute_mmd_score(
    reference_features: np.ndarray,
    candidate_features: np.ndarray,
    gamma: float,
) -> dict:
    """biased MMD² 与派生分数。"""
    reference = validate_features(reference_features, name="reference_features")
    candidate = validate_features(candidate_features, name="candidate_features")
    if reference.shape[1] != candidate.shape[1]:
        raise FeatureDistributionError(
            f"参考/候选维度不一致：{reference.shape[1]} vs {candidate.shape[1]}"
        )
    if not np.isfinite(gamma) or gamma <= 0:
        raise FeatureDistributionError(f"gamma 非法：{gamma}")

    k_xx = rbf_kernel(reference, reference, gamma).mean()
    k_yy = rbf_kernel(candidate, candidate, gamma).mean()
    k_xy = rbf_kernel(reference, candidate, gamma).mean()
    mmd2 = float(k_xx + k_yy - 2.0 * k_xy)
    if not np.isfinite(mmd2):
        raise FeatureDistributionError(f"MMD² 非有限：{mmd2}")
    return {
        "score_B": max(mmd2, 0.0),
        "mmd_squared": mmd2,
        "gamma": float(gamma),
        "reference_count": int(reference.shape[0]),
        "candidate_count": int(candidate.shape[0]),
        "feature_dimension": int(reference.shape[1]),
    }
