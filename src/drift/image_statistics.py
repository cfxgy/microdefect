#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通道 A：图像统计量漂移分数（P4-2）。

对每张灰度图计算三个统计量：

* ``mean_intensity``      灰度均值
* ``contrast``            灰度总体标准差
* ``laplacian_variance``  Laplacian 响应方差（清晰度）

对每个缺陷类别单独建立参考分布（中位数 + MAD），窗口分数为三个指标
稳健 z 分数的均方根::

    z_j = |window_mean_j - median_j| / scale_j
    score_A = sqrt(mean_j z_j^2)

``scale_j`` 默认取 ``1.4826 * MAD_j``；MAD 过小时回退到标准差，再退到
绝对下限，并在参考模型里记录实际采用的尺度来源。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

STAT_KEYS = ("mean_intensity", "contrast", "laplacian_variance")
MAD_TO_SIGMA = 1.4826
MIN_SCALE = 1e-3


class StatisticsError(ValueError):
    """输入图像或统计量非法。"""


def load_gray(path: str) -> np.ndarray:
    """读取 8 位灰度图（只读）。"""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise StatisticsError(f"无法解码图像：{path}")
    return validate_gray(image)


def validate_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise StatisticsError(f"要求二维灰度图，实际 {array.shape}")
    if array.size == 0:
        raise StatisticsError("图像为空")
    if not np.issubdtype(array.dtype, np.number):
        raise StatisticsError(f"图像 dtype 非法：{array.dtype}")
    if not np.isfinite(array.astype(np.float64)).all():
        raise StatisticsError("图像包含 NaN/Inf")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def compute_image_stats(image: np.ndarray) -> dict:
    """单张图像的三个统计量。"""
    array = validate_gray(image).astype(np.float64)
    laplacian = cv2.Laplacian(array, cv2.CV_64F)
    stats = {
        "mean_intensity": float(array.mean()),
        "contrast": float(array.std(ddof=0)),
        "laplacian_variance": float(laplacian.var()),
    }
    if not all(np.isfinite(v) for v in stats.values()):
        raise StatisticsError(f"统计量非有限：{stats}")
    return stats


def stats_matrix(stats_list: list[dict]) -> np.ndarray:
    """把统计字典列表转成 ``[N, 3]``（列顺序见 STAT_KEYS）。"""
    if not stats_list:
        raise StatisticsError("统计量列表为空")
    matrix = np.array(
        [[item[key] for key in STAT_KEYS] for item in stats_list], dtype=np.float64
    )
    if not np.isfinite(matrix).all():
        raise StatisticsError("统计量矩阵含 NaN/Inf")
    return matrix


@dataclass
class ReferenceModel:
    """某一缺陷类别的参考统计分布。"""

    class_name: str
    n_samples: int
    median: np.ndarray
    mad: np.ndarray
    std: np.ndarray
    scale: np.ndarray
    scale_source: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, stats_list: list[dict], class_name: str) -> "ReferenceModel":
        matrix = stats_matrix(stats_list)
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0)
        std = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(3)
        scale = MAD_TO_SIGMA * mad
        sources: list[str] = []
        for index in range(3):
            if scale[index] >= MIN_SCALE:
                sources.append("mad")
            elif std[index] >= MIN_SCALE:
                scale[index] = std[index]
                sources.append("std_fallback")
            else:
                scale[index] = MIN_SCALE
                sources.append("epsilon")
        return cls(
            class_name=class_name,
            n_samples=int(matrix.shape[0]),
            median=median,
            mad=mad,
            std=std,
            scale=scale,
            scale_source=sources,
        )

    def score(self, window_stats: np.ndarray) -> dict:
        """窗口统计量（3 维均值向量）→ 各指标 z 分数与 score_A。"""
        values = np.asarray(window_stats, dtype=np.float64).reshape(3)
        if not np.isfinite(values).all():
            raise StatisticsError(f"窗口统计量含 NaN/Inf：{values}")
        z = np.abs(values - self.median) / self.scale
        score = float(np.sqrt(np.mean(z ** 2)))
        return {
            "z_mean_intensity": float(z[0]),
            "z_contrast": float(z[1]),
            "z_laplacian_variance": float(z[2]),
            "score_A": score,
        }

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "n_samples": self.n_samples,
            "median": dict(zip(STAT_KEYS, self.median.tolist())),
            "mad": dict(zip(STAT_KEYS, self.mad.tolist())),
            "std": dict(zip(STAT_KEYS, self.std.tolist())),
            "scale": dict(zip(STAT_KEYS, self.scale.tolist())),
            "scale_source": dict(zip(STAT_KEYS, self.scale_source)),
            "mad_to_sigma": MAD_TO_SIGMA,
            "min_scale": MIN_SCALE,
        }


def window_stat_means(stats_list: list[dict]) -> np.ndarray:
    """窗口内各指标均值（列顺序见 STAT_KEYS）。"""
    return stats_matrix(stats_list).mean(axis=0)
