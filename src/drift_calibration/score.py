#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A 通道图像统计分数（冻结协议 3.2 节）。

公式（与 P4-2.1 阶段 3 v2.1 一致）::

    z_j(x)     = |stat_j(x) − median_j| / scale_j
    score_A(x) = sqrt( mean_j z_j(x)^2 )

其中 ``j ∈ {mean_intensity, contrast, laplacian_variance}``，
``median_j / scale_j`` 由**参考池**估计，``scale_j = 1.4826 · MAD_j``；
MAD 过小时回退到标准差，再回退到 ``1e-3``（P4-2 已验证规则）。

三个统计量的逐像素定义由冻结协议按引用委托给 P4-2 的
``src/drift/image_statistics.py``（该文件受哈希保护、未在本阶段修改）：

* ``mean_intensity``      = 灰度均值
* ``contrast``            = 灰度总体标准差（ddof=0）
* ``laplacian_variance``  = ``cv2.Laplacian(img, CV_64F).var()``

分数方向：**数值越大表示偏离参考分布越强**（越大越异常）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.drift.image_statistics import (
    STAT_KEYS,
    ReferenceModel,
    StatisticsError,
    compute_image_stats,
)


class ScoreError(ValueError):
    """分数输入非法。"""


@dataclass(frozen=True)
class ImageScore:
    """单张图像的 A 通道分数与中间量。"""

    class_name: str
    stats: tuple[float, float, float]
    z: tuple[float, float, float]
    score_A: float

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "mean_intensity": self.stats[0],
            "contrast": self.stats[1],
            "laplacian_variance": self.stats[2],
            "z_mean_intensity": self.z[0],
            "z_contrast": self.z[1],
            "z_laplacian_variance": self.z[2],
            "score_A": self.score_A,
        }


def score_direction() -> str:
    """冻结的分数方向声明。"""
    return "larger_is_more_anomalous"


class ImageScorer:
    """绑定某个 (seed, 类别) 的参考池，对单张图像计算 A 通道分数。

    Parameters
    ----------
    reference_stats:
        参考池中每张图像的统计量字典列表（长度应等于 ``reference_pool_size``）。
    class_name:
        缺陷类别名（Cr/In/Pa/PS/RS/Sc）。校准不得跨类别，故必须显式绑定。
    """

    def __init__(self, reference_stats: list[dict], class_name: str) -> None:
        if not isinstance(class_name, str) or not class_name:
            raise ScoreError("class_name 必须为非空字符串")
        if len(reference_stats) < 2:
            raise ScoreError(f"参考池样本不足：{len(reference_stats)}")
        try:
            self.model = ReferenceModel.fit(reference_stats, class_name)
        except StatisticsError as exc:
            raise ScoreError(f"参考池统计量非法：{exc}") from exc
        self.class_name = class_name
        self.reference_count = len(reference_stats)

    def score_stats(self, stats: dict) -> ImageScore:
        """对**已计算好**的统计量字典打分（不重新读图）。"""
        missing = [key for key in STAT_KEYS if key not in stats]
        if missing:
            raise ScoreError(f"统计量缺少字段：{missing}")
        values = np.array([float(stats[key]) for key in STAT_KEYS], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ScoreError(f"统计量含 NaN/Inf：{values}")
        try:
            result = self.model.score(values)
        except StatisticsError as exc:
            raise ScoreError(f"打分失败：{exc}") from exc
        return ImageScore(
            class_name=self.class_name,
            stats=tuple(float(v) for v in values),
            z=(
                float(result["z_mean_intensity"]),
                float(result["z_contrast"]),
                float(result["z_laplacian_variance"]),
            ),
            score_A=float(result["score_A"]),
        )

    def score_gray(self, image: np.ndarray) -> ImageScore:
        """对 8 位灰度图计算统计量并打分。"""
        try:
            stats = compute_image_stats(image)
        except StatisticsError as exc:
            raise ScoreError(f"图像统计失败：{exc}") from exc
        return self.score_stats(stats)

    def describe(self) -> dict:
        return {
            "class_name": self.class_name,
            "reference_count": self.reference_count,
            "score_direction": score_direction(),
            "median": dict(zip(STAT_KEYS, self.model.median.tolist())),
            "scale": dict(zip(STAT_KEYS, self.model.scale.tolist())),
            "scale_source": dict(zip(STAT_KEYS, self.model.scale_source)),
            "mad_to_sigma": float(1.4826),
            "min_scale": float(self.model.scale.min()),
        }


def reference_model(class_name: str, reference_stats: list[dict]) -> ReferenceModel:
    """便捷函数：直接返回 P4-2 的参考模型（供审计对比使用）。"""
    return ReferenceModel.fit(reference_stats, class_name)
