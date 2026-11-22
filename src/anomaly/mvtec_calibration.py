#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MVTec AD 协议 v1.0 阈值校准（P5-2.3.3）。

严格按协议 v1.0：

* 图像级：25 张 calibration 正常图像的余弦分数，取 **0.95 线性分位数**；
* 像素级：同一 25 张图像的**全部热图像素池化**，取 **0.99 线性分位数**；
* 判定：``score > threshold`` 为异常，``score == threshold`` 为正常；
* 该方法是**经验分位数阈值，不是 conformal calibration**，不提供 5%/1% 误报率保证。

本模块只做阈值计算与判定工具，不读取 test 图像、不计算任何测试指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IMAGE_QUANTILE = 0.95
PIXEL_QUANTILE = 0.99
QUANTILE_METHOD = "linear"
CALIBRATION_IMAGE_COUNT = 25


class CalibrationError(RuntimeError):
    """校准输入不合法。"""


@dataclass(frozen=True)
class ImageCalibration:
    threshold: float
    n_scores: int
    quantile: float
    method: str
    scores: np.ndarray = field(repr=False)
    exceed_count: int
    stats: dict


@dataclass(frozen=True)
class PixelCalibration:
    threshold: float
    n_pixels: int
    quantile: float
    method: str
    exceed_count: int
    stats: dict


def linear_quantile(values: np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise CalibrationError("分位数输入为空")
    if not (0.0 < float(q) < 1.0):
        raise CalibrationError(f"分位数必须在 (0,1) 内：{q}")
    if not np.isfinite(array).all():
        raise CalibrationError("分位数输入含 NaN 或 Inf")
    return float(np.quantile(array, float(q), method=QUANTILE_METHOD))


def is_anomalous(score: float, threshold: float) -> bool:
    """严格大于阈值判异常；等于阈值判正常。"""
    if not np.isfinite(score) or not np.isfinite(threshold):
        raise CalibrationError("判定输入含 NaN 或 Inf")
    return bool(float(score) > float(threshold))


def calibrate_image_scores(scores: np.ndarray, *, quantile: float = IMAGE_QUANTILE) -> ImageCalibration:
    array = np.asarray(scores, dtype=np.float64).ravel()
    if array.size != CALIBRATION_IMAGE_COUNT:
        raise CalibrationError(
            f"图像校准样本数应为 {CALIBRATION_IMAGE_COUNT}，实际 {array.size}"
        )
    if not np.isfinite(array).all():
        raise CalibrationError("图像校准分数含 NaN 或 Inf")
    threshold = linear_quantile(array, quantile)
    stats = {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p95": float(linear_quantile(array, 0.95)),
    }
    return ImageCalibration(
        threshold=threshold,
        n_scores=int(array.size),
        quantile=float(quantile),
        method=QUANTILE_METHOD,
        scores=array,
        exceed_count=int(np.count_nonzero(array > threshold)),
        stats=stats,
    )


def calibrate_pixel_heatmaps(heatmaps, *, quantile: float = PIXEL_QUANTILE) -> PixelCalibration:
    """把 25 张热图的全部像素池化后取线性分位数。"""
    arrays = [np.asarray(h, dtype=np.float32).ravel() for h in heatmaps]
    if not arrays:
        raise CalibrationError("像素校准热图为空")
    pooled = np.concatenate(arrays)
    if pooled.size == 0:
        raise CalibrationError("像素校准池化为空")
    if not np.isfinite(pooled).all():
        raise CalibrationError("像素热图含 NaN 或 Inf")
    threshold = linear_quantile(pooled, quantile)
    stats = {
        "min": float(pooled.min()),
        "max": float(pooled.max()),
        "mean": float(pooled.mean()),
        "std": float(pooled.std(ddof=1)) if pooled.size > 1 else 0.0,
    }
    return PixelCalibration(
        threshold=threshold,
        n_pixels=int(pooled.size),
        quantile=float(quantile),
        method=QUANTILE_METHOD,
        exceed_count=int(np.count_nonzero(pooled > threshold)),
        stats=stats,
    )


def calibration_metadata() -> dict:
    return {
        "image_quantile": IMAGE_QUANTILE,
        "pixel_quantile": PIXEL_QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "calibration_images": CALIBRATION_IMAGE_COUNT,
        "is_conformal": False,
        "fpr_guarantee": "none",
        "image_rule": "score > threshold",
        "pixel_rule": "score > threshold",
        "note": "经验分位数阈值，不是 conformal calibration，不提供 5% 或 1% 误报率保证",
    }
