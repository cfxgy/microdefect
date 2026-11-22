#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""产品级阈值校准（P2，可配置样本数）。

与 P5 的 MVTec 校准时序/分位数口径一致（0.95 图像 / 0.99 像素、linear），
但**不硬编码 25 张**；25 仅作为产品注册默认值。

统计边界（必须保留）：

* 图像阈值 = 校准集图像分数的 0.95 线性分位；
* 像素阈值 = 所有校准图全部像素分数的 0.99 线性分位；
* 这是经验分位数，**不是 conformal calibration**，不提供固定 FPR 保证；
* 默认 25 张；允许 20..200 张，20..24 张时标记 ``low_sample_warning``；
* 少于 20 张拒绝：0.95 分位在 n<20 时没有足够尾部观测，阈值不稳定。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_CALIBRATION_COUNT = 25
MIN_CALIBRATION_COUNT = 20
MAX_CALIBRATION_COUNT = 200
LOW_SAMPLE_THRESHOLD = 25

IMAGE_QUANTILE = 0.95
PIXEL_QUANTILE = 0.99
QUANTILE_METHOD = "linear"


class ProductCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    quantile: float
    method: str
    n: int
    stats: dict
    exceed_count: int
    dtype: str
    independent_samples: bool


def validate_calibration_count(count: int) -> None:
    if not isinstance(count, int) or isinstance(count, bool):
        raise ProductCalibrationError(f"校准数量必须是整数：{count!r}")
    if count < MIN_CALIBRATION_COUNT:
        raise ProductCalibrationError(
            f"校准图数量 {count} 过少：至少需要 {MIN_CALIBRATION_COUNT} 张"
            f"（0.95 分位在 n<{MIN_CALIBRATION_COUNT} 时缺乏尾部观测，阈值不可靠）"
        )
    if count > MAX_CALIBRATION_COUNT:
        raise ProductCalibrationError(
            f"校准图数量 {count} 过多：上限 {MAX_CALIBRATION_COUNT} 张"
        )


def calibration_sample_note(count: int) -> dict:
    return {
        "count": int(count),
        "minimum": MIN_CALIBRATION_COUNT,
        "default": DEFAULT_CALIBRATION_COUNT,
        "low_sample_warning": int(count) < LOW_SAMPLE_THRESHOLD,
        "fpr_guarantee": False,
        "note": "经验分位数阈值，不是 conformal calibration，不承诺固定 FPR。",
    }


def _stats(array: np.ndarray) -> dict:
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p95": float(np.quantile(array, 0.95, method=QUANTILE_METHOD)),
        "p99": float(np.quantile(array, 0.99, method=QUANTILE_METHOD)),
    }


def calibrate_image_scores(
    scores: np.ndarray, *, quantile: float = IMAGE_QUANTILE
) -> ThresholdResult:
    array = np.asarray(scores, dtype=np.float64).ravel()
    if array.size < 2:
        raise ProductCalibrationError(f"图像校准分数过少：{array.size}")
    if not np.isfinite(array).all():
        raise ProductCalibrationError("图像校准分数包含 NaN/Inf")
    if not (0.0 < quantile < 1.0):
        raise ProductCalibrationError(f"分位数必须在 (0,1)：{quantile}")
    threshold = float(np.quantile(array, quantile, method=QUANTILE_METHOD))
    if not np.isfinite(threshold):
        raise ProductCalibrationError("图像阈值非有限值")
    return ThresholdResult(
        threshold=threshold,
        quantile=float(quantile),
        method=QUANTILE_METHOD,
        n=int(array.size),
        stats=_stats(array),
        exceed_count=int(np.sum(array > threshold)),
        dtype="float64",
        independent_samples=True,
    )


def calibrate_pixel_scores(
    scores: np.ndarray, *, quantile: float = PIXEL_QUANTILE
) -> ThresholdResult:
    array = np.asarray(scores, dtype=np.float32).ravel()
    if array.size < 2:
        raise ProductCalibrationError(f"像素校准分数过少：{array.size}")
    if not np.isfinite(array).all():
        raise ProductCalibrationError("像素校准分数包含 NaN/Inf")
    if not (0.0 < quantile < 1.0):
        raise ProductCalibrationError(f"分位数必须在 (0,1)：{quantile}")
    threshold = float(np.quantile(array, quantile, method=QUANTILE_METHOD))
    if not np.isfinite(threshold):
        raise ProductCalibrationError("像素阈值非有限值")
    return ThresholdResult(
        threshold=threshold,
        quantile=float(quantile),
        method=QUANTILE_METHOD,
        n=int(array.size),
        stats=_stats(array.astype(np.float64, copy=False)),
        exceed_count=int(np.sum(array > threshold)),
        dtype="float32",
        independent_samples=False,
    )
