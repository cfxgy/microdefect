#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MVTec AD 评估指标（P5-2.4，严格按协议 v1.0 的 metric_definitions.md）。

* 图像级 AUROC：秩和法，并列取平均秩；
* 图像级 AP：稳定降序后 ``Σ (R_n − R_{n−1})·P_n``（等价于正样本处 precision 的均值）；
* 阈值指标：``alert ⟺ score > threshold``，等于阈值判正常；分母为 0 返回 None；
* 像素级 AUROC/AP：全部测试像素池化，逐类计算；
* 不做 bootstrap（方法与次数在协议中仍为“待人工确认”）。
"""

from __future__ import annotations

import numpy as np

PIXEL_MEMORY_CHUNK = 1 << 24


class MetricError(RuntimeError):
    """指标输入不合法。"""


def _check(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels).astype(bool)
    if scores.ndim != 1 or labels.ndim != 1 or scores.size != labels.size:
        raise MetricError(f"输入形状不匹配：{scores.shape} vs {labels.shape}")
    if scores.size == 0:
        raise MetricError("指标输入为空")
    if not np.isfinite(scores).all():
        raise MetricError("分数含 NaN 或 Inf")
    return scores, labels


def auroc_rank_average(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """秩和 AUROC，并列取平均秩。"""
    scores, labels = _check(scores, labels)
    n_pos = int(labels.sum())
    n_neg = int(scores.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    ends = np.r_[starts[1:], scores.size]
    pos_counts = np.add.reduceat(sorted_labels.astype(np.int64), starts)
    average_ranks = (starts + 1 + ends) / 2.0
    rank_sum = float((pos_counts * average_ranks).sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """稳定降序的 step AP。"""
    scores, labels = _check(scores, labels)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    positions = np.flatnonzero(labels[order]) + 1
    precision = np.arange(1, n_pos + 1) / positions
    return float(precision.mean())


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """``score > threshold`` 判异常；返回混淆矩阵与 P/R/F1（分母 0 时为 None）。"""
    scores, labels = _check(scores, labels)
    if not np.isfinite(threshold):
        raise MetricError("阈值含 NaN 或 Inf")
    predicted = scores > float(threshold)
    tp = int(np.count_nonzero(predicted & labels))
    fp = int(np.count_nonzero(predicted & ~labels))
    tn = int(np.count_nonzero(~predicted & ~labels))
    fn = int(np.count_nonzero(~predicted & labels))
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive": tp + fp,
    }


def image_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    result = {
        "auroc": auroc_rank_average(scores, labels),
        "ap": average_precision(scores, labels),
        "n_pos": int(np.asarray(labels).sum()),
        "n_neg": int(np.asarray(labels).size - np.asarray(labels).sum()),
    }
    result.update(threshold_metrics(scores, labels, threshold))
    return result


def pixel_auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """像素级 AUROC：只排序负样本 + searchsorted，精确处理并列。"""
    scores, labels = _check(scores, labels)
    n_pos = int(labels.sum())
    n_neg = int(scores.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    negatives = np.sort(scores[~labels])
    positives = scores[labels]
    less = np.searchsorted(negatives, positives, side="left")
    leq = np.searchsorted(negatives, positives, side="right")
    return float((less + 0.5 * (leq - less)).sum() / (n_pos * n_neg))


def pixel_average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """像素级 AP：打包 (score,index) 为 uint64 排序，保持稳定降序。"""
    scores, labels = _check(scores, labels)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return None
    n = scores.size
    bits = scores.view(np.uint32)
    keys = np.where(
        bits & np.uint32(0x80000000), ~bits, bits | np.uint32(0x80000000)
    ).astype(np.uint32)
    packed = (np.uint64(0xFFFFFFFF) - keys.astype(np.uint64)) << np.uint64(32)
    packed |= np.arange(n, dtype=np.uint64)
    packed.sort()
    indices = (packed & np.uint64(0xFFFFFFFF)).astype(np.int64)
    positions = np.flatnonzero(labels[indices]) + 1
    precision = np.arange(1, n_pos + 1) / positions
    return float(precision.mean())


def pixel_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    scores, labels = _check(scores, labels)
    return {
        "pixel_auroc": pixel_auroc(scores, labels),
        "pixel_ap": pixel_average_precision(scores, labels),
        "n_pixels": int(scores.size),
        "n_pos_pixels": int(labels.sum()),
        "n_neg_pixels": int(scores.size - labels.sum()),
    }


def macro_average(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def mean_std(values: list[float | None]) -> dict:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return {"mean": None, "std": None, "n": 0}
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(valid)}
