#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""漂移检测的评价指标（P4-2）。

* 二值指标（Precision / Recall / F1 / FPR / FNR / Detection Rate）按**固定阈值**
  下的判定计算；
* AUROC 使用**连续分数**，与阈值无关；
* 聚合统计默认样本标准差（ddof=1），样本不足 2 时明确返回 None。
"""

from __future__ import annotations

import numpy as np


class MetricsError(ValueError):
    """指标输入非法。"""


def validate_inputs(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if score_array.size != label_array.size:
        raise MetricsError(f"长度不一致：{score_array.size} vs {label_array.size}")
    if score_array.size == 0:
        raise MetricsError("样本为空")
    if not np.isfinite(score_array).all():
        raise MetricsError("分数含 NaN/Inf")
    if set(np.unique(label_array).tolist()) - {0, 1}:
        raise MetricsError("标签只能是 0/1")
    return score_array, label_array


def auroc(scores, labels) -> float | None:
    """基于秩的 AUROC（并列取平均秩）。单一类别时返回 None。"""
    score_array, label_array = validate_inputs(scores, labels)
    positives = int(label_array.sum())
    negatives = int(label_array.size - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.empty(score_array.size, dtype=np.float64)
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1

    positive_rank_sum = ranks[label_array == 1].sum()
    statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return float(statistic / (positives * negatives))


def binary_metrics_from_predictions(scores, labels, predictions, *, threshold: float | None = None) -> dict:
    """给定已按固定阈值产生的二值判定，计算指标（分数仅用于 AUROC/均值）。"""
    score_array, label_array = validate_inputs(scores, labels)
    predicted = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if predicted.size != label_array.size:
        raise MetricsError(f"预测长度不一致：{predicted.size} vs {label_array.size}")
    if set(np.unique(predicted).tolist()) - {0, 1}:
        raise MetricsError("预测只能是 0/1")

    tp = int(np.sum((predicted == 1) & (label_array == 1)))
    fp = int(np.sum((predicted == 1) & (label_array == 0)))
    tn = int(np.sum((predicted == 0) & (label_array == 0)))
    fn = int(np.sum((predicted == 0) & (label_array == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None

    positives = tp + fn
    negatives = fp + tn
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positives": positives,
        "negatives": negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": None if fpr is None else float(fpr),
        "fnr": None if fnr is None else float(fnr),
        "detection_rate": float(recall),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) else None,
        "auroc": auroc(score_array, label_array),
        "threshold": None if threshold is None else float(threshold),
        "mean_score_positive": float(score_array[label_array == 1].mean()) if positives else None,
        "mean_score_negative": float(score_array[label_array == 0].mean()) if negatives else None,
    }


def binary_metrics(scores, labels, threshold: float) -> dict:
    """固定阈值下的二值指标（由分数与阈值现场判定）。"""
    score_array, label_array = validate_inputs(scores, labels)
    predicted = (score_array > threshold).astype(np.int64)
    return binary_metrics_from_predictions(
        score_array, label_array, predicted, threshold=threshold
    )


def mean_std(values, *, ddof: int = 1) -> dict:
    """均值与样本标准差；样本不足 2 时 std 返回 None（不伪造）。"""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(clean, dtype=np.float64)
    if array.size > ddof:
        std = float(array.std(ddof=ddof))
    else:
        std = None
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def macro_average(records: list[dict], keys: tuple[str, ...] = ("precision", "recall", "f1", "auroc")) -> dict:
    """对若干子组做宏平均（忽略 None）。"""
    result = {}
    for key in keys:
        values = [record.get(key) for record in records if record.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    return result
