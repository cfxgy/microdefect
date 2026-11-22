#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图像级 split-conformal p 值与告警判定（冻结协议 3.4 节）。

公式（唯一允许的形式）::

    p = (1 + count(cal_score >= test_score)) / (n + 1) ,  n = 25  ⇒  分母固定为 26

判定::

    alert  ⟺  p <= alpha ,  alpha 固定为 0.05

由于 ``p_min = 1/26 ≈ 0.038462 ≤ 0.05``，而 ``2/26 ≈ 0.076923 > 0.05``，
该规则**等价于**：新图像分数**严格大于全部 25 个校准分数**才告警。
分数等于最大值时 ties 计入 ``>=``，p = 2/26，不告警。

统计解释（必须随任何结果一起报告）：

* 仅在**可交换性假设**下，提供**单张图像**的边际误报控制；
* 不保证连续图像流的累计误报率；
* 不保证 16 图窗口误报率；
* 不保证每日或每条产线的误报率；
* 不保证时间相关数据上的序列级误报控制。

本模块**不**读取任何数据、不估计任何参数、不接触测试集。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.drift_calibration.protocol import (
    ALPHA,
    N_CALIBRATION,
    P_DENOMINATOR,
    TIE_RULE,
    ProtocolError,
)


class ConformalError(ValueError):
    """conformal 输入非法。"""


@dataclass(frozen=True)
class CalibrationSet:
    """某个 (seed, 类别) 的 25 个校准分数。

    ``__post_init__`` 会强制校验：数量恰好为 25、全部有限、类别非空。
    """

    seed: int
    class_name: str
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.class_name, str) or not self.class_name:
            raise ConformalError("class_name 必须为非空字符串")
        try:
            values = tuple(float(v) for v in self.scores)
        except (TypeError, ValueError) as exc:
            raise ConformalError(f"校准分数必须是数值：{exc}") from exc
        if len(values) != N_CALIBRATION:
            raise ConformalError(
                f"校准分数数量必须为 {N_CALIBRATION}，实际 {len(values)}"
            )
        bad = [v for v in values if not math.isfinite(v)]
        if bad:
            raise ConformalError(f"校准分数含 NaN/Inf：{bad[:3]}")
        object.__setattr__(self, "scores", values)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "class_name": self.class_name,
            "n_calibration": len(self.scores),
        }


def p_min() -> float:
    """可达到的最小 p 值。"""
    return 1.0 / P_DENOMINATOR


def _validate_test_score(test_score: float) -> float:
    try:
        value = float(test_score)
    except (TypeError, ValueError) as exc:
        raise ConformalError(f"test_score 必须是数值：{exc}") from exc
    if not math.isfinite(value):
        raise ConformalError(f"test_score 必须是有限数值，实际 {value}")
    return value


def conformal_p_value(scores: Sequence[float], test_score: float) -> dict:
    """有限样本上尾 conformal p 值（ties 使用 ``>=``）。"""
    try:
        values = tuple(float(v) for v in scores)
    except (TypeError, ValueError) as exc:
        raise ConformalError(f"校准分数必须是数值：{exc}") from exc
    if len(values) != N_CALIBRATION:
        raise ConformalError(f"校准分数数量必须为 {N_CALIBRATION}，实际 {len(values)}")
    bad = [v for v in values if not math.isfinite(v)]
    if bad:
        raise ConformalError(f"校准分数含 NaN/Inf：{bad[:3]}")

    value = _validate_test_score(test_score)
    count = sum(1 for s in values if s >= value)
    p = (1 + count) / P_DENOMINATOR
    return {
        "p_value": p,
        "count_ge": count,
        "n_calibration": len(values),
        "p_min": p_min(),
        "tie_rule": TIE_RULE,
        "formula": "p = (1 + count(cal >= test)) / 26",
    }


def decide(
    calibration: CalibrationSet,
    test_score: float,
    *,
    seed: int | None = None,
    class_name: str | None = None,
    alpha: float = ALPHA,
) -> dict:
    """单张图像告警判定。

    ``seed`` / ``class_name`` 若给出，必须与校准集一致，否则拒绝——
    防止把不同 seed 或不同类别的校准分数混用。
    """
    if not isinstance(calibration, CalibrationSet):
        raise ConformalError("calibration 必须是 CalibrationSet")
    if alpha != ALPHA:
        raise ConformalError(
            f"alpha 在冻结协议中固定为 {ALPHA}，不允许改为 {alpha}"
        )
    if seed is not None and seed != calibration.seed:
        raise ConformalError(
            f"seed 不匹配：校准集 seed={calibration.seed}，调用 seed={seed}"
        )
    if class_name is not None and class_name != calibration.class_name:
        raise ConformalError(
            f"class_name 不匹配：校准集 {calibration.class_name}，调用 {class_name}"
        )

    result = conformal_p_value(calibration.scores, test_score)
    alert = result["p_value"] <= alpha
    return {
        "seed": calibration.seed,
        "class_name": calibration.class_name,
        "test_score": float(test_score),
        "p_value": result["p_value"],
        "alpha": alpha,
        "alert": bool(alert),
        "rule": "p <= alpha",
        "equivalent_rule": "test_score > max(calibration_scores)",
        "count_ge": result["count_ge"],
        "n_calibration": result["n_calibration"],
        "p_min": result["p_min"],
        "tie_rule": result["tie_rule"],
        "channel": "A",
    }


def decide_raw(scores: Sequence[float], test_score: float, *, alpha: float = ALPHA) -> dict:
    """不带 seed/类别标签的便捷入口（仅用于数学测试）。"""
    if alpha != ALPHA:
        raise ConformalError(f"alpha 固定为 {ALPHA}，实际 {alpha}")
    result = conformal_p_value(scores, test_score)
    return {
        "p_value": result["p_value"],
        "alert": bool(result["p_value"] <= alpha),
        "count_ge": result["count_ge"],
        "n_calibration": result["n_calibration"],
        "p_min": result["p_min"],
        "tie_rule": result["tie_rule"],
        "alpha": alpha,
    }


__all__ = [
    "CalibrationSet",
    "ConformalError",
    "conformal_p_value",
    "decide",
    "decide_raw",
    "p_min",
]
