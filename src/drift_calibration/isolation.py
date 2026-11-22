#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""源图像隔离审计（冻结协议 3.1 节的数据划分要求）。

协议要求：

* reference pool 是 validation 的子集；
* calibration pool = validation 中 reference 之外的余集；
* reference / calibration / test 三方 source ID（或 MD5）互不相交；
* 每个集合内部不得出现重复 ID（重复意味着同一源图被算了两次）。

任何一条不满足都必须抛 :class:`IsolationError`，不得降级为警告。
"""

from __future__ import annotations

from typing import Iterable


class IsolationError(RuntimeError):
    """源图像隔离被破坏。"""


def _as_set(ids: Iterable[str], name: str) -> set[str]:
    values = [str(item) for item in ids]
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise IsolationError(f"{name} 集合内部存在重复 ID：{duplicates[:5]}")
    return set(values)


def assert_disjoint(
    reference_ids: Iterable[str],
    calibration_ids: Iterable[str],
    test_ids: Iterable[str] | None = None,
    *,
    labels: dict[str, str] | None = None,
) -> dict:
    """校验三方隔离，返回统计摘要；失败时抛 :class:`IsolationError`。"""
    names = {"ref": "reference", "cal": "calibration", "test": "test"}
    if labels:
        names.update(labels)

    reference = _as_set(reference_ids, names["ref"])
    calibration = _as_set(calibration_ids, names["cal"])
    test = _as_set(test_ids, names["test"]) if test_ids is not None else None

    problems: list[str] = []
    ref_cal = reference & calibration
    if ref_cal:
        problems.append(f"reference ∩ calibration = {len(ref_cal)} 个：{sorted(ref_cal)[:5]}")
    if test is not None:
        ref_test = reference & test
        cal_test = calibration & test
        if ref_test:
            problems.append(f"reference ∩ test = {len(ref_test)} 个：{sorted(ref_test)[:5]}")
        if cal_test:
            problems.append(f"calibration ∩ test = {len(cal_test)} 个：{sorted(cal_test)[:5]}")

    if problems:
        raise IsolationError("源图像隔离失败： " + "；".join(problems))

    return {
        "reference_count": len(reference),
        "calibration_count": len(calibration),
        "test_count": None if test is None else len(test),
        "reference_calibration_overlap": 0,
        "reference_test_overlap": 0 if test is not None else None,
        "calibration_test_overlap": 0 if test is not None else None,
        "disjoint": True,
    }


def assert_subset(
    subset_ids: Iterable[str],
    superset_ids: Iterable[str],
    *,
    subset_name: str = "reference",
    superset_name: str = "validation",
) -> dict:
    """校验 ``subset`` 是 ``superset`` 的子集。"""
    subset = _as_set(subset_ids, subset_name)
    superset = _as_set(superset_ids, superset_name)
    outside = subset - superset
    if outside:
        raise IsolationError(
            f"{subset_name} 不是 {superset_name} 的子集：{len(outside)} 个越界，"
            f"例如 {sorted(outside)[:5]}"
        )
    return {
        "subset": subset_name,
        "superset": superset_name,
        "subset_count": len(subset),
        "superset_count": len(superset),
        "is_subset": True,
    }


__all__ = ["IsolationError", "assert_disjoint", "assert_subset"]
