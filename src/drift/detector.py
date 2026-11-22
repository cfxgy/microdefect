#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""监测窗口构建与三通道打分的组装（P4-2）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src.drift.image_statistics import ReferenceModel, window_stat_means

EPSILON = 1e-12


class DetectorError(ValueError):
    """窗口构建或打分输入非法。"""


@dataclass
class Window:
    seed: int
    class_name: str
    split: str
    window_id: str
    source_ids: list[str]
    perturbation_type: str
    perturbation_level: str
    is_drift: int

    def to_row(self) -> dict:
        return {
            "seed": self.seed,
            "class_name": self.class_name,
            "split": self.split,
            "window_id": self.window_id,
            "source_ids": "|".join(self.source_ids),
            "source_count": len(self.source_ids),
            "perturbation_type": self.perturbation_type,
            "perturbation_level": self.perturbation_level,
            "is_drift": self.is_drift,
        }


@dataclass
class ScoreResult:
    mean_intensity: float
    contrast: float
    laplacian_variance: float
    z_mean_intensity: float
    z_contrast: float
    z_laplacian_variance: float
    score_A: float
    score_B: float
    mmd_squared: float
    gamma: float
    score_fusion: float
    is_drift_A: int
    is_drift_B: int
    is_drift_fusion: int
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        base = {
            "mean_intensity": round(self.mean_intensity, 6),
            "contrast": round(self.contrast, 6),
            "laplacian_variance": round(self.laplacian_variance, 6),
            "z_mean_intensity": round(self.z_mean_intensity, 6),
            "z_contrast": round(self.z_contrast, 6),
            "z_laplacian_variance": round(self.z_laplacian_variance, 6),
            "score_A": round(self.score_A, 8),
            "score_B": round(self.score_B, 10),
            "mmd_squared": round(self.mmd_squared, 10),
            "gamma": self.gamma,
            "score_fusion": round(self.score_fusion, 8),
            "is_drift_A": self.is_drift_A,
            "is_drift_B": self.is_drift_B,
            "is_drift_fusion": self.is_drift_fusion,
        }
        base.update(self.extra)
        return base


def numeric_key(name: str) -> tuple:
    """按文件名中的数字序号排序（Cr_2 < Cr_10）。"""
    stem = name.rsplit(".", 1)[0]
    match = re.search(r"_(\d+)$", stem)
    index = int(match.group(1)) if match else 10 ** 9
    return (index, stem)


def chunk_sources(rows: list[dict], size: int) -> tuple[list[list[dict]], list[dict]]:
    """把按稳定顺序排好的源图像切成不重叠窗口，返回 (窗口列表, 尾部)。"""
    if size < 2:
        raise DetectorError(f"窗口大小至少为 2，实际 {size}")
    ordered = sorted(rows, key=lambda r: numeric_key(r["source_file"]))
    windows = [ordered[i : i + size] for i in range(0, len(ordered) - size + 1, size)]
    tail = ordered[len(windows) * size :]
    return windows, tail


def build_clean_windows(
    rows: list[dict],
    *,
    seed: int,
    class_name: str,
    split: str,
    size: int,
) -> tuple[list[Window], list[dict]]:
    chunks, tail = chunk_sources(rows, size)
    windows = [
        Window(
            seed=seed,
            class_name=class_name,
            split=split,
            window_id=f"{split}_clean_w{index:03d}",
            source_ids=[row["source_file"] for row in chunk],
            perturbation_type="clean",
            perturbation_level="none",
            is_drift=0,
        )
        for index, chunk in enumerate(chunks)
    ]
    return windows, tail


def build_perturbed_windows(
    clean_windows: list[Window],
    rows_by_condition: dict[tuple[str, str], dict[str, dict]],
) -> list[Window]:
    """按与干净窗口相同的 source ID 构造配对扰动窗口。"""
    windows: list[Window] = []
    for window in clean_windows:
        for (kind, level), mapping in rows_by_condition.items():
            missing = [sid for sid in window.source_ids if sid not in mapping]
            if missing:
                raise DetectorError(
                    f"{window.seed}/{window.class_name} 扰动 {kind}/{level} 缺少源图：{missing[:3]}"
                )
            windows.append(
                Window(
                    seed=window.seed,
                    class_name=window.class_name,
                    split=window.split,
                    window_id=f"{window.window_id}__{kind}_{level}",
                    source_ids=list(window.source_ids),
                    perturbation_type=kind,
                    perturbation_level=level,
                    is_drift=1,
                )
            )
    return windows


def fuse_scores(score_a: float, score_b: float, threshold_a: float, threshold_b: float) -> float:
    return max(score_a / max(threshold_a, EPSILON), score_b / max(threshold_b, EPSILON))


def score_window(
    window: Window,
    *,
    rows_by_file: dict[str, dict],
    stats_by_md5: dict[str, dict],
    features_by_md5: dict[str, np.ndarray],
    model: ReferenceModel,
    reference_features: np.ndarray,
    gamma: float,
    threshold_a: float,
    threshold_b: float,
    threshold_fusion: float,
) -> ScoreResult:
    """对一个窗口计算 A / B / 融合分数与二值判定。"""
    from src.drift.feature_distribution import compute_mmd_score

    stats_list = []
    feature_rows = []
    for source_id in window.source_ids:
        row = rows_by_file.get(source_id)
        if row is None:
            raise DetectorError(f"窗口引用了不存在的源图：{source_id}")
        md5 = row["output_md5"]
        if md5 not in stats_by_md5:
            raise DetectorError(f"缺少图像统计：{md5}")
        if md5 not in features_by_md5:
            raise DetectorError(f"缺少特征：{md5}")
        stats_list.append(stats_by_md5[md5])
        feature_rows.append(features_by_md5[md5])

    stats_mean = window_stat_means(stats_list)
    a = model.score(stats_mean)
    mmd = compute_mmd_score(reference_features, np.stack(feature_rows, axis=0), gamma)
    fusion = fuse_scores(a["score_A"], mmd["score_B"], threshold_a, threshold_b)

    return ScoreResult(
        mean_intensity=float(stats_mean[0]),
        contrast=float(stats_mean[1]),
        laplacian_variance=float(stats_mean[2]),
        z_mean_intensity=a["z_mean_intensity"],
        z_contrast=a["z_contrast"],
        z_laplacian_variance=a["z_laplacian_variance"],
        score_A=a["score_A"],
        score_B=mmd["score_B"],
        mmd_squared=mmd["mmd_squared"],
        gamma=gamma,
        score_fusion=fusion,
        is_drift_A=int(a["score_A"] > threshold_a),
        is_drift_B=int(mmd["score_B"] > threshold_b),
        is_drift_fusion=int(fusion > threshold_fusion),
        extra={
            "reference_count": mmd["reference_count"],
            "candidate_count": mmd["candidate_count"],
            "feature_dimension": mmd["feature_dimension"],
        },
    )
