#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""漂移监测子包（P4-2）。"""

from src.drift.calibration import CalibrationError, CalibrationResult, bootstrap_calibrate
from src.drift.detector import (
    DetectorError,
    ScoreResult,
    Window,
    build_clean_windows,
    build_perturbed_windows,
    fuse_scores,
    score_window,
)
from src.drift.feature_distribution import (
    FeatureDistributionError,
    compute_mmd_score,
    median_heuristic_gamma,
)
from src.drift.image_statistics import (
    MAD_TO_SIGMA,
    MIN_SCALE,
    STAT_KEYS,
    ReferenceModel,
    StatisticsError,
    compute_image_stats,
    load_gray,
    window_stat_means,
)
from src.drift.metrics import MetricsError, auroc, binary_metrics, macro_average, mean_std

__all__ = [
    "MAD_TO_SIGMA",
    "MIN_SCALE",
    "STAT_KEYS",
    "CalibrationError",
    "CalibrationResult",
    "DetectorError",
    "FeatureDistributionError",
    "MetricsError",
    "ReferenceModel",
    "ScoreResult",
    "StatisticsError",
    "Window",
    "auroc",
    "binary_metrics",
    "bootstrap_calibrate",
    "build_clean_windows",
    "build_perturbed_windows",
    "compute_image_stats",
    "compute_mmd_score",
    "fuse_scores",
    "load_gray",
    "macro_average",
    "mean_std",
    "median_heuristic_gamma",
    "score_window",
    "window_stat_means",
]
