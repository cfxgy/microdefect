#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2.1 阶段 4：图像级 conformal 校准模块。

公开接口只包含数学与协议校验，不读取数据、不接触测试集。
正式告警仅由通道 A 产生；通道 B 为辅助指标；Bonferroni 融合在 n=25 时
最小 p 值为 2/26 ≈ 0.0769 > 0.05，不可用于正式告警。
"""

from src.drift_calibration.conformal import (
    CalibrationSet,
    ConformalError,
    conformal_p_value,
    decide,
    decide_raw,
    p_min,
)
from src.drift_calibration.isolation import IsolationError, assert_disjoint, assert_subset
from src.drift_calibration.protocol import (
    ALPHA,
    AUXILIARY_CHANNEL,
    FROZEN_FILES,
    FUSION_AVAILABLE,
    FUSION_MIN_P,
    N_CALIBRATION,
    P_DENOMINATOR,
    PRIMARY_CHANNEL,
    PROTOCOL_ID,
    TIE_RULE,
    ProtocolError,
    ProtocolMismatchError,
    protocol_fingerprint,
    protocol_params,
    verify_frozen_files,
)
from src.drift_calibration.score import (
    ImageScore,
    ImageScorer,
    ScoreError,
    score_direction,
)

__all__ = [
    "ALPHA",
    "AUXILIARY_CHANNEL",
    "CalibrationSet",
    "ConformalError",
    "FROZEN_FILES",
    "FUSION_AVAILABLE",
    "FUSION_MIN_P",
    "ImageScore",
    "ImageScorer",
    "IsolationError",
    "N_CALIBRATION",
    "P_DENOMINATOR",
    "PRIMARY_CHANNEL",
    "PROTOCOL_ID",
    "ProtocolError",
    "ProtocolMismatchError",
    "ScoreError",
    "TIE_RULE",
    "assert_disjoint",
    "assert_subset",
    "conformal_p_value",
    "decide",
    "decide_raw",
    "p_min",
    "protocol_fingerprint",
    "protocol_params",
    "score_direction",
    "verify_frozen_files",
]
