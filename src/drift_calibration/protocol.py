#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2.1 阶段 4：冻结协议参数与哈希校验。

协议身份：``p4.2.1-stage3-v2.1``（见 experiments/results/drift_monitoring_calibration/）。

本模块只定义常量与校验，不做任何计算。修改这里的任何常量都等同于修改协议，
必须在报告中说明并重新冻结，不得静默更改。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL_ID = "p4.2.1-stage3-v2.1"

# ---- 冻结数值 ----
ALPHA = 0.05
N_CALIBRATION = 25
P_DENOMINATOR = N_CALIBRATION + 1  # 26
TIE_RULE = ">="  # cal_score >= test_score 计入计数
WINDOW_SIZE = 16  # 仅供对照；本阶段不实现窗口级告警
REFERENCE_POOL_SIZE = 25
PRIMARY_CHANNEL = "A"  # 唯一正式告警通道
AUXILIARY_CHANNEL = "B"  # 辅助指标，不参与告警
FUSION_AVAILABLE = False  # Bonferroni 在 n=25 时不可用
FUSION_MIN_P = 2.0 / (N_CALIBRATION + 1)  # 0.076923  > alpha

SCORE_FORMULA = (
    "z_j(x) = |stat_j(x) - median_j| / scale_j ; "
    "score_A(x) = sqrt(mean_j z_j(x)^2) ; larger = more anomalous"
)
P_VALUE_FORMULA = "p = (1 + count(cal_score >= test_score)) / 26"

# 冻结文件清单（相对结果目录）
FROZEN_FILES = (
    "protocol_design.md",
    "protocol_comparison.csv",
    "protocol_feasibility.json",
    "protocol_review_changes.md",
)


class ProtocolError(RuntimeError):
    """协议参数或调用非法。"""


class ProtocolMismatchError(ProtocolError):
    """冻结文件哈希与记录不一致。"""


def protocol_params() -> dict:
    """返回可序列化的协议参数（用于指纹与审计）。"""
    return {
        "protocol_id": PROTOCOL_ID,
        "alpha": ALPHA,
        "n_calibration": N_CALIBRATION,
        "p_denominator": P_DENOMINATOR,
        "tie_rule": TIE_RULE,
        "window_size": WINDOW_SIZE,
        "reference_pool_size": REFERENCE_POOL_SIZE,
        "primary_channel": PRIMARY_CHANNEL,
        "auxiliary_channel": AUXILIARY_CHANNEL,
        "fusion_available": FUSION_AVAILABLE,
        "fusion_min_p": round(FUSION_MIN_P, 12),
        "score_formula": SCORE_FORMULA,
        "p_value_formula": P_VALUE_FORMULA,
    }


def protocol_fingerprint() -> str:
    """协议参数指纹（sha256，前 16 位）。"""
    material = json.dumps(protocol_params(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_files(
    base_dir: str | Path,
    *,
    record_name: str = "protocol_design_sha256.json",
) -> dict:
    """校验冻结文件哈希是否与记录一致。

    Raises
    ------
    ProtocolMismatchError
        任一文件缺失或哈希不一致。
    """
    base = Path(base_dir)
    record_path = base / record_name
    if not record_path.is_file():
        raise ProtocolMismatchError(f"缺少冻结记录：{record_path}")

    record = json.loads(record_path.read_text(encoding="utf-8"))
    files = record.get("files_v2", {})
    if not files:
        raise ProtocolMismatchError(f"冻结记录中没有 files_v2：{record_path}")

    checked: dict[str, dict] = {}
    mismatches: list[str] = []
    for name, item in files.items():
        path = base / name
        if not path.is_file():
            mismatches.append(f"缺失文件：{name}")
            continue
        actual = _sha256_file(path)
        match = actual == item.get("sha256")
        checked[name] = {"expected": item.get("sha256"), "actual": actual, "match": match}
        if not match:
            mismatches.append(f"哈希不一致：{name}")

    if mismatches:
        raise ProtocolMismatchError("冻结文件校验失败：" + "；".join(mismatches))

    return {
        "protocol_id": record.get("version", PROTOCOL_ID),
        "files": checked,
        "all_match": True,
    }
