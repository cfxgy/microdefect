#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记忆库与阈值注册表（第一阶段软件功能）。

职责
----
* 扫描可用的 Group-disjoint（4 类）与官方划分（记录在案）记忆库；
* 校验 global/spatial 记忆库成对存在、阈值可用、形状/类型/有限值合法；
* 校验记忆库成员哈希与阈值记录一致，防止两类协议混用；
* 对不完整类别给出明确错误，不生成、不伪造任何缺失记忆库。

两套协议严格隔离：

* ``group_disjoint``：P5-3.2 记忆库 + P5-3.3 每类校准 JSON（seed=20261111，shot=5）；
* ``official``：官方记忆库（seed/shot 可配置，默认 42/5，global_full + spatial_reference）
  + P5-2.3.3 thresholds.csv。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.features import extractor as p2
from src.memory.mvtec_memory import MemoryBankError, load_bank

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUP_DISJOINT_BANKS = (
    PROJECT_ROOT / "experiments/results/mvtec_ad/stage5_3_2_group_disjoint_features/memory_banks"
)
GROUP_DISJOINT_CALIBRATION = (
    PROJECT_ROOT / "experiments/results/mvtec_ad/stage5_3_3_group_disjoint_evaluation/calibration"
)
GROUP_DISJOINT_SEED = 20261111
GROUP_DISJOINT_SHOT = 5
GROUP_DISJOINT_CATEGORIES = ("bottle", "pill", "toothbrush", "zipper")

OFFICIAL_BANKS = PROJECT_ROOT / "experiments/results/mvtec_ad/memory_banks"
OFFICIAL_THRESHOLDS = (
    PROJECT_ROOT / "experiments/results/mvtec_ad/stage5_2_3_3_calibration/thresholds.csv"
)

PROTOCOLS = ("group_disjoint", "official")

GLOBAL_DIM = 512
SPATIAL_SHAPE = (256, 14, 14)


class RegistryError(RuntimeError):
    """记忆库/阈值配置不可用。"""


@dataclass(frozen=True)
class BankConfig:
    category: str
    protocol: str
    global_bank_dir: Path
    spatial_bank_dir: Path
    calibration_path: Path
    image_threshold: float
    pixel_threshold: float
    seed: int
    shot: int
    image_quantile: float
    pixel_quantile: float
    scoring_version: str
    global_feature_version: str
    spatial_feature_version: str
    weight_sha256: str
    global_member_sha256: str
    spatial_member_sha256: str
    global_shape: tuple[int, ...]
    spatial_shape: tuple[int, ...]

    def summary(self) -> dict:
        return {
            "category": self.category,
            "protocol": self.protocol,
            "seed": self.seed,
            "shot": self.shot,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "global_shape": list(self.global_shape),
            "spatial_shape": list(self.spatial_shape),
            "global_member_sha256": self.global_member_sha256,
            "spatial_member_sha256": self.spatial_member_sha256,
        }


def _validate_bank(directory: Path, expected_type: str) -> tuple[dict, np.ndarray]:
    if not directory.is_dir():
        raise RegistryError(f"记忆库目录不存在：{directory}")
    try:
        bank = load_bank(directory)
    except MemoryBankError as exc:
        raise RegistryError(f"记忆库不可用（{directory}）：{exc}") from exc
    metadata = bank["metadata"]
    features = bank["features"]
    if metadata.get("memory_type") != expected_type:
        raise RegistryError(f"记忆库类型不符：{directory} 期望 {expected_type}，实际 {metadata.get('memory_type')}")
    if features.dtype != np.float32:
        raise RegistryError(f"记忆库 dtype 非 float32：{directory} -> {features.dtype}")
    if not np.isfinite(features).all():
        raise RegistryError(f"记忆库含 NaN/Inf：{directory}")
    if expected_type == "spatial_reference":
        if features.ndim != 4 or tuple(features.shape[1:]) != SPATIAL_SHAPE:
            raise RegistryError(f"空间记忆库形状异常：{directory} -> {features.shape}")
    else:
        if features.ndim != 2 or features.shape[1] != GLOBAL_DIM:
            raise RegistryError(f"全局记忆库形状异常：{directory} -> {features.shape}")
    if int(metadata.get("member_count", -1)) != features.shape[0]:
        raise RegistryError(f"记忆库成员数与数组不一致：{directory}")
    return metadata, features


class MemoryRegistry:
    """某个协议的可用记忆库集合。"""

    def __init__(
        self,
        protocol: str = "group_disjoint",
        *,
        bank_root: Path | str | None = None,
        calibration_root: Path | str | None = None,
        seed: int | None = None,
        shot: int | None = None,
    ) -> None:
        if protocol not in PROTOCOLS:
            raise RegistryError(f"未知协议：{protocol!r}（可选 {PROTOCOLS}）")
        self.protocol = protocol
        self.bank_root = Path(bank_root) if bank_root else (
            GROUP_DISJOINT_BANKS if protocol == "group_disjoint" else OFFICIAL_BANKS
        )
        self.calibration_root = Path(calibration_root) if calibration_root else (
            GROUP_DISJOINT_CALIBRATION if protocol == "group_disjoint" else OFFICIAL_THRESHOLDS
        )
        self.seed = int(seed) if seed is not None else (
            GROUP_DISJOINT_SEED if protocol == "group_disjoint" else 42
        )
        self.shot = int(shot) if shot is not None else GROUP_DISJOINT_SHOT

    # ---- 扫描 ----
    def available_categories(self) -> list[str]:
        if self.protocol == "group_disjoint":
            candidates = GROUP_DISJOINT_CATEGORIES
            return [
                c for c in candidates
                if (self.bank_root / c / "global_reference").is_dir()
                and (self.bank_root / c / "spatial_reference").is_dir()
                and (self.calibration_root / f"{c}_calibration.json").is_file()
            ]
        if not self.bank_root.is_dir():
            return []
        categories = []
        for path in sorted(self.bank_root.iterdir()):
            if not path.is_dir():
                continue
            base = path / f"seed_{self.seed}" / f"shot_{self.shot}"
            if (base / "global_full").is_dir() and (base / "spatial_reference").is_dir():
                categories.append(path.name)
        return categories

    # ---- 加载 ----
    def load_config(self, category: str) -> BankConfig:
        if self.protocol == "group_disjoint":
            return self._load_group_disjoint(category)
        return self._load_official(category)

    def _load_group_disjoint(self, category: str) -> BankConfig:
        if category not in GROUP_DISJOINT_CATEGORIES:
            raise RegistryError(
                f"未知类别：{category!r}（Group-disjoint 可用类别：{GROUP_DISJOINT_CATEGORIES}）"
            )
        cal_path = self.calibration_root / f"{category}_calibration.json"
        if not cal_path.is_file():
            raise RegistryError(f"缺少校准阈值文件：{cal_path}")
        try:
            calibration = json.loads(cal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"校准阈值文件损坏：{cal_path}（{exc}）") from exc

        global_dir = self.bank_root / category / "global_reference"
        spatial_dir = self.bank_root / category / "spatial_reference"
        g_meta, g_features = _validate_bank(global_dir, "global_reference")
        s_meta, s_features = _validate_bank(spatial_dir, "spatial_reference")

        if g_meta.get("protocol_version") != "v1.0" or s_meta.get("protocol_version") != "v1.0":
            raise RegistryError(f"{category}: 记忆库协议版本非 v1.0")
        if int(g_meta.get("seed", -1)) != GROUP_DISJOINT_SEED or int(s_meta.get("seed", -1)) != GROUP_DISJOINT_SEED:
            raise RegistryError(f"{category}: 记忆库 seed 与 Group-disjoint 配置不一致")
        if int(g_meta.get("shot", -1)) != self.shot or int(s_meta.get("shot", -1)) != self.shot:
            raise RegistryError(f"{category}: 记忆库 shot 与配置不一致")

        image_threshold = float(calibration["image_threshold"])
        pixel_threshold = float(calibration["pixel_threshold"])
        for name, value in (("image_threshold", image_threshold), ("pixel_threshold", pixel_threshold)):
            if not np.isfinite(value):
                raise RegistryError(f"{category}: {name} 非有限值")
        if not (0.0 <= image_threshold <= 2.0 and 0.0 <= pixel_threshold <= 2.0):
            raise RegistryError(f"{category}: 阈值超出余弦距离范围 [0,2]")

        if calibration.get("memory_global_member_sha256") != g_meta.get("member_manifest_sha256"):
            raise RegistryError(f"{category}: 图像阈值与 global 记忆库成员哈希不匹配")
        if calibration.get("memory_spatial_member_sha256") != s_meta.get("member_manifest_sha256"):
            raise RegistryError(f"{category}: 像素阈值与 spatial 记忆库成员哈希不匹配")

        versions = {
            "global": calibration.get("global_feature_version"),
            "spatial": calibration.get("spatial_feature_version"),
            "weight": calibration.get("weight_sha256"),
        }
        for meta in (g_meta, s_meta):
            if meta.get("global_feature_version") != versions["global"]:
                raise RegistryError(f"{category}: 全局特征版本不一致")
            if meta.get("spatial_feature_version") != versions["spatial"]:
                raise RegistryError(f"{category}: 空间特征版本不一致")
            if meta.get("weight_sha256") != versions["weight"]:
                raise RegistryError(f"{category}: 权重哈希不一致")

        return BankConfig(
            category=category,
            protocol="group_disjoint",
            global_bank_dir=global_dir,
            spatial_bank_dir=spatial_dir,
            calibration_path=cal_path,
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
            seed=GROUP_DISJOINT_SEED,
            shot=self.shot,
            image_quantile=float(calibration.get("image_quantile", 0.95)),
            pixel_quantile=float(calibration.get("pixel_quantile", 0.99)),
            scoring_version=str(calibration.get("scoring_version", "")),
            global_feature_version=str(versions["global"]),
            spatial_feature_version=str(versions["spatial"]),
            weight_sha256=str(versions["weight"]),
            global_member_sha256=str(g_meta["member_manifest_sha256"]),
            spatial_member_sha256=str(s_meta["member_manifest_sha256"]),
            global_shape=tuple(g_features.shape),
            spatial_shape=tuple(s_features.shape),
        )

    def _load_official(self, category: str) -> BankConfig:
        base = self.bank_root / category / f"seed_{self.seed}" / f"shot_{self.shot}"
        global_dir = base / "global_full"
        spatial_dir = base / "spatial_reference"
        g_meta, g_features = _validate_bank(global_dir, "global_full")
        s_meta, s_features = _validate_bank(spatial_dir, "spatial_reference")

        if not Path(self.calibration_root).is_file():
            raise RegistryError(f"缺少官方阈值文件：{self.calibration_root}")
        row = None
        with Path(self.calibration_root).open("r", encoding="utf-8-sig", newline="") as handle:
            for candidate in csv.DictReader(handle):
                if (candidate.get("category") == category
                        and int(candidate.get("seed", -1)) == self.seed
                        and int(candidate.get("shot", -1)) == self.shot):
                    row = candidate
                    break
        if row is None:
            raise RegistryError(f"{category}: 官方阈值表中无 seed={self.seed}, shot={self.shot} 记录")
        image_threshold = float(row["image_threshold"])
        pixel_threshold = float(row["pixel_threshold"])
        for name, value in (("image_threshold", image_threshold), ("pixel_threshold", pixel_threshold)):
            if not np.isfinite(value):
                raise RegistryError(f"{category}: {name} 非有限值")
        if row.get("global_full_member_hash") != g_meta.get("member_manifest_sha256"):
            raise RegistryError(f"{category}: 官方图像阈值与 global_full 记忆库成员哈希不匹配")
        if row.get("spatial_reference_member_hash") != s_meta.get("member_manifest_sha256"):
            raise RegistryError(f"{category}: 官方像素阈值与 spatial 记忆库成员哈希不匹配")

        return BankConfig(
            category=category,
            protocol="official",
            global_bank_dir=global_dir,
            spatial_bank_dir=spatial_dir,
            calibration_path=Path(self.calibration_root),
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
            seed=self.seed,
            shot=self.shot,
            image_quantile=0.95,
            pixel_quantile=0.99,
            scoring_version="protocol-v1.0-cosine",
            global_feature_version=str(g_meta.get("global_feature_version")),
            spatial_feature_version=str(s_meta.get("spatial_feature_version")),
            weight_sha256=str(g_meta.get("weight_sha256")),
            global_member_sha256=str(g_meta["member_manifest_sha256"]),
            spatial_member_sha256=str(s_meta["member_manifest_sha256"]),
            global_shape=tuple(g_features.shape),
            spatial_shape=tuple(s_features.shape),
        )


def expected_versions() -> dict:
    """当前冻结的特征/预处理版本（用于展示与测试）。"""
    return {
        "global_feature_version": p2.FEATURE_VERSION,
        "preprocess_version": p2.PREPROCESS_VERSION,
        "backbone": p2.BACKBONE_NAME,
        "weights_tag": p2.WEIGHTS_TAG,
    }
