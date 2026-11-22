#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单图端到端缺陷检测器（第一阶段软件功能）。

流程：读取图像 → 冻结预处理 → ResNet18 双路径特征 → 该类记忆库余弦评分
→ 图像级阈值判定 → 像素级热图 → 结构化结果。

严格复用冻结实现：
* 特征：``src.features.mvtec_spatial_extractor.MvtecSpatialExtractor``；
* 评分：``src.anomaly.mvtec_protocol_scoring``（余弦、逐 patch L2、双线性 align_corners=False）；
* 记忆库：``src.memory.mvtec_memory.load_bank``；
* 阈值：由 ``src.core.registry`` 从已冻结的校准记录读取，**不在推理时校准**。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.anomaly.mvtec_protocol_scoring import (
    SCORING_VERSION,
    score_image_cosine,
    score_spatial_cosine,
)
from src.core.registry import BankConfig, MemoryRegistry, RegistryError
from src.core.weights import WeightsUnavailableError, ensure_local_weights
from src.features import extractor as p2
from src.features.mvtec_spatial_extractor import (
    MvtecSpatialExtractor,
    load_image_rgb_mvtec,
    weights_sha256,
)
from src.memory.mvtec_memory import MemoryBankError, load_bank


class DetectorError(RuntimeError):
    """检测输入或运行错误。"""


@dataclass
class DetectionResult:
    image_path: str
    category: str
    protocol: str
    image_score: float
    image_threshold: float
    is_anomaly: bool
    pixel_threshold: float
    anomaly_heatmap: np.ndarray
    patch_distance_map: np.ndarray
    image_size: tuple[int, int]
    model_version: dict
    memory_bank_version: dict
    elapsed_time: float
    created_at: str
    heatmap_stats: dict = field(default_factory=dict)
    output_files: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "category": self.category,
            "protocol": self.protocol,
            "image_score": self.image_score,
            "image_threshold": self.image_threshold,
            "is_anomaly": self.is_anomaly,
            "pixel_threshold": self.pixel_threshold,
            "image_size": list(self.image_size),
            "heatmap_shape": list(self.anomaly_heatmap.shape),
            "patch_map_shape": list(self.patch_distance_map.shape),
            "heatmap_stats": self.heatmap_stats,
            "model_version": self.model_version,
            "memory_bank_version": self.memory_bank_version,
            "elapsed_time": self.elapsed_time,
            "created_at": self.created_at,
            "scoring_version": SCORING_VERSION,
            "output_files": self.output_files,
        }

    def save(self, out_dir: str | Path, *, stem: str | None = None) -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = stem or Path(self.image_path).stem
        json_path = out / f"{stem}_detect.json"
        heatmap_png = out / f"{stem}_heatmap.png"
        overlay_png = out / f"{stem}_overlay.png"

        _save_heatmap_png(self.anomaly_heatmap, heatmap_png)
        _save_overlay_png(self.image_path, self.anomaly_heatmap, overlay_png)

        self.output_files = {
            "result_json": str(json_path),
            "heatmap_png": str(heatmap_png),
            "overlay_png": str(overlay_png),
        }
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.output_files


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float32)
    low = float(values.min())
    high = float(values.max())
    if not np.isfinite(values).all():
        raise DetectorError("热图包含 NaN/Inf")
    if high - low <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values - low) / (high - low)


def _colorize(heatmap: np.ndarray) -> Image.Image:
    import matplotlib

    normalized = _normalize_heatmap(heatmap)
    colormap = matplotlib.colormaps["jet"]
    rgb = (colormap(normalized)[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _save_heatmap_png(heatmap: np.ndarray, path: Path) -> None:
    _colorize(heatmap).save(path)


def _save_overlay_png(image_path: str, heatmap: np.ndarray, path: Path) -> None:
    with Image.open(image_path) as handle:
        base = handle.convert("RGB")
    colored = _colorize(heatmap)
    if colored.size != base.size:
        colored = colored.resize(base.size, Image.BILINEAR)
    Image.blend(base, colored, 0.45).save(path)


class Detector:
    """单类别、单记忆库配置的检测器。"""

    def __init__(
        self,
        category: str,
        *,
        protocol: str = "group_disjoint",
        device: str = "auto",
        registry: MemoryRegistry | None = None,
        seed: int | None = None,
        shot: int | None = None,
        batch_size: int = 1,
    ) -> None:
        self.registry = registry or MemoryRegistry(protocol=protocol, seed=seed, shot=shot)
        try:
            self.config: BankConfig = self.registry.load_config(category)
        except RegistryError:
            raise
        if self.config.protocol != self.registry.protocol:
            raise DetectorError("记忆库协议与检测器协议不一致")

        self.category = category
        self.protocol = self.config.protocol
        self.device = p2.resolve_device(device)
        # 先解析本地权重并同步 torch 缓存路径，避免"默认缓存可写但为空"时误报缺失
        try:
            self.weights_path = ensure_local_weights()
        except WeightsUnavailableError as exc:
            raise DetectorError(str(exc)) from exc
        self.extractor = MvtecSpatialExtractor(
            device=self.device, batch_size=batch_size, allow_download=False
        )
        try:
            self.global_bank = load_bank(self.config.global_bank_dir)
            self.spatial_bank = load_bank(self.config.spatial_bank_dir)
        except MemoryBankError as exc:
            raise DetectorError(f"记忆库加载失败：{exc}") from exc
        if self.global_bank["features"].shape != self.config.global_shape:
            raise DetectorError("global 记忆库形状与配置不一致")
        if self.spatial_bank["features"].shape != self.config.spatial_shape:
            raise DetectorError("spatial 记忆库形状与配置不一致")
        if not (np.isfinite(self.config.image_threshold) and np.isfinite(self.config.pixel_threshold)):
            raise DetectorError("阈值非有限值")

    # ---- 信息 ----
    def describe(self) -> dict:
        return {
            "category": self.category,
            "protocol": self.protocol,
            "device": str(self.device),
            "config": self.config.summary(),
            "model_version": self._model_version(),
            "memory_bank_version": self._bank_version(),
        }

    def _model_version(self) -> dict:
        return {
            "backbone": p2.BACKBONE_NAME,
            "weights_tag": p2.WEIGHTS_TAG,
            "weight_sha256": weights_sha256(),
            "weights_path": str(self.weights_path),
            "global_feature_version": p2.FEATURE_VERSION,
            "spatial_feature_version": self.spatial_bank["metadata"]["spatial_feature_version"],
            "preprocess_version": p2.PREPROCESS_VERSION,
        }

    def _bank_version(self) -> dict:
        return {
            "protocol": self.protocol,
            "seed": int(self.config.seed),
            "shot": int(self.config.shot),
            "global_member_count": int(self.global_bank["features"].shape[0]),
            "spatial_member_count": int(self.spatial_bank["features"].shape[0]),
            "global_member_sha256": self.config.global_member_sha256,
            "spatial_member_sha256": self.config.spatial_member_sha256,
            "global_shape": list(self.config.global_shape),
            "spatial_shape": list(self.config.spatial_shape),
        }

    # ---- 检测 ----
    def detect(self, image_path: str | Path) -> DetectionResult:
        started = time.perf_counter()
        path = Path(image_path)
        if not path.is_file():
            raise DetectorError(f"图像不存在：{path}")
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            raise DetectorError(f"不支持的图像格式：{path.suffix}")
        try:
            image = load_image_rgb_mvtec(path)
        except Exception as exc:  # noqa: BLE001
            raise DetectorError(f"图像读取失败：{path}（{type(exc).__name__}: {exc}）") from exc
        width, height = image.size
        if width <= 0 or height <= 0:
            raise DetectorError(f"图像尺寸非法：{path}")

        tensor = self.extractor.preprocess(image).unsqueeze(0)
        global_feature, spatial_feature = self.extractor.extract_both_tensor(tensor)
        image_score = float(
            score_image_cosine(
                global_feature, self.global_bank["features"], device=str(self.device)
            ).distances[0]
        )
        spatial_result = score_spatial_cosine(
            spatial_feature[0],
            self.spatial_bank["features"],
            (height, width),
            device=str(self.device),
        )
        heatmap = spatial_result.heatmaps[0]
        patch_map = spatial_result.patch_distances[0]
        if not np.isfinite(image_score) or not np.isfinite(heatmap).all():
            raise DetectorError("检测结果包含 NaN/Inf")

        is_anomaly = bool(image_score > self.config.image_threshold)
        elapsed = time.perf_counter() - started
        return DetectionResult(
            image_path=str(path),
            category=self.category,
            protocol=self.protocol,
            image_score=image_score,
            image_threshold=float(self.config.image_threshold),
            is_anomaly=is_anomaly,
            pixel_threshold=float(self.config.pixel_threshold),
            anomaly_heatmap=heatmap.astype(np.float32),
            patch_distance_map=patch_map.astype(np.float32),
            image_size=(int(width), int(height)),
            model_version=self._model_version(),
            memory_bank_version=self._bank_version(),
            elapsed_time=float(elapsed),
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            heatmap_stats={
                "min": float(heatmap.min()),
                "max": float(heatmap.max()),
                "mean": float(heatmap.mean()),
            },
        )
