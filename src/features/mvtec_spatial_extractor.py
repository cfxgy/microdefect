#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MVTec AD 空间特征提取器（P5-2.2 新增，不修改 P2 冻结模块）。

复用 P2 的**同一官方 ResNet18 权重**与**同一预处理定义**，额外提供
``layer3`` 空间特征图（256×14×14 @ 224 输入），用于后续像素级定位。

与 P2 的关系
------------
* 全局 512 维特征：直接复用 ``FeatureExtractor``（P2）语义，本模块只做封装；
* 空间 256×14×14 特征：新增接口，输出**不经过** avgpool/fc；
* 两条路径共享同一模型实例与同一权重，均处于 eval 模式、参数冻结；
* P2 的 ``extractor.py`` / ``cache.py`` 不被修改。

颜色处理（关键）
----------------
MVTec 同时包含 RGB 与灰度图：

* RGB 图像：保持 RGB，**不转灰度**；
* 灰度（L）图像：复制通道为 3 通道 RGB（与 P2 对 NEU 灰度图的处理一致）；
* 其他模式（RGBA/P/CMYK）：显式 ``convert("RGB")``。

注意：P2 的 ``load_image_rgb`` 会把所有图像转 L 再复制三通道（为 NEU-CLS 设计），
因此**不能**用于 MVTec 彩色图像；本模块提供 ``load_image_rgb_mvtec``。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.features import extractor as p2

SPATIAL_LAYER = "layer3"
SPATIAL_CHANNELS = 256
SPATIAL_GRID = 14
SPATIAL_FEATURE_VERSION = "f1/resnet18-layer3-256x14x14-raw-fp32-notf32"
SPATIAL_DTYPE = np.float32


@lru_cache(maxsize=1)
def weights_sha256() -> str:
    path = p2.expected_weights_path()
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_rgb_mvtec(image_path: str | Path) -> Image.Image:
    """读取 MVTec 图像并保留颜色语义（RGB 不转灰度）。"""
    with Image.open(image_path) as handle:
        if handle.mode == "RGB":
            return handle.copy()
        if handle.mode == "L":
            gray = handle.copy()
            return Image.merge("RGB", (gray, gray, gray))
        return handle.convert("RGB")


@dataclass(frozen=True)
class SpatialInfo:
    backbone: str
    weights_tag: str
    weights_sha256: str
    preprocess_version: str
    spatial_feature_version: str
    layer: str
    channels: int
    grid: int
    dtype: str
    image_size: int
    device: str
    batch_size: int


class MvtecSpatialExtractor:
    """共享 P2 权重的双路径提取器（全局 512 维 + layer3 256×14×14）。"""

    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        allow_download: bool = False,
        batch_size: int = 8,
        model: nn.Module | None = None,
    ) -> None:
        p2.configure_backend_precision()
        if model is None:
            model, resolved = p2.build_model(device, allow_download=allow_download)
        else:
            resolved = p2.resolve_device(device)
            model = model.to(resolved)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

        self.model = model
        self.device = resolved
        self.transform = p2.build_transform()
        self.batch_size = int(batch_size)

        # 共享同一批子模块，不复制参数；仅用于 layer3 截断前向
        self._trunk = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3,
        ).to(resolved)
        self._trunk.eval()
        for parameter in self._trunk.parameters():
            parameter.requires_grad_(False)

    # ---------- 元信息 ----------
    @property
    def info(self) -> SpatialInfo:
        return SpatialInfo(
            backbone=p2.BACKBONE_NAME,
            weights_tag=p2.WEIGHTS_TAG,
            weights_sha256=weights_sha256(),
            preprocess_version=p2.PREPROCESS_VERSION,
            spatial_feature_version=SPATIAL_FEATURE_VERSION,
            layer=SPATIAL_LAYER,
            channels=SPATIAL_CHANNELS,
            grid=SPATIAL_GRID,
            dtype="float32",
            image_size=p2.IMAGE_SIZE,
            device=str(self.device),
            batch_size=self.batch_size,
        )

    def describe(self) -> dict:
        info = self.info
        return {
            "backbone": info.backbone,
            "weights": f"ResNet18_Weights.{info.weights_tag}",
            "weights_path": str(p2.expected_weights_path()),
            "weights_sha256": info.weights_sha256,
            "preprocess_version": info.preprocess_version,
            "global_feature_version": p2.FEATURE_VERSION,
            "spatial_feature_version": info.spatial_feature_version,
            "layer": info.layer,
            "global_dim": p2.EMBED_DIM,
            "spatial_shape": [info.channels, info.grid, info.grid],
            "dtype": info.dtype,
            "image_size": info.image_size,
            "device": info.device,
            "batch_size": info.batch_size,
        }

    # ---------- 预处理 ----------
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL → ``[3,224,224]``；RGB 保持颜色，L 复制为三通道。"""
        if image.mode != "RGB":
            gray = image.convert("L")
            image = Image.merge("RGB", (gray, gray, gray))
        return p2.preprocess_pil(image, self.transform)

    def preprocess_path(self, image_path: str | Path) -> torch.Tensor:
        return self.preprocess(load_image_rgb_mvtec(image_path))

    def _validate_batch(self, batch: torch.Tensor) -> None:
        if batch.ndim != 4:
            raise ValueError(f"期望 4 维张量 [B,3,H,W]，实际 {tuple(batch.shape)}")
        if batch.shape[1] != 3:
            raise ValueError(f"期望 3 通道，实际 {batch.shape[1]}")
        if batch.shape[2] != p2.IMAGE_SIZE or batch.shape[3] != p2.IMAGE_SIZE:
            raise ValueError(
                f"期望 {p2.IMAGE_SIZE}×{p2.IMAGE_SIZE}，实际 {batch.shape[2]}×{batch.shape[3]}"
            )

    # ---------- 推理 ----------
    @torch.inference_mode()
    def extract_global_tensor(self, batch: torch.Tensor) -> np.ndarray:
        """``[B,3,224,224]`` → ``[B,512]``（与 P2 完全一致的语义）。"""
        self._validate_batch(batch)
        batch = batch.to(self.device, non_blocking=False)
        features = self.model(batch)
        if features.ndim != 2 or features.shape[1] != p2.EMBED_DIM:
            raise RuntimeError(f"全局特征维度异常：{tuple(features.shape)}")
        array = features.detach().to("cpu").float().numpy()
        if not np.isfinite(array).all():
            raise RuntimeError("全局特征出现 NaN 或 Inf")
        return array

    @torch.inference_mode()
    def extract_spatial_tensor(self, batch: torch.Tensor) -> np.ndarray:
        """``[B,3,224,224]`` → ``[B,256,14,14]`` float32。"""
        self._validate_batch(batch)
        batch = batch.to(self.device, non_blocking=False)
        features = self._trunk(batch)
        expected = (batch.shape[0], SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID)
        if tuple(features.shape) != expected:
            raise RuntimeError(f"空间特征形状异常：{tuple(features.shape)} != {expected}")
        array = features.detach().to("cpu").float().numpy()
        if array.dtype != SPATIAL_DTYPE:
            raise RuntimeError(f"空间特征 dtype 异常：{array.dtype}")
        if not np.isfinite(array).all():
            raise RuntimeError("空间特征出现 NaN 或 Inf")
        return array

    @torch.inference_mode()
    def extract_both_tensor(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """一次前向同时返回全局与空间特征（共享 layer1–layer3 计算）。"""
        self._validate_batch(batch)
        batch = batch.to(self.device, non_blocking=False)
        spatial = self._trunk(batch)
        pooled = self.model.avgpool(self.model.layer4(spatial))
        global_feature = torch.flatten(pooled, 1)
        expected_spatial = (batch.shape[0], SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID)
        if tuple(spatial.shape) != expected_spatial:
            raise RuntimeError(f"空间特征形状异常：{tuple(spatial.shape)}")
        if tuple(global_feature.shape) != (batch.shape[0], p2.EMBED_DIM):
            raise RuntimeError(f"全局特征形状异常：{tuple(global_feature.shape)}")
        spatial_np = spatial.detach().to("cpu").float().numpy()
        global_np = global_feature.detach().to("cpu").float().numpy()
        if not (np.isfinite(spatial_np).all() and np.isfinite(global_np).all()):
            raise RuntimeError("特征出现 NaN 或 Inf")
        return global_np, spatial_np

    # ---------- 路径/批量接口 ----------
    def _batched(self, paths: list[str | Path]) -> list[list[str | Path]]:
        size = max(1, self.batch_size)
        return [paths[i:i + size] for i in range(0, len(paths), size)]

    def extract_global_paths(self, paths: list[str | Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, p2.EMBED_DIM), dtype=np.float32)
        chunks = []
        for chunk in self._batched(paths):
            batch = torch.stack([self.preprocess_path(p) for p in chunk], dim=0)
            chunks.append(self.extract_global_tensor(batch))
            del batch
        return np.concatenate(chunks, axis=0)

    def extract_spatial_paths(self, paths: list[str | Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID), dtype=SPATIAL_DTYPE)
        chunks = []
        for chunk in self._batched(paths):
            batch = torch.stack([self.preprocess_path(p) for p in chunk], dim=0)
            chunks.append(self.extract_spatial_tensor(batch))
            del batch
        return np.concatenate(chunks, axis=0)

    def extract_paths_both(self, paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray]:
        if not paths:
            return (
                np.zeros((0, p2.EMBED_DIM), dtype=np.float32),
                np.zeros((0, SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID), dtype=SPATIAL_DTYPE),
            )
        globals_, spatials = [], []
        for chunk in self._batched(paths):
            batch = torch.stack([self.preprocess_path(p) for p in chunk], dim=0)
            g, s = self.extract_both_tensor(batch)
            globals_.append(g)
            spatials.append(s)
            del batch
        return np.concatenate(globals_, axis=0), np.concatenate(spatials, axis=0)

    def extract_global_path(self, path: str | Path) -> np.ndarray:
        return self.extract_global_paths([path])[0]

    def extract_spatial_path(self, path: str | Path) -> np.ndarray:
        return self.extract_spatial_paths([path])[0]
