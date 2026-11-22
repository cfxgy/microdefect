#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MVTec AD 图像级与像素级异常评分（P5-2.3.2，轻量 PatchCore 风格）。

**口径说明（重要）**
本模块按 P5-2.3.2 指令实现：

* 图像级：测试图像的 512 维全局特征到 ``global_full`` 记忆库的**最小欧氏距离**；
* 像素级：测试图像每个 14×14 位置到 ``spatial_reference`` 全部 patch 的**最小欧氏距离**；
* 不增加 L2 归一化、PCA、标准化；
* 热图用双线性上采样，``align_corners=False``。

上述距离口径（欧氏）与协议 v1.0 §3/§5 的余弦距离定义**不同**，属本阶段经人工确认的
实现口径，需在后续正式实验前通过协议勘误版本对齐；本模块不修改协议与既有记忆库。

本模块只计算距离与热图：不做阈值、不做分类、不评估任何指标。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

GLOBAL_DIM = 512
SPATIAL_CHANNELS = 256
SPATIAL_GRID = 14
PATCH_COUNT = SPATIAL_GRID * SPATIAL_GRID


class ScoringError(RuntimeError):
    """评分输入/记忆库不合法。"""


@dataclass(frozen=True)
class ImageScoreResult:
    distances: np.ndarray  # (B,) float32
    nearest_indices: np.ndarray  # (B,) int64


@dataclass(frozen=True)
class SpatialScoreResult:
    patch_distances: np.ndarray  # (B,14,14) float32
    heatmaps: np.ndarray  # (B,H,W) float32


def _to_tensor(array, *, device: str, name: str) -> torch.Tensor:
    if torch.is_tensor(array):
        tensor = array.detach().to(device=device, dtype=torch.float64)
    else:
        tensor = torch.as_tensor(np.asarray(array), dtype=torch.float64, device=device)
    if not torch.isfinite(tensor).all():
        raise ScoringError(f"{name} 含 NaN 或 Inf")
    return tensor


def _validate_image_memory(memory: torch.Tensor) -> None:
    if memory.ndim != 2 or memory.shape[1] != GLOBAL_DIM:
        raise ScoringError(f"全局记忆库形状异常：{tuple(memory.shape)}，期望 [N,{GLOBAL_DIM}]")
    if memory.shape[0] == 0:
        raise ScoringError("全局记忆库为空")


def _validate_spatial_memory(memory: torch.Tensor) -> torch.Tensor:
    if memory.ndim == 4:
        if memory.shape[1:] != (SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID):
            raise ScoringError(f"空间记忆库形状异常：{tuple(memory.shape)}")
        memory = memory.permute(0, 2, 3, 1).reshape(-1, SPATIAL_CHANNELS)
    if memory.ndim != 2 or memory.shape[1] != SPATIAL_CHANNELS:
        raise ScoringError(f"空间记忆库形状异常：{tuple(memory.shape)}，期望 [K,{SPATIAL_CHANNELS}]")
    if memory.shape[0] == 0:
        raise ScoringError("空间记忆库为空")
    return memory


def pairwise_euclidean(query: np.ndarray, memory: np.ndarray, *, chunk: int = 4096) -> np.ndarray:
    """小规模/测试用：返回完整欧氏距离矩阵 ``[Q,M]``（float32）。"""
    q = np.asarray(query, dtype=np.float32)
    m = np.asarray(memory, dtype=np.float32)
    if q.ndim != 2 or m.ndim != 2 or q.shape[1] != m.shape[1]:
        raise ScoringError(f"维度不匹配：{q.shape} vs {m.shape}")
    q64 = q.astype(np.float64)
    m64 = m.astype(np.float64)
    q2 = np.sum(q64 * q64, axis=1, keepdims=True)
    out = np.empty((q64.shape[0], m64.shape[0]), dtype=np.float64)
    for start in range(0, m64.shape[0], chunk):
        block = m64[start:start + chunk]
        m2 = np.sum(block * block, axis=1, keepdims=True).T
        d2 = q2 + m2 - 2.0 * (q64 @ block.T)
        out[:, start:start + chunk] = d2
    np.maximum(out, 0.0, out=out)
    return np.sqrt(out).astype(np.float32)


def score_image(
    global_features: np.ndarray,
    global_memory: np.ndarray,
    *,
    device: str = "cpu",
    memory_chunk: int = 8192,
) -> ImageScoreResult:
    """``[D]`` 或 ``[B,D]`` → 到 ``global_memory [N,D]`` 的最小欧氏距离。"""
    features = _to_tensor(global_features, device=device, name="global_features")
    memory = _to_tensor(global_memory, device=device, name="global_memory")
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim != 2 or features.shape[1] != GLOBAL_DIM:
        raise ScoringError(f"全局特征形状异常：{tuple(features.shape)}，期望 [B,{GLOBAL_DIM}]")
    _validate_image_memory(memory)

    q2 = (features * features).sum(dim=1, keepdim=True)
    best = torch.full((features.shape[0],), float("inf"), dtype=torch.float64, device=device)
    best_index = torch.zeros((features.shape[0],), dtype=torch.long, device=device)
    for start in range(0, memory.shape[0], memory_chunk):
        block = memory[start:start + memory_chunk]
        m2 = (block * block).sum(dim=1).unsqueeze(0)
        d2 = q2 + m2 - 2.0 * (features @ block.T)
        d2 = torch.clamp(d2, min=0.0)
        block_min, block_arg = d2.min(dim=1)
        update = block_min < best
        best = torch.where(update, block_min, best)
        best_index = torch.where(update, block_arg + start, best_index)

    distances = torch.sqrt(best)
    # 完全相同向量的数值残差归零（float64 计算后 sqrt 仍可能带来 ~1e-6 残差）
    distances = torch.where(distances < 1e-5, torch.zeros_like(distances), distances)
    return ImageScoreResult(
        distances=distances.detach().to("cpu").numpy().astype(np.float32),
        nearest_indices=best_index.detach().to("cpu").numpy().astype(np.int64),
    )


def _upsample_bilinear(distance_map: np.ndarray, output_size: tuple[int, int], *, device: str) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(distance_map, dtype=np.float32), device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        tensor, size=tuple(output_size), mode="bilinear", align_corners=False
    )
    return upsampled.detach().to("cpu").numpy().astype(np.float32)[0, 0]


def score_spatial(
    spatial_features: np.ndarray,
    spatial_memory: np.ndarray,
    output_size: tuple[int, int],
    *,
    device: str = "cpu",
    memory_chunk: int = 4096,
) -> SpatialScoreResult:
    """``[256,14,14]`` 或 ``[B,256,14,14]`` → patch 距离图 + 上采样热图。"""
    memory = _validate_spatial_memory(
        _to_tensor(spatial_memory, device=device, name="spatial_memory")
    )
    features = _to_tensor(spatial_features, device=device, name="spatial_features")
    if features.ndim == 3:
        features = features.unsqueeze(0)
    if features.ndim != 4 or features.shape[1:] != (SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID):
        raise ScoringError(
            f"空间特征形状异常：{tuple(features.shape)}，期望 [B,{SPATIAL_CHANNELS},"
            f"{SPATIAL_GRID},{SPATIAL_GRID}]"
        )
    if len(output_size) != 2 or int(output_size[0]) <= 0 or int(output_size[1]) <= 0:
        raise ScoringError(f"输出尺寸非法：{output_size}")

    batch = features.shape[0]
    patches = features.reshape(batch, SPATIAL_CHANNELS, PATCH_COUNT).permute(0, 2, 1).contiguous()
    maps = torch.empty((batch, PATCH_COUNT), dtype=torch.float64, device=device)
    for index in range(batch):
        query = patches[index]
        q2 = (query * query).sum(dim=1, keepdim=True)
        best = torch.full((PATCH_COUNT,), float("inf"), dtype=torch.float64, device=device)
        for start in range(0, memory.shape[0], memory_chunk):
            block = memory[start:start + memory_chunk]
            m2 = (block * block).sum(dim=1).unsqueeze(0)
            d2 = torch.clamp(q2 + m2 - 2.0 * (query @ block.T), min=0.0)
            best = torch.minimum(best, d2.min(dim=1).values)
        maps[index] = torch.sqrt(best)

    maps = maps.reshape(batch, SPATIAL_GRID, SPATIAL_GRID)
    maps = torch.where(maps < 1e-5, torch.zeros_like(maps), maps)
    patch_np = maps.detach().to("cpu").numpy().astype(np.float32)

    heatmaps = np.stack(
        [_upsample_bilinear(patch_np[i], tuple(int(v) for v in output_size), device=device)
         for i in range(batch)],
        axis=0,
    )
    if not np.isfinite(patch_np).all() or not np.isfinite(heatmaps).all():
        raise ScoringError("距离图/热图出现 NaN 或 Inf")
    return SpatialScoreResult(patch_distances=patch_np, heatmaps=heatmaps.astype(np.float32))
