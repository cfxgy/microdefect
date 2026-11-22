#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MVTec AD 协议 v1.0 余弦异常评分（P5-2.3.2.1）。

严格对齐协议 v1.0：

* 图像级：``s(x) = 1 − max_m cos(x, m)``，m 遍历 ``global_full`` 全部 512 维特征；
* 像素级：对每个 patch 先做 **L2 归一化**，再计算
  ``d = min_m (1 − cos(query_patch, memory_patch))``；
* 热图：双线性上采样，``align_corners=False``，输出原图尺寸；
* 不做 PCA / 标准化 / 距离归零，不使用欧氏距离。

零范数向量、NaN/Inf、空记忆库、形状错误均显式报错，不静默出分。

本模块与 P5-2.3.2 的欧氏实现（``mvtec_scoring.py``）并存，互不覆盖。
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

SCORING_VERSION = "protocol-v1.0-cosine"
DISTANCE_DEFINITION = "1 - cosine_similarity"


class ProtocolScoringError(RuntimeError):
    """协议评分输入不合法。"""


@dataclass(frozen=True)
class ImageCosineResult:
    distances: np.ndarray  # (B,) float32, 1 - max cosine
    nearest_indices: np.ndarray  # (B,) int64
    max_cosine: np.ndarray  # (B,) float32


@dataclass(frozen=True)
class SpatialCosineResult:
    patch_distances: np.ndarray  # (B,14,14) float32
    heatmaps: np.ndarray  # (B,H,W) float32
    patch_max_cosine: np.ndarray  # (B,14,14) float32


def _tensor(array, *, device: str, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(array) if not torch.is_tensor(array) else array,
                             dtype=torch.float64, device=device)
    if not torch.isfinite(tensor).all():
        raise ProtocolScoringError(f"{name} 含 NaN 或 Inf")
    return tensor


def _check_nonzero(tensor: torch.Tensor, *, name: str, dim: int) -> None:
    norms = torch.linalg.vector_norm(tensor, dim=dim)
    if torch.any(norms == 0):
        count = int((norms == 0).sum().item())
        raise ProtocolScoringError(f"{name} 含 {count} 个零范数向量")


def _normalize(tensor: torch.Tensor, *, name: str, dim: int) -> torch.Tensor:
    _check_nonzero(tensor, name=name, dim=dim)
    return tensor / torch.linalg.vector_norm(tensor, dim=dim, keepdim=True)


def score_image_cosine(
    global_features: np.ndarray,
    global_memory: np.ndarray,
    *,
    device: str = "cpu",
    memory_chunk: int = 8192,
) -> ImageCosineResult:
    """``[D]``/``[B,D]`` 与 ``[N,D]`` 的 ``1 − max cos``。"""
    features = _tensor(global_features, device=device, name="global_features")
    memory = _tensor(global_memory, device=device, name="global_memory")
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim != 2 or features.shape[1] != GLOBAL_DIM:
        raise ProtocolScoringError(f"全局特征形状异常：{tuple(features.shape)}")
    if memory.ndim != 2 or memory.shape[1] != GLOBAL_DIM:
        raise ProtocolScoringError(f"全局记忆库形状异常：{tuple(memory.shape)}")
    if memory.shape[0] == 0:
        raise ProtocolScoringError("全局记忆库为空")

    query = _normalize(features, name="global_features", dim=1)
    memory_norm = _normalize(memory, name="global_memory", dim=1)

    best = torch.full((query.shape[0],), -2.0, dtype=torch.float64, device=device)
    best_index = torch.zeros((query.shape[0],), dtype=torch.long, device=device)
    for start in range(0, memory_norm.shape[0], memory_chunk):
        block = memory_norm[start:start + memory_chunk]
        sim = query @ block.T
        block_max, block_arg = sim.max(dim=1)
        update = block_max > best
        best = torch.where(update, block_max, best)
        best_index = torch.where(update, block_arg + start, best_index)

    best = torch.clamp(best, min=-1.0, max=1.0)
    distances = torch.clamp(1.0 - best, min=0.0, max=2.0)
    return ImageCosineResult(
        distances=distances.detach().to("cpu").numpy().astype(np.float32),
        nearest_indices=best_index.detach().to("cpu").numpy().astype(np.int64),
        max_cosine=best.detach().to("cpu").numpy().astype(np.float32),
    )


def _as_patch_matrix(spatial_memory: np.ndarray, *, device: str) -> torch.Tensor:
    memory = _tensor(spatial_memory, device=device, name="spatial_memory")
    if memory.ndim == 4:
        if memory.shape[1:] != (SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID):
            raise ProtocolScoringError(f"空间记忆库形状异常：{tuple(memory.shape)}")
        memory = memory.permute(0, 2, 3, 1).reshape(-1, SPATIAL_CHANNELS)
    if memory.ndim != 2 or memory.shape[1] != SPATIAL_CHANNELS:
        raise ProtocolScoringError(f"空间记忆库形状异常：{tuple(memory.shape)}")
    if memory.shape[0] == 0:
        raise ProtocolScoringError("空间记忆库为空")
    return _normalize(memory, name="spatial_memory", dim=1)


def _upsample(distance_map: np.ndarray, output_size: tuple[int, int], *, device: str) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(distance_map, dtype=np.float32), device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(tensor, size=tuple(output_size), mode="bilinear", align_corners=False)
    return upsampled.detach().to("cpu").numpy().astype(np.float32)[0, 0]


def score_spatial_cosine(
    spatial_features: np.ndarray,
    spatial_memory: np.ndarray,
    output_size: tuple[int, int],
    *,
    device: str = "cpu",
    memory_chunk: int = 4096,
) -> SpatialCosineResult:
    """``[256,14,14]``/``[B,256,14,14]`` 的逐 patch 余弦距离与热图。"""
    memory = _as_patch_matrix(spatial_memory, device=device)
    features = _tensor(spatial_features, device=device, name="spatial_features")
    if features.ndim == 3:
        features = features.unsqueeze(0)
    if features.ndim != 4 or features.shape[1:] != (SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID):
        raise ProtocolScoringError(
            f"空间特征形状异常：{tuple(features.shape)}，期望 [B,{SPATIAL_CHANNELS},"
            f"{SPATIAL_GRID},{SPATIAL_GRID}]"
        )
    if len(output_size) != 2 or int(output_size[0]) <= 0 or int(output_size[1]) <= 0:
        raise ProtocolScoringError(f"输出尺寸非法：{output_size}")

    batch = features.shape[0]
    patches = features.reshape(batch, SPATIAL_CHANNELS, PATCH_COUNT).permute(0, 2, 1).contiguous()
    patches = _normalize(patches, name="spatial_features", dim=2)

    maps = torch.empty((batch, PATCH_COUNT), dtype=torch.float64, device=device)
    cos_maps = torch.empty((batch, PATCH_COUNT), dtype=torch.float64, device=device)
    for index in range(batch):
        query = patches[index]
        best = torch.full((PATCH_COUNT,), -2.0, dtype=torch.float64, device=device)
        for start in range(0, memory.shape[0], memory_chunk):
            block = memory[start:start + memory_chunk]
            sim = query @ block.T
            best = torch.maximum(best, sim.max(dim=1).values)
        best = torch.clamp(best, min=-1.0, max=1.0)
        cos_maps[index] = best
        maps[index] = torch.clamp(1.0 - best, min=0.0, max=2.0)

    maps = maps.reshape(batch, SPATIAL_GRID, SPATIAL_GRID)
    cos_maps = cos_maps.reshape(batch, SPATIAL_GRID, SPATIAL_GRID)
    patch_np = maps.detach().to("cpu").numpy().astype(np.float32)
    cos_np = cos_maps.detach().to("cpu").numpy().astype(np.float32)

    heatmaps = np.stack(
        [_upsample(patch_np[i], tuple(int(v) for v in output_size), device=device) for i in range(batch)],
        axis=0,
    )
    if not np.isfinite(patch_np).all() or not np.isfinite(heatmaps).all():
        raise ProtocolScoringError("距离图/热图出现 NaN 或 Inf")
    return SpatialCosineResult(
        patch_distances=patch_np,
        heatmaps=heatmaps.astype(np.float32),
        patch_max_cosine=cos_np,
    )


def describe() -> dict:
    return {
        "scoring_version": SCORING_VERSION,
        "image_distance": "1 - max cosine similarity",
        "patch_distance": "1 - max cosine similarity (per-patch L2 normalization)",
        "distance_range": [0, 2],
        "normalization": "image: cosine only; patch: per-patch L2 before cosine",
        "distance_zeroing": False,
        "upsample": {"mode": "bilinear", "align_corners": False},
        "protocol_reference": "stage4_protocol_v1_0_final §3/§5",
    }
