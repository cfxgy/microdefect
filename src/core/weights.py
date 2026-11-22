#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地预训练权重解析（软件运行层）。

背景
----
``src/features/extractor.py``（P2 冻结实现）在默认缓存目录**可写但为空**时
会继续使用默认目录，导致 ``WeightsMissingError``——即使项目内
``cache/torch/hub/checkpoints`` 已存在合法权重。是否重定向取决于目录可写性，
因此在"用户目录可写"的真实运行环境里会误报权重缺失。

本模块在构造检测器之前，按以下优先级选择**确实存在权重**的缓存根目录：

1. 显式 ``TORCH_HOME``（若其中已有合法权重）；
2. 项目内 ``cache/torch``；
3. 用户默认 ``~/.cache/torch``；
4. 当前 ``torch.hub`` 目录。

找到后同时设置 ``TORCH_HOME`` 与 ``torch.hub.set_dir``，确保 P2 的
``expected_weights_path()`` 与实际加载路径一致。**不下载任何文件**。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_TORCH_HOME = PROJECT_ROOT / "cache" / "torch"
DEFAULT_TORCH_HOME = Path.home() / ".cache" / "torch"

WEIGHT_FILENAME = "resnet18-f37072fd.pth"
MIN_WEIGHT_BYTES = 1_000_000


class WeightsUnavailableError(RuntimeError):
    """本地找不到任何可用的预训练权重（不会自动下载）。"""


def weights_file_in(home: Path) -> Path:
    return Path(home) / "hub" / "checkpoints" / WEIGHT_FILENAME


def _has_weights(home: Path) -> bool:
    try:
        path = weights_file_in(home)
        return path.is_file() and path.stat().st_size >= MIN_WEIGHT_BYTES
    except OSError:
        return False


def candidate_homes() -> list[Path]:
    homes: list[Path] = []
    explicit = os.environ.get("TORCH_HOME")
    if explicit:
        homes.append(Path(explicit))
    homes.extend([PROJECT_TORCH_HOME, DEFAULT_TORCH_HOME])
    try:
        homes.append(Path(torch.hub.get_dir()).parent)
    except Exception:  # noqa: BLE001
        pass
    unique: list[Path] = []
    seen: set[str] = set()
    for home in homes:
        key = os.path.normcase(os.path.abspath(str(home)))
        if key not in seen:
            seen.add(key)
            unique.append(home)
    return unique


def find_local_weights() -> Path | None:
    for home in candidate_homes():
        if _has_weights(home):
            return weights_file_in(home)
    return None


def ensure_local_weights() -> Path:
    """选择本地权重并同步 torch 缓存路径；找不到时给出可操作错误。"""
    found = find_local_weights()
    if found is None:
        searched = "\n".join(f"  - {weights_file_in(home)}" for home in candidate_homes())
        raise WeightsUnavailableError(
            "本地未找到 ResNet18 预训练权重，且本程序不会自动联网下载。\n"
            f"已查找路径：\n{searched}\n"
            "处理方式：\n"
            f"  1) 将官方权重 {WEIGHT_FILENAME} 放入 "
            f"{weights_file_in(PROJECT_TORCH_HOME)}；\n"
            "  2) 或由管理员批准后，使用项目脚本的 --allow-download 选项下载一次；\n"
            "  3) 不要使用随机初始化权重代替预训练权重。"
        )
    home = found.parents[2]  # <home>/hub/checkpoints/<file>
    os.environ["TORCH_HOME"] = str(home)
    try:
        torch.hub.set_dir(str(home / "hub"))
    except Exception:  # noqa: BLE001
        pass
    return found
