#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冻结的 ImageNet 预训练 ResNet18 特征提取器（P2）。

本模块只做**特征提取**：不训练、不微调、不修改预训练权重、不做分类。

关键约束
--------
* backbone：``torchvision.models.resnet18``
* 权重：``ResNet18_Weights.IMAGENET1K_V1``（官方枚举，缺失时默认**不下载**，
  而是抛出 :class:`WeightsMissingError`，由调用方决定是否申请下载）
* 去掉原始 fc 分类层，保留卷积 + 全局平均池化 ⇒ 输出 512 维
* ``eval()`` + 全部参数 ``requires_grad=False`` + ``torch.inference_mode()``
* 预处理确定性：灰度 → 复制成 3 通道 RGB → 224×224 双线性 → ToTensor
  → ImageNet 均值/标准差归一化；无任何随机增强

预处理与特征版本号参与缓存键，改动其中任何一个都会让旧缓存失效。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

# ---------------------------------------------------------------------------
# 版本与常量（参与缓存键，改动必须升版本号）
# ---------------------------------------------------------------------------

BACKBONE_NAME = "resnet18"
WEIGHTS_TAG = "IMAGENET1K_V1"
PREPROCESS_VERSION = "p1/gray2rgb-resize224-bilinear-antialias-imagenetnorm"
# 特征版本号：TF32 关闭（见 configure_backend_precision），故与 v1 不兼容
FEATURE_VERSION = "f1/resnet18-gap512-raw-fp32-notf32"
EMBED_DIM = 512

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class WeightsMissingError(RuntimeError):
    """预训练权重不在本地缓存，且未允许下载。"""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_TORCH_HOME = PROJECT_ROOT / "cache" / "torch"


def _dir_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def default_torch_home() -> Path:
    """当前生效的 torch 缓存根目录（不触发下载）。"""
    return Path(os.environ.get("TORCH_HOME") or (Path.home() / ".cache" / "torch"))


_WEIGHTS_HOME_ENSURED = False


def _weights_file_at(home: Path) -> Path:
    filename = os.path.basename(urlparse(ResNet18_Weights.IMAGENET1K_V1.url).path)
    return home / "hub" / "checkpoints" / filename


def ensure_torch_home() -> Path:
    """确保 torch 缓存目录可写；不可写时回退到项目内 ``cache/torch``。

    受限环境下用户目录常不可写。若默认目录已有权重文件则保持默认；否则在
    本进程内把 ``TORCH_HOME`` 指向项目内目录（只打印一次提示）。
    """
    global _WEIGHTS_HOME_ENSURED
    if _WEIGHTS_HOME_ENSURED:
        return default_torch_home()
    _WEIGHTS_HOME_ENSURED = True

    home = default_torch_home()
    existing = _weights_file_at(home)
    try:
        if existing.is_file() and existing.stat().st_size >= 1_000_000:
            return home
    except OSError:
        pass

    if _dir_writable(home / "hub" / "checkpoints"):
        return home

    fallback = PROJECT_TORCH_HOME
    if _dir_writable(fallback / "hub" / "checkpoints"):
        os.environ["TORCH_HOME"] = str(fallback)
        print(
            f"[extractor] 默认 torch 缓存不可写（{home / 'hub' / 'checkpoints'}），"
            f"本次运行改用项目内缓存：{fallback}"
        )
        return fallback
    return home


def torch_hub_checkpoints_dir() -> Path:
    """torchvision 预训练权重的默认缓存目录。"""
    return Path(torch.hub.get_dir()) / "checkpoints"


def expected_weights_path() -> Path:
    """按官方 URL 的文件名推断本地权重文件路径。"""
    filename = os.path.basename(urlparse(ResNet18_Weights.IMAGENET1K_V1.url).path)
    return torch_hub_checkpoints_dir() / filename


def weights_cached(min_bytes: int = 1_000_000) -> bool:
    """权重是否已缓存且大小合理（不做哈希校验，仅用于快速判断）。"""
    ensure_torch_home()
    path = expected_weights_path()
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def weights_status() -> dict:
    """返回权重缓存状态，供报告与冒烟脚本使用。"""
    ensure_torch_home()
    path = expected_weights_path()
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    return {
        "backbone": BACKBONE_NAME,
        "weights_enum": f"ResNet18_Weights.{WEIGHTS_TAG}",
        "url": ResNet18_Weights.IMAGENET1K_V1.url,
        "expected_path": str(path),
        "exists": exists,
        "size_bytes": size,
        "cached": weights_cached(),
        "torch_home": os.environ.get("TORCH_HOME"),
    }


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------


def configure_backend_precision() -> dict:
    """固定推理数值精度：关闭 TF32，启用 cudnn 确定性。

    实测（RTX 4060 Laptop）：开启 TF32 时 CPU 与 GPU 特征最大绝对差约
    2.4e-3（余弦相似度 0.99999982）；关闭后降到约 3.8e-6。由于特征缓存键
    与设备无关，必须让不同设备的特征足够一致，因此这里统一关闭 TF32。
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def build_model(
    device: str | torch.device | None = None,
    *,
    allow_download: bool = False,
) -> tuple[nn.Module, torch.device]:
    """构建冻结的 ResNet18 特征提取器。

    Parameters
    ----------
    device:
        ``None``/``"auto"`` 自动选择 CUDA，否则 CPU；也可显式 ``"cpu"``/``"cuda"``。
    allow_download:
        仅当为 True 且权重缺失时才允许 torchvision 下载官方权重。
        默认 False：权重缺失时抛 :class:`WeightsMissingError`。
    """
    if not weights_cached() and not allow_download:
        status = weights_status()
        raise WeightsMissingError(
            "预训练权重不在本地缓存，且未允许下载。\n"
            f"  期望路径 : {status['expected_path']}\n"
            f"  官方 URL : {status['url']}\n"
            "  处理方式 : 由用户批准后使用 allow_download=True 或 --allow-download 下载一次。"
        )

    configure_backend_precision()
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # 去掉原始全连接分类层：avgpool 之后直接输出 512 维
    model.fc = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
        param.grad = None

    resolved = resolve_device(device)
    model = model.to(resolved)
    return model, resolved


def resolve_device(spec: str | torch.device | None) -> torch.device:
    """解析目标设备，CUDA 不可用时可回退 CPU。"""
    if spec is None or (isinstance(spec, str) and spec.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")
    return device


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------


def load_image_rgb(path: str | Path) -> Image.Image:
    """读取图像并转换为 3 通道 RGB。

    NEU-CLS 为 8 位灰度 BMP：先统一转 ``L``，再**复制同一通道三次**得到 RGB。
    这样三个通道内容完全一致，而不是把灰度值误当成不同颜色通道。
    """
    with Image.open(path) as handle:
        gray = handle.convert("L")
        rgb = Image.merge("RGB", (gray, gray, gray))
    return rgb


def build_transform() -> transforms.Compose:
    """确定性预处理流水线（无任何随机增强）。"""
    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_pil(image: Image.Image, transform: transforms.Compose | None = None) -> torch.Tensor:
    """PIL 图像 → ``[3, 224, 224]`` float32 张量。"""
    tf = transform or build_transform()
    if image.mode != "RGB":
        gray = image.convert("L")
        image = Image.merge("RGB", (gray, gray, gray))
    return tf(image)


def preprocess_path(path: str | Path, transform: transforms.Compose | None = None) -> torch.Tensor:
    return preprocess_pil(load_image_rgb(path), transform)


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行做 L2 归一化，不修改输入数组。"""
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)


# ---------------------------------------------------------------------------
# 提取器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractorInfo:
    backbone: str
    weights_tag: str
    preprocess_version: str
    feature_version: str
    embed_dim: int
    device: str
    batch_size: int


class FeatureExtractor:
    """冻结 ResNet18 的批量特征提取器。"""

    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        allow_download: bool = False,
        batch_size: int = 32,
        model: nn.Module | None = None,
    ) -> None:
        configure_backend_precision()
        if model is None:
            model, resolved = build_model(device, allow_download=allow_download)
        else:
            resolved = resolve_device(device)
            model = model.to(resolved)
        self.model = model
        self.device = resolved
        self.batch_size = int(batch_size)
        self.transform = build_transform()

    # ---- 元信息 ----
    @property
    def info(self) -> ExtractorInfo:
        return ExtractorInfo(
            backbone=BACKBONE_NAME,
            weights_tag=WEIGHTS_TAG,
            preprocess_version=PREPROCESS_VERSION,
            feature_version=FEATURE_VERSION,
            embed_dim=EMBED_DIM,
            device=str(self.device),
            batch_size=self.batch_size,
        )

    def describe(self) -> dict:
        info = self.info
        return {
            "backbone": info.backbone,
            "weights": f"ResNet18_Weights.{info.weights_tag}",
            "weights_path": str(expected_weights_path()),
            "preprocess_version": info.preprocess_version,
            "feature_version": info.feature_version,
            "embed_dim": info.embed_dim,
            "device": info.device,
            "batch_size": info.batch_size,
            "image_size": IMAGE_SIZE,
        }

    # ---- 张量推理 ----
    @torch.inference_mode()
    def extract_tensor(self, batch: torch.Tensor) -> np.ndarray:
        """``[B,3,H,W]`` → ``[B,512]`` 原始特征（float32 numpy）。"""
        if batch.ndim != 4:
            raise ValueError(f"期望 4 维张量，实际 {tuple(batch.shape)}")
        batch = batch.to(self.device, non_blocking=False)
        features = self.model(batch)
        if features.ndim != 2 or features.shape[1] != EMBED_DIM:
            raise RuntimeError(f"特征维度异常：{tuple(features.shape)}")
        array = features.detach().to("cpu").float().numpy()
        if not np.isfinite(array).all():
            raise RuntimeError("特征中出现 NaN 或 Inf")
        return array

    # ---- 路径批量推理 ----
    def extract_paths(
        self,
        paths: list[str | Path],
        *,
        batch_size: int | None = None,
        normalize: bool = False,
    ) -> np.ndarray:
        if not paths:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        size = int(batch_size or self.batch_size)
        outputs: list[np.ndarray] = []
        for start in range(0, len(paths), size):
            chunk = paths[start : start + size]
            tensors = [preprocess_path(p, self.transform) for p in chunk]
            batch = torch.stack(tensors, dim=0)
            outputs.append(self.extract_tensor(batch))
            del tensors, batch
        result = np.concatenate(outputs, axis=0)
        return l2_normalize(result) if normalize else result

    # ---- 单张推理 ----
    def extract_image(self, image: Image.Image, *, normalize: bool = False) -> np.ndarray:
        tensor = preprocess_pil(image, self.transform).unsqueeze(0)
        array = self.extract_tensor(tensor)[0]
        return l2_normalize(array) if normalize else array

    def extract_path(self, path: str | Path, *, normalize: bool = False) -> np.ndarray:
        return self.extract_image(load_image_rgb(path), normalize=normalize)

    # ---- 显存信息 ----
    def gpu_memory(self) -> dict:
        if self.device.type != "cuda":
            return {"device": "cpu", "allocated_mib": None, "max_allocated_mib": None}
        return {
            "device": torch.cuda.get_device_name(self.device),
            "allocated_mib": round(torch.cuda.memory_allocated(self.device) / 1024 ** 2, 2),
            "reserved_mib": round(torch.cuda.memory_reserved(self.device) / 1024 ** 2, 2),
            "max_allocated_mib": round(torch.cuda.max_memory_allocated(self.device) / 1024 ** 2, 2),
        }
