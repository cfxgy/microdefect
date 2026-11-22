#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特征缓存（P2）。

缓存键 = SHA256( schema | 图像 MD5 | backbone | 权重标识 | 预处理版本 | 特征版本 )

* 键只依赖**图像内容**（MD5）与模型/预处理版本，与文件名、路径无关：
  同一张图改名后仍命中同一份缓存。
* 模型权重标识、预处理版本、特征版本任一变化 ⇒ 键变化 ⇒ 旧缓存不会被误用。
* 写入采用"同目录临时文件 + ``os.replace``"原子替换，中途失败不会留下可被误读
  的完整缓存文件。
* 读取时校验 schema、键、形状、dtype 与数值有效性；损坏时计数并返回 ``None``
  （``strict=True`` 时抛 :class:`CacheCorruptedError`），绝不静默返回错误特征。
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.features.extractor import (
    BACKBONE_NAME,
    EMBED_DIM,
    FEATURE_VERSION,
    PREPROCESS_VERSION,
    WEIGHTS_TAG,
)

CACHE_SCHEMA = "microdefect.feature_cache/1"
KEY_LENGTH = 32


class CacheCorruptedError(RuntimeError):
    """缓存文件存在但内容不可用。"""


def md5_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheKeyParts:
    """构成缓存键的全部要素。"""

    image_md5: str
    backbone: str = BACKBONE_NAME
    weights_tag: str = WEIGHTS_TAG
    preprocess_version: str = PREPROCESS_VERSION
    feature_version: str = FEATURE_VERSION

    def material(self) -> str:
        return "|".join(
            [
                CACHE_SCHEMA,
                self.image_md5.lower(),
                self.backbone,
                self.weights_tag,
                self.preprocess_version,
                self.feature_version,
            ]
        )

    def key(self) -> str:
        return hashlib.sha256(self.material().encode("utf-8")).hexdigest()[:KEY_LENGTH]

    def as_dict(self) -> dict:
        return {
            "image_md5": self.image_md5,
            "backbone": self.backbone,
            "weights_tag": self.weights_tag,
            "preprocess_version": self.preprocess_version,
            "feature_version": self.feature_version,
            "schema": CACHE_SCHEMA,
            "key": self.key(),
        }


def compute_key(
    image_md5: str,
    *,
    backbone: str = BACKBONE_NAME,
    weights_tag: str = WEIGHTS_TAG,
    preprocess_version: str = PREPROCESS_VERSION,
    feature_version: str = FEATURE_VERSION,
) -> str:
    return CacheKeyParts(
        image_md5=image_md5,
        backbone=backbone,
        weights_tag=weights_tag,
        preprocess_version=preprocess_version,
        feature_version=feature_version,
    ).key()


class FeatureCache:
    """基于内容哈希的 512 维特征缓存。"""

    def __init__(
        self,
        root: str | Path,
        *,
        backbone: str = BACKBONE_NAME,
        weights_tag: str = WEIGHTS_TAG,
        preprocess_version: str = PREPROCESS_VERSION,
        feature_version: str = FEATURE_VERSION,
        embed_dim: int = EMBED_DIM,
    ) -> None:
        self.root = Path(root)
        self.backbone = backbone
        self.weights_tag = weights_tag
        self.preprocess_version = preprocess_version
        self.feature_version = feature_version
        self.embed_dim = int(embed_dim)
        self.stats = {"hits": 0, "misses": 0, "corrupted": 0, "writes": 0}
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- 键与路径 ----
    def key_for_md5(self, image_md5: str) -> str:
        return compute_key(
            image_md5,
            backbone=self.backbone,
            weights_tag=self.weights_tag,
            preprocess_version=self.preprocess_version,
            feature_version=self.feature_version,
        )

    def path_for(self, key: str) -> Path:
        if len(key) < 4:
            raise ValueError(f"缓存键长度异常：{key!r}")
        return self.root / key[:2] / key[2:4] / f"{key}.npz"

    def contains(self, key: str) -> bool:
        return self.path_for(key).is_file()

    # ---- 读 ----
    def get(self, key: str, *, strict: bool = False) -> np.ndarray | None:
        path = self.path_for(key)
        if not path.is_file():
            self.stats["misses"] += 1
            return None

        try:
            # 显式打开文件句柄：np.load(path) 在 zip 解析失败时可能不释放内部句柄
            with path.open("rb") as handle, np.load(handle, allow_pickle=False) as payload:
                if "feature" not in payload:
                    raise CacheCorruptedError(f"缓存缺少 feature 字段：{path}")
                stored_schema = str(payload["schema"]) if "schema" in payload else ""
                if stored_schema != CACHE_SCHEMA:
                    raise CacheCorruptedError(
                        f"缓存 schema 不匹配：{stored_schema!r} != {CACHE_SCHEMA!r}"
                    )
                stored_key = str(payload["key"]) if "key" in payload else ""
                if stored_key and stored_key != key:
                    raise CacheCorruptedError(
                        f"缓存键自校验失败：{stored_key!r} != {key!r}"
                    )
                feature = np.asarray(payload["feature"], dtype=np.float32)
            self._validate_vector(feature, path)
        except CacheCorruptedError:
            self.stats["corrupted"] += 1
            if strict:
                raise
            return None
        except Exception as exc:  # 文件截断、zip 损坏等
            self.stats["corrupted"] += 1
            if strict:
                raise CacheCorruptedError(f"缓存读取失败：{path}（{type(exc).__name__}: {exc}）") from exc
            return None

        self.stats["hits"] += 1
        return feature

    # ---- 写 ----
    def put(self, key: str, feature: np.ndarray) -> Path:
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        self._validate_vector(vector, None)

        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.stem}.tmp-{uuid.uuid4().hex}.npz")
        try:
            with tmp_path.open("wb") as handle:
                np.savez(
                    handle,
                    feature=vector,
                    key=key,
                    schema=CACHE_SCHEMA,
                    backbone=self.backbone,
                    weights_tag=self.weights_tag,
                    preprocess_version=self.preprocess_version,
                    feature_version=self.feature_version,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)  # 原子替换
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        self.stats["writes"] += 1
        return path

    # ---- 失效与清理 ----
    def invalidate(self, key: str) -> bool:
        path = self.path_for(key)
        if path.is_file():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """删除缓存根目录下的全部 ``.npz`` 与临时文件，返回删除数量。"""
        removed = 0
        for path in self.root.rglob("*"):
            if path.is_file() and (path.suffix == ".npz" or ".tmp-" in path.name):
                path.unlink()
                removed += 1
        return removed

    # ---- 统计 ----
    def reset_stats(self) -> None:
        for name in self.stats:
            self.stats[name] = 0

    def summary(self) -> dict:
        files = [p for p in self.root.rglob("*.npz")] if self.root.exists() else []
        return {
            "root": str(self.root),
            "schema": CACHE_SCHEMA,
            "cached_files": len(files),
            "stats": dict(self.stats),
            "versions": {
                "backbone": self.backbone,
                "weights_tag": self.weights_tag,
                "preprocess_version": self.preprocess_version,
                "feature_version": self.feature_version,
            },
        }

    # ---- 内部 ----
    def _validate_vector(self, vector: np.ndarray, path: Path | None) -> None:
        where = f"（{path}）" if path else ""
        if vector.ndim != 1 or vector.shape[0] != self.embed_dim:
            raise CacheCorruptedError(
                f"特征形状异常{where}：{tuple(vector.shape)}，期望 ({self.embed_dim},)"
            )
        if not np.isfinite(vector).all():
            raise CacheCorruptedError(f"特征含 NaN/Inf{where}")
