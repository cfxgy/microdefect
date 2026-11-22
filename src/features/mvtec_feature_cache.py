#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MVTec AD 双路径特征缓存（P5-2.2 新增，不修改 P2 的 ``cache.py``）。

与 P2 缓存的区别
----------------
P2 ``FeatureCache`` 的键只覆盖 512 维 GAP 特征（``FEATURE_VERSION``）。本模块
新增 **特征类型/层/形状/dtype** 维度，使全局与空间特征使用不同键，绝不碰撞。

键材料（全部参与 SHA256）
------------------------
``schema | image_md5 | weights_sha256 | backbone | preprocess_version |
kind | layer | shape | dtype | feature_version``

特性
----
* 原子写入：同目录临时文件 + ``os.replace``；
* 读取校验：schema、键、元数据、shape、dtype、有限性；
* 损坏检测：任何解析/校验失败都视为损坏，``strict=True`` 抛异常，否则返回 None；
* 写入前检查 NaN/Inf 与 dtype/shape，拒绝写入非法特征。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.features import extractor as p2
from src.features.mvtec_spatial_extractor import (
    SPATIAL_CHANNELS,
    SPATIAL_FEATURE_VERSION,
    SPATIAL_GRID,
    SPATIAL_LAYER,
)

CACHE_SCHEMA = "microdefect.mvtec_feature_cache/1"
KEY_LENGTH = 32


class FeatureCacheError(RuntimeError):
    """缓存使用错误或损坏。"""


class CacheCorruptedError(FeatureCacheError):
    """缓存文件存在但不可用。"""


@dataclass(frozen=True)
class FeatureSpec:
    """一类特征的完整标识。"""

    kind: str
    layer: str
    feature_version: str
    shape: tuple[int, ...]
    dtype: str = "float32"


GLOBAL_SPEC = FeatureSpec(
    kind="global",
    layer="gap",
    feature_version=p2.FEATURE_VERSION,
    shape=(p2.EMBED_DIM,),
)

SPATIAL_SPEC = FeatureSpec(
    kind="spatial",
    layer=SPATIAL_LAYER,
    feature_version=SPATIAL_FEATURE_VERSION,
    shape=(SPATIAL_CHANNELS, SPATIAL_GRID, SPATIAL_GRID),
)


def image_md5(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_material(
    image_md5_hex: str,
    *,
    weights_sha256: str,
    spec: FeatureSpec,
    backbone: str = p2.BACKBONE_NAME,
    preprocess_version: str = p2.PREPROCESS_VERSION,
) -> str:
    shape = "x".join(str(v) for v in spec.shape)
    return "|".join(
        [
            CACHE_SCHEMA,
            image_md5_hex.lower(),
            weights_sha256.lower(),
            backbone,
            preprocess_version,
            spec.kind,
            spec.layer,
            shape,
            spec.dtype,
            spec.feature_version,
        ]
    )


def compute_key(
    image_md5_hex: str,
    *,
    weights_sha256: str,
    spec: FeatureSpec,
    backbone: str = p2.BACKBONE_NAME,
    preprocess_version: str = p2.PREPROCESS_VERSION,
) -> str:
    material = key_material(
        image_md5_hex,
        weights_sha256=weights_sha256,
        spec=spec,
        backbone=backbone,
        preprocess_version=preprocess_version,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:KEY_LENGTH]


class MvtecFeatureCache:
    """双路径特征缓存。"""

    def __init__(
        self,
        root: str | Path,
        spec: FeatureSpec,
        *,
        weights_sha256: str,
        backbone: str = p2.BACKBONE_NAME,
        preprocess_version: str = p2.PREPROCESS_VERSION,
    ) -> None:
        self.root = Path(root)
        self.spec = spec
        self.weights_sha256 = weights_sha256
        self.backbone = backbone
        self.preprocess_version = preprocess_version
        self.stats = {"hits": 0, "misses": 0, "corrupted": 0, "writes": 0}
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- 键与路径 ----
    def key_for_md5(self, image_md5_hex: str) -> str:
        return compute_key(
            image_md5_hex,
            weights_sha256=self.weights_sha256,
            spec=self.spec,
            backbone=self.backbone,
            preprocess_version=self.preprocess_version,
        )

    def key_for_image(self, path: str | Path) -> str:
        return self.key_for_md5(image_md5(path))

    def path_for(self, key: str) -> Path:
        if len(key) < 4:
            raise FeatureCacheError(f"缓存键长度异常：{key!r}")
        return self.root / key[:2] / key[2:4] / f"{key}.npz"

    def contains(self, key: str) -> bool:
        return self.path_for(key).is_file()

    # ---- 写 ----
    def put(self, key: str, array: np.ndarray, *, image_md5_hex: str) -> Path:
        array = np.asarray(array)
        expected_shape = (1,) + tuple(self.spec.shape)
        if array.shape == expected_shape:
            array = array[0]
        if tuple(array.shape) != tuple(self.spec.shape):
            raise FeatureCacheError(
                f"特征形状不符：{tuple(array.shape)} != {self.spec.shape}"
            )
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        if not np.isfinite(array).all():
            raise FeatureCacheError("拒绝写入：特征包含 NaN 或 Inf")
        if self.key_for_md5(image_md5_hex) != key:
            raise FeatureCacheError("缓存键与键材料不一致")

        meta = {
            "schema": CACHE_SCHEMA,
            "key": key,
            "image_md5": image_md5_hex.lower(),
            "weights_sha256": self.weights_sha256,
            "backbone": self.backbone,
            "preprocess_version": self.preprocess_version,
            "kind": self.spec.kind,
            "layer": self.spec.layer,
            "feature_version": self.spec.feature_version,
            "shape": list(self.spec.shape),
            "dtype": self.spec.dtype,
            "stored_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            with tmp.open("wb") as handle:
                np.savez(handle, feature=array, meta=np.array(json.dumps(meta, ensure_ascii=False)))
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        self.stats["writes"] += 1
        return path

    # ---- 读 ----
    def get(self, key: str, *, strict: bool = False) -> np.ndarray | None:
        path = self.path_for(key)
        if not path.is_file():
            self.stats["misses"] += 1
            return None
        try:
            with path.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
                if "feature" not in data or "meta" not in data:
                    raise CacheCorruptedError("缓存缺少 feature/meta 字段")
                meta = json.loads(str(data["meta"].item()))
                array = np.asarray(data["feature"])
            if meta.get("schema") != CACHE_SCHEMA:
                raise CacheCorruptedError(f"schema 不符：{meta.get('schema')}")
            if meta.get("key") != key:
                raise CacheCorruptedError("缓存内键与请求键不一致")
            if meta.get("kind") != self.spec.kind or meta.get("layer") != self.spec.layer:
                raise CacheCorruptedError("特征类型/层与缓存实例不一致")
            if meta.get("feature_version") != self.spec.feature_version:
                raise CacheCorruptedError("特征版本不一致")
            if list(meta.get("shape", [])) != list(self.spec.shape):
                raise CacheCorruptedError("元数据 shape 与实例不一致")
            if tuple(array.shape) != tuple(self.spec.shape):
                raise CacheCorruptedError(f"数组 shape 异常：{tuple(array.shape)}")
            if array.dtype != np.float32:
                raise CacheCorruptedError(f"数组 dtype 异常：{array.dtype}")
            if not np.isfinite(array).all():
                raise CacheCorruptedError("数组包含 NaN 或 Inf")
            md5_value = str(meta.get("image_md5", ""))
            if len(md5_value) != 32 or any(c not in "0123456789abcdef" for c in md5_value):
                raise CacheCorruptedError("image_md5 格式非法")
        except CacheCorruptedError:
            self.stats["corrupted"] += 1
            if strict:
                raise
            return None
        except Exception as exc:  # noqa: BLE001 - 任何解析失败都视为损坏
            self.stats["corrupted"] += 1
            if strict:
                raise CacheCorruptedError(f"缓存损坏：{type(exc).__name__}: {exc}") from exc
            return None
        self.stats["hits"] += 1
        return array


def make_global_cache(root: str | Path, *, weights_sha256: str) -> MvtecFeatureCache:
    return MvtecFeatureCache(root, GLOBAL_SPEC, weights_sha256=weights_sha256)


def make_spatial_cache(root: str | Path, *, weights_sha256: str) -> MvtecFeatureCache:
    return MvtecFeatureCache(root, SPATIAL_SPEC, weights_sha256=weights_sha256)
