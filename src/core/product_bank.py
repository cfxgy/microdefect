#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""产品独立记忆库写入/校验（P2）。

与 MVTec 实验记忆库严格区分：

* schema：``microdefect.product_bank/1``
* protocol：``product-v1``
* 归属：``product_id`` + ``product_version``
* 只接受 ``global_reference`` / ``spatial_reference`` 两种正常特征库
* 不覆盖已存在目录（``allow_overwrite=False`` 语义）

特征数组仍为 float32：全局 ``(N, 512)``，空间 ``(N, 256, 14, 14)``；
成员表字段与评分/追溯所需信息一致。
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.memory.mvtec_memory import feature_content_sha256, member_manifest_sha256

PRODUCT_BANK_SCHEMA = "microdefect.product_bank/1"
PRODUCT_BANK_PROTOCOL = "product-v1"

GLOBAL_DIM = 512
SPATIAL_SHAPE = (256, 14, 14)


class ProductBankError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProductBankInfo:
    directory: Path
    metadata: dict
    n_members: int


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _validate_features(memory_type: str, features: np.ndarray) -> np.ndarray:
    array = np.asarray(features)
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if array.ndim == 0 or array.shape[0] == 0:
        raise ProductBankError("空记忆库：至少需要一个成员")
    if memory_type == "spatial_reference":
        if array.ndim != 4 or tuple(array.shape[1:]) != SPATIAL_SHAPE:
            raise ProductBankError(f"空间记忆形状异常：{array.shape}")
    elif memory_type == "global_reference":
        if array.ndim != 2 or array.shape[1] != GLOBAL_DIM:
            raise ProductBankError(f"全局记忆形状异常：{array.shape}")
    else:
        raise ProductBankError(f"不支持的产品记忆类型：{memory_type!r}")
    if not np.isfinite(array).all():
        raise ProductBankError("特征包含 NaN 或 Inf")
    return np.ascontiguousarray(array, dtype=np.float32)


def save_product_bank(
    directory: str | Path,
    features: np.ndarray,
    members: list[dict],
    *,
    product_id: str,
    product_version: int,
    memory_type: str,
    extractor_info: dict,
) -> dict:
    """写入一个产品记忆库目录（features.npy / members.csv / metadata.json）。"""
    target = Path(directory)
    if target.exists() and any(target.iterdir()):
        raise ProductBankError(f"目标记忆库目录已存在且非空，拒绝覆盖：{target}")
    array = _validate_features(memory_type, features)
    if len(members) != array.shape[0]:
        raise ProductBankError(f"成员数 {len(members)} 与特征行数 {array.shape[0]} 不一致")
    required = ("rel_path", "image_sha256", "product_id", "role", "order_index")
    for member in members:
        missing = [key for key in required if key not in member]
        if missing:
            raise ProductBankError(f"成员字段缺失：{missing}")
        if member["product_id"] != product_id:
            raise ProductBankError(
                f"成员 product_id 串用：{member['product_id']} != {product_id}"
            )

    metadata = {
        "schema_version": PRODUCT_BANK_SCHEMA,
        "protocol_version": PRODUCT_BANK_PROTOCOL,
        "product_id": product_id,
        "product_version": int(product_version),
        "memory_type": memory_type,
        "backbone": extractor_info["backbone"],
        "weights_tag": extractor_info["weights_tag"],
        "weight_sha256": extractor_info["weight_sha256"],
        "global_feature_version": extractor_info["global_feature_version"],
        "spatial_feature_version": extractor_info["spatial_feature_version"],
        "preprocess_version": extractor_info["preprocess_version"],
        "feature_dtype": "float32",
        "input_size": int(extractor_info["image_size"]),
        "creation_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "member_count": len(members),
        "member_manifest_sha256": member_manifest_sha256(members),
        "feature_content_sha256": feature_content_sha256(array),
        "shape": list(array.shape),
    }

    target.mkdir(parents=True, exist_ok=True)
    feature_path = target / "features.npy"
    tmp = feature_path.with_name(f"features.npy.tmp-{uuid.uuid4().hex}")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, feature_path)

    member_path = target / "members.csv"
    with member_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["order_index", "rel_path", "image_sha256", "product_id",
                      "product_version", "role", "shot"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for member in members:
            writer.writerow({
                "order_index": int(member["order_index"]),
                "rel_path": member["rel_path"],
                "image_sha256": member["image_sha256"],
                "product_id": member["product_id"],
                "product_version": int(product_version),
                "role": member["role"],
                "shot": int(member.get("shot", 5)),
            })
    _atomic_write(target / "metadata.json",
                  json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
    verify_product_bank(target)
    return metadata


def load_product_bank(directory: str | Path) -> dict:
    directory = Path(directory)
    meta_path = directory / "metadata.json"
    if not meta_path.is_file():
        raise ProductBankError(f"缺少 metadata.json：{directory}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != PRODUCT_BANK_SCHEMA:
        raise ProductBankError(f"非产品记忆库 schema：{metadata.get('schema_version')!r}")
    features = np.load(directory / "features.npy", allow_pickle=False)
    with (directory / "members.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        members = list(csv.DictReader(handle))
    return {"metadata": metadata, "features": features, "members": members}


def verify_product_bank(directory: str | Path) -> ProductBankInfo:
    directory = Path(directory)
    payload = load_product_bank(directory)
    metadata, features, members = payload["metadata"], payload["features"], payload["members"]
    problems: list[str] = []
    if metadata.get("protocol_version") != PRODUCT_BANK_PROTOCOL:
        problems.append("protocol_version 不是 product-v1")
    if features.dtype != np.float32:
        problems.append(f"dtype={features.dtype}")
    if tuple(features.shape) != tuple(metadata.get("shape", [])):
        problems.append("shape 与 metadata 不一致")
    if not np.isfinite(features).all():
        problems.append("non-finite")
    if len(members) != int(metadata.get("member_count", -1)):
        problems.append("成员数与 metadata 不一致")
    if len(members) != features.shape[0]:
        problems.append("成员数与特征行数不一致")
    if any(m.get("product_id") != metadata.get("product_id") for m in members):
        problems.append("成员 product_id 与 metadata 不一致")
    if feature_content_sha256(features) != metadata.get("feature_content_sha256"):
        problems.append("feature_content_sha256 不匹配")
    if member_manifest_sha256(members) != metadata.get("member_manifest_sha256"):
        problems.append("member_manifest_sha256 不匹配")
    _validate_features(metadata["memory_type"], features)
    if problems:
        raise ProductBankError(f"产品记忆库校验失败（{directory}）：{problems}")
    return ProductBankInfo(directory=directory, metadata=metadata, n_members=len(members))
