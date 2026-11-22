#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""新产品注册核心流程（P2）。

流程::

    校验输入 → SHA256 去重/隔离 → 冻结特征提取（参考图/校准图/可选缺陷图）
    → 产品记忆库 → 独立校准集计算阈值 → 写入版本 staging（含 manifest）
    → 原子提交 + 激活（同一产品锁内）

严格边界：

* 参考图 ≥5；校准图默认 25、允许 20..200，参考/校准必须完全独立；
* 校准图只用于阈值，不进入记忆库；
* 缺陷样本可选、独立存储、不参与当前评分；
* 独立测试集可选；未提供时允许注册，但不标记为“通过独立性能验收”；
* 不训练、不微调、不下载权重；复用冻结特征提取器与既有余弦评分。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.anomaly.mvtec_protocol_scoring import score_image_cosine, score_spatial_cosine
from src.core.product_bank import (
    PRODUCT_BANK_SCHEMA,
    save_product_bank,
)
from src.core.product_calibration import (
    DEFAULT_CALIBRATION_COUNT,
    IMAGE_QUANTILE,
    PIXEL_QUANTILE,
    ProductCalibrationError,
    calibrate_image_scores,
    calibrate_pixel_scores,
    calibration_sample_note,
    validate_calibration_count,
)
from src.core.products import (
    ProductIntegrityError,
    ProductRecord,
    ProductStore,
    ProductVersion,
    sha256_file,
)
from src.core.weights import ensure_local_weights
from src.features import extractor as p2
from src.features.mvtec_spatial_extractor import MvtecSpatialExtractor
from src.memory.mvtec_memory import feature_content_sha256, member_manifest_sha256

REFERENCE_MIN = 5
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class RegistrationError(ValueError):
    """注册输入/流程错误（不会提交版本）。"""


class DuplicateRegistrationError(RegistrationError):
    """相同注册指纹已存在；不会创建重复产品。"""

    def __init__(self, fingerprint: str, existing_product_id: str, existing_version: int,
                 manifest_sha256: str = "") -> None:
        self.fingerprint = fingerprint
        self.existing_product_id = existing_product_id
        self.existing_version = existing_version
        self.manifest_sha256 = manifest_sha256
        super().__init__(
            f"相同注册内容已存在：product_id={existing_product_id}, "
            f"version={existing_version}, fingerprint={fingerprint[:16]}…"
        )


@dataclass
class RegistrationRequest:
    name: str
    reference_paths: list[Path]
    calibration_paths: list[Path]
    calibration_count: int = DEFAULT_CALIBRATION_COUNT
    category: str = ""
    product_id: str | None = None
    defect_paths: list[Path] = field(default_factory=list)
    test_normal_paths: list[Path] = field(default_factory=list)
    test_defect_paths: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_original_names: dict[str, str] = field(default_factory=dict)
    image_quantile: float = IMAGE_QUANTILE
    pixel_quantile: float = PIXEL_QUANTILE
    activate: bool = True


@dataclass
class RegistrationResult:
    product: ProductRecord
    version: ProductVersion
    image_threshold: float
    pixel_threshold: float
    n_reference: int
    n_calibration: int
    low_sample_warning: bool
    independent_test_provided: bool
    independent_test_validated: bool
    manifest_sha256: str
    warnings: list[str]

    def summary(self) -> dict:
        return {
            "product_id": self.product.product_id,
            "name": self.product.name,
            "status": self.product.status,
            "current_version": self.product.current_version,
            "version": self.version.version,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "n_reference": self.n_reference,
            "n_calibration": self.n_calibration,
            "low_sample_warning": self.low_sample_warning,
            "independent_test_provided": self.independent_test_provided,
            "independent_test_validated": self.independent_test_validated,
            "manifest_sha256": self.manifest_sha256,
            "source_status": self.version.source_status,
            "registration_fingerprint": self.version.registration_fingerprint,
            "warnings": self.warnings,
        }


def collect_images(directory: str | Path, *, role: str) -> list[Path]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise RegistrationError(f"{role}目录不存在：{directory}")
    images = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not images:
        raise RegistrationError(f"{role}目录没有可用图片：{directory}")
    return images


def _check_readable(paths: list[Path], role: str) -> None:
    for path in paths:
        if not path.is_file():
            raise RegistrationError(f"{role}图片不存在：{path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:  # noqa: BLE001
            raise RegistrationError(f"{role}图片损坏：{path}（{exc}）") from exc


def _dedupe_by_sha(labelled: list[tuple[str, Path]]) -> dict[str, str]:
    """返回 sha -> 首次出现角色；发现重复则抛错。"""
    seen: dict[str, tuple[str, Path]] = {}
    for role, path in labelled:
        digest = sha256_file(path)
        if digest in seen:
            other_role, other_path = seen[digest]
            raise RegistrationError(
                f"检测到相同文件内容（SHA256={digest[:16]}…）："
                f"{other_role}={other_path} 与 {role}={path}；"
                "参考图、校准图、缺陷图、测试集必须互相独立。"
            )
        seen[digest] = (role, path)
    return {digest: role for digest, (role, _) in seen.items()}


def _register_request_validation(request: RegistrationRequest) -> tuple[list[tuple[str, Path]], dict]:
    reference = [Path(p).resolve() for p in request.reference_paths]
    calibration = [Path(p).resolve() for p in request.calibration_paths]
    defects = [Path(p).resolve() for p in request.defect_paths]
    test_normal = [Path(p).resolve() for p in request.test_normal_paths]
    test_defect = [Path(p).resolve() for p in request.test_defect_paths]

    if len(reference) < REFERENCE_MIN:
        raise RegistrationError(
            f"参考图不足：需要至少 {REFERENCE_MIN} 张正常参考图，实际 {len(reference)} 张"
        )
    try:
        validate_calibration_count(request.calibration_count)
    except ProductCalibrationError as exc:
        raise RegistrationError(str(exc)) from exc
    if len(calibration) != request.calibration_count:
        raise RegistrationError(
            f"校准图数量与配置不一致：配置 {request.calibration_count} 张，实际 {len(calibration)} 张；"
            "请调整目录内容或 --calibration-count。"
        )

    for paths, role in ((reference, "reference"), (calibration, "calibration"),
                        (defects, "defect"), (test_normal, "test_normal"),
                        (test_defect, "test_defect")):
        _check_readable(paths, role)

    labelled = ([("reference", p) for p in reference]
                + [("calibration", p) for p in calibration]
                + [("defect", p) for p in defects]
                + [("test_normal", p) for p in test_normal]
                + [("test_defect", p) for p in test_defect])
    _dedupe_by_sha(labelled)
    roles = {
        "reference": reference, "calibration": calibration, "defect": defects,
        "test_normal": test_normal, "test_defect": test_defect,
    }
    return labelled, roles


def _extractor_info(extractor: MvtecSpatialExtractor) -> dict:
    info = extractor.info
    return {
        "backbone": info.backbone,
        "weights_tag": info.weights_tag,
        "weight_sha256": info.weights_sha256,
        "global_feature_version": p2.FEATURE_VERSION,
        "spatial_feature_version": info.spatial_feature_version,
        "preprocess_version": info.preprocess_version,
        "image_size": info.image_size,
    }


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _safe_source_filename(original: str, index: int) -> str:
    name = Path(original or "image").name
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem)[:48] or "image"
    return f"{index:03d}_{stem}{suffix}"


def compute_registration_fingerprint(
    *,
    name: str,
    category: str,
    calibration_count: int,
    groups: dict[str, list[str]],
    image_quantile: float,
    pixel_quantile: float,
    feature_extractor_version: str,
    weights_sha256: str,
) -> str:
    """确定性注册指纹（规范化 JSON + SHA256）。

    组内 SHA 排序（注册算法对组内顺序不敏感）；不包含原始文件名，
    因此同一图片仅改名不会改变指纹；角色边界、名称、类别、校准数量、
    分位数、特征版本与权重均纳入。
    """
    canonical = {
        "schema": "microdefect.registration_fingerprint/1",
        "name": _normalize_text(name),
        "category": _normalize_text(category),
        "calibration_count": int(calibration_count),
        "image_quantile": float(image_quantile),
        "pixel_quantile": float(pixel_quantile),
        "feature_extractor_version": str(feature_extractor_version),
        "weights_sha256": str(weights_sha256),
        "groups": {
            role: sorted(str(item) for item in groups.get(role, []))
            for role in ("reference", "calibration", "defect", "test_normal", "test_defect")
        },
    }
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_records(roles: dict[str, list[Path]],
                    request: RegistrationRequest) -> tuple[list[dict], dict[str, list[dict]]]:
    records: list[dict] = []
    by_role: dict[str, list[dict]] = {}
    for role in ("reference", "calibration", "defect", "test_normal", "test_defect"):
        role_records: list[dict] = []
        for index, path in enumerate(roles.get(role, [])):
            original = request.source_original_names.get(str(path)) or Path(path).name
            original = Path(original).name
            rel_path = f"samples/{role}/{_safe_source_filename(original, index)}"
            role_records.append({
                "role": role,
                "rel_path": rel_path,
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
                "original_filename": original,
                "source_path": path,
            })
        by_role[role] = role_records
        records.extend(role_records)
    return records, by_role


def _public_source_samples(records: list[dict]) -> list[dict]:
    return [
        {key: item[key] for key in
         ("role", "rel_path", "sha256", "size_bytes", "original_filename")}
        for item in records
    ]


def _fingerprint_for_version(
    store: ProductStore,
    product: ProductRecord,
    version: ProductVersion,
) -> dict:
    """返回 {"status": "available"|"unavailable", "fingerprint": str, "reason": str"}。"""
    if version.registration_fingerprint:
        value = str(version.registration_fingerprint)
        if re.fullmatch(r"[0-9a-f]{64}", value):
            return {"status": "available", "fingerprint": value, "reason": ""}
        return {"status": "unavailable", "fingerprint": "",
                "reason": "registration_fingerprint 格式非法"}

    samples = version.source_samples or []
    if not samples:
        return {"status": "unavailable", "fingerprint": "",
                "reason": "缺少 source_samples，无法重建指纹"}
    groups: dict[str, list[str]] = {
        role: [] for role in
        ("reference", "calibration", "defect", "test_normal", "test_defect")
    }
    seen: set[str] = set()
    for item in samples:
        role = item.get("role")
        digest = item.get("sha256") or item.get("image_sha256")
        if role not in groups:
            return {"status": "unavailable", "fingerprint": "",
                    "reason": f"source_samples 含未知角色：{role!r}"}
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            return {"status": "unavailable", "fingerprint": "",
                    "reason": f"source_samples SHA256 非法：{role}"}
        digest = digest.lower()
        if digest in seen:
            return {"status": "unavailable", "fingerprint": "",
                    "reason": "source_samples 存在重复 SHA256"}
        seen.add(digest)
        groups[role].append(digest)

    registration_path = store.version_directory(
        product.product_id, version.version) / "registration.json"
    try:
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "fingerprint": "",
                "reason": f"缺少/无法读取 registration.json：{exc}"}
    try:
        expected = {
            "reference": int(registration["n_reference"]),
            "calibration": int(registration["n_calibration"]),
            "defect": int(registration.get("n_defect", 0)),
            "test_normal": int(registration.get("n_test_normal", 0)),
            "test_defect": int(registration.get("n_test_defect", 0)),
        }
    except (KeyError, TypeError, ValueError):
        return {"status": "unavailable", "fingerprint": "",
                "reason": "registration.json 缺少样本数量元数据"}
    metadata = version.calibration_metadata or {}
    try:
        metadata_calibration = int(metadata["n_calibration"])
    except (KeyError, TypeError, ValueError):
        return {"status": "unavailable", "fingerprint": "",
                "reason": "calibration_metadata 缺少 n_calibration"}
    if metadata_calibration != expected["calibration"]:
        return {"status": "unavailable", "fingerprint": "",
                "reason": "calibration_metadata 与 registration.json 校准数量不一致"}
    for role, count in expected.items():
        if not groups[role]:
            if count > 0:
                return {"status": "unavailable", "fingerprint": "",
                        "reason": f"{role} 缺失（期望 {count} 张）"}
        if len(groups[role]) != count:
            return {"status": "unavailable", "fingerprint": "",
                    "reason": f"{role} 数量 {len(groups[role])} 与元数据 {count} 不一致"}
    if not version.feature_extractor_version or not version.weights_sha256:
        return {"status": "unavailable", "fingerprint": "",
                "reason": "版本缺少特征版本或权重 SHA256"}
    version_dir = store.version_directory(product.product_id, version.version)
    for item in samples:
        relative = item.get("rel_path")
        if not relative:
            continue
        path = Path(relative)
        if not path.is_absolute():
            path = version_dir / path
        if path.is_file():
            digest = (item.get("sha256") or item.get("image_sha256") or "").lower()
            if sha256_file(path) != digest:
                return {"status": "unavailable", "fingerprint": "",
                        "reason": f"source_samples 文件哈希不匹配：{relative}"}
            if item.get("size_bytes") is not None and path.stat().st_size != item["size_bytes"]:
                return {"status": "unavailable", "fingerprint": "",
                        "reason": f"source_samples 文件大小不匹配：{relative}"}
    return {
        "status": "available",
        "fingerprint": compute_registration_fingerprint(
            name=product.name,
            category=product.category,
            calibration_count=len(groups["calibration"]),
            groups=groups,
            image_quantile=float(metadata.get("image_quantile", IMAGE_QUANTILE)),
            pixel_quantile=float(metadata.get("pixel_quantile", PIXEL_QUANTILE)),
            feature_extractor_version=version.feature_extractor_version,
            weights_sha256=version.weights_sha256,
        ),
        "reason": "",
    }


def find_registration_duplicate(store: ProductStore, fingerprint: str) -> dict:
    """只扫描 active 产品的有效 current 版本。

    返回::

        {
          "duplicate": bool,
          "matched": {"product_id", "version", "manifest_sha256"} | None,
          "unavailable": [{"product_id", "version", "reason"}, ...]
        }

    历史 superseded/非 current、draft/disabled/archived 不参与阻塞式判重。
    """
    result = {"duplicate": False, "matched": None, "unavailable": []}
    for product in store.list_products(include_archived=True):
        if product.status != "active" or product.current_version is None:
            continue
        try:
            validation = store.validate_product(product.product_id)
            if not validation.get("ok"):
                result["unavailable"].append({
                    "product_id": product.product_id,
                    "version": product.current_version,
                    "reason": "active 产品完整性校验失败：" + "; ".join(validation.get("problems") or []),
                })
                continue
            version = store.get_version(product.product_id, product.current_version)
        except Exception as exc:  # noqa: BLE001
            result["unavailable"].append({
                "product_id": product.product_id,
                "version": product.current_version,
                "reason": f"当前版本不可读取：{type(exc).__name__}: {exc}",
            })
            continue
        if version.status != "active":
            continue
        candidate = _fingerprint_for_version(store, product, version)
        if candidate["status"] != "available":
            result["unavailable"].append({
                "product_id": product.product_id,
                "version": version.version,
                "reason": candidate["reason"],
            })
            continue
        if candidate["fingerprint"] == fingerprint:
            result["duplicate"] = True
            result["matched"] = {
                "product_id": product.product_id,
                "version": version.version,
                "manifest_sha256": version.manifest_sha256,
            }
    return result


def _members(paths: list[Path], *, role: str, product_id: str, shot: int,
             rel_paths: list[str] | None = None) -> list[dict]:
    return [{
        "order_index": index,
        "rel_path": rel_paths[index] if rel_paths is not None else str(path),
        "image_sha256": sha256_file(path),
        "product_id": product_id,
        "role": role,
        "shot": shot,
    } for index, path in enumerate(paths)]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)


def _register_product_impl(
    store: ProductStore,
    request: RegistrationRequest,
    *,
    extractor: MvtecSpatialExtractor | None = None,
    device: str = "auto",
    batch_size: int = 8,
    fingerprint: str = "",
    source_records: list[dict] | None = None,
    source_by_role: dict[str, list[dict]] | None = None,
) -> RegistrationResult:
    """执行一次完整注册（调用方必须已持有指纹锁）。"""
    labelled, roles = _register_request_validation(request)
    reference = roles["reference"]
    calibration = roles["calibration"]
    defects = roles["defect"]
    test_normal = roles["test_normal"]
    test_defect = roles["test_defect"]
    if source_records is None or source_by_role is None:
        raise RegistrationError("注册内部错误：缺少持久化样本记录")

    if extractor is None:
        ensure_local_weights()
        extractor = MvtecSpatialExtractor(device=device, batch_size=batch_size,
                                          allow_download=False)
    extractor_info = _extractor_info(extractor)
    score_device = str(getattr(extractor, "device", "cpu"))

    try:
        ref_global, ref_spatial = extractor.extract_paths_both([str(p) for p in reference])
        cal_global, cal_spatial = extractor.extract_paths_both([str(p) for p in calibration])
        defect_global = defect_spatial = None
        if defects:
            defect_global, defect_spatial = extractor.extract_paths_both(
                [str(p) for p in defects]
            )
    except Exception as exc:  # noqa: BLE001
        raise RegistrationError(f"特征提取失败：{type(exc).__name__}: {exc}") from exc

    if ref_global.shape != (len(reference), 512) or ref_spatial.shape != (
        len(reference), 256, 14, 14
    ):
        raise RegistrationError("参考图特征形状异常")
    if cal_global.shape != (len(calibration), 512) or cal_spatial.shape != (
        len(calibration), 256, 14, 14
    ):
        raise RegistrationError("校准图特征形状异常")

    image_scores = score_image_cosine(cal_global, ref_global, device=score_device).distances
    pixel_chunks: list[np.ndarray] = []
    pixel_map_shapes: list[list[int]] = []
    for index, path in enumerate(calibration):
        with Image.open(path) as handle:
            width, height = handle.size
        heatmap = score_spatial_cosine(
            cal_spatial[index], ref_spatial, (height, width), device=score_device
        ).heatmaps[0]
        pixel_chunks.append(np.asarray(heatmap, dtype=np.float32).ravel())
        pixel_map_shapes.append([int(height), int(width)])
    pixel_scores = np.concatenate(pixel_chunks).astype(np.float32, copy=False)
    del pixel_chunks
    pixel_scores_sha256 = hashlib.sha256(
        np.ascontiguousarray(pixel_scores, dtype=np.float32).tobytes(order="C")
    ).hexdigest()

    try:
        image_cal = calibrate_image_scores(image_scores, quantile=request.image_quantile)
        pixel_cal = calibrate_pixel_scores(pixel_scores, quantile=request.pixel_quantile)
    except ProductCalibrationError as exc:
        raise RegistrationError(f"阈值校准失败：{exc}") from exc

    sample_note = calibration_sample_note(len(calibration))
    warnings = []
    if sample_note["low_sample_warning"]:
        warnings.append(
            f"校准图仅 {len(calibration)} 张（<{sample_note['default']}）：阈值统计不稳定，"
            "不承诺固定 FPR，建议补充校准样本。"
        )
    independent_test_provided = bool(test_normal or test_defect)
    if not independent_test_provided:
        warnings.append("未提供独立测试集：注册可用，但未通过独立性能验收。")

    product = None
    if request.product_id:
        product = store.get_product(request.product_id)
    else:
        product = store.create_product(
            request.name, request.category, metadata=request.metadata,
            feature_extractor_version=extractor_info["global_feature_version"],
        )
    product_id = product.product_id

    ref_members = _members(reference, role="reference", product_id=product_id,
                           shot=len(reference),
                           rel_paths=[r["rel_path"] for r in source_by_role["reference"]])
    cal_members = _members(calibration, role="calibration", product_id=product_id,
                           shot=len(calibration),
                           rel_paths=[r["rel_path"] for r in source_by_role["calibration"]])
    ref_member_sha = member_manifest_sha256(ref_members)
    cal_member_sha = member_manifest_sha256(cal_members)
    ref_feature_sha = feature_content_sha256(ref_global)
    ref_spatial_feature_sha = feature_content_sha256(ref_spatial)
    calibration_sha_list = [m["image_sha256"] for m in cal_members]
    heatmap_config = {"mode": "bilinear", "align_corners": False}
    pixel_score_method = (
        "1 - max cosine per patch (patch L2-normalized); "
        "bilinear upsample to original size; all pixels pooled"
    )

    source_samples = _public_source_samples(source_records)

    calibration_metadata = {
        "n_calibration": len(calibration),
        "image_quantile": image_cal.quantile,
        "pixel_quantile": pixel_cal.quantile,
        "method": image_cal.method,
        "image_quantile_method": image_cal.method,
        "pixel_quantile_method": pixel_cal.method,
        "image_threshold": image_cal.threshold,
        "pixel_threshold": pixel_cal.threshold,
        "image_stats": image_cal.stats,
        "pixel_stats": pixel_cal.stats,
        "image_exceed_count": image_cal.exceed_count,
        "pixel_exceed_count": pixel_cal.exceed_count,
        "pixel_n": int(pixel_cal.n),
        "pixel_dtype": pixel_cal.dtype,
        "pixel_stats_dtype": "float64",
        "pixel_threshold_dtype": "float32",
        "image_score_dtype": "float32",
        "image_threshold_dtype": "float64",
        "pixel_independent_samples": False,
        "image_independent_samples": True,
        "pixel_score_method": pixel_score_method,
        "pixel_map_shapes": pixel_map_shapes,
        "pixel_scores_sha256": pixel_scores_sha256,
        "pixel_scores_shape": [int(pixel_scores.size)],
        "heatmap_upsample": heatmap_config,
        "preprocess_version": extractor_info["preprocess_version"],
        "calibration_image_sha256": calibration_sha_list,
        "reference_member_sha256": ref_member_sha,
        "calibration_member_sha256": cal_member_sha,
        "reference_feature_sha256": ref_feature_sha,
        "reference_spatial_feature_sha256": ref_spatial_feature_sha,
        "sample_note": sample_note,
        "fpr_guarantee": False,
        "independent_test_provided": independent_test_provided,
        "independent_test_validated": False,
    }

    def populate(staging: Path, version: int) -> None:
        # 原始样本必须与记忆库/阈值一起进入同一个 staging/manifest/原子提交。
        for item in source_records:
            target = staging / item["rel_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["source_path"], target)
            if sha256_file(target) != item["sha256"]:
                raise RegistrationError(
                    f"持久化样本 SHA256 不一致：{item['rel_path']}"
                )
            if target.stat().st_size != item["size_bytes"]:
                raise RegistrationError(
                    f"持久化样本大小不一致：{item['rel_path']}"
                )
        reference_dir = staging / "reference"
        global_meta = save_product_bank(
            reference_dir / "global_reference", ref_global, ref_members,
            product_id=product_id, product_version=version,
            memory_type="global_reference", extractor_info=extractor_info,
        )
        spatial_meta = save_product_bank(
            reference_dir / "spatial_reference", ref_spatial, ref_members,
            product_id=product_id, product_version=version,
            memory_type="spatial_reference", extractor_info=extractor_info,
        )
        if global_meta["member_manifest_sha256"] != ref_member_sha:
            raise RegistrationError("全局记忆库成员哈希与预期不一致")
        if spatial_meta["member_manifest_sha256"] != ref_member_sha:
            raise RegistrationError("空间记忆库成员哈希与预期不一致")

        calibration_dir = staging / "calibration"
        _write_csv(
            calibration_dir / "members.csv",
            ["order_index", "rel_path", "image_sha256", "product_id", "role"],
            [{"order_index": m["order_index"], "rel_path": m["rel_path"],
              "image_sha256": m["image_sha256"], "product_id": product_id,
              "role": "calibration"} for m in cal_members],
        )
        _write_csv(
            calibration_dir / "image_scores.csv",
            ["order_index", "rel_path", "image_sha256", "image_score"],
            [{"order_index": index, "rel_path": source_by_role["calibration"][index]["rel_path"],
              "image_sha256": cal_members[index]["image_sha256"],
              "image_score": f"{float(image_scores[index]):.10f}"}
             for index, path in enumerate(calibration)],
        )
        _write_json(calibration_dir / "thresholds.json", {
            "schema_version": "microdefect.product_calibration/1",
            "product_id": product_id,
            "product_version": version,
            "n_calibration": len(calibration),
            "image_quantile": image_cal.quantile,
            "pixel_quantile": pixel_cal.quantile,
            "method": image_cal.method,
            "image_threshold": image_cal.threshold,
            "pixel_threshold": pixel_cal.threshold,
            "image_stats": image_cal.stats,
            "pixel_stats": pixel_cal.stats,
            "image_exceed_count": image_cal.exceed_count,
            "pixel_exceed_count": pixel_cal.exceed_count,
            "pixel_n": int(pixel_cal.n),
            "pixel_dtype": pixel_cal.dtype,
            "pixel_stats_dtype": "float64",
            "pixel_threshold_dtype": "float32",
            "image_quantile_method": image_cal.method,
            "pixel_quantile_method": pixel_cal.method,
            "image_score_dtype": "float32",
            "image_threshold_dtype": "float64",
            "pixel_independent_samples": False,
            "image_independent_samples": True,
            "pixel_score_method": pixel_score_method,
            "pixel_map_shapes": pixel_map_shapes,
            "pixel_scores_sha256": pixel_scores_sha256,
            "pixel_scores_shape": [int(pixel_scores.size)],
            "heatmap_upsample": heatmap_config,
            "preprocess_version": extractor_info["preprocess_version"],
            "calibration_image_sha256": calibration_sha_list,
            "reference_global_member_sha256": ref_member_sha,
            "reference_spatial_member_sha256": ref_member_sha,
            "reference_feature_sha256": ref_feature_sha,
            "reference_spatial_feature_sha256": ref_spatial_feature_sha,
            "calibration_member_sha256": cal_member_sha,
            "extractor": extractor_info,
            "sample_note": sample_note,
            "fpr_guarantee": False,
            "independent_test_provided": independent_test_provided,
            "independent_test_validated": False,
        })

        if defects:
            defects_dir = staging / "defects"
            np.save(defects_dir / "global_defect.npy", defect_global, allow_pickle=False)
            np.save(defects_dir / "spatial_defect.npy", defect_spatial, allow_pickle=False)
            defect_members = _members(defects, role="defect", product_id=product_id,
                                      shot=len(defects),
                                      rel_paths=[r["rel_path"] for r in source_by_role["defect"]])
            _write_csv(
                defects_dir / "members.csv",
                ["order_index", "rel_path", "image_sha256", "product_id", "role"],
                [{"order_index": m["order_index"], "rel_path": m["rel_path"],
                  "image_sha256": m["image_sha256"], "product_id": product_id,
                  "role": "defect"} for m in defect_members],
            )
            _write_json(defects_dir / "metadata.json", {
                "schema_version": "microdefect.product_defect_bank/1",
                "product_id": product_id,
                "product_version": version,
                "n_defect": len(defects),
                "global_shape": list(defect_global.shape),
                "spatial_shape": list(defect_spatial.shape),
                "used_for_scoring": False,
                "note": "第一版仅独立存储，不参与正常记忆库与阈值。",
            })

        if independent_test_provided:
            tests_dir = staging / "tests"
            rows = []
            for role, paths in (("test_normal", test_normal), ("test_defect", test_defect)):
                for index, path in enumerate(paths):
                    rows.append({
                        "order_index": len(rows), "role": role,
                        "rel_path": source_by_role[role][index]["rel_path"],
                        "image_sha256": sha256_file(path),
                    })
            _write_csv(tests_dir / "members.csv",
                       ["order_index", "role", "rel_path", "image_sha256"], rows)
            _write_json(tests_dir / "test_metadata.json", {
                "schema_version": "microdefect.product_test/1",
                "product_id": product_id,
                "product_version": version,
                "independent_test_provided": True,
                "independent_test_validated": False,
                "n_test_normal": len(test_normal),
                "n_test_defect": len(test_defect),
                "note": "仅登记测试集，未在本阶段计算性能或标记验收通过。",
            })

        _write_json(staging / "registration.json", {
            "schema_version": "microdefect.product_registration/1",
            "product_id": product_id,
            "product_version": version,
            "n_reference": len(reference),
            "n_calibration": len(calibration),
            "n_defect": len(defects),
            "independent_test_provided": independent_test_provided,
            "independent_test_validated": False,
            "image_threshold": image_cal.threshold,
            "pixel_threshold": pixel_cal.threshold,
            "sample_note": sample_note,
            "warnings": warnings,
        })

    try:
        product, version = store.register_version(
            product_id,
            populate=populate,
            activate=request.activate,
            threshold=image_cal.threshold,
            calibration_metadata=calibration_metadata,
            feature_extractor_version=extractor_info["global_feature_version"],
            weights_sha256=extractor_info["weight_sha256"],
            defect_bank_path="defects" if defects else None,
            source_samples=source_samples,
            source_status="persisted",
            registration_fingerprint=fingerprint,
        )
    except (ProductIntegrityError, ProductCalibrationError) as exc:
        raise RegistrationError(f"注册提交失败：{exc}") from exc

    validation = store.validate_product(product_id)
    if not validation["ok"]:
        raise RegistrationError(f"注册后完整性校验失败：{validation['problems']}")

    return RegistrationResult(
        product=product,
        version=version,
        image_threshold=image_cal.threshold,
        pixel_threshold=pixel_cal.threshold,
        n_reference=len(reference),
        n_calibration=len(calibration),
        low_sample_warning=sample_note["low_sample_warning"],
        independent_test_provided=independent_test_provided,
        independent_test_validated=False,
        manifest_sha256=version.manifest_sha256,
        warnings=warnings,
    )


def register_product(
    store: ProductStore,
    request: RegistrationRequest,
    *,
    extractor: MvtecSpatialExtractor | None = None,
    device: str = "auto",
    batch_size: int = 8,
) -> RegistrationResult:
    """注册入口：先按指纹加跨进程锁，再判重并原子提交。

    * 新注册（request.product_id=None）：相同指纹最多成功一次；
    * 显式 product_id 的重新注册仍按 P2.1 创建新版本，不做全局判重；
    * 锁文件：<root>/.locks/registration-<fingerprint>.lock，超时抛出明确错误。
    """
    _, roles = _register_request_validation(request)
    if extractor is None:
        ensure_local_weights()
        extractor = MvtecSpatialExtractor(
            device=device, batch_size=batch_size, allow_download=False
        )
    extractor_info = _extractor_info(extractor)
    records, by_role = _source_records(roles, request)
    groups = {
        role: [item["sha256"] for item in by_role.get(role, [])]
        for role in ("reference", "calibration", "defect", "test_normal", "test_defect")
    }
    fingerprint = compute_registration_fingerprint(
        name=request.name,
        category=request.category,
        calibration_count=request.calibration_count,
        groups=groups,
        image_quantile=request.image_quantile,
        pixel_quantile=request.pixel_quantile,
        feature_extractor_version=extractor_info["global_feature_version"],
        weights_sha256=extractor_info["weight_sha256"],
    )

    with store.registration_lock(fingerprint):
        if request.product_id is None:
            duplicate = find_registration_duplicate(store, fingerprint)
            if duplicate["duplicate"]:
                existing = duplicate["matched"]
                raise DuplicateRegistrationError(
                    fingerprint,
                    existing["product_id"],
                    int(existing["version"]),
                    str(existing.get("manifest_sha256", "")),
                )
        return _register_product_impl(
            store, request,
            extractor=extractor,
            device=device,
            batch_size=batch_size,
            fingerprint=fingerprint,
            source_records=records,
            source_by_role=by_role,
        )
