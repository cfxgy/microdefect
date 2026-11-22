#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""产品级单图检测引擎（P3.1）。

流程::

    product_id → ProductStore 校验产品 → current_version →
    version 完整性/manifest → 产品记忆库 → 校准阈值 →
    冻结特征提取 → 图像/像素余弦评分 → 与产品阈值比较 → 结果

与旧 MVTec 检测路径完全分离：本模块只读取 ``memory/products/<product_id>``，
不读取 MVTec 预置记忆库，也不使用固定 25 张的总协议阈值。

只读保证：检测过程不写产品目录，不修改 current_version、manifest 或阈值。
"""

from __future__ import annotations

import json
import math
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
from src.core.detector import _save_heatmap_png, _save_overlay_png
from src.core.product_bank import load_product_bank, verify_product_bank
from src.core.products import (
    ProductLockTimeoutError,
    ProductStore,
    ProductStoreError,
    sha256_file,
)
from src.core.weights import WeightsUnavailableError, ensure_local_weights
from src.features import extractor as p2
from src.features.mvtec_spatial_extractor import (
    MvtecSpatialExtractor,
    load_image_rgb_mvtec,
    weights_sha256,
)

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ProductDetectorError(RuntimeError):
    """产品检测不可用（产品、版本、记忆库、阈值或权重问题）。"""


@dataclass
class ProductDetectionResult:
    product_id: str
    product_name: str
    version: int
    manifest_sha256: str
    bank_version: str
    image_path: str
    image_sha256: str
    input_image_sha256: str
    image_size: tuple[int, int]
    image_score: float
    image_threshold: float
    image_status: str
    is_anomaly: bool
    pixel_threshold: float
    anomaly_heatmap: np.ndarray
    anomaly_mask: np.ndarray
    anomaly_pixels: int
    anomaly_ratio: float
    feature_extractor_version: str
    spatial_feature_version: str
    preprocess_version: str
    weights_sha256: str
    memory_bank: dict
    elapsed_time: float
    inference_time: float
    created_at: str
    heatmap_stats: dict = field(default_factory=dict)
    output_files: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "version": self.version,
            "protocol": "product",
            "manifest_sha256": self.manifest_sha256,
            "bank_version": self.bank_version,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "input_image_sha256": self.input_image_sha256,
            "image_size": list(self.image_size),
            "image_score": self.image_score,
            "image_threshold": self.image_threshold,
            "image_status": self.image_status,
            "is_anomaly": self.is_anomaly,
            "pixel_threshold": self.pixel_threshold,
            "heatmap_shape": list(self.anomaly_heatmap.shape),
            "heatmap_stats": self.heatmap_stats,
            "anomaly_pixels": self.anomaly_pixels,
            "anomaly_ratio": self.anomaly_ratio,
            "feature_extractor_version": self.feature_extractor_version,
            "spatial_feature_version": self.spatial_feature_version,
            "preprocess_version": self.preprocess_version,
            "weights_sha256": self.weights_sha256,
            "memory_bank": self.memory_bank,
            "elapsed_time": self.elapsed_time,
            "inference_time": self.inference_time,
            "created_at": self.created_at,
            "scoring_version": SCORING_VERSION,
            "output_files": self.output_files,
        }

    def save(self, out_dir: str | Path, *, stem: str | None = None) -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = stem or f"{self.product_id[:8]}_v{self.version}_{Path(self.image_path).stem}_{self.image_sha256[:8]}"
        json_path = out / f"{stem}_detect.json"
        heatmap_path = out / f"{stem}_heatmap.png"
        overlay_path = out / f"{stem}_overlay.png"
        mask_path = out / f"{stem}_anomaly_mask.png"

        _save_heatmap_png(self.anomaly_heatmap, heatmap_path)
        _save_overlay_png(self.image_path, self.anomaly_heatmap, overlay_path)
        Image.fromarray((self.anomaly_mask.astype(np.uint8) * 255)).save(mask_path)

        self.output_files = {
            "result_json": str(json_path),
            "heatmap_png": str(heatmap_path),
            "overlay_png": str(overlay_path),
            "anomaly_mask_png": str(mask_path),
        }
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.output_files


def _safe_join(base: Path, relative: str, field_name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ProductDetectorError(f"{field_name} 为空")
    if "\\" in relative or ":" in relative:
        raise ProductDetectorError(f"{field_name} 含非法字符：{relative!r}")
    from pathlib import PurePosixPath

    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ProductDetectorError(f"{field_name} 不是安全相对路径：{relative!r}")
    target = (base / pure).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as exc:
        raise ProductDetectorError(f"{field_name} 越出版本目录") from exc
    return target


class ProductDetector:
    """单产品、单版本的只读检测器。"""

    def __init__(
        self,
        product_id: str,
        *,
        root: str | Path | None = None,
        store: ProductStore | None = None,
        device: str = "auto",
        batch_size: int = 1,
        version: int | None = None,
    ) -> None:
        self.product_id = product_id
        self.store = store or ProductStore(root)
        try:
            # 与 set_current_version() 使用同一把产品级跨进程锁：
            # 产品状态、current_version、版本校验、阈值与记忆库读取期间，
            # 其他进程不能切换 current_version（TOCTOU 修复）。
            with self.store._product_lock(product_id):
                self._initialize_under_lock(
                    product_id, store=self.store, device=device, batch_size=batch_size,
                    requested_version=version,
                )
            # 锁已释放：后续模型加载不阻塞版本切换；结果仍绑定已验证版本。
            self._initialize_extractor(device=device, batch_size=batch_size)
        except ProductLockTimeoutError as exc:
            raise ProductDetectorError(
                f"等待产品锁超时，拒绝无锁读取或使用旧版本：{exc}"
            ) from exc
        except ProductDetectorError:
            raise
        except ProductStoreError as exc:
            raise ProductDetectorError(f"产品加载失败：{exc}") from exc

    def _initialize_under_lock(
        self,
        product_id: str,
        *,
        root: str | Path | None = None,
        store: ProductStore | None = None,
        device: str = "auto",
        batch_size: int = 1,
        requested_version: int | None = None,
    ) -> None:
        self.product_id = product_id
        self.store = store or ProductStore(root)
        try:
            product = self.store.get_product(product_id)
        except ProductStoreError as exc:
            raise ProductDetectorError(f"产品不可用：{exc}") from exc
        if product.status != "active":
            raise ProductDetectorError(
                f"产品状态为 {product.status!r}，只有 active 产品可以检测"
            )
        version_number = product.current_version if requested_version is None else requested_version
        if version_number is None:
            raise ProductDetectorError("产品没有 current_version，无法检测")
        validation = self.store.validate_product(product_id)
        if not validation["ok"]:
            raise ProductDetectorError(
                f"产品完整性校验失败：{validation['problems']}"
            )
        try:
            version = self.store.get_version(product_id, version_number)
        except ProductStoreError as exc:
            raise ProductDetectorError(f"当前版本不可用：{exc}") from exc
        if requested_version is None:
            if version.status != "active":
                raise ProductDetectorError(
                    f"当前版本派生状态为 {version.status!r}，不是 active"
                )
        else:
            if version.status == "archived":
                raise ProductDetectorError("已归档版本不能用于检测")
            target_validation = self.store.validate_version(product_id, version_number)
            if not target_validation["ok"]:
                raise ProductDetectorError(
                    f"目标版本完整性校验失败：{target_validation['problems']}"
                )

        self.product = product
        self.version = version
        self.version_dir = self.store.version_directory(product_id, version.version)
        calibration_dir = _safe_join(self.version_dir, version.calibration_path, "calibration_path")
        threshold_path = calibration_dir / "thresholds.json"
        if not threshold_path.is_file():
            raise ProductDetectorError(f"缺少产品阈值文件：{threshold_path}")
        try:
            thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProductDetectorError(f"阈值文件损坏：{threshold_path}") from exc
        if thresholds.get("product_id") != product_id:
            raise ProductDetectorError("阈值文件的 product_id 与产品不一致")
        if int(thresholds.get("product_version", -1)) != version.version:
            raise ProductDetectorError("阈值文件的 version 与当前版本不一致")
        try:
            image_threshold = float(thresholds["image_threshold"])
            pixel_threshold = float(thresholds["pixel_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductDetectorError(f"阈值字段缺失或非法：{exc}") from exc
        for name, value in (("image_threshold", image_threshold),
                            ("pixel_threshold", pixel_threshold)):
            if not math.isfinite(value) or not (0.0 <= value <= 2.0):
                raise ProductDetectorError(f"{name} 非法：{value!r}")
        if version.threshold is None or not math.isfinite(float(version.threshold)):
            raise ProductDetectorError("版本元数据缺少 image threshold")
        if abs(float(version.threshold) - image_threshold) > 1e-9:
            raise ProductDetectorError(
                "version.json 的 threshold 与 thresholds.json 不一致"
            )
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold
        self.thresholds = thresholds

        try:
            ensure_local_weights()
            self.runtime_weight_sha256 = weights_sha256()
        except WeightsUnavailableError as exc:
            raise ProductDetectorError(str(exc)) from exc
        if not version.weights_sha256:
            raise ProductDetectorError("版本未记录 weights_sha256")
        if version.feature_extractor_version != p2.FEATURE_VERSION:
            raise ProductDetectorError("版本记录的特征提取器版本与当前运行时不符")
        if self.runtime_weight_sha256 != version.weights_sha256:
            raise ProductDetectorError(
                "当前权重 SHA256 与注册版本不一致，拒绝使用其他权重检测"
            )

        global_dir = self.version_dir / "reference" / "global_reference"
        spatial_dir = self.version_dir / "reference" / "spatial_reference"
        try:
            verify_product_bank(global_dir)
            verify_product_bank(spatial_dir)
            g_bank = load_product_bank(global_dir)
            s_bank = load_product_bank(spatial_dir)
        except Exception as exc:  # noqa: BLE001
            raise ProductDetectorError(f"产品记忆库不可用：{exc}") from exc
        for name, bank in (("global", g_bank), ("spatial", s_bank)):
            metadata = bank["metadata"]
            if metadata.get("product_id") != product_id:
                raise ProductDetectorError(f"{name} 记忆库 product_id 不匹配")
            if int(metadata.get("product_version", -1)) != version.version:
                raise ProductDetectorError(f"{name} 记忆库版本不匹配")
            if metadata.get("weight_sha256") != self.runtime_weight_sha256:
                raise ProductDetectorError(f"{name} 记忆库权重哈希不匹配")
            if metadata.get("global_feature_version") != p2.FEATURE_VERSION:
                raise ProductDetectorError(f"{name} 记忆库全局特征版本不匹配")
        if g_bank["metadata"].get("member_manifest_sha256") != thresholds.get(
            "reference_global_member_sha256"
        ):
            raise ProductDetectorError("global 记忆库成员哈希与阈值记录不一致")
        if s_bank["metadata"].get("member_manifest_sha256") != thresholds.get(
            "reference_spatial_member_sha256"
        ):
            raise ProductDetectorError("spatial 记忆库成员哈希与阈值记录不一致")

        # 顶层 bank_version 完全由本次实际加载并校验的 bank metadata 构造，
        # 不根据产品版本号推断、不硬编码。
        self.bank_version = (
            f"{g_bank['metadata'].get('protocol_version')}"
            f":g={g_bank['metadata'].get('member_manifest_sha256')}"
            f":s={s_bank['metadata'].get('member_manifest_sha256')}"
        )

        self.global_bank = g_bank
        self.spatial_bank = s_bank
        self.global_features = g_bank["features"]
        self.spatial_features = s_bank["features"]

    def _initialize_extractor(self, *, device: str, batch_size: int) -> None:
        """模型加载放在产品锁外，避免长时间占用产品锁。"""
        try:
            self.extractor = MvtecSpatialExtractor(
                device=device, batch_size=batch_size, allow_download=False
            )
        except WeightsUnavailableError as exc:
            raise ProductDetectorError(str(exc)) from exc
        metadata = self.spatial_bank["metadata"]
        if self.extractor.info.spatial_feature_version != metadata.get(
            "spatial_feature_version"
        ):
            raise ProductDetectorError("空间特征版本与注册记忆库不一致")
        if self.extractor.info.preprocess_version != metadata.get(
            "preprocess_version"
        ):
            raise ProductDetectorError("预处理版本与注册记忆库不一致")
        self.device = str(getattr(self.extractor, "device", "cpu"))

    def describe(self) -> dict:
        return {
            "product_id": self.product_id,
            "product_name": self.product.name,
            "version": self.version.version,
            "manifest_sha256": self.version.manifest_sha256,
            "bank_version": self.bank_version,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "feature_extractor_version": p2.FEATURE_VERSION,
            "spatial_feature_version": self.extractor.info.spatial_feature_version,
            "weights_sha256": self.runtime_weight_sha256,
            "global_member_sha256": self.global_bank["metadata"]["member_manifest_sha256"],
            "spatial_member_sha256": self.spatial_bank["metadata"]["member_manifest_sha256"],
        }

    def detect(self, image_path: str | Path) -> ProductDetectionResult:
        started = time.perf_counter()
        path = Path(image_path)
        if not path.is_file():
            raise ProductDetectorError(f"图像不存在：{path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ProductDetectorError(f"不支持的图像格式：{path.suffix}")
        try:
            image = load_image_rgb_mvtec(path)
        except Exception as exc:  # noqa: BLE001
            raise ProductDetectorError(
                f"图像读取失败：{path}（{type(exc).__name__}: {exc}）"
            ) from exc
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ProductDetectorError(f"图像尺寸非法：{path}")
        image_sha = sha256_file(path)

        tensor = self.extractor.preprocess(image).unsqueeze(0)
        global_feature, spatial_feature = self.extractor.extract_both_tensor(tensor)
        image_score = float(
            score_image_cosine(global_feature, self.global_features, device=self.device).distances[0]
        )
        spatial_result = score_spatial_cosine(
            spatial_feature[0], self.spatial_features, (height, width), device=self.device
        )
        heatmap = np.asarray(spatial_result.heatmaps[0], dtype=np.float32)
        if heatmap.shape != (height, width):
            raise ProductDetectorError("热图尺寸与输入图像不一致")
        if not math.isfinite(image_score) or not np.isfinite(heatmap).all():
            raise ProductDetectorError("检测结果包含 NaN/Inf")
        if not (0.0 <= image_score <= 2.0):
            raise ProductDetectorError(f"图像分数超出范围：{image_score}")

        mask = heatmap > self.pixel_threshold
        is_anomaly = bool(image_score > self.image_threshold)
        elapsed = time.perf_counter() - started
        return ProductDetectionResult(
            product_id=self.product_id,
            product_name=self.product.name,
            version=self.version.version,
            manifest_sha256=self.version.manifest_sha256,
            bank_version=self.bank_version,
            image_path=str(path),
            image_sha256=image_sha,
            input_image_sha256=image_sha,
            image_size=(int(width), int(height)),
            image_score=image_score,
            image_threshold=self.image_threshold,
            image_status="anomaly" if is_anomaly else "normal",
            is_anomaly=is_anomaly,
            pixel_threshold=self.pixel_threshold,
            anomaly_heatmap=heatmap,
            anomaly_mask=mask,
            anomaly_pixels=int(mask.sum()),
            anomaly_ratio=float(mask.mean()),
            feature_extractor_version=p2.FEATURE_VERSION,
            spatial_feature_version=self.extractor.info.spatial_feature_version,
            preprocess_version=self.extractor.info.preprocess_version,
            weights_sha256=self.runtime_weight_sha256,
            memory_bank={
                "product_id": self.product_id,
                "version": self.version.version,
                "bank_version": self.bank_version,
                "global_member_sha256": self.global_bank["metadata"]["member_manifest_sha256"],
                "spatial_member_sha256": self.spatial_bank["metadata"]["member_manifest_sha256"],
                "global_member_count": int(self.global_features.shape[0]),
                "spatial_member_count": int(self.spatial_features.shape[0]),
                "global_shape": list(self.global_features.shape),
                "spatial_shape": list(self.spatial_features.shape),
            },
            elapsed_time=elapsed,
            inference_time=elapsed,
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            heatmap_stats={
                "min": float(heatmap.min()),
                "max": float(heatmap.max()),
                "mean": float(heatmap.mean()),
            },
        )
