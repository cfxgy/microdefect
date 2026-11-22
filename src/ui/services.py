#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""界面服务层：选项发现、上传校验、检测调用与结果落盘。

与 Streamlit 解耦，便于单元测试；页面层只负责渲染与交互。
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import uuid
from pathlib import Path

from PIL import Image

from src.core.detector import DetectionResult, Detector, DetectorError
from src.core.product_calibration import IMAGE_QUANTILE, PIXEL_QUANTILE
from src.core.product_batch import (
    DEFAULT_PRODUCT_BATCH_ROOT,
    ProductBatchError,
    ProductBatchRun,
    batch_file_bytes,
    batch_zip,
    list_product_batches as core_list_product_batches,
    load_product_batch as core_load_product_batch,
)
from src.core.product_drift import (
    DEFAULT_DRIFT_ROOT,
    DriftError,
    DriftMonitor,
    drift_file_bytes as core_drift_file_bytes,
    list_drift_windows as core_list_drift_windows,
    load_drift_window as core_load_drift_window,
)
from src.core.product_feedback import (
    FeedbackConflictError,
    FeedbackError,
    activate_update as core_activate_update,
    create_feedback as core_create_feedback,
    list_feedback as core_list_feedback,
    list_updates as core_list_updates,
    prepare_update as core_prepare_update,
    review_feedback as core_review_feedback,
)
from src.core.product_detector import ProductDetector, ProductDetectorError
from src.core.product_registration import (
    DuplicateRegistrationError,
    RegistrationError,
    RegistrationRequest,
    compute_registration_fingerprint,
    find_registration_duplicate,
    register_product as core_register_product,
    validate_calibration_count,
)
from src.core.products import ProductStore, ProductStoreError
from src.core.registry import MemoryRegistry, RegistryError
from src.features import extractor as p2
from src.features.mvtec_spatial_extractor import weights_sha256
from src.core.weights import ensure_local_weights

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "cache" / "ui_uploads"
RESULT_DIR = PROJECT_ROOT / "cache" / "ui_results"
PRODUCT_UPLOAD_DIR = PROJECT_ROOT / "cache" / "ui_product_uploads"
PRODUCT_RESULT_DIR = PROJECT_ROOT / "cache" / "ui_product_results"
FORMAL_PRODUCT_ROOT = PROJECT_ROOT / "memory" / "products"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
SUPPORTED_MIME = ("image/png", "image/jpeg", "image/bmp")
PROTOCOL_LABELS = {
    "group_disjoint": "Group-disjoint（4 类近似分组）",
    "official": "Official（官方划分）",
}


class ImageValidationError(ValueError):
    """上传文件不是可解析的、受支持的图像。"""


def discover_options() -> dict[str, list[str]]:
    """只返回注册器校验通过的协议与类别，不完整的条目不出现。"""
    options: dict[str, list[str]] = {}
    for protocol in ("group_disjoint", "official"):
        try:
            registry = MemoryRegistry(protocol=protocol)
        except RegistryError:
            continue
        valid: list[str] = []
        for category in registry.available_categories():
            try:
                registry.load_config(category)
            except RegistryError:
                continue
            valid.append(category)
        if valid:
            options[protocol] = valid
    return options


def validate_image_bytes(data: bytes, filename: str) -> tuple[int, int]:
    if not data:
        raise ImageValidationError("文件为空，无法检测。")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"文件过大：{len(data) / 1024 / 1024:.1f} MB，上限 {MAX_UPLOAD_BYTES // 1024 // 1024} MB。"
        )
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ImageValidationError(
            f"不支持的图像格式：{suffix or '（无扩展名）'}；仅支持 PNG/JPG/JPEG/BMP。"
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception as exc:  # noqa: BLE001
        raise ImageValidationError(f"无法解析图像文件：{type(exc).__name__}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ImageValidationError("图像尺寸非法。")
    return width, height


def _unique_stem(filename: str, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()[:10]
    return f"{Path(filename).stem}_{digest}"


def save_upload(data: bytes, filename: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower()
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)
    return path


def hash_signature(data: bytes, category: str, protocol: str, device: str) -> str:
    return hashlib.sha256(
        data + category.encode() + protocol.encode() + device.encode()
    ).hexdigest()


def run_detection(
    data: bytes,
    filename: str,
    category: str,
    protocol: str,
    *,
    device: str = "auto",
    detector: Detector | None = None,
) -> dict:
    """校验并执行一次真实检测，返回结果对象与落盘文件路径。"""
    validate_image_bytes(data, filename)
    upload_path = save_upload(data, filename)
    engine = detector or Detector(category, protocol=protocol, device=device)
    try:
        result: DetectionResult = engine.detect(upload_path)
    except DetectorError:
        raise
    out_dir = RESULT_DIR / uuid.uuid4().hex
    stem = _unique_stem(filename, data)
    files = result.save(out_dir, stem=stem)
    return {
        "result": result,
        "files": files,
        "upload_path": str(upload_path),
        "signature": hash_signature(data, category, protocol, device),
    }


def load_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


# ---------------------------------------------------------------------------
# P3.2 产品级服务（薄封装；算法全部调用 P2.1 / P3.1.1 核心）
# ---------------------------------------------------------------------------

class ProductUIServiceError(RuntimeError):
    """产品 UI 服务层错误。"""


def product_root() -> Path:
    """正式产品根目录；测试/验收可用环境变量指向独立临时根目录。"""
    override = os.environ.get("MICRODEFECT_PRODUCT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return FORMAL_PRODUCT_ROOT


def product_signature(product_id: str, version: int, image_sha256: str,
                      root: Path, device: str) -> str:
    return hashlib.sha256(
        f"{product_id}|{version}|{image_sha256}|{root}|{device}".encode("utf-8")
    ).hexdigest()


def _safe_upload_name(filename: str, index: int, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename or "image").stem)[:48] or "image"
    return f"{index:03d}_{stem}{suffix}"


def _normalize_uploads(items) -> list[tuple[str, bytes]]:
    normalized: list[tuple[str, bytes]] = []
    for item in items or []:
        if isinstance(item, dict):
            name = item.get("name", "")
            data = item.get("data", b"")
        else:
            name, data = item
        normalized.append((str(name), bytes(data)))
    return normalized


def _write_uploads(directory: Path, items, role: str) -> list[Path]:
    paths: list[Path] = []
    directory.mkdir(parents=True, exist_ok=True)
    for index, (name, data) in enumerate(_normalize_uploads(items)):
        validate_image_bytes(data, name)
        suffix = Path(name).suffix.lower()
        target = directory / _safe_upload_name(name, index, suffix)
        target.write_bytes(data)
        paths.append(target)
    return paths


def list_product_cards(root: str | Path | None = None) -> list[dict]:
    """返回产品列表与可检测原因；不修改任何产品文件。"""
    base = Path(root) if root is not None else product_root()
    try:
        store = ProductStore(base)
        products = store.list_products(include_archived=True)
    except ProductStoreError as exc:
        raise ProductUIServiceError(f"产品列表读取失败：{exc}") from exc

    cards: list[dict] = []
    for product in products:
        reasons: list[str] = []
        version = None
        validation = {"ok": False, "problems": ["未校验"]}
        try:
            validation = store.validate_product(product.product_id)
        except Exception as exc:  # noqa: BLE001
            validation = {"ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}
        if product.current_version is not None:
            try:
                version = store.get_version(product.product_id, product.current_version)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"当前版本不可读取：{type(exc).__name__}: {exc}")
        else:
            reasons.append("缺少 current_version")
        if product.status != "active":
            reasons.append(f"产品状态为 {product.status}，不可检测")
        if version is not None and version.status != "active":
            reasons.append(f"当前版本派生状态为 {version.status}，不可检测")
        if not validation.get("ok"):
            problems = validation.get("problems") or []
            reasons.append("完整性校验失败" + (f"：{problems}" if problems else ""))
        cards.append({
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "status": product.status,
            "current_version": product.current_version,
            "version_status": version.status if version is not None else None,
            "created_at": version.created_at if version is not None else "",
            "image_threshold": version.threshold if version is not None else None,
            "manifest_sha256": version.manifest_sha256 if version is not None else "",
            "source_status": version.source_status if version is not None else "unavailable",
            "source_samples": list(version.source_samples) if version is not None else [],
            "validation_ok": bool(validation.get("ok")),
            "detectable": not reasons,
            "reason": "；".join(reasons),
        })
    return cards


def _original_name_map(items, paths: list[Path]) -> dict[str, str]:
    normalized = _normalize_uploads(items)
    return {
        str(path): Path(name).name
        for (name, _data), path in zip(normalized, paths)
    }


def registration_fingerprint_for_uploads(
    *,
    name: str,
    category: str,
    calibration_count: int,
    reference_files,
    calibration_files,
    defect_files=None,
    test_normal_files=None,
    test_defect_files=None,
) -> tuple[str, dict[str, list[str]]]:
    """在 UI 层计算与核心一致的候选指纹（用于提交前提示已有产品）。"""
    groups: dict[str, list[str]] = {}
    for role, items in (
        ("reference", reference_files), ("calibration", calibration_files),
        ("defect", defect_files), ("test_normal", test_normal_files),
        ("test_defect", test_defect_files),
    ):
        groups[role] = [
            hashlib.sha256(data).hexdigest()
            for _name, data in _normalize_uploads(items)
        ]
    ensure_local_weights()
    fingerprint = compute_registration_fingerprint(
        name=name,
        category=category,
        calibration_count=calibration_count,
        groups=groups,
        image_quantile=IMAGE_QUANTILE,
        pixel_quantile=PIXEL_QUANTILE,
        feature_extractor_version=p2.FEATURE_VERSION,
        weights_sha256=weights_sha256(),
    )
    return fingerprint, groups


def find_duplicate_registration(
    *,
    name: str,
    category: str,
    calibration_count: int,
    reference_files,
    calibration_files,
    defect_files=None,
    test_normal_files=None,
    test_defect_files=None,
    root: str | Path | None = None,
) -> dict | None:
    fingerprint, _ = registration_fingerprint_for_uploads(
        name=name, category=category, calibration_count=calibration_count,
        reference_files=reference_files, calibration_files=calibration_files,
        defect_files=defect_files, test_normal_files=test_normal_files,
        test_defect_files=test_defect_files,
    )
    return find_registration_duplicate(
        ProductStore(Path(root) if root is not None else product_root()),
        fingerprint,
    )


def register_product_from_uploads(
    *,
    name: str,
    category: str = "",
    reference_files,
    calibration_files,
    calibration_count: int,
    defect_files=None,
    test_normal_files=None,
    test_defect_files=None,
    root: str | Path | None = None,
    device: str = "auto",
) -> dict:
    """把上传文件写入受控临时目录，调用 P2.1 正式注册，并复核产品。"""
    if not str(name).strip():
        raise ProductUIServiceError("产品名称不能为空。")
    try:
        validate_calibration_count(int(calibration_count))
    except RegistrationError as exc:
        raise ProductUIServiceError(str(exc)) from exc
    references = _normalize_uploads(reference_files)
    calibration = _normalize_uploads(calibration_files)
    if len(references) < 5:
        raise ProductUIServiceError("正常参考图片至少需要 5 张。")
    if len(calibration) != int(calibration_count):
        raise ProductUIServiceError(
            f"校准图片数量与设置不一致：设置 {calibration_count} 张，实际 {len(calibration)} 张。"
        )

    base = Path(root) if root is not None else product_root()
    upload_root = PRODUCT_UPLOAD_DIR / f"register_{uuid.uuid4().hex}"
    try:
        reference_paths = _write_uploads(upload_root / "reference", references, "reference")
        calibration_paths = _write_uploads(upload_root / "calibration", calibration, "calibration")
        defect_paths = _write_uploads(upload_root / "defect", defect_files, "defect")
        test_normal_paths = _write_uploads(upload_root / "test_normal", test_normal_files, "test_normal")
        test_defect_paths = _write_uploads(upload_root / "test_defect", test_defect_files, "test_defect")
        source_original_names: dict[str, str] = {}
        for items, paths in (
            (reference_files, reference_paths),
            (calibration_files, calibration_paths),
            (defect_files, defect_paths),
            (test_normal_files, test_normal_paths),
            (test_defect_files, test_defect_paths),
        ):
            source_original_names.update(_original_name_map(items, paths))

        request = RegistrationRequest(
            name=str(name).strip(),
            category=str(category or "").strip(),
            reference_paths=reference_paths,
            calibration_paths=calibration_paths,
            calibration_count=int(calibration_count),
            defect_paths=defect_paths,
            test_normal_paths=test_normal_paths,
            test_defect_paths=test_defect_paths,
            source_original_names=source_original_names,
        )
        try:
            result = core_register_product(ProductStore(base), request, device=device)
        except DuplicateRegistrationError as exc:
            return {
                "duplicate": True,
                "fingerprint": exc.fingerprint,
                "existing": {
                    "product_id": exc.existing_product_id,
                    "version": exc.existing_version,
                    "manifest_sha256": exc.manifest_sha256,
                },
            }
        except Exception as exc:  # noqa: BLE001
            raise ProductUIServiceError(
                f"产品注册失败：{type(exc).__name__}: {exc}"
            ) from exc

        store = ProductStore(base)
        try:
            product = store.get_product(result.product.product_id)
            validation = store.validate_product(product.product_id)
            version = store.get_version(product.product_id, product.current_version)
        except Exception as exc:  # noqa: BLE001
            raise ProductUIServiceError(f"注册后复核失败：{exc}") from exc
        if not validation.get("ok"):
            raise ProductUIServiceError(
                f"注册后完整性校验失败：{validation.get('problems')}"
            )
        return {
            "summary": result.summary(),
            "product": product,
            "version": version,
            "validation": validation,
            "low_sample_warning": result.low_sample_warning,
            "independent_test_provided": result.independent_test_provided,
            "warnings": result.warnings,
            "source_status": version.source_status,
        }
    finally:
        shutil.rmtree(upload_root, ignore_errors=True)


def run_product_detection(
    product_id: str,
    data: bytes,
    filename: str,
    *,
    root: str | Path | None = None,
    device: str = "auto",
) -> dict:
    """校验上传图并调用 P3.1.1 ProductDetector；不写产品目录。"""
    validate_image_bytes(data, filename)
    base = Path(root) if root is not None else product_root()
    out_dir = PRODUCT_RESULT_DIR / uuid.uuid4().hex
    out_dir.mkdir(parents=True, exist_ok=False)
    suffix = Path(filename).suffix.lower()
    upload_path = out_dir / f"input{suffix}"
    upload_path.write_bytes(data)
    try:
        detector = ProductDetector(product_id, root=base, device=device)
        result = detector.detect(upload_path)
        files = result.save(out_dir)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return {
        "result": result,
        "files": files,
        "upload_path": str(upload_path),
        "signature": product_signature(
            product_id, result.version, result.input_image_sha256, base, device
        ),
    }


def create_product_batch(
    product_id: str,
    uploads,
    *,
    root: str | Path | None = None,
    batch_root: str | Path | None = None,
    device: str = "auto",
) -> ProductBatchRun:
    """把上传图片写入受控临时目录，再创建固定产品版本的批次。"""
    items = _normalize_uploads(uploads)
    if not items:
        raise ProductBatchError("批量检测至少需要一张图片")
    base = Path(root) if root is not None else product_root()
    temp = PRODUCT_UPLOAD_DIR / f"batch_{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    names: list[str] = []
    try:
        for index, (name, data) in enumerate(items):
            suffix = Path(name).suffix.lower() or ".bin"
            path = temp / f"{index:04d}{suffix}"
            path.write_bytes(data)
            paths.append(path)
            names.append(Path(name).name)
        return ProductBatchRun.create(
            product_id, paths, original_names=names,
            product_root=base, batch_root=batch_root or DEFAULT_PRODUCT_BATCH_ROOT,
            device=device,
        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def list_product_batches(root: str | Path | None = None,
                         product_id: str | None = None) -> list[dict]:
    return core_list_product_batches(root or DEFAULT_PRODUCT_BATCH_ROOT, product_id)


def load_product_batch(batch_id: str, root: str | Path | None = None) -> dict:
    return core_load_product_batch(batch_id, root or DEFAULT_PRODUCT_BATCH_ROOT)


def product_batch_zip(batch_id: str, root: str | Path | None = None) -> Path:
    return batch_zip(batch_id, root or DEFAULT_PRODUCT_BATCH_ROOT)


def product_batch_file_bytes(batch_id: str, relative: str,
                             root: str | Path | None = None) -> bytes:
    return batch_file_bytes(batch_id, relative, root or DEFAULT_PRODUCT_BATCH_ROOT)


def analyze_product_drift(
    product_id: str,
    uploads,
    *,
    root: str | Path | None = None,
    window_root: str | Path | None = None,
    device: str = "auto",
) -> dict:
    """保存上传图到受控临时目录，执行一次产品漂移窗口分析。"""
    items = _normalize_uploads(uploads)
    if not items:
        raise DriftError("监测窗口至少需要一张图片")
    temp = PRODUCT_UPLOAD_DIR / f"drift_{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    names: list[str] = []
    try:
        for index, (name, data) in enumerate(items):
            suffix = Path(name).suffix.lower() or ".bin"
            path = temp / f"{index:04d}{suffix}"
            path.write_bytes(data)
            paths.append(path)
            names.append(Path(name).name)
        monitor = DriftMonitor(
            product_id, product_root=root, device=device,
        )
        return monitor.analyze(paths, original_names=names,
                               window_root=window_root or DEFAULT_DRIFT_ROOT)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def list_drift_windows(root: str | Path | None = None,
                       product_id: str | None = None) -> list[dict]:
    return core_list_drift_windows(root or DEFAULT_DRIFT_ROOT, product_id)


def load_drift_window(window_id: str, root: str | Path | None = None) -> dict:
    return core_load_drift_window(window_id, root or DEFAULT_DRIFT_ROOT)


def drift_file_bytes(window_id: str, relative: str,
                     root: str | Path | None = None) -> bytes:
    return core_drift_file_bytes(window_id, relative, root or DEFAULT_DRIFT_ROOT)


def create_product_feedback(
    product_id: str,
    *,
    product_version: int,
    source_type: str,
    source_id: str | None,
    image_path: str | Path,
    image_score: float | None = None,
    image_threshold: float | None = None,
    image_status: str | None = None,
    drift_level: str | None = None,
    drift_reasons: list[str] | None = None,
    original_filename: str | None = None,
    root: str | Path | None = None,
) -> dict:
    return core_create_feedback(
        ProductStore(Path(root) if root is not None else product_root()),
        product_id=product_id, product_version=product_version,
        source_type=source_type, source_id=source_id,
        image_path=image_path, image_sha256=hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        image_score=image_score, image_threshold=image_threshold,
        image_status=image_status, drift_level=drift_level,
        drift_reasons=drift_reasons, original_filename=original_filename,
    )


def create_product_feedback_from_batch(
    product_id: str, *, product_version: int, batch_id: str, item: dict,
    root: str | Path | None = None,
) -> dict:
    files = item.get("files") or {}
    relative = item.get("input_path") or files.get("input_path")
    if not relative:
        raise DriftError("批次项没有可用的原图")
    data = product_batch_file_bytes(batch_id, relative)
    temp = PRODUCT_UPLOAD_DIR / f"feedback_batch_{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    path = temp / Path(relative).name
    try:
        path.write_bytes(data)
        return create_product_feedback(
            product_id, product_version=product_version, source_type="batch",
            source_id=batch_id, image_path=path,
            image_score=item.get("image_score"), image_threshold=item.get("image_threshold"),
            image_status=("anomaly" if item.get("is_anomaly") else "normal"),
            original_filename=item.get("original_filename"), root=root,
        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def create_product_feedback_from_drift(
    product_id: str, *, product_version: int, window_id: str, item: dict,
    drift_level: str | None = None, drift_reasons: list[str] | None = None,
    root: str | Path | None = None,
) -> dict:
    relative = item.get("input_path")
    if not relative:
        raise DriftError("漂移项没有可用的原图")
    data = core_drift_file_bytes(window_id, relative)
    temp = PRODUCT_UPLOAD_DIR / f"feedback_drift_{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    path = temp / Path(relative).name
    try:
        path.write_bytes(data)
        return create_product_feedback(
            product_id, product_version=product_version, source_type="drift",
            source_id=window_id, image_path=path,
            image_score=item.get("global_score"), image_threshold=None,
            image_status="drift", drift_level=drift_level,
            drift_reasons=drift_reasons,
            original_filename=item.get("original_filename"), root=root,
        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def list_product_feedback(product_id: str, *, status: str | None = None,
                          root: str | Path | None = None) -> list[dict]:
    return core_list_feedback(
        ProductStore(Path(root) if root is not None else product_root()),
        product_id, status=status,
    )


def review_product_feedback(product_id: str, feedback_id: str, conclusion: str,
                            note: str = "", root: str | Path | None = None) -> dict:
    return core_review_feedback(
        ProductStore(Path(root) if root is not None else product_root()),
        product_id, feedback_id, conclusion, note,
    )


def prepare_product_update(
    product_id: str,
    feedback_ids: list[str],
    validation_normal_uploads,
    validation_anomaly_uploads,
    *,
    note: str = "",
    root: str | Path | None = None,
) -> dict:
    temp = PRODUCT_UPLOAD_DIR / f"p5_validation_{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    normal_paths: list[Path] = []
    anomaly_paths: list[Path] = []
    try:
        for index, (name, data) in enumerate(_normalize_uploads(validation_normal_uploads)):
            path = temp / f"normal_{index:04d}{Path(name).suffix.lower() or '.png'}"
            path.write_bytes(data)
            normal_paths.append(path)
        for index, (name, data) in enumerate(_normalize_uploads(validation_anomaly_uploads)):
            path = temp / f"anomaly_{index:04d}{Path(name).suffix.lower() or '.png'}"
            path.write_bytes(data)
            anomaly_paths.append(path)
        return core_prepare_update(
            ProductStore(Path(root) if root is not None else product_root()),
            product_id, feedback_ids=feedback_ids,
            validation_normal_paths=normal_paths,
            validation_anomaly_paths=anomaly_paths,
            note=note,
        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def list_product_updates(product_id: str, root: str | Path | None = None) -> list[dict]:
    return core_list_updates(
        ProductStore(Path(root) if root is not None else product_root()), product_id)


def activate_product_update(product_id: str, update_id: str,
                            root: str | Path | None = None) -> dict:
    return core_activate_update(
        ProductStore(Path(root) if root is not None else product_root()),
        product_id, update_id)


def feedback_image_bytes(product_id: str, feedback: dict,
                         root: str | Path | None = None) -> bytes:
    store = ProductStore(Path(root) if root is not None else product_root())
    return (store.product_directory(product_id) / feedback["image_path"]).read_bytes()




