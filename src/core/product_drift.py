#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""产品级漂移监测（P4）。

只读产品数据；复用 P3.1.1 的冻结特征提取器、参考 bank 与余弦评分。
窗口聚合指标 + 注册校准基线；输出实验性预警，不更新任何产品资产。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.anomaly.mvtec_protocol_scoring import score_image_cosine, score_spatial_cosine
from src.core.product_detector import ProductDetector, ProductDetectorError
from src.core.products import sha256_file
from src.features.mvtec_spatial_extractor import load_image_rgb_mvtec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRIFT_ROOT = PROJECT_ROOT / "cache" / "product_drift"
SCHEMA = "microdefect.product_drift/1"
ALGORITHM_VERSION = "drift-v1"
MIN_WINDOW_SAMPLES = 5
OBSERVE_Z = 2.0
WARNING_Z = 3.0
EPS = 1e-9
LEVELS = ("normal", "observation", "drift_warning", "insufficient", "failed")


class DriftError(RuntimeError):
    """漂移监测不可用/写入失败。"""


class DriftBaselineError(DriftError):
    """基线不完整（旧产品未持久化 source samples 等）。"""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
    tmp.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def _stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "std": None, "p95": None, "min": None, "max": None}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _z(value: float, baseline: dict) -> float:
    std = baseline.get("std")
    if std is None or not np.isfinite(std) or std < EPS:
        return 0.0
    return float((value - baseline["mean"]) / std)


def _image_stats(path: Path) -> tuple[float, float, tuple[int, int]]:
    with Image.open(path) as handle:
        gray = np.asarray(handle.convert("L"), dtype=np.float32)
        size = handle.size
    return float(gray.mean()), float(gray.std()), (int(size[0]), int(size[1]))


def _safe_name(name: str, index: int, digest: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name or "image").stem)[:48] or "image"
    suffix = Path(name or "").suffix.lower()
    return f"{index:04d}_{digest[:8]}_{stem}{suffix}"


def _window_id_valid(window_id: str) -> bool:
    return bool(re.fullmatch(r"drift_[0-9]{8}_[0-9]{6}_[0-9a-f]{6}", window_id or ""))


class DriftMonitor:
    def __init__(self, product_id: str, *, product_root: str | Path | None = None,
                 device: str = "auto", batch_size: int = 1) -> None:
        self.product_id = product_id
        try:
            self.detector = ProductDetector(product_id, root=product_root, device=device,
                                            batch_size=batch_size)
        except ProductDetectorError as exc:
            raise DriftBaselineError(f"产品基线不可用：{exc}") from exc
        if self.detector.version.source_status != "persisted":
            raise DriftBaselineError(
                "产品来源样本未持久化（source_status=legacy），无法建立漂移基线"
            )
        self.version_dir = self.detector.version_dir
        self.reference_paths = self._role_paths("reference")
        self.calibration_paths = self._role_paths("calibration")
        if len(self.reference_paths) != len(self.detector.global_features):
            raise DriftBaselineError("reference source_samples 数量与记忆库成员数不一致")
        meta = self.detector.version.calibration_metadata or {}
        expected_cal = int(meta.get("n_calibration", -1))
        if expected_cal != len(self.calibration_paths) or expected_cal <= 0:
            raise DriftBaselineError("calibration source_samples 数量与元数据不一致")
        self.baseline = self._build_baseline()

    def _role_paths(self, role: str) -> list[Path]:
        paths = []
        for item in self.detector.version.source_samples:
            if item.get("role") != role:
                continue
            target = self.version_dir / str(item.get("rel_path", ""))
            if not target.is_file():
                raise DriftBaselineError(f"{role} 来源样本不存在：{item.get('rel_path')}")
            if item.get("sha256") and sha256_file(target) != item["sha256"]:
                raise DriftBaselineError(f"{role} 来源样本哈希不匹配：{item.get('rel_path')}")
            paths.append(target)
        if not paths:
            raise DriftBaselineError(f"缺少 {role} 来源样本")
        return paths

    def _build_baseline(self) -> dict:
        cal_global, cal_spatial = self.detector.extractor.extract_paths_both(
            [str(p) for p in self.calibration_paths]
        )
        global_scores = score_image_cosine(
            cal_global, self.detector.global_features, device=self.detector.device
        ).distances
        global_scores = [float(x) for x in global_scores]
        spatial_means = []
        spatial_p95 = []
        for index, path in enumerate(self.calibration_paths):
            with Image.open(path) as handle:
                width, height = handle.size
            heat = score_spatial_cosine(
                cal_spatial[index], self.detector.spatial_features,
                (height, width), device=self.detector.device,
            ).heatmaps[0]
            spatial_means.append(float(np.mean(heat)))
            spatial_p95.append(float(np.quantile(heat, 0.95)))
        cal_brightness = []
        cal_contrast = []
        for path in self.calibration_paths:
            mean, std, _ = _image_stats(path)
            cal_brightness.append(mean)
            cal_contrast.append(std)
        ref_brightness = []
        ref_contrast = []
        for path in self.reference_paths:
            mean, std, _ = _image_stats(path)
            ref_brightness.append(mean)
            ref_contrast.append(std)
        return {
            "reference_images": len(self.reference_paths),
            "calibration_images": len(self.calibration_paths),
            "global": _stats(global_scores),
            "spatial_mean": _stats(spatial_means),
            "spatial_p95": _stats(spatial_p95),
            "brightness_reference": _stats(ref_brightness),
            "contrast_reference": _stats(ref_contrast),
            "brightness": _stats(cal_brightness),
            "contrast": _stats(cal_contrast),
            "image_threshold": self.detector.image_threshold,
            "pixel_threshold": self.detector.pixel_threshold,
        }

    def analyze(
        self,
        paths: list[Path],
        *,
        original_names: list[str] | None = None,
        window_root: str | Path | None = None,
    ) -> dict:
        if not paths:
            raise DriftError("监测窗口至少需要一张图片")
        directory = self._create_window_dir(window_root)
        original_names = list(original_names or [p.name for p in paths])
        input_dir = directory / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        items = []
        valid_paths: list[Path] = []
        valid_names: list[str] = []
        valid_indices: list[int] = []
        for index, path in enumerate(paths):
            digest = sha256_file(path)
            stored = _safe_name(original_names[index], index, digest)
            target = input_dir / stored
            shutil.copyfile(path, target)
            item = {
                "index": index,
                "original_filename": original_names[index],
                "stored_filename": stored,
                "input_path": target.relative_to(directory).as_posix(),
                "input_sha256": digest,
                "size_bytes": int(path.stat().st_size),
                "status": "unprocessed",
                "global_score": None,
                "spatial_mean": None,
                "spatial_p95": None,
                "brightness": None,
                "contrast": None,
                "error": None,
            }
            items.append(item)
            try:
                with Image.open(path) as handle:
                    handle.verify()
                valid_paths.append(path)
                valid_names.append(original_names[index])
                valid_indices.append(index)
            except Exception as exc:  # noqa: BLE001
                item["status"] = "failed"
                item["error"] = f"{type(exc).__name__}: {exc}"

        global_scores = []
        spatial_means = []
        spatial_p95s = []
        brightness = []
        contrast = []
        for local_index, path in enumerate(valid_paths):
            item = items[valid_indices[local_index]]
            try:
                global_feature, spatial_feature = self.detector.extractor.extract_paths_both(
                    [str(path)]
                )
                score = float(score_image_cosine(
                    global_feature, self.detector.global_features,
                    device=self.detector.device,
                ).distances[0])
                with Image.open(path) as handle:
                    width, height = handle.size
                heat = score_spatial_cosine(
                    spatial_feature[0], self.detector.spatial_features,
                    (height, width), device=self.detector.device,
                ).heatmaps[0]
                mean_b, std_b, _ = _image_stats(path)
                item.update({
                    "status": "success",
                    "global_score": score,
                    "spatial_mean": float(np.mean(heat)),
                    "spatial_p95": float(np.quantile(heat, 0.95)),
                    "brightness": mean_b,
                    "contrast": std_b,
                })
                global_scores.append(score)
                spatial_means.append(float(np.mean(heat)))
                spatial_p95s.append(float(np.quantile(heat, 0.95)))
                brightness.append(mean_b)
                contrast.append(std_b)
            except Exception as exc:  # noqa: BLE001
                item["status"] = "failed"
                item["error"] = f"{type(exc).__name__}: {exc}"

        payload = self._evaluate(directory.name, items, global_scores, spatial_means,
                                 spatial_p95s, brightness, contrast)
        # 先写 CSV，最后写 window.json；JSON 不存在时历史读取不会把本次当成功记录。
        self._write_csv(directory, payload)
        _write_json(directory / "window.json", payload)
        return payload

    @staticmethod
    def _create_window_dir(root: str | Path | None) -> Path:
        base = Path(root or DEFAULT_DRIFT_ROOT)
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        window_id = f"drift_{stamp}_{uuid.uuid4().hex[:6]}"
        directory = base / window_id
        directory.mkdir(parents=False, exist_ok=False)
        return directory

    def _evaluate(self, window_id: str, items: list[dict], global_scores: list[float],
                  spatial_means: list[float], spatial_p95s: list[float],
                  brightness: list[float], contrast: list[float]) -> dict:
        success = sum(1 for item in items if item["status"] == "success")
        failed = sum(1 for item in items if item["status"] == "failed")
        metrics = {
            "global": _stats(global_scores),
            "spatial_mean": _stats(spatial_means),
            "spatial_p95": _stats(spatial_p95s),
            "brightness": _stats(brightness),
            "contrast": _stats(contrast),
        }
        reasons: list[str] = []
        level = "normal"
        if success == 0:
            level = "failed"
            reasons.append("所有图片均无法提取特征")
        elif success < MIN_WINDOW_SAMPLES:
            level = "insufficient"
            reasons.append(f"有效图片 {success} 张，少于最小窗口 {MIN_WINDOW_SAMPLES} 张；不给出可靠漂移结论")
        else:
            checks = [
                ("global", metrics["global"], "全局特征距离", "global", True),
                ("spatial_mean", metrics["spatial_mean"], "局部空间距离", "spatial_mean", True),
                ("brightness", metrics["brightness"], "亮度", "brightness", False),
                ("contrast", metrics["contrast"], "对比度", "contrast", False),
            ]
            for key, stats, label, baseline_key, directional in checks:
                if stats["mean"] is None:
                    continue
                z = _z(float(stats["mean"]), self.baseline[baseline_key])
                trigger_z = z if directional else abs(z)
                if trigger_z >= WARNING_Z:
                    level = "drift_warning"
                    reasons.append(
                        f"{label}均值 {stats['mean']:.6f}，相对校准基线 "
                        f"{self.baseline[baseline_key]['mean']:.6f} 偏移 {z:+.2f}σ（实验性阈值）"
                    )
                elif trigger_z >= OBSERVE_Z and level == "normal":
                    level = "observation"
                    reasons.append(
                        f"{label}均值偏移 {z:+.2f}σ，建议观察（实验性阈值）"
                    )
        payload = {
            "schema_version": SCHEMA,
            "algorithm_version": ALGORITHM_VERSION,
            "window_id": window_id,
            "product_id": self.product_id,
            "product_name": self.detector.product.name,
            "product_version": self.detector.version.version,
            "manifest_sha256": self.detector.version.manifest_sha256,
            "bank_version": self.detector.bank_version,
            "created_at": _now(),
            "total": len(items),
            "success": success,
            "failed": failed,
            "min_window_samples": MIN_WINDOW_SAMPLES,
            "observe_z": OBSERVE_Z,
            "warning_z": WARNING_Z,
            "threshold_source": "注册校准图像的窗口均值分布（均值/标准差）；实验性，无 FPR/显著性保证",
            "level": level,
            "reasons": reasons,
            "baseline": self.baseline,
            "metrics": metrics,
            "items": items,
        }
        return payload

    @staticmethod
    def _write_csv(directory: Path, payload: dict) -> None:
        fields = [
            "index", "original_filename", "stored_filename", "input_sha256",
            "size_bytes", "status", "global_score", "spatial_mean", "spatial_p95",
            "brightness", "contrast", "error",
        ]
        tmp = directory / "results.csv.tmp"
        with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in payload["items"]:
                writer.writerow({key: item.get(key) for key in fields})
        tmp.replace(directory / "results.csv")


def load_drift_window(window_id: str, root: str | Path | None = None) -> dict:
    if not _window_id_valid(window_id):
        raise DriftError(f"非法监测窗口 ID：{window_id!r}")
    directory = Path(root or DEFAULT_DRIFT_ROOT) / window_id
    path = directory / "window.json"
    if not path.is_file():
        raise FileNotFoundError(f"监测窗口不存在：{directory}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriftError(f"监测记录损坏：{exc}") from exc
    if payload.get("schema_version") != SCHEMA:
        raise DriftError("监测 schema 不支持")
    return payload


def list_drift_windows(root: str | Path | None = None,
                       product_id: str | None = None) -> list[dict]:
    base = Path(root or DEFAULT_DRIFT_ROOT)
    if not base.is_dir():
        return []
    windows = []
    for directory in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not directory.is_dir() or not _window_id_valid(directory.name):
            continue
        try:
            payload = load_drift_window(directory.name, base)
        except (OSError, DriftError):
            continue
        if product_id and payload.get("product_id") != product_id:
            continue
        payload.pop("items", None)
        windows.append(payload)
    return windows


def drift_file_bytes(window_id: str, relative: str,
                     root: str | Path | None = None) -> bytes:
    base = Path(root or DEFAULT_DRIFT_ROOT) / window_id
    target = (base / relative).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as exc:
        raise DriftError("监测文件路径越界") from exc
    if not target.is_file():
        raise FileNotFoundError(target)
    return target.read_bytes()
