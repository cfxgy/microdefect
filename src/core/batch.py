#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量检测：文件夹扫描、批次执行、结果落盘与历史记录。

只复用 ``src.core.detector.Detector``；本模块不含任何检测算法。
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image

from src.core.detector import DetectionResult, Detector

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_BATCH_ROOT = Path(__file__).resolve().parents[2] / "cache" / "batch_outputs"

RESULT_FIELDS = [
    "index", "image_path", "file_name", "category", "image_score", "image_threshold",
    "pixel_threshold", "is_anomaly", "elapsed_time", "status", "error",
    "result_json", "heatmap_png", "overlay_png",
]


@dataclass
class BatchScan:
    folder: Path
    recursive: bool
    valid: list[Path] = field(default_factory=list)
    invalid: list[tuple[Path, str]] = field(default_factory=list)
    total_files: int = 0

    def summary(self) -> dict:
        return {
            "folder": str(self.folder),
            "recursive": self.recursive,
            "total_files": self.total_files,
            "valid": len(self.valid),
            "invalid": len(self.invalid),
            "invalid_items": [{"path": str(p), "reason": r} for p, r in self.invalid[:100]],
        }


def _check_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except Exception:  # noqa: BLE001
        return "invalid_or_corrupt_image"
    if width <= 0 or height <= 0:
        return "invalid_dimensions"
    return None


def scan_folder(folder: str | Path, recursive: bool = True) -> BatchScan:
    path = Path(folder).expanduser()
    if not path.is_dir():
        raise NotADirectoryError(f"文件夹不存在或不是目录：{path}")
    result = BatchScan(folder=path.resolve(), recursive=recursive)
    iterator = path.rglob("*") if recursive else path.iterdir()
    for candidate in sorted(iterator):
        if not candidate.is_file():
            continue
        result.total_files += 1
        if candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
            result.invalid.append((candidate, "unsupported_extension"))
            continue
        reason = _check_image(candidate)
        if reason is None:
            result.valid.append(candidate)
        else:
            result.invalid.append((candidate, reason))
    return result


def create_batch_dir(root: str | Path) -> Path:
    base = Path(root).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"batch_{stamp}_{uuid.uuid4().hex[:6]}"
    directory = base / batch_id
    directory.mkdir(parents=False, exist_ok=False)
    return directory


class BatchRun:
    """一次批量检测的执行状态；每调用一次 ``step()`` 处理一张图。"""

    def __init__(
        self,
        paths: list[Path],
        category: str,
        protocol: str,
        batch_dir: Path,
        detector: Detector,
        *,
        device: str = "auto",
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.category = category
        self.protocol = protocol
        self.batch_dir = Path(batch_dir)
        self.detector = detector
        self.device = device
        self.index = 0
        self.rows: list[dict] = []
        self.errors: list[dict] = []
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.started_perf = time.perf_counter()
        self.finished_at: str | None = None
        self.stop_requested = False
        self.finished = False
        self.versions = self._versions()

    def _versions(self) -> dict:
        try:
            description = self.detector.describe()
        except Exception:  # noqa: BLE001
            return {}
        return {
            "model_version": description.get("model_version", {}),
            "memory_bank_version": description.get("memory_bank_version", {}),
        }

    # ---- 状态 ----
    @property
    def total(self) -> int:
        return len(self.paths)

    @property
    def success(self) -> int:
        return sum(1 for row in self.rows if row["status"] == "success")

    @property
    def failed(self) -> int:
        return sum(1 for row in self.rows if row["status"] == "failed")

    @property
    def stopped(self) -> int:
        return sum(1 for row in self.rows if row["status"] == "stopped")

    @property
    def normal(self) -> int:
        return sum(
            1 for row in self.rows
            if row["status"] == "success" and str(row["is_anomaly"]) == "0"
        )

    @property
    def anomaly(self) -> int:
        return sum(
            1 for row in self.rows
            if row["status"] == "success" and str(row["is_anomaly"]) == "1"
        )

    def request_stop(self) -> None:
        self.stop_requested = True

    # ---- 执行 ----
    def step(self) -> dict | None:
        if self.finished:
            return None
        if self.index >= self.total:
            self.finish()
            return None
        if self.stop_requested:
            self._mark_remaining_stopped()
            self.finish()
            return None

        path = self.paths[self.index]
        index = self.index
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
        stem = f"{index:04d}_{path.stem}_{digest}"
        row = {
            "index": index,
            "image_path": str(path.resolve()),
            "file_name": path.name,
            "category": self.category,
            "image_score": "",
            "image_threshold": "",
            "pixel_threshold": "",
            "is_anomaly": "",
            "elapsed_time": "",
            "status": "failed",
            "error": "",
            "result_json": "",
            "heatmap_png": "",
            "overlay_png": "",
        }
        try:
            result = self.detector.detect(path)
            files = result.save(self.batch_dir, stem=stem)
            row.update({
                "image_score": f"{result.image_score:.8f}",
                "image_threshold": f"{result.image_threshold:.8f}",
                "pixel_threshold": f"{result.pixel_threshold:.8f}",
                "is_anomaly": "1" if result.is_anomaly else "0",
                "elapsed_time": f"{result.elapsed_time:.6f}",
                "status": "success",
                "result_json": Path(files["result_json"]).name,
                "heatmap_png": Path(files["heatmap_png"]).name,
                "overlay_png": Path(files["overlay_png"]).name,
            })
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.errors.append({
                "index": index,
                "image_path": str(path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
        self.rows.append(row)
        self.index += 1
        self.persist()
        return row

    def _mark_remaining_stopped(self) -> None:
        for offset, path in enumerate(self.paths[self.index:], start=self.index):
            self.rows.append({
                "index": offset, "image_path": str(path.resolve()), "file_name": path.name,
                "category": self.category, "image_score": "", "image_threshold": "",
                "pixel_threshold": "", "is_anomaly": "", "elapsed_time": "",
                "status": "stopped", "error": "user_stopped",
                "result_json": "", "heatmap_png": "", "overlay_png": "",
            })
        self.index = self.total

    def finish(self) -> dict:
        if self.finished:
            return self.summary()
        if self.index < self.total:
            self._mark_remaining_stopped()
        self.finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self.finished = True
        self.persist()
        (self.batch_dir / "batch_summary.json").write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.summary()

    # ---- 持久化 ----
    def persist(self) -> None:
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        with (self.batch_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        errors_path = self.batch_dir / "errors.csv"
        if self.errors:
            with errors_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["index", "image_path", "error_type", "error_message"]
                )
                writer.writeheader()
                writer.writerows(self.errors)
        (self.batch_dir / "batch_summary.json").write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def summary(self) -> dict:
        elapsed = time.perf_counter() - self.started_perf
        return {
            "schema": "microdefect.batch/1",
            "batch_id": self.batch_dir.name,
            "batch_dir": str(self.batch_dir),
            "protocol": self.protocol,
            "category": self.category,
            "device": self.device,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(elapsed, 3),
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "stopped": self.stopped,
            "normal": self.normal,
            "anomaly": self.anomaly,
            "finished": self.finished,
            "stop_requested": self.stop_requested,
            "model_version": self.versions.get("model_version", {}),
            "memory_bank_version": self.versions.get("memory_bank_version", {}),
            "files": {
                "summary": "batch_summary.json",
                "results": "results.csv",
                "errors": "errors.csv" if self.errors else None,
            },
            "images": self.rows,
        }


def list_batches(root: str | Path) -> list[dict]:
    base = Path(root).expanduser()
    if not base.is_dir():
        return []
    batches = []
    for candidate in sorted(base.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        summary_path = candidate / "batch_summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        batches.append(summary)
    return batches


def load_batch(batch_dir: str | Path) -> dict:
    directory = Path(batch_dir)
    summary_path = directory / "batch_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"批次不存在或缺少 batch_summary.json：{directory}")
    return json.loads(summary_path.read_text(encoding="utf-8"))
