#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""产品级批量检测与批次持久化（P3.3）。

复用 P3.1.1 ``ProductDetector``，一个批次固定一个产品当前版本；
逐张处理、失败隔离、支持停止，批次 JSON/CSV 原子写入，历史只读。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.core.batch import create_batch_dir
from src.core.product_detector import ProductDetector, ProductDetectorError
from src.core.products import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCT_BATCH_ROOT = PROJECT_ROOT / "cache" / "product_batches"
SCHEMA = "microdefect.product_batch/1"
ITEM_STATUSES = ("success", "failed", "unprocessed")
BATCH_STATUSES = ("running", "completed", "partial", "failed", "cancelled")


class ProductBatchError(RuntimeError):
    """产品批次不可用/写入失败。"""


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


def _safe_name(name: str, index: int, digest: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name or "image").stem)[:48] or "image"
    suffix = Path(name or "").suffix.lower()
    return f"{index:04d}_{digest[:8]}_{stem}{suffix}"


def _batch_id_valid(batch_id: str) -> bool:
    return bool(re.fullmatch(r"batch_[0-9]{8}_[0-9]{6}_[0-9a-f]{6}", batch_id or ""))


@dataclass
class ProductBatchRun:
    product_id: str
    batch_dir: Path
    detector: ProductDetector
    item_paths: list[Path]
    original_names: list[str]
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    index: int = 0
    status: str = "running"
    stop_requested: bool = False
    items: list[dict] = field(default_factory=list)
    batch_id: str = ""

    @classmethod
    def create(
        cls,
        product_id: str,
        paths: list[Path],
        *,
        original_names: list[str] | None = None,
        product_root: str | Path | None = None,
        batch_root: str | Path | None = None,
        device: str = "auto",
        batch_size: int = 1,
    ) -> "ProductBatchRun":
        if not paths:
            raise ProductBatchError("批量检测至少需要一张图片")
        detector = ProductDetector(
            product_id, root=product_root, device=device, batch_size=batch_size
        )
        directory = create_batch_dir(batch_root or DEFAULT_PRODUCT_BATCH_ROOT)
        original_names = list(original_names or [p.name for p in paths])
        run = cls(
            product_id=product_id,
            batch_dir=directory,
            detector=detector,
            item_paths=[Path(p) for p in paths],
            original_names=original_names,
            batch_id=directory.name,
        )
        run._prepare_items()
        run._persist()
        return run

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def finished(self) -> bool:
        return self.status != "running"

    def _prepare_items(self) -> None:
        input_dir = self.batch_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(self.item_paths):
            if not path.is_file():
                raise ProductBatchError(f"输入图片不存在：{path}")
            digest = sha256_file(path)
            size = path.stat().st_size
            stored = _safe_name(self.original_names[index], index, digest)
            target = input_dir / stored
            shutil.copyfile(path, target)
            if sha256_file(target) != digest or target.stat().st_size != size:
                raise ProductBatchError(f"输入图片复制校验失败：{path}")
            self.items.append({
                "index": index,
                "original_filename": self.original_names[index],
                "stored_filename": stored,
                "input_path": target.relative_to(self.batch_dir).as_posix(),
                "input_sha256": digest,
                "size_bytes": int(size),
                "status": "unprocessed",
                "image_score": None,
                "image_threshold": self.detector.image_threshold,
                "image_status": None,
                "is_anomaly": None,
                "inference_time": None,
                "error": None,
                "files": {},
            })

    def request_stop(self) -> None:
        self.stop_requested = True

    def _counts(self) -> dict:
        statuses = [item["status"] for item in self.items]
        return {
            "total": len(statuses),
            "success": statuses.count("success"),
            "failed": statuses.count("failed"),
            "unprocessed": statuses.count("unprocessed"),
            "normal": sum(1 for item in self.items
                          if item["status"] == "success" and not item["is_anomaly"]),
            "anomaly": sum(1 for item in self.items
                           if item["status"] == "success" and item["is_anomaly"]),
        }

    def _persist(self) -> None:
        payload = self.to_dict()
        try:
            _write_json(self.batch_dir / "batch.json", payload)
            self._write_csv()
        except OSError as exc:
            raise ProductBatchError(f"批次记录写入失败：{exc}") from exc

    def _write_csv(self) -> None:
        fields = [
            "index", "original_filename", "stored_filename", "input_sha256",
            "size_bytes", "status", "image_score", "image_threshold",
            "image_status", "is_anomaly", "inference_time", "error",
            "result_json", "heatmap_png", "overlay_png", "anomaly_mask_png",
        ]
        tmp = self.batch_dir / "results.csv.tmp"
        with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in self.items:
                row = {key: item.get(key) for key in fields if key in item}
                files = item.get("files") or {}
                for key in ("result_json", "heatmap_png", "overlay_png", "anomaly_mask_png"):
                    row[key] = files.get(key, "")
                writer.writerow(row)
        tmp.replace(self.batch_dir / "results.csv")

    def step(self) -> dict | None:
        if self.finished:
            return None
        if self.stop_requested:
            self._finish_remaining("用户停止，未处理")
            return None
        item = self.items[self.index]
        try:
            result = self.detector.detect(self.batch_dir / item["input_path"])
            result_dir = self.batch_dir / "results"
            result_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{item['index']:04d}_{item['input_sha256'][:8]}"
            files = result.save(result_dir, stem=stem)
            item.update({
                "status": "success",
                "image_score": result.image_score,
                "image_threshold": result.image_threshold,
                "image_status": result.image_status,
                "is_anomaly": result.is_anomaly,
                "inference_time": result.inference_time,
                "files": {key: Path(value).relative_to(self.batch_dir).as_posix()
                          for key, value in files.items()},
                "error": None,
            })
        except Exception as exc:  # noqa: BLE001
            item.update({
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
        self.index += 1
        if self.index >= len(self.items):
            self._finalize()
        else:
            self._persist()
        return item

    def _finish_remaining(self, reason: str) -> None:
        for item in self.items[self.index:]:
            if item["status"] == "unprocessed":
                item["status"] = "unprocessed"
                item["error"] = reason
        self._finalize()

    def _finalize(self) -> None:
        if self.finished:
            return
        counts = self._counts()
        if counts["unprocessed"] > 0:
            self.status = "cancelled"
        elif counts["success"] == counts["total"]:
            self.status = "completed"
        elif counts["success"] > 0:
            self.status = "partial"
        else:
            self.status = "failed"
        self.finished_at = _now()
        self._persist()

    def to_dict(self) -> dict:
        counts = self._counts()
        return {
            "schema_version": SCHEMA,
            "batch_id": self.batch_id,
            "product_id": self.product_id,
            "product_name": self.detector.product.name,
            "product_version": self.detector.version.version,
            "manifest_sha256": self.detector.version.manifest_sha256,
            "bank_version": self.detector.bank_version,
            "image_threshold": self.detector.image_threshold,
            "pixel_threshold": self.detector.pixel_threshold,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "stop_requested": self.stop_requested,
            **counts,
            "items": self.items,
        }

    def summary(self) -> dict:
        payload = self.to_dict()
        payload.pop("items", None)
        return payload


def load_product_batch(batch_id: str, root: str | Path | None = None) -> dict:
    if not _batch_id_valid(batch_id):
        raise ProductBatchError(f"非法批次 ID：{batch_id!r}")
    directory = Path(root or DEFAULT_PRODUCT_BATCH_ROOT) / batch_id
    path = directory / "batch.json"
    if not path.is_file():
        raise FileNotFoundError(f"批次不存在或缺少 batch.json：{directory}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductBatchError(f"批次记录损坏：{exc}") from exc
    if payload.get("schema_version") != SCHEMA:
        raise ProductBatchError("批次 schema 不支持")
    return payload


def list_product_batches(root: str | Path | None = None,
                         product_id: str | None = None) -> list[dict]:
    base = Path(root or DEFAULT_PRODUCT_BATCH_ROOT)
    if not base.is_dir():
        return []
    summaries: list[dict] = []
    for directory in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not directory.is_dir() or not _batch_id_valid(directory.name):
            continue
        try:
            payload = load_product_batch(directory.name, base)
        except (OSError, ProductBatchError):
            continue
        if product_id and payload.get("product_id") != product_id:
            continue
        payload.pop("items", None)
        summaries.append(payload)
    return summaries


def batch_zip(batch_id: str, root: str | Path | None = None) -> Path:
    payload = load_product_batch(batch_id, root)
    base = Path(root or DEFAULT_PRODUCT_BATCH_ROOT) / batch_id
    target = base / f"{batch_id}.zip"
    if target.is_file():
        return target
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path == target:
                continue
            archive.write(path, path.relative_to(base).as_posix())
    return target


def batch_file_bytes(batch_id: str, relative: str,
                     root: str | Path | None = None) -> bytes:
    base = Path(root or DEFAULT_PRODUCT_BATCH_ROOT) / batch_id
    target = (base / relative).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as exc:
        raise ProductBatchError("批次文件路径越界") from exc
    if not target.is_file():
        raise FileNotFoundError(target)
    return target.read_bytes()
