#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人工审核、反馈记录与受控记忆更新（P5）。

产品反馈存于产品 feedback/ 目录；更新通过 P2.1 显式 product_id 创建新版本，
使用独立验证集比较旧/新版本，只有显式激活才切换 current_version。
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

from src.core.product_detector import ProductDetector, ProductDetectorError
from src.core.product_registration import RegistrationRequest, register_product
from src.core.products import (
    ProductStateConflictError,
    ProductStore,
    sha256_file,
)

SCHEMA = "microdefect.product_feedback/1"
UPDATE_SCHEMA = "microdefect.product_update/1"
ACTIVATION_SCHEMA = "microdefect.product_update_activation/1"
CONCLUSIONS = ("normal", "anomaly", "environment", "reject")
STATUSES = ("pending", "confirmed", "rejected", "applied", "validation_failed")
MIN_VALIDATION_PER_CLASS = 5


class FeedbackError(RuntimeError):
    """反馈/更新不可用。"""


class FeedbackConflictError(FeedbackError):
    def __init__(self, feedback_id: str) -> None:
        self.feedback_id = feedback_id
        super().__init__(f"相同来源与图片的反馈已存在：{feedback_id}")


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


def _update_dir(product_dir: Path, update_id: str) -> Path:
    return _updates_dir(product_dir) / update_id


def _report_path(product_dir: Path, update_id: str) -> Path:
    return _update_dir(product_dir, update_id) / "report.json"


def _report_sha_path(product_dir: Path, update_id: str) -> Path:
    return _update_dir(product_dir, update_id) / "report.sha256"


def _integrity_path(product_dir: Path) -> Path:
    return _updates_dir(product_dir) / "integrity.json"


def _write_report(product_dir: Path, update_id: str, report: dict) -> None:
    update_dir = _update_dir(product_dir, update_id)
    data = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write(update_dir / "report.json", data)
    digest = hashlib.sha256(data).hexdigest()
    _atomic_write(update_dir / "report.sha256", (digest + "\n").encode("ascii"))
    integrity_path = _integrity_path(product_dir)
    try:
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        integrity = {}
    integrity[update_id] = digest
    _write_json(integrity_path, integrity)


def _activation_path(product_dir: Path, update_id: str) -> Path:
    return _update_dir(product_dir, update_id) / "activation.json"


def _write_activation(product_dir: Path, update_id: str, state: str,
                      *, old_version: int, new_version: int, error: str = "") -> dict:
    payload = {
        "schema_version": ACTIVATION_SCHEMA,
        "update_id": update_id,
        "state": state,
        "old_version": old_version,
        "new_version": new_version,
        "updated_at": _now(),
        "error": error,
    }
    _write_json(_activation_path(product_dir, update_id), payload)
    return payload


def _read_activation(product_dir: Path, update_id: str) -> dict | None:
    path = _activation_path(product_dir, update_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _fix_validation(update_dir: Path, normal: list[Path], anomaly: list[Path]) -> tuple[list[Path], list[Path], list[dict]]:
    """复制验证图到 update 专属目录，并返回固定副本及 SHA 元数据。"""
    records = []
    fixed_normal = []
    fixed_anomaly = []
    target_dir = update_dir / "validation"
    for label, paths, target_list in (
        ("normal", normal, fixed_normal), ("anomaly", anomaly, fixed_anomaly),
    ):
        for index, path in enumerate(paths):
            path = Path(path)
            if not path.is_file():
                raise FeedbackError(f"验证图片不存在：{path}")
            digest = sha256_file(path)
            target = target_dir / label / f"{index:04d}_{digest[:8]}{path.suffix.lower()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if sha256_file(target) != digest:
                raise FeedbackError(f"验证图片固定副本 SHA 不一致：{path}")
            target_list.append(target)
            records.append({
                "label": label,
                "sha256": digest,
                "rel_path": target.relative_to(update_dir).as_posix(),
            })
    return fixed_normal, fixed_anomaly, records


def _idempotency_path(product_dir: Path) -> Path:
    return _updates_dir(product_dir) / "idempotency.json"


def _idempotency_key(product_id: str, current_version: int,
                     feedback_ids: list[str], validation_records: list[dict]) -> str:
    payload = {
        "product_id": product_id,
        "current_version": int(current_version),
        "feedback_ids": sorted(feedback_ids),
        "validation": sorted(
            [[item["label"], item["sha256"]] for item in validation_records]
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_idempotency(product_dir: Path) -> dict:
    path = _idempotency_path(product_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_idempotency(product_dir: Path, payload: dict) -> None:
    _write_json(_idempotency_path(product_dir), payload)


def _existing_update(store: ProductStore, product_id: str, key: str) -> dict | None:
    product_dir = store.product_directory(product_id)
    index = _load_idempotency(product_dir)
    update_id = index.get(key)
    if not update_id:
        return None
    try:
        return load_update(store, product_id, update_id)
    except Exception:  # noqa: BLE001
        index.pop(key, None)
        _write_idempotency(product_dir, index)
        return None


def _verify_fixed_validation(update_dir: Path, records: list[dict]) -> None:
    for item in records:
        path = update_dir / item["rel_path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise FeedbackError(f"固定验证图片缺失或被修改：{item['rel_path']}")


def _feedback_dir(product_dir: Path) -> Path:
    return product_dir / "feedback"


def _records_dir(product_dir: Path) -> Path:
    return _feedback_dir(product_dir) / "records"


def _images_dir(product_dir: Path) -> Path:
    return _feedback_dir(product_dir) / "images"


def _updates_dir(product_dir: Path) -> Path:
    return _feedback_dir(product_dir) / "updates"


def _load_feedback(product_dir: Path, feedback_id: str) -> tuple[Path, dict]:
    if not re.fullmatch(r"[0-9a-f]{32}", feedback_id or ""):
        raise FeedbackError(f"非法反馈 ID：{feedback_id!r}")
    path = _records_dir(product_dir) / f"{feedback_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def create_feedback(
    store: ProductStore,
    *,
    product_id: str,
    product_version: int,
    source_type: str,
    source_id: str | None,
    image_path: str | Path,
    image_sha256: str,
    image_score: float | None = None,
    image_threshold: float | None = None,
    image_status: str | None = None,
    drift_level: str | None = None,
    drift_reasons: list[str] | None = None,
    original_filename: str | None = None,
) -> dict:
    if source_type not in ("detection", "batch", "drift"):
        raise FeedbackError(f"非法来源类型：{source_type!r}")
    path = Path(image_path)
    if not path.is_file():
        raise FeedbackError(f"来源图片不存在：{path}")
    digest = sha256_file(path)
    if digest != image_sha256:
        raise FeedbackError("来源图片 SHA256 不一致")
    product = store.get_product(product_id)
    version = store.get_version(product_id, product_version)
    product_dir = store.product_directory(product_id)
    with store.update_lock(product_id):
        records = _records_dir(product_dir)
        records.mkdir(parents=True, exist_ok=True)
        for existing_path in records.glob("*.json"):
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if (existing.get("source_type") == source_type
                    and existing.get("source_id") == source_id
                    and existing.get("image_sha256") == digest):
                raise FeedbackConflictError(existing["feedback_id"])
        feedback_id = uuid.uuid4().hex
        suffix = path.suffix.lower() or ".png"
        image_target = _images_dir(product_dir) / f"{feedback_id}_{digest[:8]}{suffix}"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, image_target)
        if sha256_file(image_target) != digest:
            raise FeedbackError("反馈图片复制校验失败")
        payload = {
            "schema_version": SCHEMA,
            "feedback_id": feedback_id,
            "product_id": product_id,
            "product_version": product_version,
            "source_type": source_type,
            "source_id": source_id,
            "image_sha256": digest,
            "image_path": image_target.relative_to(product_dir).as_posix(),
            "original_filename": original_filename or path.name,
            "image_score": image_score,
            "image_threshold": image_threshold,
            "image_status": image_status,
            "drift_level": drift_level,
            "drift_reasons": list(drift_reasons or []),
            "created_at": _now(),
            "status": "pending",
            "conclusion": None,
            "review_note": "",
            "reviewed_at": None,
            "reviewed_product_version": None,
            "version_manifest_sha256": version.manifest_sha256,
        }
        _write_json(records / f"{feedback_id}.json", payload)
        return payload


def list_feedback(store: ProductStore, product_id: str,
                  *, status: str | None = None) -> list[dict]:
    product_dir = store.product_directory(product_id)
    directory = _records_dir(product_dir)
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status and payload.get("status") != status:
            continue
        records.append(payload)
    return records


def review_feedback(store: ProductStore, product_id: str, feedback_id: str,
                    conclusion: str, note: str = "") -> dict:
    if conclusion not in CONCLUSIONS:
        raise FeedbackError(f"非法审核结论：{conclusion!r}")
    product_dir = store.product_directory(product_id)
    with store.update_lock(product_id):
        path, payload = _load_feedback(product_dir, feedback_id)
        if payload.get("status") != "pending":
            raise FeedbackError("该反馈已完成审核，不能重复审核")
        status = "confirmed" if conclusion in ("normal", "anomaly") else "rejected"
        payload.update({
            "status": status,
            "conclusion": conclusion,
            "review_note": str(note or ""),
            "reviewed_at": _now(),
            "reviewed_product_version": store.get_product(product_id).current_version,
        })
        _write_json(path, payload)
        return payload


def feedback_image_path(store: ProductStore, product_id: str, feedback: dict) -> Path:
    return store.product_directory(product_id) / feedback["image_path"]


def _version_source_paths(store: ProductStore, product_id: str,
                          version_number: int) -> dict[str, list[Path]]:
    version = store.get_version(product_id, version_number)
    if version.source_status != "persisted":
        raise FeedbackError("产品来源样本未持久化，不能进行记忆更新")
    vdir = store.version_directory(product_id, version_number)
    roles: dict[str, list[Path]] = {
        role: [] for role in ("reference", "calibration", "defect", "test_normal", "test_defect")
    }
    for item in version.source_samples:
        role = item.get("role")
        if role not in roles:
            continue
        path = vdir / item["rel_path"]
        if not path.is_file():
            raise FeedbackError(f"来源样本缺失：{item['rel_path']}")
        if item.get("sha256") and sha256_file(path) != item["sha256"]:
            raise FeedbackError(f"来源样本哈希不匹配：{item['rel_path']}")
        roles[role].append(path)
    if not roles["reference"] or not roles["calibration"]:
        raise FeedbackError("参考/校准来源样本不完整")
    return roles


def _dedupe_validation(store: ProductStore, product_id: str, version_number: int,
                       normal: list[Path], anomaly: list[Path],
                       update_hashes: set[str]) -> tuple[list[Path], list[Path], set[str]]:
    source_hashes = set(update_hashes)
    roles = _version_source_paths(store, product_id, version_number)
    for paths in roles.values():
        source_hashes.update(sha256_file(p) for p in paths)
    normal_hashes = [sha256_file(p) for p in normal]
    anomaly_hashes = [sha256_file(p) for p in anomaly]
    all_hashes = normal_hashes + anomaly_hashes
    if len(set(all_hashes)) != len(all_hashes):
        raise FeedbackError("验证集内部存在重复图片")
    if set(all_hashes) & source_hashes:
        raise FeedbackError("验证集与参考/校准/更新样本存在重复 SHA256")
    return normal, anomaly, set(all_hashes)


def _auroc(normal: list[float], anomaly: list[float]) -> float:
    wins = 0.0
    total = len(normal) * len(anomaly)
    for n in normal:
        for a in anomaly:
            if a > n:
                wins += 1.0
            elif a == n:
                wins += 0.5
    return wins / total if total else 0.0


def _average_precision(normal: list[float], anomaly: list[float]) -> float:
    values = [(s, 0) for s in normal] + [(s, 1) for s in anomaly]
    values.sort(key=lambda item: item[0], reverse=True)
    hits = 0
    total_pos = len(anomaly)
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(values, 1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total_pos if total_pos else 0.0


def _metrics(normal: list[float], anomaly: list[float], threshold: float) -> dict:
    fp = sum(1 for score in normal if score > threshold)
    fn = sum(1 for score in anomaly if score <= threshold)
    tp = len(anomaly) - fn
    tn = len(normal) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "normal_count": len(normal),
        "anomaly_count": len(anomaly),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "auroc": _auroc(normal, anomaly),
        "average_precision": _average_precision(normal, anomaly),
        "f1": f1,
        "threshold": threshold,
    }


def _evaluate_version(store: ProductStore, product_id: str, version_number: int,
                      normal_paths: list[Path], anomaly_paths: list[Path]) -> dict:
    detector = ProductDetector(product_id, store=store, version=version_number, device="auto")
    normal_scores = [detector.detect(p).image_score for p in normal_paths]
    anomaly_scores = [detector.detect(p).image_score for p in anomaly_paths]
    return _metrics(normal_scores, anomaly_scores, detector.image_threshold)


def prepare_update(
    store: ProductStore,
    product_id: str,
    *,
    feedback_ids: list[str],
    validation_normal_paths: list[Path],
    validation_anomaly_paths: list[Path],
    note: str = "",
) -> dict:
    product = store.get_product(product_id)
    current = product.current_version
    if current is None:
        raise FeedbackError("产品没有 current_version")
    with store.update_lock(product_id):
        current = store.get_product(product_id).current_version
        roles = _version_source_paths(store, product_id, current)
        selected = []
        for feedback_id in feedback_ids:
            _path, payload = _load_feedback(store.product_directory(product_id), feedback_id)
            if payload["product_id"] != product_id or payload["product_version"] != current:
                raise FeedbackError("反馈不属于当前产品版本")
            if payload["status"] != "confirmed":
                raise FeedbackError("只有 confirmed 反馈可参与更新")
            if payload["conclusion"] not in ("normal", "anomaly"):
                raise FeedbackError("环境变化/拒绝反馈不能参与更新")
            selected.append(payload)
        if not selected:
            raise FeedbackError("没有可用的已确认反馈")
        update_hashes = {item["image_sha256"] for item in selected}
        if len(update_hashes) != len(selected):
            raise FeedbackError("更新候选存在重复图片")
        all_source_hashes = set()
        for paths in roles.values():
            all_source_hashes.update(sha256_file(p) for p in paths)
        if update_hashes & all_source_hashes:
            raise FeedbackError("更新候选与现有来源样本重复")
        normal_paths = list(roles["reference"])
        anomaly_paths = list(roles["defect"])
        for item in selected:
            image = store.product_directory(product_id) / item["image_path"]
            if not image.is_file() or sha256_file(image) != item["image_sha256"]:
                raise FeedbackError("反馈图片缺失或哈希不匹配")
            if item["conclusion"] == "normal":
                normal_paths.append(image)
            else:
                anomaly_paths.append(image)
        original_names = {
            str(p): p.name for p in normal_paths + list(roles["calibration"])
            + anomaly_paths + list(roles["test_normal"]) + list(roles["test_defect"])
        }
        request = RegistrationRequest(
            name=product.name, category=product.category, product_id=product_id,
            reference_paths=normal_paths,
            calibration_paths=list(roles["calibration"]),
            calibration_count=len(roles["calibration"]),
            defect_paths=anomaly_paths,
            test_normal_paths=list(roles["test_normal"]),
            test_defect_paths=list(roles["test_defect"]),
            source_original_names=original_names,
            activate=False,
        )
        update_id = uuid.uuid4().hex
        product_dir = store.product_directory(product_id)
        update_dir = _updates_dir(product_dir) / update_id
        fixed_normal, fixed_anomaly, validation_files = _fix_validation(
            update_dir, list(validation_normal_paths), list(validation_anomaly_paths)
        )
        _dedupe_validation(store, product_id, current,
                           fixed_normal, fixed_anomaly, update_hashes)
        key = _idempotency_key(product_id, current, feedback_ids, validation_files)
        existing = _existing_update(store, product_id, key)
        if existing is not None:
            shutil.rmtree(update_dir, ignore_errors=True)
            return existing
        index = _load_idempotency(product_dir)
        index[key] = update_id
        _write_idempotency(product_dir, index)
        try:
            result = register_product(store, request, device="auto")
            new_version = result.version.version
            _verify_fixed_validation(update_dir, validation_files)
            old_metrics = _evaluate_version(store, product_id, current,
                                            fixed_normal, fixed_anomaly)
            new_metrics = _evaluate_version(store, product_id, new_version,
                                            fixed_normal, fixed_anomaly)
            _verify_fixed_validation(update_dir, validation_files)
            enough = (len(fixed_normal) >= MIN_VALIDATION_PER_CLASS
                      and len(fixed_anomaly) >= MIN_VALIDATION_PER_CLASS)
            if not enough:
                verdict = "inconclusive"
                reasons = ["验证集数量不足"]
            elif (new_metrics["false_positives"] <= old_metrics["false_positives"]
                  and new_metrics["false_negatives"] <= old_metrics["false_negatives"]
                  and new_metrics["auroc"] >= old_metrics["auroc"] - 1e-12):
                verdict = "pass"
                reasons = ["新版本未增加误报/漏报，AUROC 未下降"]
            else:
                verdict = "fail"
                reasons = ["新版本误报/漏报增加或 AUROC 下降"]
            report = {
                "schema_version": UPDATE_SCHEMA,
                "update_id": update_id,
                "product_id": product_id,
                "old_version": current,
                "new_version": new_version,
                "feedback_ids": feedback_ids,
                "update_samples": [
                    {"feedback_id": item["feedback_id"], "sha256": item["image_sha256"]}
                    for item in selected
                ],
                "update_sha256": sorted(update_hashes),
                "validation_normal_sha256": [
                    item["sha256"] for item in validation_files if item["label"] == "normal"
                ],
                "validation_anomaly_sha256": [
                    item["sha256"] for item in validation_files if item["label"] == "anomaly"
                ],
                "validation_files": validation_files,
                "validation_min_per_class": MIN_VALIDATION_PER_CLASS,
                "old_metrics": old_metrics,
                "new_metrics": new_metrics,
                "verdict": verdict,
                "reasons": reasons,
                "activated": False,
                "created_at": _now(),
                "activated_at": None,
                "note": str(note or ""),
                "new_manifest_sha256": result.version.manifest_sha256,
            }
            _write_report(product_dir, update_id, report)
            if verdict != "pass":
                for feedback_id in feedback_ids:
                    path, payload = _load_feedback(product_dir, feedback_id)
                    payload["status"] = "validation_failed"
                    payload["update_id"] = update_id
                    _write_json(path, payload)
            return report
        except Exception:
            current_index = _load_idempotency(product_dir)
            if current_index.get(key) == update_id:
                current_index.pop(key, None)
                _write_idempotency(product_dir, current_index)
            raise


def list_updates(store: ProductStore, product_id: str) -> list[dict]:
    recover_pending_updates(store, product_id)
    directory = _updates_dir(store.product_directory(product_id))
    if not directory.is_dir():
        return []
    updates = []
    for path in sorted(directory.glob("*/report.json"), key=lambda p: p.stat().st_mtime,
                       reverse=True):
        try:
            updates.append(load_update(store, product_id, path.parent.name))
        except (OSError, FeedbackError):
            continue
    return updates


def load_update(store: ProductStore, product_id: str, update_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", update_id or ""):
        raise FeedbackError("非法更新 ID")
    product_dir = store.product_directory(product_id)
    update_dir = _update_dir(product_dir, update_id)
    report_path = update_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    data = report_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    sha_path = update_dir / "report.sha256"
    if not sha_path.is_file() or sha_path.read_text(encoding="ascii").strip() != digest:
        raise FeedbackError("更新报告完整性摘要不匹配")
    try:
        integrity = json.loads(_integrity_path(product_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeedbackError("缺少更新完整性索引") from exc
    if integrity.get(update_id) != digest:
        raise FeedbackError("更新报告与完整性索引不匹配")
    try:
        report = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise FeedbackError("更新报告 JSON 损坏") from exc
    if report.get("schema_version") != UPDATE_SCHEMA:
        raise FeedbackError("更新报告 schema 不支持")
    if report.get("update_id") != update_id or report.get("product_id") != product_id:
        raise FeedbackError("更新报告与产品/目录不匹配")
    try:
        old_version = int(report["old_version"])
        new_version = int(report["new_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeedbackError("更新报告版本字段非法") from exc
    try:
        old = store.get_version(product_id, old_version)
        new = store.get_version(product_id, new_version)
    except Exception as exc:  # noqa: BLE001
        raise FeedbackError(f"更新报告版本不存在：{exc}") from exc
    if new.manifest_sha256 != report.get("new_manifest_sha256"):
        raise FeedbackError("更新报告的新版本 manifest 不匹配")
    feedback_ids = report.get("feedback_ids") or []
    if len(set(feedback_ids)) != len(feedback_ids):
        raise FeedbackError("更新报告 feedback_ids 重复")
    update_samples = report.get("update_samples") or []
    if len(update_samples) != len(feedback_ids):
        raise FeedbackError("更新报告样本 SHA 与反馈数量不一致")
    by_id = {item.get("feedback_id"): item.get("sha256") for item in update_samples}
    allowed_statuses = {"confirmed", "applied", "validation_failed"}
    for feedback_id in feedback_ids:
        try:
            _path, feedback = _load_feedback(product_dir, feedback_id)
        except (OSError, json.JSONDecodeError) as exc:
            raise FeedbackError(f"更新报告引用的反馈不存在：{feedback_id}") from exc
        if feedback.get("product_id") != product_id:
            raise FeedbackError("更新报告引用的反馈属于其他产品")
        if feedback.get("status") not in allowed_statuses:
            raise FeedbackError("更新报告引用的反馈状态非法")
        if feedback.get("image_sha256") != by_id.get(feedback_id):
            raise FeedbackError("更新报告引用的反馈 SHA 不匹配")
    validation = report.get("validation_files") or []
    if not validation:
        raise FeedbackError("更新报告缺少固定验证集元数据")
    _verify_fixed_validation(update_dir, validation)
    normal_count = sum(1 for item in validation if item.get("label") == "normal")
    anomaly_count = sum(1 for item in validation if item.get("label") == "anomaly")
    old_metrics = report.get("old_metrics") or {}
    new_metrics = report.get("new_metrics") or {}
    expected_verdict = "inconclusive"
    if normal_count >= MIN_VALIDATION_PER_CLASS and anomaly_count >= MIN_VALIDATION_PER_CLASS:
        if (new_metrics.get("false_positives", 10**9) <= old_metrics.get("false_positives", -1)
                and new_metrics.get("false_negatives", 10**9) <= old_metrics.get("false_negatives", -1)
                and new_metrics.get("auroc", -1) >= old_metrics.get("auroc", 1) - 1e-12):
            expected_verdict = "pass"
        else:
            expected_verdict = "fail"
    if report.get("verdict") != expected_verdict:
        raise FeedbackError("更新报告 verdict 与指标不一致")
    if old_version not in (store.get_product(product_id).current_version,):
        # 验证基于旧 current；激活时另做 current 检查，这里只保证旧版本存在
        pass
    return report


def _complete_activation_locked(store: ProductStore, product_id: str,
                                 report: dict, update_id: str) -> dict:
    product_dir = store.product_directory(product_id)
    report["activated"] = True
    report["activated_at"] = _now()
    _write_report(product_dir, update_id, report)
    for feedback_id in report["feedback_ids"]:
        path, payload = _load_feedback(product_dir, feedback_id)
        payload["status"] = "applied"
        payload["applied_update_id"] = update_id
        payload["applied_at"] = _now()
        _write_json(path, payload)
    _write_activation(product_dir, update_id, "committed",
                      old_version=int(report["old_version"]),
                      new_version=int(report["new_version"]))
    return report


def _recover_update_locked(store: ProductStore, product_id: str,
                           update_id: str) -> dict:
    product_dir = store.product_directory(product_id)
    activation = _read_activation(product_dir, update_id)
    if activation is None or activation.get("state") == "committed":
        return load_update(store, product_id, update_id)
    report = load_update(store, product_id, update_id)
    current = store.get_product(product_id).current_version
    old = int(report["old_version"])
    new = int(report["new_version"])
    if current == new:
        return _complete_activation_locked(store, product_id, report, update_id)
    if current == old:
        _write_activation(product_dir, update_id, "failed",
                          old_version=old, new_version=new,
                          error="current 未切换；激活未完成")
        return report
    _write_activation(product_dir, update_id, "conflict",
                      old_version=old, new_version=new,
                      error=f"current={current} 与事务记录不匹配")
    raise FeedbackError("激活事务与当前版本不一致，需人工核查")


def recover_pending_updates(store: ProductStore, product_id: str) -> list[dict]:
    product_dir = store.product_directory(product_id)
    if not _updates_dir(product_dir).is_dir():
        return []
    recovered = []
    with store.update_lock(product_id):
        for path in sorted(_updates_dir(product_dir).glob("*/activation.json")):
            update_id = path.parent.name
            activation = _read_activation(product_dir, update_id)
            if activation is None or activation.get("state") == "committed":
                continue
            try:
                recovered.append(_recover_update_locked(store, product_id, update_id))
            except Exception:
                continue
    return recovered


def activate_update(store: ProductStore, product_id: str, update_id: str) -> dict:
    with store.update_lock(product_id):
        product_dir = store.product_directory(product_id)
        activation = _read_activation(product_dir, update_id)
        if activation is not None and activation.get("state") != "committed":
            report = _recover_update_locked(store, product_id, update_id)
            if report.get("activated"):
                return report
        report = load_update(store, product_id, update_id)
        if report.get("verdict") != "pass":
            raise FeedbackError("验证未通过，不能激活")
        if report.get("activated"):
            if store.get_product(product_id).current_version == report["new_version"]:
                return report
            raise FeedbackError("更新已标记激活但 current 不一致")
        if store.get_product(product_id).current_version != report["old_version"]:
            raise FeedbackError("当前版本已变化，拒绝激活旧更新")
        _verify_fixed_validation(_update_dir(product_dir, update_id),
                                 report.get("validation_files") or [])
        for feedback_id in report["feedback_ids"]:
            _p, payload = _load_feedback(product_dir, feedback_id)
            if payload.get("status") not in ("confirmed", "applied"):
                raise FeedbackError("更新反馈状态不再是 confirmed，拒绝激活")
        old = int(report["old_version"])
        new = int(report["new_version"])
        _write_activation(product_dir, update_id, "prepared",
                          old_version=old, new_version=new)
        try:
            store.compare_and_set_current(product_id, old, new)
        except ProductStateConflictError as exc:
            _write_activation(product_dir, update_id, "failed",
                              old_version=old, new_version=new, error=str(exc))
            raise FeedbackError(f"current_version 并发变化，拒绝激活：{exc}") from exc
        _write_activation(product_dir, update_id, "switched",
                          old_version=old, new_version=new)
        try:
            return _complete_activation_locked(store, product_id, report, update_id)
        except Exception:
            # current 已切换；保留事务日志，下一次加载/激活时恢复并完成记录。
            _recover_update_locked(store, product_id, update_id)
            raise
