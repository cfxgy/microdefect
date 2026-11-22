#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""产品与产品版本的数据模型、目录隔离、完整性校验与原子持久化（P1）。

本阶段只负责“产品元数据 + 版本目录 + manifest + 状态机”，**不生成特征记忆库、
不做特征提取、不做阈值校准**（这些属于 P2）。

目录约定::

    <root>/                         # 默认 D:\\microdefect\\memory\\products
      .staging/                     # 注册/写入中的临时目录，永远不是可用产品
      .archive/                     # 预留归档区（本模块不物理移动产品）
      <product_id>/                 # UUID4 hex，32 位
        product.json
        feedback/records.jsonl      # 预留：人工反馈
        performance/history.jsonl   # 预留：历史性能
        tests/                      # 预留：独立测试记录
        versions/
          v001/
            version.json
            manifest.json
            manifest.sha256
            reference/.keep         # P2 写入真实记忆库
            calibration/.keep       # P2 写入校准素材
            tests/.keep

设计要点
--------
* product_id 只接受 32 位小写 hex UUID，目录名与之相同；名称不参与路径。
* 版本号只接受 1..999 的整数，目录名为 ``vNNN``；新版本永不覆盖旧版本。
* 写入先落到 ``.staging``，完成 manifest 后原子 rename 到正式位置；
  失败时保留 staging 供人工排查，不生成 active 产品。
* ``manifest.json`` 覆盖除自身与 ``manifest.sha256`` 外的全部文件；
  ``manifest.sha256`` 是 manifest 文件的 SHA256。
* 任何读取都校验 product_id、版本、相对路径、文件哈希与 manifest 一致性。
* 不提供物理删除；停用/归档只是状态变化。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any

if os.name == "nt":  # pragma: no cover - Windows 运行环境
    import msvcrt
else:  # pragma: no cover - POSIX 兼容
    import fcntl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "memory" / "products"

PRODUCT_SCHEMA = "microdefect.product/1"
VERSION_SCHEMA = "microdefect.product_version/1"
MANIFEST_SCHEMA = "microdefect.product_manifest/1"

PRODUCT_STATUSES = ("draft", "validating", "active", "disabled", "archived")
VERSION_STATUSES = ("validating", "committed", "active", "superseded", "archived")
PRODUCT_TRANSITIONS = {
    "draft": {"validating", "disabled", "archived"},
    "validating": {"draft", "active", "disabled"},
    "active": {"disabled", "archived"},
    "disabled": {"active", "archived"},
    "archived": set(),
}
MAX_VERSION = 999

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SAFE_NAME_MAX = 120


class ProductStoreError(RuntimeError):
    """产品存储错误基类。"""


class ProductNotFoundError(ProductStoreError):
    pass


class ProductConflictError(ProductStoreError):
    pass


class ProductIntegrityError(ProductStoreError):
    pass


class ProductPathError(ProductStoreError):
    pass


class ProductStateError(ProductStoreError):
    pass


class ProductLockTimeoutError(ProductStoreError):
    """等待产品级跨进程锁超时。"""

    def __init__(self, product_id: str, lock_path: Path, timeout: float) -> None:
        super().__init__(
            f"等待产品 {product_id} 的写锁超时（{timeout:.2f}s）：{lock_path}；"
            "另一个进程正在写入该产品，请稍后重试。"
        )
        self.product_id = product_id
        self.lock_path = lock_path
        self.timeout = timeout


class ProductStateConflictError(ProductStoreError):
    """current_version 与预期不一致（并发切换）。"""


class _ProductFileLock:
    """基于操作系统文件锁的跨进程互斥（Windows: msvcrt，POSIX: fcntl）。

    * 锁文件位于 ``<root>/.locks/<product_id>.lock``，不属于任何版本目录，
      因此不会被写入 manifest；
    * 进程异常退出时由操作系统自动释放锁；
    * 不删除锁文件（删除会破坏其他等待者的锁语义）；
    * 等待超时抛 :class:`ProductLockTimeoutError`。
    """

    def __init__(self, path: Path, timeout: float, product_id: str) -> None:
        self.path = path
        self.timeout = timeout
        self.product_id = product_id
        self.fd: int | None = None

    def __enter__(self) -> "_ProductFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.lseek(fd, 0, os.SEEK_END)
            if os.lseek(fd, 0, os.SEEK_CUR) == 0:
                os.write(fd, b"\0")
        except OSError:
            pass
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._acquire(fd)
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise ProductLockTimeoutError(self.product_id, self.path, self.timeout)
                time.sleep(0.05)
                continue
            self.fd = fd
            return self

    def _acquire(self, fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        if os.name == "nt":
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is None:
            return
        try:
            os.lseek(self.fd, 0, os.SEEK_SET)
            if os.name == "nt":
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None


def _serialized(method):
    """按第一个参数 product_id 串行化写方法。"""

    @wraps(method)
    def wrapper(self, product_id, *args, **kwargs):
        with self._product_lock(product_id):
            return method(self, product_id, *args, **kwargs)

    return wrapper


def _serialized_create(method):
    """create_product 的首参是名称，这里先生成/校验 product_id 再加锁。"""

    @wraps(method)
    def wrapper(self, name, category="", *, metadata=None,
                feature_extractor_version="", product_id=None):
        pid = product_id or uuid.uuid4().hex
        validate_product_id(pid)
        with self._product_lock(pid):
            return method(
                self, name, category, metadata=metadata,
                feature_extractor_version=feature_extractor_version, product_id=pid,
            )

    return wrapper


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_product_id(product_id: str) -> str:
    if not isinstance(product_id, str) or not _ID_RE.fullmatch(product_id):
        raise ProductPathError(
            f"非法产品 ID：{product_id!r}（必须是 32 位小写 hex UUID，不含路径字符）"
        )
    return product_id


def version_dirname(version: int) -> str:
    if not isinstance(version, int) or isinstance(version, bool) or version < 1 or version > MAX_VERSION:
        raise ProductPathError(f"非法版本号：{version!r}（允许 1..{MAX_VERSION}）")
    return f"v{version:03d}"


def _safe_relative(base: Path, relative: str, *, field_name: str) -> Path:
    """把相对路径限制在 base 内，拒绝绝对路径、.. 与盘符。"""
    if not isinstance(relative, str) or not relative.strip():
        raise ProductPathError(f"{field_name} 不能为空")
    if "\\" in relative or ":" in relative:
        raise ProductPathError(f"{field_name} 含非法字符：{relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ProductPathError(f"{field_name} 必须是安全的相对路径：{relative!r}")
    target = (base / pure).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError as exc:
        raise ProductPathError(f"{field_name} 越出版本目录：{relative!r}") from exc
    return target


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )


def _read_json(path: Path, *, what: str) -> dict:
    if not path.is_file():
        raise ProductNotFoundError(f"缺少{what}：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductIntegrityError(f"{what}损坏：{path}（{exc}）") from exc
    if not isinstance(payload, dict):
        raise ProductIntegrityError(f"{what}格式错误：{path}")
    return payload


@dataclass
class ProductRecord:
    product_id: str
    name: str
    category: str
    status: str
    created_at: str
    updated_at: str
    current_version: int | None
    feature_extractor_version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = PRODUCT_SCHEMA

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_version": self.current_version,
            "feature_extractor_version": self.feature_extractor_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ProductRecord":
        return cls(
            product_id=str(payload["product_id"]),
            name=str(payload["name"]),
            category=str(payload.get("category", "")),
            status=str(payload["status"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            current_version=payload.get("current_version"),
            feature_extractor_version=str(payload.get("feature_extractor_version", "")),
            metadata=dict(payload.get("metadata") or {}),
            schema_version=str(payload.get("schema_version", PRODUCT_SCHEMA)),
        )


@dataclass
class ProductVersion:
    product_id: str
    version: int
    status: str
    created_at: str
    feature_extractor_version: str
    weights_sha256: str
    reference_bank_path: str
    calibration_path: str
    defect_bank_path: str | None
    threshold: float | None
    calibration_metadata: dict[str, Any]
    manifest_sha256: str
    source_samples: list[dict] = field(default_factory=list)
    source_status: str = "legacy"
    registration_fingerprint: str = ""
    test_records: list[dict] = field(default_factory=list)
    schema_version: str = VERSION_SCHEMA

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "product_id": self.product_id,
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at,
            "feature_extractor_version": self.feature_extractor_version,
            "weights_sha256": self.weights_sha256,
            "reference_bank_path": self.reference_bank_path,
            "calibration_path": self.calibration_path,
            "defect_bank_path": self.defect_bank_path,
            "threshold": self.threshold,
            "calibration_metadata": self.calibration_metadata,
            "manifest_sha256": self.manifest_sha256,
            "source_samples": self.source_samples,
            "source_status": self.source_status,
            "registration_fingerprint": self.registration_fingerprint,
            "test_records": self.test_records,
        }

    def storage_dict(self) -> dict:
        """写入 version.json 的内容（不含由 manifest 推导的 manifest_sha256）。"""
        payload = self.to_dict()
        payload.pop("manifest_sha256", None)
        return payload


class ProductStore:
    """产品与版本的安全存储。"""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        create_root: bool = True,
        lock_timeout: float = 10.0,
    ) -> None:
        if not isinstance(lock_timeout, (int, float)) or lock_timeout <= 0:
            raise ProductStateError(f"lock_timeout 必须为正数：{lock_timeout!r}")
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.root = self.root.resolve()
        if create_root:
            self.root.mkdir(parents=True, exist_ok=True)
        self.staging_root = self.root / ".staging"
        self.lock_root = self.root / ".locks"
        self.lock_timeout = float(lock_timeout)

    @contextmanager
    def _product_lock(self, product_id: str):
        validate_product_id(product_id)
        lock_path = self.lock_root / f"{product_id}.lock"
        with _ProductFileLock(lock_path, self.lock_timeout, product_id):
            yield

    @contextmanager
    def update_lock(self, product_id: str):
        """产品反馈/更新事务锁（与 P1.1 相同的文件锁实现，独立锁文件）。"""
        validate_product_id(product_id)
        lock_path = self.lock_root / f"update-{product_id}.lock"
        with _ProductFileLock(lock_path, self.lock_timeout, f"update:{product_id}"):
            yield

    @contextmanager
    def registration_lock(self, fingerprint: str):
        """按注册指纹的跨进程锁（与 P1.1 使用同一 _ProductFileLock 实现）。

        锁文件：``<root>/.locks/registration-<fingerprint>.lock``；
        只串行化相同指纹，不同指纹可并行。锁超时抛 ProductLockTimeoutError。
        """
        if not re.fullmatch(r"[0-9a-f]{64}", str(fingerprint or "")):
            raise ProductStateError("注册指纹必须是 64 位小写 hex")
        lock_path = self.lock_root / f"registration-{fingerprint}.lock"
        with _ProductFileLock(lock_path, self.lock_timeout, f"registration:{fingerprint[:16]}"):
            yield

    # ------------------------------------------------------------------
    # 内部路径
    # ------------------------------------------------------------------
    def _product_dir(self, product_id: str) -> Path:
        validate_product_id(product_id)
        return self.root / product_id

    def _versions_dir(self, product_id: str) -> Path:
        return self._product_dir(product_id) / "versions"

    def _version_dir(self, product_id: str, version: int) -> Path:
        return self._versions_dir(product_id) / version_dirname(version)

    def _require_product(self, product_id: str) -> ProductRecord:
        record = self.get_product(product_id)
        return record

    # ------------------------------------------------------------------
    # 创建 / 读取产品
    # ------------------------------------------------------------------
    @_serialized_create
    def create_product(
        self,
        name: str,
        category: str = "",
        *,
        metadata: dict | None = None,
        feature_extractor_version: str = "",
        product_id: str | None = None,
    ) -> ProductRecord:
        clean_name = (name or "").strip()
        if not clean_name or len(clean_name) > _SAFE_NAME_MAX:
            raise ProductStateError(f"产品名称非法（1..{_SAFE_NAME_MAX} 字符）：{name!r}")
        pid = product_id or uuid.uuid4().hex
        validate_product_id(pid)
        final_dir = self._product_dir(pid)
        if final_dir.exists():
            raise ProductConflictError(f"产品已存在，拒绝覆盖：{pid}")

        created = _now()
        record = ProductRecord(
            product_id=pid,
            name=clean_name,
            category=(category or "").strip(),
            status="draft",
            created_at=created,
            updated_at=created,
            current_version=None,
            feature_extractor_version=feature_extractor_version,
            metadata=dict(metadata or {}),
        )

        staging = self._make_staging(f"{pid}-{uuid.uuid4().hex[:8]}")
        try:
            _atomic_write_json(staging / "product.json", record.to_dict())
            (staging / "feedback").mkdir(parents=True, exist_ok=True)
            (staging / "performance").mkdir(parents=True, exist_ok=True)
            (staging / "tests").mkdir(parents=True, exist_ok=True)
            _atomic_write(staging / "feedback" / "records.jsonl", b"")
            _atomic_write(staging / "performance" / "history.jsonl", b"")
            (staging / "versions").mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                raise ProductConflictError(f"产品已存在，拒绝覆盖：{pid}")
            os.rename(staging, final_dir)
        except Exception:
            # 保留 staging 供排查，不产生半成品产品目录
            raise
        return record

    def get_product(self, product_id: str) -> ProductRecord:
        directory = self._product_dir(product_id)
        product_file = directory / "product.json"
        if not product_file.is_file():
            raise ProductNotFoundError(f"产品不存在：{product_id}")
        payload = _read_json(product_file, what="product.json")
        if payload.get("product_id") != product_id:
            raise ProductIntegrityError(
                f"product.json 的 product_id 与目录不一致：{payload.get('product_id')!r} != {product_id}"
            )
        if payload.get("schema_version") != PRODUCT_SCHEMA:
            raise ProductIntegrityError(f"不支持的产品 schema：{payload.get('schema_version')!r}")
        record = ProductRecord.from_dict(payload)
        if record.status not in PRODUCT_STATUSES:
            raise ProductIntegrityError(f"非法产品状态：{record.status!r}")
        return record

    def list_products(self, *, include_archived: bool = False) -> list[ProductRecord]:
        if not self.root.is_dir():
            return []
        records: list[ProductRecord] = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if not _ID_RE.fullmatch(entry.name):
                continue
            try:
                record = self.get_product(entry.name)
            except ProductStoreError:
                continue
            if record.status == "archived" and not include_archived:
                continue
            records.append(record)
        return sorted(records, key=lambda r: (r.created_at, r.product_id))

    # ------------------------------------------------------------------
    # 创建 / 读取版本
    # ------------------------------------------------------------------
    @_serialized
    def create_version(self, product_id: str, **kwargs) -> ProductVersion:
        return self._create_version_locked(product_id, **kwargs)

    def _create_version_locked(
        self,
        product_id: str,
        *,
        reference_bank_path: str = "reference",
        calibration_path: str = "calibration",
        defect_bank_path: str | None = None,
        threshold: float | None = None,
        calibration_metadata: dict | None = None,
        feature_extractor_version: str = "",
        weights_sha256: str = "",
        version: int | None = None,
        source_samples: list[dict] | None = None,
        source_status: str = "legacy",
        registration_fingerprint: str = "",
        populate=None,
        initial_status: str = "validating",
    ) -> ProductVersion:
        if initial_status not in VERSION_STATUSES:
            raise ProductStateError(f"非法初始版本状态：{initial_status!r}")
        product = self._require_product(product_id)
        if product.status == "archived":
            raise ProductStateError(f"产品已归档，不能新增版本：{product_id}")

        if version is None:
            existing = [
                int(p.name[1:]) for p in self._versions_dir(product_id).iterdir()
                if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
            ] if self._versions_dir(product_id).is_dir() else []
            version = max(existing, default=0) + 1
        dirname = version_dirname(version)
        target = self._versions_dir(product_id) / dirname
        if target.exists():
            raise ProductConflictError(f"版本已存在，拒绝覆盖：{product_id}/{dirname}")

        for name, value in (("reference_bank_path", reference_bank_path),
                            ("calibration_path", calibration_path)):
            _safe_relative(Path("."), value, field_name=name)
        if defect_bank_path is not None:
            _safe_relative(Path("."), defect_bank_path, field_name="defect_bank_path")
        if threshold is not None and not isinstance(threshold, (int, float)):
            raise ProductStateError("threshold 必须是数值或 None")
        if threshold is not None and not math.isfinite(float(threshold)):
            raise ProductStateError("threshold 必须是有限数值")

        record = ProductVersion(
            product_id=product_id,
            version=version,
            status=initial_status,
            created_at=_now(),
            feature_extractor_version=feature_extractor_version,
            weights_sha256=weights_sha256,
            reference_bank_path=reference_bank_path,
            calibration_path=calibration_path,
            defect_bank_path=defect_bank_path,
            threshold=float(threshold) if threshold is not None else None,
            calibration_metadata=dict(calibration_metadata or {}),
            manifest_sha256="",
            source_samples=list(source_samples or []),
            source_status=source_status,
            registration_fingerprint=registration_fingerprint,
        )

        staging = self._make_staging(f"{product_id}-{dirname}-{uuid.uuid4().hex[:8]}")
        try:
            for relative in ("reference", "calibration", "tests"):
                (staging / relative).mkdir(parents=True, exist_ok=True)
                _atomic_write(staging / relative / ".keep", b"")
            if defect_bank_path is not None:
                (staging / "defects").mkdir(parents=True, exist_ok=True)
                _atomic_write(staging / "defects" / ".keep", b"")
            _atomic_write_json(staging / "version.json", record.storage_dict())
            if populate is not None:
                populate(staging, version)
            manifest = self._build_manifest(staging, product_id, version)
            _atomic_write_json(staging / "manifest.json", manifest)
            manifest_sha = sha256_file(staging / "manifest.json")
            _atomic_write(staging / "manifest.sha256", (manifest_sha + "\n").encode("ascii"))
            record.manifest_sha256 = manifest_sha

            self.validate_version_tree(staging, product_id, version, expected_manifest_sha=manifest_sha)
            if target.exists():
                raise ProductConflictError(f"版本已存在，拒绝覆盖：{product_id}/{dirname}")
            os.rename(staging, target)
        except Exception:
            raise

        # 产品 updated_at 只做轻量更新；失败不影响已提交版本
        product.updated_at = _now()
        _atomic_write_json(self._product_dir(product_id) / "product.json", product.to_dict())
        return self.get_version(product_id, version)

    def register_version(
        self,
        product_id: str,
        *,
        populate=None,
        activate: bool = True,
        **kwargs,
    ) -> tuple[ProductRecord, ProductVersion]:
        """在**同一个产品锁**内完成“建版本 + 提交 + 激活”。

        这是给注册服务使用的事务接口：特征提取/校准在锁外完成，
        但版本写入、manifest 校验、提交与 current 指针切换不可被其他进程插入。
        """
        with self._product_lock(product_id):
            version = self._create_version_locked(
                product_id, populate=populate, initial_status="committed", **kwargs
            )
            validation = self.validate_version(product_id, version.version)
            if not validation["ok"]:
                raise ProductIntegrityError(
                    f"新版本完整性校验失败，未激活：{validation['problems']}"
                )
            if not activate:
                return self._require_product(product_id), self.get_version(
                    product_id, version.version
                )
            product = self._activate_version_pointer_locked(product_id, version.version)
            version = self.get_version(product_id, version.version)
        return product, version

    def _activate_version_pointer_locked(self, product_id: str, version: int) -> ProductRecord:
        """只切换 product.current_version，不改写任何既有版本文件。

        注册场景要求“旧版本 manifest 与文件哈希保持不变”，因此激活不修改旧版本
        的 version.json；`set_current_version()` 仍保留 P1 的显式 superseded 语义。
        """
        product = self._require_product(product_id)
        if product.status == "archived":
            raise ProductStateError(f"产品已归档，不能激活版本：{product_id}")
        target = self.get_version(product_id, version)
        if target.status == "archived":
            raise ProductStateError("已归档版本不能激活")
        validation = self.validate_version(product_id, version)
        if not validation["ok"]:
            raise ProductIntegrityError(f"版本校验失败：{validation['problems']}")
        product.current_version = version
        product.status = "active"
        product.updated_at = _now()
        _atomic_write_json(self._product_dir(product_id) / "product.json", product.to_dict())
        return self.get_product(product_id)

    @staticmethod
    def _derive_version_status(raw_status: str, current_version: int | None,
                               version: int) -> str:
        """把持久化状态派生为对外语义（不修改任何版本文件）。

        * 归档版本始终 archived；
        * 产品存在 current 指针时：current → active，其余一律 superseded；
        * 产品尚无 current 指针（draft/activate=False）时：保留 validating/committed，
          保证 P1 的“新建但未激活”语义与旧文件兼容。
        """
        if raw_status == "archived":
            return "archived"
        if current_version is not None:
            return "active" if version == current_version else "superseded"
        return raw_status if raw_status in ("validating", "committed") else "superseded"

    def get_version(self, product_id: str, version: int) -> ProductVersion:
        directory = self._version_dir(product_id, version)
        if not directory.is_dir():
            raise ProductNotFoundError(f"版本不存在：{product_id}/v{version}")
        payload = _read_json(directory / "version.json", what="version.json")
        if payload.get("product_id") != product_id:
            raise ProductIntegrityError("version.json 的 product_id 不匹配")
        if int(payload.get("version", -1)) != version:
            raise ProductIntegrityError("version.json 的 version 不匹配")
        if payload.get("status") not in VERSION_STATUSES:
            raise ProductIntegrityError(f"非法版本状态：{payload.get('status')!r}")
        product = self.get_product(product_id)
        derived_status = self._derive_version_status(
            str(payload["status"]), product.current_version, version
        )
        manifest_path = directory / "manifest.json"
        sha_path = directory / "manifest.sha256"
        if not manifest_path.is_file() or not sha_path.is_file():
            raise ProductIntegrityError(f"版本缺少 manifest：{directory}")
        expected = sha_path.read_text(encoding="ascii").strip()
        actual = sha256_file(manifest_path)
        if expected != actual:
            raise ProductIntegrityError(
                f"manifest.sha256 与 manifest 不一致：{expected[:12]} != {actual[:12]}"
            )
        self._verify_manifest_files(directory, product_id, version)
        return ProductVersion(
            product_id=payload["product_id"],
            version=int(payload["version"]),
            status=derived_status,
            created_at=str(payload["created_at"]),
            feature_extractor_version=str(payload.get("feature_extractor_version", "")),
            weights_sha256=str(payload.get("weights_sha256", "")),
            reference_bank_path=str(payload["reference_bank_path"]),
            calibration_path=str(payload["calibration_path"]),
            defect_bank_path=payload.get("defect_bank_path"),
            threshold=payload.get("threshold"),
            calibration_metadata=dict(payload.get("calibration_metadata") or {}),
            manifest_sha256=expected,
            source_samples=list(payload.get("source_samples") or []),
            source_status=str(payload.get("source_status", "legacy")),
            registration_fingerprint=str(payload.get("registration_fingerprint", "")),
            test_records=list(payload.get("test_records") or []),
        )

    def list_versions(self, product_id: str) -> list[ProductVersion]:
        directory = self._versions_dir(product_id)
        if not directory.is_dir():
            return []
        versions = []
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir() or not re.fullmatch(r"v\d{3}", entry.name):
                continue
            try:
                versions.append(self.get_version(product_id, int(entry.name[1:])))
            except ProductStoreError:
                continue
        return versions

    # ------------------------------------------------------------------
    # 当前版本与状态
    # ------------------------------------------------------------------
    @_serialized
    def set_current_version(self, product_id: str, version: int) -> ProductRecord:
        return self._set_current_version_locked(product_id, version)

    @_serialized
    def compare_and_set_current(self, product_id: str, expected_version: int,
                                version: int) -> ProductRecord:
        """在同一产品锁内完成 current 比较与切换。"""
        product = self._require_product(product_id)
        if product.current_version != expected_version:
            raise ProductStateConflictError(
                f"current_version 已变化：预期 {expected_version}，"
                f"实际 {product.current_version}"
            )
        return self._activate_version_pointer_locked(product_id, version)

    def _set_current_version_locked(self, product_id: str, version: int) -> ProductRecord:
        """只切换 product.current_version；绝不改写任何版本文件。"""
        return self._activate_version_pointer_locked(product_id, version)

    @_serialized
    def update_product_status(self, product_id: str, status: str) -> ProductRecord:
        if status not in PRODUCT_STATUSES:
            raise ProductStateError(f"非法产品状态：{status!r}")
        product = self._require_product(product_id)
        if status == product.status:
            return product
        allowed = PRODUCT_TRANSITIONS.get(product.status, set())
        if status not in allowed:
            raise ProductStateError(f"不允许的状态迁移：{product.status} -> {status}")
        if status == "active":
            if product.current_version is None:
                raise ProductStateError("没有当前版本，不能激活产品")
            validation = self.validate_version(product_id, product.current_version)
            if not validation["ok"]:
                raise ProductIntegrityError(f"当前版本校验失败：{validation['problems']}")
        product.status = status
        product.updated_at = _now()
        _atomic_write_json(self._product_dir(product_id) / "product.json", product.to_dict())
        return self.get_product(product_id)

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate_product(self, product_id: str) -> dict:
        problems: list[str] = []
        record: dict | None = None
        try:
            product = self.get_product(product_id)
            record = product.to_dict()
            directory = self._product_dir(product_id)
            if not directory.is_dir():
                problems.append("产品目录不存在")
            for required in ("feedback", "performance", "tests", "versions"):
                if not (directory / required).is_dir():
                    problems.append(f"缺少目录：{required}")
            if product.status == "active" and product.current_version is None:
                problems.append("active 产品没有 current_version")
            if product.status == "draft" and product.current_version is not None:
                problems.append("draft 产品不应有 current_version")
            if product.current_version is not None:
                version_result = self.validate_version(product_id, product.current_version)
                if not version_result["ok"]:
                    problems.extend(f"current_version: {p}" for p in version_result["problems"])
                else:
                    current = self.get_version(product_id, product.current_version)
                    if current.status != "active":
                        problems.append("current_version 未表现为 active")
            versions = self.list_versions(product_id)
            active_versions = [v.version for v in versions if v.status == "active"]
            if product.current_version is None:
                if active_versions:
                    problems.append(f"没有 current_version 但存在 active 版本：{active_versions}")
            elif active_versions != [product.current_version]:
                problems.append(
                    f"有效 active 版本应恰好是 current_version={product.current_version}，"
                    f"实际为 {active_versions}"
                )
        except ProductStoreError as exc:
            problems.append(f"{type(exc).__name__}: {exc}")
        return {"ok": not problems, "product_id": product_id, "problems": problems, "product": record}

    def validate_version(self, product_id: str, version: int) -> dict:
        problems: list[str] = []
        manifest_sha = ""
        try:
            directory = self._version_dir(product_id, version)
            if not directory.is_dir():
                return {"ok": False, "product_id": product_id, "version": version,
                        "problems": ["版本目录不存在"], "manifest_sha256": ""}
            version_record = self.get_version(product_id, version)
            manifest = _read_json(directory / "manifest.json", what="manifest.json")
            expected_sha = (directory / "manifest.sha256").read_text(encoding="ascii").strip()
            actual_sha = sha256_file(directory / "manifest.json")
            manifest_sha = actual_sha
            if expected_sha != actual_sha:
                problems.append("manifest.sha256 不匹配")
            if manifest.get("product_id") != product_id:
                problems.append("manifest product_id 不匹配")
            if int(manifest.get("version", -1)) != version:
                problems.append("manifest version 不匹配")
            if manifest.get("schema_version") != MANIFEST_SCHEMA:
                problems.append("manifest schema 不支持")

            listed = manifest.get("files") or {}
            actual_files = self._scan_version_files(directory)
            if set(listed) != set(actual_files):
                missing = sorted(set(listed) - set(actual_files))
                extra = sorted(set(actual_files) - set(listed))
                if missing:
                    problems.append(f"manifest 列出但文件缺失：{missing[:5]}")
                if extra:
                    problems.append(f"存在 manifest 未列出的文件：{extra[:5]}")
            for relative, expected in listed.items():
                if relative not in actual_files:
                    continue
                target = directory / relative
                try:
                    if target.is_symlink():
                        problems.append(f"不允许符号链接：{relative}")
                        continue
                    if sha256_file(target) != expected.get("sha256"):
                        problems.append(f"文件哈希不匹配：{relative}")
                    if target.stat().st_size != expected.get("size"):
                        problems.append(f"文件大小不匹配：{relative}")
                except OSError as exc:
                    problems.append(f"文件读取失败：{relative}（{exc}）")

            for field_name, value in (("reference_bank_path", version_record.reference_bank_path),
                                      ("calibration_path", version_record.calibration_path)):
                try:
                    target = _safe_relative(directory, value, field_name=field_name)
                except ProductPathError as exc:
                    problems.append(str(exc))
                    continue
                if not target.exists():
                    problems.append(f"{field_name} 指向的路径不存在：{value}")
            if version_record.defect_bank_path is not None:
                try:
                    target = _safe_relative(directory, version_record.defect_bank_path,
                                            field_name="defect_bank_path")
                    if not target.exists():
                        problems.append("defect_bank_path 指向的路径不存在")
                except ProductPathError as exc:
                    problems.append(str(exc))
            if version_record.source_status == "persisted":
                for item in version_record.source_samples:
                    relative = item.get("rel_path")
                    try:
                        target = _safe_relative(directory, str(relative), field_name="source_samples.rel_path")
                    except ProductPathError as exc:
                        problems.append(str(exc))
                        continue
                    if not target.is_file():
                        problems.append(f"持久化来源样本不存在：{relative}")
                        continue
                    if sha256_file(target) != item.get("sha256"):
                        problems.append(f"持久化来源样本哈希不匹配：{relative}")
                    if target.stat().st_size != item.get("size_bytes"):
                        problems.append(f"持久化来源样本大小不匹配：{relative}")
            if version_record.threshold is not None:
                if not isinstance(version_record.threshold, (int, float)) or not math.isfinite(
                    float(version_record.threshold)
                ):
                    problems.append("threshold 非有限数值")
        except ProductStoreError as exc:
            problems.append(f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{type(exc).__name__}: {exc}")
        return {"ok": not problems, "product_id": product_id, "version": version,
                "problems": problems, "manifest_sha256": manifest_sha}

    def validate_version_tree(
        self, directory: str | Path, product_id: str, version: int, *,
        expected_manifest_sha: str | None = None,
    ) -> None:
        """校验任意目录（含 staging）的 manifest 与文件树，失败抛异常。"""
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        sha_path = directory / "manifest.sha256"
        if not manifest_path.is_file() or not sha_path.is_file():
            raise ProductIntegrityError("缺少 manifest.json 或 manifest.sha256")
        actual = sha256_file(manifest_path)
        expected = sha_path.read_text(encoding="ascii").strip()
        if expected != actual:
            raise ProductIntegrityError("manifest.sha256 与 manifest 不一致")
        if expected_manifest_sha is not None and expected_manifest_sha != actual:
            raise ProductIntegrityError("manifest 哈希与预期不一致")
        manifest = _read_json(manifest_path, what="manifest.json")
        if manifest.get("product_id") != product_id or int(manifest.get("version", -1)) != version:
            raise ProductIntegrityError("manifest 的 product_id/version 不匹配")
        listed = manifest.get("files") or {}
        actual_files = self._scan_version_files(directory)
        if set(listed) != set(actual_files):
            raise ProductIntegrityError("manifest 文件清单与目录不一致")
        for relative, item in listed.items():
            target = directory / relative
            if target.is_symlink():
                raise ProductIntegrityError(f"不允许符号链接：{relative}")
            if sha256_file(target) != item.get("sha256"):
                raise ProductIntegrityError(f"文件哈希不匹配：{relative}")

    # ------------------------------------------------------------------
    # staging
    # ------------------------------------------------------------------
    def list_staging(self) -> list[dict]:
        if not self.staging_root.is_dir():
            return []
        entries = []
        for entry in sorted(self.staging_root.iterdir()):
            if not entry.is_dir():
                continue
            files = [p for p in entry.rglob("*") if p.is_file()]
            entries.append({
                "name": entry.name,
                "path": str(entry),
                "file_count": len(files),
                "reason": "incomplete_write_or_interrupted_registration",
            })
        return entries

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _make_staging(self, label: str) -> Path:
        self.staging_root.mkdir(parents=True, exist_ok=True)
        staging = self.staging_root / label
        staging.mkdir(parents=False, exist_ok=False)
        return staging

    def _verify_manifest_files(self, directory: Path, product_id: str, version: int) -> dict:
        manifest = _read_json(directory / "manifest.json", what="manifest.json")
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise ProductIntegrityError(f"manifest schema 不支持：{manifest.get('schema_version')!r}")
        if manifest.get("product_id") != product_id:
            raise ProductIntegrityError("manifest 的 product_id 不匹配")
        if int(manifest.get("version", -1)) != version:
            raise ProductIntegrityError("manifest 的 version 不匹配")
        listed = manifest.get("files") or {}
        actual = self._scan_version_files(directory)
        if set(listed) != set(actual):
            raise ProductIntegrityError("manifest 文件清单与目录不一致")
        for relative, item in listed.items():
            target = directory / relative
            if target.is_symlink():
                raise ProductIntegrityError(f"不允许符号链接：{relative}")
            if sha256_file(target) != item.get("sha256"):
                raise ProductIntegrityError(f"文件哈希不匹配：{relative}")
            if target.stat().st_size != item.get("size"):
                raise ProductIntegrityError(f"文件大小不匹配：{relative}")
        return manifest

    def _scan_version_files(self, directory: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ProductIntegrityError(f"不允许符号链接：{path}")
            if not path.is_file():
                continue
            relative = path.relative_to(directory).as_posix()
            if relative in ("manifest.json", "manifest.sha256"):
                continue
            files[relative] = path
        return files

    def _build_manifest(self, directory: Path, product_id: str, version: int) -> dict:
        files = {}
        for relative, path in self._scan_version_files(directory).items():
            files[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}
        return {
            "schema_version": MANIFEST_SCHEMA,
            "product_id": product_id,
            "version": version,
            "created_at": _now(),
            "files": files,
        }

    def _write_version_file(self, product_id: str, record: ProductVersion) -> None:
        directory = self._version_dir(product_id, record.version)
        _atomic_write_json(directory / "version.json", record.storage_dict())
        manifest = self._build_manifest(directory, product_id, record.version)
        _atomic_write_json(directory / "manifest.json", manifest)
        _atomic_write(directory / "manifest.sha256",
                      (sha256_file(directory / "manifest.json") + "\n").encode("ascii"))

    def product_directory(self, product_id: str) -> Path:
        """只读查询产品目录（测试/诊断用；调用方不得绕过接口写入）。"""
        return self._product_dir(product_id)

    def version_directory(self, product_id: str, version: int) -> Path:
        """只读查询版本目录（供产品检测等只读消费者使用）。"""
        return self._version_dir(product_id, version)
