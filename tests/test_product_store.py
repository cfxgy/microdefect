#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 产品数据模型与原子存储测试。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.products import (  # noqa: E402
    ProductConflictError,
    ProductIntegrityError,
    ProductLockTimeoutError,
    ProductNotFoundError,
    ProductPathError,
    ProductStateError,
    ProductStore,
    sha256_file,
)

TESTS_ROOT = PROJECT_ROOT / "cache" / "tests"

WORKER_SOURCE = r'''# -*- coding: utf-8 -*-
import json
import os
import sys
import time
from pathlib import Path

root, result, op, pid = sys.argv[1:5]
arg1 = sys.argv[5] if len(sys.argv) > 5 else ""
arg2 = sys.argv[6] if len(sys.argv) > 6 else ""
sys.path.insert(0, r"__PROJECT_ROOT__")
from src.core.products import (
    ProductLockTimeoutError,
    ProductStore,
    ProductStoreError,
)

timeout = float(os.environ.get("PRODUCT_LOCK_TIMEOUT", "10"))
store = ProductStore(Path(root), lock_timeout=timeout)
payload = {"ok": True}
try:
    if op == "create":
        store.create_product(arg1 or "P", product_id=pid)
    elif op == "version":
        store.create_version(pid, version=int(arg1))
    elif op == "setcurrent":
        store.set_current_version(pid, int(arg1))
    elif op == "hold":
        with store._product_lock(pid):
            Path(arg2).write_text("locked", encoding="utf-8")
            time.sleep(float(arg1))
    elif op == "crash":
        with store._product_lock(pid):
            Path(arg1).write_text("locked", encoding="utf-8")
            os._exit(1)
    else:
        raise RuntimeError("unknown op")
except ProductLockTimeoutError as exc:
    payload = {"ok": False, "error": "ProductLockTimeoutError", "message": str(exc)}
except ProductStoreError as exc:
    payload = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
except Exception as exc:  # noqa: BLE001
    payload = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
Path(result).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
'''


def worker_script() -> Path:
    TESTS_ROOT.mkdir(parents=True, exist_ok=True)
    path = TESTS_ROOT / "product_store_worker.py"
    if not path.is_file():
        path.write_text(WORKER_SOURCE.replace("__PROJECT_ROOT__", str(PROJECT_ROOT)),
                        encoding="utf-8")
    return path


class ProductStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        TESTS_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = TESTS_ROOT / f"products_{uuid.uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.store = ProductStore(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _active(self, name: str = "产品A") -> tuple[str, int]:
        product = self.store.create_product(name, "washer")
        version = self.store.create_version(product.product_id)
        self.store.set_current_version(product.product_id, version.version)
        return product.product_id, version.version

    # 1
    def test_01_create_product(self) -> None:
        record = self.store.create_product("垫片-A", "washer", metadata={"line": "L1"})
        self.assertEqual(record.status, "draft")
        self.assertIsNone(record.current_version)
        self.assertEqual(len(record.product_id), 32)
        directory = self.store.product_directory(record.product_id)
        self.assertTrue((directory / "product.json").is_file())
        self.assertTrue((directory / "feedback" / "records.jsonl").is_file())
        self.assertTrue((directory / "performance" / "history.jsonl").is_file())
        self.assertTrue((directory / "versions").is_dir())
        self.assertEqual(self.store.get_product(record.product_id).name, "垫片-A")

    # 2
    def test_02_uuid_unique(self) -> None:
        a = self.store.create_product("同名")
        b = self.store.create_product("同名")
        self.assertNotEqual(a.product_id, b.product_id)
        self.assertEqual(len({a.product_id, b.product_id}), 2)

    # 3
    def test_03_same_name_independent(self) -> None:
        a = self.store.create_product("同名产品")
        b = self.store.create_product("同名产品")
        self.assertTrue(self.store.product_directory(a.product_id).is_dir())
        self.assertTrue(self.store.product_directory(b.product_id).is_dir())
        self.assertNotEqual(self.store.product_directory(a.product_id), self.store.product_directory(b.product_id))

    # 4
    def test_04_list_and_get(self) -> None:
        a = self.store.create_product("A")
        b = self.store.create_product("B")
        listing = self.store.list_products()
        self.assertEqual({p.product_id for p in listing}, {a.product_id, b.product_id})
        self.assertEqual(self.store.get_product(a.product_id).name, "A")
        with self.assertRaises(ProductNotFoundError):
            self.store.get_product("f" * 32)

    # 5
    def test_05_create_version(self) -> None:
        product = self.store.create_product("A")
        version = self.store.create_version(product.product_id)
        self.assertEqual(version.version, 1)
        self.assertEqual(version.status, "validating")
        self.assertIsNone(version.threshold)
        self.assertEqual(version.reference_bank_path, "reference")
        self.assertEqual(len(version.manifest_sha256), 64)
        directory = self.root / product.product_id / "versions" / "v001"
        for name in ("version.json", "manifest.json", "manifest.sha256",
                     "reference/.keep", "calibration/.keep", "tests/.keep"):
            self.assertTrue((directory / name).is_file(), name)
        self.assertEqual([v.version for v in self.store.list_versions(product.product_id)], [1])

    # 6
    def test_06_old_version_untouched(self) -> None:
        product = self.store.create_product("A")
        v1 = self.store.create_version(product.product_id)
        v1_dir = self.root / product.product_id / "versions" / "v001"
        before = {p.relative_to(v1_dir).as_posix(): sha256_file(p) for p in v1_dir.rglob("*") if p.is_file()}
        v2 = self.store.create_version(product.product_id)
        after = {p.relative_to(v1_dir).as_posix(): sha256_file(p) for p in v1_dir.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(v2.version, 2)
        self.assertTrue((self.root / product.product_id / "versions" / "v002").is_dir())

    # 7
    def test_07_switch_current_version(self) -> None:
        product = self.store.create_product("A")
        v1 = self.store.create_version(product.product_id)
        self.store.set_current_version(product.product_id, v1.version)
        self.assertEqual(self.store.get_product(product.product_id).current_version, 1)
        self.assertEqual(self.store.get_version(product.product_id, 1).status, "active")

        v2 = self.store.create_version(product.product_id)
        self.store.set_current_version(product.product_id, v2.version)
        record = self.store.get_product(product.product_id)
        self.assertEqual(record.status, "active")
        self.assertEqual(record.current_version, 2)
        self.assertEqual(self.store.get_version(product.product_id, 1).status, "superseded")
        self.assertEqual(self.store.get_version(product.product_id, 2).status, "active")

    # 8
    def test_08_disable_and_archive(self) -> None:
        pid, _ = self._active()
        disabled = self.store.update_product_status(pid, "disabled")
        self.assertEqual(disabled.status, "disabled")
        self.assertIn(pid, {p.product_id for p in self.store.list_products()})
        active = self.store.update_product_status(pid, "active")
        self.assertEqual(active.status, "active")
        archived = self.store.update_product_status(pid, "archived")
        self.assertEqual(archived.status, "archived")
        self.assertNotIn(pid, {p.product_id for p in self.store.list_products()})
        self.assertIn(pid, {p.product_id for p in self.store.list_products(include_archived=True)})
        with self.assertRaises(ProductStateError):
            self.store.update_product_status(pid, "active")
        self.assertTrue(self.store.product_directory(pid).is_dir())

    # 9
    def test_09_duplicate_creation_rejected(self) -> None:
        fixed = uuid.uuid4().hex
        first = self.store.create_product("A", product_id=fixed)
        with self.assertRaises(ProductConflictError):
            self.store.create_product("A2", product_id=fixed)
        v1 = self.store.create_version(first.product_id, version=1)
        self.assertEqual(v1.version, 1)
        with self.assertRaises(ProductConflictError):
            self.store.create_version(first.product_id, version=1)
        v2 = self.store.create_version(first.product_id)
        self.assertEqual(v2.version, 2)

    # 10
    def test_10_invalid_ids_and_traversal(self) -> None:
        for bad in ("", "..", "../x", "abc", "A" * 32, "a" * 31, "a" * 33, "a/b" * 10, "a" * 31 + "/"):
            with self.assertRaises(ProductPathError, msg=bad):
                self.store.get_product(bad)
        product = self.store.create_product("A")
        for bad_path in ("../evil", "/abs", "C:/x", "a\\b", "reference/../../x"):
            with self.assertRaises(ProductPathError, msg=bad_path):
                self.store.create_version(product.product_id, reference_bank_path=bad_path)
        for bad_version in (0, -1, 1000, True):
            with self.assertRaises(ProductPathError):
                self.store.get_version(product.product_id, bad_version)

    # 11
    def test_11_manifest_hash_validation(self) -> None:
        product = self.store.create_product("A")
        version = self.store.create_version(product.product_id)
        self.assertTrue(self.store.validate_version(product.product_id, version.version)["ok"])

        keep = self.root / product.product_id / "versions" / "v001" / "reference" / ".keep"
        keep.write_text("tampered", encoding="utf-8")
        result = self.store.validate_version(product.product_id, version.version)
        self.assertFalse(result["ok"])
        self.assertTrue(any("哈希" in p or "hash" in p.lower() for p in result["problems"]))

        with self.assertRaises(ProductIntegrityError):
            self.store.get_version(product.product_id, version.version)

    # 12
    def test_12_corruption_detection(self) -> None:
        product = self.store.create_product("A")
        version = self.store.create_version(product.product_id)
        directory = self.root / product.product_id / "versions" / "v001"
        (directory / "manifest.sha256").write_text("0" * 64, encoding="ascii")
        with self.assertRaises(ProductIntegrityError):
            self.store.get_version(product.product_id, version.version)

        (directory / "version.json").unlink()
        with self.assertRaises((ProductNotFoundError, ProductIntegrityError)):
            self.store.get_version(product.product_id, version.version)

    # 13
    def test_13_interrupted_write_leaves_no_active(self) -> None:
        with mock.patch("src.core.products.os.rename", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.create_product("中断产品")
        self.assertEqual(self.store.list_products(), [])
        staging = self.store.list_staging()
        self.assertEqual(len(staging), 1)
        self.assertIn("interrupted", staging[0]["reason"])
        self.assertTrue(Path(staging[0]["path"]).is_dir())

    # 14
    def test_14_staging_not_product(self) -> None:
        staging = self.root / ".staging" / (uuid.uuid4().hex)
        staging.mkdir(parents=True)
        (staging / "product.json").write_text('{"broken": true}', encoding="utf-8")
        self.assertEqual(self.store.list_products(), [])
        with self.assertRaises(ProductNotFoundError):
            self.store.get_product(staging.name)

    # 15
    def test_15_product_isolation(self) -> None:
        a_pid, a_ver = self._active("A")
        b_pid, b_ver = self._active("B")
        a_dir = self.root / a_pid
        b_dir = self.root / b_pid
        self.assertNotEqual(a_dir, b_dir)
        a_manifest = (a_dir / "versions" / "v001" / "manifest.sha256").read_text(encoding="ascii")
        self.store.create_version(b_pid)  # B 新增版本
        self.assertEqual((a_dir / "versions" / "v001" / "manifest.sha256").read_text(encoding="ascii"), a_manifest)
        with self.assertRaises(ProductNotFoundError):
            self.store.get_version(b_pid, a_ver + 99)
        self.assertEqual(self.store.get_version(a_pid, a_ver).product_id, a_pid)
        self.assertTrue(self.store.validate_product(a_pid)["ok"])
        self.assertTrue(self.store.validate_product(b_pid)["ok"])

    # 17
    def test_17_non_finite_threshold_rejected(self) -> None:
        product = self.store.create_product("A")
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ProductStateError):
                self.store.create_version(product.product_id, threshold=bad)

    # 18
    def test_18_product_version_inconsistency_detected(self) -> None:
        pid, _ = self._active()
        product_file = self.store.product_directory(pid) / "product.json"
        payload = json.loads(product_file.read_text(encoding="utf-8"))
        payload["current_version"] = 99
        product_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        result = self.store.validate_product(pid)
        self.assertFalse(result["ok"])
        self.assertTrue(any("current_version" in p or "99" in p or "不存在" in p for p in result["problems"]))

    # 16
    def test_16_active_requires_current_version(self) -> None:
        product = self.store.create_product("A")
        with self.assertRaises(ProductStateError):
            self.store.update_product_status(product.product_id, "active")
        # draft 且无当前版本是合法状态
        self.assertTrue(self.store.validate_product(product.product_id)["ok"])


class ConcurrencyTest(unittest.TestCase):
    """跨进程锁测试：同一产品的写操作串行化。"""

    def setUp(self) -> None:
        TESTS_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = TESTS_ROOT / f"concurrency_{uuid.uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.store = ProductStore(self.root, lock_timeout=10.0)
        self.worker = worker_script()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _spawn(self, op: str, product_id: str, arg1: str = "", arg2: str = "",
               lock_timeout: float = 10.0):
        result = TESTS_ROOT / f"result_{uuid.uuid4().hex}.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["PRODUCT_LOCK_TIMEOUT"] = str(lock_timeout)
        process = subprocess.Popen(
            [sys.executable, str(self.worker), str(self.root), str(result), op,
             product_id, arg1, arg2],
            cwd=str(PROJECT_ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return process, result

    @staticmethod
    def _finish(process: subprocess.Popen, result: Path) -> dict:
        process.wait(timeout=90)
        return json.loads(result.read_text(encoding="utf-8"))

    @staticmethod
    def _wait_file(path: Path, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                return
            time.sleep(0.05)
        raise AssertionError(f"等待锁信号超时：{path}")

    # 1. 两进程同时创建同一产品
    def test_20_concurrent_create_same_product(self) -> None:
        pid = uuid.uuid4().hex
        p1, r1 = self._spawn("create", pid, "A")
        p2, r2 = self._spawn("create", pid, "B")
        res1 = self._finish(p1, r1)
        res2 = self._finish(p2, r2)
        self.assertEqual(sorted([res1["ok"], res2["ok"]]), [False, True])
        errors = {r.get("error") for r in (res1, res2) if not r["ok"]}
        self.assertEqual(errors, {"ProductConflictError"})
        record = self.store.get_product(pid)
        self.assertEqual(record.status, "draft")
        self.assertTrue(self.store.validate_product(pid)["ok"])
        product_dirs = [p for p in self.root.iterdir() if p.is_dir() and len(p.name) == 32]
        self.assertEqual(len(product_dirs), 1)

    # 2. 两进程同时创建同一产品的相同版本
    def test_21_concurrent_same_version(self) -> None:
        product = self.store.create_product("A")
        p1, r1 = self._spawn("version", product.product_id, "1")
        p2, r2 = self._spawn("version", product.product_id, "1")
        res1 = self._finish(p1, r1)
        res2 = self._finish(p2, r2)
        self.assertEqual(sorted([res1["ok"], res2["ok"]]), [False, True])
        self.assertEqual({r.get("error") for r in (res1, res2) if not r["ok"]},
                         {"ProductConflictError"})
        versions = self.store.list_versions(product.product_id)
        self.assertEqual([v.version for v in versions], [1])
        self.assertTrue(self.store.validate_version(product.product_id, 1)["ok"])

    # 3. 两进程同时创建不同版本
    def test_22_concurrent_different_versions(self) -> None:
        product = self.store.create_product("A")
        p1, r1 = self._spawn("version", product.product_id, "1")
        p2, r2 = self._spawn("version", product.product_id, "2")
        res1 = self._finish(p1, r1)
        res2 = self._finish(p2, r2)
        self.assertTrue(res1["ok"], res1)
        self.assertTrue(res2["ok"], res2)
        self.assertEqual(sorted(v.version for v in self.store.list_versions(product.product_id)), [1, 2])
        for version in (1, 2):
            self.assertTrue(self.store.validate_version(product.product_id, version)["ok"])
        self.assertIsNone(self.store.get_product(product.product_id).current_version)
        self.assertTrue(self.store.validate_product(product.product_id)["ok"])

    # 4. 两进程同时修改当前版本指针
    def test_23_concurrent_set_current(self) -> None:
        product = self.store.create_product("A")
        self.store.create_version(product.product_id, version=1)
        self.store.create_version(product.product_id, version=2)
        p1, r1 = self._spawn("setcurrent", product.product_id, "1")
        p2, r2 = self._spawn("setcurrent", product.product_id, "2")
        res1 = self._finish(p1, r1)
        res2 = self._finish(p2, r2)
        self.assertTrue(res1["ok"], res1)
        self.assertTrue(res2["ok"], res2)
        record = self.store.get_product(product.product_id)
        self.assertEqual(record.status, "active")
        self.assertIn(record.current_version, (1, 2))
        versions = {v.version: v.status for v in self.store.list_versions(product.product_id)}
        self.assertEqual(versions[record.current_version], "active")
        self.assertEqual(sum(1 for status in versions.values() if status == "active"), 1)
        self.assertTrue(self.store.validate_product(product.product_id)["ok"])

    # 5. 持锁进程异常退出，另一个进程可继续
    def test_24_crash_releases_lock(self) -> None:
        product = self.store.create_product("A")
        sentinel = TESTS_ROOT / f"crash_{uuid.uuid4().hex}.flag"
        process, _ = self._spawn("crash", product.product_id, str(sentinel))
        self._wait_file(sentinel)
        process.wait(timeout=30)
        self.assertNotEqual(process.returncode, 0)
        fast = ProductStore(self.root, lock_timeout=5.0)
        version = fast.create_version(product.product_id, version=1)
        self.assertEqual(version.version, 1)
        self.assertTrue(self.store.validate_version(product.product_id, 1)["ok"])

    # 6. 锁等待超时报告
    def test_25_lock_timeout_reported(self) -> None:
        product = self.store.create_product("A")
        sentinel = TESTS_ROOT / f"hold_{uuid.uuid4().hex}.flag"
        holder, _ = self._spawn("hold", product.product_id, "5", str(sentinel))
        self._wait_file(sentinel)
        short = ProductStore(self.root, lock_timeout=0.5)
        started = time.monotonic()
        with self.assertRaises(ProductLockTimeoutError) as context:
            short.create_version(product.product_id, version=1)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.4)
        self.assertIn(product.product_id, str(context.exception))
        self.assertIn("超时", str(context.exception))
        holder.wait(timeout=90)
        version = self.store.create_version(product.product_id, version=1)
        self.assertEqual(version.version, 1)
        self.assertTrue(self.store.validate_version(product.product_id, 1)["ok"])

    # 7. 并发冲突后 manifest 与状态不被破坏；锁文件不进入 manifest
    def test_26_conflict_does_not_corrupt_and_lock_not_in_manifest(self) -> None:
        pid = uuid.uuid4().hex
        p1, r1 = self._spawn("create", pid, "A")
        p2, r2 = self._spawn("create", pid, "B")
        self._finish(p1, r1)
        self._finish(p2, r2)
        lock_file = self.root / ".locks" / f"{pid}.lock"
        self.assertTrue(lock_file.is_file())
        product = self.store.get_product(pid)
        version = self.store.create_version(product.product_id, version=1)
        manifest = json.loads(
            (self.root / pid / "versions" / "v001" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertFalse(any("lock" in key.lower() for key in manifest["files"]))
        self.assertNotIn(".locks", str(manifest["files"]))
        self.assertTrue(self.store.validate_version(pid, version.version)["ok"])
        self.assertTrue(self.store.validate_product(pid)["ok"])
        self.assertEqual(self.store.get_product(pid).status, "draft")


    # 8. 不同产品之间不互相阻塞
    def test_27_unrelated_products_not_blocked(self) -> None:
        a = self.store.create_product("A")
        b = self.store.create_product("B")
        sentinel = TESTS_ROOT / f"holdA_{uuid.uuid4().hex}.flag"
        holder, _ = self._spawn("hold", a.product_id, "3", str(sentinel))
        self._wait_file(sentinel)
        started = time.monotonic()
        version = self.store.create_version(b.product_id, version=1)
        elapsed = time.monotonic() - started
        self.assertEqual(version.version, 1)
        self.assertLess(elapsed, 2.0)
        holder.wait(timeout=60)
        self.assertTrue(self.store.validate_version(b.product_id, 1)["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
