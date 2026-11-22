#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3.3 产品级批量检测测试。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.product_batch import (  # noqa: E402
    ProductBatchError,
    ProductBatchRun,
    list_product_batches,
    load_product_batch,
    batch_zip,
)
from src.core.product_registration import RegistrationRequest, register_product  # noqa: E402
from src.core.products import ProductStore, sha256_file  # noqa: E402

DATA = PROJECT_ROOT / "datasets/mvtec_ad/bottle"
TRAIN = DATA / "train/good"
TEST_GOOD = DATA / "test/good"
TEST_DEFECT = DATA / "test/broken_large"
TESTS_ROOT = PROJECT_ROOT / "cache/tests"


def _paths(root: Path, start: int, count: int) -> list[Path]:
    return [root / f"{index:03d}.png" for index in range(start, start + count)]


def _run_all(run: ProductBatchRun) -> None:
    while not run.finished:
        run.step()


class ProductBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TESTS_ROOT.mkdir(parents=True, exist_ok=True)
        cls.root = TESTS_ROOT / f"p33_product_{uuid.uuid4().hex[:8]}"
        cls.root.mkdir(parents=True, exist_ok=False)
        cls.store = ProductStore(cls.root, lock_timeout=60)
        cls.product_a = register_product(
            cls.store,
            RegistrationRequest(
                name="批量产品A", category="washer",
                reference_paths=_paths(TRAIN, 0, 5),
                calibration_paths=_paths(TRAIN, 5, 25),
            ),
            device="auto",
        )
        cls.product_b = register_product(
            cls.store,
            RegistrationRequest(
                name="批量产品B", category="gear",
                reference_paths=_paths(TRAIN, 60, 5),
                calibration_paths=_paths(TRAIN, 65, 25),
            ),
            device="auto",
        )
        cls.batch_root = cls.root / "batches"

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _run(self, product_id: str, paths: list[Path], names=None) -> ProductBatchRun:
        return ProductBatchRun.create(
            product_id, paths, original_names=names,
            product_root=self.root, batch_root=self.batch_root, device="cpu",
        )

    def test_01_completed_batch_and_persistence(self) -> None:
        paths = [TEST_GOOD / "000.png", TEST_GOOD / "001.png",
                 TEST_DEFECT / "000.png"]
        run = self._run(self.product_a.product.product_id, paths)
        _run_all(run)
        self.assertTrue(run.finished)
        self.assertEqual(run.status, "completed")
        summary = run.summary()
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["success"], 3)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["product_version"], 1)
        self.assertTrue((run.batch_dir / "batch.json").is_file())
        self.assertTrue((run.batch_dir / "results.csv").is_file())
        loaded = load_product_batch(run.batch_id, self.batch_root)
        self.assertEqual(loaded["status"], "completed")
        self.assertEqual(len(loaded["items"]), 3)
        self.assertEqual(len(list_product_batches(self.batch_root)), 1)
        self.assertTrue(batch_zip(run.batch_id, self.batch_root).is_file())

    def test_02_duplicate_names_are_unique(self) -> None:
        temp = self.root / f"names_{uuid.uuid4().hex[:6]}"
        temp.mkdir()
        a = temp / "a.png"
        b = temp / "b.png"
        shutil.copyfile(TEST_GOOD / "000.png", a)
        shutil.copyfile(TEST_GOOD / "001.png", b)
        run = self._run(self.product_a.product.product_id, [a, b],
                        names=["same.png", "same.png"])
        _run_all(run)
        items = run.to_dict()["items"]
        self.assertEqual([x["original_filename"] for x in items], ["same.png", "same.png"])
        self.assertNotEqual(items[0]["stored_filename"], items[1]["stored_filename"])
        self.assertTrue((run.batch_dir / items[0]["input_path"]).is_file())
        self.assertTrue((run.batch_dir / items[1]["input_path"]).is_file())

    def test_03_invalid_files_isolated(self) -> None:
        temp = self.root / f"invalid_{uuid.uuid4().hex[:6]}"
        temp.mkdir()
        corrupt = temp / "corrupt.png"
        corrupt.write_bytes(b"not-an-image")
        empty = temp / "empty.png"
        empty.write_bytes(b"")
        unsupported = temp / "image.gif"
        shutil.copyfile(TEST_GOOD / "000.png", unsupported)
        run = self._run(self.product_a.product.product_id,
                        [TEST_GOOD / "000.png", corrupt, empty, unsupported])
        _run_all(run)
        items = run.to_dict()["items"]
        self.assertEqual(items[0]["status"], "success")
        self.assertEqual([x["status"] for x in items[1:]], ["failed", "failed", "failed"])
        self.assertEqual(run.status, "partial")
        self.assertTrue(all(x["error"] for x in items[1:]))

    def test_04_stop_marks_unprocessed_and_persists(self) -> None:
        paths = _paths(TEST_GOOD, 0, 4)
        run = self._run(self.product_a.product.product_id, paths)
        run.step()
        run.request_stop()
        run.step()
        self.assertTrue(run.finished)
        self.assertEqual(run.status, "cancelled")
        payload = load_product_batch(run.batch_id, self.batch_root)
        self.assertEqual(payload["status"], "cancelled")
        self.assertEqual(payload["success"], 1)
        self.assertEqual(payload["unprocessed"], 3)
        self.assertTrue(all(x["status"] == "unprocessed" for x in payload["items"][1:]))

    def test_05_batch_version_does_not_follow_current_switch(self) -> None:
        pid = self.product_a.product.product_id
        run = self._run(pid, [TEST_GOOD / "000.png"])
        run.step()
        self.store.set_current_version(pid, 1)
        # 注册 v2 并切换；旧批次仍应记录 v1
        register_product(
            self.store,
            RegistrationRequest(
                name="ignored", product_id=pid,
                reference_paths=_paths(TRAIN, 30, 5),
                calibration_paths=_paths(TRAIN, 35, 25),
            ),
            device="auto",
        )
        self.store.set_current_version(pid, 2)
        loaded = load_product_batch(run.batch_id, self.batch_root)
        self.assertEqual(loaded["product_version"], 1)
        self.assertEqual(loaded["manifest_sha256"], run.detector.version.manifest_sha256)
        self.assertNotEqual(loaded["manifest_sha256"],
                            self.store.get_version(pid, 2).manifest_sha256)

    def test_06_product_files_are_not_modified(self) -> None:
        pid = self.product_a.product.product_id
        self.store.set_current_version(pid, 1)
        base = self.root / pid

        def hashes():
            return {p.relative_to(base).as_posix(): sha256_file(p)
                    for p in base.rglob("*") if p.is_file()}

        before = hashes()
        run = self._run(pid, [TEST_GOOD / "000.png", TEST_DEFECT / "000.png"])
        _run_all(run)
        self.assertEqual(hashes(), before)
        self.assertEqual(self.store.get_product(pid).current_version, 1)

    def test_07_batch_write_failure_does_not_mark_completed(self) -> None:
        run = self._run(self.product_a.product.product_id, [TEST_GOOD / "000.png"])
        with mock.patch("src.core.product_batch._write_json", side_effect=OSError("disk full")):
            with self.assertRaises(ProductBatchError):
                run.step()
        payload = json.loads((run.batch_dir / "batch.json").read_text(encoding="utf-8"))
        self.assertNotEqual(payload["status"], "completed")

    def test_08_invalid_product_rejected(self) -> None:
        with self.assertRaises(Exception):
            self._run("f" * 32, [TEST_GOOD / "000.png"])

    def test_09_failed_status_and_product_isolation(self) -> None:
        temp = self.root / f"fail_{uuid.uuid4().hex[:6]}"
        temp.mkdir()
        bad1 = temp / "bad1.png"
        bad2 = temp / "bad2.png"
        bad1.write_bytes(b"x")
        bad2.write_bytes(b"y")
        failed = self._run(self.product_a.product.product_id, [bad1, bad2])
        _run_all(failed)
        self.assertEqual(failed.status, "failed")
        other = self._run(self.product_b.product.product_id, [TEST_GOOD / "000.png"])
        _run_all(other)
        self.assertEqual(other.status, "completed")
        self.assertEqual(other.to_dict()["product_id"], self.product_b.product.product_id)
        self.assertEqual(other.to_dict()["product_name"], "批量产品B")
        self.assertNotEqual(other.to_dict()["manifest_sha256"],
                            self.product_a.version.manifest_sha256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
