#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4 产品级漂移监测测试。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

from PIL import Image, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.product_drift import (  # noqa: E402
    DriftBaselineError,
    DriftError,
    DriftMonitor,
    list_drift_windows,
    load_drift_window,
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


class ProductDriftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TESTS_ROOT.mkdir(parents=True, exist_ok=True)
        cls.root = TESTS_ROOT / f"p4_drift_{uuid.uuid4().hex[:8]}"
        cls.root.mkdir(parents=True, exist_ok=False)
        cls.store = ProductStore(cls.root, lock_timeout=60)
        cls.product_a = register_product(
            cls.store,
            RegistrationRequest(
                name="漂移产品A", category="washer",
                reference_paths=_paths(TRAIN, 0, 5),
                calibration_paths=_paths(TRAIN, 5, 25),
            ),
            device="auto",
        )
        cls.product_b = register_product(
            cls.store,
            RegistrationRequest(
                name="漂移产品B", category="gear",
                reference_paths=_paths(TRAIN, 60, 5),
                calibration_paths=_paths(TRAIN, 65, 25),
            ),
            device="auto",
        )
        cls.drift_root = cls.root / "drift"

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _analyze(self, product_id: str, paths: list[Path], names=None):
        monitor = DriftMonitor(product_id, product_root=self.root, device="cpu")
        return monitor.analyze(paths, original_names=names, window_root=self.drift_root)

    def _variants(self, mode: str, count: int = 5) -> list[Path]:
        directory = self.root / f"{mode}_{uuid.uuid4().hex[:6]}"
        directory.mkdir()
        paths = []
        for index, src in enumerate(_paths(TEST_GOOD, 0, count)):
            with Image.open(src) as handle:
                image = handle.convert("RGB")
            if mode == "bright":
                image = ImageEnhance.Brightness(image).enhance(1.5)
            elif mode == "contrast":
                image = ImageEnhance.Contrast(image).enhance(0.3)
            target = directory / f"{mode}_{index}.png"
            image.save(target)
            paths.append(target)
        return paths

    def test_01_baseline_window_persisted(self) -> None:
        payload = self._analyze(self.product_a.product.product_id,
                                _paths(TEST_GOOD, 0, 5))
        self.assertIn(payload["level"], ("normal", "observation"))
        self.assertEqual(payload["product_version"], 1)
        self.assertEqual(payload["success"], 5)
        self.assertEqual(payload["failed"], 0)
        self.assertTrue((self.drift_root / payload["window_id"] / "window.json").is_file())
        self.assertTrue((self.drift_root / payload["window_id"] / "results.csv").is_file())
        loaded = load_drift_window(payload["window_id"], self.drift_root)
        self.assertEqual(loaded["algorithm_version"], payload["algorithm_version"])

    def test_02_brightness_change_reflected(self) -> None:
        base = self._analyze(self.product_a.product.product_id,
                             _paths(TEST_GOOD, 0, 5))
        bright = self._analyze(self.product_a.product.product_id,
                               self._variants("bright"))
        base_b = base["metrics"]["brightness"]["mean"]
        bright_b = bright["metrics"]["brightness"]["mean"]
        self.assertGreater(bright_b, base_b)
        self.assertIn(bright["level"], ("observation", "drift_warning"))
        self.assertTrue(any("亮度" in reason for reason in bright["reasons"]))

    def test_03_contrast_change_reflected(self) -> None:
        base = self._analyze(self.product_a.product.product_id,
                             _paths(TEST_GOOD, 0, 5))
        low = self._analyze(self.product_a.product.product_id,
                            self._variants("contrast"))
        self.assertLess(low["metrics"]["contrast"]["mean"],
                        base["metrics"]["contrast"]["mean"])
        self.assertIn(low["level"], ("observation", "drift_warning"))

    def test_04_feature_distribution_change_reflected(self) -> None:
        base = self._analyze(self.product_a.product.product_id,
                             _paths(TEST_GOOD, 0, 5))
        changed = self._analyze(self.product_a.product.product_id,
                                _paths(TEST_DEFECT, 0, 5))
        self.assertGreater(changed["metrics"]["global"]["mean"],
                           base["metrics"]["global"]["mean"])
        self.assertIn(changed["level"], ("observation", "drift_warning"))

    def test_05_invalid_single_image_isolation(self) -> None:
        temp = self.root / f"bad_{uuid.uuid4().hex[:6]}"
        temp.mkdir()
        bad = temp / "bad.png"
        bad.write_bytes(b"not-an-image")
        payload = self._analyze(self.product_a.product.product_id,
                                _paths(TEST_GOOD, 0, 4) + [bad])
        self.assertEqual(payload["success"], 4)
        self.assertEqual(payload["failed"], 1)
        self.assertEqual(payload["level"], "insufficient")

    def test_06_insufficient_samples(self) -> None:
        payload = self._analyze(self.product_a.product.product_id,
                                _paths(TEST_GOOD, 0, 3))
        self.assertEqual(payload["level"], "insufficient")
        self.assertTrue(payload["reasons"])

    def test_07_products_are_isolated(self) -> None:
        a = self._analyze(self.product_a.product.product_id, _paths(TEST_GOOD, 0, 5))
        b = self._analyze(self.product_b.product.product_id, _paths(TEST_GOOD, 0, 5))
        self.assertNotEqual(a["product_id"], b["product_id"])
        self.assertNotEqual(a["bank_version"], b["bank_version"])
        self.assertNotEqual(a["manifest_sha256"], b["manifest_sha256"])

    def test_08_version_switch_keeps_old_window_version(self) -> None:
        pid = self.product_a.product.product_id
        self.store.set_current_version(pid, 1)
        payload = self._analyze(pid, _paths(TEST_GOOD, 0, 5))
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
        loaded = load_drift_window(payload["window_id"], self.drift_root)
        self.assertEqual(loaded["product_version"], 1)
        self.assertNotEqual(loaded["manifest_sha256"],
                            self.store.get_version(pid, 2).manifest_sha256)
        self.store.set_current_version(pid, 1)

    def test_09_history_and_product_unchanged(self) -> None:
        pid = self.product_a.product.product_id
        self.store.set_current_version(pid, 1)
        base = self.root / pid
        before = {p.relative_to(base).as_posix(): sha256_file(p)
                  for p in base.rglob("*") if p.is_file()}
        payload = self._analyze(pid, _paths(TEST_GOOD, 0, 5))
        after = {p.relative_to(base).as_posix(): sha256_file(p)
                 for p in base.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        windows = list_drift_windows(self.drift_root, product_id=pid)
        self.assertTrue(any(item["window_id"] == payload["window_id"] for item in windows))
        self.assertEqual(self.store.get_product(pid).current_version, 1)

    def test_10_write_failure_does_not_create_window(self) -> None:
        monitor = DriftMonitor(self.product_a.product.product_id,
                               product_root=self.root, device="cpu")
        before = len(list_drift_windows(self.drift_root,
                                        product_id=self.product_a.product.product_id))
        with mock.patch("src.core.product_drift._write_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                monitor.analyze(_paths(TEST_GOOD, 0, 5), window_root=self.drift_root)
        after = len(list_drift_windows(self.drift_root,
                                       product_id=self.product_a.product.product_id))
        self.assertEqual(before, after)

    def test_11_legacy_baseline_rejected(self) -> None:
        product = self.store.create_product("旧漂移产品")
        version = self.store.create_version(product.product_id)
        self.store.set_current_version(product.product_id, version.version)
        with self.assertRaises(DriftBaselineError):
            DriftMonitor(product.product_id, product_root=self.root, device="cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
