#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3.1 产品级单图检测引擎测试。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.product_detector import ProductDetector, ProductDetectorError  # noqa: E402
from src.core.product_registration import (  # noqa: E402
    RegistrationRequest,
    register_product,
)
from src.core.products import ProductStore, sha256_file  # noqa: E402
from src.core.weights import WeightsUnavailableError  # noqa: E402

DATA = PROJECT_ROOT / "datasets/mvtec_ad/bottle"
TRAIN = DATA / "train/good"
TEST_GOOD = DATA / "test/good"
TEST_DEFECT = DATA / "test/broken_large"
TESTS_ROOT = PROJECT_ROOT / "cache/tests"


def _paths(root: Path, start: int, count: int) -> list[Path]:
    return [root / f"{index:03d}.png" for index in range(start, start + count)]


class ProductDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TESTS_ROOT.mkdir(parents=True, exist_ok=True)
        cls.root = TESTS_ROOT / f"p31_detector_{uuid.uuid4().hex[:8]}"
        cls.root.mkdir(parents=True, exist_ok=False)
        cls.store = ProductStore(cls.root, lock_timeout=60)
        cls.product_a = register_product(
            cls.store,
            RegistrationRequest(
                name="产品A", category="washer",
                reference_paths=_paths(TRAIN, 0, 5),
                calibration_paths=_paths(TRAIN, 5, 25),
            ),
            device="auto",
        )
        cls.product_b = register_product(
            cls.store,
            RegistrationRequest(
                name="产品B", category="gear",
                reference_paths=_paths(TRAIN, 60, 5),
                calibration_paths=_paths(TRAIN, 65, 25),
            ),
            device="auto",
        )
        cls.detector_a = ProductDetector(cls.product_a.product.product_id, store=cls.store)
        cls.baseline_good = cls.detector_a.detect(TEST_GOOD / "000.png")
        cls.baseline_bad = cls.detector_a.detect(TEST_DEFECT / "000.png")

        # 并发测试专用：同一产品两个版本
        cls.product_s = register_product(
            cls.store,
            RegistrationRequest(
                name="并发产品", category="washer",
                reference_paths=_paths(TRAIN, 120, 5),
                calibration_paths=_paths(TRAIN, 125, 25),
            ),
            device="auto",
        )
        cls.product_s_v2 = register_product(
            cls.store,
            RegistrationRequest(
                name="ignored", product_id=cls.product_s.product.product_id,
                reference_paths=_paths(TRAIN, 150, 5),
                calibration_paths=_paths(TRAIN, 155, 25),
            ),
            device="auto",
        )
        cls.store.set_current_version(cls.product_s.product.product_id, 1)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _product_dir(self, product_id: str) -> Path:
        return self.root / product_id

    def _file_hashes(self, product_id: str) -> dict:
        base = self._product_dir(product_id)
        return {
            p.relative_to(base).as_posix(): sha256_file(p)
            for p in base.rglob("*") if p.is_file()
        }

    # 1
    def test_01_detect_normal_with_product_fields(self) -> None:
        detector = ProductDetector(self.product_a.product.product_id, store=self.store)
        result = detector.detect(TEST_GOOD / "000.png")
        self.assertEqual(result.product_id, self.product_a.product.product_id)
        self.assertEqual(result.product_name, "产品A")
        self.assertEqual(result.version, 1)
        self.assertEqual(result.image_status, "normal")
        self.assertFalse(result.is_anomaly)
        self.assertLess(result.image_score, result.image_threshold)
        self.assertEqual(result.image_sha256, sha256_file(TEST_GOOD / "000.png"))
        self.assertEqual(result.feature_extractor_version, "f1/resnet18-gap512-raw-fp32-notf32")
        self.assertEqual(result.weights_sha256,
                         "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec")
        self.assertEqual(result.memory_bank["global_member_count"], 5)
        self.assertEqual(result.memory_bank["spatial_member_count"], 5)
        self.assertEqual(result.memory_bank["product_id"], result.product_id)

    # 2
    def test_02_detect_anomaly_uses_image_threshold(self) -> None:
        detector = ProductDetector(self.product_a.product.product_id, store=self.store)
        result = detector.detect(TEST_DEFECT / "000.png")
        self.assertEqual(result.image_status, "anomaly")
        self.assertTrue(result.is_anomaly)
        self.assertGreater(result.image_score, result.image_threshold)
        # 该样本分数低于 pixel_threshold，但图像级仍必须按 image_threshold 判异常
        self.assertLess(result.image_score, result.pixel_threshold)
        self.assertFalse(result.image_score > result.pixel_threshold)

    # 3
    def test_03_json_and_files(self) -> None:
        detector = ProductDetector(self.product_a.product.product_id, store=self.store)
        result = detector.detect(TEST_GOOD / "000.png")
        out = self.root / f"out_{uuid.uuid4().hex[:6]}"
        files = result.save(out)
        for key in ("result_json", "heatmap_png", "overlay_png", "anomaly_mask_png"):
            self.assertTrue(Path(files[key]).is_file(), key)
        payload = json.loads(Path(files["result_json"]).read_text(encoding="utf-8"))
        for key in (
            "product_id", "product_name", "version", "manifest_sha256",
            "image_score", "image_threshold", "image_status", "pixel_threshold",
            "image_sha256", "weights_sha256", "feature_extractor_version",
            "anomaly_pixels", "anomaly_ratio", "memory_bank", "elapsed_time",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["image_sha256"], sha256_file(TEST_GOOD / "000.png"))
        self.assertEqual(payload["anomaly_pixels"], int(result.anomaly_mask.sum()))
        self.assertEqual(result.heatmap_stats["max"], float(result.anomaly_heatmap.max()))

    # 4
    def test_04_repeat_stable(self) -> None:
        detector = ProductDetector(self.product_a.product.product_id, store=self.store)
        first = detector.detect(TEST_GOOD / "000.png")
        second = detector.detect(TEST_GOOD / "000.png")
        self.assertEqual(first.image_score, second.image_score)
        self.assertEqual(first.image_status, second.image_status)
        self.assertTrue((first.anomaly_heatmap == second.anomaly_heatmap).all())
        self.assertEqual(first.image_sha256, second.image_sha256)

    # 5
    def test_05_products_isolated(self) -> None:
        a_before = self._file_hashes(self.product_a.product.product_id)
        a = ProductDetector(self.product_a.product.product_id, store=self.store)
        b = ProductDetector(self.product_b.product.product_id, store=self.store)
        self.assertEqual(a.product_id, self.product_a.product.product_id)
        self.assertEqual(b.product_id, self.product_b.product.product_id)
        self.assertNotEqual(a.global_bank["metadata"]["member_manifest_sha256"],
                            b.global_bank["metadata"]["member_manifest_sha256"])
        a_result = a.detect(TEST_GOOD / "000.png")
        b_result = b.detect(TEST_GOOD / "000.png")
        self.assertEqual(a_result.product_id, a.product_id)
        self.assertEqual(b_result.product_id, b.product_id)
        self.assertEqual(a_result.image_score, self.baseline_good.image_score)
        self.assertEqual(self._file_hashes(self.product_a.product.product_id), a_before)

    # 6
    def test_06_missing_product_rejected(self) -> None:
        with self.assertRaises(ProductDetectorError):
            ProductDetector("f" * 32, store=self.store)

    # 7
    def test_07_status_rejected(self) -> None:
        draft = self.store.create_product("草稿产品")
        with self.assertRaises(ProductDetectorError):
            ProductDetector(draft.product_id, store=self.store)

        disabled = self.store.create_product("停用产品")
        self.store.update_product_status(disabled.product_id, "disabled")
        with self.assertRaises(ProductDetectorError):
            ProductDetector(disabled.product_id, store=self.store)

        archived = self.store.create_product("归档产品")
        self.store.update_product_status(archived.product_id, "archived")
        with self.assertRaises(ProductDetectorError):
            ProductDetector(archived.product_id, store=self.store)

    # 8
    def test_08_manifest_corruption_rejected(self) -> None:
        copy_root = TESTS_ROOT / f"p31_corrupt_{uuid.uuid4().hex[:8]}"
        try:
            pid = self.product_a.product.product_id
            shutil.copytree(self._product_dir(pid), copy_root / pid)
            feature = copy_root / pid / "versions" / "v001" / "reference" / "global_reference" / "features.npy"
            feature.write_bytes(feature.read_bytes() + b"corrupt")
            store = ProductStore(copy_root)
            with self.assertRaises(ProductDetectorError):
                ProductDetector(pid, store=store)
        finally:
            shutil.rmtree(copy_root, ignore_errors=True)

    # 9
    def test_09_missing_bank_rejected(self) -> None:
        copy_root = TESTS_ROOT / f"p31_nobank_{uuid.uuid4().hex[:8]}"
        try:
            pid = self.product_a.product.product_id
            shutil.copytree(self._product_dir(pid), copy_root / pid)
            shutil.rmtree(copy_root / pid / "versions" / "v001" / "reference" / "spatial_reference")
            store = ProductStore(copy_root)
            with self.assertRaises(ProductDetectorError):
                ProductDetector(pid, store=store)
        finally:
            shutil.rmtree(copy_root, ignore_errors=True)

    # 10
    def test_10_missing_threshold_rejected(self) -> None:
        product = self.store.create_product("无阈值产品")
        version = self.store.create_version(product.product_id)
        self.store.set_current_version(product.product_id, version.version)
        with self.assertRaises(ProductDetectorError) as context:
            ProductDetector(product.product_id, store=self.store)
        self.assertIn("阈值", str(context.exception))

    # 11
    def test_11_weights_missing_rejected(self) -> None:
        with mock.patch(
            "src.core.product_detector.ensure_local_weights",
            side_effect=WeightsUnavailableError("weights missing"),
        ):
            with self.assertRaises(ProductDetectorError):
                ProductDetector(self.product_a.product.product_id, store=self.store)

    # 12
    def test_12_invalid_image_rejected(self) -> None:
        detector = ProductDetector(self.product_a.product.product_id, store=self.store)
        with self.assertRaises(ProductDetectorError):
            detector.detect(self.root / "not_exists.png")
        bad = self.root / "bad.png"
        bad.write_bytes(b"not an image")
        with self.assertRaises(ProductDetectorError):
            detector.detect(bad)
        wrong = self.root / "image.gif"
        wrong.write_bytes(b"GIF89a")
        with self.assertRaises(ProductDetectorError):
            detector.detect(wrong)

    # 13
    def test_13_failure_does_not_modify_product(self) -> None:
        pid = self.product_a.product.product_id
        before = self._file_hashes(pid)
        detector = ProductDetector(pid, store=self.store)
        with self.assertRaises(ProductDetectorError):
            detector.detect(self.root / "missing.png")
        self.assertEqual(self._file_hashes(pid), before)
        self.assertEqual(self.store.get_product(pid).current_version, 1)

    # 14
    def test_14_cli_end_to_end(self) -> None:
        pid = self.product_a.product.product_id
        before = self._file_hashes(pid)
        out = self.root / f"cli_{uuid.uuid4().hex[:6]}"
        results = []
        for image in (TEST_GOOD / "000.png", TEST_DEFECT / "000.png"):
            process = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts/detect_product.py"),
                 "--product-id", pid,
                 "--image", str(image),
                 "--output-dir", str(out / image.stem),
                 "--root", str(self.root),
                 "--device", "cpu"],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=300,
            )
            self.assertEqual(process.returncode, 0, process.stderr[-1000:])
            payload = json.loads(process.stdout[process.stdout.index("{"):])
            self.assertTrue(payload["ok"])
            self.assertTrue(Path(payload["files"]["result_json"]).is_file())
            results.append(payload)
        self.assertEqual(results[0]["image_status"], "normal")
        self.assertEqual(results[1]["image_status"], "anomaly")
        self.assertEqual(results[0]["product_id"], results[1]["product_id"])
        self.assertEqual(results[0]["version"], 1)
        for payload in results:
            self.assertIn("bank_version", payload)
            self.assertTrue(payload["bank_version"])
            self.assertIn("input_image_sha256", payload)
            self.assertEqual(payload["input_image_sha256"], payload["image_sha256"])
            self.assertIn("inference_time", payload)
            self.assertGreaterEqual(payload["inference_time"], 0.0)
        self.assertEqual(self._file_hashes(pid), before)

    def _rebuild_manifest(self, store: ProductStore, product_id: str, version: int) -> None:
        version_dir = store.version_directory(product_id, version)
        manifest = store._build_manifest(version_dir, product_id, version)
        (version_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        digest = sha256_file(version_dir / "manifest.json")
        (version_dir / "manifest.sha256").write_text(digest + "\n", encoding="ascii")

    # 15
    def test_15_registering_c_does_not_change_a(self) -> None:
        pid = self.product_a.product.product_id
        before = self._file_hashes(pid)
        register_product(
            self.store,
            RegistrationRequest(
                name="产品C", category="screw",
                reference_paths=_paths(TRAIN, 90, 5),
                calibration_paths=_paths(TRAIN, 95, 25),
            ),
            device="auto",
        )
        self.assertEqual(self._file_hashes(pid), before)
        detector = ProductDetector(pid, store=self.store)
        result = detector.detect(TEST_GOOD / "000.png")
        self.assertEqual(result.image_score, self.baseline_good.image_score)

    # 16
    def test_16_bank_feature_hash_mismatch_rejected(self) -> None:
        copy_root = TESTS_ROOT / f"p31_bankhash_{uuid.uuid4().hex[:8]}"
        try:
            pid = self.product_a.product.product_id
            shutil.copytree(self._product_dir(pid), copy_root / pid)
            store = ProductStore(copy_root)
            feature = copy_root / pid / "versions" / "v001" / "reference" / "global_reference" / "features.npy"
            array = np.load(feature)
            array[0, 0] = array[0, 0] + 1.0
            np.save(feature, array, allow_pickle=False)
            self._rebuild_manifest(store, pid, 1)
            with self.assertRaises(ProductDetectorError):
                ProductDetector(pid, store=store)
        finally:
            shutil.rmtree(copy_root, ignore_errors=True)

    # 17
    def test_17_illegal_threshold_rejected(self) -> None:
        copy_root = TESTS_ROOT / f"p31_badthr_{uuid.uuid4().hex[:8]}"
        try:
            pid = self.product_a.product.product_id
            shutil.copytree(self._product_dir(pid), copy_root / pid)
            store = ProductStore(copy_root)
            threshold_path = copy_root / pid / "versions" / "v001" / "calibration" / "thresholds.json"
            payload = json.loads(threshold_path.read_text(encoding="utf-8"))
            payload["image_threshold"] = float("nan")
            threshold_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self._rebuild_manifest(store, pid, 1)
            with self.assertRaises(ProductDetectorError) as context:
                ProductDetector(pid, store=store)
            self.assertIn("image_threshold", str(context.exception))
        finally:
            shutil.rmtree(copy_root, ignore_errors=True)

    def _worker_cmd(self, *args: str) -> list[str]:
        return [
            sys.executable,
            str(PROJECT_ROOT / "tests" / "product_detector_worker.py"),
            *args,
        ]

    def _wait_file(self, path: Path, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if path.is_file():
                return
            time.sleep(0.02)
        self.fail(f"等待文件超时：{path}")

    def _run_worker(self, *args: str, timeout: float = 120.0) -> dict:
        result_path = self.root / f"worker_{uuid.uuid4().hex}.json"
        process = subprocess.run(
            self._worker_cmd(*args, "--result", str(result_path)),
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        self.assertTrue(result_path.is_file(), process.stderr[-1000:])
        return json.loads(result_path.read_text(encoding="utf-8"))

    def _start_holder(self, product_id: str):
        token = uuid.uuid4().hex
        sentinel = self.root / f"hold_{token}.sentinel"
        release = self.root / f"hold_{token}.release"
        result = self.root / f"hold_{token}.json"
        process = subprocess.Popen(
            self._worker_cmd(
                "--root", str(self.root), "--product-id", product_id,
                "--mode", "hold", "--result", str(result),
                "--sentinel", str(sentinel), "--release", str(release),
                "--lock-timeout", "30",
            ),
            cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_file(sentinel)
        return process, release, result

    def _release_holder(self, holder, timeout: float = 60.0) -> dict:
        process, release, result = holder
        release.write_text("release", encoding="utf-8")
        process.wait(timeout=timeout)
        self.assertEqual(process.returncode, 0)
        return json.loads(result.read_text(encoding="utf-8"))

    def _bank_version_from_metadata(self, product_id: str, version: int) -> str:
        version_dir = self.store.version_directory(product_id, version)
        g_meta = json.loads(
            (version_dir / "reference" / "global_reference" / "metadata.json")
            .read_text(encoding="utf-8")
        )
        s_meta = json.loads(
            (version_dir / "reference" / "spatial_reference" / "metadata.json")
            .read_text(encoding="utf-8")
        )
        return (
            f"{g_meta.get('protocol_version')}"
            f":g={g_meta.get('member_manifest_sha256')}"
            f":s={s_meta.get('member_manifest_sha256')}"
        )

    # 18
    def test_18_bank_version_from_loaded_metadata(self) -> None:
        pid = self.product_a.product.product_id
        result = self.detector_a.detect(TEST_GOOD / "000.png")
        expected = self._bank_version_from_metadata(pid, 1)
        self.assertEqual(result.bank_version, expected)
        self.assertEqual(result.memory_bank["bank_version"], expected)
        payload = result.to_dict()
        self.assertEqual(payload["bank_version"], expected)
        out = self.root / f"bank_{uuid.uuid4().hex[:6]}"
        result.save(out)
        saved = json.loads(next(out.glob("*_detect.json")).read_text(encoding="utf-8"))
        self.assertEqual(saved["bank_version"], expected)

    def test_18b_bank_metadata_mismatch_rejected(self) -> None:
        copy_root = TESTS_ROOT / f"p311_bankmeta_{uuid.uuid4().hex[:8]}"
        try:
            pid = self.product_a.product.product_id
            shutil.copytree(self._product_dir(pid), copy_root / pid)
            store = ProductStore(copy_root)
            meta_path = (copy_root / pid / "versions" / "v001" / "reference"
                         / "global_reference" / "metadata.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["product_version"] = 99
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            self._rebuild_manifest(store, pid, 1)
            with self.assertRaises(ProductDetectorError):
                ProductDetector(pid, store=store)
        finally:
            shutil.rmtree(copy_root, ignore_errors=True)

    # 19
    def test_19_input_image_sha256_standard_field(self) -> None:
        pid = self.product_a.product.product_id
        image_path = TEST_GOOD / "000.png"
        result = self.detector_a.detect(image_path)
        expected = sha256_file(image_path)
        self.assertEqual(result.input_image_sha256, expected)
        self.assertEqual(result.image_sha256, expected)
        self.assertEqual(result.to_dict()["input_image_sha256"], expected)

        changed = self.root / f"changed_{uuid.uuid4().hex[:6]}.png"
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        image.putpixel((0, 0), (1, 2, 3))
        image.save(changed)
        changed_result = ProductDetector(pid, store=self.store).detect(changed)
        self.assertNotEqual(changed_result.input_image_sha256, expected)
        self.assertEqual(changed_result.input_image_sha256, sha256_file(changed))
        self.assertEqual(changed_result.input_image_sha256, changed_result.image_sha256)

    # 20
    def test_20_inference_time_standard_field(self) -> None:
        result = self.detector_a.detect(TEST_GOOD / "000.png")
        self.assertTrue(np.isfinite(result.inference_time))
        self.assertGreaterEqual(result.inference_time, 0.0)
        self.assertEqual(result.inference_time, result.elapsed_time)
        payload = result.to_dict()
        self.assertEqual(payload["inference_time"], result.inference_time)
        self.assertEqual(payload["elapsed_time"], result.elapsed_time)

    # 21
    def test_21_concurrent_switch_and_detect_no_mixed_version(self) -> None:
        pid = self.product_s.product.product_id
        self.store.set_current_version(pid, 1)
        before = {
            k: v for k, v in self._file_hashes(pid).items()
            if k.startswith("versions/")
        }
        holder = self._start_holder(pid)
        token = uuid.uuid4().hex
        detect_started = self.root / f"detect_{token}.started"
        switch_started = self.root / f"switch_{token}.started"
        detect_result = self.root / f"detect_{token}.json"
        switch_result = self.root / f"switch_{token}.json"
        detect_proc = subprocess.Popen(
            self._worker_cmd(
                "--root", str(self.root), "--product-id", pid, "--mode", "detect",
                "--result", str(detect_result),
                "--image", str(TEST_GOOD / "000.png"),
                "--output-dir", str(self.root / f"detect_{token}"),
                "--lock-timeout", "20", "--started", str(detect_started),
            ),
            cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._wait_file(detect_started)
        switch_proc = subprocess.Popen(
            self._worker_cmd(
                "--root", str(self.root), "--product-id", pid, "--mode", "switch",
                "--version", "2", "--result", str(switch_result),
                "--lock-timeout", "20", "--started", str(switch_started),
            ),
            cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._wait_file(switch_started)
        self._release_holder(holder)
        detect_proc.wait(timeout=120)
        switch_proc.wait(timeout=120)
        self.assertEqual(detect_proc.returncode, 0)
        self.assertEqual(switch_proc.returncode, 0)
        detect_payload = json.loads(detect_result.read_text(encoding="utf-8"))
        self.assertTrue(detect_payload["ok"], detect_payload)
        version = detect_payload["version"]
        self.assertIn(version, (1, 2))
        version_record = self.store.get_version(pid, version)
        self.assertEqual(detect_payload["manifest_sha256"], version_record.manifest_sha256)
        self.assertEqual(detect_payload["bank_version"],
                         self._bank_version_from_metadata(pid, version))
        # 切换完成后新检测必须使用新的 current_version
        self.assertEqual(self.store.get_product(pid).current_version, 2)
        after_switch = ProductDetector(pid, store=self.store).detect(TEST_GOOD / "000.png")
        self.assertEqual(after_switch.version, 2)
        # 历史版本文件未被检测/切换改写
        after = {
            k: v for k, v in self._file_hashes(pid).items()
            if k.startswith("versions/")
        }
        self.assertEqual(after, before)

    # 22
    def test_22_lock_timeout_blocks_detection_without_fallback(self) -> None:
        pid = self.product_a.product.product_id
        holder = self._start_holder(pid)
        try:
            payload = self._run_worker(
                "--root", str(self.root), "--product-id", pid, "--mode", "detect",
                "--image", str(TEST_GOOD / "000.png"),
                "--output-dir", str(self.root / f"timeout_{uuid.uuid4().hex[:6]}"),
                "--lock-timeout", "0.5",
            )
        finally:
            self._release_holder(holder)
        self.assertFalse(payload["ok"], payload)
        self.assertEqual(payload["error_type"], "ProductDetectorError")
        self.assertTrue("锁" in payload["error"] or "超时" in payload["error"], payload)

    # 23
    def test_23_different_products_do_not_block(self) -> None:
        holder = self._start_holder(self.product_a.product.product_id)
        try:
            payload = self._run_worker(
                "--root", str(self.root),
                "--product-id", self.product_b.product.product_id,
                "--mode", "detect",
                "--image", str(TEST_GOOD / "000.png"),
                "--output-dir", str(self.root / f"other_{uuid.uuid4().hex[:6]}"),
                "--lock-timeout", "5",
            )
        finally:
            self._release_holder(holder)
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["version"], 1)

    # 24
    def test_24_switch_then_detect_uses_new_current(self) -> None:
        pid = self.product_s.product.product_id
        self.store.set_current_version(pid, 1)
        v1 = ProductDetector(pid, store=self.store).detect(TEST_GOOD / "000.png")
        self.assertEqual(v1.version, 1)
        self.store.set_current_version(pid, 2)
        v2 = ProductDetector(pid, store=self.store).detect(TEST_GOOD / "000.png")
        self.assertEqual(v2.version, 2)
        self.assertNotEqual(v1.manifest_sha256, v2.manifest_sha256)
        self.assertNotEqual(v1.bank_version, v2.bank_version)


    def test_90_current_version_and_legacy_default(self) -> None:
        pid = self.product_a.product.product_id
        v1_before = self._file_hashes(pid)
        self._v2 = register_product(
            self.store,
            RegistrationRequest(
                name="ignored", product_id=pid,
                reference_paths=_paths(TRAIN, 30, 5),
                calibration_paths=_paths(TRAIN, 35, 25),
            ),
            device="auto",
        )
        self.assertEqual(self.store.get_product(pid).current_version, 2)
        detector_v2 = ProductDetector(pid, store=self.store)
        result_v2 = detector_v2.detect(TEST_GOOD / "000.png")
        self.assertEqual(result_v2.version, 2)
        self.assertNotEqual(result_v2.manifest_sha256, self.baseline_good.manifest_sha256)

        self.store.set_current_version(pid, 1)
        detector_v1 = ProductDetector(pid, store=self.store)
        result_v1 = detector_v1.detect(TEST_GOOD / "000.png")
        self.assertEqual(result_v1.version, 1)
        self.assertEqual(result_v1.manifest_sha256, self.baseline_good.manifest_sha256)
        self.assertEqual(result_v1.image_score, self.baseline_good.image_score)
        # v1 文件未被 v2 注册或指针切换改写
        v1_after = {
            k: v for k, v in self._file_hashes(pid).items()
            if k.startswith("versions/v001/")
        }
        v1_before_only = {
            k: v for k, v in v1_before.items() if k.startswith("versions/v001/")
        }
        self.assertEqual(v1_after, v1_before_only)


if __name__ == "__main__":
    unittest.main(verbosity=2)
