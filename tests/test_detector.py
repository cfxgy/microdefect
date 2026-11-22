#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单图检测器与记忆库注册表测试（第一阶段）。"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.core.detector import Detector, DetectorError  # noqa: E402
from src.core.registry import (  # noqa: E402
    GROUP_DISJOINT_CALIBRATION,
    GROUP_DISJOINT_CATEGORIES,
    MemoryRegistry,
    RegistryError,
)

SPLIT = PROJECT_ROOT / "experiments/results/mvtec_ad/stage5_3_1_group_disjoint_splits/group_disjoint_splits.csv"
DATA = PROJECT_ROOT / "datasets/mvtec_ad"


def sample_images() -> dict[str, Path]:
    samples: dict[str, Path] = {}
    with SPLIT.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["group_assignment"] == "test" and row["normal_or_anomaly"] == "normal":
                samples.setdefault(row["category"], DATA / row["image_path"])
    return samples


class RegistryTest(unittest.TestCase):
    def test_01_group_disjoint_categories(self) -> None:
        registry = MemoryRegistry(protocol="group_disjoint")
        categories = registry.available_categories()
        self.assertEqual(categories, list(GROUP_DISJOINT_CATEGORIES))
        for category in categories:
            config = registry.load_config(category)
            self.assertEqual(config.protocol, "group_disjoint")
            self.assertEqual(config.global_shape, (5, 512))
            self.assertEqual(config.spatial_shape, (5, 256, 14, 14))
            self.assertTrue(np.isfinite(config.image_threshold))
            self.assertTrue(np.isfinite(config.pixel_threshold))
            self.assertTrue(0.0 <= config.image_threshold <= 2.0)
            self.assertTrue(0.0 <= config.pixel_threshold <= 2.0)
            self.assertTrue(config.calibration_path.is_file())
            self.assertNotEqual(config.global_member_sha256, "")
            self.assertNotEqual(config.spatial_member_sha256, "")

    def test_02_official_protocol(self) -> None:
        registry = MemoryRegistry(protocol="official", seed=42, shot=5)
        config = registry.load_config("bottle")
        self.assertEqual(config.protocol, "official")
        self.assertEqual(config.global_shape[1], 512)
        self.assertEqual(len(config.global_shape), 2)
        self.assertEqual(config.spatial_shape[1:], (256, 14, 14))

    def test_03_unknown_category(self) -> None:
        registry = MemoryRegistry(protocol="group_disjoint")
        with self.assertRaises(RegistryError):
            registry.load_config("not_a_class")

    def test_04_missing_bank_root(self) -> None:
        empty = PROJECT_ROOT / "cache" / "tests" / f"empty_banks_{uuid.uuid4().hex[:8]}"
        empty.mkdir(parents=True, exist_ok=True)
        try:
            registry = MemoryRegistry(protocol="group_disjoint", bank_root=empty)
            with self.assertRaises(RegistryError):
                registry.load_config("bottle")
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_05_bad_threshold_rejected(self) -> None:
        tmp = PROJECT_ROOT / "cache" / "tests" / f"bad_cal_{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            source = GROUP_DISJOINT_CALIBRATION / "bottle_calibration.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["image_threshold"] = float("nan")
            (tmp / "bottle_calibration.json").write_text(json.dumps(payload), encoding="utf-8")
            registry = MemoryRegistry(
                protocol="group_disjoint", calibration_root=tmp
            )
            with self.assertRaises(RegistryError):
                registry.load_config("bottle")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class DetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = sample_images()
        cls.detector = Detector("bottle", protocol="group_disjoint", device="cpu")

    def test_06_detect_all_categories(self) -> None:
        for category, image_path in self.samples.items():
            detector = self.detector if category == "bottle" else Detector(
                category, protocol="group_disjoint", device="cpu"
            )
            result = detector.detect(image_path)
            self.assertEqual(result.category, category)
            self.assertEqual(result.protocol, "group_disjoint")
            self.assertGreaterEqual(result.image_score, 0.0)
            self.assertLessEqual(result.image_score, 2.0)
            self.assertIsInstance(result.is_anomaly, bool)
            self.assertTrue(np.isfinite(result.anomaly_heatmap).all())
            with Image.open(image_path) as handle:
                width, height = handle.size
            self.assertEqual(result.anomaly_heatmap.shape, (height, width))
            self.assertEqual(result.patch_distance_map.shape, (14, 14))
            self.assertGreater(result.elapsed_time, 0.0)

    def test_07_repeat_stable(self) -> None:
        image = self.samples["bottle"]
        first = self.detector.detect(image)
        second = self.detector.detect(image)
        self.assertEqual(first.image_score, second.image_score)
        self.assertEqual(first.is_anomaly, second.is_anomaly)
        np.testing.assert_array_equal(first.anomaly_heatmap, second.anomaly_heatmap)

    def test_08_result_fields(self) -> None:
        result = self.detector.detect(self.samples["bottle"])
        payload = result.to_dict()
        for key in ("image_path", "category", "image_score", "image_threshold", "is_anomaly",
                    "pixel_threshold", "image_size", "model_version", "memory_bank_version",
                    "elapsed_time", "heatmap_stats"):
            self.assertIn(key, payload)
        self.assertEqual(payload["memory_bank_version"]["shot"], 5)
        self.assertEqual(payload["memory_bank_version"]["global_member_count"], 5)
        self.assertTrue(payload["model_version"]["weight_sha256"])
        json.dumps(payload)  # 可序列化

    def test_09_save_outputs(self) -> None:
        out = PROJECT_ROOT / "cache" / "tests" / f"detect_out_{uuid.uuid4().hex[:8]}"
        try:
            result = self.detector.detect(self.samples["bottle"])
            files = result.save(out)
            for key in ("result_json", "heatmap_png", "overlay_png"):
                path = Path(files[key])
                self.assertTrue(path.is_file(), key)
            payload = json.loads(Path(files["result_json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["category"], "bottle")
            with Image.open(self.samples["bottle"]) as handle:
                size = handle.size
            with Image.open(files["heatmap_png"]) as heat, Image.open(files["overlay_png"]) as overlay:
                self.assertEqual(heat.size, size)
                self.assertEqual(overlay.size, size)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_10_missing_image(self) -> None:
        with self.assertRaises(DetectorError):
            self.detector.detect(DATA / "bottle" / "does_not_exist.png")

    def test_11_cli(self) -> None:
        out = PROJECT_ROOT / "cache" / "tests" / f"cli_out_{uuid.uuid4().hex[:8]}"
        try:
            process = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts/run_detect.py"),
                 "--image", str(self.samples["bottle"]),
                 "--category", "bottle",
                 "--out", str(out),
                 "--device", "cpu"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=300,
            )
            self.assertEqual(process.returncode, 0, process.stderr[-500:])
            payload = json.loads(process.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(Path(payload["files"]["result_json"]).is_file())
            self.assertTrue(Path(payload["files"]["heatmap_png"]).is_file())
            self.assertTrue(Path(payload["files"]["overlay_png"]).is_file())
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
