#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-2.1 MVTec AD 数据读取模块测试。

运行：
    D:\\microdefect\\.venv\\Scripts\\python.exe -m unittest tests.test_mvtec_dataset -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.data.mvtec_dataset import (  # noqa: E402
    CATEGORIES,
    MVTecDataset,
    MissingMaskError,
    UnknownCategoryError,
)

DATA_ROOT = PROJECT_ROOT / "datasets" / "mvtec_ad"
AUDIT_JSON = PROJECT_ROOT / "experiments" / "results" / "mvtec_ad" / "stage3_audit" / "audit_report.json"

EXPECTED_TRAIN = {
    "bottle": 209, "cable": 224, "capsule": 219, "carpet": 280, "grid": 264,
    "hazelnut": 391, "leather": 245, "metal_nut": 220, "pill": 267, "screw": 320,
    "tile": 230, "toothbrush": 60, "transistor": 213, "wood": 247, "zipper": 240,
}
EXPECTED_TEST_GOOD = {
    "bottle": 20, "cable": 58, "capsule": 23, "carpet": 28, "grid": 21,
    "hazelnut": 40, "leather": 32, "metal_nut": 22, "pill": 26, "screw": 41,
    "tile": 33, "toothbrush": 12, "transistor": 60, "wood": 19, "zipper": 32,
}
EXPECTED_MASKS = {
    "bottle": 63, "cable": 92, "capsule": 109, "carpet": 89, "grid": 57,
    "hazelnut": 70, "leather": 92, "metal_nut": 93, "pill": 141, "screw": 119,
    "tile": 84, "toothbrush": 30, "transistor": 40, "wood": 60, "zipper": 119,
}


class MVTecDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = MVTecDataset(DATA_ROOT)

    def test_01_categories(self) -> None:
        self.assertEqual(self.dataset.list_categories(), CATEGORIES)
        self.assertEqual(len(CATEGORIES), 15)
        self.assertEqual(len(set(CATEGORIES)), 15)

    def test_02_train_counts_match_audit(self) -> None:
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        total = 0
        for category in CATEGORIES:
            files = self.dataset.list_train_good(category)
            self.assertEqual(len(files), EXPECTED_TRAIN[category], category)
            self.assertEqual(len(files), audit["per_class"][category]["train_good"], category)
            self.assertTrue(all(p.suffix.lower() == ".png" for p in files))
            self.assertEqual(files, sorted(files, key=lambda p: p.name), f"{category} 排序不稳定")
            total += len(files)
        self.assertEqual(total, 3629)

    def test_03_test_counts(self) -> None:
        total_good = total_defect = 0
        for category in CATEGORIES:
            self.assertEqual(len(self.dataset.list_test_good(category)), EXPECTED_TEST_GOOD[category])
            defects = self.dataset.list_test_defects(category)
            self.assertNotIn("good", defects)
            total_good += len(self.dataset.list_test_good(category))
            total_defect += sum(len(v) for v in defects.values())
        self.assertEqual(total_good, 467)
        self.assertEqual(total_defect, 1258)

    def test_04_mask_pairing(self) -> None:
        paired = 0
        for category in CATEGORIES:
            masks = self.dataset.list_ground_truth(category)
            self.assertEqual(sum(len(v) for v in masks.values()), EXPECTED_MASKS[category])
            for defect, images in self.dataset.list_test_defects(category).items():
                for image in images:
                    mask = self.dataset.get_ground_truth(image)
                    self.assertIsNotNone(mask)
                    assert mask is not None
                    self.assertTrue(mask.is_file())
                    self.assertEqual(mask.stem, f"{image.stem}_mask")
                    self.assertIn(f"ground_truth/{defect}/", mask.as_posix())
                    paired += 1
        self.assertEqual(paired, 1258)

    def test_05_normal_test_has_no_mask(self) -> None:
        for category in CATEGORIES:
            for image in self.dataset.list_test_good(category):
                self.assertIsNone(self.dataset.get_ground_truth(image))

    def test_06_all_images_decode(self) -> None:
        failures = []
        count = 0
        for category in CATEGORIES:
            paths = (
                self.dataset.list_train_good(category)
                + self.dataset.list_test_images(category)
                + [m for v in self.dataset.list_ground_truth(category).values() for m in v]
            )
            for path in paths:
                count += 1
                try:
                    with Image.open(path) as im:
                        im.verify()
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{path}: {type(exc).__name__}")
        self.assertEqual(count, 6612)
        self.assertEqual(failures, [])

    def test_07_paths_unique_and_inside_root(self) -> None:
        seen: set[str] = set()
        root = self.dataset.root
        for category in CATEGORIES:
            paths = self.dataset.list_train_good(category) + self.dataset.list_test_images(category)
            for path in paths:
                rel = self.dataset.relative(path)
                self.assertNotIn(rel, seen)
                seen.add(rel)
                self.assertTrue(str(path.resolve()).startswith(str(root)))
                self.assertNotIn("..", rel)

    def test_08_load_image_preserves_mode(self) -> None:
        gray = self.dataset.load_image(self.dataset.list_train_good("bottle")[0])
        self.assertIn(gray.mode, ("L", "RGB"))
        rgb = self.dataset.to_rgb(gray)
        self.assertEqual(rgb.mode, "RGB")
        small = self.dataset.resize(rgb, 224)
        self.assertEqual(small.size, (224, 224))

    def test_09_unknown_category(self) -> None:
        with self.assertRaises(UnknownCategoryError):
            self.dataset.list_train_good("not_a_category")

    def test_10_missing_mask_detection(self) -> None:
        missing = self.dataset.root / "bottle" / "test" / "broken_large" / "does_not_exist.png"
        with self.assertRaises(MissingMaskError):
            self.dataset.mask_path_for(missing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
