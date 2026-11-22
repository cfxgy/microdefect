#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-2.3.1 记忆库模块测试。"""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.data.mvtec_dataset import MVTecDataset  # noqa: E402
from src.data.mvtec_split import build_split  # noqa: E402
from src.memory.mvtec_memory import (  # noqa: E402
    MemoryBankCorruptedError,
    MemoryBankError,
    feature_content_sha256,
    load_bank,
    member_manifest_sha256,
    save_bank,
    verify_bank,
)

DATA = PROJECT_ROOT / "datasets" / "mvtec_ad"
EXTRACTOR_INFO = {
    "backbone": "resnet18",
    "weights_tag": "IMAGENET1K_V1",
    "weight_sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "global_feature_version": "f1/resnet18-gap512-raw-fp32-notf32",
    "spatial_feature_version": "f1/resnet18-layer3-256x14x14-raw-fp32-notf32",
    "preprocess_version": "p1/gray2rgb-resize224-bilinear-antialias-imagenorm",
    "image_size": 224,
}


def member_rows(paths, category, dataset, shot, role):
    rows = []
    for index, path in enumerate(paths):
        rows.append({
            "rel_path": dataset.relative(path),
            "image_sha256": f"{index:064x}",  # 测试用占位内容哈希
            "category": category,
            "role": role,
            "shot": shot,
            "order_index": index,
            "seed": 42,
        })
    return rows


class MemoryBankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = PROJECT_ROOT / "cache" / "tests" / f"memory_bank_{uuid.uuid4().hex[:8]}"
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.dataset = MVTecDataset(DATA)
        cls.split = build_split(cls.dataset, "bottle", 42)
        cls.rng = np.random.default_rng(0)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _save_global(self, name, paths, shot, role="reference", value=0.0):
        directory = self.root / name
        features = self.rng.normal(size=(len(paths), 512)).astype(np.float32) + value
        members = member_rows(paths, "bottle", self.dataset, shot, role)
        meta = save_bank(
            directory, features, members,
            category="bottle", seed=42, shot=shot,
            memory_type=f"global_{role}", extractor_info=EXTRACTOR_INFO,
        )
        return directory, features, members, meta

    def test_01_global_reference_roundtrip(self) -> None:
        directory, features, members, meta = self._save_global("g_ref_1", self.split.reference[1], 1)
        loaded = load_bank(directory)
        self.assertTrue(np.array_equal(loaded["features"], features))
        self.assertEqual(loaded["metadata"]["member_count"], 1)
        self.assertEqual(loaded["members"][0]["rel_path"], members[0]["rel_path"])
        self.assertTrue(meta["feature_content_sha256"] == feature_content_sha256(features))

    def test_02_global_extended_and_full(self) -> None:
        directory_ext, features_ext, _, _ = self._save_global(
            "g_ext_1", self.split.extended[1], 1, role="extended")
        directory_full, features_full, members_full, _ = self._save_global(
            "g_full_1", self.split.memory_full[1], 1, role="full")
        self.assertEqual(load_bank(directory_ext)["features"].shape[1], 512)
        self.assertEqual(load_bank(directory_full)["features"].shape[0], len(members_full))
        self.assertEqual(
            member_manifest_sha256(load_bank(directory_full)["members"]),
            member_manifest_sha256(members_full),
        )

    def test_03_spatial_reference(self) -> None:
        directory = self.root / "s_ref_5"
        paths = self.split.reference[5]
        features = self.rng.normal(size=(5, 256, 14, 14)).astype(np.float32)
        members = member_rows(paths, "bottle", self.dataset, 5, "reference")
        save_bank(
            directory, features, members,
            category="bottle", seed=42, shot=5,
            memory_type="spatial_reference", extractor_info=EXTRACTOR_INFO,
        )
        loaded = load_bank(directory)
        self.assertEqual(loaded["features"].shape, (5, 256, 14, 14))
        self.assertEqual(loaded["features"].dtype, np.float32)
        self.assertTrue(np.array_equal(loaded["features"], features))
        self.assertTrue(np.isfinite(loaded["features"]).all())

    def test_04_no_calibration_no_test(self) -> None:
        cal = {p.as_posix() for p in self.split.calibration}
        test_images = set()
        for category_files in self.dataset.list_ground_truth("bottle").values():
            test_images |= {p.as_posix() for p in category_files}
        for k in (1, 3, 5):
            full = {p.as_posix() for p in self.split.memory_full[k]}
            self.assertTrue(full.isdisjoint(cal))
            self.assertTrue(all("/train/good/" in p for p in full))

    def test_05_full_members_identical_across_shots(self) -> None:
        hashes = {
            k: member_manifest_sha256([
                {"rel_path": self.dataset.relative(p), "image_sha256": p.name}
                for p in self.split.memory_full[k]
            ])
            for k in (1, 3, 5)
        }
        self.assertEqual(len(set(hashes.values())), 1)

    def test_06_order_traceability(self) -> None:
        directory, features, members, _ = self._save_global(
            "g_order", self.split.memory_full[5], 5, role="full")
        loaded = load_bank(directory)
        self.assertEqual([m["order_index"] for m in loaded["members"]], list(range(len(members))))
        self.assertEqual([m["rel_path"] for m in loaded["members"]], [m["rel_path"] for m in members])

    def test_07_corruption_detected(self) -> None:
        directory, _, _, _ = self._save_global("g_corrupt", self.split.reference[3], 3)
        (directory / "features.npy").write_bytes(b"broken")
        self.assertFalse(verify_bank(directory)["ok"])
        with self.assertRaises(MemoryBankCorruptedError):
            load_bank(directory)

    def test_08_truncated_metadata(self) -> None:
        directory, _, _, _ = self._save_global("g_meta", self.split.reference[1], 1)
        (directory / "metadata.json").write_text("{bad json", encoding="utf-8")
        self.assertFalse(verify_bank(directory)["ok"])

    def test_09_refuse_mismatched_overwrite(self) -> None:
        directory, _, _, _ = self._save_global("g_conflict", self.split.reference[1], 1)
        with self.assertRaises(MemoryBankError):
            save_bank(
                directory, np.zeros((1, 512), dtype=np.float32),
                member_rows(self.split.reference[1], "bottle", self.dataset, 3, "reference"),
                category="bottle", seed=42, shot=3,
                memory_type="global_reference", extractor_info=EXTRACTOR_INFO,
            )

    def test_10_rebuild_same_config_allowed(self) -> None:
        directory, features, members, _ = self._save_global("g_rebuild", self.split.reference[1], 1)
        save_bank(
            directory, features, members,
            category="bottle", seed=42, shot=1,
            memory_type="global_reference", extractor_info=EXTRACTOR_INFO,
        )
        self.assertTrue(verify_bank(directory)["ok"])

    def test_11_empty_and_bad_shape_rejected(self) -> None:
        with self.assertRaises(MemoryBankError):
            save_bank(
                self.root / "empty", np.zeros((0, 512), dtype=np.float32), [],
                category="bottle", seed=42, shot=1,
                memory_type="global_reference", extractor_info=EXTRACTOR_INFO,
            )
        with self.assertRaises(MemoryBankError):
            save_bank(
                self.root / "badshape", np.zeros((2, 256), dtype=np.float32),
                member_rows(self.split.reference[1], "bottle", self.dataset, 1, "reference"),
                category="bottle", seed=42, shot=1,
                memory_type="global_reference", extractor_info=EXTRACTOR_INFO,
            )

    def test_12_category_mismatch_rejected(self) -> None:
        members = member_rows(self.split.reference[1], "cable", self.dataset, 1, "reference")
        with self.assertRaises(MemoryBankError):
            save_bank(
                self.root / "cross", np.zeros((1, 512), dtype=np.float32), members,
                category="bottle", seed=42, shot=1,
                memory_type="global_reference", extractor_info=EXTRACTOR_INFO,
            )

    def test_13_member_count_mismatch(self) -> None:
        with self.assertRaises(MemoryBankError):
            save_bank(
                self.root / "count", np.zeros((2, 512), dtype=np.float32),
                member_rows(self.split.reference[1], "bottle", self.dataset, 1, "reference"),
                category="bottle", seed=42, shot=1,
                memory_type="global_reference", extractor_info=EXTRACTOR_INFO,
            )

    def test_14_nan_rejected(self) -> None:
        features = np.zeros((1, 512), dtype=np.float32)
        features[0, 0] = np.nan
        with self.assertRaises(MemoryBankError):
            save_bank(
                self.root / "nan", features,
                member_rows(self.split.reference[1], "bottle", self.dataset, 1, "reference"),
                category="bottle", seed=42, shot=1,
                memory_type="global_reference", extractor_info=EXTRACTOR_INFO,
            )

    def test_15_no_temp_leftovers(self) -> None:
        directory, _, _, _ = self._save_global("g_tmp", self.split.reference[1], 1)
        leftovers = [p.name for p in directory.iterdir() if ".tmp-" in p.name]
        self.assertEqual(leftovers, [])

    def test_16_seed_isolation_in_metadata(self) -> None:
        directory, _, _, meta = self._save_global("g_seed", self.split.reference[1], 1)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(load_bank(directory)["metadata"]["seed"], 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
