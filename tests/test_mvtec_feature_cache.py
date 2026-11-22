#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-2.2 双路径特征缓存测试。"""

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

from src.features.mvtec_feature_cache import (  # noqa: E402
    SPATIAL_SPEC,
    CacheCorruptedError,
    FeatureCacheError,
    FeatureSpec,
    GLOBAL_SPEC,
    compute_key,
    image_md5,
    make_global_cache,
    make_spatial_cache,
)

DATA = PROJECT_ROOT / "datasets" / "mvtec_ad"
RGB_A = DATA / "bottle/train/good/000.png"
RGB_B = DATA / "bottle/train/good/001.png"
WEIGHTS_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


class FeatureCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = PROJECT_ROOT / "cache" / "tests" / f"mvtec_cache_{uuid.uuid4().hex[:8]}"
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.global_cache = make_global_cache(cls.root / "global", weights_sha256=WEIGHTS_SHA)
        cls.spatial_cache = make_spatial_cache(cls.root / "spatial", weights_sha256=WEIGHTS_SHA)
        cls.rng = np.random.default_rng(0)
        cls.g = cls.rng.normal(size=(512,)).astype(np.float32)
        cls.s = cls.rng.normal(size=(256, 14, 14)).astype(np.float32)
        cls.md5 = image_md5(RGB_A)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_01_miss_then_hit(self) -> None:
        key = self.global_cache.key_for_md5(self.md5)
        self.assertFalse(self.global_cache.contains(key))
        self.assertIsNone(self.global_cache.get(key))
        self.global_cache.put(key, self.g, image_md5_hex=self.md5)
        loaded = self.global_cache.get(key)
        self.assertIsNotNone(loaded)
        self.assertTrue(np.array_equal(loaded, self.g))
        self.assertGreaterEqual(self.global_cache.stats["hits"], 1)

    def test_02_roundtrip_spatial(self) -> None:
        key = self.spatial_cache.key_for_md5(self.md5)
        self.spatial_cache.put(key, self.s, image_md5_hex=self.md5)
        loaded = self.spatial_cache.get(key)
        self.assertTrue(np.array_equal(loaded, self.s))
        self.assertEqual(loaded.dtype, np.float32)
        self.assertEqual(loaded.shape, (256, 14, 14))

    def test_03_key_distinction(self) -> None:
        k_g = compute_key(self.md5, weights_sha256=WEIGHTS_SHA, spec=GLOBAL_SPEC)
        k_s = compute_key(self.md5, weights_sha256=WEIGHTS_SHA, spec=SPATIAL_SPEC)
        self.assertNotEqual(k_g, k_s)
        self.assertNotEqual(self.global_cache.path_for(k_g), self.spatial_cache.path_for(k_s))

    def test_04_version_changes_key(self) -> None:
        alt = FeatureSpec(kind="spatial", layer="layer3", feature_version="v-alt", shape=(256, 14, 14))
        k1 = compute_key(self.md5, weights_sha256=WEIGHTS_SHA, spec=SPATIAL_SPEC)
        k2 = compute_key(self.md5, weights_sha256=WEIGHTS_SHA, spec=alt)
        self.assertNotEqual(k1, k2)

    def test_05_weights_change_key(self) -> None:
        k1 = compute_key(self.md5, weights_sha256=WEIGHTS_SHA, spec=GLOBAL_SPEC)
        k2 = compute_key(self.md5, weights_sha256="0" * 64, spec=GLOBAL_SPEC)
        self.assertNotEqual(k1, k2)

    def test_06_content_change_key(self) -> None:
        k1 = self.global_cache.key_for_image(RGB_A)
        k2 = self.global_cache.key_for_image(RGB_B)
        self.assertNotEqual(k1, k2)

    def test_07_corrupted_detected(self) -> None:
        key = self.global_cache.key_for_md5(self.md5)
        path = self.global_cache.path_for(key + "corrupt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a npz file")
        self.assertIsNone(self.global_cache.get(key + "corrupt"))
        with self.assertRaises(CacheCorruptedError):
            self.global_cache.get(key + "corrupt", strict=True)

    def test_08_truncated_detected(self) -> None:
        key = self.global_cache.key_for_md5(self.md5)
        good = self.global_cache.path_for(key)
        self.assertTrue(good.is_file())
        broken_key = key + "trunc"
        broken = self.global_cache.path_for(broken_key)
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(good.read_bytes()[:50])
        self.assertIsNone(self.global_cache.get(broken_key))

    def test_09_meta_mismatch_detected(self) -> None:
        key = self.global_cache.key_for_md5(self.md5)
        bad_key = key + "meta"
        path = self.global_cache.path_for(bad_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"schema": "wrong", "key": bad_key}
        with path.open("wb") as handle:
            np.savez(handle, feature=self.g, meta=np.array(__import__("json").dumps(meta)))
        self.assertIsNone(self.global_cache.get(bad_key))
        with self.assertRaises(CacheCorruptedError):
            self.global_cache.get(bad_key, strict=True)

    def test_10_nan_rejected(self) -> None:
        arr = self.g.copy()
        arr[0] = np.nan
        key = self.global_cache.key_for_md5(self.md5)
        with self.assertRaises(FeatureCacheError):
            self.global_cache.put(key, arr, image_md5_hex=self.md5)

    def test_11_shape_rejected(self) -> None:
        key = self.global_cache.key_for_md5(self.md5)
        with self.assertRaises(FeatureCacheError):
            self.global_cache.put(key, np.zeros((256, 14, 14), dtype=np.float32), image_md5_hex=self.md5)

    def test_12_atomic_no_temp_files(self) -> None:
        key = self.global_cache.key_for_md5(self.md5)
        self.global_cache.put(key, self.g, image_md5_hex=self.md5)
        directory = self.global_cache.path_for(key).parent
        leftovers = [p.name for p in directory.iterdir() if ".tmp-" in p.name]
        self.assertEqual(leftovers, [])

    def test_13_put_with_key_mismatch_rejected(self) -> None:
        with self.assertRaises(FeatureCacheError):
            self.global_cache.put("0" * 32, self.g, image_md5_hex=self.md5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
