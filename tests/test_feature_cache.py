#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P2 特征缓存测试（不依赖预训练权重）。

    D:\\microdefect\\.venv\\Scripts\\python.exe -m unittest tests.test_feature_cache -v
"""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import cache as cache_module  # noqa: E402
from src.features.cache import (  # noqa: E402
    CACHE_SCHEMA,
    CacheCorruptedError,
    CacheKeyParts,
    FeatureCache,
    compute_key,
    md5_file,
)
from src.features.extractor import (  # noqa: E402
    EMBED_DIM,
    FEATURE_VERSION,
    PREPROCESS_VERSION,
    WEIGHTS_TAG,
)

DATA_ROOT = PROJECT_ROOT / "data" / "NEU-CLS"


def sample_vector(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=EMBED_DIM).astype(np.float32)


class FeatureCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # 系统临时目录与 tempfile 的 0700 目录在受限沙箱下不可写/不可访问，
        # 因此显式在项目根目录下用默认权限创建临时目录。
        self.tmp_path = PROJECT_ROOT / f"tmp_feature_cache_{uuid.uuid4().hex[:8]}"
        self.tmp_path.mkdir()
        self.cache = FeatureCache(self.tmp_path / "features")
        self.md5 = "0123456789abcdef0123456789abcdef"
        self.key = self.cache.key_for_md5(self.md5)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    # ---- 键构成 ----
    def test_01_key_from_md5_only(self) -> None:
        """同一 MD5（即使文件名不同）得到同一键。"""
        self.assertEqual(compute_key(self.md5), compute_key(self.md5.upper().lower()))
        parts = CacheKeyParts(image_md5=self.md5)
        self.assertEqual(parts.key(), compute_key(self.md5))
        self.assertEqual(len(self.key), 32)

    def test_02_key_changes_with_md5(self) -> None:
        other = compute_key("ffffffffffffffffffffffffffffffff")
        self.assertNotEqual(self.key, other)

    def test_03_key_changes_with_preprocess_version(self) -> None:
        changed = compute_key(self.md5, preprocess_version=PREPROCESS_VERSION + "-v2")
        self.assertNotEqual(self.key, changed)

    def test_04_key_changes_with_weights_tag(self) -> None:
        changed = compute_key(self.md5, weights_tag=WEIGHTS_TAG + "-other")
        self.assertNotEqual(self.key, changed)

    def test_05_key_changes_with_feature_version(self) -> None:
        changed = compute_key(self.md5, feature_version=FEATURE_VERSION + "-v2")
        self.assertNotEqual(self.key, changed)

    # ---- 读写 ----
    def test_06_put_then_get_roundtrip(self) -> None:
        vector = sample_vector(1)
        path = self.cache.put(self.key, vector)
        self.assertTrue(path.is_file())
        loaded = self.cache.get(self.key)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(vector, loaded)
        self.assertEqual(loaded.dtype, np.float32)
        self.assertEqual(loaded.shape, (EMBED_DIM,))

    def test_07_miss_and_hit_counters(self) -> None:
        self.assertIsNone(self.cache.get(self.key))
        self.assertEqual(self.cache.stats["misses"], 1)
        self.assertEqual(self.cache.stats["hits"], 0)

        self.cache.put(self.key, sample_vector(2))
        self.assertIsNotNone(self.cache.get(self.key))
        self.assertEqual(self.cache.stats["hits"], 1)
        self.assertEqual(self.cache.stats["misses"], 1)

    def test_08_different_content_never_shares_slot(self) -> None:
        key_a = self.cache.key_for_md5("aaaa" * 8)
        key_b = self.cache.key_for_md5("bbbb" * 8)
        self.assertNotEqual(self.cache.path_for(key_a), self.cache.path_for(key_b))
        self.cache.put(key_a, sample_vector(3))
        self.assertIsNone(self.cache.get(key_b))

    # ---- 损坏检测 ----
    def test_09_truncated_file_detected(self) -> None:
        path = self.cache.put(self.key, sample_vector(4))
        path.write_bytes(path.read_bytes()[:40])  # 截断
        self.cache.reset_stats()
        self.assertIsNone(self.cache.get(self.key))
        self.assertEqual(self.cache.stats["corrupted"], 1)
        self.assertEqual(self.cache.stats["hits"], 0)
        with self.assertRaises(CacheCorruptedError):
            self.cache.get(self.key, strict=True)

    def test_10_wrong_shape_detected(self) -> None:
        path = self.cache.path_for(self.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, feature=np.zeros(10, np.float32), key=self.key, schema=CACHE_SCHEMA)
        self.cache.reset_stats()
        self.assertIsNone(self.cache.get(self.key))
        self.assertEqual(self.cache.stats["corrupted"], 1)

    def test_11_nan_feature_detected(self) -> None:
        path = self.cache.path_for(self.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        bad = np.zeros(EMBED_DIM, np.float32)
        bad[0] = np.nan
        np.savez(path, feature=bad, key=self.key, schema=CACHE_SCHEMA)
        self.cache.reset_stats()
        self.assertIsNone(self.cache.get(self.key))
        self.assertEqual(self.cache.stats["corrupted"], 1)

    def test_12_put_rejects_invalid_vector(self) -> None:
        with self.assertRaises(CacheCorruptedError):
            self.cache.put(self.key, np.zeros(10, np.float32))
        bad = np.zeros(EMBED_DIM, np.float32)
        bad[3] = np.inf
        with self.assertRaises(CacheCorruptedError):
            self.cache.put(self.key, bad)
        self.assertFalse(self.cache.contains(self.key))

    # ---- 原子性 ----
    def test_13_no_temp_files_left_after_success(self) -> None:
        self.cache.put(self.key, sample_vector(5))
        leftovers = [p for p in self.cache.root.rglob("*") if ".tmp-" in p.name]
        self.assertEqual(leftovers, [])

    def test_14_failed_write_leaves_nothing_readable(self) -> None:
        """写入中途异常：不得留下可被误读的完整缓存文件，也不得留下临时文件。"""
        vector = sample_vector(6)

        def failing_savez(handle, *args, **kwargs):
            handle.write(b"partial-data")  # 模拟中途写入
            raise OSError("simulated disk failure")

        with mock.patch("numpy.savez", side_effect=failing_savez):
            with self.assertRaises(OSError):
                self.cache.put(self.key, vector)

        self.assertFalse(self.cache.path_for(self.key).exists())
        leftovers = [p for p in self.cache.root.rglob("*") if p.is_file()]
        self.assertEqual(leftovers, [], f"残留文件：{leftovers}")
        self.assertIsNone(self.cache.get(self.key))

    # ---- 失效 ----
    def test_15_invalidate(self) -> None:
        self.cache.put(self.key, sample_vector(7))
        self.assertTrue(self.cache.invalidate(self.key))
        self.assertFalse(self.cache.contains(self.key))
        self.assertIsNone(self.cache.get(self.key))
        self.assertFalse(self.cache.invalidate(self.key))

    def test_16_version_bump_does_not_reuse_old_slot(self) -> None:
        old = FeatureCache(self.tmp_path / "features")
        old.put(old.key_for_md5(self.md5), sample_vector(8))

        new_pp = FeatureCache(self.tmp_path / "features", preprocess_version="p2-other")
        new_w = FeatureCache(self.tmp_path / "features", weights_tag="OTHER")
        self.assertNotEqual(old.key_for_md5(self.md5), new_pp.key_for_md5(self.md5))
        self.assertNotEqual(old.key_for_md5(self.md5), new_w.key_for_md5(self.md5))
        self.assertIsNone(new_pp.get(new_pp.key_for_md5(self.md5)))
        self.assertIsNone(new_w.get(new_w.key_for_md5(self.md5)))

    # ---- 真实文件：内容相同则键相同 ----
    def test_17_real_duplicate_images_share_key(self) -> None:
        """Pa_101.bmp 与 Pa_105.bmp 内容相同 ⇒ MD5 相同 ⇒ 缓存键相同。"""
        first = DATA_ROOT / "Pa_101.bmp"
        second = DATA_ROOT / "Pa_105.bmp"
        self.assertTrue(first.is_file() and second.is_file())
        md5_a, md5_b = md5_file(first), md5_file(second)
        self.assertEqual(md5_a, md5_b)
        self.assertEqual(self.cache.key_for_md5(md5_a), self.cache.key_for_md5(md5_b))

    def test_18_cache_dir_created_and_summary(self) -> None:
        nested = FeatureCache(self.tmp_path / "a" / "b" / "c")
        self.assertTrue(nested.root.is_dir())
        nested.put(nested.key_for_md5(self.md5), sample_vector(9))
        summary = nested.summary()
        self.assertEqual(summary["cached_files"], 1)
        self.assertEqual(summary["schema"], CACHE_SCHEMA)
        self.assertEqual(summary["versions"]["weights_tag"], WEIGHTS_TAG)


if __name__ == "__main__":
    unittest.main(verbosity=2)
