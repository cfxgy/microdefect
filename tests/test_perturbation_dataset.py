#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-1 扰动数据集构建测试（合成小规模夹具，不依赖真实数据与全量生成）。"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.perturbation_dataset import (  # noqa: E402
    MANIFEST_FIELDS,
    DatasetConfig,
    DatasetConflictError,
    atomic_write_bytes,
    check_split_isolation,
    generate_seed,
    md5_path,
    verify_seed,
)
from src.data.perturbations import build_specs  # noqa: E402

SPLIT_FIELDS = (
    "file_name",
    "rel_path",
    "abs_path",
    "class_prefix",
    "class_name",
    "width",
    "height",
    "mode",
    "size_bytes",
    "md5",
    "leak_group",
    "dup_group",
    "is_duplicate",
    "split",
    "seed",
    "shot",
)
CLASS_NAMES = {"Cr": "Crazing", "In": "Inclusion", "Pa": "Patches", "Sc": "Scratches"}


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def make_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(200, 200), dtype=np.uint8)


class Fixture:
    """构造最小可用的 images + splits 目录结构。"""

    def __init__(self, root: Path, *, duplicate_across_splits: bool = False) -> None:
        self.root = root
        self.images = root / "images"
        self.splits = root / "splits"
        self.out = root / "out"
        self.images.mkdir(parents=True)
        self.splits.mkdir(parents=True)

        spec = {
            "validation": [("Cr_1.bmp", "Cr"), ("In_2.bmp", "In")],
            "test": [("Pa_3.bmp", "Pa"), ("Sc_4.bmp", "Sc")],
        }
        payloads: dict[str, bytes] = {}
        for split, items in spec.items():
            for index, (name, cls) in enumerate(items):
                array = make_image(hash((split, name)) % 1000)
                path = self.images / name
                Image.fromarray(array, mode="L").save(path, format="BMP")
                if duplicate_across_splits and name == "Sc_4.bmp":
                    # 让 test 的 Sc_4 与 validation 的 Cr_1 内容相同
                    path.write_bytes((self.images / "Cr_1.bmp").read_bytes())
                payloads[name] = path.read_bytes()

        seed_dir = self.splits / "seed_42"
        seed_dir.mkdir(parents=True)
        for split, items in spec.items():
            rows = []
            for name, cls in items:
                path = self.images / name
                data = path.read_bytes()
                rows.append(
                    {
                        "file_name": name,
                        "rel_path": name,
                        "abs_path": str(path),
                        "class_prefix": cls,
                        "class_name": CLASS_NAMES[cls],
                        "width": 200,
                        "height": 200,
                        "mode": "L",
                        "size_bytes": len(data),
                        "md5": md5_bytes(data),
                        "leak_group": "g_" + md5_bytes(data)[:10],
                        "dup_group": "",
                        "is_duplicate": "False",
                        "split": split,
                        "seed": 42,
                        "shot": "na",
                    }
                )
            with (seed_dir / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(SPLIT_FIELDS))
                writer.writeheader()
                writer.writerows(rows)

    def config(self, **kwargs) -> DatasetConfig:
        params = dict(
            manifest_path=self.root / "manifest.csv",
            splits_root=self.splits,
            data_root=self.images,
            out_root=self.out,
            seeds=(42,),
            splits=("validation", "test"),
        )
        params.update(kwargs)
        return DatasetConfig(**params)


class GenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = PROJECT_ROOT / f"tmp_p41_{uuid.uuid4().hex[:8]}"
        self.root.mkdir()
        self.fx = Fixture(self.root)
        self.config = self.fx.config()
        self.specs = build_specs()
        self.report = generate_seed(self.config, 42, self.specs, log=lambda *a, **k: None)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def manifest(self) -> list[dict]:
        path = self.fx.out / "seed_42" / "perturbation_manifest.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_01_counts(self) -> None:
        self.assertEqual(self.report["counts"]["sources"], 4)
        self.assertEqual(self.report["counts"]["originals"], 4)
        self.assertEqual(self.report["counts"]["perturbations"], 40)  # 4 × 10
        self.assertEqual(self.report["counts"]["total_files"], 44)
        self.assertEqual(self.report["status"], "complete")
        for spec in self.specs:
            self.assertEqual(self.report["by_spec"][spec.name], 4)

    def test_02_manifest_fields(self) -> None:
        rows = self.manifest()
        self.assertEqual(len(rows), 44)
        self.assertEqual(list(rows[0].keys()), list(MANIFEST_FIELDS))
        for row in rows:
            self.assertEqual(row["seed"], "42")
            self.assertIn(row["split"], ("validation", "test"))
            self.assertTrue(row["source_md5"])
            self.assertTrue(row["output_md5"])
            self.assertEqual(row["width"], "200")
            self.assertEqual(row["height"], "200")
            self.assertEqual(row["mode"], "L")
            self.assertTrue(int(row["output_size_bytes"]) > 0)

    def test_03_md5_traceability(self) -> None:
        seed_dir = self.fx.out / "seed_42"
        for row in self.manifest():
            source = self.fx.images / row["source_file"]
            self.assertEqual(md5_path(source), row["source_md5"])
            target = seed_dir / row["output_rel_path"]
            self.assertTrue(target.is_file())
            self.assertEqual(md5_path(target), row["output_md5"])

    def test_04_original_bytes_identical(self) -> None:
        seed_dir = self.fx.out / "seed_42"
        originals = [r for r in self.manifest() if r["perturbation_type"] == "original"]
        self.assertEqual(len(originals), 4)
        for row in originals:
            source = self.fx.images / row["source_file"]
            target = seed_dir / row["output_rel_path"]
            self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_05_class_consistency(self) -> None:
        for row in self.manifest():
            self.assertEqual(Path(row["source_file"]).stem.split("_")[0], row["source_class"])

    def test_06_verify_ok(self) -> None:
        result = verify_seed(self.fx.out / "seed_42", log=lambda *a, **k: None)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["files_checked"], 44)
        self.assertEqual(result["missing"], 0)
        self.assertEqual(result["decode_errors"], 0)

    def test_07_verify_detects_missing(self) -> None:
        target = self.fx.out / "seed_42" / "test" / "brightness_low" / "Pa_3__brightness_low.bmp"
        self.assertTrue(target.is_file())
        target.unlink()
        result = verify_seed(self.fx.out / "seed_42", log=lambda *a, **k: None)
        self.assertFalse(result["ok"])
        self.assertEqual(result["missing"], 1)

    def test_08_verify_detects_corruption(self) -> None:
        target = self.fx.out / "seed_42" / "test" / "brightness_low" / "Pa_3__brightness_low.bmp"
        target.write_bytes(b"not-an-image")
        result = verify_seed(self.fx.out / "seed_42", log=lambda *a, **k: None)
        self.assertFalse(result["ok"])
        self.assertGreater(result["md5_mismatch"] + result["decode_errors"], 0)

    def test_09_rerun_without_resume_conflicts(self) -> None:
        with self.assertRaises(DatasetConflictError):
            generate_seed(self.config, 42, self.specs, log=lambda *a, **k: None)

    def test_10_resume_skips(self) -> None:
        report_path = self.fx.out / "seed_42" / "perturbation_report.json"
        before = report_path.stat().st_mtime_ns
        resume_config = self.fx.config(resume=True)
        report = generate_seed(resume_config, 42, self.specs, log=lambda *a, **k: None)
        self.assertEqual(report["seed"], 42)
        self.assertEqual(report_path.stat().st_mtime_ns, before)

    def test_11_original_source_untouched(self) -> None:
        digests = {p.name: md5_path(p) for p in self.fx.images.glob("*.bmp")}
        shutil.rmtree(self.fx.out)
        generate_seed(self.fx.config(), 42, self.specs, log=lambda *a, **k: None)
        for name, digest in digests.items():
            self.assertEqual(md5_path(self.fx.images / name), digest)


class AtomicAndIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = PROJECT_ROOT / f"tmp_p41_{uuid.uuid4().hex[:8]}"
        self.root.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_12_atomic_write_failure_leaves_nothing(self) -> None:
        target = self.root / "sub" / "out.bmp"
        with mock.patch("src.data.perturbation_dataset.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                atomic_write_bytes(target, b"data")
        self.assertFalse(target.exists())
        leftovers = [p for p in (self.root / "sub").rglob("*") if p.is_file()]
        self.assertEqual(leftovers, [])

    def test_13_cross_split_md5_rejected(self) -> None:
        fx = Fixture(self.root / "dup", duplicate_across_splits=True)
        with self.assertRaises(DatasetConflictError):
            generate_seed(fx.config(), 42, build_specs(), log=lambda *a, **k: None)

    def test_14_isolation_check_direct(self) -> None:
        fx = Fixture(self.root / "ok")
        val = [{"md5": "a" * 32, "file_name": "Cr_1.bmp"}]
        test = [{"md5": "b" * 32, "file_name": "Sc_4.bmp"}]
        info = check_split_isolation(val, test)
        self.assertEqual(info["cross_split_md5_overlap"], 0)
        self.assertEqual(info["validation_sources"], 1)

    def test_15_seed_directories_isolated(self) -> None:
        fx = Fixture(self.root / "seeds")
        specs = build_specs()
        generate_seed(fx.config(), 42, specs, log=lambda *a, **k: None)
        # 为 123 复用同一份划分内容，但目录按 seed 隔离
        (fx.splits / "seed_123").mkdir(parents=True, exist_ok=True)
        for name in ("validation.csv", "test.csv"):
            shutil.copyfile(fx.splits / "seed_42" / name, fx.splits / "seed_123" / name)
        generate_seed(fx.config(seeds=(42, 123)), 123, specs, log=lambda *a, **k: None)

        for seed in (42, 123):
            rows_path = fx.out / f"seed_{seed}" / "perturbation_manifest.csv"
            with rows_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual({r["seed"] for r in rows}, {str(seed)})
            self.assertTrue(all(r["output_rel_path"].startswith(("validation/", "test/")) for r in rows))

        dirs = sorted(p.name for p in fx.out.iterdir() if p.is_dir())
        self.assertEqual(dirs, ["seed_123", "seed_42"])
        files_42 = {p.relative_to(fx.out / "seed_42") for p in (fx.out / "seed_42").rglob("*.bmp")}
        files_123 = {p.relative_to(fx.out / "seed_123") for p in (fx.out / "seed_123").rglob("*.bmp")}
        self.assertEqual(files_42, files_123)  # 结构一致但物理隔离

    def test_16_manifest_reading_counts(self) -> None:
        fx = Fixture(self.root / "counts")
        generate_seed(fx.config(), 42, build_specs(), log=lambda *a, **k: None)
        report = json.loads((fx.out / "seed_42" / "perturbation_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["by_split"]["validation"]["sources"], 2)
        self.assertEqual(report["by_split"]["test"]["sources"], 2)
        self.assertEqual(report["isolation"]["cross_split_md5_overlap"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
