#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量检测模块测试（第三阶段）。"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.batch import (  # noqa: E402
    BatchRun,
    create_batch_dir,
    list_batches,
    load_batch,
    scan_folder,
)
from src.core.detector import Detector  # noqa: E402

DATA = PROJECT_ROOT / "datasets/mvtec_ad"
WORK = PROJECT_ROOT / "cache" / "tests" / f"batch_{uuid.uuid4().hex[:8]}"


def _copy(rel: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA / rel, target)


class BatchModuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.input_dir = WORK / "input"
        _copy("bottle/test/good/000.png", cls.input_dir / "good_a.png")
        _copy("bottle/test/good/001.png", cls.input_dir / "good_b.png")
        _copy("bottle/test/broken_large/000.png", cls.input_dir / "defect_a.png")
        _copy("bottle/test/broken_large/001.png", cls.input_dir / "nested" / "defect_b.png")
        (cls.input_dir / "notes.txt").write_text("not an image", encoding="utf-8")
        (cls.input_dir / "broken.png").write_bytes(b"not-an-image")
        cls.detector = Detector("bottle", protocol="group_disjoint", device="cpu")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(WORK, ignore_errors=True)

    def test_01_scan_recursive(self) -> None:
        scan = scan_folder(self.input_dir, recursive=True)
        self.assertEqual(len(scan.valid), 4)
        self.assertEqual(len(scan.invalid), 2)
        self.assertEqual(scan.total_files, 6)
        names = {p.name for p in scan.valid}
        self.assertIn("defect_b.png", names)
        reasons = {reason for _, reason in scan.invalid}
        self.assertIn("unsupported_extension", reasons)
        self.assertIn("invalid_or_corrupt_image", reasons)

    def test_02_scan_non_recursive(self) -> None:
        scan = scan_folder(self.input_dir, recursive=False)
        self.assertEqual(len(scan.valid), 3)
        self.assertNotIn("defect_b.png", {p.name for p in scan.valid})

    def test_03_scan_missing_folder(self) -> None:
        with self.assertRaises(NotADirectoryError):
            scan_folder(WORK / "does_not_exist")

    def test_04_batch_run(self) -> None:
        scan = scan_folder(self.input_dir, recursive=True)
        batch_dir = create_batch_dir(WORK / "outputs")
        run = BatchRun(scan.valid, "bottle", "group_disjoint", batch_dir, self.detector)
        while not run.finished:
            run.step()
        summary = run.finish()
        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["success"], 4)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["normal"], 2)
        self.assertEqual(summary["anomaly"], 2)
        self.assertEqual(summary["stopped"], 0)
        self.assertTrue((batch_dir / "batch_summary.json").is_file())
        self.assertTrue((batch_dir / "results.csv").is_file())
        with (batch_dir / "results.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row["status"], "success")
            for key in ("result_json", "heatmap_png", "overlay_png"):
                self.assertTrue((batch_dir / row[key]).is_file(), row[key])
        loaded = load_batch(batch_dir)
        self.assertEqual(loaded["success"], 4)

    def test_05_failure_isolation(self) -> None:
        good = self.input_dir / "good_a.png"
        bad = self.input_dir / "broken.png"
        batch_dir = create_batch_dir(WORK / "outputs")
        run = BatchRun([good, bad, self.input_dir / "defect_a.png"], "bottle",
                       "group_disjoint", batch_dir, self.detector)
        while not run.finished:
            run.step()
        summary = run.finish()
        self.assertEqual(summary["success"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertTrue((batch_dir / "errors.csv").is_file())
        with (batch_dir / "errors.csv").open(encoding="utf-8-sig") as handle:
            errors = list(csv.DictReader(handle))
        self.assertEqual(len(errors), 1)
        self.assertEqual(summary["images"][1]["status"], "failed")

    def test_06_stop_marks_remaining(self) -> None:
        scan = scan_folder(self.input_dir, recursive=True)
        batch_dir = create_batch_dir(WORK / "outputs")
        run = BatchRun(scan.valid, "bottle", "group_disjoint", batch_dir, self.detector)
        run.step()
        run.request_stop()
        while not run.finished:
            run.step()
        summary = run.finish()
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["stopped"], 3)
        self.assertEqual(summary["total"], 4)
        self.assertFalse(any(row["status"] == "success" and row["index"] > 0
                             for row in summary["images"]))

    def test_07_same_name_no_overwrite(self) -> None:
        first = WORK / "a" / "000.png"
        second = WORK / "b" / "000.png"
        _copy("bottle/test/good/000.png", first)
        _copy("bottle/test/good/001.png", second)
        batch_dir = create_batch_dir(WORK / "outputs")
        run = BatchRun([first, second], "bottle", "group_disjoint", batch_dir, self.detector)
        while not run.finished:
            run.step()
        run.finish()
        names = [row["result_json"] for row in run.rows]
        self.assertEqual(len(set(names)), 2)
        for name in names:
            self.assertTrue((batch_dir / name).is_file())

    def test_08_history(self) -> None:
        batches = list_batches(WORK / "outputs")
        self.assertGreaterEqual(len(batches), 1)
        self.assertTrue(all("batch_id" in item for item in batches))
        self.assertIn("total", batches[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
