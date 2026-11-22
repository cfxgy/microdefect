#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 实验结果完整性与数据隔离测试。

读取正式结果目录（默认 experiments/results/fewshot，可用环境变量
MICRODEFECT_FEWSHOT_DIR 覆盖）与 P1-2 的划分文件，检查：
15 组结果齐全、预测明细完整、原型/阈值有效、值自洽、输入未被改动。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEEDS = (42, 123, 2024, 7, 999)
SHOTS = (1, 3, 5)
CLASSES = ("Cr", "In", "Pa", "PS", "RS", "Sc")
DIM = 512

FEWSHOT_DIR = Path(
    os.environ.get("MICRODEFECT_FEWSHOT_DIR", str(PROJECT_ROOT / "experiments" / "results" / "fewshot"))
)
SPLITS_DIR = PROJECT_ROOT / "experiments" / "results" / "splits"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FewshotResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        summary_path = FEWSHOT_DIR / "summary.json"
        if not summary_path.is_file():
            raise unittest.SkipTest(f"结果目录不存在或缺少 summary.json：{FEWSHOT_DIR}")
        cls.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cls.rows = read_csv(FEWSHOT_DIR / "summary.csv")

    # ---- 15 组齐全 ----
    def test_01_all_15_groups_present(self) -> None:
        pairs = {(int(r["seed"]), int(r["shot"])) for r in self.rows}
        expected = {(seed, shot) for seed in SEEDS for shot in SHOTS}
        self.assertEqual(pairs, expected)
        self.assertEqual(len(self.rows), 15)
        for seed in SEEDS:
            for shot in SHOTS:
                group = FEWSHOT_DIR / f"seed_{seed}" / f"shot_{shot}"
                self.assertTrue(group.is_dir(), f"缺少目录：{group}")

    def test_02_group_files_complete(self) -> None:
        required = (
            "predictions_test.csv",
            "predictions_validation.csv",
            "prototypes.csv",
            "thresholds.json",
            "metrics.json",
            "confusion_matrix.csv",
        )
        for seed in SEEDS:
            for shot in SHOTS:
                group = FEWSHOT_DIR / f"seed_{seed}" / f"shot_{shot}"
                for name in required:
                    path = group / name
                    self.assertTrue(path.is_file(), f"缺少文件：{path}")
                    self.assertGreater(path.stat().st_size, 0, f"空文件：{path}")

    def test_03_predictions_columns_and_rows(self) -> None:
        fields = {
            "file_name",
            "md5",
            "true_label",
            "predicted_label",
            "minimum_distance",
            "accepted",
            "seed",
            "shot",
        }
        for seed in SEEDS:
            for shot in SHOTS:
                group = FEWSHOT_DIR / f"seed_{seed}" / f"shot_{shot}"
                rows = read_csv(group / "predictions_test.csv")
                self.assertTrue(rows), f"{group} 预测为空"
                self.assertTrue(fields <= set(rows[0]), f"{group} 缺列")
                test_rows = read_csv(SPLITS_DIR / f"seed_{seed}" / "test.csv")
                self.assertEqual(len(rows), len(test_rows), f"{group} 行数不符")
                self.assertEqual({r["md5"] for r in rows}, {r["md5"] for r in test_rows})
                for row in rows:
                    self.assertIn(row["true_label"], CLASSES)
                    self.assertIn(row["predicted_label"], CLASSES)
                    self.assertGreaterEqual(float(row["minimum_distance"]), 0.0)
                    self.assertLessEqual(float(row["minimum_distance"]), 2.0)

    # ---- 三档 shot 共享 validation/test ----
    def test_04_shots_share_validation_and_test(self) -> None:
        for seed in SEEDS:
            seed_dir = SPLITS_DIR / f"seed_{seed}"
            test_ref = [r["file_name"] for r in read_csv(seed_dir / "test.csv")]
            val_ref = [r["file_name"] for r in read_csv(seed_dir / "validation.csv")]
            for shot in SHOTS:
                group = FEWSHOT_DIR / f"seed_{seed}" / f"shot_{shot}"
                test_now = [r["file_name"] for r in read_csv(group / "predictions_test.csv")]
                val_now = [r["file_name"] for r in read_csv(group / "predictions_validation.csv")]
                self.assertEqual(test_now, test_ref, f"seed={seed} shot={shot} test 不一致")
                self.assertEqual(val_now, val_ref, f"seed={seed} shot={shot} validation 不一致")

    def test_05_support_nesting_used(self) -> None:
        """1-shot ⊂ 3-shot ⊂ 5-shot 的 support 关系来自 P1-2，实验必须沿用。"""
        for seed in SEEDS:
            sets = {}
            for shot in SHOTS:
                rows = read_csv(SPLITS_DIR / f"seed_{seed}" / f"support_{shot}shot.csv")
                sets[shot] = {r["file_name"] for r in rows}
            self.assertTrue(sets[1] <= sets[3] <= sets[5], f"seed={seed} support 不嵌套")
            self.assertEqual(len(sets[5]), 30)

    # ---- 原型 ----
    def test_06_prototypes_valid(self) -> None:
        for seed in SEEDS:
            for shot in SHOTS:
                rows = read_csv(FEWSHOT_DIR / f"seed_{seed}" / f"shot_{shot}" / "prototypes.csv")
                self.assertEqual(len(rows), 6)
                self.assertEqual({r["class_prefix"] for r in rows}, set(CLASSES))
                for row in rows:
                    dims = [float(row[f"d{i}"]) for i in range(DIM)]
                    self.assertEqual(len(dims), DIM)
                    norm = sum(v * v for v in dims) ** 0.5
                    self.assertAlmostEqual(norm, 1.0, places=4)
                    self.assertTrue(all(abs(v) < 1e6 for v in dims))
                    self.assertEqual(int(row["support_count"]), shot)

    # ---- 阈值 ----
    def test_07_thresholds_valid(self) -> None:
        for seed in SEEDS:
            for shot in SHOTS:
                payload = json.loads(
                    (FEWSHOT_DIR / f"seed_{seed}" / f"shot_{shot}" / "thresholds.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(payload["seed"], seed)
                self.assertEqual(payload["shot"], shot)
                self.assertAlmostEqual(payload["quantile"], 0.95)
                self.assertEqual(payload["source"], "validation")
                self.assertEqual(set(payload["thresholds"]), set(CLASSES))
                for cls, tau in payload["thresholds"].items():
                    self.assertGreaterEqual(tau, 0.0, f"{cls} 阈值越界")
                    self.assertLessEqual(tau, 2.0, f"{cls} 阈值越界")
                    self.assertEqual(payload["validation_stats"][cls]["n"], 50)

    # ---- 指标自洽 ----
    def test_08_metrics_consistent(self) -> None:
        summary_index = {(int(r["seed"]), int(r["shot"])): r for r in self.rows}
        for (seed, shot), summary_row in summary_index.items():
            metrics = json.loads(
                (FEWSHOT_DIR / f"seed_{seed}" / f"shot_{shot}" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            closed = metrics["classification_closed_set"]
            rejection = metrics["rejection"]
            confusion = closed["confusion_matrix"]
            total = sum(sum(row) for row in confusion)
            self.assertEqual(total, closed["total"])
            self.assertEqual(total, rejection["test_total"])
            self.assertEqual(sum(confusion[i][i] for i in range(6)), closed["correct"])
            self.assertAlmostEqual(closed["accuracy"], closed["correct"] / total, places=6)
            self.assertAlmostEqual(float(summary_row["accuracy"]), closed["accuracy"], places=6)
            self.assertAlmostEqual(
                float(summary_row["macro_f1"]), closed["macro_f1"], places=6
            )
            self.assertEqual(
                rejection["accepted_count"] + rejection["rejected_count"], rejection["test_total"]
            )
            if rejection["accepted_count"] == 0:
                self.assertIsNone(rejection["accepted_accuracy"])
            else:
                self.assertAlmostEqual(
                    rejection["accepted_accuracy"],
                    rejection["accepted_correct"] / rejection["accepted_count"],
                    places=6,
                )

    # ---- 汇总 ----
    def test_09_aggregate_matches_summary(self) -> None:
        import statistics

        for shot in SHOTS:
            values = [float(r["accuracy"]) for r in self.rows if int(r["shot"]) == shot]
            aggregate = self.summary["aggregate"][f"shot_{shot}"]["accuracy"]
            self.assertAlmostEqual(aggregate["mean"], statistics.mean(values), places=6)
            self.assertEqual(aggregate["count"], 5)
            if len(values) > 1:
                self.assertAlmostEqual(aggregate["std"], statistics.stdev(values), places=6)

    # ---- 输入未被改动 ----
    def test_10_inputs_unchanged_since_run(self) -> None:
        recorded = self.summary["inputs_sha256"]
        for name, digest in recorded.items():
            if name == "manifest":
                path = PROJECT_ROOT / "experiments" / "results" / "neu_cls_manifest.csv"
            else:
                path = SPLITS_DIR / name
            self.assertTrue(path.is_file(), f"缺少输入文件：{path}")
            self.assertEqual(sha256_file(path), digest, f"输入文件已被修改：{path}")

    def test_11_run_metadata_present(self) -> None:
        config = self.summary["config"]
        self.assertEqual(tuple(config["seeds"]), SEEDS)
        self.assertEqual(tuple(config["shots"]), SHOTS)
        self.assertAlmostEqual(config["quantile"], 0.95)
        self.assertEqual(config["embed_dim"], DIM)
        self.assertIn("features", self.summary)
        self.assertGreater(self.summary["features"]["unique_contents"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
