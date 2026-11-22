#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5-2.1 MVTec AD 协议 v1.0 划分测试。

覆盖：15 类 × 5 seed × {1,3,5}-shot = 75 组，全部集合关系、嵌套、
互斥、memory_full 等价性与成员哈希一致性。

运行：
    D:\\microdefect\\.venv\\Scripts\\python.exe -m unittest tests.test_mvtec_split -v
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mvtec_dataset import CATEGORIES, MVTecDataset  # noqa: E402
from src.data.mvtec_split import (  # noqa: E402
    CALIBRATION_SIZE,
    DEFAULT_SEEDS,
    SHOTS,
    build_group_disjoint_split,
    build_split,
    file_sha256,
    load_similarity_groups,
    member_hash,
    protocol_seed,
    totals,
    verify_split,
)

DATA_ROOT = PROJECT_ROOT / "datasets" / "mvtec_ad"
PAIRS_CSV = PROJECT_ROOT / "experiments/results/mvtec_ad/stage3_1_similarity_review/similarity_pairs.csv"


class MVTecSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = MVTecDataset(DATA_ROOT)
        cls.splits = {
            (category, seed): build_split(cls.dataset, category, seed)
            for category in CATEGORIES
            for seed in DEFAULT_SEEDS
        }

    def test_01_all_75_groups_verified(self) -> None:
        self.assertEqual(len(self.splits), 75)
        failures = []
        for (category, seed), split in self.splits.items():
            result = verify_split(self.dataset, split)
            if not result["ok"]:
                failures.append((category, seed, result["issues"][:3]))
        self.assertEqual(failures, [])

    def test_02_counts_and_totals(self) -> None:
        for (category, seed), split in self.splits.items():
            self.assertEqual(len(split.calibration), CALIBRATION_SIZE)
            for k in SHOTS:
                self.assertEqual(len(split.reference[k]), k)
                self.assertEqual(len(split.extended[k]), len(split.order) - k - CALIBRATION_SIZE)
                self.assertEqual(len(split.memory_full[k]), len(split.order) - CALIBRATION_SIZE)

        seed42 = [self.splits[(c, 42)] for c in CATEGORIES]
        t = totals(seed42)
        self.assertEqual(t["groups"], 15)
        self.assertEqual(t["train_good"], 3629)
        self.assertEqual(t["calibration"], 375)
        self.assertEqual(t["memory_full"], 3254)
        self.assertEqual(t["extended_k1"], 3239)
        self.assertEqual(t["extended_k3"], 3209)
        self.assertEqual(t["extended_k5"], 3179)

        all_t = totals(list(self.splits.values()))
        self.assertEqual(all_t["groups"], 75)
        self.assertEqual(all_t["train_good"], 3629 * 5)
        self.assertEqual(all_t["calibration"], 375 * 5)
        self.assertEqual(all_t["memory_full"], 3254 * 5)

    def test_03_set_relations(self) -> None:
        for (category, seed), split in self.splits.items():
            train = set(split.order)
            cal = set(split.calibration)
            refs = {k: set(split.reference[k]) for k in SHOTS}
            self.assertTrue(refs[1] <= refs[3] <= refs[5])
            self.assertTrue(cal.isdisjoint(refs[5]))
            for k in SHOTS:
                self.assertTrue(set(split.extended[k]).isdisjoint(refs[k]))
                self.assertTrue(set(split.extended[k]).isdisjoint(cal))
                self.assertEqual(set(split.memory_full[k]), train - cal)
            self.assertEqual(
                len({split.hashes[f"memory_full_{k}"] for k in SHOTS}), 1
            )

    def test_04_no_test_images(self) -> None:
        for split in self.splits.values():
            members = list(split.calibration)
            for k in SHOTS:
                members += split.reference[k] + split.extended[k] + split.memory_full[k]
            for path in members:
                self.assertIn("/train/good/", path.as_posix())

    def test_05_toothbrush(self) -> None:
        split = self.splits[("toothbrush", 42)]
        self.assertEqual(len(split.order), 60)
        self.assertEqual(len(split.calibration), 25)
        self.assertEqual(len(split.reference[5]), 5)
        self.assertEqual(len(split.extended[5]), 30)
        self.assertEqual(len(split.memory_full[5]), 35)

    def test_06_deterministic(self) -> None:
        a = build_split(self.dataset, "capsule", 2024)
        b = build_split(self.dataset, "capsule", 2024)
        self.assertEqual([p.as_posix() for p in a.order], [p.as_posix() for p in b.order])
        self.assertEqual(a.hashes, b.hashes)

    def test_07_protocol_seed(self) -> None:
        expected = int(
            hashlib.sha256(b"mvtec-ad|v1.0|42|bottle").hexdigest()[:16], 16
        )
        self.assertEqual(protocol_seed(42, "bottle"), expected)

    def test_08_member_hash_definition(self) -> None:
        split = self.splits[("bottle", 7)]
        manual = hashlib.sha256(
            "\n".join(
                sorted(f"{self.dataset.relative(p)}\t{file_sha256(p)}" for p in split.calibration)
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(member_hash(split.calibration, self.dataset), manual)
        self.assertEqual(split.hashes["calibration"], manual)

    def test_09_unique_paths(self) -> None:
        for split in self.splits.values():
            cal = [p.as_posix() for p in split.calibration]
            self.assertEqual(len(cal), len(set(cal)))
            for k in SHOTS:
                ref = [p.as_posix() for p in split.reference[k]]
                self.assertEqual(len(ref), len(set(ref)))

    def test_10_group_disjoint_interface(self) -> None:
        groups = load_similarity_groups(self.dataset, PAIRS_CSV)
        self.assertEqual(len(groups), 15)

        capsule = build_group_disjoint_split(self.dataset, "capsule", 42, groups)
        self.assertFalse(capsule["feasible"])
        self.assertEqual(capsule["excluded_train"], 216)
        self.assertEqual(capsule["remaining_train"], 3)
        self.assertEqual(capsule["calibration_shortfall"], 22)
        self.assertEqual(capsule["total_shortfall"], 27)
        self.assertEqual(capsule["reference"], [])

        bottle = build_group_disjoint_split(self.dataset, "bottle", 42, groups)
        self.assertTrue(bottle["feasible"])
        self.assertEqual(bottle["excluded_train"], 33)
        self.assertEqual(len(bottle["reference"]), 5)
        self.assertEqual(len(bottle["calibration"]), 25)
        self.assertEqual(len(bottle["test_defect"]), 63)

        feasible = [
            c for c in CATEGORIES
            if build_group_disjoint_split(self.dataset, c, 42, groups)["feasible"]
        ]
        self.assertEqual(len(feasible), 14)
        self.assertNotIn("capsule", feasible)


if __name__ == "__main__":
    unittest.main(verbosity=2)
