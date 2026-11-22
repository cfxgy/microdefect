#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-2 数据划分自动化测试。

同一份文件既可由 pytest 收集运行，也可用标准库 unittest 直接运行：

    D:\\microdefect\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v
    D:\\microdefect\\.venv\\Scripts\\python.exe -m pytest tests -v   （需先装 pytest）

测试目标：可复现、集合隔离、重复图不跨集合、shot 嵌套、完整覆盖、无旧路径。
"""

from __future__ import annotations

import csv
import filecmp
import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import (  # noqa: E402
    CLASS_ORDER,
    DEFAULT_N_SUPPORT,
    DEFAULT_N_VAL,
    DEFAULT_SEEDS,
    SplitConfig,
    build_seed_split,
    load_manifest,
    seed_outputs,
    validate_seed_split,
    write_csv,
)

MANIFEST = PROJECT_ROOT / "experiments" / "results" / "neu_cls_manifest.csv"
DATA_ROOT = PROJECT_ROOT / "data" / "NEU-CLS"
LEGACY_ROOT = PROJECT_ROOT / "NEU-CLS"
SPLITS_ROOT = PROJECT_ROOT / "experiments" / "results" / "splits"

CSV_NAMES = (
    "support_5shot.csv",
    "support_3shot.csv",
    "support_1shot.csv",
    "validation.csv",
    "test.csv",
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


class SplitTestCase(unittest.TestCase):
    """所有测试共享一次性加载的清单与五个种子的划分。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_manifest(MANIFEST)
        cls.config = SplitConfig(
            manifest_path=MANIFEST,
            data_root=DATA_ROOT,
            n_support=DEFAULT_N_SUPPORT,
            n_val=DEFAULT_N_VAL,
            seeds=DEFAULT_SEEDS,
        )
        cls.splits = {
            seed: build_seed_split(cls.rows, seed, cls.config) for seed in DEFAULT_SEEDS
        }
        cls.validations = {
            seed: validate_seed_split(
                cls.splits[seed],
                cls.rows,
                data_root=DATA_ROOT,
                legacy_root=LEGACY_ROOT,
                n_support=DEFAULT_N_SUPPORT,
                n_val=DEFAULT_N_VAL,
            )
            for seed in DEFAULT_SEEDS
        }

    # ---- 1. 可复现 ----
    def test_01_same_seed_reproducible(self) -> None:
        """同一 seed 连续生成两次，CSV 逐字节一致。"""
        # 系统临时目录在受限沙箱下不可写，改用项目内默认权限目录
        tmp_path = PROJECT_ROOT / f"tmp_split_test_{uuid.uuid4().hex[:8]}"
        tmp_path.mkdir()
        try:
            for run in ("run_a", "run_b"):
                split = build_seed_split(self.rows, 42, self.config)
                out = tmp_path / run
                for name, rows in seed_outputs(split).items():
                    write_csv(out / name, rows)

            for name in CSV_NAMES:
                a = tmp_path / "run_a" / name
                b = tmp_path / "run_b" / name
                self.assertTrue(a.is_file() and b.is_file(), f"{name} 未生成")
                self.assertTrue(
                    filecmp.cmp(a, b, shallow=False),
                    f"同一 seed 两次生成的 {name} 不一致",
                )
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    # ---- 2. 不同 seed 合法且确实不同 ----
    def test_02_different_seeds_valid_and_distinct(self) -> None:
        """不同 seed 均通过校验，且 support 集合不完全相同。"""
        for seed, validation in self.validations.items():
            self.assertTrue(validation["all_pass"], f"seed={seed} 校验未通过")

        def support_names(seed: int) -> set[str]:
            return {r["file_name"] for r in self.splits[seed].flatten("support")}

        pairs = [(42, 123), (42, 2024), (7, 999)]
        for a, b in pairs:
            self.assertNotEqual(
                support_names(a), support_names(b), f"seed={a} 与 seed={b} 的 support 完全相同"
            )

    # ---- 3. 六类齐全 ----
    def test_03_all_six_classes_present(self) -> None:
        """六个类别在每个 seed 的每个集合中都有样本。"""
        for seed, split in self.splits.items():
            self.assertEqual(set(split.per_class), set(CLASS_ORDER), f"seed={seed} 类别不全")
            for cls in CLASS_ORDER:
                cs = split.per_class[cls]
                self.assertTrue(cs.support, f"seed={seed} {cls} support 为空")
                self.assertTrue(cs.validation, f"seed={seed} {cls} validation 为空")
                self.assertTrue(cs.test, f"seed={seed} {cls} test 为空")

    # ---- 4/5. 集合隔离 ----
    def test_04_no_file_overlap(self) -> None:
        """support / validation / test 之间文件名无交集。"""
        for seed, split in self.splits.items():
            names = {
                s: {r["file_name"] for r in split.flatten(s)} for s in ("support", "validation", "test")
            }
            self.assertFalse(names["support"] & names["validation"], f"seed={seed} support∩validation")
            self.assertFalse(names["support"] & names["test"], f"seed={seed} support∩test")
            self.assertFalse(names["validation"] & names["test"], f"seed={seed} validation∩test")

    def test_05_no_md5_overlap(self) -> None:
        """support / validation / test 之间完整 MD5 无交集。"""
        for seed, split in self.splits.items():
            md5s = {
                s: {r["md5"] for r in split.flatten(s)} for s in ("support", "validation", "test")
            }
            self.assertFalse(md5s["support"] & md5s["validation"], f"seed={seed} support∩validation MD5")
            self.assertFalse(md5s["support"] & md5s["test"], f"seed={seed} support∩test MD5")
            self.assertFalse(md5s["validation"] & md5s["test"], f"seed={seed} validation∩test MD5")

    # ---- 6. 重复图同侧 ----
    def test_06_duplicate_pair_same_split(self) -> None:
        """Pa_101.bmp 与 Pa_105.bmp 在每个 seed 中都属于同一集合。"""
        target = {"Pa_101.bmp", "Pa_105.bmp"}
        for seed, split in self.splits.items():
            location = {}
            for split_name in ("support", "validation", "test"):
                for row in split.flatten(split_name):
                    if row["file_name"] in target:
                        location[row["file_name"]] = split_name
            self.assertEqual(set(location), target, f"seed={seed} 重复图缺失")
            self.assertEqual(
                location["Pa_101.bmp"],
                location["Pa_105.bmp"],
                f"seed={seed} 重复图被拆到不同集合：{location}",
            )
            # 进一步用 MD5 确认（不只看文件名）
            md5s = {r["md5"] for r in split.flatten(location["Pa_101.bmp"]) if r["file_name"] in target}
            self.assertEqual(len(md5s), 1, f"seed={seed} 重复图 MD5 不一致")

    # ---- 7/8. shot 嵌套 ----
    def test_07_one_shot_subset_of_three_shot(self) -> None:
        for seed, split in self.splits.items():
            one = {r["file_name"] for r in split.shot_rows(1)}
            three = {r["file_name"] for r in split.shot_rows(3)}
            self.assertTrue(one <= three, f"seed={seed} 1-shot 不是 3-shot 子集")

    def test_08_three_shot_subset_of_five_shot(self) -> None:
        for seed, split in self.splits.items():
            three = {r["file_name"] for r in split.shot_rows(3)}
            five = {r["file_name"] for r in split.shot_rows(5)}
            self.assertTrue(three <= five, f"seed={seed} 3-shot 不是 5-shot 子集")

    # ---- 9. shot 之间共享 validation/test ----
    def test_09_shots_share_same_validation_and_test(self) -> None:
        """三档 shot 使用同一份 validation / test（逐行比对）。"""
        for seed, split in self.splits.items():
            val = [r["file_name"] for r in split.flatten("validation")]
            test = [r["file_name"] for r in split.flatten("test")]
            for shot in (1, 3, 5):
                split.shot_rows(shot)  # 构造参考集不应改变 val/test
                self.assertEqual(val, [r["file_name"] for r in split.flatten("validation")])
                self.assertEqual(test, [r["file_name"] for r in split.flatten("test")])

    # ---- 10. 完整覆盖且仅一次 ----
    def test_10_full_coverage_exactly_once(self) -> None:
        manifest_names = {r["file_name"] for r in self.rows}
        for seed, split in self.splits.items():
            assigned: list[str] = []
            for split_name in ("support", "validation", "test"):
                assigned.extend(r["file_name"] for r in split.flatten(split_name))
            self.assertEqual(len(assigned), len(set(assigned)), f"seed={seed} 有文件被重复分配")
            self.assertEqual(set(assigned), manifest_names, f"seed={seed} 覆盖不完整")
            self.assertEqual(len(assigned), len(self.rows), f"seed={seed} 分配总数不符")

    # ---- 11. 非图像文件未纳入 ----
    def test_11_no_non_image_files(self) -> None:
        for seed, split in self.splits.items():
            for split_name in ("support", "validation", "test"):
                for row in split.flatten(split_name):
                    self.assertTrue(
                        row["file_name"].lower().endswith(".bmp"),
                        f"seed={seed} 出现非 BMP 文件：{row['file_name']}",
                    )
                    self.assertNotIn("thumbs", row["file_name"].lower())
                    self.assertNotIn("thumbs", row["abs_path"].lower())

    # ---- 12. 每类数量符合设计 ----
    def test_12_per_class_counts(self) -> None:
        """每类 support=5、validation=50、test=245；重复组导致独立内容数变化需可解释。"""
        for seed, split in self.splits.items():
            for cls in CLASS_ORDER:
                cs = split.per_class[cls]
                self.assertEqual(len(cs.support), DEFAULT_N_SUPPORT, f"seed={seed} {cls} support")
                self.assertEqual(len(cs.validation), DEFAULT_N_VAL, f"seed={seed} {cls} validation")
                total = len(cs.support) + len(cs.validation) + len(cs.test)
                expected_total = sum(1 for r in self.rows if r["class_prefix"] == cls)
                self.assertEqual(total, expected_total, f"seed={seed} {cls} 总数不符")
                self.assertEqual(len(cs.test), expected_total - DEFAULT_N_SUPPORT - DEFAULT_N_VAL)

                # 独立内容数：support 必须全不重复
                support_md5 = [r["md5"] for r in cs.support]
                self.assertEqual(
                    len(support_md5), len(set(support_md5)), f"seed={seed} {cls} support 内部有重复内容"
                )

                # Pa 类允许的一个集合因重复组而少 1 个独立内容
                if cls == "Pa":
                    unique = {
                        "support": len({r["md5"] for r in cs.support}),
                        "validation": len({r["md5"] for r in cs.validation}),
                        "test": len({r["md5"] for r in cs.test}),
                    }
                    self.assertIn(
                        unique["validation"],
                        (DEFAULT_N_VAL, DEFAULT_N_VAL - 1),
                        f"seed={seed} Pa validation 独立内容数异常：{unique}",
                    )
                    self.assertIn(
                        unique["test"], (245, 244), f"seed={seed} Pa test 独立内容数异常：{unique}"
                    )
                    self.assertEqual(
                        unique["support"] + unique["validation"] + unique["test"], 299,
                        f"seed={seed} Pa 独立内容总数应为 299，实际 {unique}",
                    )
                else:
                    self.assertEqual(len({r["md5"] for r in cs.validation}), DEFAULT_N_VAL)
                    self.assertEqual(len({r["md5"] for r in cs.test}), 245)

    # ---- 13. 五个 seed 可重新生成且与磁盘结果一致 ----
    def test_13_five_seeds_regenerable_and_match_disk(self) -> None:
        for seed in DEFAULT_SEEDS:
            seed_dir = SPLITS_ROOT / f"seed_{seed}"
            self.assertTrue(seed_dir.is_dir(), f"缺少已生成的划分目录：{seed_dir}")
            regenerated = seed_outputs(self.splits[seed])
            for name in CSV_NAMES:
                disk_path = seed_dir / name
                self.assertTrue(disk_path.is_file(), f"缺少文件：{disk_path}")
                disk_rows = read_csv(disk_path)
                self.assertEqual(
                    len(disk_rows), len(regenerated[name]), f"{disk_path} 行数不一致"
                )
                disk_names = [r["file_name"] for r in disk_rows]
                regen_names = [r["file_name"] for r in regenerated[name]]
                self.assertEqual(disk_names, regen_names, f"{disk_path} 内容与重新生成结果不一致")

    # ---- 14. 无旧路径 ----
    def test_14_no_legacy_path(self) -> None:
        legacy = str(LEGACY_ROOT).lower().replace("\\", "/")
        data_root = str(DATA_ROOT).lower().replace("\\", "/")
        for seed, split in self.splits.items():
            for split_name in ("support", "validation", "test"):
                for row in split.flatten(split_name):
                    path = row["abs_path"].lower().replace("\\", "/")
                    self.assertTrue(path.startswith(data_root), f"seed={seed} 路径不在数据源内：{path}")
                    self.assertNotIn(legacy, path, f"seed={seed} 出现旧路径：{path}")
                    self.assertTrue(Path(row["abs_path"]).is_file(), f"文件不存在：{row['abs_path']}")

    # ---- 补充：整体校验全通过 ----
    def test_15_validation_checks_all_pass(self) -> None:
        for seed, validation in self.validations.items():
            failed = [k for k, v in validation["checks"].items() if not v["pass"]]
            self.assertEqual(failed, [], f"seed={seed} 校验项失败：{failed}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
