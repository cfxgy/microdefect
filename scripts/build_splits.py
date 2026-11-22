#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 NEU-CLS 五个随机种子的实验划分（P1-2）。

用法（必须使用项目虚拟环境的解释器）：
    D:\\microdefect\\.venv\\Scripts\\python.exe scripts\\build_splits.py

读取：
    experiments/results/neu_cls_manifest.csv   （P1-1 产物，只读，不修改）

输出：
    experiments/results/splits/seed_<seed>/
        support_5shot.csv
        support_3shot.csv
        support_1shot.csv
        validation.csv
        test.csv
        split_report.json

本脚本不读图像像素、不提特征、不训练模型。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import (  # noqa: E402
    CLASS_ORDER,
    DEFAULT_N_SUPPORT,
    DEFAULT_N_VAL,
    DEFAULT_SEEDS,
    MANIFEST_FIELDS,
    SplitConfig,
    SplitError,
    build_seed_report,
    build_seed_split,
    load_manifest,
    seed_outputs,
    summarize,
    validate_seed_split,
    write_csv,
)

GENERATOR = "scripts/build_splits.py"

DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "results" / "neu_cls_manifest.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "NEU-CLS"
DEFAULT_OUT = PROJECT_ROOT / "experiments" / "results" / "splits"
LEGACY_ROOT = PROJECT_ROOT / "NEU-CLS"

CSV_NAMES = (
    "support_5shot.csv",
    "support_3shot.csv",
    "support_1shot.csv",
    "validation.csv",
    "test.csv",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_seed_dir(base: Path, seed: int) -> tuple[Path, bool]:
    """避免覆盖非本脚本生成的结果。返回 (目录, 是否改名)。"""
    target = base / f"seed_{seed}"
    if not target.exists():
        return target, False
    report = target / "split_report.json"
    if report.is_file():
        try:
            old = json.loads(report.read_text(encoding="utf-8"))
            if old.get("generator") == "src/data/split.py":
                return target, False
        except Exception:
            pass
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"seed_{seed}_{stamp}", True


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("至少需要一个 seed")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description="微缺智学 P1-2 NEU-CLS 数据划分")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT), help="数据源根目录")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="划分输出根目录")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--n-support", type=int, default=DEFAULT_N_SUPPORT)
    parser.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    data_root = Path(args.root)
    out_root = Path(args.out)
    seeds = parse_seeds(args.seeds)

    print("=" * 70)
    print("微缺智学 P1-2 NEU-CLS 实验数据划分")
    print("=" * 70)
    print(f"清单      : {manifest_path}")
    print(f"数据源    : {data_root}")
    print(f"输出根目录: {out_root}")
    print(f"随机种子  : {list(seeds)}")
    print(f"划分规模  : support={args.n_support}  validation={args.n_val}  其余为 test")

    if not data_root.is_dir():
        print(f"[FAIL] 数据源目录不存在：{data_root}")
        return 2
    if LEGACY_ROOT.exists():
        print(f"[warn] 旧路径仍存在（仅报告）：{LEGACY_ROOT}")

    try:
        rows = load_manifest(manifest_path)
    except SplitError as exc:
        print(f"[FAIL] 读取清单失败：{exc}")
        return 2

    print(f"清单样本数: {len(rows)}")
    field_check = all(f in rows[0] for f in MANIFEST_FIELDS)
    print(f"清单字段  : {'完整' if field_check else '缺失'}")

    config = SplitConfig(
        manifest_path=manifest_path,
        data_root=data_root,
        n_support=args.n_support,
        n_val=args.n_val,
        seeds=seeds,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict] = []
    written: dict[int, dict] = {}

    for seed in seeds:
        seed_split = build_seed_split(rows, seed, config)
        validation = validate_seed_split(
            seed_split,
            rows,
            data_root=data_root,
            legacy_root=LEGACY_ROOT,
            n_support=args.n_support,
            n_val=args.n_val,
        )
        report = build_seed_report(seed_split, validation, config)

        seed_dir, renamed = resolve_seed_dir(out_root, seed)
        seed_dir.mkdir(parents=True, exist_ok=True)

        files = seed_outputs(seed_split)
        hashes = {}
        for name in CSV_NAMES:
            path = seed_dir / name
            write_csv(path, files[name])
            hashes[name] = sha256_of(path)

        report["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        report["outputs"] = {
            "directory": str(seed_dir),
            "renamed_due_to_conflict": renamed,
            "files": {name: {"rows": len(files[name]), "sha256": hashes[name]} for name in CSV_NAMES},
        }
        (seed_dir / "split_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        seed_results.append(validation)
        written[seed] = {
            "dir": seed_dir,
            "report": report,
            "validation": validation,
        }

        status = "PASS" if validation["all_pass"] else "FAIL"
        print(
            f"[{status}] seed={seed:<5} support={validation['totals']['support_files']:>3} "
            f"val={validation['totals']['validation_files']:>3} test={validation['totals']['test_files']:>3} "
            f"unique(md5)={validation['totals']['support_unique_md5']}/"
            f"{validation['totals']['validation_unique_md5']}/"
            f"{validation['totals']['test_unique_md5']} "
            f"-> {seed_dir.name}"
        )

    summary = summarize(seed_results)

    print("-" * 70)
    print("按类别核对（文件数 / 独立MD5），以 seed=42 为例：")
    first = seed_results[0]
    print(f"{'类别':<4}{'support':>12}{'validation':>14}{'test':>14}")
    for cls in CLASS_ORDER:
        pc = first["per_class"][cls]
        print(
            f"{cls:<4}"
            f"{pc['support']['files']:>6}/{pc['support']['unique_md5']:<5}"
            f"{pc['validation']['files']:>8}/{pc['validation']['unique_md5']:<5}"
            f"{pc['test']['files']:>8}/{pc['test']['unique_md5']:<5}"
        )
    print("-" * 70)
    print("重复组归属（每个种子）：")
    for seed in seeds:
        groups = written[seed]["validation"]["duplicate_groups"]
        for group in groups:
            print(f"  seed={seed:<5} {group['dup_group']} {group['files']} -> {group['splits']}")
    print("-" * 70)
    print("跨种子汇总：")
    for seed in seeds:
        info = summary["per_seed"][str(seed)]
        print(
            f"  seed={seed:<5} support={info['support_files']} val={info['validation_files']} "
            f"test={info['test_files']} pass={info['all_pass']}"
        )
    print("-" * 70)
    print(f"总体结果: {'PASS' if summary['all_pass'] else 'FAIL'}")
    print(f"输出目录: {out_root}")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
