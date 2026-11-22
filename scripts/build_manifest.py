#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-1 NEU-CLS 数据集扫描与清单生成（微缺智学）。

只读扫描：不删除、不移动、不修改任何原始数据。

用法（必须使用项目虚拟环境的解释器）：
    D:\\microdefect\\.venv\\Scripts\\python.exe scripts\\build_manifest.py

默认数据源（本阶段唯一认可路径）：
    D:\\microdefect\\data\\NEU-CLS

输出：
    experiments/results/neu_cls_scan_report.json
    experiments/results/neu_cls_manifest.csv

清单字段：
    file_name, rel_path, abs_path, class_prefix, class_name,
    width, height, mode, size_bytes, md5, leak_group, dup_group, is_duplicate

  * leak_group：用于后续防泄漏划分的分组键（内容哈希派生）；内容相同的图片
    一定落在同一 leak_group，划分时必须整组同侧。
  * dup_group：仅内容重复的图片才填写（如 dup_1），其余为空。
  * is_duplicate：该图片是否为某个重复组的成员（True/False）。

脚本可重复运行；若目标文件已存在但不是本脚本格式生成的，会改用带时间戳的
新文件名，不覆盖未知内容。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

GENERATOR = "scripts/build_manifest.py"
SCAN_SCHEMA = "microdefect.neu_cls_scan_report/1"
MANIFEST_SCHEMA = "microdefect.neu_cls_manifest/1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "NEU-CLS"
LEGACY_ROOT = PROJECT_ROOT / "NEU-CLS"

# 通行映射约定（数据集本身未附带类别说明文件，此处为项目约定）
CLASS_NAMES = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "PS": "Pitted Surface",
    "RS": "Rolled-in Scale",
    "Sc": "Scratches",
}
CLASS_ORDER = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
EXPECTED_PER_CLASS = 300
EXPECTED_TOTAL = EXPECTED_PER_CLASS * len(CLASS_ORDER)
EXPECTED_SIZE = (200, 200)
EXPECTED_MODE = "L"
TARGET_DUP_PAIR = ("Pa_101.bmp", "Pa_105.bmp")

CSV_HEADER = [
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
]

NAME_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)_(?P<index>\d+)$")


def md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_name(stem: str) -> tuple[str | None, int | None]:
    match = NAME_RE.match(stem)
    if not match:
        return None, None
    prefix = match.group("prefix")
    if prefix not in CLASS_NAMES:
        return None, int(match.group("index"))
    return prefix, int(match.group("index"))


def sort_key(row: dict) -> tuple:
    prefix = row["class_prefix"]
    order = CLASS_ORDER.index(prefix) if prefix in CLASS_ORDER else len(CLASS_ORDER)
    index = row.get("_index")
    return (order, index if index is not None else 10 ** 9, row["file_name"])


def safe_json_out(path: Path, payload: dict) -> tuple[Path, bool]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                old = json.load(fh)
            mine = isinstance(old, dict) and old.get("generator") == GENERATOR
        except Exception:
            mine = False
        if not mine:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return path.with_name(f"{path.stem}_{stamp}{path.suffix}"), True
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path, False


def safe_csv_out(path: Path, rows: list[dict]) -> tuple[Path, bool]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                first_line = fh.readline().strip()
            mine = first_line.split(",") == CSV_HEADER
        except Exception:
            mine = False
        if not mine:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            mine = True
            renamed = True
        else:
            renamed = False
    else:
        renamed = False
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADER, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path, renamed


def main() -> int:
    parser = argparse.ArgumentParser(description="微缺智学 P1-1 NEU-CLS 扫描与清单")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="数据集根目录")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "experiments" / "results"),
        help="输出目录",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅扫描前 N 张（0=全部，调试用）")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("微缺智学 P1-1 NEU-CLS 数据集扫描")
    print("=" * 66)
    print(f"数据源根目录 : {root}")
    print(f"输出目录     : {out_dir}")

    if not root.is_dir():
        print(f"[FAIL] 数据源目录不存在：{root}")
        return 2

    legacy_exists = LEGACY_ROOT.exists()
    print(f"旧路径 {LEGACY_ROOT} : {'仍存在（仅报告，未处理）' if legacy_exists else '不存在'}")

    all_files = sorted([p for p in root.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    all_files = [p for p in all_files if p.name not in {"Thumbs.db"}]

    bmp_files: list[Path] = []
    excluded: list[dict] = []
    for path in all_files:
        if path.suffix.lower() == ".bmp":
            bmp_files.append(path)
        else:
            excluded.append(
                {
                    "file_name": path.name,
                    "reason": "非 .bmp 文件，已排除",
                    "size_bytes": path.stat().st_size,
                }
            )
    # Thumbs.db 单独记录，便于审计
    thumbs = root / "Thumbs.db"
    if thumbs.exists():
        excluded.append(
            {
                "file_name": thumbs.name,
                "reason": "Windows 缩略图缓存，非图像，已排除",
                "size_bytes": thumbs.stat().st_size,
            }
        )

    if args.limit > 0:
        bmp_files = bmp_files[: args.limit]

    print(f"BMP 文件数   : {len(bmp_files)}")

    rows: list[dict] = []
    corrupt: list[dict] = []
    size_counter: Counter = Counter()
    mode_counter: Counter = Counter()

    from PIL import Image  # 延迟导入，便于 --help 快速返回

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in bmp_files:
            prefix, index = classify_name(path.stem)
            md5 = md5_of(path)
            size_bytes = path.stat().st_size
            width = height = None
            mode = None
            readable = False
            error = None
            try:
                with Image.open(path) as img:
                    img.load()
                    width, height = img.size
                    mode = img.mode
                    readable = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                corrupt.append({"file_name": path.name, "error": error})
            if readable:
                size_counter[f"{width}x{height}"] += 1
                mode_counter[mode] += 1
            rows.append(
                {
                    "file_name": path.name,
                    "rel_path": path.relative_to(root).as_posix(),
                    "abs_path": str(path),
                    "class_prefix": prefix or "",
                    "class_name": CLASS_NAMES.get(prefix, "") if prefix else "",
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "size_bytes": size_bytes,
                    "md5": md5,
                    "leak_group": f"g_{md5[:10]}",
                    "dup_group": "",
                    "is_duplicate": False,
                    "readable": readable,
                    "error": error,
                    "_index": index,
                }
            )

    # ---- 重复检测 ----
    by_md5: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_md5[row["md5"]].append(row)
    dup_groups_raw = {k: v for k, v in by_md5.items() if len(v) > 1}
    dup_groups = sorted(dup_groups_raw.items(), key=lambda kv: sorted(r["file_name"] for r in kv[1]))
    for group_idx, (md5, members) in enumerate(dup_groups, start=1):
        group_id = f"dup_{group_idx}"
        for member in members:
            member["dup_group"] = group_id
            member["is_duplicate"] = True

    rows.sort(key=sort_key)

    # ---- 统计 ----
    class_counter = Counter(row["class_prefix"] for row in rows if row["class_prefix"])
    unknown_class = [row["file_name"] for row in rows if not row["class_prefix"]]
    total_bytes = sum(row["size_bytes"] for row in rows)

    pair_info = {}
    for name in TARGET_DUP_PAIR:
        matches = [r for r in rows if r["file_name"] == name]
        pair_info[name] = {
            "exists": bool(matches),
            "md5": matches[0]["md5"] if matches else None,
            "size_bytes": matches[0]["size_bytes"] if matches else None,
            "dup_group": matches[0]["dup_group"] if matches else None,
        }
    pair_equal = (
        all(v["exists"] for v in pair_info.values())
        and len({v["md5"] for v in pair_info.values()}) == 1
    )

    # ---- 验收项 ----
    checks = {
        "dataset_root_exists": {"pass": root.is_dir(), "actual": str(root)},
        "bmp_count_1800": {"pass": len(rows) == EXPECTED_TOTAL, "actual": len(rows), "expected": EXPECTED_TOTAL},
        "class_count_6": {"pass": len(class_counter) == len(CLASS_ORDER), "actual": dict(sorted(class_counter.items()))},
        "each_class_300": {
            "pass": all(class_counter.get(c) == EXPECTED_PER_CLASS for c in CLASS_ORDER),
            "actual": {c: class_counter.get(c, 0) for c in CLASS_ORDER},
            "expected": EXPECTED_PER_CLASS,
        },
        "all_200x200": {
            "pass": set(size_counter.keys()) == {f"{EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}"},
            "actual": dict(size_counter),
            "expected": f"{EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}",
        },
        "all_mode_L": {"pass": set(mode_counter.keys()) == {EXPECTED_MODE}, "actual": dict(mode_counter)},
        "no_corrupt_images": {"pass": len(corrupt) == 0, "actual": len(corrupt)},
        "all_names_parsed": {"pass": not unknown_class, "actual": unknown_class[:10]},
        "target_duplicate_pair_verified": {
            "pass": pair_equal,
            "actual": pair_info,
        },
        "legacy_path_absent": {"pass": not legacy_exists, "actual": str(LEGACY_ROOT), "note": "旧路径存在时仅报告，不删除"},
    }
    all_pass = all(v["pass"] for v in checks.values())

    payload = {
        "schema": SCAN_SCHEMA,
        "generator": GENERATOR,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "dataset_root_resolved": str(root.resolve()),
        "declared_unique_source": "D:\\microdefect\\data\\NEU-CLS",
        "legacy_root": {"path": str(LEGACY_ROOT), "exists": legacy_exists},
        "scan_options": {"only_suffix": ".bmp", "limit": args.limit},
        "totals": {
            "bmp_count": len(rows),
            "class_count": len(class_counter),
            "total_bytes": total_bytes,
            "total_mib": round(total_bytes / 1024 ** 2, 2),
            "readable": sum(1 for r in rows if r["readable"]),
            "unreadable": len(corrupt),
        },
        "class_counts": {c: class_counter.get(c, 0) for c in CLASS_ORDER},
        "class_names": CLASS_NAMES,
        "size_distribution": dict(size_counter),
        "mode_distribution": dict(mode_counter),
        "excluded_files": excluded,
        "corrupt_images": corrupt,
        "unknown_class_files": unknown_class,
        "duplicates": {
            "group_count": len(dup_groups),
            "image_count_in_duplicate_groups": sum(len(v) for v in dup_groups_raw.values()),
            "groups": [
                {
                    "dup_group": f"dup_{i}",
                    "md5": md5,
                    "size_bytes": members[0]["size_bytes"],
                    "files": [r["file_name"] for r in members],
                }
                for i, (md5, members) in enumerate(dup_groups, start=1)
            ],
            "target_pair": pair_info,
            "target_pair_equal": pair_equal,
        },
        "checks": checks,
        "all_pass": all_pass,
        "notes": [
            "类别全名（Crazing/Inclusion/...）为项目内通行映射约定，数据集目录中未发现类别说明文件。",
            "六类为钢材表面缺陷类别，不是六种产品；该数据集不含正常类别，不能单独用于证明正常/异常检测能力。",
            "leak_group 由内容 MD5 派生，后续划分必须整组同侧以避免重复图片泄漏。",
            "本次扫描未删除、未移动、未修改任何原始图片。",
        ],
    }

    scan_path, scan_renamed = safe_json_out(out_dir / "neu_cls_scan_report.json", payload)
    manifest_path, manifest_renamed = safe_csv_out(out_dir / "neu_cls_manifest.csv", rows)

    # ---- 控制台摘要 ----
    print("-" * 66)
    for cls in CLASS_ORDER:
        print(f"  {cls:<3} {CLASS_NAMES[cls]:<16} {class_counter.get(cls, 0):>5} 张")
    print(f"  {'合计':<3} {'':<16} {len(rows):>5} 张")
    print("-" * 66)
    print(f"总大小       : {payload['totals']['total_mib']} MiB")
    print(f"尺寸分布     : {dict(size_counter)}")
    print(f"模式分布     : {dict(mode_counter)}")
    print(f"损坏图片     : {len(corrupt)}")
    print(f"重复组       : {len(dup_groups)} 组 / {payload['duplicates']['image_count_in_duplicate_groups']} 张")
    for group in payload["duplicates"]["groups"]:
        print(f"  - {group['dup_group']} md5={group['md5']} files={group['files']}")
    print(
        f"Pa_101/Pa_105: 存在={all(v['exists'] for v in pair_info.values())} "
        f"MD5相同={pair_equal}"
    )
    print("-" * 66)
    for key, val in checks.items():
        print(f"{'PASS' if val['pass'] else 'FAIL'}  {key}")
    print("-" * 66)
    print(f"扫描报告     : {scan_path}" + ("（改用新文件名）" if scan_renamed else ""))
    print(f"数据清单     : {manifest_path}" + ("（改用新文件名）" if manifest_renamed else ""))
    print(f"总体结果     : {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
