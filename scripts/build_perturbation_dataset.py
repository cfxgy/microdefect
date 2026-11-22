#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-1 受控扰动数据集构建脚本。

用法（必须使用项目虚拟环境解释器）：
    D:\\microdefect\\.venv\\Scripts\\python.exe scripts\\build_perturbation_dataset.py

默认：
    划分    experiments/results/splits
    输出    experiments/data/perturbations
    数据    data/NEU-CLS
    种子    42,123,2024,7,999

只读 P1-1 清单与 P1-2 划分；不触碰 P2/P3 产物与原始图片。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.perturbation_dataset import (  # noqa: E402
    SCHEMA,
    SPLITS,
    DatasetConfig,
    DatasetConflictError,
    directory_size,
    generate_seed,
    verify_seed,
)
from src.data.perturbations import build_specs, specs_config_signature  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "results" / "neu_cls_manifest.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "experiments" / "results" / "splits"
DEFAULT_DATA = PROJECT_ROOT / "data" / "NEU-CLS"
DEFAULT_OUT = PROJECT_ROOT / "experiments" / "data" / "perturbations"


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def merge_verification_into_global(out_root: Path, verification: dict) -> bool:
    """把独立校验进程的结果合并回 global_report.json（幂等）。"""
    global_path = out_root / "global_report.json"
    if not global_path.is_file():
        return False
    report = json.loads(global_path.read_text(encoding="utf-8"))
    per_seed = verification.get("per_seed", {})
    report["verification"] = {
        "all_ok": verification.get("all_ok"),
        "duplicate_output_groups_total": sum(
            item.get("duplicate_output_groups", 0) for item in per_seed.values()
        ),
        "elapsed_seconds": verification.get("elapsed_seconds"),
        "per_seed": per_seed,
    }
    for item in report.get("per_seed", []):
        key = f"seed_{item['seed']}"
        if key in per_seed:
            item["verification"] = per_seed[key]
    global_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="微缺智学 P4-1 受控扰动数据集构建")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--splits-root", default=str(DEFAULT_SPLITS))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="42,123,2024,7,999")
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--limit", type=int, default=0, help="每个 split 仅取前 N 张（冒烟用）")
    parser.add_argument("--resume", action="store_true", help="跳过已完成且签名一致的 seed")
    parser.add_argument("--no-verify", action="store_true", help="跳过逐文件校验")
    parser.add_argument("--verify-only", action="store_true", help="只做校验（需已有生成结果）")
    parser.add_argument("--fsync", action="store_true", help="写入后 fsync（更慢，抗断电）")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    out_root = Path(args.out)
    specs = build_specs()

    print("=" * 74)
    print("微缺智学 P4-1 受控扰动数据集构建")
    print("=" * 74)
    print(f"划分      : {args.splits_root}")
    print(f"输出      : {out_root}")
    print(f"seeds     : {list(seeds)}  splits: {list(splits)}  limit: {args.limit or '全部'}")
    print(f"扰动配置  : {[s.name for s in specs]}")
    print(f"配置签名  : {specs_config_signature()}")

    config = DatasetConfig(
        manifest_path=Path(args.manifest),
        splits_root=Path(args.splits_root),
        data_root=Path(args.data_root),
        out_root=out_root,
        seeds=seeds,
        splits=splits,
        limit=args.limit,
        resume=args.resume,
        fsync=args.fsync,
    )

    started = time.perf_counter()
    seed_reports: list[dict] = []
    verifications: dict[str, dict] = {}
    failures: list[str] = []

    if args.verify_only:
        for seed in seeds:
            seed_dir = out_root / f"seed_{seed}"
            report_path = seed_dir / "perturbation_report.json"
            if not report_path.is_file():
                failures.append(f"seed_{seed}: 缺少 perturbation_report.json，无法校验")
                continue
            result = verify_seed(seed_dir)
            verifications[f"seed_{seed}"] = result
            stored = json.loads(report_path.read_text(encoding="utf-8"))
            stored["integrity"] = result
            report_path.write_text(
                json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        summary = {
            "schema": SCHEMA + "/verification",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "all_ok": all(v["ok"] for v in verifications.values()) if verifications else False,
            "per_seed": verifications,
            "failures": failures,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        (out_root / "verification_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        merge_verification_into_global(out_root, summary)
        print("-" * 74)
        for name, result in verifications.items():
            print(
                f"{name}: 文件 {result['files_checked']}, 缺 {result['missing']}, "
                f"损坏 {result['decode_errors']}, MD5 不符 {result['md5_mismatch']}, "
                f"重复组 {result['duplicate_output_groups']}"
            )
        print(f"全部通过: {summary['all_ok']}，耗时 {summary['elapsed_seconds']}s")
        return 0 if summary["all_ok"] and not failures else 1

    for seed in seeds:
        try:
            report = generate_seed(config, seed, specs)
            seed_reports.append(report)
            if not args.no_verify:
                verifications[f"seed_{seed}"] = verify_seed(out_root / f"seed_{seed}")
                report_path = out_root / f"seed_{seed}" / "perturbation_report.json"
                stored = json.loads(report_path.read_text(encoding="utf-8"))
                stored["integrity"] = verifications[f"seed_{seed}"]
                report_path.write_text(
                    json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except DatasetConflictError as exc:
            print(f"[FAIL] seed={seed} 冲突：{exc}")
            failures.append(f"seed_{seed}: {exc}")
        except Exception as exc:  # 记录后继续判定整体失败
            print(f"[FAIL] seed={seed} 异常：{type(exc).__name__}: {exc}")
            failures.append(f"seed_{seed}: {type(exc).__name__}: {exc}")

    total_seconds = time.perf_counter() - started
    total_files = sum(r["counts"]["total_files"] for r in seed_reports)
    total_bytes = sum(r["counts"]["bytes"] for r in seed_reports)
    total_perturbations = sum(r["counts"]["perturbations"] for r in seed_reports)
    disk_bytes = directory_size(out_root) if out_root.exists() else 0

    verify_all_ok = all(v["ok"] for v in verifications.values()) if verifications else None
    aggregate_duplicates = sum(v["duplicate_output_groups"] for v in verifications.values())

    global_report = {
        "schema": SCHEMA + "/global",
        "generator": "scripts/build_perturbation_dataset.py",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete" if not failures else "failed",
        "config": {
            "seeds": list(seeds),
            "splits": list(splits),
            "limit": args.limit,
            "config_signature": specs_config_signature(),
            "perturbation_specs": [
                {"name": s.name, "kind": s.kind, "level": s.level, "params": s.params}
                for s in specs
            ],
            "source_manifest": str(Path(args.manifest)),
            "splits_root": str(Path(args.splits_root)),
        },
        "totals": {
            "seeds_completed": len(seed_reports),
            "files": total_files,
            "perturbations": total_perturbations,
            "declared_bytes": total_bytes,
            "declared_mib": round(total_bytes / 1024 ** 2, 2),
            "on_disk_bytes": disk_bytes,
            "on_disk_mib": round(disk_bytes / 1024 ** 2, 2),
            "wall_seconds": round(total_seconds, 2),
        },
        "per_seed": [
            {
                "seed": r["seed"],
                "sources": r["counts"]["sources"],
                "originals": r["counts"]["originals"],
                "perturbations": r["counts"]["perturbations"],
                "total_files": r["counts"]["total_files"],
                "mib": r["counts"]["mib"],
                "by_split": r["by_split"],
                "generation_seconds": r["timing"]["generation_seconds"],
                "verification": verifications.get(f"seed_{r['seed']}"),
            }
            for r in seed_reports
        ],
        "verification": {
            "all_ok": verify_all_ok,
            "duplicate_output_groups_total": aggregate_duplicates,
            "per_seed": verifications,
        },
        "failures": failures,
        "notes": [
            "本数据集由 NEU-CLS 原始图像人工生成，属于受控模拟扰动，不是真实产线采集数据。",
            "图像扰动不等同于真实相机、照明、工件运动或生产线变化。",
            "重复输出 MD5 来自 NEU-CLS 中本就重复的原图（Pa_101/Pa_105），属于预期现象。",
            "本阶段不包含任何漂移检测性能指标。",
        ],
    }
    (out_root / "global_report.json").write_text(
        json.dumps(global_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-" * 74)
    for item in global_report["per_seed"]:
        print(
            f"seed={item['seed']:<5} 原图 {item['originals']:>5}  扰动 {item['perturbations']:>6}  "
            f"共 {item['total_files']:>6} 文件  {item['mib']:>8} MiB  {item['generation_seconds']:>6}s"
        )
    print("-" * 74)
    print(f"总文件 {total_files}，扰动 {total_perturbations}，磁盘 {global_report['totals']['on_disk_mib']} MiB")
    print(f"校验全部通过: {verify_all_ok}")
    print(f"总耗时 {total_seconds:.1f}s")
    if failures:
        print(f"[FAIL] {len(failures)} 项失败：")
        for item in failures:
            print("  -", item)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
