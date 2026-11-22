#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2 结果独立审计。

只读：不修改任何结果文件。输出 ``audit_report.json`` 到结果目录。

审计分三类：
1. 历史产物完好性（P1/P2/P3/P4-1 关键文件哈希/数量）；
2. 防泄漏与可追溯性（参考池是否只来自 validation、窗口 source 是否可追溯）；
3. 结果自洽性（指标能否从逐窗口分数重算、是否有 NaN/Inf、窗口是否缺失/重复）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import CLASS_ORDER  # noqa: E402
from src.drift.metrics import binary_metrics_from_predictions, mean_std  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "experiments" / "results" / "drift_monitoring"
DEFAULT_PERT = PROJECT_ROOT / "experiments" / "data" / "perturbations"
DEFAULT_SPLITS = PROJECT_ROOT / "experiments" / "results" / "splits"

EXPECTED_HASHES = {
    "experiments/results/neu_cls_manifest.csv":
        "A7F6D67755E764B2C68BC2E5716A045550FE80851351FF8D949786C7234DC631",
    "experiments/results/splits/seed_42/test.csv":
        "9BD3DD13523E07585A12F78DF56EEDFAD217A8E9028D73A695DBC3BF42DB58F5",
    "experiments/results/splits/seed_7/validation.csv":
        "10D1156A0E7DF811AF947054CCD51AAF1DE03467D0DED569BF170C23981BD40E",
    "experiments/results/fewshot/summary.json":
        "630317147F5783780B7B06801E0B0B2D2740BBF38C41E955FC859FC0821C18EA",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(out_dir: Path, pert_root: Path) -> dict:
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, **details) -> None:
        checks[name] = {"pass": bool(passed), **details}

    # ---- 1. 历史产物 ----
    hash_results = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = PROJECT_ROOT / relative
        actual = sha256_file(path) if path.is_file() else None
        hash_results[relative] = {
            "expected": expected.lower(),
            "actual": actual.lower() if actual else None,
            "match": actual is not None and actual.lower() == expected.lower(),
        }
    record("historical_hashes_unchanged", all(v["match"] for v in hash_results.values()), files=hash_results)

    bmp_files = list((PROJECT_ROOT / "data" / "NEU-CLS").glob("*.bmp"))
    record("original_images_intact", len(bmp_files) == 1800, count=len(bmp_files))

    global_report = json.loads((pert_root / "global_report.json").read_text(encoding="utf-8"))
    record(
        "p41_global_report_ok",
        global_report.get("status") == "complete"
        and global_report.get("verification", {}).get("all_ok") is True,
        status=global_report.get("status"),
        verification_all_ok=global_report.get("verification", {}).get("all_ok"),
    )

    # ---- 2. 防泄漏与可追溯 ----
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    seeds = config["seeds"]
    leakage: dict[str, dict] = {}
    traceability: dict[str, dict] = {}
    completeness: dict[str, dict] = {}

    for seed in seeds:
        pert_rows = read_csv(pert_root / f"seed_{seed}" / "perturbation_manifest.csv")
        originals = {
            (row["split"], row["source_file"]): row
            for row in pert_rows
            if row["perturbation_type"] == "original"
        }
        reference = read_csv(out_dir / f"seed_{seed}" / "reference_pool.csv")
        pool_ids = {row["source_file"] for row in reference}
        val_ids = {row["source_file"] for row in pert_rows if row["split"] == "validation" and row["perturbation_type"] == "original"}
        test_ids = {row["source_file"] for row in pert_rows if row["split"] == "test" and row["perturbation_type"] == "original"}

        leakage[f"seed_{seed}"] = {
            "reference_in_validation": pool_ids <= val_ids,
            "reference_in_test": len(pool_ids & test_ids),
            "validation_test_source_overlap": len(val_ids & test_ids),
            "pool_size": len(pool_ids),
        }

        windows = read_csv(out_dir / f"seed_{seed}" / "test_window_scores.csv")
        unknown = []
        for row in windows:
            for sid in row["source_ids"].split("|"):
                if ("test", sid) not in originals:
                    unknown.append(sid)
        traceability[f"seed_{seed}"] = {
            "windows": len(windows),
            "untraceable_sources": len(unknown),
            "examples": unknown[:3],
        }

        expected_windows = sum(1 for r in windows if r["is_drift"] == "0")
        per_class_clean = Counter(r["class_name"] for r in windows if r["is_drift"] == "0")
        duplicate_ids = len(windows) - len({(r["class_name"], r["window_id"]) for r in windows})
        completeness[f"seed_{seed}"] = {
            "windows": len(windows),
            "clean_windows": expected_windows,
            "per_class_clean": dict(per_class_clean),
            "duplicate_window_keys": duplicate_ids,
        }

    record(
        "no_test_in_reference_pool",
        all(v["reference_in_test"] == 0 for v in leakage.values()),
        detail=leakage,
    )
    record(
        "no_validation_test_source_overlap",
        all(v["validation_test_source_overlap"] == 0 for v in leakage.values()),
        detail={k: v["validation_test_source_overlap"] for k, v in leakage.items()},
    )
    record(
        "all_windows_traceable",
        all(v["untraceable_sources"] == 0 for v in traceability.values()),
        detail=traceability,
    )
    record(
        "no_duplicate_windows",
        all(v["duplicate_window_keys"] == 0 for v in completeness.values()),
        detail=completeness,
    )
    record("all_seeds_completed", len(seeds) == 5 and all(
        (out_dir / f"seed_{seed}" / "metrics_overall.json").is_file() for seed in seeds
    ), seeds=seeds)

    # ---- 3. 结果自洽 ----
    consistency: dict[str, dict] = {}
    nan_issues = []
    for seed in seeds:
        windows = read_csv(out_dir / f"seed_{seed}" / "test_window_scores.csv")
        threshold_rows = json.loads((out_dir / f"seed_{seed}" / "calibration_thresholds.json").read_text(encoding="utf-8"))
        thresholds = {row["class_name"]: row for row in threshold_rows["thresholds"]}
        stored = read_csv(out_dir / f"seed_{seed}" / "metrics_by_class.csv")

        for row in windows:
            for key in ("score_A", "score_B", "score_fusion"):
                value = float(row[key])
                if value != value or value in (float("inf"), float("-inf")):
                    nan_issues.append(f"{seed}:{row['window_id']}:{key}")

        recomputed = []
        for cls in CLASS_ORDER:
            rows = [r for r in windows if r["class_name"] == cls]
            for channel in ("A", "B", "fusion"):
                score_key = f"score_{channel}" if channel != "fusion" else "score_fusion"
                result = binary_metrics_from_predictions(
                    [float(r[score_key]) for r in rows],
                    [int(r["is_drift"]) for r in rows],
                    [int(r[f"is_drift_{channel}"]) for r in rows],
                )
                recomputed.append((cls, channel, result))

        mismatches = []
        for cls, channel, result in recomputed:
            match = next((r for r in stored if r["class_name"] == cls and r["channel"] == channel), None)
            if match is None:
                mismatches.append(f"{seed}/{cls}/{channel}: 缺失存储指标")
                continue
            for key in ("precision", "recall", "f1"):
                if abs(float(match[key]) - result[key]) > 1e-6:
                    mismatches.append(f"{seed}/{cls}/{channel}:{key}")
        consistency[f"seed_{seed}"] = {"mismatches": mismatches, "classes": len(CLASS_ORDER)}

    record("no_nan_or_inf_scores", not nan_issues, examples=nan_issues[:5])
    record(
        "metrics_recomputable_from_windows",
        all(len(v["mismatches"]) == 0 for v in consistency.values()),
        detail={k: v["mismatches"][:3] for k, v in consistency.items()},
    )

    # 阈值按类别独立
    per_class_thresholds = {}
    for seed in seeds:
        payload = json.loads((out_dir / f"seed_{seed}" / "calibration_thresholds.json").read_text(encoding="utf-8"))
        names = [row["class_name"] for row in payload["thresholds"]]
        per_class_thresholds[f"seed_{seed}"] = {
            "classes": sorted(names),
            "distinct_fusion_thresholds": len({round(row["threshold_fusion"], 9) for row in payload["thresholds"]}),
        }
    record(
        "thresholds_per_class",
        all(set(v["classes"]) == set(CLASS_ORDER) for v in per_class_thresholds.values()),
        detail=per_class_thresholds,
    )

    # 图表
    plots = sorted((out_dir / "plots").glob("*.png")) if (out_dir / "plots").is_dir() else []
    record("plots_present", len(plots) >= 6, count=len(plots), files=[p.name for p in plots])

    # 可复现性：同一 seed 的 summary 文件是否与 summary.csv 一致
    summary_rows = read_csv(out_dir / "summary.csv")
    by_seed_channel = {(int(r["seed"]), r["channel"]): r for r in summary_rows}
    consistent_summary = True
    for seed in seeds:
        metrics = json.loads((out_dir / f"seed_{seed}" / "metrics_overall.json").read_text(encoding="utf-8"))
        for channel in ("A", "B", "fusion"):
            stored_row = by_seed_channel.get((seed, channel))
            if stored_row is None:
                consistent_summary = False
                continue
            if abs(float(stored_row["f1"]) - float(metrics["overall"][channel]["f1"])) > 1e-6:
                consistent_summary = False
    record("summary_matches_per_seed_metrics", consistent_summary)

    passed = all(item["pass"] for item in checks.values())
    return {
        "schema": "microdefect.drift_monitoring/audit",
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "checks": checks,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P4-2 结果审计")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pert-root", default=str(DEFAULT_PERT))
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not (out_dir / "summary.json").is_file():
        print(f"[FAIL] 结果目录缺少 summary.json：{out_dir}")
        return 2
    report = audit(out_dir, Path(args.pert_root))
    target = out_dir / "audit_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("P4-2 结果审计")
    print("=" * 70)
    for name, item in report["checks"].items():
        print(f"[{'PASS' if item['pass'] else 'FAIL'}] {name}")
    print("-" * 70)
    print(f"总体: {'PASS' if report['pass'] else 'FAIL'}")
    print(f"报告: {target}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
