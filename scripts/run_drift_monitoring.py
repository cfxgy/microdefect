#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2 漂移监测实验编排脚本。

阶段：输入审计 → 特征/统计准备 → bootstrap 校准 → validation 验证 →
test 配对评估 → 指标 → 图表 → 汇总。

所有产物默认写入 ``experiments/results/drift_monitoring``；可用 ``--out`` 改到
可写目录（例如沙箱受限时先写 cache 再复制）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "mpl"))

import numpy as np  # noqa: E402

from src.data.split import CLASS_ORDER  # noqa: E402
from src.drift.calibration import bootstrap_calibrate  # noqa: E402
from src.drift.detector import (  # noqa: E402
    DetectorError,
    Window,
    build_clean_windows,
    build_perturbed_windows,
    numeric_key,
)
from src.drift.image_statistics import ReferenceModel, compute_image_stats, load_gray  # noqa: E402
from src.drift.metrics import binary_metrics, binary_metrics_from_predictions, macro_average, mean_std  # noqa: E402
from src.features.cache import FeatureCache  # noqa: E402
from src.features.extractor import FeatureExtractor, weights_cached  # noqa: E402

SCHEMA = "microdefect.drift_monitoring/1"
DEFAULT_PERT = PROJECT_ROOT / "experiments" / "data" / "perturbations"
DEFAULT_SPLITS = PROJECT_ROOT / "experiments" / "results" / "splits"
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "features"
DEFAULT_OUT = PROJECT_ROOT / "experiments" / "results" / "drift_monitoring"
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "results" / "neu_cls_manifest.csv"
SEEDS = (42, 123, 2024, 7, 999)
CONDITIONS = (
    ("brightness", "low"),
    ("brightness", "high"),
    ("contrast", "low"),
    ("contrast", "high"),
    ("gaussian_noise", "low"),
    ("gaussian_noise", "high"),
    ("gaussian_blur", "low"),
    ("gaussian_blur", "high"),
    ("translation", "low"),
    ("translation", "high"),
)
CHANNELS = ("A", "B", "fusion")
REFERENCE_POOL_SIZE = 25


class RunError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fields or list(rows[0].keys())
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 阶段 2：输入审计
# ---------------------------------------------------------------------------


def build_input_audit(pert_root: Path, seeds: tuple[int, ...]) -> dict:
    seeds_info = []
    total_rows = 0
    all_md5 = set()
    global_missing = 0
    for seed in seeds:
        seed_dir = pert_root / f"seed_{seed}"
        manifest = seed_dir / "perturbation_manifest.csv"
        if not manifest.is_file():
            raise RunError(f"缺少 P4-1 清单：{manifest}")
        rows = read_csv(manifest)
        total_rows += len(rows)

        per_class = defaultdict(lambda: {"validation": 0, "test": 0})
        counts = defaultdict(int)
        pairs = defaultdict(set)
        missing = 0
        val_md5, test_md5 = set(), set()
        for row in rows:
            if not (seed_dir / row["output_rel_path"]).is_file():
                missing += 1
            all_md5.add(row["output_md5"])
            counts[f"{row['perturbation_type']}_{row['perturbation_level']}"] += 1
            pairs[(row["split"], row["source_file"])].add(
                (row["perturbation_type"], row["perturbation_level"])
            )
            if row["perturbation_type"] == "original":
                per_class[row["source_class"]][row["split"]] += 1
                (val_md5 if row["split"] == "validation" else test_md5).add(row["source_md5"])
        global_missing += missing
        bad_pairs = sum(1 for value in pairs.values() if len(value) != 11)
        seeds_info.append(
            {
                "seed": seed,
                "manifest": str(manifest),
                "rows": len(rows),
                "originals": sum(1 for r in rows if r["perturbation_type"] == "original"),
                "perturbations": sum(1 for r in rows if r["perturbation_type"] != "original"),
                "per_class_originals": {cls: per_class[cls] for cls in CLASS_ORDER},
                "condition_counts": dict(sorted(counts.items())),
                "missing_files": missing,
                "incomplete_pairs": bad_pairs,
                "validation_test_md5_overlap": len(val_md5 & test_md5),
                "unique_output_md5": len({r["output_md5"] for r in rows}),
            }
        )
        if missing or bad_pairs:
            raise RunError(f"seed={seed} 数据不完整：missing={missing}, bad_pairs={bad_pairs}")

    global_report_path = pert_root / "global_report.json"
    global_report = (
        json.loads(global_report_path.read_text(encoding="utf-8"))
        if global_report_path.is_file()
        else {}
    )
    verification = global_report.get("verification", {})
    p41_ok = bool(verification.get("all_ok")) and global_report.get("status") == "complete"

    audit = {
        "schema": SCHEMA + "/input_audit",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(pert_root),
        "manifest_fields": list(read_csv(pert_root / f"seed_{seeds[0]}" / "perturbation_manifest.csv")[0].keys()),
        "seed_count": len(seeds),
        "seeds": seeds_info,
        "totals": {
            "rows": total_rows,
            "unique_output_md5": len(all_md5),
            "missing_files": global_missing,
        },
        "perturbation_types": sorted({c[0] for c in CONDITIONS}),
        "strengths": sorted({c[1] for c in CONDITIONS}),
        "pairing": {
            "expected_rows_per_source": 11,
            "source_id_field": "source_file",
            "output_id_field": "output_rel_path",
            "source_md5_field": "source_md5",
            "complete": all(item["incomplete_pairs"] == 0 for item in seeds_info),
        },
        "cross_split": {
            "validation_test_overlap_by_seed": {
                str(item["seed"]): item["validation_test_md5_overlap"] for item in seeds_info
            },
            "pass": all(item["validation_test_md5_overlap"] == 0 for item in seeds_info),
        },
        "p41_global_report": {
            "path": str(global_report_path),
            "status": global_report.get("status"),
            "verification_all_ok": verification.get("all_ok"),
            "passed": p41_ok,
        },
        "pass": p41_ok and global_missing == 0,
    }
    return audit


# ---------------------------------------------------------------------------
# 特征与统计准备
# ---------------------------------------------------------------------------


def collect_unique_outputs(pert_root: Path, seeds: tuple[int, ...]) -> dict[str, dict]:
    unique: dict[str, dict] = {}
    for seed in seeds:
        for row in read_csv(pert_root / f"seed_{seed}" / "perturbation_manifest.csv"):
            unique.setdefault(row["output_md5"], {"path": str(pert_root / f"seed_{seed}" / row["output_rel_path"])})
    return unique


def prepare_features(
    unique: dict[str, dict],
    cache: FeatureCache,
    *,
    device: str,
    batch_size: int,
    log=print,
) -> tuple[dict, dict, dict]:
    """返回 (stats_by_md5, features_by_md5, timing)。"""
    stats_by_md5: dict[str, dict] = {}
    features_by_md5: dict[str, np.ndarray] = {}

    t0 = time.perf_counter()
    for index, (md5, item) in enumerate(unique.items(), start=1):
        stats_by_md5[md5] = compute_image_stats(load_gray(item["path"]))
        if index % 5000 == 0:
            log(f"  [stats] {index}/{len(unique)}")
    stats_seconds = time.perf_counter() - t0

    hits_before = cache.stats["hits"]
    misses_before = cache.stats["misses"]
    missing: list[tuple[str, str]] = []
    for md5, item in unique.items():
        vector = cache.get(cache.key_for_md5(md5))
        if vector is None:
            missing.append((md5, item["path"]))
        else:
            features_by_md5[md5] = vector
    cache_read_seconds = time.perf_counter() - t0 - stats_seconds

    extract_seconds = 0.0
    extractor_info = None
    if missing:
        if not weights_cached():
            raise RunError("预训练权重缺失，无法提取新特征")
        extractor = FeatureExtractor(device, batch_size=batch_size)
        extractor_info = extractor.describe()
        extractor.extract_paths([missing[0][1]], batch_size=1)  # 预热（不计入）
        t1 = time.perf_counter()
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            vectors = extractor.extract_paths([path for _, path in chunk], batch_size=batch_size)
            for (md5, _), vector in zip(chunk, vectors):
                features_by_md5[md5] = vector
                cache.put(cache.key_for_md5(md5), vector)
        extract_seconds = time.perf_counter() - t1

    timing = {
        "unique_images": len(unique),
        "stats_seconds": round(stats_seconds, 3),
        "cache_read_seconds": round(cache_read_seconds, 3),
        "feature_extract_seconds": round(extract_seconds, 3),
        "cache_hits": int(cache.stats["hits"] - hits_before),
        "cache_misses": int(missing.__len__()),
        "cache_stats_total": dict(cache.stats),
        "extractor": extractor_info,
    }
    log(
        f"  [features] 唯一图像 {len(unique)}，命中 {timing['cache_hits']}，"
        f"新提取 {len(missing)}，提取 {extract_seconds:.1f}s"
    )
    return stats_by_md5, features_by_md5, timing


# ---------------------------------------------------------------------------
# 每个 seed 的处理
# ---------------------------------------------------------------------------


def select_reference_pool(rows: list[dict], size: int, seed: int, class_name: str) -> list[dict]:
    ordered = sorted(rows, key=lambda r: numeric_key(r["source_file"]))
    if len(ordered) < size:
        raise RunError(
            f"seed={seed} {class_name} validation 原图 {len(ordered)} 张，"
            f"不足参考池大小 {size}"
        )
    material = f"p4.2|{seed}|{class_name}".encode("utf-8")
    derived = int(hashlib.sha256(material).hexdigest()[:16], 16)
    rng = np.random.default_rng(derived)
    index = np.sort(rng.choice(len(ordered), size=size, replace=False))
    return [ordered[i] for i in index]


def score_windows(
    windows: list[Window],
    *,
    rows_by_file: dict[str, dict],
    stats_by_md5: dict,
    features_by_md5: dict,
    model: ReferenceModel,
    reference_features: np.ndarray,
    gamma: float,
    thresholds: tuple[float, float, float],
) -> list[dict]:
    from src.drift.detector import score_window

    output = []
    for window in windows:
        result = score_window(
            window,
            rows_by_file=rows_by_file,
            stats_by_md5=stats_by_md5,
            features_by_md5=features_by_md5,
            model=model,
            reference_features=reference_features,
            gamma=gamma,
            threshold_a=thresholds[0],
            threshold_b=thresholds[1],
            threshold_fusion=thresholds[2],
        )
        row = window.to_row()
        row.update(result.to_row())
        output.append(row)
    return output


def evaluate_group(rows: list[dict], *, pred_key: str, score_key: str, threshold: float | None = None) -> dict:
    """用窗口自带的二值判定（该校准阈值下的结果）计算指标。"""
    from src.drift.metrics import binary_metrics_from_predictions

    scores = [float(row[score_key]) for row in rows]
    labels = [int(row["is_drift"]) for row in rows]
    predictions = [int(row[pred_key]) for row in rows]
    return binary_metrics_from_predictions(scores, labels, predictions, threshold=threshold)


def build_seed(
    seed: int,
    *,
    config: dict,
    pert_root: Path,
    stats_by_md5: dict,
    features_by_md5: dict,
    out_dir: Path,
    classes: tuple[str, ...],
    conditions: tuple[tuple[str, str], ...],
    window_limit: int,
    log=print,
) -> dict:
    started = time.perf_counter()
    rows = read_csv(pert_root / f"seed_{seed}" / "perturbation_manifest.csv")

    by_split_class: dict[tuple[str, str], list[dict]] = defaultdict(list)
    condition_rows: dict[tuple[str, str, str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["split"], row["source_class"])
        if row["perturbation_type"] == "original":
            by_split_class[key].append(row)
        else:
            condition_rows[(row["split"], row["source_class"], row["perturbation_type"], row["perturbation_level"])][
                row["source_file"]
            ] = row

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    reference_rows: list[dict] = []
    calibration_rows: list[dict] = []
    validation_rows: list[dict] = []
    test_rows: list[dict] = []
    threshold_rows: list[dict] = []
    null_rows: list[dict] = []
    reference_pool_info: dict[str, dict] = {}
    timings = {"calibration_seconds": 0.0, "scoring_seconds": 0.0}

    for class_name in classes:
        validation_originals = by_split_class[("validation", class_name)]
        test_originals = by_split_class[("test", class_name)]
        pool = select_reference_pool(validation_originals, config["reference_pool_size"], seed, class_name)
        pool_ids = {row["source_file"] for row in pool}
        remaining_validation = [row for row in validation_originals if row["source_file"] not in pool_ids]

        reference_stats = [stats_by_md5[row["output_md5"]] for row in pool]
        reference_features = np.stack([features_by_md5[row["output_md5"]] for row in pool], axis=0)
        model = ReferenceModel.fit(reference_stats, class_name)

        for row in pool:
            reference_rows.append(
                {
                    "seed": seed,
                    "class_name": class_name,
                    "split": "validation",
                    "source_file": row["source_file"],
                    "source_md5": row["source_md5"],
                    "output_md5": row["output_md5"],
                    "source_rel_path": row["source_rel_path"],
                }
            )

        # ---- 校准（只用 validation 参考池）----
        t0 = time.perf_counter()
        calibration = bootstrap_calibrate(
            reference_stats,
            reference_features,
            seed=seed,
            class_name=class_name,
            window_size=config["window_size"],
            iterations=config["bootstrap_iterations"],
            random_seed=config["random_seed"] + int(hashlib.sha256(f"{seed}{class_name}".encode()).hexdigest()[:6], 16),
        )
        timings["calibration_seconds"] += time.perf_counter() - t0
        thresholds = (calibration.threshold_A, calibration.threshold_B, calibration.threshold_fusion)
        threshold_rows.append(calibration.summary())
        for index in range(calibration.iterations):
            null_rows.append(
                {
                    "seed": seed,
                    "class_name": class_name,
                    "iteration": index,
                    "null_score_A": float(calibration.null_scores_A[index]),
                    "null_score_B": float(calibration.null_scores_B[index]),
                    "null_score_fusion": float(calibration.null_scores_fusion[index]),
                }
            )
        reference_pool_info[class_name] = {
            "pool_size": len(pool),
            "remaining_validation_originals": len(remaining_validation),
            "gamma": calibration.gamma,
            "reference_model": model.to_dict(),
        }

        # 仅用原图行建立 source_file → 原图输出 的映射（避免被扰动行覆盖）
        originals_by_file = {
            row["source_file"]: row
            for row in rows
            if row["source_class"] == class_name and row["perturbation_type"] == "original"
        }

        # ---- validation 窗口（剩余 validation 原图，非重叠）----
        t0 = time.perf_counter()
        val_windows, val_tail = build_clean_windows(
            remaining_validation, seed=seed, class_name=class_name, split="validation", size=config["window_size"]
        )
        if window_limit:
            val_windows = val_windows[:window_limit]
        validation_rows.extend(
            score_windows(
                val_windows,
                rows_by_file=originals_by_file,
                stats_by_md5=stats_by_md5,
                features_by_md5=features_by_md5,
                model=model,
                reference_features=reference_features,
                gamma=calibration.gamma,
                thresholds=thresholds,
            )
        )

        # ---- test 窗口 ----
        clean_windows, tail = build_clean_windows(
            test_originals, seed=seed, class_name=class_name, split="test", size=config["window_size"]
        )
        if window_limit:
            clean_windows = clean_windows[:window_limit]
        clean_rows = score_windows(
            clean_windows,
            rows_by_file=originals_by_file,
            stats_by_md5=stats_by_md5,
            features_by_md5=features_by_md5,
            model=model,
            reference_features=reference_features,
            gamma=calibration.gamma,
            thresholds=thresholds,
        )
        test_rows.extend(clean_rows)

        selected = list(conditions)
        mapping = {
            condition: condition_rows[("test", class_name, condition[0], condition[1])]
            for condition in selected
        }
        # 每个扰动条件使用自己的 source_file → 扰动输出 映射
        for condition, cond_map in mapping.items():
            perturbed_windows = build_perturbed_windows(clean_windows, {condition: cond_map})
            perturbed_rows = score_windows(
                perturbed_windows,
                rows_by_file=cond_map,
                stats_by_md5=stats_by_md5,
                features_by_md5=features_by_md5,
                model=model,
                reference_features=reference_features,
                gamma=calibration.gamma,
                thresholds=thresholds,
            )
            test_rows.extend(perturbed_rows)
        timings["scoring_seconds"] += time.perf_counter() - t0

        # 尾部样本记录（不静默丢弃）
        for row in val_tail:
            calibration_rows.append({"seed": seed, "class_name": class_name, "split": "validation", "unused_tail": row["source_file"]})
        for row in tail:
            calibration_rows.append({"seed": seed, "class_name": class_name, "split": "test", "unused_tail": row["source_file"]})

    write_csv(out_dir / f"seed_{seed}" / "reference_pool.csv", reference_rows)
    write_csv(out_dir / f"seed_{seed}" / "calibration_thresholds.csv", threshold_rows)
    write_csv(out_dir / f"seed_{seed}" / "bootstrap_null_scores.csv", null_rows)
    write_csv(out_dir / f"seed_{seed}" / "validation_window_scores.csv", validation_rows)
    write_csv(out_dir / f"seed_{seed}" / "test_window_scores.csv", test_rows)
    write_json(
        out_dir / f"seed_{seed}" / "calibration_thresholds.json",
        {"seed": seed, "config": config, "classes": reference_pool_info, "thresholds": threshold_rows},
    )
    write_csv(out_dir / f"seed_{seed}" / "unused_tail_samples.csv", calibration_rows)

    metrics = compute_metrics(seed, test_rows, threshold_rows)
    write_json(out_dir / f"seed_{seed}" / "metrics_overall.json", metrics)
    write_csv(out_dir / f"seed_{seed}" / "metrics_by_perturbation.csv", metrics["rows_by_perturbation"])
    write_csv(out_dir / f"seed_{seed}" / "metrics_by_class.csv", metrics["rows_by_class"])
    write_csv(out_dir / f"seed_{seed}" / "metrics_by_strength.csv", metrics["rows_by_strength"])
    write_csv(out_dir / f"seed_{seed}" / "paired_window_scores.csv", metrics["paired_rows"])

    elapsed = time.perf_counter() - started
    log(f"[ok] seed={seed} 完成，用时 {elapsed:.1f}s，test 窗口 {len(test_rows)}")
    return {
        "seed": seed,
        "test_windows": len(test_rows),
        "validation_windows": len(validation_rows),
        "phase_seconds": round(elapsed, 3),
        "calibration_seconds": round(timings["calibration_seconds"], 3),
        "scoring_seconds": round(timings["scoring_seconds"], 3),
        "metrics": metrics["overall"],
    }


def _split_by(rows: list[dict], key_fn) -> dict:
    grouped: dict = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return grouped


def compute_metrics(seed: int, test_rows: list[dict], threshold_rows: list[dict]) -> dict:
    thresholds = {(t["class_name"]): t for t in threshold_rows}

    def threshold_for(class_name: str, channel: str) -> float | None:
        row = thresholds.get(class_name)
        if row is None:
            return None
        key = {"A": "threshold_A", "B": "threshold_B", "fusion": "threshold_fusion"}[channel]
        return float(row[key])

    rows_by_perturbation = []
    rows_by_class = []
    rows_by_strength = []
    overall_rows = []

    for class_name, class_rows in _split_by(test_rows, lambda r: r["class_name"]).items():
        clean = [r for r in class_rows if r["is_drift"] == 0]
        for condition, condition_rows in _split_by(
            [r for r in class_rows if r["is_drift"] == 1],
            lambda r: (r["perturbation_type"], r["perturbation_level"]),
        ).items():
            combined = clean + condition_rows
            for channel in CHANNELS:
                score_key = f"score_{channel}" if channel != "fusion" else "score_fusion"
                result = evaluate_group(
                    combined,
                    pred_key="is_drift_" + channel,
                    score_key=score_key,
                    threshold=threshold_for(class_name, channel),
                )
                rows_by_perturbation.append(
                    {
                        "seed": seed,
                        "class_name": class_name,
                        "perturbation_type": condition[0],
                        "perturbation_level": condition[1],
                        "channel": channel,
                        "threshold": result["threshold"],
                        "positives": result["positives"],
                        "negatives": result["negatives"],
                        "precision": result["precision"],
                        "recall": result["recall"],
                        "f1": result["f1"],
                        "fpr": result["fpr"],
                        "fnr": result["fnr"],
                        "detection_rate": result["detection_rate"],
                        "auroc": result["auroc"],
                        "mean_score_positive": result["mean_score_positive"],
                        "mean_score_negative": result["mean_score_negative"],
                    }
                )
        # 类别层：把所有扰动与干净窗口合并（该类的整体检测能力）
        for channel in CHANNELS:
            score_key = f"score_{channel}" if channel != "fusion" else "score_fusion"
            result = evaluate_group(
                class_rows,
                pred_key="is_drift_" + channel,
                score_key=score_key,
                threshold=threshold_for(class_name, channel),
            )
            rows_by_class.append(
                {
                    "seed": seed,
                    "class_name": class_name,
                    "channel": channel,
                    "threshold": result["threshold"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "fpr": result["fpr"],
                    "fnr": result["fnr"],
                    "auroc": result["auroc"],
                }
            )

    # 强度层与 overall（跨类别合并）
    for level in ("low", "high"):
        level_rows = [r for r in test_rows if r["is_drift"] == 1 and r["perturbation_level"] == level]
        clean_rows = [r for r in test_rows if r["is_drift"] == 0]
        combined = clean_rows + level_rows
        for channel in CHANNELS:
            score_key = f"score_{channel}" if channel != "fusion" else "score_fusion"
            result = evaluate_group(combined, pred_key="is_drift_" + channel, score_key=score_key)
            rows_by_strength.append(
                {
                    "seed": seed,
                    "perturbation_level": level,
                    "channel": channel,
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1": result["f1"],
                    "fpr": result["fpr"],
                    "auroc": result["auroc"],
                }
            )

    for channel in CHANNELS:
        score_key = f"score_{channel}" if channel != "fusion" else "score_fusion"
        result = evaluate_group(test_rows, pred_key="is_drift_" + channel, score_key=score_key)
        overall_rows.append(
            {
                "seed": seed,
                "channel": channel,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "fpr": result["fpr"],
                "fnr": result["fnr"],
                "detection_rate": result["detection_rate"],
                "auroc": result["auroc"],
                "mean_score_positive": result["mean_score_positive"],
                "mean_score_negative": result["mean_score_negative"],
            }
        )

    # 配对差值
    paired_rows = []
    clean_index = {
        (r["class_name"], r["window_id"]): r for r in test_rows if r["is_drift"] == 0
    }
    for row in test_rows:
        if row["is_drift"] != 1:
            continue
        base_id = row["window_id"].split("__")[0]
        clean = clean_index.get((row["class_name"], base_id))
        if clean is None:
            continue
        paired_rows.append(
            {
                "seed": seed,
                "class_name": row["class_name"],
                "window_id": base_id,
                "perturbation_type": row["perturbation_type"],
                "perturbation_level": row["perturbation_level"],
                "delta_A": round(row["score_A"] - clean["score_A"], 8),
                "delta_B": round(row["score_B"] - clean["score_B"], 10),
                "delta_fusion": round(row["score_fusion"] - clean["score_fusion"], 8),
            }
        )

    return {
        "seed": seed,
        "overall": {row["channel"]: row for row in overall_rows},
        "rows_by_perturbation": rows_by_perturbation,
        "rows_by_class": rows_by_class,
        "rows_by_strength": rows_by_strength,
        "paired_rows": paired_rows,
        "thresholds": {name: {"A": t["threshold_A"], "B": t["threshold_B"], "fusion": t["threshold_fusion"]} for name, t in thresholds.items()},
    }


# ---------------------------------------------------------------------------
# 汇总与图表
# ---------------------------------------------------------------------------


def aggregate_across_seeds(seed_overalls: list[dict]) -> dict:
    result = {}
    for channel in CHANNELS:
        values = {key: [] for key in ("precision", "recall", "f1", "fpr", "fnr", "detection_rate", "auroc")}
        for item in seed_overalls:
            record = item["overall"].get(channel, {})
            for key in values:
                values[key].append(record.get(key))
        result[channel] = {key: mean_std(value) for key, value in values.items()}
    return result


def make_plots(out_dir: Path, seed_metrics: list[dict], summary_rows: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    seed_names = [str(item["seed"]) for item in seed_metrics]

    # 1. 三通道整体指标
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, metric in zip(axes, ("precision", "recall", "f1", "auroc")):
        for channel in CHANNELS:
            values = [item["overall"][channel][metric] for item in seed_metrics]
            axis.plot(seed_names, values, marker="o", label=channel)
        axis.set_title(metric.upper())
        axis.set_xlabel("seed")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("Drift detection by channel (per seed, NEU-CLS classes)")
    fig.tight_layout()
    path = plot_dir / "channel_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    created.append(str(path))

    # 2. 各扰动类型 F1
    per_type = defaultdict(lambda: defaultdict(list))
    for row in summary_rows["by_perturbation"]:
        per_type[row["perturbation_type"]][row["channel"]].append(float(row["f1"]))
    types = sorted(per_type)
    fig, axis = plt.subplots(figsize=(10, 4.5))
    width = 0.25
    for offset, channel in enumerate(CHANNELS):
        values = [float(np.mean(per_type[t][channel])) for t in types]
        axis.bar(np.arange(len(types)) + offset * width, values, width, label=channel)
    axis.set_xticks(np.arange(len(types)) + width)
    axis.set_xticklabels(types, rotation=20)
    axis.set_ylabel("F1")
    axis.set_ylim(0, 1.05)
    axis.set_title("Detection F1 by perturbation type (all seeds pooled)")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = plot_dir / "f1_by_perturbation_type.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    created.append(str(path))

    # 3. low/high
    per_level = defaultdict(lambda: defaultdict(list))
    for row in summary_rows["by_strength"]:
        per_level[row["perturbation_level"]][row["channel"]].append(float(row["f1"]))
    fig, axis = plt.subplots(figsize=(6, 4))
    levels = ["low", "high"]
    for offset, channel in enumerate(CHANNELS):
        values = [float(np.mean(per_level[lv][channel])) for lv in levels]
        axis.bar(np.arange(len(levels)) + offset * 0.25, values, 0.25, label=channel)
    axis.set_xticks(np.arange(len(levels)) + 0.25)
    axis.set_xticklabels(levels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("F1")
    axis.set_title("Detection F1 by perturbation strength")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = plot_dir / "f1_by_strength.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    created.append(str(path))

    # 4. 类别 F1
    per_class = defaultdict(lambda: defaultdict(list))
    for row in summary_rows["by_class"]:
        per_class[row["class_name"]][row["channel"]].append(float(row["f1"]))
    classes = [c for c in CLASS_ORDER if c in per_class]
    fig, axis = plt.subplots(figsize=(8, 4))
    for offset, channel in enumerate(CHANNELS):
        values = [float(np.mean(per_class[c][channel])) for c in classes]
        axis.bar(np.arange(len(classes)) + offset * 0.25, values, 0.25, label=channel)
    axis.set_xticks(np.arange(len(classes)) + 0.25)
    axis.set_xticklabels(classes)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("F1")
    axis.set_title("Detection F1 by defect class (NEU-CLS categories, not products)")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = plot_dir / "f1_by_class.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    created.append(str(path))

    # 5. 分数分布（fusion）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, channel in zip(axes, CHANNELS):
        key = f"score_{channel}" if channel != "fusion" else "score_fusion"
        clean, drift = [], []
        for item in seed_metrics:
            for row in item["windows"]:
                (drift if row["is_drift"] else clean).append(float(row[key]))
        axis.hist(clean, bins=40, alpha=0.6, label=f"clean (n={len(clean)})", density=True)
        axis.hist(drift, bins=40, alpha=0.6, label=f"perturbed (n={len(drift)})", density=True)
        axis.set_title(f"channel {channel}: {key}")
        axis.set_xlabel("score")
        axis.set_ylabel("density")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    path = plot_dir / "score_distributions.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    created.append(str(path))

    # 6. 阈值与测试分数（每类，fusion）
    fig, axis = plt.subplots(figsize=(10, 4.5))
    positions = []
    labels = []
    for index, cls in enumerate(classes):
        thresholds = [item["thresholds"][cls]["fusion"] for item in seed_metrics if cls in item["thresholds"]]
        scores = [float(row["score_fusion"]) for item in seed_metrics for row in item["windows"] if row["class_name"] == cls]
        axis.boxplot([scores], positions=[index * 2], widths=0.6, showfliers=False)
        axis.scatter([index * 2] * len(thresholds), thresholds, marker="_", s=200, color="red", zorder=3)
        positions.append(index * 2)
        labels.append(cls)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("fusion score")
    axis.set_title("Test fusion scores by class (box) and bootstrap 95% thresholds (red)")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = plot_dir / "thresholds_and_scores.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    created.append(str(path))

    # 7. 强度趋势（每类扰动 low→high 的平均分数）
    trend = defaultdict(lambda: defaultdict(list))
    for row in summary_rows["by_perturbation"]:
        if row["channel"] != "fusion":
            continue
        trend[row["perturbation_type"]][row["perturbation_level"]].append(float(row["mean_score_positive"]))
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for ptype in sorted(trend):
        low = float(np.mean(trend[ptype]["low"])) if trend[ptype]["low"] else np.nan
        high = float(np.mean(trend[ptype]["high"])) if trend[ptype]["high"] else np.nan
        axis.plot(["low", "high"], [low, high], marker="o", label=ptype)
    axis.set_ylabel("mean fusion score (perturbed windows)")
    axis.set_title("Drift score vs perturbation strength")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    path = plot_dir / "score_vs_strength.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    created.append(str(path))

    return created


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="微缺智学 P4-2 漂移监测实验")
    parser.add_argument("--pert-root", default=str(DEFAULT_PERT))
    parser.add_argument("--splits-root", default=str(DEFAULT_SPLITS))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--classes", nargs="+", default=list(CLASS_ORDER))
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20261110)
    parser.add_argument("--reference-pool-size", type=int, default=REFERENCE_POOL_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--limit-windows", type=int, default=0, help="每类每 split 只用前 N 个窗口（smoke）")
    parser.add_argument("--smoke", action="store_true", help="单 seed 单类别 brightness_low 冒烟")
    args = parser.parse_args()

    pert_root = Path(args.pert_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = tuple(args.seeds)
    classes = tuple(args.classes)
    conditions = CONDITIONS
    if args.smoke:
        seeds = (42,)
        classes = ("Cr",)
        conditions = (("brightness", "low"),)
        args.limit_windows = args.limit_windows or 3
        args.bootstrap_iterations = min(args.bootstrap_iterations, 100)

    config = {
        "schema": SCHEMA,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seeds": list(seeds),
        "classes": list(classes),
        "conditions": [list(c) for c in conditions],
        "window_size": args.window_size,
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_seed": args.random_seed,
        "reference_pool_size": args.reference_pool_size,
        "quantile": 0.95,
        "device": args.device,
        "batch_size": args.batch_size,
        "smoke": args.smoke,
        "limit_windows": args.limit_windows,
        "pert_root": str(pert_root),
    }
    config["signature"] = hashlib.sha256(
        json.dumps({k: config[k] for k in ("seeds", "classes", "conditions", "window_size",
                                           "bootstrap_iterations", "random_seed",
                                           "reference_pool_size", "limit_windows")},
                   sort_keys=True).encode()
    ).hexdigest()[:16]

    log_path = out_dir / "run_log.txt"
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    log("=" * 78)
    log("微缺智学 P4-2 双通道漂移监测实验")
    log("=" * 78)
    log(f"seeds={list(seeds)} classes={list(classes)} conditions={len(conditions)}")
    log(f"window={args.window_size} bootstrap={args.bootstrap_iterations} seed={args.random_seed}")

    if args.verify_only:
        failures = []
        for seed in seeds:
            metrics_path = out_dir / f"seed_{seed}" / "metrics_overall.json"
            if not metrics_path.is_file():
                failures.append(f"seed_{seed}: 缺少 metrics_overall.json")
        log(f"[verify] 缺失 {len(failures)} 项")
        for item in failures:
            log("  - " + item)
        return 0 if not failures else 1

    # 输入审计
    audit_path = out_dir / "input_audit.json"
    if audit_path.is_file() and args.resume:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        log("[resume] 复用已有 input_audit.json")
    else:
        audit = build_input_audit(pert_root, seeds)
        write_json(audit_path, audit)
        log(f"[audit] 通过={audit['pass']}，唯一内容 {audit['totals']['unique_output_md5']}")
        if not audit["pass"]:
            log("[FAIL] 输入审计未通过，停止")
            return 2

    write_json(out_dir / "config.json", config)

    started = time.perf_counter()
    unique = collect_unique_outputs(pert_root, seeds)
    log(f"[data] 唯一输出图像 {len(unique)}")

    cache = FeatureCache(Path(args.cache_dir))
    stats_by_md5, features_by_md5, feature_timing = prepare_features(
        unique, cache, device=args.device, batch_size=args.batch_size, log=log
    )

    seed_metrics: list[dict] = []
    seed_reports: list[dict] = []
    for seed in seeds:
        metrics_path = out_dir / f"seed_{seed}" / "metrics_overall.json"
        if args.resume and metrics_path.is_file():
            log(f"[resume] seed={seed} 已完成，跳过")
            stored = json.loads(metrics_path.read_text(encoding="utf-8"))
            stored["windows"] = read_csv(out_dir / f"seed_{seed}" / "test_window_scores.csv")
            stored["test_windows"] = len(stored["windows"])
            seed_metrics.append(stored)
            seed_reports.append({"seed": seed, "test_windows": stored.get("test_windows", 0), "skipped": True})
            continue
        report = build_seed(
            seed,
            config=config,
            pert_root=pert_root,
            stats_by_md5=stats_by_md5,
            features_by_md5=features_by_md5,
            out_dir=out_dir,
            classes=classes,
            conditions=conditions,
            window_limit=args.limit_windows,
            log=log,
        )
        seed_reports.append(report)
        stored = json.loads((out_dir / f"seed_{seed}" / "metrics_overall.json").read_text(encoding="utf-8"))
        stored["windows"] = read_csv(out_dir / f"seed_{seed}" / "test_window_scores.csv")
        stored["thresholds"] = {row["class_name"]: {"A": float(row["threshold_A"]),
                                                    "B": float(row["threshold_B"]),
                                                    "fusion": float(row["threshold_fusion"])}
                                for row in json.loads((out_dir / f"seed_{seed}" /
                                                       "calibration_thresholds.json").read_text(encoding="utf-8"))["thresholds"]}
        stored["test_windows"] = len(stored["windows"])
        seed_metrics.append(stored)

    # 汇总
    summary_rows = {
        "by_perturbation": [],
        "by_class": [],
        "by_strength": [],
        "paired": [],
    }
    for seed in seeds:
        summary_rows["by_perturbation"].extend(read_csv(out_dir / f"seed_{seed}" / "metrics_by_perturbation.csv"))
        summary_rows["by_class"].extend(read_csv(out_dir / f"seed_{seed}" / "metrics_by_class.csv"))
        summary_rows["by_strength"].extend(read_csv(out_dir / f"seed_{seed}" / "metrics_by_strength.csv"))
        summary_rows["paired"].extend(read_csv(out_dir / f"seed_{seed}" / "paired_window_scores.csv"))

    for key in ("by_perturbation", "by_class", "by_strength", "paired"):
        write_csv(out_dir / f"summary_{key}.csv", summary_rows[key])

    aggregate = aggregate_across_seeds(seed_metrics)
    overall_summary = []
    for seed_item in seed_metrics:
        for channel in CHANNELS:
            record = seed_item["overall"][channel]
            overall_summary.append(
                {
                    "seed": seed_item["seed"],
                    "channel": channel,
                    "precision": record["precision"],
                    "recall": record["recall"],
                    "f1": record["f1"],
                    "fpr": record["fpr"],
                    "fnr": record["fnr"],
                    "detection_rate": record["detection_rate"],
                    "auroc": record["auroc"],
                }
            )
    write_csv(out_dir / "summary.csv", overall_summary)

    # 分组宏平均
    def group_macro(rows, group_key, channel) -> list[dict]:
        grouped = defaultdict(list)
        for row in rows:
            if row["channel"] != channel:
                continue
            grouped[row[group_key]].append(row)
        output = []
        for name, items in sorted(grouped.items()):
            output.append(
                {
                    "group": group_key,
                    "value": name,
                    "channel": channel,
                    "n": len(items),
                    "f1": mean_std([float(i["f1"]) for i in items]),
                    "auroc": mean_std([float(i["auroc"]) for i in items if i.get("auroc") not in (None, "", "None")]),
                }
            )
        return output

    macro = []
    for channel in CHANNELS:
        macro += group_macro(summary_rows["by_perturbation"], "perturbation_type", channel)
        macro += group_macro(summary_rows["by_perturbation"], "perturbation_level", channel)
        macro += group_macro(summary_rows["by_class"], "class_name", channel)
    write_csv(out_dir / "macro_by_group.csv", macro)

    summary_json = {
        "schema": SCHEMA,
        "config": config,
        "feature_timing": feature_timing,
        "aggregate_across_seeds": aggregate,
        "seeds": [
            {
                "seed": item["seed"],
                "test_windows": item.get("test_windows"),
                "overall": item["overall"],
            }
            for item in seed_metrics
        ],
        "seed_reports": seed_reports,
        "macro_groups": macro,
        "total_seconds": round(time.perf_counter() - started, 2),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(out_dir / "summary.json", summary_json)

    plots = make_plots(out_dir, seed_metrics, summary_rows)
    log(f"[plots] 生成 {len(plots)} 张图")

    log("-" * 78)
    def fmt_stat(value) -> str:
        return "NA" if value is None else f"{value:.4f}"

    for channel in CHANNELS:
        stats = aggregate[channel]
        log(
            f"channel {channel}: F1={fmt_stat(stats['f1']['mean'])}±{fmt_stat(stats['f1']['std'])}  "
            f"AUROC={fmt_stat(stats['auroc']['mean'])}±{fmt_stat(stats['auroc']['std'])}  "
            f"P={fmt_stat(stats['precision']['mean'])} R={fmt_stat(stats['recall']['mean'])}"
        )
    log(f"总耗时 {summary_json['total_seconds']}s")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
