#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 少样本原型注册与识别实验（NEU-CLS，5 seed × 3 shot）。

流程
----
1. 读取 P1-2 的划分 CSV（support_{1,3,5}shot / validation / test）。
2. 按完整 MD5 去重，从 P2 特征缓存取特征；缺失的才用冻结 ResNet18 提取一次。
3. 每个 (seed, shot)：用 L2 归一化 Support 特征求均值 → 再归一化，得到类别原型。
4. 用 Validation 的 95% 分位距离校准每类阈值。
5. 在 Test 上做最近原型分类（闭集指标）+ 阈值拒识（拒识指标）。
6. 输出预测明细、原型、阈值、指标、汇总。

不训练、不微调、不用 Test 调参、不重新划分数据。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.data.split import CLASS_ORDER  # noqa: E402
from src.experiments.metrics import (  # noqa: E402
    classification_metrics,
    summarize,
)
from src.features.cache import FeatureCache  # noqa: E402
from src.features.extractor import (  # noqa: E402
    EMBED_DIM,
    FEATURE_VERSION,
    PREPROCESS_VERSION,
    WEIGHTS_TAG,
    FeatureExtractor,
)
from src.memory.prototype import build_prototypes, predict_nearest  # noqa: E402
from src.memory.threshold import (  # noqa: E402
    DEFAULT_QUANTILE,
    calibrate_thresholds,
    rejection_metrics,
)

DEFAULT_SPLITS = PROJECT_ROOT / "experiments" / "results" / "splits"
DEFAULT_CACHE = PROJECT_ROOT / "cache" / "features"
DEFAULT_OUT = PROJECT_ROOT / "experiments" / "results" / "fewshot"
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "results" / "neu_cls_manifest.csv"
SEEDS = (42, 123, 2024, 7, 999)
SHOTS = (1, 3, 5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_seed_splits(splits_root: Path, seed: int) -> dict:
    seed_dir = splits_root / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"缺少划分目录：{seed_dir}")
    return {
        "support": {shot: read_csv(seed_dir / f"support_{shot}shot.csv") for shot in SHOTS},
        "validation": read_csv(seed_dir / "validation.csv"),
        "test": read_csv(seed_dir / "test.csv"),
    }


def gather_features(
    rows: list[dict],
    feature_by_md5: dict[str, np.ndarray],
) -> np.ndarray:
    vectors = [feature_by_md5[row["md5"]] for row in rows]
    if not vectors:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32)


def split_by_class(rows: list[dict], feature_by_md5: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[str]] = {cls: [] for cls in CLASS_ORDER}
    for row in rows:
        grouped[row["class_prefix"]].append(row["md5"])
    return {
        cls: np.stack([feature_by_md5[m] for m in md5s], axis=0).astype(np.float32)
        for cls, md5s in grouped.items()
        if md5s
    }


def prediction_rows(
    rows: list[dict],
    labels: list[str],
    distances: np.ndarray,
    min_distance: np.ndarray,
    accepted: np.ndarray,
    *,
    seed: int,
    shot: int,
) -> list[dict]:
    output = []
    for index, row in enumerate(rows):
        record = {
            "file_name": row["file_name"],
            "md5": row["md5"],
            "true_label": row["class_prefix"],
            "predicted_label": labels[index],
            "minimum_distance": round(float(min_distance[index]), 6),
            "accepted": bool(accepted[index]),
            "seed": seed,
            "shot": shot,
        }
        for cls_index, cls in enumerate(CLASS_ORDER):
            record[f"d_{cls}"] = round(float(distances[index, cls_index]), 6)
        output.append(record)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="微缺智学 P3 少样本原型实验")
    parser.add_argument("--splits-root", default=str(DEFAULT_SPLITS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--shots", default=",".join(str(s) for s in SHOTS))
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timing-samples", type=int, default=8)
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    shots = tuple(int(s) for s in args.shots.split(",") if s.strip())
    splits_root = Path(args.splits_root)
    cache_dir = Path(args.cache_dir)

    out_root = Path(args.out)
    if out_root.exists() and (out_root / "summary.json").is_file():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = out_root.with_name(f"{out_root.name}_{stamp}")
        print(f"[info] 目标目录已有结果，改用新目录：{out_root.name}")
    out_root.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    print("=" * 72)
    print("微缺智学 P3 少样本原型注册与识别实验")
    print("=" * 72)
    print(f"seeds={list(seeds)}  shots={list(shots)}  quantile={args.quantile}")

    # ---------- 读取划分 ----------
    splits = {seed: load_seed_splits(splits_root, seed) for seed in seeds}
    all_rows: list[dict] = []
    for seed in seeds:
        for shot in shots:
            all_rows.extend(splits[seed]["support"][shot])
        all_rows.extend(splits[seed]["validation"])
        all_rows.extend(splits[seed]["test"])

    unique: dict[str, dict] = {}
    for row in all_rows:
        unique.setdefault(row["md5"], row)
    print(f"涉及记录 {len(all_rows)} 条，按 MD5 去重后 {len(unique)} 种独立内容")

    # ---------- 特征：缓存优先 ----------
    cache = FeatureCache(cache_dir)
    feature_by_md5: dict[str, np.ndarray] = {}
    missing: list[dict] = []

    t_read = time.perf_counter()
    for md5, row in unique.items():
        vector = cache.get(cache.key_for_md5(md5))
        if vector is None:
            missing.append(row)
        else:
            feature_by_md5[md5] = vector
    cache_read_ms = (time.perf_counter() - t_read) * 1000
    stats_before_extract = dict(cache.stats)
    print(f"缓存命中 {len(feature_by_md5)} / 未命中 {len(missing)}，读取耗时 {cache_read_ms:.1f} ms")

    extract_ms = 0.0
    extractor_info = None
    if missing:
        extractor = FeatureExtractor(args.device, batch_size=args.batch_size)
        extractor_info = extractor.describe()
        print(f"提取设备 {extractor.device}，需要提取 {len(missing)} 张")
        t_extract = time.perf_counter()
        paths = [row["abs_path"] for row in missing]
        md5s = [row["md5"] for row in missing]
        for start in range(0, len(paths), args.batch_size):
            chunk_paths = paths[start : start + args.batch_size]
            chunk_md5s = md5s[start : start + args.batch_size]
            vectors = extractor.extract_paths(chunk_paths, batch_size=args.batch_size)
            for md5, vector in zip(chunk_md5s, vectors):
                feature_by_md5[md5] = vector
                cache.put(cache.key_for_md5(md5), vector)
        extract_ms = (time.perf_counter() - t_extract) * 1000
        print(f"提取完成，耗时 {extract_ms:.1f} ms")
    else:
        print("全部特征已命中缓存，无需提取")

    # ---------- 推理计时（预热后） ----------
    timing = {"device": None, "single_ms": None, "batch_ms": None, "per_image_ms": None}
    if args.timing_samples > 0:
        sample_rows = splits[seeds[0]]["test"][: args.timing_samples]
        sample_paths = [row["abs_path"] for row in sample_rows]
        timer = FeatureExtractor(args.device, batch_size=args.batch_size)
        timing["device"] = str(timer.device)
        timer.extract_paths(sample_paths[:2], batch_size=2)  # 预热
        t0 = time.perf_counter()
        for path in sample_paths:
            timer.extract_path(path)
        timing["single_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        timer.extract_paths(sample_paths)
        timing["batch_ms"] = (time.perf_counter() - t0) * 1000
        timing["per_image_ms"] = timing["batch_ms"] / len(sample_paths)
        timing["samples"] = len(sample_paths)
        if extractor_info is None:
            extractor_info = timer.describe()

    # ---------- 实验 ----------
    summary_rows: list[dict] = []
    per_group: dict[str, dict] = {}

    for seed in seeds:
        for shot in shots:
            support_rows = splits[seed]["support"][shot]
            val_rows = splits[seed]["validation"]
            test_rows = splits[seed]["test"]

            t_lookup = time.perf_counter()
            support_by_class = split_by_class(support_rows, feature_by_md5)
            val_by_class = split_by_class(val_rows, feature_by_md5)
            test_features = gather_features(test_rows, feature_by_md5)
            lookup_ms = (time.perf_counter() - t_lookup) * 1000

            # 注册：原型构建
            t_reg = time.perf_counter()
            prototypes = build_prototypes(support_by_class, CLASS_ORDER)
            per_class_ms = {}
            for cls in CLASS_ORDER:
                t_one = time.perf_counter()
                build_prototypes({cls: support_by_class[cls]}, (cls,))
                per_class_ms[cls] = (time.perf_counter() - t_one) * 1000
            registration_ms = (time.perf_counter() - t_reg) * 1000

            # 阈值：仅使用 Validation
            thresholds, threshold_stats = calibrate_thresholds(
                val_by_class, prototypes, CLASS_ORDER, quantile=args.quantile
            )

            # Test 预测
            prediction = predict_nearest(
                test_features, prototypes, CLASS_ORDER, thresholds=thresholds
            )
            accepted = prediction.accepted

            y_true = [row["class_prefix"] for row in test_rows]
            closed = classification_metrics(y_true, prediction.labels, CLASS_ORDER)
            rejection = rejection_metrics(y_true, prediction.labels, accepted, CLASS_ORDER)

            # Validation 预测（用于报告，不参与调参）
            val_prediction = predict_nearest(
                gather_features(val_rows, feature_by_md5), prototypes, CLASS_ORDER
            )
            val_pred_rows = prediction_rows(
                val_rows,
                val_prediction.labels,
                val_prediction.distances,
                val_prediction.min_distance,
                np.ones(len(val_rows), dtype=bool),
                seed=seed,
                shot=shot,
            )
            test_pred_rows = prediction_rows(
                test_rows,
                prediction.labels,
                prediction.distances,
                prediction.min_distance,
                accepted,
                seed=seed,
                shot=shot,
            )

            group_dir = out_root / f"seed_{seed}" / f"shot_{shot}"
            group_dir.mkdir(parents=True, exist_ok=True)
            pred_fields = [
                "file_name",
                "md5",
                "true_label",
                "predicted_label",
                "minimum_distance",
                "accepted",
                "seed",
                "shot",
            ] + [f"d_{cls}" for cls in CLASS_ORDER]
            write_csv(group_dir / "predictions_test.csv", test_pred_rows, pred_fields)
            write_csv(group_dir / "predictions_validation.csv", val_pred_rows, pred_fields)

            proto_rows = []
            for cls in CLASS_ORDER:
                vector = prototypes[cls]
                record = {
                    "class_prefix": cls,
                    "support_count": len(support_by_class[cls]),
                    "support_files": "|".join(
                        row["file_name"] for row in support_rows if row["class_prefix"] == cls
                    ),
                    "norm": round(float(np.linalg.norm(vector)), 6),
                    "build_ms": round(per_class_ms[cls], 3),
                }
                for dim in range(EMBED_DIM):
                    record[f"d{dim}"] = float(vector[dim])
                proto_rows.append(record)
            write_csv(
                group_dir / "prototypes.csv",
                proto_rows,
                ["class_prefix", "support_count", "support_files", "norm", "build_ms"]
                + [f"d{i}" for i in range(EMBED_DIM)],
            )

            (group_dir / "thresholds.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "shot": shot,
                        "quantile": args.quantile,
                        "source": "validation",
                        "thresholds": thresholds,
                        "validation_stats": threshold_stats,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            confusion_rows = []
            for i, cls in enumerate(CLASS_ORDER):
                record = {"true_label": cls}
                for j, pred_cls in enumerate(CLASS_ORDER):
                    record[pred_cls] = closed["confusion_matrix"][i][j]
                confusion_rows.append(record)
            write_csv(
                group_dir / "confusion_matrix.csv",
                confusion_rows,
                ["true_label"] + list(CLASS_ORDER),
            )

            metrics_payload = {
                "seed": seed,
                "shot": shot,
                "feature_version": FEATURE_VERSION,
                "preprocess_version": PREPROCESS_VERSION,
                "weights_tag": WEIGHTS_TAG,
                "class_order": list(CLASS_ORDER),
                "counts": {
                    "support": len(support_rows),
                    "validation": len(val_rows),
                    "test": len(test_rows),
                },
                "classification_closed_set": closed,
                "rejection": rejection,
                "timing": {
                    "prototype_registration_ms": round(registration_ms, 3),
                    "per_class_registration_ms": {
                        cls: round(ms, 3) for cls, ms in per_class_ms.items()
                    },
                    "feature_lookup_ms": round(lookup_ms, 3),
                },
            }
            (group_dir / "metrics.json").write_text(
                json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            row_summary = {
                "seed": seed,
                "shot": shot,
                "accuracy": round(closed["accuracy"], 6),
                "macro_f1": round(closed["macro_f1"], 6),
                "rejection_rate": round(rejection["rejection_rate"], 6),
                "accepted_accuracy": (
                    None
                    if rejection["accepted_accuracy"] is None
                    else round(rejection["accepted_accuracy"], 6)
                ),
                "accepted_count": rejection["accepted_count"],
                "test_count": rejection["test_total"],
                "registration_ms": round(registration_ms, 3),
                "lookup_ms": round(lookup_ms, 3),
                "threshold_mean": round(float(np.mean(list(thresholds.values()))), 6),
            }
            summary_rows.append(row_summary)
            per_group[f"seed_{seed}_shot_{shot}"] = {
                "summary": row_summary,
                "per_class": closed["per_class"],
                "rejection": rejection,
                "thresholds": thresholds,
            }
            print(
                f"[ok] seed={seed:<5} shot={shot}  acc={closed['accuracy']:.4f}  "
                f"f1={closed['macro_f1']:.4f}  rej={rejection['rejection_rate']:.4f}  "
                f"acc_acc={rejection['accepted_accuracy']}  reg={registration_ms:.2f}ms"
            )

    # ---------- 汇总 ----------
    write_csv(
        out_root / "summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )

    aggregate = {}
    for shot in shots:
        rows = [r for r in summary_rows if r["shot"] == shot]
        aggregate[f"shot_{shot}"] = {
            "seeds": [r["seed"] for r in rows],
            "accuracy": summarize([r["accuracy"] for r in rows]),
            "macro_f1": summarize([r["macro_f1"] for r in rows]),
            "rejection_rate": summarize([r["rejection_rate"] for r in rows]),
            "accepted_accuracy": summarize([r["accepted_accuracy"] for r in rows]),
            "registration_ms": summarize([r["registration_ms"] for r in rows]),
        }

    inputs_sha = {"manifest": sha256_file(Path(args.manifest))}
    for seed in seeds:
        for name in ("support_1shot.csv", "support_3shot.csv", "support_5shot.csv", "validation.csv", "test.csv"):
            path = splits_root / f"seed_{seed}" / name
            inputs_sha[f"seed_{seed}/{name}"] = sha256_file(path)

    summary = {
        "schema": "microdefect.fewshot_summary/1",
        "generator": "scripts/run_fewshot_experiment.py",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            "seeds": list(seeds),
            "shots": list(shots),
            "quantile": args.quantile,
            "method": "nearest prototype, cosine distance",
            "weights_tag": WEIGHTS_TAG,
            "feature_version": FEATURE_VERSION,
            "preprocess_version": PREPROCESS_VERSION,
            "embed_dim": EMBED_DIM,
        },
        "features": {
            "unique_contents": len(unique),
            "rows": len(all_rows),
            "cache_hits": len(feature_by_md5) - len(missing),
            "cache_misses": len(missing),
            "cache_read_ms": round(cache_read_ms, 3),
            "extraction_ms": round(extract_ms, 3),
            "cache_stats_before_extract": stats_before_extract,
            "cache_stats_after": dict(cache.stats),
        },
        "inference_timing": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in timing.items()},
        "extractor": extractor_info,
        "aggregate": aggregate,
        "groups": per_group,
        "inputs_sha256": inputs_sha,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log_lines = [
        "微缺智学 P3 实验日志",
        f"开始: {started_at}",
        f"结束: {summary['finished_at']}",
        f"seed: {list(seeds)}  shot: {list(shots)}  quantile: {args.quantile}",
        f"独立内容数: {len(unique)}  记录数: {len(all_rows)}",
        f"缓存命中/未命中: {len(feature_by_md5) - len(missing)}/{len(missing)}",
        f"缓存读取: {cache_read_ms:.1f} ms  特征提取: {extract_ms:.1f} ms",
        f"推理设备: {timing['device']}  单张 {timing['single_ms']} ms  批量 {timing['batch_ms']} ms",
        "",
        "seed,shot,accuracy,macro_f1,rejection_rate,accepted_accuracy,registration_ms",
    ]
    for row in summary_rows:
        log_lines.append(
            f"{row['seed']},{row['shot']},{row['accuracy']},{row['macro_f1']},"
            f"{row['rejection_rate']},{row['accepted_accuracy']},{row['registration_ms']}"
        )
    (out_root / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("-" * 72)
    for shot in shots:
        agg = aggregate[f"shot_{shot}"]
        print(
            f"shot={shot}  acc={agg['accuracy']['mean']:.4f}±{agg['accuracy']['std']:.4f}  "
            f"f1={agg['macro_f1']['mean']:.4f}±{agg['macro_f1']['std']:.4f}  "
            f"rej={agg['rejection_rate']['mean']:.4f}  "
            f"reg={agg['registration_ms']['mean']:.2f}ms"
        )
    print(f"输出目录: {out_root}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
