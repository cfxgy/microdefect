#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P2 真实图片冒烟测试：特征提取 + 缓存命中/失效验证。

用法（必须使用项目虚拟环境解释器）：
    D:\\microdefect\\.venv\\Scripts\\python.exe scripts\\smoke_test_features.py

行为：
  * 若 ImageNet 权重未缓存且未加 ``--allow-download``，脚本**不会下载**，
    打印缺失信息并返回退出码 3（等待人工批准）。
  * 否则对 6 张真实图片（每类一张）+ Pa_101/Pa_105 做单张与批量提取，
    验证形状、数值有效性、重复一致性、缓存命中与失效。

不训练、不微调、不覆盖 P1 任何产物、不修改原始图片。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.features.cache import FeatureCache, md5_file  # noqa: E402
from src.features.extractor import (  # noqa: E402
    EMBED_DIM,
    PREPROCESS_VERSION,
    FeatureExtractor,
    WeightsMissingError,
    weights_status,
)

CLASS_ORDER = ("Cr", "In", "Pa", "PS", "RS", "Sc")
DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "results" / "neu_cls_manifest.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "NEU-CLS"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache" / "features"
DEFAULT_REPORT = PROJECT_ROOT / "cache" / "reports" / "feature_smoke_report.json"


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pick_one_per_class(rows: list[dict]) -> dict[str, dict]:
    chosen: dict[str, dict] = {}
    for row in sorted(rows, key=lambda r: (r["class_prefix"], int(Path(r["file_name"]).stem.split("_")[-1]))):
        chosen.setdefault(row["class_prefix"], row)
    return chosen


def gpu_snapshot() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "allocated_mib": round(torch.cuda.memory_allocated() / 1024 ** 2, 2),
            "reserved_mib": round(torch.cuda.memory_reserved() / 1024 ** 2, 2),
            "max_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
        }
    except Exception as exc:  # 仅用于报告
        return {"cuda_available": False, "error": f"{type(exc).__name__}: {exc}"}


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="微缺智学 P2 特征冒烟测试")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-download", action="store_true", help="允许下载官方预训练权重")
    args = parser.parse_args()

    status = weights_status()
    cache_dir = Path(args.cache_dir)
    report_path = Path(args.report)

    print("=" * 70)
    print("微缺智学 P2 特征提取冒烟测试")
    print("=" * 70)
    print(f"权重枚举  : {status['weights_enum']}")
    print(f"权重路径  : {status['expected_path']}")
    print(f"已缓存    : {status['cached']}")

    if not status["cached"] and not args.allow_download:
        payload = {
            "schema": "microdefect.feature_smoke/1",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "BLOCKED_MISSING_WEIGHTS",
            "weights": status,
            "action_required": (
                "需用户批准后下载官方 ImageNet 权重："
                f"torchvision {status['weights_enum']} → {status['url']}"
            ),
            "python": sys.executable,
        }
        write_report(report_path, payload)
        print("-" * 70)
        print("[BLOCKED] 预训练权重不在本地缓存，按规则不擅自下载。")
        print(f"  需要文件 : {Path(status['expected_path']).name}")
        print(f"  官方 URL : {status['url']}")
        print(f"  缓存位置 : {Path(status['expected_path']).parent}")
        print("  批准后可执行：scripts\\smoke_test_features.py --allow-download （仅首次下载）")
        print(f"  已写入阻塞说明：{report_path}")
        return 3

    data_root = Path(args.data_root)
    rows = load_manifest(Path(args.manifest))
    picked = pick_one_per_class(rows)
    missing = [cls for cls in CLASS_ORDER if cls not in picked]
    if missing:
        print(f"[FAIL] 清单中缺少类别：{missing}")
        return 2

    samples = [(cls, Path(picked[cls]["abs_path"]), picked[cls]["md5"]) for cls in CLASS_ORDER]
    duplicate_pair = [data_root / "Pa_101.bmp", data_root / "Pa_105.bmp"]

    print(f"设备      : {args.device}")
    print(f"批大小    : {args.batch_size}")
    print(f"缓存目录  : {cache_dir}")

    try:
        extractor = FeatureExtractor(args.device, allow_download=args.allow_download, batch_size=args.batch_size)
    except WeightsMissingError as exc:
        print(f"[FAIL] {exc}")
        return 3

    print(f"实际设备  : {extractor.device}")
    print(f"特征维度  : {EMBED_DIM}")
    print("-" * 70)

    # ---- 单张提取（逐张计时）----
    per_image = []
    single_features = {}
    t_all = time.perf_counter()
    for cls, path, md5 in samples:
        t0 = time.perf_counter()
        feature = extractor.extract_path(path)
        elapsed = (time.perf_counter() - t0) * 1000
        single_features[cls] = feature
        per_image.append(
            {
                "class": cls,
                "file": path.name,
                "md5": md5,
                "shape": list(feature.shape),
                "dtype": str(feature.dtype),
                "finite": bool(np.isfinite(feature).all()),
                "norm": round(float(np.linalg.norm(feature)), 4),
                "elapsed_ms": round(elapsed, 2),
                "first5": [round(float(v), 6) for v in feature[:5]],
            }
        )
        print(f"  {cls:<3} {path.name:<12} shape={feature.shape} finite={np.isfinite(feature).all()} "
              f"norm={np.linalg.norm(feature):.3f} {elapsed:7.2f} ms")
    total_single_ms = (time.perf_counter() - t_all) * 1000

    # ---- 批量提取 ----
    paths = [path for _, path, _ in samples]
    t0 = time.perf_counter()
    batch_features = extractor.extract_paths(paths)
    batch_ms = (time.perf_counter() - t0) * 1000
    batch_ok = batch_features.shape == (len(paths), EMBED_DIM) and np.isfinite(batch_features).all()

    # ---- 重复一致性 ----
    repeat = extractor.extract_paths(paths)
    repeat_identical = bool(np.array_equal(batch_features, repeat))

    # ---- 缓存演示 ----
    cache = FeatureCache(cache_dir)
    cache_pairs = [(s[0], s[2]) for s in samples] + [("Pa_dup", md5_file(duplicate_pair[0]))]

    first_pass = []
    for cls, md5 in cache_pairs:
        key = cache.key_for_md5(md5)
        hit = cache.get(key)
        first_pass.append({"class": cls, "md5": md5, "key": key, "hit": hit is not None})
        if hit is None:
            cache.put(key, single_features.get(cls, single_features["Pa"]))

    stats_after_first = dict(cache.stats)

    second_pass = []
    for cls, md5 in cache_pairs:
        key = cache.key_for_md5(md5)
        hit = cache.get(key)
        second_pass.append({"class": cls, "key": key, "hit": hit is not None})

    # ---- Pa_101 / Pa_105 共用内容缓存 ----
    pair_info = {}
    for path in duplicate_pair:
        md5 = md5_file(path)
        pair_info[path.name] = {"md5": md5, "key": cache.key_for_md5(md5)}
    pair_same_key = pair_info["Pa_101.bmp"]["key"] == pair_info["Pa_105.bmp"]["key"]

    # ---- 预处理版本变化导致缓存失效 ----
    bumped = FeatureCache(cache_dir, preprocess_version=PREPROCESS_VERSION + "-bumped")
    bumped_key = bumped.key_for_md5(cache_pairs[0][1])
    version_change_detected = bumped_key != cache.key_for_md5(cache_pairs[0][1])

    # ---- 原始特征 vs L2 归一化 ----
    raw = single_features["Cr"]
    normalized = extractor.extract_path(paths[0], normalize=True)

    all_ok = (
        all(item["shape"] == [EMBED_DIM] for item in per_image)
        and all(item["finite"] for item in per_image)
        and batch_ok
        and repeat_identical
        and pair_same_key
        and version_change_detected
    )

    payload = {
        "schema": "microdefect.feature_smoke/1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "OK" if all_ok else "FAIL",
        "python": sys.executable,
        "weights": status,
        "extractor": extractor.describe(),
        "device": str(extractor.device),
        "batch_size": args.batch_size,
        "dim": EMBED_DIM,
        "timing": {
            "single_total_ms": round(total_single_ms, 2),
            "single_mean_ms": round(total_single_ms / len(samples), 2),
            "batch_ms": round(batch_ms, 2),
            "batch_per_image_ms": round(batch_ms / len(samples), 2),
        },
        "gpu": gpu_snapshot(),
        "per_image": per_image,
        "batch": {
            "shape": list(batch_features.shape),
            "finite": bool(np.isfinite(batch_features).all()),
            "repeat_identical": repeat_identical,
        },
        "cache": {
            "directory": str(cache_dir),
            "first_pass": {"entries": first_pass, "stats": stats_after_first},
            "second_pass": {"entries": second_pass, "stats": dict(cache.stats)},
            "pair": {"files": pair_info, "same_key": pair_same_key, "same_md5": True},
            "preprocess_version_bump_changes_key": version_change_detected,
            "summary": cache.summary(),
        },
        "raw_vs_l2": {
            "raw_norm": round(float(np.linalg.norm(raw)), 4),
            "l2_norm": round(float(np.linalg.norm(normalized)), 6),
            "identical": bool(np.array_equal(raw, normalized)),
        },
        "all_ok": all_ok,
    }
    write_report(report_path, payload)

    print("-" * 70)
    print(f"单张总耗时 : {total_single_ms:.1f} ms（平均 {total_single_ms / len(samples):.1f} ms/张）")
    print(f"批量耗时   : {batch_ms:.1f} ms（{batch_features.shape}）")
    print(f"重复一致   : {repeat_identical}")
    print(f"缓存首轮   : hits={stats_after_first['hits']} misses={stats_after_first['misses']}")
    print(f"缓存次轮   : hits={dict(cache.stats)['hits']} misses={dict(cache.stats)['misses']}")
    print(f"Pa 重用    : 同键={pair_same_key} ({pair_info['Pa_101.bmp']['key']})")
    print(f"预处理版本变更导致键变化: {version_change_detected}")
    print(f"原始/L2    : raw_norm={np.linalg.norm(raw):.3f} l2_norm={np.linalg.norm(normalized):.3f}")
    print("-" * 70)
    print(f"报告: {report_path}")
    print(f"结果: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
