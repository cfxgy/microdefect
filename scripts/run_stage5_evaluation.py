#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2.1 阶段 5：冻结协议下的正式评估。

执行顺序（严格）：

1. **冻结前审计**（不读取任何 test 分数）
   - 核验 v2.2 协议文件与哈希记录；
   - 核验阶段 4 代码哈希；
   - 校验 A 通道实现语义（合成数据，不触碰真实 test）；
   - 数据隔离审计（reference / calibration / test，source ID + MD5 双重）；
   - 写 stage5_frozen/，并声明 test 尚未读取。
2. **正式评估**（只在此后读取 test 图像）
   - 每 seed/class：reference model → calibration 分数 → conformal p 值；
   - 指标：FPR + Clopper-Pearson、各扰动条件 Recall / AUROC / PR-AUC（配对 source bootstrap）；
   - 五 seed 的均值 ± 标准差（划分敏感性）。

只使用 A 通道；B 通道不参与告警；不使用融合。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from scipy.stats import beta, rankdata  # noqa: E402

from src.data.split import CLASS_ORDER  # noqa: E402
from src.drift.image_statistics import compute_image_stats, load_gray  # noqa: E402
from src.drift.metrics import auroc, mean_std  # noqa: E402
from src.drift_calibration.conformal import CalibrationSet, decide  # noqa: E402
from src.drift_calibration.isolation import IsolationError, assert_disjoint  # noqa: E402
from src.drift_calibration.protocol import ALPHA, N_CALIBRATION  # noqa: E402
from src.drift_calibration.score import ImageScorer  # noqa: E402

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
PERT_ROOT = PROJECT_ROOT / "experiments" / "data" / "perturbations"
P42_ROOT = PROJECT_ROOT / "experiments" / "results" / "drift_monitoring"
CAL_ROOT = PROJECT_ROOT / "experiments" / "results" / "drift_monitoring_calibration"
DEFAULT_OUT = CAL_ROOT / "stage5_evaluation"
DEFAULT_FROZEN = CAL_ROOT / "stage5_frozen"
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED_BASE = 20261110


class Stage5Error(RuntimeError):
    """阶段 5 的审计或执行失败。"""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


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


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average Precision（PR 曲线下面积，连续分数）。"""
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / np.maximum(tp + fp, 1)
    total_positive = int(sorted_labels.sum())
    if total_positive == 0:
        return float("nan")
    recall = tp / total_positive
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def _fast_auroc(scores: np.ndarray, n_positive: int) -> float:
    """基于平均秩的 AUROC（前半为负样本、后半为正样本的拼接顺序）。"""
    n = scores.size
    n_negative = n - n_positive
    ranks = rankdata(scores)
    positive_rank_sum = ranks[n_negative:].sum()
    statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return float(statistic / (n_positive * n_negative))


def bootstrap_pair_metrics(
    negative: np.ndarray,
    positive: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict:
    """配对 source bootstrap：每次同时抽取 (s⁻, s⁺)。"""
    rng = np.random.default_rng(seed)
    n = negative.size
    auc_values = np.empty(iterations)
    ap_values = np.empty(iterations)
    labels_all = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    for i in range(iterations):
        index = rng.integers(0, n, size=n)
        neg = negative[index]
        pos = positive[index]
        scores = np.concatenate([neg, pos])
        auc_values[i] = _fast_auroc(scores, n)
        ap_values[i] = average_precision(scores, labels_all)
    return {
        "auroc_ci_low": float(np.percentile(auc_values, 2.5)),
        "auroc_ci_high": float(np.percentile(auc_values, 97.5)),
        "pr_auc_ci_low": float(np.percentile(ap_values, 2.5)),
        "pr_auc_ci_high": float(np.percentile(ap_values, 97.5)),
        "auroc_boot_mean": float(auc_values.mean()),
        "pr_auc_boot_mean": float(ap_values.mean()),
    }


def bootstrap_seed(seed: int, class_name: str, condition: str) -> int:
    material = f"p4.2.1-stage5|{BOOTSTRAP_SEED_BASE}|{seed}|{class_name}|{condition}".encode()
    return int(hashlib.sha256(material).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------
# 阶段 A：冻结前审计
# ---------------------------------------------------------------------------


def verify_protocol_v22() -> dict:
    record_path = CAL_ROOT / "protocol_v2.2_sha256.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    files = record["files_v2_2"]
    checked = {}
    for name, item in files.items():
        path = CAL_ROOT / name
        if not path.is_file():
            raise Stage5Error(f"缺少 v2.2 协议文件：{path}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise Stage5Error(f"v2.2 协议哈希不一致：{name}")
        checked[name] = {"sha256": actual, "size_bytes": path.stat().st_size}
    return {
        "record": str(record_path),
        "record_sha256": sha256_file(record_path),
        "files": checked,
        "all_match": True,
    }


def verify_stage4_code() -> dict:
    record = json.loads((CAL_ROOT / "stage4_implementation" / "stage4_sha256.json").read_text(encoding="utf-8"))
    checked = {}
    for group in ("code_files", "test_files"):
        for rel, item in record[group].items():
            path = PROJECT_ROOT / rel
            if not path.is_file():
                raise Stage5Error(f"缺少阶段 4 文件：{rel}")
            actual = sha256_file(path)
            if actual != item["sha256"]:
                raise Stage5Error(f"阶段 4 代码哈希不一致：{rel}")
            checked[rel] = actual
    return {"files": checked, "all_match": True, "record_sha256": sha256_file(CAL_ROOT / "stage4_implementation" / "stage4_sha256.json")}


def verify_score_semantics() -> dict:
    """用合成数据校验 A 通道实现与 v2.2 一致（不读取真实 test）。"""
    rng = np.random.default_rng(0)
    reference_stats = [
        compute_image_stats(rng.integers(0, 256, size=(64, 64), dtype=np.uint8)) for _ in range(25)
    ]
    scorer = ImageScorer(reference_stats, "Cr")
    image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    result = scorer.score_gray(image)

    matrix = np.array(
        [[s[k] for k in ("mean_intensity", "contrast", "laplacian_variance")] for s in reference_stats]
    )
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    std = matrix.std(axis=0, ddof=1)
    scale = 1.4826 * mad
    for index in range(3):
        if scale[index] < 1e-3:
            scale[index] = std[index] if std[index] >= 1e-3 else 1e-3
    manual = float(np.sqrt(np.mean((np.abs(np.array(result.stats) - median) / scale) ** 2)))

    checks = {
        "score_matches_manual_formula": abs(result.score_A - manual) < 1e-9,
        "score_is_non_negative": result.score_A >= 0.0,
        "score_finite": bool(np.isfinite(result.score_A)),
        "alpha_is_0_05": ALPHA == 0.05,
        "calibration_size_is_25": N_CALIBRATION == 25,
    }
    if not all(checks.values()):
        raise Stage5Error(f"A 通道语义校验失败：{checks}")
    return checks


def load_seed_data(seed: int) -> dict:
    """读取某 seed 的 P4-1 清单并整理 ref/cal/test 集合（不计算任何分数）。"""
    rows = read_csv(PERT_ROOT / f"seed_{seed}" / "perturbation_manifest.csv")
    originals: dict[tuple[str, str], dict] = {}
    perturbed: dict[tuple[str, str, str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["perturbation_type"] == "original":
            originals[(row["split"], row["source_file"])] = row
        else:
            perturbed[(row["split"], row["source_class"], row["perturbation_type"], row["perturbation_level"])][
                row["source_file"]
            ] = row

    reference_rows = read_csv(P42_ROOT / f"seed_{seed}" / "reference_pool.csv")
    references: dict[str, list[dict]] = defaultdict(list)
    for row in reference_rows:
        references[row["class_name"]].append(row)

    return {
        "rows": rows,
        "originals": originals,
        "perturbed": perturbed,
        "references": references,
    }


def audit_isolation(seed: int, data: dict) -> dict:
    per_class = {}
    for cls in CLASS_ORDER:
        ref = data["references"][cls]
        ref_ids = [row["source_file"] for row in ref]
        ref_md5 = [row["source_md5"] for row in ref]

        validation_originals = [
            row for (split, _), row in data["originals"].items()
            if split == "validation" and row["source_class"] == cls
        ]
        ref_id_set = set(ref_ids)
        calibration = [row for row in validation_originals if row["source_file"] not in ref_id_set]
        test = [
            row for (split, _), row in data["originals"].items()
            if split == "test" and row["source_class"] == cls
        ]

        cal_ids = [row["source_file"] for row in calibration]
        cal_md5 = [row["source_md5"] for row in calibration]
        test_ids = [row["source_file"] for row in test]
        test_md5 = [row["source_md5"] for row in test]

        assert_disjoint(ref_ids, cal_ids, test_ids)

        # MD5 层面交叉检查（同一内容不得跨集合）
        ref_md5_set, cal_md5_set, test_md5_set = set(ref_md5), set(cal_md5), set(test_md5)
        overlap = {
            "ref_cal": sorted(ref_md5_set & cal_md5_set),
            "ref_test": sorted(ref_md5_set & test_md5_set),
            "cal_test": sorted(cal_md5_set & test_md5_set),
        }
        if any(overlap.values()):
            raise Stage5Error(f"seed={seed} {cls} MD5 跨集合重复：{overlap}")

        per_class[cls] = {
            "reference_files": len(ref_ids),
            "reference_unique_md5": len(ref_md5_set),
            "calibration_files": len(cal_ids),
            "calibration_unique_md5": len(cal_md5_set),
            "test_files": len(test_ids),
            "test_unique_md5": len(test_md5_set),
            "source_id_overlap": 0,
            "md5_overlap": 0,
            "reference_subset_of_validation": ref_id_set <= {row["source_file"] for row in validation_originals},
        }
        if len(cal_ids) != N_CALIBRATION:
            raise Stage5Error(f"seed={seed} {cls} calibration 数量 {len(cal_ids)} != 25")
    return per_class


def phase_freeze(frozen_dir: Path, out_dir: Path) -> dict:
    frozen_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone().isoformat(timespec="seconds")

    protocol = verify_protocol_v22()
    code = verify_stage4_code()
    semantics = verify_score_semantics()

    isolation = {}
    data_hashes = {}
    for seed in SEEDS:
        data = load_seed_data(seed)
        isolation[f"seed_{seed}"] = audit_isolation(seed, data)
        for name in ("perturbation_manifest.csv",):
            path = PERT_ROOT / f"seed_{seed}" / name
            data_hashes[f"p41/seed_{seed}/{name}"] = sha256_file(path)
        for name in ("reference_pool.csv", "calibration_thresholds.json"):
            path = P42_ROOT / f"seed_{seed}" / name
            if path.is_file():
                data_hashes[f"p42/seed_{seed}/{name}"] = sha256_file(path)
        for name in ("validation.csv", "test.csv"):
            path = PROJECT_ROOT / "experiments" / "results" / "splits" / f"seed_{seed}" / name
            data_hashes[f"p12/seed_{seed}/{name}"] = sha256_file(path)

    freeze = {
        "schema": "microdefect.drift_calibration/stage5_freeze",
        "frozen_at": started,
        "test_not_read": True,
        "test_read_statement": "本记录生成时，尚未读取、计算或查看任何 test 分数或 test 标签统计。",
        "protocol": {
            "version": "p4.2.1-stage3-v2.2",
            "design_file": "protocol_design_v2.2.md",
            "hashes": protocol,
        },
        "code": code,
        "score_semantics_checks": semantics,
        "parameters": {
            "alpha": ALPHA,
            "n_calibration": N_CALIBRATION,
            "window_size": 16,
            "score_channel": "A",
            "primary_alert_channel": "A",
            "fusion_used": False,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
            "bootstrap_seed_rule": "sha256('p4.2.1-stage5|<base>|<seed>|<class>|<condition>')[:16] as int",
            "tie_rule": ">=",
            "alert_rule": "p <= 0.05",
            "seeds": list(SEEDS),
            "classes": list(CLASS_ORDER),
            "conditions": [list(c) for c in CONDITIONS],
        },
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "data_hashes": data_hashes,
        "isolation_audit": isolation,
        "notes": [
            "reference pool 来自 validation，是 validation 的子集，不等同于与整个 validation 独立。",
            "calibration pool 是 validation 中 reference 之外的余集。",
            "seed=7 的 Pa 类含重复内容（Pa_101/Pa_105），文件数与独立 MD5 数分别记录。",
        ],
    }
    write_json(frozen_dir / "freeze_record.json", freeze)
    write_json(frozen_dir / "data_isolation_audit.json", isolation)
    write_json(frozen_dir / "protocol_hashes.json", protocol)
    write_json(frozen_dir / "code_hashes.json", code)
    write_json(frozen_dir / "data_hashes.json", data_hashes)

    print(f"[freeze] 完成：{started}，test_not_read=True")
    for seed in SEEDS:
        cls_check = isolation[f"seed_{seed}"]["Pa"]
        print(
            f"  seed={seed}: ref/cal/test = "
            f"{cls_check['reference_files']}/{cls_check['calibration_files']}/{cls_check['test_files']} (Pa)"
        )
    return freeze


# ---------------------------------------------------------------------------
# 阶段 B：正式评估
# ---------------------------------------------------------------------------


def collect_stats(seeds: tuple[int, ...], data_by_seed: dict) -> tuple[dict, dict]:
    """计算所需图像的统计量（reference/validation originals + test 全部）。"""
    needed: dict[str, str] = {}
    for seed in seeds:
        data = data_by_seed[seed]
        for (split, _), row in data["originals"].items():
            needed.setdefault(row["output_md5"], row["abs_path"] if "abs_path" in row else
                              str(PERT_ROOT / f"seed_{seed}" / row["output_rel_path"]))
        for key, mapping in data["perturbed"].items():
            split, cls, ptype, level = key
            if split != "test":
                continue
            for row in mapping.values():
                needed.setdefault(row["output_md5"], str(PERT_ROOT / f"seed_{seed}" / row["output_rel_path"]))

    stats_by_md5: dict[str, dict] = {}
    started = time.perf_counter()
    for index, (md5, path) in enumerate(needed.items(), start=1):
        stats_by_md5[md5] = compute_image_stats(load_gray(path))
        if index % 10000 == 0:
            print(f"  [stats] {index}/{len(needed)}")
    elapsed = time.perf_counter() - started
    return stats_by_md5, {"unique_images": len(needed), "seconds": round(elapsed, 2)}


def evaluate(out_dir: Path, seeds: tuple[int, ...], stats_by_md5: dict) -> dict:
    per_seed_class = []
    per_condition = []
    bootstrap_rows = []
    prediction_rows = []
    alert_rows = []

    for seed in seeds:
        data = load_seed_data(seed)
        for cls in CLASS_ORDER:
            ref_rows = data["references"][cls]
            reference_stats = [stats_by_md5[row["source_md5"]] for row in ref_rows]
            scorer = ImageScorer(reference_stats, cls)

            ref_ids = {row["source_file"] for row in ref_rows}
            calibration_rows = [
                row for (split, _), row in data["originals"].items()
                if split == "validation" and row["source_class"] == cls and row["source_file"] not in ref_ids
            ]
            calibration_scores = np.array(
                [scorer.score_stats(stats_by_md5[row["source_md5"]]).score_A for row in calibration_rows]
            )
            calibration = CalibrationSet(seed=seed, class_name=cls, scores=tuple(calibration_scores))

            test_rows = [
                row for (split, _), row in data["originals"].items()
                if split == "test" and row["source_class"] == cls
            ]
            test_rows.sort(key=lambda r: int(r["source_file"].rsplit("_", 1)[1].split(".")[0]))
            clean_scores = np.array(
                [scorer.score_stats(stats_by_md5[row["source_md5"]]).score_A for row in test_rows]
            )
            clean_decisions = [
                decide(calibration, float(score), seed=seed, class_name=cls) for score in clean_scores
            ]
            clean_alerts = sum(1 for d in clean_decisions if d["alert"])
            n_clean = len(clean_decisions)
            fpr = clean_alerts / n_clean
            fpr_low, fpr_high = clopper_pearson(clean_alerts, n_clean)

            for row, score, decision in zip(test_rows, clean_scores, clean_decisions):
                prediction_rows.append(
                    {
                        "seed": seed,
                        "class_name": cls,
                        "split": "test",
                        "condition": "unperturbed",
                        "source_file": row["source_file"],
                        "source_md5": row["source_md5"],
                        "score_A": round(float(score), 8),
                        "p_value": round(decision["p_value"], 8),
                        "alert": int(decision["alert"]),
                    }
                )
                if decision["alert"]:
                    alert_rows.append(
                        {
                            "seed": seed,
                            "class_name": cls,
                            "condition": "unperturbed",
                            "source_file": row["source_file"],
                            "score_A": round(float(score), 8),
                            "p_value": round(decision["p_value"], 8),
                            "alert": 1,
                        }
                    )

            per_seed_class.append(
                {
                    "seed": seed,
                    "class_name": cls,
                    "n_calibration": len(calibration_scores),
                    "calibration_score_mean": round(float(calibration_scores.mean()), 6),
                    "calibration_score_max": round(float(calibration_scores.max()), 6),
                    "n_test_unperturbed": n_clean,
                    "false_alarms": clean_alerts,
                    "fpr": round(fpr, 8),
                    "fpr_ci_low": None if fpr_low is None else round(fpr_low, 8),
                    "fpr_ci_high": None if fpr_high is None else round(fpr_high, 8),
                    "clean_score_mean": round(float(clean_scores.mean()), 6),
                }
            )

            for condition in CONDITIONS:
                mapping = data["perturbed"][("test", cls, condition[0], condition[1])]
                sources = [row["source_file"] for row in test_rows]
                perturbed_scores = np.array(
                    [scorer.score_stats(stats_by_md5[mapping[source]["output_md5"]]).score_A for source in sources]
                )
                perturbed_decisions = [
                    decide(calibration, float(score), seed=seed, class_name=cls) for score in perturbed_scores
                ]
                recalls = sum(1 for d in perturbed_decisions if d["alert"])
                recall = recalls / len(perturbed_decisions)

                labels = np.concatenate(
                    [np.zeros(n_clean, dtype=np.int64), np.ones(len(perturbed_scores), dtype=np.int64)]
                )
                scores_all = np.concatenate([clean_scores, perturbed_scores])
                auc = auroc(scores_all, labels)
                ap = average_precision(scores_all, labels)
                condition_name = f"{condition[0]}_{condition[1]}"
                boot = bootstrap_pair_metrics(
                    clean_scores,
                    perturbed_scores,
                    iterations=BOOTSTRAP_ITERATIONS,
                    seed=bootstrap_seed(seed, cls, condition_name),
                )
                bootstrap_rows.append(
                    {
                        "seed": seed,
                        "class_name": cls,
                        "condition": condition_name,
                        "auroc": None if auc is None else round(float(auc), 8),
                        **{k: round(v, 8) for k, v in boot.items()},
                    }
                )
                per_condition.append(
                    {
                        "seed": seed,
                        "class_name": cls,
                        "perturbation_type": condition[0],
                        "perturbation_level": condition[1],
                        "condition": condition_name,
                        "n_perturbed": len(perturbed_scores),
                        "detections": recalls,
                        "recall": round(recall, 8),
                        "auroc": None if auc is None else round(float(auc), 8),
                        "pr_auc": round(ap, 8),
                        "auroc_ci_low": round(boot["auroc_ci_low"], 8),
                        "auroc_ci_high": round(boot["auroc_ci_high"], 8),
                        "pr_auc_ci_low": round(boot["pr_auc_ci_low"], 8),
                        "pr_auc_ci_high": round(boot["pr_auc_ci_high"], 8),
                        "clean_score_mean": round(float(clean_scores.mean()), 6),
                        "perturbed_score_mean": round(float(perturbed_scores.mean()), 6),
                        "delta_score": round(float(perturbed_scores.mean() - clean_scores.mean()), 6),
                    }
                )
                for source, score, decision in zip(sources, perturbed_scores, perturbed_decisions):
                    prediction_rows.append(
                        {
                            "seed": seed,
                            "class_name": cls,
                            "split": "test",
                            "condition": condition_name,
                            "source_file": source,
                            "source_md5": mapping[source]["source_md5"],
                            "score_A": round(float(score), 8),
                            "p_value": round(decision["p_value"], 8),
                            "alert": int(decision["alert"]),
                        }
                    )

    write_csv(out_dir / "per_seed_class_metrics.csv", per_seed_class)
    write_csv(out_dir / "per_perturbation_metrics.csv", per_condition)
    write_csv(out_dir / "bootstrap_ci.csv", bootstrap_rows)
    write_csv(out_dir / "prediction_scores.csv", prediction_rows)
    write_csv(out_dir / "conformal_alerts.csv", alert_rows)
    return {
        "per_seed_class": per_seed_class,
        "per_condition": per_condition,
        "bootstrap": bootstrap_rows,
        "predictions": prediction_rows,
        "alerts": alert_rows,
    }


def summarize(rows: dict) -> dict:
    per_seed_class = rows["per_seed_class"]
    per_condition = rows["per_condition"]

    seeds = sorted({row["seed"] for row in per_seed_class})
    per_seed = []
    for seed in seeds:
        subset = [row for row in per_seed_class if row["seed"] == seed]
        false_alarms = sum(row["false_alarms"] for row in subset)
        total = sum(row["n_test_unperturbed"] for row in subset)
        low, high = clopper_pearson(false_alarms, total)
        per_seed.append(
            {
                "seed": seed,
                "false_alarms": false_alarms,
                "n_test": total,
                "fpr": false_alarms / total,
                "fpr_ci_low": low,
                "fpr_ci_high": high,
            }
        )

    conditions = [f"{t}_{lvl}" for t, lvl in CONDITIONS]
    per_condition_macro = {}
    for condition in conditions:
        subset = [row for row in per_condition if row["condition"] == condition]
        per_condition_macro[condition] = {
            "n": len(subset),
            "recall": mean_std([row["recall"] for row in subset]),
            "auroc": mean_std([row["auroc"] for row in subset]),
            "pr_auc": mean_std([row["pr_auc"] for row in subset]),
            "delta_score": mean_std([row["delta_score"] for row in subset]),
        }

    macro_by_seed = {}
    for seed in seeds:
        subset = [row for row in per_condition if row["seed"] == seed]
        macro_by_seed[str(seed)] = {
            "recall": mean_std([row["recall"] for row in subset]),
            "auroc": mean_std([row["auroc"] for row in subset]),
            "pr_auc": mean_std([row["pr_auc"] for row in subset]),
        }

    fpr_values = [row["fpr"] for row in per_seed]
    return {
        "per_seed_fpr": per_seed,
        "fpr_across_seeds": mean_std(fpr_values),
        "per_condition_macro": per_condition_macro,
        "per_seed_condition_macro": macro_by_seed,
        "n_predictions": len(rows["predictions"]),
        "n_alerts": len(rows["alerts"]),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def rebuild_derived_outputs(out_dir: Path) -> dict:
    """从已产生的 prediction_scores.csv 派生出告警清单并修正 summary 计数。

    这是**输出完整性修复**，不重新计算分数、不改变判定、不重跑评估：
    ``conformal_alerts.csv`` 应包含全部告警（未扰动 + 受控扰动），
    而首次实现只写入了未扰动告警。
    """
    predictions = read_csv(out_dir / "prediction_scores.csv")
    alerts = [row for row in predictions if row["alert"] == "1"]
    write_csv(out_dir / "conformal_alerts.csv", alerts)

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    unperturbed = sum(1 for row in alerts if row["condition"] == "unperturbed")
    perturbed = len(alerts) - unperturbed
    summary["n_predictions"] = len(predictions)
    summary["n_alerts"] = len(alerts)
    summary["n_alerts_unperturbed"] = unperturbed
    summary["n_alerts_perturbed"] = perturbed
    summary["derived_rebuild"] = {
        "source": "prediction_scores.csv",
        "reason": "首次实现只把未扰动告警写入 conformal_alerts.csv，属输出完整性缺陷；"
                  "本重建仅从已存储的逐图预测派生，不重新评分、不改变判定、不重跑评估。",
        "predictions": len(predictions),
        "alerts_total": len(alerts),
        "alerts_unperturbed": unperturbed,
        "alerts_perturbed": perturbed,
    }
    write_json(summary_path, summary)
    return summary["derived_rebuild"]


def main() -> int:
    parser = argparse.ArgumentParser(description="P4-2.1 阶段 5：冻结协议下的正式评估")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--frozen", default=str(DEFAULT_FROZEN))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--rebuild-derived", action="store_true", help="从已存预测派生出告警清单")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.alpha != ALPHA:
        print(f"[FAIL] alpha 冻结为 {ALPHA}，不允许改为 {args.alpha}")
        return 2
    if args.bootstrap_iterations != BOOTSTRAP_ITERATIONS:
        print(f"[FAIL] bootstrap 次数冻结为 {BOOTSTRAP_ITERATIONS}")
        return 2

    out_dir = Path(args.out)
    frozen_dir = Path(args.frozen)
    seeds = tuple(args.seeds)

    if args.rebuild_derived:
        info = rebuild_derived_outputs(out_dir)
        print("[rebuild] 派生修复完成：", json.dumps(info, ensure_ascii=False))
        return 0

    if args.verify_only:
        required = (
            "evaluation_config.json",
            "per_seed_class_metrics.csv",
            "per_perturbation_metrics.csv",
            "bootstrap_ci.csv",
            "prediction_scores.csv",
            "conformal_alerts.csv",
            "summary.json",
            "final_report.md",
            "sha256_manifest.json",
        )
        missing = [name for name in required if not (out_dir / name).is_file()]
        print(f"[verify] 缺失 {len(missing)} 项")
        for name in missing:
            print("  -", name)
        return 0 if not missing else 1

    print("=" * 78)
    print("P4-2.1 阶段 5：冻结协议下的正式图像级 conformal 评估")
    print("=" * 78)

    freeze = phase_freeze(frozen_dir, out_dir)
    if args.freeze_only:
        print("PASS (freeze only)")
        return 0

    print("[eval] 开始读取 test 图像并计算分数（此前 test 未被读取）")
    data_by_seed = {seed: load_seed_data(seed) for seed in seeds}
    stats_by_md5, stats_timing = collect_stats(seeds, data_by_seed)
    print(f"[eval] 统计量完成：{stats_timing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = evaluate(out_dir, seeds, stats_by_md5)
    summary = summarize(rows)
    summary["timing"] = {
        "stats_seconds": stats_timing["seconds"],
        "unique_images": stats_timing["unique_images"],
        "evaluation_seconds": round(time.perf_counter() - started, 2),
    }
    config = {
        "schema": "microdefect.drift_calibration/stage5_evaluation_config",
        "protocol": "p4.2.1-stage3-v2.2",
        "alpha": ALPHA,
        "n_calibration": N_CALIBRATION,
        "channel": "A",
        "fusion_used": False,
        "seeds": list(seeds),
        "classes": list(CLASS_ORDER),
        "conditions": [list(c) for c in CONDITIONS],
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
        "freeze_record_sha256": sha256_file(frozen_dir / "freeze_record.json"),
        "test_not_read_before_freeze": True,
    }
    write_json(out_dir / "evaluation_config.json", config)
    write_json(out_dir / "data_isolation_audit.json", freeze["isolation_audit"])
    write_json(out_dir / "summary.json", summary)

    print("-" * 78)
    for item in summary["per_seed_fpr"]:
        print(
            f"seed={item['seed']:<5} FPR={item['fpr']:.4f} "
            f"[{item['fpr_ci_low']:.4f}, {item['fpr_ci_high']:.4f}] "
            f"({item['false_alarms']}/{item['n_test']})"
        )
    for condition, metrics in summary["per_condition_macro"].items():
        print(
            f"  {condition:<18} recall={metrics['recall']['mean']:.4f} "
            f"AUROC={metrics['auroc']['mean'] if metrics['auroc']['mean'] is not None else float('nan'):.4f} "
            f"PR-AUC={metrics['pr_auc']['mean']:.4f}"
        )
    print(f"[eval] 预测行 {summary['n_predictions']}，告警 {summary['n_alerts']}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
