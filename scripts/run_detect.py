#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单图缺陷检测命令行入口（第一阶段）。

用法::

    D:\\microdefect\\.venv\\Scripts\\python.exe scripts\\run_detect.py ^
        --image "D:\\path\\to\\image.png" --category bottle --out "D:\\path\\to\\out"

输出：``*_detect.json``、``*_heatmap.png``、``*_overlay.png``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.detector import Detector, DetectorError  # noqa: E402
from src.core.registry import MemoryRegistry, RegistryError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微缺智学：单图缺陷检测")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--category", required=True, help="类别（如 bottle / pill / toothbrush / zipper）")
    parser.add_argument("--out", required=True, help="输出目录（不存在则创建）")
    parser.add_argument("--protocol", default="group_disjoint",
                        choices=("group_disjoint", "official"), help="记忆库协议")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--seed", type=int, default=None, help="官方协议使用的 seed（默认 42）")
    parser.add_argument("--shot", type=int, default=None, help="官方协议使用的 shot（默认 5）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        registry = MemoryRegistry(protocol=args.protocol, seed=args.seed, shot=args.shot)
        detector = Detector(
            args.category, protocol=args.protocol, device=args.device,
            registry=registry, seed=args.seed, shot=args.shot,
        )
        result = detector.detect(args.image)
        # 文件名加入路径短哈希，避免不同目录下的同名图片互相覆盖
        digest = hashlib.sha1(str(Path(args.image).resolve()).encode("utf-8")).hexdigest()[:8]
        stem = f"{Path(args.image).stem}_{digest}"
        files = result.save(args.out, stem=stem)
    except (RegistryError, DetectorError, FileNotFoundError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2

    print(json.dumps({
        "ok": True,
        "category": result.category,
        "protocol": result.protocol,
        "image_score": round(result.image_score, 6),
        "image_threshold": round(result.image_threshold, 6),
        "is_anomaly": result.is_anomaly,
        "pixel_threshold": round(result.pixel_threshold, 6),
        "image_size": list(result.image_size),
        "elapsed_time_s": round(result.elapsed_time, 4),
        "files": files,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
