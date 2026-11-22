#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""产品级单图检测 CLI（P3.1）。

示例::

    D:\\microdefect\\.venv\\Scripts\\python.exe scripts\\detect_product.py ^
        --product-id <PRODUCT_ID> ^
        --image D:\\data\\part.png ^
        --output-dir D:\\data\\out ^
        --root D:\\microdefect\\memory\\products
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.product_detector import ProductDetector, ProductDetectorError  # noqa: E402
from src.core.products import ProductStore, ProductStoreError  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微缺智学：产品级单图检测")
    parser.add_argument("--product-id", required=True, help="已注册产品 ID")
    parser.add_argument("--image", required=True, help="待检测图片路径")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "memory" / "products"),
                        help="产品存储根目录")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        store = ProductStore(args.root)
        detector = ProductDetector(
            args.product_id, store=store, device=args.device,
            batch_size=args.batch_size,
        )
        result = detector.detect(args.image)
        files = result.save(args.output_dir)
    except (ProductDetectorError, ProductStoreError, FileNotFoundError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))
        return 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))
        return 3

    print(json.dumps({
        "ok": True,
        "product_id": result.product_id,
        "product_name": result.product_name,
        "version": result.version,
        "manifest_sha256": result.manifest_sha256,
        "bank_version": result.bank_version,
        "image_path": result.image_path,
        "image_sha256": result.image_sha256,
        "input_image_sha256": result.input_image_sha256,
        "image_score": round(result.image_score, 8),
        "image_threshold": round(result.image_threshold, 8),
        "image_status": result.image_status,
        "pixel_threshold": round(result.pixel_threshold, 8),
        "anomaly_pixels": result.anomaly_pixels,
        "anomaly_ratio": round(result.anomaly_ratio, 8),
        "feature_extractor_version": result.feature_extractor_version,
        "weights_sha256": result.weights_sha256,
        "elapsed_time_s": round(result.elapsed_time, 4),
        "inference_time": round(result.inference_time, 4),
        "files": files,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
