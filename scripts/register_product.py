#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""产品注册命令行入口（P2）。

示例::

    D:\\microdefect\\.venv\\Scripts\\python.exe scripts\\register_product.py ^
        --name "垫片-A" --category washer ^
        --reference-dir D:\\data\\washer\\reference ^
        --calibration-dir D:\\data\\washer\\calibration ^
        --calibration-count 25 ^
        --defect-dir D:\\data\\washer\\defects ^
        --test-normal-dir D:\\data\\washer\\test_normal ^
        --test-defect-dir D:\\data\\washer\\test_defect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.product_registration import (  # noqa: E402
    DEFAULT_CALIBRATION_COUNT,
    RegistrationError,
    RegistrationRequest,
    collect_images,
    register_product,
)
from src.core.products import ProductStore, ProductStoreError  # noqa: E402


def _optional_images(directory: str | None, role: str) -> list[Path]:
    if not directory:
        return []
    return collect_images(directory, role=role)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微缺智学：新产品注册")
    parser.add_argument("--name", required=True, help="产品名称")
    parser.add_argument("--category", default="", help="产品类别/型号（可选）")
    parser.add_argument("--product-id", default=None,
                        help="已有产品 ID：注册为新版本；不传则创建新产品")
    parser.add_argument("--reference-dir", required=True, help="正常参考图目录（≥5 张）")
    parser.add_argument("--calibration-dir", required=True, help="独立正常校准图目录")
    parser.add_argument("--calibration-count", type=int, default=DEFAULT_CALIBRATION_COUNT,
                        help=f"校准图数量（默认 {DEFAULT_CALIBRATION_COUNT}，允许 20..200）")
    parser.add_argument("--defect-dir", default=None, help="可选：缺陷样本目录")
    parser.add_argument("--test-normal-dir", default=None, help="可选：独立测试集-正常")
    parser.add_argument("--test-defect-dir", default=None, help="可选：独立测试集-异常")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "memory" / "products"),
                        help="产品存储根目录")
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-activate", action="store_true",
                        help="只创建版本，不切换当前版本")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        reference = collect_images(args.reference_dir, role="reference")
        calibration = collect_images(args.calibration_dir, role="calibration")
        defects = _optional_images(args.defect_dir, "defect")
        test_normal = _optional_images(args.test_normal_dir, "test_normal")
        test_defect = _optional_images(args.test_defect_dir, "test_defect")
        request = RegistrationRequest(
            name=args.name,
            category=args.category,
            product_id=args.product_id,
            reference_paths=reference,
            calibration_paths=calibration,
            calibration_count=args.calibration_count,
            defect_paths=defects,
            test_normal_paths=test_normal,
            test_defect_paths=test_defect,
            activate=not args.no_activate,
        )
        store = ProductStore(args.root)
        result = register_product(
            store, request, device=args.device, batch_size=args.batch_size
        )
    except (RegistrationError, ProductStoreError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))
        return 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))
        return 3

    payload = {"ok": True, **result.summary()}
    payload["batch_dir"] = str(store.product_directory(result.product.product_id))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
