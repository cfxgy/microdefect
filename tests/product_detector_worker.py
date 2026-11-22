#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3.1.1 并发测试辅助进程（非 test_*.py，不参与自动发现）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.product_detector import ProductDetector  # noqa: E402
from src.core.products import ProductStore  # noqa: E402


def write_result(path: Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--mode", required=True, choices=("detect", "switch", "hold"))
    parser.add_argument("--result", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--lock-timeout", type=float, default=10.0)
    parser.add_argument("--sentinel", default=None)
    parser.add_argument("--release", default=None)
    parser.add_argument("--started", default=None)
    args = parser.parse_args()

    if args.started and args.mode in ("detect", "switch"):
        Path(args.started).write_text("started", encoding="utf-8")

    store = ProductStore(args.root, lock_timeout=args.lock_timeout)
    try:
        if args.mode == "detect":
            detector = ProductDetector(
                args.product_id, store=store, device="cpu", batch_size=1
            )
            result = detector.detect(args.image)
            files = result.save(args.output_dir)
            write_result(args.result, {
                "ok": True,
                "version": result.version,
                "manifest_sha256": result.manifest_sha256,
                "bank_version": result.bank_version,
                "image_score": result.image_score,
                "image_threshold": result.image_threshold,
                "image_status": result.image_status,
                "input_image_sha256": result.input_image_sha256,
                "files": files,
            })
        elif args.mode == "switch":
            record = store.set_current_version(args.product_id, int(args.version))
            write_result(args.result, {
                "ok": True,
                "current_version": record.current_version,
                "status": record.status,
            })
        else:  # hold
            with store._product_lock(args.product_id):
                Path(args.sentinel).write_text("locked", encoding="utf-8")
                deadline = time.time() + 30
                while not Path(args.release).is_file():
                    if time.time() > deadline:
                        write_result(args.result, {"ok": False, "error": "release timeout"})
                        return 3
                    time.sleep(0.05)
            write_result(args.result, {"ok": True, "released": True})
    except Exception as exc:  # noqa: BLE001
        write_result(args.result, {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
