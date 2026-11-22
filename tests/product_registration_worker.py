#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3.2.1 注册并发测试辅助进程（非 test_*.py）。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.product_registration import (  # noqa: E402
    DuplicateRegistrationError,
    RegistrationRequest,
    register_product,
)
from src.core.products import ProductStore  # noqa: E402

TRAIN = PROJECT_ROOT / "datasets/mvtec_ad/bottle/train/good"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--mode", choices=("register", "hold"), default="register")
    parser.add_argument("--fingerprint", default=None)
    parser.add_argument("--release", default=None)
    parser.add_argument("--name", default="垫片-A")
    parser.add_argument("--ref-start", type=int, default=0)
    parser.add_argument("--cal-start", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "hold":
        from src.core.products import _ProductFileLock  # noqa: PLC0415
        lock_path = Path(args.root) / ".locks" / f"registration-{args.fingerprint}.lock"
        lock = _ProductFileLock(lock_path, 120, f"registration:{args.fingerprint[:16]}")
        with lock:
            Path(args.start).write_text("held", encoding="utf-8")
            deadline = time.time() + 120
            while not Path(args.release).is_file():
                if time.time() > deadline:
                    return 4
                time.sleep(0.05)
        Path(args.result).write_text(json.dumps({"status": "released"}), encoding="utf-8")
        return 0
    deadline = time.time() + 120
    while not Path(args.start).is_file():
        if time.time() > deadline:
            Path(args.result).write_text(json.dumps({"status": "error", "error": "start timeout"}),
                                         encoding="utf-8")
            return 3
        time.sleep(0.05)
    request = RegistrationRequest(
        name=args.name,
        category="washer",
        reference_paths=[TRAIN / f"{i:03d}.png" for i in range(args.ref_start, args.ref_start + 5)],
        calibration_paths=[TRAIN / f"{i:03d}.png" for i in range(args.cal_start, args.cal_start + 25)],
    )
    try:
        result = register_product(ProductStore(args.root, lock_timeout=120), request,
                                  device="auto")
        payload = {
            "status": "ok",
            "product_id": result.product.product_id,
            "version": result.version.version,
            "fingerprint": result.version.registration_fingerprint,
        }
    except DuplicateRegistrationError as exc:
        payload = {
            "status": "duplicate",
            "product_id": exc.existing_product_id,
            "version": exc.existing_version,
            "fingerprint": exc.fingerprint,
        }
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    Path(args.result).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
