#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地预训练权重解析测试（修复网页 WeightsMissingError）。"""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import weights  # noqa: E402
from src.core.detector import Detector, DetectorError  # noqa: E402

DATA = PROJECT_ROOT / "datasets/mvtec_ad"


class WeightsTest(unittest.TestCase):
    def test_01_project_weights_present(self) -> None:
        path = weights.weights_file_in(weights.PROJECT_TORCH_HOME)
        self.assertTrue(path.is_file())
        self.assertGreaterEqual(path.stat().st_size, 46_000_000)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            digest, "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
        )

    def test_02_empty_torch_home_falls_back_to_project(self) -> None:
        empty = PROJECT_ROOT / "cache" / "tests" / f"empty_torch_{uuid.uuid4().hex[:8]}"
        empty.mkdir(parents=True, exist_ok=True)
        try:
            with mock.patch.dict(os.environ, {"TORCH_HOME": str(empty)}, clear=False):
                found = weights.ensure_local_weights()
                self.assertEqual(found, weights.weights_file_in(weights.PROJECT_TORCH_HOME))
                self.assertEqual(
                    os.path.normcase(os.environ["TORCH_HOME"]),
                    os.path.normcase(str(weights.PROJECT_TORCH_HOME)),
                )
                detector = Detector("bottle", protocol="group_disjoint", device="cpu")
                self.assertEqual(detector.weights_path, found)
                result = detector.detect(DATA / "bottle/test/good/000.png")
                self.assertFalse(result.is_anomaly)
        finally:
            import shutil
            shutil.rmtree(empty, ignore_errors=True)

    def test_03_missing_weights_clear_error(self) -> None:
        empty = PROJECT_ROOT / "cache" / "tests" / f"no_weights_{uuid.uuid4().hex[:8]}"
        (empty / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)
        try:
            with mock.patch.object(weights, "PROJECT_TORCH_HOME", empty), \
                 mock.patch.object(weights, "DEFAULT_TORCH_HOME", empty), \
                 mock.patch.dict(os.environ, {"TORCH_HOME": str(empty)}, clear=False), \
                 mock.patch.object(weights.torch.hub, "get_dir", return_value=str(empty / "hub")):
                with self.assertRaises(weights.WeightsUnavailableError) as context:
                    weights.ensure_local_weights()
                message = str(context.exception)
                self.assertIn("不会自动联网下载", message)
                self.assertIn("resnet18-f37072fd.pth", message)
                with self.assertRaises(DetectorError):
                    Detector("bottle", protocol="group_disjoint", device="cpu")
        finally:
            import shutil
            shutil.rmtree(empty, ignore_errors=True)

    def test_04_no_download_attempted(self) -> None:
        # allow_download=False 是检测器的硬编码；确认 Detector 不暴露下载入口
        with self.assertRaises(TypeError):
            Detector("bottle", allow_download=True)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main(verbosity=2)
