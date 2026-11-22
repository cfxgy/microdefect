#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P4-2.1 阶段 4：协议校验与源图像隔离审计测试。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift_calibration.isolation import (  # noqa: E402
    IsolationError,
    assert_disjoint,
    assert_subset,
)
from src.drift_calibration.protocol import (  # noqa: E402
    ALPHA,
    FUSION_AVAILABLE,
    FUSION_MIN_P,
    N_CALIBRATION,
    P_DENOMINATOR,
    PROTOCOL_ID,
    TIE_RULE,
    ProtocolMismatchError,
    protocol_fingerprint,
    protocol_params,
    verify_frozen_files,
)

FROZEN_DIR = PROJECT_ROOT / "experiments" / "results" / "drift_monitoring_calibration"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProtocolConstantsTest(unittest.TestCase):
    def test_01_frozen_constants(self) -> None:
        self.assertEqual(N_CALIBRATION, 25)
        self.assertEqual(P_DENOMINATOR, 26)
        self.assertEqual(TIE_RULE, ">=")
        self.assertAlmostEqual(ALPHA, 0.05)
        self.assertFalse(FUSION_AVAILABLE)
        self.assertGreater(FUSION_MIN_P, ALPHA)  # 2/26 > 0.05

    def test_02_fingerprint_stable_and_informative(self) -> None:
        first = protocol_fingerprint()
        second = protocol_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        params = protocol_params()
        self.assertEqual(params["alpha"], 0.05)
        self.assertEqual(params["n_calibration"], 25)
        self.assertEqual(params["primary_channel"], "A")
        self.assertIn("p = (1 + count(cal_score >= test_score)) / 26", params["p_value_formula"])
        self.assertEqual(params["protocol_id"], PROTOCOL_ID)


class FrozenFileVerificationTest(unittest.TestCase):
    def test_03_real_frozen_files_match_record(self) -> None:
        report = verify_frozen_files(FROZEN_DIR)
        self.assertTrue(report["all_match"])
        self.assertGreaterEqual(len(report["files"]), 4)
        for name, item in report["files"].items():
            self.assertTrue(item["match"], name)

    def test_04_tampered_record_fails(self) -> None:
        tmp = PROJECT_ROOT / f"tmp_stage4_proto_{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            target = tmp / "protocol_design.md"
            target.write_text("tampered", encoding="utf-8")
            record = {
                "version": "test",
                "files_v2": {
                    "protocol_design.md": {
                        "sha256": "0" * 64,
                        "size_bytes": target.stat().st_size,
                    }
                },
            }
            (tmp / "protocol_design_sha256.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            with self.assertRaises(ProtocolMismatchError):
                verify_frozen_files(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_05_missing_file_fails(self) -> None:
        tmp = PROJECT_ROOT / f"tmp_stage4_proto_{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            record = {"version": "test", "files_v2": {"missing.md": {"sha256": "f" * 64}}}
            (tmp / "protocol_design_sha256.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            with self.assertRaises(ProtocolMismatchError):
                verify_frozen_files(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_06_missing_record_fails(self) -> None:
        tmp = PROJECT_ROOT / f"tmp_stage4_proto_{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            with self.assertRaises(ProtocolMismatchError):
                verify_frozen_files(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class IsolationAuditTest(unittest.TestCase):
    def test_07_disjoint_passes(self) -> None:
        result = assert_disjoint(
            [f"Cr_{i}.bmp" for i in range(25)],
            [f"Cr_{i}.bmp" for i in range(25, 50)],
            [f"Cr_{i}.bmp" for i in range(100, 120)],
        )
        self.assertTrue(result["disjoint"])
        self.assertEqual(result["reference_calibration_overlap"], 0)

    def test_08_reference_calibration_overlap_fails(self) -> None:
        with self.assertRaises(IsolationError):
            assert_disjoint(["a", "b", "c"], ["c", "d"])

    def test_09_calibration_test_overlap_fails(self) -> None:
        with self.assertRaises(IsolationError):
            assert_disjoint(["a"], ["b", "c"], ["c"])

    def test_10_duplicate_ids_fail(self) -> None:
        with self.assertRaises(IsolationError):
            assert_disjoint(["a", "a"], ["b"])

    def test_11_subset_check(self) -> None:
        assert_subset(["a", "b"], ["a", "b", "c"])
        with self.assertRaises(IsolationError):
            assert_subset(["a", "z"], ["a", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
