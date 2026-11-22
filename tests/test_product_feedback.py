#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P5 人工反馈与受控更新测试。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.product_detector import ProductDetector  # noqa: E402
from src.core.product_feedback import (  # noqa: E402
    FeedbackConflictError,
    FeedbackError,
    activate_update,
    create_feedback,
    list_feedback,
    list_updates,
    load_update,
    prepare_update,
    recover_pending_updates,
    review_feedback,
)
from src.core.product_registration import RegistrationRequest, register_product  # noqa: E402
from src.core.products import (  # noqa: E402
    ProductStateConflictError,
    ProductStore,
    sha256_file,
)

DATA = PROJECT_ROOT / "datasets/mvtec_ad/bottle"
TRAIN = DATA / "train/good"
TEST_GOOD = DATA / "test/good"
TEST_DEFECT = DATA / "test/broken_large"
TESTS_ROOT = PROJECT_ROOT / "cache/tests"


def _paths(root: Path, start: int, count: int) -> list[Path]:
    return [root / f"{index:03d}.png" for index in range(start, start + count)]


class ProductFeedbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TESTS_ROOT.mkdir(parents=True, exist_ok=True)
        cls.root = TESTS_ROOT / f"p5_feedback_{uuid.uuid4().hex[:8]}"
        cls.root.mkdir(parents=True, exist_ok=False)
        cls.store = ProductStore(cls.root, lock_timeout=60)
        cls.product_a = register_product(
            cls.store,
            RegistrationRequest(
                name="反馈产品A", category="washer",
                reference_paths=_paths(TRAIN, 0, 5),
                calibration_paths=_paths(TRAIN, 5, 25),
            ),
            device="auto",
        )
        cls.product_b = register_product(
            cls.store,
            RegistrationRequest(
                name="反馈产品B", category="gear",
                reference_paths=_paths(TRAIN, 60, 5),
                calibration_paths=_paths(TRAIN, 65, 25),
            ),
            device="auto",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self) -> None:
        self.store.set_current_version(self.product_a.product.product_id, 1)

    def _feedback(self, image: Path, source_id: str, *, product=None, status="normal"):
        product = product or self.product_a
        digest = sha256_file(image)
        return create_feedback(
            self.store,
            product_id=product.product.product_id,
            product_version=1,
            source_type="detection",
            source_id=source_id,
            image_path=image,
            image_sha256=digest,
            image_score=0.02 if status == "normal" else 0.05,
            image_threshold=0.021,
            image_status=status,
        )

    def _confirmed_pair(self, tag: str):
        normal = self._feedback(TEST_GOOD / f"{100 + hash(tag) % 50:03d}.png"
                                if (TEST_GOOD / f"{100 + hash(tag) % 50:03d}.png").is_file()
                                else TEST_GOOD / "005.png", f"{tag}-n", status="normal")
        anomaly = self._feedback(TEST_DEFECT / "005.png", f"{tag}-a", status="anomaly")
        review_feedback(self.store, self.product_a.product.product_id,
                        normal["feedback_id"], "normal", "确认正常")
        review_feedback(self.store, self.product_a.product.product_id,
                        anomaly["feedback_id"], "anomaly", "确认异常")
        return normal, anomaly

    def test_01_feedback_lifecycle_and_dedup(self) -> None:
        normal = self._feedback(TEST_GOOD / "010.png", "life-n", status="normal")
        self.assertEqual(normal["status"], "pending")
        with self.assertRaises(FeedbackConflictError):
            self._feedback(TEST_GOOD / "010.png", "life-n", status="normal")
        reviewed = review_feedback(
            self.store, self.product_a.product.product_id,
            normal["feedback_id"], "normal", "确认",
        )
        self.assertEqual(reviewed["status"], "confirmed")
        self.assertEqual(reviewed["conclusion"], "normal")
        self.assertTrue(any(item["feedback_id"] == normal["feedback_id"]
                            for item in list_feedback(self.store,
                                                      self.product_a.product.product_id)))
        with self.assertRaises(FeedbackError):
            review_feedback(self.store, self.product_a.product.product_id,
                            normal["feedback_id"], "anomaly")

    def test_02_environment_and_reject_are_not_candidates(self) -> None:
        env = self._feedback(TEST_GOOD / "011.png", "env")
        rej = self._feedback(TEST_GOOD / "012.png", "rej")
        review_feedback(self.store, self.product_a.product.product_id,
                        env["feedback_id"], "environment", "光源变化")
        review_feedback(self.store, self.product_a.product.product_id,
                        rej["feedback_id"], "reject", "无法判断")
        self.assertEqual(list_feedback(self.store, self.product_a.product.product_id,
                                       status="confirmed")[0]["status"], "confirmed")
        for item in (env, rej):
            with self.assertRaises(FeedbackError):
                prepare_update(
                    self.store, self.product_a.product.product_id,
                    feedback_ids=[item["feedback_id"]],
                    validation_normal_paths=_paths(TEST_GOOD, 0, 5),
                    validation_anomaly_paths=_paths(TEST_DEFECT, 0, 5),
                )

    def test_03_validation_must_be_disjoint(self) -> None:
        normal, anomaly = self._confirmed_pair("disjoint")
        with self.assertRaises(FeedbackError):
            prepare_update(
                self.store, self.product_a.product.product_id,
                feedback_ids=[normal["feedback_id"], anomaly["feedback_id"]],
                validation_normal_paths=[TEST_GOOD / "005.png"] + _paths(TEST_GOOD, 0, 4),
                validation_anomaly_paths=_paths(TEST_DEFECT, 0, 5),
            )

    def test_04_successful_update_and_explicit_activation(self) -> None:
        normal, anomaly = self._confirmed_pair("success")
        old_version = self.store.get_product(self.product_a.product.product_id).current_version
        report = prepare_update(
            self.store, self.product_a.product.product_id,
            feedback_ids=[normal["feedback_id"], anomaly["feedback_id"]],
            validation_normal_paths=_paths(TEST_GOOD, 0, 5),
            validation_anomaly_paths=_paths(TEST_DEFECT, 0, 5),
        )
        self.assertEqual(report["verdict"], "pass")
        self.assertEqual(self.store.get_product(self.product_a.product.product_id).current_version,
                         old_version)
        activated = activate_update(self.store, self.product_a.product.product_id,
                                    report["update_id"])
        self.assertTrue(activated["activated"])
        self.assertEqual(self.store.get_product(self.product_a.product.product_id).current_version,
                         report["new_version"])
        statuses = {item["feedback_id"]: item["status"]
                    for item in list_feedback(self.store, self.product_a.product.product_id)}
        self.assertEqual(statuses[normal["feedback_id"]], "applied")
        self.assertEqual(statuses[anomaly["feedback_id"]], "applied")
        old_detector = ProductDetector(self.product_a.product.product_id,
                                       store=self.store, version=report["old_version"],
                                       device="cpu")
        self.assertEqual(old_detector.detect(TEST_GOOD / "000.png").version,
                         report["old_version"])

    def test_05_insufficient_validation_cannot_activate(self) -> None:
        normal, anomaly = self._confirmed_pair("short")
        old_version = self.store.get_product(self.product_a.product.product_id).current_version
        report = prepare_update(
            self.store, self.product_a.product.product_id,
            feedback_ids=[normal["feedback_id"], anomaly["feedback_id"]],
            validation_normal_paths=[TEST_GOOD / "000.png"],
            validation_anomaly_paths=[TEST_DEFECT / "000.png"],
        )
        self.assertEqual(report["verdict"], "inconclusive")
        self.assertEqual(self.store.get_product(self.product_a.product.product_id).current_version,
                         old_version)
        with self.assertRaises(FeedbackError):
            activate_update(self.store, self.product_a.product.product_id, report["update_id"])
        statuses = {item["feedback_id"]: item["status"]
                    for item in list_feedback(self.store, self.product_a.product.product_id)}
        self.assertEqual(statuses[normal["feedback_id"]], "validation_failed")

    def test_06_product_isolation_and_persistence(self) -> None:
        b = self._feedback(TEST_GOOD / "013.png", "b-feedback", product=self.product_b)
        self.assertFalse(any(item["feedback_id"] == b["feedback_id"]
                             for item in list_feedback(self.store,
                                                       self.product_a.product.product_id)))
        reopened = ProductStore(self.root)
        records = list_feedback(reopened, self.product_b.product.product_id)
        self.assertTrue(any(item["feedback_id"] == b["feedback_id"] for item in records))
        self.assertTrue(list_updates(reopened, self.product_a.product.product_id))

    def test_07_image_hash_mismatch_rejected(self) -> None:
        with self.assertRaises(FeedbackError):
            create_feedback(
                self.store,
                product_id=self.product_a.product.product_id,
                product_version=1,
                source_type="detection",
                source_id="hash-bad",
                image_path=TEST_GOOD / "014.png",
                image_sha256="0" * 64,
            )

    def _prepare_pass(self, tag: str) -> dict:
        normal, anomaly = self._confirmed_pair(tag)
        return prepare_update(
            self.store, self.product_a.product.product_id,
            feedback_ids=[normal["feedback_id"], anomaly["feedback_id"]],
            validation_normal_paths=_paths(TEST_GOOD, 0, 5),
            validation_anomaly_paths=_paths(TEST_DEFECT, 0, 5),
        )

    def test_08_report_tamper_rejected(self) -> None:
        report = self._prepare_pass("tamper")
        path = (self.store.product_directory(self.product_a.product.product_id)
                / "feedback/updates" / report["update_id"] / "report.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["verdict"] = "fail"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.assertRaises(FeedbackError):
            load_update(self.store, self.product_a.product.product_id, report["update_id"])
        with self.assertRaises(FeedbackError):
            activate_update(self.store, self.product_a.product.product_id, report["update_id"])
        self.assertEqual(self.store.get_product(self.product_a.product.product_id).current_version, 1)

    def test_09_report_sidecar_missing_rejected(self) -> None:
        report = self._prepare_pass("sidecar")
        path = (self.store.product_directory(self.product_a.product.product_id)
                / "feedback/updates" / report["update_id"] / "report.sha256")
        path.unlink()
        with self.assertRaises(FeedbackError):
            activate_update(self.store, self.product_a.product.product_id, report["update_id"])
        self.assertEqual(self.store.get_product(self.product_a.product.product_id).current_version, 1)

    def test_10_update_idempotency(self) -> None:
        normal, anomaly = self._confirmed_pair("idem")
        first = prepare_update(
            self.store, self.product_a.product.product_id,
            feedback_ids=[normal["feedback_id"], anomaly["feedback_id"]],
            validation_normal_paths=_paths(TEST_GOOD, 0, 5),
            validation_anomaly_paths=_paths(TEST_DEFECT, 0, 5),
        )
        before = len(self.store.list_versions(self.product_a.product.product_id))
        second = prepare_update(
            self.store, self.product_a.product.product_id,
            feedback_ids=[normal["feedback_id"], anomaly["feedback_id"]],
            validation_normal_paths=_paths(TEST_GOOD, 0, 5),
            validation_anomaly_paths=_paths(TEST_DEFECT, 0, 5),
        )
        self.assertEqual(first["update_id"], second["update_id"])
        self.assertEqual(before, len(self.store.list_versions(self.product_a.product.product_id)))

    def test_11_fixed_validation_tamper_blocks_activation(self) -> None:
        report = self._prepare_pass("fixed")
        update_dir = (self.store.product_directory(self.product_a.product.product_id)
                      / "feedback/updates" / report["update_id"])
        fixed = update_dir / report["validation_files"][0]["rel_path"]
        fixed.write_bytes(b"tampered")
        with self.assertRaises(FeedbackError):
            activate_update(self.store, self.product_a.product.product_id, report["update_id"])
        self.assertEqual(self.store.get_product(self.product_a.product.product_id).current_version, 1)

    def test_12_activation_recovers_after_post_switch_failure(self) -> None:
        report = self._prepare_pass("recover")
        with mock.patch("src.core.product_feedback._complete_activation_locked",
                        side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                activate_update(self.store, self.product_a.product.product_id, report["update_id"])
        self.assertEqual(self.store.get_product(self.product_a.product.product_id).current_version,
                         report["new_version"])
        recovered = recover_pending_updates(self.store, self.product_a.product.product_id)
        self.assertTrue(recovered)
        loaded = load_update(self.store, self.product_a.product.product_id, report["update_id"])
        self.assertTrue(loaded["activated"])
        for item in list_feedback(self.store, self.product_a.product.product_id):
            if item["feedback_id"] in report["feedback_ids"]:
                self.assertEqual(item["status"], "applied")

    def test_13_current_change_after_validation_rejects_activation(self) -> None:
        report = self._prepare_pass("current-change")
        register_product(
            self.store,
            RegistrationRequest(
                name="ignored", product_id=self.product_a.product.product_id,
                reference_paths=_paths(TRAIN, 30, 5),
                calibration_paths=_paths(TRAIN, 35, 25),
            ),
            device="auto",
        )
        latest = max(v.version for v in self.store.list_versions(self.product_a.product.product_id))
        self.store.set_current_version(self.product_a.product.product_id, latest)
        with self.assertRaises(FeedbackError):
            activate_update(self.store, self.product_a.product.product_id, report["update_id"])
        self.assertEqual(self.store.get_product(self.product_a.product.product_id).current_version,
                         latest)

    def test_14_current_cas_conflict_keeps_old_current(self) -> None:
        report = self._prepare_pass("cas")
        with mock.patch.object(ProductStore, "compare_and_set_current",
                               side_effect=ProductStateConflictError("concurrent switch")):
            with self.assertRaises(FeedbackError):
                activate_update(self.store, self.product_a.product.product_id, report["update_id"])
        self.assertEqual(self.store.get_product(self.product_a.product.product_id).current_version, 1)

    def test_15_batch_and_drift_source_feedback(self) -> None:
        for source_type, source_id, image in (
            ("batch", "batch-x", TEST_GOOD / "010.png"),
            ("drift", "drift-x", TEST_GOOD / "011.png"),
        ):
            feedback = create_feedback(
                self.store,
                product_id=self.product_a.product.product_id,
                product_version=1,
                source_type=source_type,
                source_id=source_id,
                image_path=image,
                image_sha256=sha256_file(image),
                image_score=0.01,
                image_status="normal",
            )
            self.assertEqual(feedback["source_type"], source_type)
            self.assertEqual(feedback["source_id"], source_id)
            self.assertEqual(feedback["status"], "pending")


if __name__ == "__main__":
    unittest.main(verbosity=2)
