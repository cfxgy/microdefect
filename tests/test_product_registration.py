#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P2 产品注册核心流程测试。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly.mvtec_protocol_scoring import (  # noqa: E402
    score_image_cosine,
    score_spatial_cosine,
)
from src.core.product_bank import load_product_bank  # noqa: E402
from src.core.product_calibration import (  # noqa: E402
    calibrate_image_scores,
    calibrate_pixel_scores,
)
from src.core.product_registration import (  # noqa: E402
    DuplicateRegistrationError,
    RegistrationError,
    RegistrationRequest,
    compute_registration_fingerprint,
    find_registration_duplicate,
    register_product,
)
from src.core.products import ProductLockTimeoutError, ProductStore, sha256_file  # noqa: E402
from src.features.mvtec_spatial_extractor import MvtecSpatialExtractor  # noqa: E402
from src.core.weights import ensure_local_weights  # noqa: E402
from src.features import extractor as p2  # noqa: E402
from src.features.mvtec_spatial_extractor import weights_sha256  # noqa: E402

DATA = PROJECT_ROOT / "datasets/mvtec_ad/bottle"
TRAIN = DATA / "train/good"
TEST_GOOD = DATA / "test/good"
TEST_DEFECT = DATA / "test/broken_large"
TESTS_ROOT = PROJECT_ROOT / "cache/tests"


def _paths(root: Path, start: int, count: int) -> list[Path]:
    return [root / f"{index:03d}.png" for index in range(start, start + count)]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_legacy_version(
    store: ProductStore,
    root: Path,
    *,
    name: str,
    calibration_count: int = 25,
    reference_count: int = 5,
    include_calibration: bool = True,
    source_count_override: int | None = None,
    registration_fingerprint: str = "",
    status: str = "active",
) -> tuple[str, int]:
    """构造 P1 风格旧版本：source_samples + registration.json，无新指纹字段。"""
    product = store.create_product(name, "washer")
    samples = []
    for index in range(reference_count):
        samples.append({"role": "reference", "rel_path": f"legacy/ref/{index}.png",
                        "sha256": _hash(f"{name}-ref-{index}")})
    if include_calibration:
        count = calibration_count if source_count_override is None else source_count_override
        for index in range(count):
            samples.append({"role": "calibration", "rel_path": f"legacy/cal/{index}.png",
                            "sha256": _hash(f"{name}-cal-{index}")})
    ensure_local_weights()
    version = store.create_version(
        product.product_id,
        threshold=0.1,
        calibration_metadata={
            "n_calibration": calibration_count,
            "image_quantile": 0.95,
            "pixel_quantile": 0.99,
        },
        feature_extractor_version=p2.FEATURE_VERSION,
        weights_sha256=weights_sha256(),
        source_samples=samples,
        registration_fingerprint=registration_fingerprint,
    )
    vdir = store.version_directory(product.product_id, version.version)
    (vdir / "registration.json").write_text(json.dumps({
        "n_reference": reference_count,
        "n_calibration": calibration_count,
        "n_defect": 0,
        "n_test_normal": 0,
        "n_test_defect": 0,
    }, ensure_ascii=False), encoding="utf-8")
    manifest = store._build_manifest(vdir, product.product_id, version.version)
    (vdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (vdir / "manifest.sha256").write_text(
        sha256_file(vdir / "manifest.json") + "\n", encoding="ascii")
    store.set_current_version(product.product_id, version.version)
    if status != "active":
        store.update_product_status(product.product_id, status)
    return product.product_id, version.version


class RegistrationTest(unittest.TestCase):
    def setUp(self) -> None:
        TESTS_ROOT.mkdir(parents=True, exist_ok=True)
        self.root = TESTS_ROOT / f"p2_register_{uuid.uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.store = ProductStore(self.root, lock_timeout=60)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _request(self, *, start: int = 0, calibration: int = 25, **kwargs) -> RegistrationRequest:
        return RegistrationRequest(
            name=kwargs.pop("name", "垫片-A"),
            category=kwargs.pop("category", "washer"),
            reference_paths=_paths(TRAIN, start, 5),
            calibration_paths=_paths(TRAIN, start + 5, calibration),
            calibration_count=calibration,
            **kwargs,
        )

    def _register(self, request: RegistrationRequest, **kwargs):
        return register_product(self.store, request, device="auto",
                                batch_size=8, **kwargs)

    # 1
    def test_01_five_reference_default_calibration(self) -> None:
        result = self._register(self._request())
        self.assertEqual(result.product.status, "active")
        self.assertEqual(result.version.version, 1)
        self.assertAlmostEqual(result.image_threshold, result.version.threshold)
        self.assertGreater(result.pixel_threshold, 0)
        self.assertFalse(result.low_sample_warning)
        self.assertEqual(result.n_reference, 5)
        self.assertEqual(result.n_calibration, 25)
        pid = result.product.product_id
        self.assertTrue(self.store.validate_product(pid)["ok"])
        version_dir = self.root / pid / "versions" / "v001"
        for name in (
            "reference/global_reference/features.npy",
            "reference/global_reference/metadata.json",
            "reference/spatial_reference/features.npy",
            "calibration/thresholds.json",
            "calibration/image_scores.csv",
            "calibration/members.csv",
            "registration.json",
        ):
            self.assertTrue((version_dir / name).is_file(), name)
        thresholds = json.loads((version_dir / "calibration" / "thresholds.json").read_text(encoding="utf-8"))
        self.assertEqual(thresholds["n_calibration"], 25)
        self.assertEqual(thresholds["product_id"], pid)
        self.assertFalse(thresholds["sample_note"]["fpr_guarantee"])

    # 2
    def test_02_reference_below_five_rejected(self) -> None:
        request = self._request()
        request.reference_paths = request.reference_paths[:4]
        with self.assertRaises(RegistrationError):
            self._register(request)
        self.assertEqual(self.store.list_products(), [])

    # 3
    def test_03_calibration_count_configurable_and_minimum(self) -> None:
        result = self._register(self._request(calibration=20))
        self.assertTrue(result.low_sample_warning)
        self.assertEqual(result.n_calibration, 20)

        for bad in (19, 10, 201):
            request = self._request(calibration=bad)
            with self.assertRaises(RegistrationError):
                self._register(request)

    # 4
    def test_04_duplicate_reference_calibration_rejected(self) -> None:
        request = self._request()
        request.calibration_paths[0] = request.reference_paths[0]
        with self.assertRaises(RegistrationError) as context:
            self._register(request)
        self.assertIn("相同文件内容", str(context.exception))

        request = self._request()
        request.reference_paths[1] = request.reference_paths[0]
        with self.assertRaises(RegistrationError):
            self._register(request)

    # 5
    def test_05_calibration_count_mismatch_rejected(self) -> None:
        request = self._request()
        request.calibration_paths = request.calibration_paths[:24]
        with self.assertRaises(RegistrationError):
            self._register(request)

    # 6
    def test_06_optional_defect_and_test(self) -> None:
        result = self._register(self._request())
        version_dir = self.root / result.product.product_id / "versions" / "v001"
        self.assertFalse((version_dir / "defects").exists())
        self.assertFalse(result.independent_test_provided)
        self.assertFalse(result.independent_test_validated)
        self.assertTrue(any("未提供独立测试集" in w for w in result.warnings))

    # 7
    def test_07_independent_test_provided_but_not_validated(self) -> None:
        request = self._request()
        request.test_normal_paths = _paths(TEST_GOOD, 0, 5)
        request.test_defect_paths = _paths(TEST_DEFECT, 0, 5)
        result = self._register(request)
        self.assertTrue(result.independent_test_provided)
        self.assertFalse(result.independent_test_validated)
        version_dir = self.root / result.product.product_id / "versions" / "v001"
        self.assertTrue((version_dir / "tests" / "members.csv").is_file())
        metadata = json.loads((version_dir / "tests" / "test_metadata.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["independent_test_validated"])

    # 8
    def test_08_test_overlap_rejected(self) -> None:
        request = self._request()
        request.test_normal_paths = [request.reference_paths[0]]
        with self.assertRaises(RegistrationError) as context:
            self._register(request)
        self.assertIn("相同文件内容", str(context.exception))

    # 9
    def test_09_defect_samples_stored_but_not_used_for_scoring(self) -> None:
        request = self._request(defect_paths=_paths(TEST_DEFECT, 0, 5))
        result = self._register(request)
        version_dir = self.root / result.product.product_id / "versions" / "v001"
        self.assertTrue((version_dir / "defects" / "members.csv").is_file())
        meta = json.loads((version_dir / "defects" / "metadata.json").read_text(encoding="utf-8"))
        self.assertFalse(meta["used_for_scoring"])

    # 10
    def test_10_new_version_keeps_old_untouched(self) -> None:
        first = self._register(self._request())
        old_dir = self.root / first.product.product_id / "versions" / "v001"
        old_hashes = {p.relative_to(old_dir).as_posix(): p.read_bytes() for p in old_dir.rglob("*") if p.is_file()}
        second_request = self._request(start=30)
        second_request.product_id = first.product.product_id
        second = self._register(second_request)
        self.assertEqual(second.version.version, 2)
        self.assertEqual(self.store.get_product(first.product.product_id).current_version, 2)
        new_hashes = {p.relative_to(old_dir).as_posix(): p.read_bytes() for p in old_dir.rglob("*") if p.is_file()}
        self.assertEqual(old_hashes, new_hashes)

    # 11
    def test_11_product_b_does_not_change_a(self) -> None:
        a = self._register(self._request())
        a_record = self.store.get_product(a.product.product_id)
        a_version = self.store.get_version(a.product.product_id, 1)
        a_manifest = a_version.manifest_sha256
        b = self._register(RegistrationRequest(
            name="齿轮-B", category="gear",
            reference_paths=_paths(TRAIN, 60, 5),
            calibration_paths=_paths(TRAIN, 65, 25),
        ))
        self.assertNotEqual(a.product.product_id, b.product.product_id)
        a_after = self.store.get_product(a.product.product_id)
        self.assertEqual(a_after.current_version, a_record.current_version)
        self.assertEqual(self.store.get_version(a.product.product_id, 1).manifest_sha256, a_manifest)
        self.assertTrue(self.store.validate_product(a.product.product_id)["ok"])

    # 12
    def test_12_failure_does_not_change_active(self) -> None:
        first = self._register(self._request())
        pid = first.product.product_id
        before_manifest = self.store.get_version(pid, 1).manifest_sha256
        before_current = self.store.get_product(pid).current_version

        with mock.patch("src.core.product_registration.save_product_bank",
                        side_effect=OSError("disk full")):
            with self.assertRaises(Exception):
                self._register(self._request(start=30))

        record = self.store.get_product(pid)
        self.assertEqual(record.current_version, before_current)
        self.assertEqual(self.store.get_version(pid, 1).manifest_sha256, before_manifest)
        self.assertEqual(len(self.store.list_versions(pid)), 1)
        self.assertTrue(self.store.validate_product(pid)["ok"])
        self.assertGreaterEqual(len(self.store.list_staging()), 1)

    # 13
    def test_13_manifest_failure_leaves_no_version(self) -> None:
        pid = self.store.create_product("中断产品").product_id
        request = self._request()
        request.product_id = pid
        with mock.patch("src.core.products.ProductStore._build_manifest",
                        side_effect=OSError("manifest write failed")):
            with self.assertRaises(Exception):
                self._register(request)
        self.assertEqual(self.store.list_versions(pid), [])
        self.assertIsNone(self.store.get_product(pid).current_version)
        self.assertGreaterEqual(len(self.store.list_staging()), 1)

    # 14
    def test_14_restart_and_reload(self) -> None:
        result = self._register(self._request())
        reopened = ProductStore(self.root)
        product = reopened.get_product(result.product.product_id)
        version = reopened.get_version(result.product.product_id, 1)
        self.assertEqual(product.current_version, 1)
        self.assertEqual(version.version, 1)
        self.assertTrue(reopened.validate_product(product.product_id)["ok"])

    # 15
    def test_15_concurrent_registration_same_product(self) -> None:
        product = self.store.create_product("并发注册产品")
        pid = product.product_id
        work = TESTS_ROOT / f"p2_concurrent_{uuid.uuid4().hex[:8]}"
        try:
            for label, start in (("a", 0), ("b", 30)):
                ref_dir = work / label / "ref"
                cal_dir = work / label / "cal"
                ref_dir.mkdir(parents=True)
                cal_dir.mkdir(parents=True)
                for index in range(5):
                    shutil.copyfile(TRAIN / f"{start + index:03d}.png", ref_dir / f"{index:03d}.png")
                for index in range(25):
                    shutil.copyfile(TRAIN / f"{start + 5 + index:03d}.png", cal_dir / f"{index:03d}.png")
            env = dict(**__import__("os").environ)
            env["PYTHONPATH"] = str(PROJECT_ROOT)
            commands = []
            for label in ("a", "b"):
                commands.append(subprocess.Popen(
                    [sys.executable, str(PROJECT_ROOT / "scripts/register_product.py"),
                     "--name", "并发注册产品", "--product-id", pid,
                     "--reference-dir", str(work / label / "ref"),
                     "--calibration-dir", str(work / label / "cal"),
                     "--calibration-count", "25", "--root", str(self.root),
                     "--device", "cpu"],
                    cwd=str(PROJECT_ROOT), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace",
                ))
            outputs = []
            for command in commands:
                stdout, stderr = command.communicate(timeout=600)
                self.assertEqual(command.returncode, 0, stderr[-1000:])
                outputs.append(json.loads(stdout[stdout.index("{"):]))
            self.assertTrue(all(item["ok"] for item in outputs), outputs)
            versions = self.store.list_versions(pid)
            self.assertEqual(sorted(v.version for v in versions), [1, 2])
            for version in versions:
                self.assertTrue(self.store.validate_version(pid, version.version)["ok"])
            record = self.store.get_product(pid)
            self.assertIn(record.current_version, (1, 2))
            self.assertTrue(self.store.validate_product(pid)["ok"])
        finally:
            shutil.rmtree(work, ignore_errors=True)


    # 16
    def test_16_single_active_version_semantics(self) -> None:
        first = self._register(self._request(start=0))
        pid = first.product.product_id
        self.assertEqual([v.status for v in self.store.list_versions(pid)], ["active"])

        second = self._register(RegistrationRequest(
            name="ignored", product_id=pid,
            reference_paths=_paths(TRAIN, 30, 5),
            calibration_paths=_paths(TRAIN, 35, 25),
        ))
        self.assertEqual(second.version.version, 2)
        statuses = {v.version: v.status for v in self.store.list_versions(pid)}
        self.assertEqual(statuses, {1: "superseded", 2: "active"})

        third = self._register(RegistrationRequest(
            name="ignored", product_id=pid,
            reference_paths=_paths(TRAIN, 60, 5),
            calibration_paths=_paths(TRAIN, 65, 25),
        ))
        self.assertEqual(third.version.version, 3)
        statuses = {v.version: v.status for v in self.store.list_versions(pid)}
        self.assertEqual(statuses, {1: "superseded", 2: "superseded", 3: "active"})
        self.assertEqual(sum(1 for s in statuses.values() if s == "active"), 1)
        self.assertTrue(self.store.validate_product(pid)["ok"])

    # 17
    def test_17_legacy_multiple_raw_active_fields_derived(self) -> None:
        first = self._register(self._request(start=0))
        pid = first.product.product_id
        self._register(RegistrationRequest(
            name="ignored", product_id=pid,
            reference_paths=_paths(TRAIN, 30, 5),
            calibration_paths=_paths(TRAIN, 35, 25),
        ))
        v1 = self.store.get_version(pid, 1)
        v1.status = "active"  # 人为制造历史 active 字段
        self.store._write_version_file(pid, v1)
        raw = json.loads(
            (self.root / pid / "versions" / "v001" / "version.json").read_text(encoding="utf-8")
        )
        self.assertEqual(raw["status"], "active")
        self.assertEqual(self.store.get_version(pid, 1).status, "superseded")
        statuses = [v.status for v in self.store.list_versions(pid)]
        self.assertEqual(statuses.count("active"), 1)
        self.assertTrue(self.store.validate_product(pid)["ok"])

    # 18
    def test_18_validate_rejects_pointer_mismatch(self) -> None:
        result = self._register(self._request())
        pid = result.product.product_id
        product_file = self.store.product_directory(pid) / "product.json"
        payload = json.loads(product_file.read_text(encoding="utf-8"))
        payload["current_version"] = 99
        product_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        validation = self.store.validate_product(pid)
        self.assertFalse(validation["ok"])
        self.assertTrue(any("current_version" in p or "99" in p for p in validation["problems"]))

    # 19
    def test_19_switch_and_rollback_keep_version_files_unchanged(self) -> None:
        first = self._register(self._request(start=0))
        pid = first.product.product_id
        self._register(RegistrationRequest(
            name="ignored", product_id=pid,
            reference_paths=_paths(TRAIN, 30, 5),
            calibration_paths=_paths(TRAIN, 35, 25),
        ))

        def hashes() -> dict:
            base = self.root / pid / "versions"
            return {
                p.relative_to(base).as_posix(): sha256_file(p)
                for p in base.rglob("*") if p.is_file()
            }

        before = hashes()
        for target in (1, 2, 1, 2):
            self.store.set_current_version(pid, target)
            statuses = {v.version: v.status for v in self.store.list_versions(pid)}
            self.assertEqual(statuses[target], "active")
            self.assertEqual(sum(1 for s in statuses.values() if s == "active"), 1)
        self.assertEqual(hashes(), before)
        self.assertTrue(self.store.validate_product(pid)["ok"])

    # 20
    def test_20_post_commit_validation_failure_no_active_orphan(self) -> None:
        first = self._register(self._request(start=0))
        pid = first.product.product_id
        request = self._request(start=30)
        request.product_id = pid
        with mock.patch.object(
            ProductStore, "validate_version",
            return_value={"ok": False, "problems": ["injected"]},
        ):
            with self.assertRaises(RegistrationError):
                self._register(request)
        record = self.store.get_product(pid)
        self.assertEqual(record.current_version, 1)
        statuses = {v.version: v.status for v in self.store.list_versions(pid)}
        self.assertEqual(statuses, {1: "active", 2: "superseded"})
        self.assertTrue(self.store.validate_product(pid)["ok"])

    # 21
    def test_21_activation_failure_keeps_previous_current(self) -> None:
        first = self._register(self._request(start=0))
        pid = first.product.product_id
        request = self._request(start=30)
        request.product_id = pid
        with mock.patch.object(
            ProductStore, "_activate_version_pointer_locked",
            side_effect=OSError("activation failed"),
        ):
            with self.assertRaises(Exception):
                self._register(request)
        record = self.store.get_product(pid)
        self.assertEqual(record.current_version, 1)
        statuses = {v.version: v.status for v in self.store.list_versions(pid)}
        self.assertEqual(statuses, {1: "active", 2: "superseded"})
        self.assertTrue(self.store.validate_product(pid)["ok"])

    # 22
    def test_22_pixel_metadata_and_threshold_recompute(self) -> None:
        result = self._register(self._request())
        pid = result.product.product_id
        version_dir = self.root / pid / "versions" / "v001"
        thresholds = json.loads(
            (version_dir / "calibration" / "thresholds.json").read_text(encoding="utf-8")
        )
        calibration_paths = _paths(TRAIN, 5, 25)
        expected_pixels = 0
        shapes = []
        for path in calibration_paths:
            with Image.open(path) as handle:
                width, height = handle.size
            expected_pixels += width * height
            shapes.append([height, width])
        self.assertEqual(thresholds["pixel_n"], expected_pixels)
        self.assertEqual(thresholds["pixel_dtype"], "float32")
        self.assertEqual(thresholds["pixel_threshold_dtype"], "float32")
        self.assertEqual(thresholds["pixel_stats_dtype"], "float64")
        self.assertEqual(thresholds["pixel_quantile"], 0.99)
        self.assertEqual(thresholds["pixel_quantile_method"], "linear")
        self.assertFalse(thresholds["pixel_independent_samples"])
        self.assertFalse(thresholds["fpr_guarantee"])
        self.assertEqual(thresholds["heatmap_upsample"],
                         {"mode": "bilinear", "align_corners": False})
        self.assertEqual(thresholds["calibration_image_sha256"],
                         [sha256_file(p) for p in calibration_paths])
        self.assertEqual(thresholds["pixel_map_shapes"], shapes)
        self.assertEqual(len(thresholds["pixel_scores_sha256"]), 64)

        global_bank = load_product_bank(version_dir / "reference" / "global_reference")
        spatial_bank = load_product_bank(version_dir / "reference" / "spatial_reference")
        extractor = MvtecSpatialExtractor(device="auto", batch_size=8, allow_download=False)
        cal_global, cal_spatial = extractor.extract_paths_both([str(p) for p in calibration_paths])
        device = str(getattr(extractor, "device", "cpu"))
        image_scores = score_image_cosine(
            cal_global, global_bank["features"], device=device
        ).distances
        image_cal = calibrate_image_scores(image_scores, quantile=0.95)
        self.assertAlmostEqual(image_cal.threshold, thresholds["image_threshold"], places=6)

        chunks = []
        for index, path in enumerate(calibration_paths):
            with Image.open(path) as handle:
                width, height = handle.size
            heatmap = score_spatial_cosine(
                cal_spatial[index], spatial_bank["features"], (height, width), device=device
            ).heatmaps[0]
            chunks.append(np.asarray(heatmap, dtype=np.float32).ravel())
        scores = np.concatenate(chunks).astype(np.float32, copy=False)
        pixel_cal = calibrate_pixel_scores(scores, quantile=0.99)
        self.assertEqual(pixel_cal.n, expected_pixels)
        self.assertAlmostEqual(pixel_cal.threshold, thresholds["pixel_threshold"], places=6)
        digest = hashlib.sha256(np.ascontiguousarray(scores).tobytes(order="C")).hexdigest()
        self.assertEqual(digest, thresholds["pixel_scores_sha256"])

    # 23
    def test_23_legacy_version_without_new_calibration_fields(self) -> None:
        product = self.store.create_product("旧版本兼容")
        version = self.store.create_version(product.product_id)
        self.store.set_current_version(product.product_id, version.version)
        loaded = self.store.get_version(product.product_id, version.version)
        self.assertEqual(loaded.status, "active")
        self.assertEqual(loaded.calibration_metadata, {})
        self.assertTrue(self.store.validate_product(product.product_id)["ok"])

    # 24
    def test_24_commit_without_activation_is_not_active(self) -> None:
        first = self._register(self._request(start=0))
        pid = first.product.product_id
        request = self._request(start=30)
        request.product_id = pid
        request.activate = False
        result = self._register(request)
        self.assertEqual(result.product.current_version, 1)
        statuses = {v.version: v.status for v in self.store.list_versions(pid)}
        self.assertEqual(statuses, {1: "active", 2: "superseded"})
        reopened = ProductStore(self.root)
        self.assertTrue(reopened.validate_product(pid)["ok"])
        self.assertEqual(
            {v.version: v.status for v in reopened.list_versions(pid)},
            {1: "active", 2: "superseded"},
        )

    # 25
    def test_25_persistent_source_samples_and_manifest(self) -> None:
        result = self._register(self._request(
            defect_paths=_paths(TEST_DEFECT, 0, 1),
            test_normal_paths=_paths(TEST_GOOD, 0, 1),
            test_defect_paths=_paths(TEST_DEFECT, 1, 1),
        ))
        version = result.version
        self.assertEqual(version.source_status, "persisted")
        self.assertTrue(version.registration_fingerprint)
        vdir = self.root / result.product.product_id / "versions" / "v001"
        self.assertEqual(len(version.source_samples), 5 + 25 + 1 + 1 + 1)
        roles = {item["role"] for item in version.source_samples}
        self.assertEqual(roles, {"reference", "calibration", "defect", "test_normal", "test_defect"})
        manifest = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
        for item in version.source_samples:
            self.assertFalse(Path(item["rel_path"]).is_absolute())
            self.assertTrue(item["rel_path"].startswith("samples/"))
            self.assertIn(item["rel_path"], manifest["files"])
            path = vdir / item["rel_path"]
            self.assertTrue(path.is_file())
            self.assertEqual(sha256_file(path), item["sha256"])
            self.assertEqual(path.stat().st_size, item["size_bytes"])
            self.assertTrue(item["original_filename"])
        calibration_members = (vdir / "calibration" / "members.csv").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn("samples/calibration/", calibration_members)
        self.assertNotIn("cache", calibration_members)
        sample = version.source_samples[0]
        (vdir / sample["rel_path"]).write_bytes(b"tampered")
        self.assertFalse(self.store.validate_version(result.product.product_id, 1)["ok"])

    # 26
    def test_26_fingerprint_canonical_and_duplicate_rejection(self) -> None:
        first = self._register(self._request())
        count = len(self.store.list_products())
        renamed = self._request(source_original_names={
            str(path): f"renamed_{path.name}" for path in _paths(TRAIN, 0, 30)
        })
        with self.assertRaises(DuplicateRegistrationError):
            self._register(renamed)
        self.assertEqual(len(self.store.list_products()), count)
        self.assertTrue(first.version.registration_fingerprint)

    # 27
    def test_27_same_name_and_count_different_image_allowed(self) -> None:
        first = self._register(self._request())
        second = self._register(RegistrationRequest(
            name="垫片-A", category="washer",
            reference_paths=_paths(TRAIN, 0, 5),
            calibration_paths=_paths(TRAIN, 5, 24) + _paths(TRAIN, 100, 1),
            calibration_count=25,
        ))
        self.assertNotEqual(first.product.product_id, second.product.product_id)
        self.assertNotEqual(first.version.registration_fingerprint,
                            second.version.registration_fingerprint)

    # 28
    def test_28_role_or_category_change_changes_fingerprint(self) -> None:
        req = self._request()
        result = self._register(req)
        baseline = result.version.registration_fingerprint
        other = self._register(RegistrationRequest(
            name="垫片-A", category="gear",
            reference_paths=_paths(TRAIN, 0, 5),
            calibration_paths=_paths(TRAIN, 5, 25),
            calibration_count=25,
        ))
        self.assertNotEqual(other.version.registration_fingerprint, baseline)
        # 同一图片交换角色也必须改变指纹
        ensure_local_weights()
        h_ref = sha256_file(_paths(TRAIN, 0, 1)[0])
        h_cal = sha256_file(_paths(TRAIN, 5, 1)[0])
        common = dict(
            name="垫片-A", category="washer", calibration_count=25,
            image_quantile=0.95, pixel_quantile=0.99,
            feature_extractor_version=p2.FEATURE_VERSION,
            weights_sha256=weights_sha256(),
        )
        fp_a = compute_registration_fingerprint(
            groups={"reference": [h_ref], "calibration": [h_cal],
                    "defect": [], "test_normal": [], "test_defect": []}, **common)
        fp_b = compute_registration_fingerprint(
            groups={"reference": [h_cal], "calibration": [h_ref],
                    "defect": [], "test_normal": [], "test_defect": []}, **common)
        self.assertNotEqual(fp_a, fp_b)

    # 29
    def test_29_legacy_version_source_status(self) -> None:
        product = self.store.create_product("旧产品")
        version = self.store.create_version(product.product_id)
        self.store.set_current_version(product.product_id, version.version)
        loaded = self.store.get_version(product.product_id, version.version)
        self.assertEqual(loaded.source_status, "legacy")
        self.assertEqual(loaded.registration_fingerprint, "")

    # 30
    def test_30_concurrent_same_fingerprint_single_product(self) -> None:
        start = self.root / "start.flag"
        out1 = self.root / "w1.json"
        out2 = self.root / "w2.json"
        worker = PROJECT_ROOT / "tests" / "product_registration_worker.py"
        procs = []
        for result_path in (out1, out2):
            proc = subprocess.Popen(
                [sys.executable, str(worker), "--root", str(self.root),
                 "--start", str(start), "--result", str(result_path)],
                cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            procs.append(proc)
        start.write_text("go", encoding="utf-8")
        for proc in procs:
            proc.wait(timeout=300)
        payloads = [json.loads(p.read_text(encoding="utf-8")) for p in (out1, out2)]
        self.assertEqual(sum(1 for p in payloads if p["status"] == "ok"), 1, payloads)
        self.assertEqual(sum(1 for p in payloads if p["status"] == "duplicate"), 1, payloads)
        self.assertEqual(len(self.store.list_products(include_archived=True)), 1)


    # 31
    def test_31_registration_fingerprint_lock_timeout_no_fallback(self) -> None:
        first = self._register(self._request())
        fingerprint = first.version.registration_fingerprint
        worker = PROJECT_ROOT / "tests" / "product_registration_worker.py"
        start = self.root / "held.flag"
        release = self.root / "release.flag"
        result = self.root / "holder.json"
        holder = subprocess.Popen(
            [sys.executable, str(worker), "--root", str(self.root),
             "--start", str(start), "--result", str(result),
             "--mode", "hold", "--fingerprint", fingerprint,
             "--release", str(release)],
            cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = __import__("time").time() + 30
            while not start.is_file():
                if __import__("time").time() > deadline:
                    self.fail("holder did not acquire lock")
                __import__("time").sleep(0.05)
            with self.assertRaises(ProductLockTimeoutError):
                register_product(
                    ProductStore(self.root, lock_timeout=0.3),
                    self._request(), device="auto",
                )
            self.assertEqual(len(self.store.list_products(include_archived=True)), 1)
        finally:
            release.write_text("release", encoding="utf-8")
            holder.wait(timeout=60)


    # 32
    def test_32_different_fingerprints_can_register_concurrently(self) -> None:
        start = self.root / "start_diff.flag"
        worker = PROJECT_ROOT / "tests" / "product_registration_worker.py"
        specs = (("产品甲", 0, 5, self.root / "a.json"),
                 ("产品乙", 60, 65, self.root / "b.json"))
        procs = []
        for name, ref_start, cal_start, result_path in specs:
            proc = subprocess.Popen(
                [sys.executable, str(worker), "--root", str(self.root),
                 "--start", str(start), "--result", str(result_path),
                 "--name", name, "--ref-start", str(ref_start),
                 "--cal-start", str(cal_start)],
                cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            procs.append(proc)
        start.write_text("go", encoding="utf-8")
        for proc in procs:
            proc.wait(timeout=300)
        payloads = [json.loads(result_path.read_text(encoding="utf-8"))
                    for *_prefix, result_path in specs]
        self.assertEqual([p["status"] for p in payloads], ["ok", "ok"], payloads)
        self.assertEqual(len(self.store.list_products(include_archived=True)), 2)


    # 33
    def test_33_crashed_registration_lock_is_released(self) -> None:
        request = self._request()
        ensure_local_weights()
        groups = {
            "reference": [sha256_file(p) for p in request.reference_paths],
            "calibration": [sha256_file(p) for p in request.calibration_paths],
            "defect": [], "test_normal": [], "test_defect": [],
        }
        fingerprint = compute_registration_fingerprint(
            name=request.name, category=request.category,
            calibration_count=request.calibration_count, groups=groups,
            image_quantile=request.image_quantile,
            pixel_quantile=request.pixel_quantile,
            feature_extractor_version=p2.FEATURE_VERSION,
            weights_sha256=weights_sha256(),
        )
        start = self.root / "crash_hold.flag"
        release = self.root / "crash_release.flag"
        worker = PROJECT_ROOT / "tests" / "product_registration_worker.py"
        holder = subprocess.Popen(
            [sys.executable, str(worker), "--root", str(self.root),
             "--start", str(start), "--result", str(self.root / "crash.json"),
             "--mode", "hold", "--fingerprint", fingerprint,
             "--release", str(release)],
            cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = __import__("time").time() + 30
            while not start.is_file():
                if __import__("time").time() > deadline:
                    self.fail("holder did not acquire lock")
                __import__("time").sleep(0.05)
            holder.kill()
            holder.wait(timeout=30)
            result = self._register(request)
            self.assertTrue(result.version.registration_fingerprint)
            self.assertEqual(len(self.store.list_products(include_archived=True)), 1)
        finally:
            if holder.poll() is None:
                holder.kill()


    # 34
    def test_34_legacy_missing_calibration_unavailable(self) -> None:
        pid, ver = _write_legacy_version(
            self.store, self.root, name="旧缺校准", include_calibration=False,
        )
        result = find_registration_duplicate(self.store, _hash("not-matching"))
        self.assertFalse(result["duplicate"])
        self.assertTrue(any(item["product_id"] == pid for item in result["unavailable"]))
        self.assertTrue(any("calibration" in item["reason"] for item in result["unavailable"]))

    # 35
    def test_35_legacy_sample_count_mismatch_unavailable(self) -> None:
        pid, ver = _write_legacy_version(
            self.store, self.root, name="旧数量不符",
            calibration_count=25, source_count_override=24,
        )
        result = find_registration_duplicate(self.store, _hash("not-matching"))
        self.assertFalse(result["duplicate"])
        self.assertTrue(any(item["product_id"] == pid for item in result["unavailable"]))
        self.assertTrue(any("calibration" in item["reason"] for item in result["unavailable"]))

    # 36
    def test_36_legacy_complete_rebuild_and_duplicate(self) -> None:
        pid, ver = _write_legacy_version(self.store, self.root, name="旧完整")
        version = self.store.get_version(pid, ver)
        groups = {
            role: [item["sha256"] for item in version.source_samples if item["role"] == role]
            for role in ("reference", "calibration", "defect", "test_normal", "test_defect")
        }
        ensure_local_weights()
        fingerprint = compute_registration_fingerprint(
            name="旧完整", category="washer", calibration_count=25, groups=groups,
            image_quantile=0.95, pixel_quantile=0.99,
            feature_extractor_version=p2.FEATURE_VERSION,
            weights_sha256=weights_sha256(),
        )
        result = find_registration_duplicate(self.store, fingerprint)
        self.assertTrue(result["duplicate"])
        self.assertEqual(result["matched"]["product_id"], pid)
        self.assertEqual(result["matched"]["version"], ver)

    # 37
    def test_37_disabled_archived_do_not_block(self) -> None:
        for status in ("disabled", "archived"):
            pid, ver = _write_legacy_version(
                self.store, self.root, name=f"旧{status}", status=status,
            )
            version = self.store.get_version(pid, ver)
            groups = {
                role: [item["sha256"] for item in version.source_samples if item["role"] == role]
                for role in ("reference", "calibration", "defect", "test_normal", "test_defect")
            }
            fingerprint = compute_registration_fingerprint(
                name=f"旧{status}", category="washer", calibration_count=25, groups=groups,
                image_quantile=0.95, pixel_quantile=0.99,
                feature_extractor_version=p2.FEATURE_VERSION,
                weights_sha256=weights_sha256(),
            )
            result = find_registration_duplicate(self.store, fingerprint)
            self.assertFalse(result["duplicate"], status)

    # 38
    def test_38_committed_noncurrent_does_not_block(self) -> None:
        pid, current = _write_legacy_version(self.store, self.root, name="旧当前")
        expected = _hash("noncurrent")
        self.store.create_version(
            pid, initial_status="committed", registration_fingerprint=expected,
            threshold=0.1,
            calibration_metadata={"n_calibration": 25, "image_quantile": 0.95,
                                  "pixel_quantile": 0.99},
            feature_extractor_version=p2.FEATURE_VERSION,
            weights_sha256=weights_sha256(),
        )
        result = find_registration_duplicate(self.store, expected)
        self.assertFalse(result["duplicate"])

    # 39
    def test_39_activation_failure_retry_succeeds(self) -> None:
        request = self._request(start=0)
        with mock.patch.object(
            ProductStore, "_activate_version_pointer_locked",
            side_effect=OSError("activation failed"),
        ):
            with self.assertRaises(OSError):
                self._register(request)
        # 失败留下的 committed 非 current 版本不得阻止相同内容重试
        result = self._register(request)
        self.assertEqual(result.product.status, "active")
        self.assertEqual(result.version.status, "active")


if __name__ == "__main__":
    unittest.main(verbosity=2)
