#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""界面服务层与 Streamlit 页面测试（第二阶段）。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TMP = PROJECT_ROOT / "cache" / "ui_test_tmp"
_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(_TMP)
os.environ["TEMP"] = str(_TMP)
tempfile.tempdir = str(_TMP)

import sys  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 沙箱禁止 chmod，Streamlit 上传会创建 TemporaryDirectory 并在解释器
# 退出时清理失败；这里让测试进程静默清理，避免误报退出码（不影响被测代码）。
import tempfile  # noqa: E402

tempfile._resetperms = lambda path, *args, **kwargs: None  # type: ignore[attr-defined]


def _quiet_cleanup(cls, name, warn_message, ignore_errors=False, delete=True):
    if not delete:
        return
    try:
        cls._rmtree(name, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


tempfile.TemporaryDirectory._cleanup = classmethod(_quiet_cleanup)  # type: ignore[assignment]

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.ui import services  # noqa: E402
from src.core.product_registration import (  # noqa: E402
    RegistrationRequest,
    register_product,
)
from src.core.products import ProductStore  # noqa: E402
from src.core.weights import ensure_local_weights  # noqa: E402
from src.features import extractor as p2  # noqa: E402
from src.features.mvtec_spatial_extractor import weights_sha256  # noqa: E402

APP = str(PROJECT_ROOT / "src/ui/app.py")
DATA = PROJECT_ROOT / "datasets/mvtec_ad"


def _bytes(rel: str) -> bytes:
    return (DATA / rel).read_bytes()


class ServicesTest(unittest.TestCase):
    def test_01_options(self) -> None:
        options = services.discover_options()
        self.assertIn("group_disjoint", options)
        self.assertEqual(options["group_disjoint"], ["bottle", "pill", "toothbrush", "zipper"])
        self.assertIn("official", options)
        self.assertGreaterEqual(len(options["official"]), 10)

    def test_02_validate_image(self) -> None:
        width, height = services.validate_image_bytes(_bytes("bottle/test/good/000.png"), "a.png")
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        with self.assertRaises(services.ImageValidationError):
            services.validate_image_bytes(b"not an image", "bad.png")
        with self.assertRaises(services.ImageValidationError):
            services.validate_image_bytes(_bytes("bottle/test/good/000.png"), "a.gif")
        with self.assertRaises(services.ImageValidationError):
            services.validate_image_bytes(b"", "empty.png")

    def test_03_run_detection(self) -> None:
        data = _bytes("bottle/test/good/000.png")
        state = services.run_detection(data, "000.png", "bottle", "group_disjoint", device="cpu")
        self.assertFalse(state["result"].is_anomaly)
        for key in ("result_json", "heatmap_png", "overlay_png"):
            self.assertTrue(Path(state["files"][key]).is_file())
        again = services.run_detection(data, "000.png", "bottle", "group_disjoint", device="cpu")
        self.assertEqual(Path(state["files"]["result_json"]).name,
                         Path(again["files"]["result_json"]).name)

    def test_04_unique_names(self) -> None:
        first = services.run_detection(_bytes("bottle/test/good/000.png"), "000.png",
                                      "bottle", "group_disjoint", device="cpu")
        second = services.run_detection(_bytes("bottle/test/good/001.png"), "000.png",
                                        "bottle", "group_disjoint", device="cpu")
        self.assertNotEqual(Path(first["files"]["result_json"]).name,
                            Path(second["files"]["result_json"]).name)


class AppTestUI(unittest.TestCase):
    def _app(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()
        return at

    def test_10_page_loads(self) -> None:
        at = self._app()
        self.assertEqual(list(at.exception), [])
        self.assertEqual(at.selectbox(key="protocol").options,
                         [services.PROTOCOL_LABELS["group_disjoint"],
                          services.PROTOCOL_LABELS["official"]])
        self.assertEqual(at.selectbox(key="category").options,
                         ["bottle", "pill", "toothbrush", "zipper"])
        self.assertTrue(at.button(key="detect_button").disabled)

    def test_11_normal_image(self) -> None:
        at = self._app()
        at.file_uploader(key="uploader").set_value(
            ("000.png", _bytes("bottle/test/good/000.png"), "image/png")
        )
        at.run()
        self.assertFalse(at.button(key="detect_button").disabled)
        at.button(key="detect_button").click()
        at.run(timeout=300)
        self.assertEqual(list(at.exception), [])
        state = at.session_state["detection"]
        self.assertFalse(state["result"].is_anomaly)
        keys = {item.key for item in at.download_button}
        self.assertTrue({"download_json", "download_heatmap", "download_overlay"} <= keys)
        badges = [m.value for m in at.markdown if "badge" in m.value and "span" in m.value]
        self.assertTrue(any("正常" in b for b in badges))

    def test_12_anomaly_image(self) -> None:
        at = self._app()
        at.file_uploader(key="uploader").set_value(
            ("000.png", _bytes("bottle/test/broken_large/000.png"), "image/png")
        )
        at.run()
        at.button(key="detect_button").click()
        at.run(timeout=300)
        self.assertEqual(list(at.exception), [])
        state = at.session_state["detection"]
        self.assertTrue(state["result"].is_anomaly)
        keys = {item.key for item in at.download_button}
        self.assertTrue({"download_json", "download_heatmap", "download_overlay"} <= keys)

    def test_13_invalid_image(self) -> None:
        at = self._app()
        at.file_uploader(key="uploader").set_value(("bad.png", b"not-an-image", "image/png"))
        at.run()
        self.assertEqual(list(at.exception), [])
        self.assertGreaterEqual(len(at.error), 1)
        self.assertTrue(at.button(key="detect_button").disabled)
        self.assertNotIn("detection", at.session_state)

    def test_15_protocol_switch(self) -> None:
        at = self._app()
        at.selectbox(key="protocol").select(services.PROTOCOL_LABELS["official"])
        at.run()
        self.assertEqual(list(at.exception), [])
        categories = at.selectbox(key="category").options
        self.assertGreaterEqual(len(categories), 10)
        self.assertIn("bottle", categories)

    def test_16_batch_flow(self) -> None:
        import shutil

        work = PROJECT_ROOT / "cache" / "tests" / f"batch_ui_{uuid.uuid4().hex[:8]}"
        images = work / "input"
        images.mkdir(parents=True)
        for rel, name in (("bottle/test/good/000.png", "a_good.png"),
                          ("bottle/test/good/001.png", "b_good.png"),
                          ("bottle/test/broken_large/000.png", "c_defect.png")):
            shutil.copyfile(DATA / rel, images / name)
        out = work / "outputs"
        try:
            at = self._app()
            at.text_input(key="batch_folder").set_value(str(images))
            at.text_input(key="batch_out_root").set_value(str(out))
            at.run()
            at.button(key="batch_scan_button").click()
            at.run()
            self.assertEqual(list(at.exception), [])
            at.button(key="batch_start_button").click()
            at.run(timeout=300)
            self.assertEqual(list(at.exception), [])
            run = at.session_state["batch_run"]
            self.assertTrue(run.finished)
            summary = run.summary()
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["success"], 3)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["normal"] + summary["anomaly"], 3)
            for name in ("batch_summary.json", "results.csv"):
                self.assertTrue((run.batch_dir / name).is_file())
            self.assertFalse(any(item.key in {"download_json", "download_heatmap",
                                               "download_overlay"}
                                 for item in at.download_button))

            # 历史批次
            at2 = self._app()
            at2.text_input(key="history_root").set_value(str(out))
            at2.run()
            self.assertEqual(list(at2.exception), [])
            options = at2.selectbox(key="history_batch").options
            self.assertEqual(len(options), 1)
            at2.selectbox(key="history_batch").select(options[0])
            at2.run()
            self.assertEqual(list(at2.exception), [])
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_17_batch_empty_folder(self) -> None:
        import shutil

        empty = PROJECT_ROOT / "cache" / "tests" / f"batch_empty_{uuid.uuid4().hex[:8]}"
        empty.mkdir(parents=True)
        try:
            at = self._app()
            at.text_input(key="batch_folder").set_value(str(empty))
            at.run()
            at.button(key="batch_scan_button").click()
            at.run()
            self.assertEqual(list(at.exception), [])
            self.assertTrue(at.button(key="batch_start_button").disabled)
            self.assertTrue(any("没有可检测" in w.value for w in at.warning))
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_18_stale_running_flag_does_not_block(self) -> None:
        """回归：旧版本遗留的 running=True 不能永久禁用检测按钮。"""
        at = self._app()
        at.session_state["running"] = True
        at.file_uploader(key="uploader").set_value(
            ("000.png", _bytes("bottle/test/good/000.png"), "image/png")
        )
        at.run()
        self.assertEqual(list(at.exception), [])
        self.assertFalse(at.button(key="detect_button").disabled)
        at.button(key="detect_button").click()
        at.run(timeout=300)
        state = at.session_state["detection"]
        self.assertFalse(state["result"].is_anomaly)

    def test_19_new_upload_hides_old_result(self) -> None:
        """换图后不得继续显示上一张图的结果。"""
        at = self._app()
        at.file_uploader(key="uploader").set_value(
            ("000.png", _bytes("bottle/test/good/000.png"), "image/png")
        )
        at.run()
        at.button(key="detect_button").click()
        at.run(timeout=300)
        self.assertGreaterEqual(len(at.metric), 4)
        at.file_uploader(key="uploader").set_value(
            ("000.png", _bytes("bottle/test/broken_large/000.png"), "image/png")
        )
        at.run()
        self.assertEqual(list(at.exception), [])
        self.assertNotIn("图像级异常分数", [item.label for item in at.metric])
        self.assertTrue(any("已选择新的图像" in item.value for item in at.info))
        at.button(key="detect_button").click()
        at.run(timeout=300)
        self.assertTrue(at.session_state["detection"]["result"].is_anomaly)

    def test_14_duplicate_click_reuses(self) -> None:
        at = self._app()
        at.file_uploader(key="uploader").set_value(
            ("000.png", _bytes("bottle/test/good/000.png"), "image/png")
        )
        at.run()
        at.button(key="detect_button").click()
        at.run(timeout=300)
        first = at.session_state["detection"]["result"].image_score
        at.button(key="detect_button").click()
        at.run(timeout=300)
        self.assertEqual(list(at.exception), [])
        self.assertAlmostEqual(first, at.session_state["detection"]["result"].image_score, places=12)
        self.assertTrue(any("复用" in info.value for info in at.info))


class ProductUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = PROJECT_ROOT / "cache" / "tests" / f"p32_ui_{uuid.uuid4().hex[:8]}"
        cls.root.mkdir(parents=True, exist_ok=False)
        cls.old_root_env = os.environ.get("MICRODEFECT_PRODUCT_ROOT")
        os.environ["MICRODEFECT_PRODUCT_ROOT"] = str(cls.root)
        cls.store = ProductStore(cls.root, lock_timeout=60)

        def paths(start: int, count: int) -> list[Path]:
            return [DATA / "bottle/train/good" / f"{i:03d}.png" for i in range(start, start + count)]

        cls.product_a = register_product(
            cls.store,
            RegistrationRequest(
                name="产品A", category="washer",
                reference_paths=paths(0, 5), calibration_paths=paths(5, 25),
            ),
            device="auto",
        )
        cls.product_b = register_product(
            cls.store,
            RegistrationRequest(
                name="产品B", category="gear",
                reference_paths=paths(60, 5), calibration_paths=paths(65, 25),
            ),
            device="auto",
        )
        draft = cls.store.create_product("草稿产品")
        disabled = cls.store.create_product("停用产品")
        cls.store.update_product_status(disabled.product_id, "disabled")
        archived = cls.store.create_product("归档产品")
        cls.store.update_product_status(archived.product_id, "archived")
        cls.draft_id = draft.product_id
        cls.disabled_id = disabled.product_id
        cls.archived_id = archived.product_id
        corrupt = cls.store.create_product("损坏产品")
        corrupt_version = cls.store.create_version(corrupt.product_id)
        cls.store.set_current_version(corrupt.product_id, corrupt_version.version)
        keep = cls.store.version_directory(corrupt.product_id, 1) / "calibration" / ".keep"
        keep.write_bytes(keep.read_bytes() + b"corrupt")
        cls.corrupt_id = corrupt.product_id
        # 旧来源不完整产品：用于 UI 同时展示 duplicate 与 unavailable
        legacy = cls.store.create_product("旧来源不可判重")
        ensure_local_weights()
        legacy_version = cls.store.create_version(
            legacy.product_id,
            threshold=0.1,
            calibration_metadata={"n_calibration": 25, "image_quantile": 0.95,
                                  "pixel_quantile": 0.99},
            feature_extractor_version=p2.FEATURE_VERSION,
            weights_sha256=weights_sha256(),
            source_samples=[
                {"role": "reference", "rel_path": f"legacy/{i}.png",
                 "sha256": hashlib.sha256(f"legacy-{i}".encode()).hexdigest()}
                for i in range(5)
            ],
        )
        legacy_dir = cls.store.version_directory(legacy.product_id, 1)
        (legacy_dir / "registration.json").write_text(json.dumps({
            "n_reference": 5, "n_calibration": 25, "n_defect": 0,
            "n_test_normal": 0, "n_test_defect": 0,
        }), encoding="utf-8")
        manifest = cls.store._build_manifest(legacy_dir, legacy.product_id, 1)
        (legacy_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        (legacy_dir / "manifest.sha256").write_text(
            hashlib.sha256((legacy_dir / "manifest.json").read_bytes()).hexdigest() + "\n",
            encoding="ascii")
        cls.store.set_current_version(legacy.product_id, legacy_version.version)
        cls.legacy_unavailable_id = legacy.product_id

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.old_root_env is None:
            os.environ.pop("MICRODEFECT_PRODUCT_ROOT", None)
        else:
            os.environ["MICRODEFECT_PRODUCT_ROOT"] = cls.old_root_env
        shutil.rmtree(cls.root, ignore_errors=True)

    def _app(self) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()
        return at

    def _uploads(self, paths: list[Path]) -> list[tuple[str, bytes, str]]:
        return [(p.name, p.read_bytes(), "image/png") for p in paths]

    def _paths(self, start: int, count: int) -> list[Path]:
        return [DATA / "bottle/train/good" / f"{i:03d}.png" for i in range(start, start + count)]

    def test_20_register_ui_success_and_cleanup(self) -> None:
        before = set(services.PRODUCT_UPLOAD_DIR.glob("register_*"))
        at = self._app()
        at.text_input(key="product_name").set_value("UI新产品")
        at.text_input(key="product_category").set_value("washer")
        at.number_input(key="product_calibration_count").set_value(25)
        at.file_uploader(key="product_reference_uploader").set_value(self._uploads(self._paths(0, 5)))
        at.file_uploader(key="product_calibration_uploader").set_value(self._uploads(self._paths(5, 25)))
        at.run()
        self.assertEqual(list(at.exception), [])
        self.assertFalse(at.button(key="product_register_button").disabled)
        at.button(key="product_register_button").click()
        at.run(timeout=600)
        self.assertEqual(list(at.exception), [])
        self.assertIn("product_register_result", at.session_state)
        summary = at.session_state["product_register_result"]["summary"]
        self.assertEqual(summary["name"], "UI新产品")
        self.assertEqual(summary["status"], "active")
        self.assertEqual(summary["current_version"], 1)
        self.assertEqual(summary["n_reference"], 5)
        self.assertEqual(summary["n_calibration"], 25)
        self.assertTrue(at.session_state["product_register_result"]["validation"]["ok"])
        product = self.store.get_product(summary["product_id"])
        self.assertEqual(product.status, "active")
        self.assertTrue(self.store.validate_product(product.product_id)["ok"])
        version = self.store.get_version(product.product_id, product.current_version)
        self.assertEqual(version.source_status, "persisted")
        self.assertTrue(version.registration_fingerprint)
        self.assertEqual(len(version.source_samples), 30)
        self.assertIn("000.png", [item["original_filename"] for item in version.source_samples])
        for item in version.source_samples:
            self.assertFalse(Path(item["rel_path"]).is_absolute())
            self.assertTrue(Path(item["rel_path"]).as_posix().startswith("samples/"))
            sample_path = self.store.version_directory(
                product.product_id, version.version) / item["rel_path"]
            self.assertTrue(sample_path.is_file())
            self.assertEqual(hashlib.sha256(sample_path.read_bytes()).hexdigest(), item["sha256"])
        self.assertTrue(at.button(key="product_register_button").disabled)
        self.assertTrue(any(
            "已成功注册" in info.value or "已存在相同注册内容" in info.value
            for info in list(at.info) + list(at.warning)
        ))
        # 上传临时目录已清理
        after = set(services.PRODUCT_UPLOAD_DIR.glob("register_*"))
        self.assertEqual(after, before)

    def test_21_register_insufficient_reference(self) -> None:
        at = self._app()
        at.text_input(key="product_name").set_value("不足参考")
        at.number_input(key="product_calibration_count").set_value(25)
        at.file_uploader(key="product_reference_uploader").set_value(self._uploads(self._paths(0, 4)))
        at.file_uploader(key="product_calibration_uploader").set_value(self._uploads(self._paths(5, 25)))
        at.run()
        self.assertEqual(list(at.exception), [])
        self.assertTrue(at.button(key="product_register_button").disabled)
        self.assertTrue(any("至少需要 5 张" in e.value for e in at.error))

    def test_22_register_low_sample_warning(self) -> None:
        at = self._app()
        at.text_input(key="product_name").set_value("低样本产品")
        at.number_input(key="product_calibration_count").set_value(20)
        at.file_uploader(key="product_reference_uploader").set_value(self._uploads(self._paths(0, 5)))
        at.file_uploader(key="product_calibration_uploader").set_value(self._uploads(self._paths(5, 20)))
        at.run()
        self.assertEqual(list(at.exception), [])
        self.assertFalse(at.button(key="product_register_button").disabled)
        self.assertTrue(any("20–24" in w.value or "偏少" in w.value for w in at.warning))

    def test_23_register_duplicate_reference_calibration_rejected(self) -> None:
        at = self._app()
        refs = self._paths(0, 5)
        cals = [refs[0]] + self._paths(5, 24)
        at.text_input(key="product_name").set_value("重复图片产品")
        at.number_input(key="product_calibration_count").set_value(25)
        at.file_uploader(key="product_reference_uploader").set_value(self._uploads(refs))
        at.file_uploader(key="product_calibration_uploader").set_value(self._uploads(cals))
        at.run()
        at.button(key="product_register_button").click()
        at.run(timeout=600)
        self.assertEqual(list(at.exception), [])
        self.assertTrue(any("注册失败" in e.value for e in at.error))
        self.assertNotIn("product_register_result", at.session_state)

    def test_24_product_list_and_detectable_selection(self) -> None:
        at = self._app()
        options = at.selectbox(key="product_select").options
        self.assertTrue(any("产品A" in item for item in options))
        self.assertTrue(any("产品B" in item for item in options))
        self.assertFalse(any("草稿产品" in item for item in options))
        self.assertFalse(any("停用产品" in item for item in options))
        self.assertFalse(any("归档产品" in item for item in options))
        self.assertFalse(any("损坏产品" in item for item in options))
        frames = [df.value for df in at.dataframe if "产品名称" in df.value.columns]
        self.assertTrue(frames)
        frame = frames[0]
        self.assertIn("不可检测原因", frame.columns)
        corrupt_rows = frame[frame["产品名称"] == "损坏产品"]
        self.assertEqual(len(corrupt_rows), 1)
        self.assertEqual(corrupt_rows.iloc[0]["可检测"], "否")
        self.assertTrue(corrupt_rows.iloc[0]["不可检测原因"])

    def test_28_product_b_detection_isolated(self) -> None:
        at = self._app()
        options = at.selectbox(key="product_select").options
        label = next(item for item in options if "产品B" in item)
        at.selectbox(key="product_select").select(label)
        at.run()
        at.file_uploader(key="product_detect_uploader").set_value(
            ("good.png", (DATA / "bottle/test/good/000.png").read_bytes(), "image/png")
        )
        at.run()
        at.button(key="product_detect_button").click()
        at.run(timeout=600)
        self.assertEqual(list(at.exception), [])
        result = at.session_state["product_detection"]["result"]
        self.assertEqual(result.product_id, self.product_b.product.product_id)
        self.assertEqual(result.product_name, "产品B")
        self.assertEqual(result.version, 1)

    def test_29_empty_product_directory(self) -> None:
        empty = PROJECT_ROOT / "cache" / "tests" / f"p32_empty_{uuid.uuid4().hex[:8]}"
        empty.mkdir(parents=True)
        old = os.environ.get("MICRODEFECT_PRODUCT_ROOT")
        os.environ["MICRODEFECT_PRODUCT_ROOT"] = str(empty)
        try:
            at = self._app()
            self.assertEqual(list(at.exception), [])
            self.assertTrue(any("产品目录为空" in info.value for info in at.info))
            self.assertEqual([box for box in at.selectbox if box.key == "product_select"], [])
        finally:
            if old is None:
                os.environ.pop("MICRODEFECT_PRODUCT_ROOT", None)
            else:
                os.environ["MICRODEFECT_PRODUCT_ROOT"] = old
            shutil.rmtree(empty, ignore_errors=True)

    def test_25_detect_normal_and_anomaly(self) -> None:
        at = self._app()
        options = at.selectbox(key="product_select").options
        label = next(item for item in options if "产品A" in item)
        at.selectbox(key="product_select").select(label)
        at.run()
        at.file_uploader(key="product_detect_uploader").set_value(
            ("good.png", (DATA / "bottle/test/good/000.png").read_bytes(), "image/png")
        )
        at.run()
        self.assertFalse(at.button(key="product_detect_button").disabled)
        at.button(key="product_detect_button").click()
        at.run(timeout=600)
        self.assertEqual(list(at.exception), [])
        state = at.session_state["product_detection"]
        result = state["result"]
        self.assertEqual(result.product_id, self.product_a.product.product_id)
        self.assertEqual(result.version, 1)
        self.assertEqual(result.image_status, "normal")
        self.assertFalse(result.is_anomaly)
        self.assertEqual(result.input_image_sha256, result.image_sha256)
        self.assertTrue(result.bank_version)
        self.assertGreaterEqual(result.inference_time, 0.0)
        self.assertGreaterEqual(len(at.download_button), 4)
        self.assertTrue(any("bank_version" in c.value for c in at.caption))

        at.file_uploader(key="product_detect_uploader").set_value(
            ("bad.png", (DATA / "bottle/test/broken_large/000.png").read_bytes(), "image/png")
        )
        at.run()
        at.button(key="product_detect_button").click()
        at.run(timeout=600)
        self.assertEqual(list(at.exception), [])
        anomaly = at.session_state["product_detection"]["result"]
        self.assertTrue(anomaly.is_anomaly)
        self.assertEqual(anomaly.image_status, "anomaly")
        self.assertGreater(anomaly.image_score, anomaly.image_threshold)

    def test_26_invalid_image_failure_recovery(self) -> None:
        at = self._app()
        options = at.selectbox(key="product_select").options
        label = next(item for item in options if "产品A" in item)
        at.selectbox(key="product_select").select(label)
        at.run()
        at.file_uploader(key="product_detect_uploader").set_value(
            ("bad.png", b"not-an-image", "image/png")
        )
        at.run()
        self.assertEqual(list(at.exception), [])
        self.assertTrue(at.button(key="product_detect_button").disabled)
        self.assertTrue(any("图片不可用" in e.value for e in at.error))
        self.assertNotIn("product_detection", at.session_state)

        at.file_uploader(key="product_detect_uploader").set_value(
            ("good.png", (DATA / "bottle/test/good/000.png").read_bytes(), "image/png")
        )
        at.run()
        self.assertFalse(at.button(key="product_detect_button").disabled)
        at.button(key="product_detect_button").click()
        at.run(timeout=600)
        self.assertEqual(list(at.exception), [])
        self.assertIn("product_detection", at.session_state)

    def test_27_current_version_switch_updates_detection(self) -> None:
        pid = self.product_a.product.product_id
        self.store.set_current_version(pid, 1)
        register_product(
            self.store,
            RegistrationRequest(
                name="ignored", product_id=pid,
                reference_paths=self._paths(30, 5), calibration_paths=self._paths(35, 25),
            ),
            device="auto",
        )
        self.store.set_current_version(pid, 1)
        at = self._app()
        options = at.selectbox(key="product_select").options
        at.selectbox(key="product_select").select(next(item for item in options if "产品A" in item))
        at.run()
        at.file_uploader(key="product_detect_uploader").set_value(
            ("good.png", (DATA / "bottle/test/good/000.png").read_bytes(), "image/png")
        )
        at.run()
        at.button(key="product_detect_button").click()
        at.run(timeout=600)
        self.assertEqual(at.session_state["product_detection"]["result"].version, 1)

        self.store.set_current_version(pid, 2)
        at.run()
        options = at.selectbox(key="product_select").options
        label = next(item for item in options if "产品A" in item)
        self.assertIn("v2", label)
        at.selectbox(key="product_select").select(label)
        at.run()
        at.button(key="product_detect_button").click()
        at.run(timeout=600)
        self.assertEqual(at.session_state["product_detection"]["result"].version, 2)


    def test_30_duplicate_same_content_across_new_session(self) -> None:
        self.store.set_current_version(self.product_a.product.product_id, 1)
        before = len(self.store.list_products(include_archived=True))
        at = self._app()
        at.text_input(key="product_name").set_value("产品A")
        at.text_input(key="product_category").set_value("washer")
        at.number_input(key="product_calibration_count").set_value(25)
        at.file_uploader(key="product_reference_uploader").set_value(self._uploads(self._paths(0, 5)))
        at.file_uploader(key="product_calibration_uploader").set_value(self._uploads(self._paths(5, 25)))
        at.run()
        self.assertEqual(list(at.exception), [])
        self.assertTrue(at.button(key="product_register_button").disabled)
        self.assertTrue(any("已存在相同注册内容" in w.value for w in at.warning))
        self.assertEqual(len(self.store.list_products(include_archived=True)), before)

    def test_31_same_name_count_changed_image_not_locked(self) -> None:
        self.store.set_current_version(self.product_a.product.product_id, 1)
        at = self._app()
        at.text_input(key="product_name").set_value("产品A")
        at.text_input(key="product_category").set_value("washer")
        at.number_input(key="product_calibration_count").set_value(25)
        at.file_uploader(key="product_reference_uploader").set_value(self._uploads(self._paths(0, 5)))
        changed = self._paths(5, 24) + self._paths(100, 1)
        at.file_uploader(key="product_calibration_uploader").set_value(self._uploads(changed))
        at.run()
        self.assertEqual(list(at.exception), [])
        self.assertFalse(at.button(key="product_register_button").disabled)
        self.assertFalse(any("已存在相同注册内容" in w.value for w in at.warning))
        self.assertTrue(any("无法可靠判重" in w.value for w in at.warning))


    def test_32_duplicate_and_unavailable_both_shown(self) -> None:
        self.store.set_current_version(self.product_a.product.product_id, 1)
        at = self._app()
        at.text_input(key="product_name").set_value("产品A")
        at.text_input(key="product_category").set_value("washer")
        at.number_input(key="product_calibration_count").set_value(25)
        at.file_uploader(key="product_reference_uploader").set_value(self._uploads(self._paths(0, 5)))
        at.file_uploader(key="product_calibration_uploader").set_value(self._uploads(self._paths(5, 25)))
        at.run()
        self.assertEqual(list(at.exception), [])
        values = [w.value for w in at.warning]
        self.assertTrue(any("已存在相同注册内容" in v for v in values), values)
        self.assertTrue(any("无法可靠判重" in v for v in values), values)
        self.assertTrue(at.button(key="product_register_button").disabled)


    def test_33_product_batch_ui_and_history(self) -> None:
        self.store.set_current_version(self.product_a.product.product_id, 1)
        at = self._app()
        options = at.selectbox(key="product_batch_select").options
        at.selectbox(key="product_batch_select").select(
            next(item for item in options if "产品A" in item)
        )
        at.run()
        good = (DATA / "bottle/test/good/000.png").read_bytes()
        good2 = (DATA / "bottle/test/good/001.png").read_bytes()
        at.file_uploader(key="product_batch_uploader").set_value([
            ("same.png", good, "image/png"),
            ("same.png", good2, "image/png"),
            ("bad.png", b"not-an-image", "image/png"),
            ("empty.png", b"", "image/png"),
        ])
        at.run()
        self.assertFalse(at.button(key="product_batch_start").disabled)
        at.button(key="product_batch_start").click()
        at.run(timeout=600)
        self.assertEqual(list(at.exception), [])
        run = at.session_state["product_batch_run"]
        self.assertTrue(run.finished)
        self.assertEqual(run.total, 4)
        self.assertEqual(run.summary()["success"], 2)
        self.assertEqual(run.summary()["failed"], 2)
        self.assertEqual(run.summary()["status"], "partial")
        self.assertEqual(run.detector.version.version, 1)
        batch_id = run.batch_id

        # 确保存在 v2 后再切换；历史仍显示原版本
        try:
            self.store.get_version(self.product_a.product.product_id, 2)
        except Exception:
            register_product(
                self.store,
                RegistrationRequest(
                    name="ignored", product_id=self.product_a.product.product_id,
                    reference_paths=self._paths(30, 5),
                    calibration_paths=self._paths(35, 25),
                ),
                device="auto",
            )
        self.store.set_current_version(self.product_a.product.product_id, 2)
        try:
            hist = self._app()
            labels = hist.selectbox(key="product_batch_history_batch").options
            chosen = next(item for item in labels if item.startswith(batch_id))
            hist.selectbox(key="product_batch_history_batch").select(chosen)
            hist.run()
            self.assertEqual(list(hist.exception), [])
            payload = services.load_product_batch(batch_id)
            self.assertEqual(payload["product_version"], 1)
            self.assertEqual(payload["status"], "partial")
            self.assertGreaterEqual(len(hist.download_button), 3)
        finally:
            self.store.set_current_version(self.product_a.product.product_id, 1)


    def test_34_product_drift_ui(self) -> None:
        self.store.set_current_version(self.product_a.product.product_id, 1)
        at = self._app()
        at.selectbox(key="drift_product").select(
            next(item for item in at.selectbox(key="drift_product").options
                 if "产品A" in item)
        )
        at.run()
        at.file_uploader(key="drift_uploader").set_value(
            self._uploads(self._paths(0, 5))
        )
        at.run()
        self.assertFalse(at.button(key="drift_analyze").disabled)
        at.button(key="drift_analyze").click()
        at.run(timeout=600)
        self.assertEqual(list(at.exception), [])
        payload = at.session_state["drift_result"]
        self.assertEqual(payload["product_version"], 1)
        self.assertEqual(payload["success"], 5)
        self.assertIn(payload["level"], ("normal", "observation"))
        self.assertEqual(payload["product_id"], self.product_a.product.product_id)
        self.assertGreaterEqual(len(at.download_button), 2)


    def test_35_feedback_review_and_update_preview_ui(self) -> None:
        self.store.set_current_version(self.product_a.product.product_id, 1)
        pid = self.product_a.product.product_id
        image = DATA / "bottle/test/good/005.png"
        feedback = services.create_product_feedback(
            pid, product_version=1, source_type="detection",
            source_id="ui-feedback-1", image_path=image,
            image_score=0.01, image_threshold=0.021, image_status="normal",
            original_filename=image.name, root=self.root,
        )
        at = self._app()
        at.selectbox(key="feedback_product").select(
            next(item for item in at.selectbox(key="feedback_product").options
                 if "产品A" in item)
        )
        at.run()
        services.review_product_feedback(
            pid, feedback["feedback_id"], "normal", "UI审核", root=self.root)
        at.run()
        records = services.list_product_feedback(pid, root=self.root)
        reviewed = next(item for item in records
                        if item["feedback_id"] == feedback["feedback_id"])
        self.assertEqual(reviewed["status"], "confirmed")

        # 更新预览（不激活）
        at.multiselect(key="feedback_update_ids").set_value([feedback["feedback_id"]])
        at.file_uploader(key="validation_normal").set_value(
            self._uploads([DATA / "bottle/test/good" / f"{i:03d}.png" for i in range(5)]))
        at.file_uploader(key="validation_anomaly").set_value(
            self._uploads([DATA / "bottle/test/broken_large" / f"{i:03d}.png"
                           for i in range(5)]))
        at.run()
        at.button(key="feedback_prepare_update").click()
        at.run(timeout=600)
        self.assertEqual(list(at.exception), [])
        report = at.session_state["p5_update_report"]
        self.assertIn(report["verdict"], ("pass", "fail", "inconclusive"))
        self.assertFalse(report["activated"])
        self.assertEqual(self.store.get_product(pid).current_version, 1)
        # 重复提交同一候选/验证集必须返回同一 update，不创建新版本
        before_versions = len(self.store.list_versions(pid))
        at.button(key="feedback_prepare_update").click()
        at.run(timeout=600)
        self.assertEqual(list(at.exception), [])
        repeated = at.session_state["p5_update_report"]
        self.assertEqual(repeated["update_id"], report["update_id"])
        self.assertEqual(before_versions, len(self.store.list_versions(pid)))

    def test_36_batch_history_feedback_entry(self) -> None:
        self.store.set_current_version(self.product_a.product.product_id, 1)
        pid = self.product_a.product.product_id
        run = services.create_product_batch(
            pid,
            [(f"{i}.png", (DATA / "bottle/test/good" / f"{i:03d}.png").read_bytes())
             for i in range(2)],
            root=self.root,
        )
        while not run.finished:
            run.step()
        at = self._app()
        labels = at.selectbox(key="product_batch_history_batch").options
        chosen = next(item for item in labels if run.batch_id in item)
        at.selectbox(key="product_batch_history_batch").select(chosen)
        at.run()
        self.assertEqual(list(at.exception), [])
        buttons = [b for b in at.button if b.key and b.key.startswith("batch_feedback_")]
        self.assertTrue(buttons)
        buttons[0].click()
        at.run()
        self.assertEqual(list(at.exception), [])
        records = services.list_product_feedback(pid, root=self.root)
        self.assertTrue(any(item["source_type"] == "batch"
                            and item["source_id"] == run.batch_id for item in records))

    def test_37_drift_history_feedback_entry(self) -> None:
        self.store.set_current_version(self.product_a.product.product_id, 1)
        pid = self.product_a.product.product_id
        payload = services.analyze_product_drift(
            pid,
            [(f"{i}.png", (DATA / "bottle/test/good" / f"{i:03d}.png").read_bytes())
             for i in range(5)],
            root=self.root,
        )
        at = self._app()
        labels = at.selectbox(key="drift_history_window").options
        chosen = next(item for item in labels if payload["window_id"] in item)
        at.selectbox(key="drift_history_window").select(chosen)
        at.run()
        self.assertEqual(list(at.exception), [])
        buttons = [b for b in at.button if b.key and b.key.startswith("drift_feedback_")]
        self.assertTrue(buttons)
        buttons[0].click()
        at.run()
        self.assertEqual(list(at.exception), [])
        records = services.list_product_feedback(pid, root=self.root)
        self.assertTrue(any(item["source_type"] == "drift"
                            and item["source_id"] == payload["window_id"]
                            for item in records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
