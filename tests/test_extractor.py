#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P2 特征提取器测试。

需要预训练权重的用例在权重缺失时会被标记为 skip（并给出原因），
不会伪造通过；权重就位后重新运行即可得到完整结果。

    D:\\microdefect\\.venv\\Scripts\\python.exe -m unittest tests.test_extractor -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extractor import (  # noqa: E402
    EMBED_DIM,
    IMAGE_SIZE,
    WEIGHTS_TAG,
    FeatureExtractor,
    WeightsMissingError,
    build_model,
    build_transform,
    expected_weights_path,
    l2_normalize,
    load_image_rgb,
    preprocess_pil,
    weights_cached,
    weights_status,
)

DATA_ROOT = PROJECT_ROOT / "data" / "NEU-CLS"
SAMPLE_IMAGES = [
    DATA_ROOT / "Cr_1.bmp",
    DATA_ROOT / "In_1.bmp",
    DATA_ROOT / "Pa_1.bmp",
    DATA_ROOT / "PS_1.bmp",
    DATA_ROOT / "RS_1.bmp",
    DATA_ROOT / "Sc_1.bmp",
]

HAS_WEIGHTS = weights_cached()
SKIP_REASON = (
    f"ImageNet 预训练权重未缓存（期望路径 {expected_weights_path()}）；"
    "在获得批准并下载后本用例可完整执行"
)


class PreprocessTestCase(unittest.TestCase):
    """不依赖模型权重的预处理测试。"""

    def test_01_grayscale_to_three_identical_channels(self) -> None:
        image = load_image_rgb(SAMPLE_IMAGES[0])
        self.assertEqual(image.mode, "RGB")
        array = np.asarray(image)
        self.assertEqual(array.shape[2], 3)
        # 三个通道必须完全一致（同一灰度复制），而不是不同颜色通道
        np.testing.assert_array_equal(array[:, :, 0], array[:, :, 1])
        np.testing.assert_array_equal(array[:, :, 1], array[:, :, 2])

    def test_02_preprocess_shape_dtype(self) -> None:
        tensor = preprocess_pil(load_image_rgb(SAMPLE_IMAGES[1]))
        self.assertEqual(tuple(tensor.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all().item())

    def test_03_preprocess_deterministic(self) -> None:
        image = load_image_rgb(SAMPLE_IMAGES[2])
        first = preprocess_pil(image)
        second = preprocess_pil(image)
        self.assertTrue(torch.equal(first, second))

    def test_04_transform_has_no_random_ops(self) -> None:
        transform = build_transform()
        names = [type(op).__name__ for op in transform.transforms]
        self.assertEqual(names, ["Resize", "ToTensor", "Normalize"])
        self.assertFalse(any("Random" in n for n in names))

    def test_05_weights_status_reports_expected_path(self) -> None:
        status = weights_status()
        self.assertEqual(status["weights_enum"], f"ResNet18_Weights.{WEIGHTS_TAG}")
        self.assertTrue(status["url"].endswith(".pth"))
        self.assertEqual(Path(status["expected_path"]).name, Path(status["url"]).name)
        self.assertEqual(status["cached"], HAS_WEIGHTS)


class WeightsMissingTestCase(unittest.TestCase):
    """权重缺失时不得静默下载或使用随机权重。"""

    @unittest.skipIf(HAS_WEIGHTS, "权重已缓存，本用例只在缺失场景下有意义")
    def test_06_missing_weights_raises_instead_of_random_init(self) -> None:
        with self.assertRaises(WeightsMissingError):
            build_model(allow_download=False)
        self.assertFalse(expected_weights_path().exists())


@unittest.skipUnless(HAS_WEIGHTS, SKIP_REASON)
class ModelTestCase(unittest.TestCase):
    """需要真实 ImageNet 权重的用例。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.device = build_model("auto")
        cls.extractor = FeatureExtractor("auto", batch_size=4)

    def test_10_model_loaded_and_fc_removed(self) -> None:
        self.assertIsInstance(self.model.fc, torch.nn.Identity)
        self.assertNotIsInstance(self.model.fc, torch.nn.Linear)

    def test_11_all_params_frozen(self) -> None:
        trainable = [name for name, p in self.model.named_parameters() if p.requires_grad]
        self.assertEqual(trainable, [])

    def test_12_model_in_eval_mode(self) -> None:
        self.assertFalse(self.model.training)

    def test_13_single_image_shape(self) -> None:
        feature = self.extractor.extract_path(SAMPLE_IMAGES[0])
        self.assertEqual(feature.shape, (EMBED_DIM,))
        self.assertEqual(feature.dtype, np.float32)

    def test_14_batch_shape(self) -> None:
        features = self.extractor.extract_paths(SAMPLE_IMAGES)
        self.assertEqual(features.shape, (len(SAMPLE_IMAGES), EMBED_DIM))

    def test_15_no_nan_or_inf(self) -> None:
        features = self.extractor.extract_paths(SAMPLE_IMAGES[:3])
        self.assertTrue(np.isfinite(features).all())

    def test_16_raw_and_l2_interfaces(self) -> None:
        raw = self.extractor.extract_path(SAMPLE_IMAGES[0])
        normalized = self.extractor.extract_path(SAMPLE_IMAGES[0], normalize=True)
        # 归一化不应覆盖原始特征：二者独立返回
        self.assertFalse(np.allclose(raw, normalized))
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=5)
        self.assertNotAlmostEqual(float(np.linalg.norm(raw)), 1.0, places=3)
        again = self.extractor.extract_path(SAMPLE_IMAGES[0])
        np.testing.assert_array_equal(raw, again)

    def test_17_repeat_consistency(self) -> None:
        first = self.extractor.extract_paths(SAMPLE_IMAGES[:4])
        second = self.extractor.extract_paths(SAMPLE_IMAGES[:4])
        np.testing.assert_array_equal(first, second)

    def test_18_duplicate_images_produce_identical_features(self) -> None:
        a = self.extractor.extract_path(DATA_ROOT / "Pa_101.bmp")
        b = self.extractor.extract_path(DATA_ROOT / "Pa_105.bmp")
        np.testing.assert_array_equal(a, b)

    def test_19_cpu_fallback(self) -> None:
        cpu_extractor = FeatureExtractor("cpu")
        self.assertEqual(cpu_extractor.device.type, "cpu")
        feature = cpu_extractor.extract_path(SAMPLE_IMAGES[0])
        self.assertEqual(feature.shape, (EMBED_DIM,))
        self.assertTrue(np.isfinite(feature).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 不可用")
    def test_20_gpu_inference(self) -> None:
        gpu_extractor = FeatureExtractor("cuda")
        self.assertEqual(gpu_extractor.device.type, "cuda")
        gpu_feature = gpu_extractor.extract_path(SAMPLE_IMAGES[0])
        cpu_feature = FeatureExtractor("cpu").extract_path(SAMPLE_IMAGES[0])
        self.assertEqual(gpu_feature.shape, (EMBED_DIM,))
        np.testing.assert_allclose(gpu_feature, cpu_feature, rtol=1e-4, atol=1e-4)
        memory = gpu_extractor.gpu_memory()
        self.assertEqual(memory["device"], torch.cuda.get_device_name(0))

    def test_21_l2_normalize_helper_does_not_mutate(self) -> None:
        raw = self.extractor.extract_paths(SAMPLE_IMAGES[:3])
        snapshot = raw.copy()
        normalized = l2_normalize(raw)
        np.testing.assert_array_equal(raw, snapshot)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
