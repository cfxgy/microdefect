#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特征提取与特征缓存子包（P2）。"""

from src.features.cache import (
    CACHE_SCHEMA,
    CacheCorruptedError,
    CacheKeyParts,
    FeatureCache,
    compute_key,
    md5_file,
)
from src.features.extractor import (
    BACKBONE_NAME,
    EMBED_DIM,
    FEATURE_VERSION,
    PREPROCESS_VERSION,
    WEIGHTS_TAG,
    ExtractorInfo,
    FeatureExtractor,
    WeightsMissingError,
    build_model,
    build_transform,
    expected_weights_path,
    load_image_rgb,
    preprocess_path,
    weights_cached,
    weights_status,
)

__all__ = [
    "BACKBONE_NAME",
    "CACHE_SCHEMA",
    "EMBED_DIM",
    "FEATURE_VERSION",
    "PREPROCESS_VERSION",
    "WEIGHTS_TAG",
    "CacheCorruptedError",
    "CacheKeyParts",
    "ExtractorInfo",
    "FeatureCache",
    "FeatureExtractor",
    "WeightsMissingError",
    "build_model",
    "build_transform",
    "compute_key",
    "expected_weights_path",
    "load_image_rgb",
    "md5_file",
    "preprocess_path",
    "weights_cached",
    "weights_status",
]
