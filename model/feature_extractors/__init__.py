"""Reusable frozen feature extractors that are not standalone MoE experts."""

from model.feature_extractors.whisper_top2 import WhisperTop2FeatureExtractor

__all__ = ["WhisperTop2FeatureExtractor"]
