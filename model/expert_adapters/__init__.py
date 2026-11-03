"""Lightweight shared interfaces for future fine-tuned expert adapters."""

from .base import ExpertAdapter, validate_expert_output

__all__ = ["ExpertAdapter", "validate_expert_output"]
