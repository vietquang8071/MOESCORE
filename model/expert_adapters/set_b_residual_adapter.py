"""Strict E6 ExpertAdapter over precomputed frozen component outputs."""

from collections.abc import Mapping

import torch

from model.composite_experts.set_b_residual import SetBResidualHead
from model.expert_adapters.base import ExpertAdapter, validate_expert_output
from model.expert_adapters.paired_cross_features import build_paired_cross_features


class SetBResidualAdapter(ExpertAdapter):
    def __init__(self, head: SetBResidualHead | None = None):
        super().__init__(4609)
        if head is not None and not isinstance(head, SetBResidualHead):
            raise TypeError("head must be SetBResidualHead")
        self.head = head if head is not None else SetBResidualHead()

    def _forward(self, batch):
        if not isinstance(batch, Mapping) or set(batch) != {"component_outputs", "whisper_audio", "whisper_text"}:
            raise ValueError("E6 batch must contain exactly component_outputs, whisper_audio, whisper_text")
        components = batch["component_outputs"]
        if not isinstance(components, Mapping) or set(components) != {"top2_laion"}:
            raise ValueError("E6 component must be top2_laion")
        laion = components["top2_laion"]
        if not isinstance(laion, Mapping) or set(laion) != {"score", "features"}:
            raise ValueError("top2_laion requires exactly score and features")
        score, features = validate_expert_output(laion["score"], laion["features"], feature_dim=2049,
                                                  expert_name="top2_laion")
        tensors = (score, features, batch["whisper_audio"], batch["whisper_text"])
        for name, value, width in (("whisper_audio", tensors[2], 1280), ("whisper_text", tensors[3], 1280)):
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape != (score.shape[0], width):
                raise ValueError(f"{name} must have shape [B,{width}]")
        if any(t.dtype != torch.float32 for t in tensors):
            raise TypeError("E6 component outputs must be float32")
        if any(t.device != score.device for t in tensors):
            raise ValueError("E6 component device mismatch")
        if any(not torch.isfinite(t).all().item() for t in tensors):
            raise ValueError("E6 components must be finite")
        expected = build_paired_cross_features(features[:, :512], features[:, 512:1024])
        if not torch.allclose(features, expected, rtol=1e-5, atol=1e-6):
            raise ValueError("LAION paired feature representation mismatch")
        if not torch.allclose(score, expected[:, -1:], rtol=1e-5, atol=1e-6):
            raise ValueError("LAION cosine score mismatch")
        features6 = torch.cat((features, tensors[2], tensors[3]), dim=1).detach()
        return self.head(score.detach(), features6), features6
