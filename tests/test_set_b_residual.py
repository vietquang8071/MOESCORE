import pytest
import torch

from model.expert_adapters.paired_cross_features import build_paired_cross_features
from model.expert_adapters.set_b_residual_adapter import SetBResidualAdapter


def batch(size=2):
    audio = torch.randn(size, 512)
    text = torch.randn(size, 512)
    laion = build_paired_cross_features(audio, text)
    return {
        "component_outputs": {"top2_laion": {"score": laion[:, -1:].clone(), "features": laion}},
        "whisper_audio": torch.randn(size, 1280),
        "whisper_text": torch.randn(size, 1280),
    }


@pytest.mark.parametrize("size", [1, 3])
def test_initial_score_and_feature_order(size):
    data = batch(size)
    model = SetBResidualAdapter()
    score, features = model(data)
    assert score.shape == (size, 1)
    assert features.shape == (size, 4609)
    assert torch.allclose(score, data["component_outputs"]["top2_laion"]["score"])
    assert torch.equal(features[:, :2049], data["component_outputs"]["top2_laion"]["features"])
    assert torch.equal(features[:, 2049:3329], data["whisper_audio"])
    assert torch.equal(features[:, 3329:], data["whisper_text"])


def test_component_features_are_frozen_and_head_gets_gradients():
    data = batch()
    data["whisper_audio"].requires_grad_()
    data["component_outputs"]["top2_laion"]["features"].requires_grad_()
    model = SetBResidualAdapter()
    model(data)[0].sum().backward()
    assert data["whisper_audio"].grad is None
    assert data["component_outputs"]["top2_laion"]["features"].grad is None
    assert model.head.scale.grad is not None
    assert model.head.residual_mlp[-1].weight.grad is not None


def test_rejects_inconsistent_laion_and_nonfinite_whisper():
    data = batch()
    data["component_outputs"]["top2_laion"]["score"] += 0.1
    with pytest.raises(ValueError):
        SetBResidualAdapter()(data)
    data = batch()
    data["whisper_text"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        SetBResidualAdapter()(data)


def test_affine_calibration_and_residual_equation():
    data = batch(1)
    model = SetBResidualAdapter()
    with torch.no_grad():
        model.head.scale.fill_(1.5)
        model.head.bias.fill_(-0.2)
        model.head.residual_mlp[-1].bias.fill_(0.3)
    actual, features = model(data)
    baseline = data["component_outputs"]["top2_laion"]["score"]
    assert torch.allclose(actual, 1.5 * baseline + 0.1)
    assert features.shape[1] == 4609


@pytest.mark.parametrize("change", ["extra", "width", "dtype", "laion_block"])
def test_invalid_component_contract_rejected(change):
    data = batch(2)
    if change == "extra":
        data["unexpected"] = 1
    elif change == "width":
        data["whisper_audio"] = data["whisper_audio"][:, :-1]
    elif change == "dtype":
        data["whisper_text"] = data["whisper_text"].double()
    else:
        data["component_outputs"]["top2_laion"]["features"][:, 1024] += 0.2
    with pytest.raises((ValueError, TypeError)):
        SetBResidualAdapter()(data)
