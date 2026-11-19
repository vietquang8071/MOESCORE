import pytest
import torch

from model.expert_adapters.registry import profile_contract, validate_and_assemble_expert_outputs
from model.moe_gating_core import GatingNetwork


def test_full7_order_width_and_slices():
    order, dims, width = profile_contract("full_7")
    assert order == tuple(f"expert{i}" for i in range(1, 8))
    assert list(dims.values()) == [2049, 4097, 3840, 1024, 11267, 4609, 3073]
    assert width == 29959
    outputs = {name: (torch.full((2, 1), float(i)), torch.full((2, dims[name]), float(i)))
               for i, name in enumerate(order, 1)}
    scores, features = validate_and_assemble_expert_outputs(outputs, profile="full_7")
    assert scores.shape == (2, 7) and features.shape == (2, 29959)
    start = 0
    for i, name in enumerate(order, 1):
        assert torch.all(features[:, start:start + dims[name]] == i)
        assert torch.all(scores[:, i - 1] == i)
        start += dims[name]
    gate = GatingNetwork(width, 7, profile="full_7")
    weights = gate(features)
    assert weights.shape == (2, 7)
    assert torch.allclose(weights.sum(1), torch.ones(2))
    prediction = (scores * weights).sum(dim=1, keepdim=True)
    expert_scores = torch.arange(
        1,
        8,
        dtype=weights.dtype,
        device=weights.device,
    )

    expected_prediction = (weights @ expert_scores).unsqueeze(1)

    assert prediction.shape == (2, 1)
    torch.testing.assert_close(prediction, expected_prediction)


def test_full7_rejects_full9_source_name():
    order, dims, _ = profile_contract("full_7")
    outputs = {name: (torch.zeros(1, 1), torch.zeros(1, dims[name])) for name in order}
    outputs["m2d_lora"] = outputs.pop("expert7")
    with pytest.raises(ValueError, match="registry mismatch"):
        validate_and_assemble_expert_outputs(outputs, profile="full_7")


@pytest.mark.parametrize("defect", ["missing", "width", "score_shape", "dtype", "nonfinite", "batch"])
def test_full7_rejects_invalid_component(defect):
    order, dims, _ = profile_contract("full_7")
    outputs = {name: (torch.zeros(2, 1), torch.zeros(2, dims[name])) for name in order}
    if defect == "missing":
        del outputs["expert6"]
    elif defect == "width":
        outputs["expert6"] = (torch.zeros(2, 1), torch.zeros(2, dims["expert6"] - 1))
    elif defect == "score_shape":
        outputs["expert6"] = (torch.zeros(2), outputs["expert6"][1])
    elif defect == "dtype":
        outputs["expert6"] = (torch.zeros(2, 1, dtype=torch.float64), outputs["expert6"][1])
    elif defect == "nonfinite":
        outputs["expert6"][0][0, 0] = float("nan")
    else:
        outputs["expert6"] = (torch.zeros(1, 1), torch.zeros(1, dims["expert6"]))
    with pytest.raises((ValueError, TypeError)):
        validate_and_assemble_expert_outputs(outputs, profile="full_7")
