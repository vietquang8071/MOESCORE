import pytest
import torch

from loss.domain_spearman import SoftSpearmanLoss, DomainWiseSoftSpearmanLoss, HybridSoftSpearmanLoss, _average_ranks
from loss.l2_soft_rank import l2_soft_rank


def test_soft_spearman_orientation_ties_and_gradients():
    loss = SoftSpearmanLoss()
    target = torch.tensor([0., 1., 1., 3., 4.])
    prediction = torch.tensor([0., 1., 2., 3., 4.], requires_grad=True)
    forward = loss(prediction, target)
    reverse = loss(-prediction, target)
    assert forward < reverse
    assert 0 <= forward < 0.1
    assert 1.9 < reverse <= 2
    forward.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_soft_spearman_constant_prediction_and_invalid_target():
    loss = SoftSpearmanLoss()
    prediction = torch.ones(9, requires_grad=True)
    result = loss(prediction, torch.arange(9.))
    result.backward()
    assert torch.isfinite(result) and torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError, match="constant target"):
        loss(prediction, torch.ones(9))
    near = (1 + torch.arange(9., dtype=torch.float64) * 1e-7).requires_grad_()
    near_loss = loss(near, torch.arange(9., dtype=torch.float64))
    near_loss.backward()
    assert torch.isfinite(near_loss) and torch.isfinite(near.grad).all()
    assert _average_ranks(torch.tensor([3., 1., 1., 4.])).tolist() == [3., 1.5, 1.5, 4.]


def test_domain_loss_excludes_constant_target_and_hybrid_is_exact_half_mix():
    prediction = torch.arange(32., requires_grad=True)
    targets = torch.cat((torch.arange(16.), torch.ones(16)))
    domains = torch.tensor([0] * 16 + [1] * 16)
    domain_loss, valid, invalid = DomainWiseSoftSpearmanLoss()(prediction, targets, domains)
    global_loss, _, total, details = HybridSoftSpearmanLoss()(prediction, targets, domains)
    assert valid == [0] and invalid == [1]
    assert torch.allclose(total, 0.5 * (global_loss + domain_loss))
    assert details["valid_domains"] == [0]
    assert details["invalid_reasons"] == {1: "constant_target"}
    with pytest.raises(ValueError, match="all domain groups"):
        DomainWiseSoftSpearmanLoss()(prediction, torch.ones(32), domains)


def test_l2_rank_matches_permutahedron_projection_and_finite_difference():
    # All four entries pool in one isotonic block: exact projection is affine.
    values = torch.tensor([0., .02, .04, .06], dtype=torch.float64, requires_grad=True)
    ranks = l2_soft_rank(values, 0.1)
    expected = 2.5 + (values - values.mean()) / 0.1
    assert torch.allclose(ranks, expected, atol=1e-10)
    weighted = (ranks * torch.tensor([1., 0., 0., -1.], dtype=torch.float64)).sum()
    weighted.backward()
    epsilon = 1e-6
    displaced = values.detach().clone()
    displaced[0] += epsilon
    plus = (l2_soft_rank(displaced, 0.1) * torch.tensor([1., 0., 0., -1.])).sum()
    displaced[0] -= 2 * epsilon
    minus = (l2_soft_rank(displaced, 0.1) * torch.tensor([1., 0., 0., -1.])).sum()
    assert values.grad[0].item() == pytest.approx(((plus - minus) / (2 * epsilon)).item(), rel=1e-5)


def test_l2_rank_matches_independent_isotonic_solver():
    from scipy.optimize import isotonic_regression

    for seed in range(5):
        values = torch.randn(32, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        sorted_values, order = torch.sort(values / 0.1, descending=True, stable=True)
        w = torch.arange(32, 0, -1, dtype=torch.float64)
        projected = sorted_values.numpy() - isotonic_regression(
            (sorted_values - w).numpy(), increasing=False
        ).x
        inverse = torch.argsort(order)
        assert torch.allclose(l2_soft_rank(values, 0.1), torch.tensor(projected)[inverse], atol=1e-10)
