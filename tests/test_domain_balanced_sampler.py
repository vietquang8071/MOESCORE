from collections import Counter

from training.domain_balanced_sampler import DomainBalancedBatchSampler


def test_balanced_batches_and_epoch_resume():
    ids = [domain for domain in range(9) for _ in range(40)]
    sampler = DomainBalancedBatchSampler(ids, seed=2027, steps_per_epoch=59)
    batches = list(sampler)
    assert len(batches) == 59
    appearances = Counter()
    for batch in batches:
        assert len(batch) == 128 and len(set(batch)) == 128
        counts = Counter(ids[i] for i in batch)
        assert len(counts) == 4 and set(counts.values()) == {32}
        appearances.update(counts.keys())
    assert set(appearances) == set(range(9))
    assert set(appearances.values()) <= {26, 27}
    state = sampler.state_dict()
    next_batches = list(sampler)
    resumed = DomainBalancedBatchSampler(ids, seed=2027, steps_per_epoch=59)
    resumed.load_state_dict(state)
    assert list(resumed) == next_batches
    assert sampler.cumulative_unique_count == resumed.cumulative_unique_count
