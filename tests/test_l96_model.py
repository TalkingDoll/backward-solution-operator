import numpy as np

from experiments.lorenz96.exp_lorenz96 import (
    _band_means,
    _polynomial_exponents,
    _stabilize_transition,
    bso_rollout,
)


def test_l96_dictionary_feature_count() -> None:
    exponents = _polynomial_exponents(4, 2)
    assert exponents.shape == (10, 4)
    assert np.all(exponents.sum(axis=1) == 2)
    assert 20 + len(exponents) == 30


def test_l96_stability_cap_preserves_stable_mode() -> None:
    transition = np.diag([1.1, 0.8])
    stable, raw_radius, _ = _stabilize_transition(transition, 0.99)
    assert np.isclose(raw_radius, 1.1)
    assert np.allclose(np.diag(stable), [0.99, 0.8])


def test_l96_bso_rollout_matches_linear_recurrence() -> None:
    model = {
        "field_mean": np.zeros(2),
        "field_scale": np.ones(2),
        "pca": np.asarray([[1.0], [0.0]]),
        "score_scale": np.ones(1),
        "exponents": np.empty((0, 1), dtype=np.int16),
        "feature_mean": np.zeros(2),
        "feature_scale": np.ones(2),
        "decoder": np.eye(2),
        "A": np.diag([0.9, 0.8]),
    }
    rollout = bso_rollout(model, np.asarray([[1.0, 2.0]]), 2)
    expected = np.asarray([[[1.0, 2.0]], [[0.9, 1.6]], [[0.81, 1.28]]])
    assert np.allclose(rollout, expected)


def test_l96_band_means_are_disjoint() -> None:
    times = np.asarray([0.0, 0.3, 0.32, 1.0, 1.02, 6.0])
    errors = np.asarray([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    means = _band_means(times, errors)
    assert np.isclose(means["short_t_le_0_3"], 2.0)
    assert np.isclose(means["mid_0_3_lt_t_le_1"], 6.0)
    assert np.isclose(means["long_1_lt_t_le_6"], 10.0)
