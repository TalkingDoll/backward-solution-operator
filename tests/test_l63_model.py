import numpy as np

from experiments.lorenz63.exp_lorenz63 import (
    _polynomial_exponents,
    _stabilize_transition,
    bso_rollout,
)


def test_l63_dictionary_feature_count() -> None:
    exponents = _polynomial_exponents(5, 4)
    assert exponents.shape == (120, 5)
    assert np.all(exponents.sum(axis=1) >= 2)
    assert np.all(exponents.sum(axis=1) <= 4)


def test_l63_stability_cap_preserves_stable_mode() -> None:
    transition = np.diag([1.1, 0.8])
    stable, raw_radius, _ = _stabilize_transition(transition, 0.97)
    assert np.isclose(raw_radius, 1.1)
    assert np.allclose(np.diag(stable), [0.97, 0.8])


def test_l63_bso_rollout_matches_linear_recurrence() -> None:
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
