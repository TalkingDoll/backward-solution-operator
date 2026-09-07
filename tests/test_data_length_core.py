import inspect

import pytest
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from experiments._data_length import (_coordinates, _mark_M, _method_label,
                                       _one_step, _valid_time,
                                       dictionary_rollout, fit_projected_edmd,
                                       gram_diagnostics, operator_ensemble,
                                       run_sweep)


@pytest.mark.parametrize(
    ("method", "target_m", "expected"),
    [
        ("data_edmd_npairs", 20, "Data EDMD (rank = min(n, 20))"),
        ("data_edmd_npairs", 40, "Data EDMD (rank = min(n, 40))"),
        ("data_edmd_forced_ridge", 20,
         "Data EDMD (rank = 20, ridge)"),
        ("data_edmd_forced_pinv", 40,
         "Data EDMD (rank = 40, min-norm)"),
    ],
)
def test_rank_labels_include_the_selected_target(method, target_m, expected):
    assert _method_label(method, target_m, rank_labels=True) == expected


def test_target_reference_line_can_be_drawn_without_an_annotation():
    fig, ax = plt.subplots()
    try:
        _mark_M(ax, 20, annotate=False)
        assert len(ax.lines) == 1
        assert list(ax.lines[0].get_xdata()) == [20, 20]
        assert not ax.texts
    finally:
        plt.close(fig)


class _Identity(nn.Module):
    def forward(self, fields, grid):
        return fields


class _Blowup(nn.Module):
    """Multiplies by 10 each step, so rolled states leave the RMS band."""

    def forward(self, fields, grid):
        return fields * 10.0


class _Result:
    """The two attributes ``fno_rollout`` reads off a ``BSOResult``."""

    def __init__(self, model, p):
        self.operator_pure = model
        self.grid = np.linspace(0.0, 1.0, p, endpoint=False)


def _orthonormal_setup(M):
    """psi and weights with encode/decode exact inverses: psi=sqrt(M) I, w=1/M."""
    psi = np.sqrt(M) * np.eye(M)
    weights = np.full(M, 1.0 / M)
    return psi, weights


def test_projected_edmd_recovers_a_pinned_linear_system():
    rng = np.random.default_rng(0)
    M, N = 5, 200
    psi, w = _orthonormal_setup(M)
    K = rng.normal(size=(M, M))
    K[:, 0] = 0.0                      # pinned constant mode
    K[0, 0] = 1.0
    c0 = rng.normal(size=(M, N))
    c1 = K.T @ c0                      # convention: U Psi ~= K^T Psi
    u0 = (psi.T @ c0).T                # decode -> fields (N, M)
    uT = (psi.T @ c1).T
    fit = fit_projected_edmd(psi, w, u0, uT, dim=M, reg_alpha=1e-12)
    np.testing.assert_allclose(fit, K, atol=1e-6)


def test_projected_edmd_dimension_matches_request():
    rng = np.random.default_rng(1)
    M = 8
    psi, w = _orthonormal_setup(M)
    u0 = rng.normal(size=(50, M))
    uT = rng.normal(size=(50, M))
    for dim in (2, 4, 8):
        k = fit_projected_edmd(psi, w, u0, uT, dim=dim, reg_alpha=1e-6)
        assert k.shape == (dim, dim)
        np.testing.assert_allclose(k[:, 0], np.eye(dim)[:, 0])  # constant pinned


def test_one_step_is_zero_for_an_exactly_linear_holdout():
    rng = np.random.default_rng(2)
    M = 4
    psi, w = _orthonormal_setup(M)
    K = rng.normal(size=(M, M))
    K[:, 0] = 0.0
    K[0, 0] = 1.0
    c0 = rng.normal(size=(M, 30))
    test_u0 = (psi.T @ c0).T
    test_u1 = (psi.T @ (K.T @ c0)).T
    assert _one_step(psi, K, w, test_u0, test_u1) < 1e-9


def test_coordinates_invert_the_decoder():
    M = 6
    psi, w = _orthonormal_setup(M)
    fields = np.random.default_rng(3).normal(size=(12, M))
    coords = _coordinates(psi, w, fields)          # (M, 12)
    reconstructed = (psi.T @ coords).T
    np.testing.assert_allclose(reconstructed, fields, atol=1e-10)


def test_dictionary_rollout_reproduces_an_exact_linear_trajectory():
    rng = np.random.default_rng(4)
    M, n, H = 4, 5, 6
    psi, w = _orthonormal_setup(M)
    K = rng.normal(size=(M, M))
    K[:, 0] = 0.0
    K[0, 0] = 1.0
    c0 = rng.normal(size=(M, n))
    u0 = (psi.T @ c0).T
    truth = []
    c = c0
    for _ in range(H):
        c = K.T @ c
        truth.append((psi.T @ c).T)
    truth = np.stack(truth)                        # (H, n, M)
    pred = dictionary_rollout(psi, K, w, u0, H)
    np.testing.assert_allclose(pred, truth, atol=1e-8)


def test_pinv_solver_recovers_a_full_rank_system_exactly():
    """With n_s >= m and no ridge, the min-norm fit is the exact transition."""
    rng = np.random.default_rng(10)
    M, N = 5, 200
    psi, w = _orthonormal_setup(M)
    K = rng.normal(size=(M, M))
    K[:, 0] = 0.0
    K[0, 0] = 1.0
    c0 = rng.normal(size=(M, N))
    u0 = (psi.T @ c0).T
    uT = (psi.T @ (K.T @ c0)).T
    fit = fit_projected_edmd(psi, w, u0, uT, dim=M, reg_alpha=0.0,
                             solver="pinv")
    np.testing.assert_allclose(fit, K, atol=1e-8)


def test_forced_fit_below_m_is_rank_deficient_but_still_returns_a_matrix():
    """The N<M regime the forced curves probe: rank(G) <= n_s, not m."""
    rng = np.random.default_rng(11)
    M, N = 8, 3                                   # far fewer pairs than columns
    psi, w = _orthonormal_setup(M)
    u0 = rng.normal(size=(N, M))
    uT = rng.normal(size=(N, M))
    rank, cond = gram_diagnostics(psi, w, u0, dim=M)
    assert rank <= N < M                          # measurably rank deficient
    assert not np.isfinite(cond) or cond > 1e12   # singular Gram
    for solver in ("ridge", "pinv"):
        fit = fit_projected_edmd(psi, w, u0, uT, dim=M, reg_alpha=1e-4,
                                 solver=solver)
        assert fit.shape == (M, M)                # defined, just not determined
        assert np.isfinite(fit).all()
        np.testing.assert_allclose(fit[:, 0], np.eye(M)[:, 0])


def test_unknown_solver_is_rejected():
    psi, w = _orthonormal_setup(4)
    x = np.random.default_rng(12).normal(size=(10, 4))
    with pytest.raises(ValueError, match="solver"):
        fit_projected_edmd(psi, w, x, x, dim=4, reg_alpha=1e-6, solver="lstsq")


def test_valid_time_is_the_first_lead_exceeding_the_threshold():
    rel = np.array([[0.1, 0.1],
                    [0.2, 0.6],
                    [0.7, 0.1],
                    [0.9, 0.9]])                   # (H=4, n=2)
    vt = _valid_time(rel, step_time=2.0, threshold=0.5)
    np.testing.assert_allclose(vt, [4.0, 2.0])     # origin0 exceeds at lead 2
    never = np.full((3, 1), 0.1)
    np.testing.assert_allclose(_valid_time(never, 1.5, 0.5), [4.5])


def _unit_states(n, p, rng):
    """States on the weighted unit sphere, matching an attractor RMS of 1."""
    x = rng.normal(size=(n, p))
    return x / np.sqrt(np.mean(x ** 2, axis=1, keepdims=True))


def test_operator_ensemble_pads_small_n_to_the_seed_target():
    p, n_obs, depth, target = 8, 3, 2, 256
    rng = np.random.default_rng(0)
    result = _Result(_Identity(), p)
    ens = operator_ensemble(result, "sensor1d", _unit_states(n_obs, p, rng),
                            np.full(p, 1.0 / p), seed=7, sigma=0.0,
                            target_seeds=target, depth=depth)
    # 3 observed states padded by ceil(256/3)=86 noise-free copies, then rolled
    # `depth` times under the identity: every state survives the RMS filter.
    assert len(ens) == n_obs * (1 + 86) * (1 + depth)


def test_operator_ensemble_is_reproducible_per_seed():
    p = 8
    seeds = _unit_states(4, p, np.random.default_rng(1))
    w = np.full(p, 1.0 / p)
    kw = dict(seed=2026, target_seeds=32, depth=1)
    a = operator_ensemble(_Result(_Identity(), p), "sensor1d", seeds, w, **kw)
    b = operator_ensemble(_Result(_Identity(), p), "sensor1d", seeds, w, **kw)
    np.testing.assert_array_equal(a, b)
    kw["seed"] = 2027
    assert not np.array_equal(a, operator_ensemble(
        _Result(_Identity(), p), "sensor1d", seeds, w, **kw))


def test_operator_ensemble_drops_diverged_states():
    """A blowing-up operator contributes nothing past the RMS band."""
    p, n_obs = 8, 4
    seeds = _unit_states(n_obs, p, np.random.default_rng(2))
    ens = operator_ensemble(_Result(_Blowup(), p), "sensor1d", seeds,
                            np.full(p, 1.0 / p), seed=0, sigma=0.0,
                            target_seeds=n_obs, depth=3)
    # seeds have RMS 1; every rolled state is at 10, 100, 1000 -- all rejected.
    rms = np.sqrt(np.sum(np.full(p, 1.0 / p)[None] * ens ** 2, axis=1))
    assert np.all(rms < 1.5)
    assert len(ens) == n_obs * 2          # the seeds and their one copy


def test_operator_ensemble_rejects_non_2d_seed_states():
    with pytest.raises(ValueError, match="must be"):
        operator_ensemble(_Result(_Identity(), 4), "sensor1d",
                          np.zeros(4), np.full(4, 0.25), seed=0)


def test_run_sweep_defaults_to_the_quadrature_bso():
    """ENSO and EEG share run_sweep; the ensemble path must be opt-in."""
    assert (inspect.signature(run_sweep).parameters["bso_source"].default
            == "quadrature")


def test_run_sweep_rejects_an_unknown_bso_source():
    with pytest.raises(ValueError, match="bso_source"):
        run_sweep(system="x", make_dataset_n=lambda n, s: {},
                  test_traj=np.zeros((1, 2, 3)), weights=np.ones(3),
                  kind="sensor1d", cfg_builder=lambda n, s: {}, target_M=3,
                  lengths=(1,), seeds=(0,), checkpoint_root=None,
                  reg_alpha=1e-4, bso_source="quadratute")
