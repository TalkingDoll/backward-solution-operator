import pytest
import numpy as np

from experiments._data_length import (_coordinates, _one_step, _valid_time,
                                       dictionary_rollout, fit_projected_edmd,
                                       gram_diagnostics)


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
