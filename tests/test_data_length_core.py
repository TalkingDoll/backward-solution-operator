import numpy as np

from experiments._data_length import (_coordinates, _one_step, _valid_time,
                                       dictionary_rollout, fit_projected_edmd)


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


def test_valid_time_is_the_first_lead_exceeding_the_threshold():
    rel = np.array([[0.1, 0.1],
                    [0.2, 0.6],
                    [0.7, 0.1],
                    [0.9, 0.9]])                   # (H=4, n=2)
    vt = _valid_time(rel, step_time=2.0, threshold=0.5)
    np.testing.assert_allclose(vt, [4.0, 2.0])     # origin0 exceeds at lead 2
    never = np.full((3, 1), 0.1)
    np.testing.assert_allclose(_valid_time(never, 1.5, 0.5), [4.5])
