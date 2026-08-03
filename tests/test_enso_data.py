import numpy as np

from experiments.enso.enso_data import (ENSOData, area_mean, build_grid_dataset,
                                         fit_eofs)


def _synthetic(T=40, H=4, W=5, seed=0):
    rng = np.random.default_rng(seed)
    p = H * W
    weights = np.full(p, 1.0 / p)
    anomalies = rng.normal(size=(T, p))
    grid = np.zeros((H, W, 2))
    return ENSOData(
        times=np.arange(T).astype("datetime64[M]"),
        lat=np.arange(H).astype(float), lon=np.arange(W).astype(float),
        shape=(H, W), anomalies=anomalies, weights=weights, fno_grid=grid,
        basis_coords=grid.reshape(-1, 2),
        nino_masks={"nino34": weights > 0}, scale=1.0,
        dev_end=30, val_start=10, primary_end=36)


def test_area_mean_averages_only_the_masked_cells():
    field = np.array([[1.0, 3.0, 100.0, 100.0]])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    mask = np.array([True, True, False, False])
    np.testing.assert_allclose(area_mean(field, weights, mask), [2.0])


def test_eofs_are_area_orthonormal_and_have_the_right_shapes():
    data = _synthetic()
    eof = fit_eofs(data, fit_end=data.dev_end, n_modes=6)
    assert eof.patterns.shape == (6, data.weights.shape[0])
    gram = eof.patterns @ (data.weights[:, None] * eof.patterns.T)
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-8)
    assert eof.fixed_columns().shape == (data.weights.shape[0], 5)


def test_grid_dataset_pairs_are_consecutive_and_leakage_free():
    data = _synthetic()
    ds = build_grid_dataset(data, start=2, end=data.dev_end - 2)
    assert ds["kind"] == "grid2d"
    n = ds["u0"].shape[0]
    # uT is one month after u0; u2T two months after; nothing reaches dev_end
    flat0 = ds["u0"].reshape(n, -1)
    flat1 = ds["uT"].reshape(n, -1)
    np.testing.assert_allclose(flat1[:-1], flat0[1:])
    assert 2 + n + 2 <= data.dev_end
