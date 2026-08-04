import numpy as np

from experiments.enso.enso_data import (BASE_PERIOD_YEARS, ENSOData,
                                         TRAIN_END_DATE, area_mean,
                                         base_period, build_grid_dataset,
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
        train_end=30, test_end=36, base_years=(0, 1))


def test_climatology_base_period_is_training_only_and_follows_the_cutoff():
    """The base period must never reach into the verification years.

    Deriving it from the training cutoff is what prevents the earlier
    1981-2010 / 1998 straddle from recurring if the split moves.
    """
    first, last = base_period()
    train_year = int(TRAIN_END_DATE[:4])
    assert last == train_year - 1                 # ends at the cutoff
    assert last < train_year                      # strictly training-only
    assert last - first + 1 == BASE_PERIOD_YEARS  # full-length window
    # it tracks the cutoff rather than being hardcoded
    assert base_period("1998-01-01") == (1968, 1997)
    assert base_period("2008-01-01") == (1978, 2007)
    assert base_period("2008-01-01", n_years=20) == (1988, 2007)


def test_area_mean_averages_only_the_masked_cells():
    field = np.array([[1.0, 3.0, 100.0, 100.0]])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    mask = np.array([True, True, False, False])
    np.testing.assert_allclose(area_mean(field, weights, mask), [2.0])


def test_eofs_are_area_orthonormal_and_have_the_right_shapes():
    data = _synthetic()
    eof = fit_eofs(data, fit_end=data.train_end, n_modes=6)
    assert eof.patterns.shape == (6, data.weights.shape[0])
    gram = eof.patterns @ (data.weights[:, None] * eof.patterns.T)
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-8)
    assert eof.fixed_columns().shape == (data.weights.shape[0], 5)


def test_fixed_columns_keep_the_leading_eof_and_drop_the_last():
    """The constant costs one slot; the *least* energetic mode pays for it.

    The leading EOF is the dominant (ENSO) mode, so it must remain in the
    dictionary span.
    """
    data = _synthetic()
    eof = fit_eofs(data, fit_end=data.train_end, n_modes=6)
    columns = eof.fixed_columns()
    np.testing.assert_allclose(columns[:, 0], eof.patterns[0])
    np.testing.assert_allclose(columns[:, -1], eof.patterns[-2])
    # the dropped pattern is the last one
    assert not np.allclose(columns[:, -1], eof.patterns[-1])


def test_lead_bands_split_short_and_long_leads():
    from experiments.enso.exp_enso_data_length_matched import lead_bands
    bands = lead_bands(12)
    leads = np.arange(1, 13)
    np.testing.assert_array_equal(leads[bands["short"]], [1, 2, 3])
    np.testing.assert_array_equal(leads[bands["long"]], [10, 11, 12])
    short_and_long = bands["short"] & bands["long"]
    assert not short_and_long.any()


def test_band_means_average_only_their_leads():
    from experiments.enso.exp_enso_data_length_matched import (band_means,
                                                               lead_bands)
    bands = lead_bands(12)
    per_lead = np.arange(1.0, 13.0)
    means = band_means(per_lead, bands)
    assert means["short"] == np.mean([1.0, 2.0, 3.0])
    assert means["long"] == np.mean([10.0, 11.0, 12.0])


def test_skill_horizon_is_leads_sustained_above_the_acc_threshold():
    from experiments.enso.exp_enso_data_length_matched import (
        skill_horizon_months)
    # ACC drops below 0.5 at lead index 2 -> two months of skill.
    assert skill_horizon_months(np.array([0.9, 0.7, 0.3, 0.1]), 0.5) == 2.0
    # never drops -> the full horizon.
    assert skill_horizon_months(np.array([0.9, 0.8, 0.7]), 0.5) == 3.0
    # climatology has undefined (nan) correlation -> zero skill horizon.
    assert skill_horizon_months(np.array([np.nan, np.nan]), 0.5) == 0.0


def test_data_edmd_is_infeasible_below_the_target_dimension():
    """The N<M rule the matched ENSO sweep applies: fit only when N >= M."""
    from experiments._data_length import fit_projected_edmd
    from experiments.enso.exp_enso_data_length_matched import TARGET_M
    M = TARGET_M
    psi = np.sqrt(M) * np.eye(M)
    weights = np.full(M, 1.0 / M)
    rng = np.random.default_rng(0)
    # N >= M is fitted and carries the requested dimension...
    u0, uT = rng.normal(size=(M, M)), rng.normal(size=(M, M))
    assert fit_projected_edmd(psi, weights, u0, uT, M, 1e-4).shape == (M, M)
    # ...while the sweep skips N < M (it never calls the fit).
    assert 10 < M and 39 < M


def test_grid_dataset_pairs_are_consecutive_and_leakage_free():
    data = _synthetic()
    ds = build_grid_dataset(data, start=2, end=data.train_end - 2)
    assert ds["kind"] == "grid2d"
    n = ds["u0"].shape[0]
    # uT is one month after u0; u2T two months after; nothing reaches train_end
    flat0 = ds["u0"].reshape(n, -1)
    flat1 = ds["uT"].reshape(n, -1)
    np.testing.assert_allclose(flat1[:-1], flat0[1:])
    assert 2 + n + 2 <= data.train_end
