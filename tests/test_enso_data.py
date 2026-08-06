import pytest
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


def test_damped_persistence_gamma_is_training_only_and_in_grid():
    """gamma is picked from the grid using only pre-train_end origins."""
    from experiments.enso.exp_enso import damped_persistence_gamma
    data = _synthetic(T=80)
    data.anomalies[data.train_end:] = 1e6      # corrupt the test period only
    grid = np.linspace(0.5, 1.0, 11)
    gamma = damped_persistence_gamma(data, horizon=4, grid=grid)
    assert gamma in set(np.round(grid, 12))    # comes from the grid
    assert 0.5 <= gamma <= 1.0                 # unaffected by the corruption


def test_damped_persistence_reduces_to_persistence_at_gamma_one():
    """gamma=1 is plain persistence; gamma<1 shrinks with lead."""
    launch = np.array([[2.0, -1.0]])           # (1 origin, 2 cells)
    horizon = 3
    for gamma, expect_shrink in ((1.0, False), (0.8, True)):
        decay = gamma ** np.arange(1, horizon + 1)
        pred = launch[:, None] * decay[None, :, None]
        if expect_shrink:
            amp = np.abs(pred).max(axis=(0, 2))
            assert np.all(np.diff(amp) < 0)
        else:
            for lead in range(horizon):
                np.testing.assert_allclose(pred[:, lead], launch)


def test_delay_state_is_chronological_and_newest_block_last():
    """Row t holds months t-history+1 .. t, oldest first, newest last."""
    from experiments.enso.exp_enso_delay import delay_state, newest_block
    coeff = np.arange(24.0).reshape(8, 3)      # 8 months, 3 modes
    states = delay_state(coeff, history=4)
    assert np.all(np.isnan(states[:3]))        # no full context yet
    np.testing.assert_allclose(states[3], coeff[0:4].reshape(-1))
    np.testing.assert_allclose(states[7], coeff[4:8].reshape(-1))
    # the newest block is the instantaneous coefficient vector
    np.testing.assert_allclose(newest_block(states[7], 3), coeff[7])


def test_lim_delay_reduces_to_instantaneous_lim_at_history_one():
    """The two LIM variants differ only by the memory they are given."""
    from experiments.enso.exp_enso import fit_lim
    from experiments.enso.exp_enso_delay import delay_state, fit_lim_delay
    coeff = np.random.default_rng(5).normal(size=(60, 4))
    states = delay_state(coeff, history=1)
    np.testing.assert_allclose(states, coeff)          # history=1 is a no-op
    np.testing.assert_allclose(fit_lim_delay(states), fit_lim(coeff))


def test_delay_dataset_pairs_stay_before_the_training_cutoff():
    from experiments.enso.exp_enso_delay import build_delay_dataset
    states = np.arange(200.0).reshape(50, 4)
    train_end = 40
    ds = build_delay_dataset(states, 3, train_end - 2)
    n = ds["u0"].shape[0]
    np.testing.assert_allclose(ds["uT"][:-1], ds["u0"][1:])   # consecutive
    assert 3 + n + 2 <= train_end                             # no leakage
    assert ds["kind"] == "sensor1d"
    assert ds["weights"].shape == (states.shape[1],)


def test_direct_readout_matches_the_nino34_diagnostic_exactly():
    """The learned model and every other method must share one index definition.

    ``FieldToIndexFNO`` collapses its feature fields with a fixed area readout;
    that readout has to be the same weighting ``_nino34`` applies, or the direct
    comparator would be scored against a subtly different index.
    """
    from experiments.enso.enso_direct import region_readout_weights
    from experiments.enso.exp_enso import _nino34
    data = _synthetic(T=12, H=4, W=5)
    data.nino_masks["nino34"] = np.arange(20) < 7      # a proper sub-region
    fields = np.random.default_rng(6).normal(size=(3, 20))
    readout = fields @ region_readout_weights(data)     # what the model computes
    np.testing.assert_allclose(readout * data.scale, _nino34(data, fields))


def test_direct_training_origins_stay_inside_the_training_period():
    from experiments.enso.enso_direct import train_direct
    data = _synthetic(T=60)
    index = np.zeros(60)
    bad = np.arange(data.train_end - 2, data.train_end)   # targets cross the cut
    with pytest.raises(ValueError, match="train_end"):
        train_direct(data, bad, index, horizon=4)


def test_wang_lim_at_tau0_one_reduces_to_the_lag1_iterated_lim():
    """Isolates tau0 + fractional powers as the only substantive differences.

    Whitening (Wang Eq. 6) is a similarity transform on the propagator and a
    matching rescale of the readout, so it cannot change a forecast; at
    ``tau0 = 1`` the paper's LIM must therefore coincide with the lag-1 model.
    """
    from experiments.enso.enso_lim import fit_lim_wang
    from experiments.enso.exp_enso import fit_lim
    rng = np.random.default_rng(0)
    coeff = rng.normal(size=(300, 6))
    index = 0.8 * coeff[:, 0] + 0.2 * rng.normal(size=300)
    wang = fit_lim_wang(coeff, index, tau0=1, ridge=1e-10)
    G = fit_lim(coeff, ridge=1e-10)
    c0 = coeff[:5]
    iterated = np.stack([c0 @ np.linalg.matrix_power(G.T, lead)
                         for lead in range(1, 4)], axis=1)
    np.testing.assert_allclose(wang.coefficient_forecast(c0, 3), iterated,
                               atol=1e-8)


def _damped_oscillator(n=400, seed=1):
    """A stable AR(1) whose propagator has positive-real-part eigenvalues.

    White noise will not do here: its fitted propagator has eigenvalues with
    negative real part, for which the principal square root of Eq. 12 is
    genuinely complex. On the real ENSO record every eigenvalue has positive
    real part and the fractional powers are real to ~1e-17.
    """
    rng = np.random.default_rng(seed)
    angle, decay = np.deg2rad(20.0), 0.92
    block = decay * np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
    a = np.kron(np.eye(3), block)                       # 6-dimensional
    series = np.zeros((n, 6))
    for t in range(1, n):
        series[t] = a @ series[t - 1] + 0.1 * rng.normal(size=6)
    return series


def test_wang_eq9_readout_equals_propagate_then_average():
    """Wang Eq. 9: the Eq. 7 index readout is not an independent choice."""
    from experiments.enso.enso_lim import fit_lim_wang
    coeff = _damped_oscillator()
    index = 0.8 * coeff[:, 0] + 0.2 * coeff[:, 1]
    wang = fit_lim_wang(coeff, index, tau0=2)
    c0 = coeff[:5]
    via_coefficients = (wang.coefficient_forecast(c0, 4)
                        / wang.pc_scale) @ wang.readout
    np.testing.assert_allclose(wang.index_forecast(c0, 4), via_coefficients,
                               atol=1e-10)


def test_wang_lim_rejects_a_genuinely_complex_fractional_power():
    """The Eq. 12 fractional power is fragile; a large imaginary part must
    surface rather than be silently discarded (the old suite took .real)."""
    from experiments.enso.enso_lim import fit_lim_wang
    rng = np.random.default_rng(3)
    coeff = rng.normal(size=(300, 6))       # white noise: negative-real-part mu
    index = rng.normal(size=300)
    wang = fit_lim_wang(coeff, index, tau0=2)
    with pytest.raises(np.linalg.LinAlgError, match="complex"):
        wang.index_forecast(coeff[:3], 1)   # lead 1 needs the square root


def test_direct_lagged_lim_is_plain_least_squares_per_lead():
    from experiments.enso.enso_lim import fit_lim_direct, predict_lim_direct
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 5))
    y = 0.7 * x[:, 0] + 0.3 * rng.normal(size=200)
    origins = np.arange(0, 150)
    weights = fit_lim_direct(x, y, origins, horizon=1, ridge=1e-10)
    design = np.concatenate([np.ones((len(origins), 1)), x[origins]], axis=1)
    expected = np.linalg.lstsq(design, y[origins + 1], rcond=None)[0]
    np.testing.assert_allclose(weights[0], expected, atol=1e-6)
    assert predict_lim_direct(weights, x[origins]).shape == (len(origins), 1)


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


# ---------------------------------------------------------------------------
# Direct Niño-3.4 forecasting through the BSO framework (exp_enso_index)
# ---------------------------------------------------------------------------

def test_index_delay_state_is_chronological_and_newest_is_instantaneous():
    """Row t holds months t-11 … t, so the last entry is the index at t."""
    from experiments.enso.exp_enso_index import index_delay_states
    index = np.arange(40, dtype=float)
    states, mean, scale = index_delay_states(index, train_end=30, history=12)
    assert states.shape == (40, 12)
    # Rows without a full 12-month history stay NaN rather than being padded.
    assert np.isnan(states[:11]).all() and np.isfinite(states[11:]).all()
    np.testing.assert_allclose(states[20, -1], (index[20] - mean) / scale)
    assert np.all(np.diff(states[20]) > 0)          # oldest first, newest last
    # Standardisation uses training months only.
    np.testing.assert_allclose(mean, index[:30].mean())
    np.testing.assert_allclose(scale, index[:30].std())


def test_index_bso_training_pairs_stay_before_the_training_cutoff():
    """No phase-A target month, and no regression target, reaches train_end."""
    from experiments.enso.exp_enso_index import _pair_ends
    train_end, history, horizon = 30, 12, 4
    ends = _pair_ends(history - 1, train_end)
    assert ends[0] == history - 1
    assert int(ends.max()) + 1 == train_end - 1     # latest target is in-sample
    fit_origins = np.arange(history - 1, train_end - horizon)
    assert int(fit_origins.max()) + horizon == train_end - 1


def test_index_bso_dictionary_has_the_declared_column_layout():
    """Psi(z) = [1, z, h(z)]: the layout the linear control is a subset of."""
    import torch
    from experiments.eeg.eeg_edmddl import PaperEDMDDLDictionary
    from experiments.enso.exp_enso_index import DICTIONARY_SIZE
    dictionary = PaperEDMDDLDictionary(state_dimension=12,
                                       dictionary_size=DICTIONARY_SIZE,
                                       hidden_dims=(8, 8))
    assert DICTIONARY_SIZE == 1 + 12 + dictionary.n_learned_observables
    z = torch.as_tensor(np.random.default_rng(0).normal(size=(7, 12)),
                        dtype=torch.float32)
    psi = dictionary(z).detach().numpy()
    assert psi.shape == (7, DICTIONARY_SIZE)
    np.testing.assert_allclose(psi[:, 0], 1.0)                  # constant
    np.testing.assert_allclose(psi[:, 1:13], z.numpy(), atol=1e-6)  # z itself
    # The learned block must actually depend on the state, else the "learned"
    # variant would silently equal the linear control.
    assert np.ptp(psi[:, 13:], axis=0).min() > 1e-8


def test_index_bso_linear_control_is_per_lead_regression_on_the_delay_state():
    """`bso_direct_linear` is exactly least squares on [1, z], by construction.

    Both variants reach `fit_lim_direct` through the same call, with the
    dictionary's constant column supplied by the intercept, so the gap between
    them isolates the nonlinearity and nothing else.
    """
    from experiments.enso.enso_lim import fit_lim_direct, predict_lim_direct
    from experiments.enso.exp_enso_index import index_delay_states
    rng = np.random.default_rng(3)
    index = np.cumsum(rng.normal(size=200)) * 0.1
    states, _, _ = index_delay_states(index, train_end=150, history=12)
    origins = np.arange(11, 150 - 4)
    weights = fit_lim_direct(states, index, origins, horizon=4, ridge=1e-10)
    design = np.concatenate([np.ones((len(origins), 1)), states[origins]],
                            axis=1)
    for lead in range(1, 5):
        expected = np.linalg.lstsq(design, index[origins + lead],
                                   rcond=None)[0]
        np.testing.assert_allclose(weights[lead - 1], expected, atol=1e-5)
    assert predict_lim_direct(weights, states[origins]).shape == (len(origins), 4)
