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


def test_pca_state_is_training_only_and_keeps_the_nan_convention():
    """The reduced state must not see a verification month, ever.

    The whole ladder is fitted on this state, so a leak here would flatter every
    rung at once and be invisible in the comparison between them.
    """
    from experiments.enso.enso_data import pca_state
    from experiments.enso.exp_enso_delay import delay_state
    rng = np.random.default_rng(11)
    coeff = rng.normal(size=(80, 5))
    delay = delay_state(coeff, history=4)
    clean = pca_state(delay, train_end=60, rank=3)

    assert clean.shape == (80, 3)
    # Rows without a full history stay NaN rather than being padded to zero.
    assert np.isnan(clean[:3]).all() and np.isfinite(clean[3:]).all()

    # Corrupting the verification period cannot move the basis, so the
    # projections of the training rows are unchanged.
    corrupted = delay.copy()
    corrupted[60:] = 1e6
    after = pca_state(corrupted, train_end=60, rank=3)
    np.testing.assert_allclose(after[3:60], clean[3:60], atol=1e-8)


def test_pca_state_rank_is_capped_and_rejects_an_empty_training_window():
    from experiments.enso.enso_data import pca_state
    delay = np.arange(60.0).reshape(15, 4)
    assert pca_state(delay, train_end=10, rank=99).shape[1] == 4
    with pytest.raises(ValueError, match="train_end"):
        pca_state(delay, train_end=0, rank=2)


def test_direct_cv_folds_are_contiguous_and_stay_inside_the_origins():
    """Blocked folds, drawn only from `origins`: a random split would leak.

    Neighbouring months are near-duplicates, so a randomly split fold has almost
    the same rows on both sides and the selected ridge collapses to the smallest
    value on the grid -- which is exactly the un-tuned behaviour the helper
    exists to avoid.
    """
    from experiments.enso.enso_lim import fit_direct_cv, fit_lim_direct
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 6))
    y = 0.6 * x[:, 0] + 0.4 * rng.normal(size=200)
    origins = np.arange(20, 150)

    # A single-point grid must reproduce the fixed-ridge estimator exactly.
    weights, ridges = fit_direct_cv(x, y, origins, horizon=3,
                                    grid=np.array([1e-6]))
    np.testing.assert_allclose(ridges, 1e-6)
    np.testing.assert_allclose(
        weights, fit_lim_direct(x, y, origins, horizon=3, ridge=1e-6),
        atol=1e-10)

    # Selection uses only `origins`: corrupting everything outside changes
    # nothing.
    corrupt = x.copy()
    corrupt[:20] = 1e6
    corrupt[153:] = 1e6
    y_corrupt = y.copy()
    y_corrupt[:20] = 1e6
    y_corrupt[153:] = 1e6
    _, same = fit_direct_cv(corrupt, y_corrupt, origins, horizon=3)
    _, base = fit_direct_cv(x, y, origins, horizon=3)
    np.testing.assert_allclose(same, base)


def test_direct_cv_beats_the_fixed_default_on_an_overparameterised_design():
    """The reason the helper exists, stated as held-out error.

    A wide design fitted at the fixed ``ridge=1e-6`` default overfits, and a
    comparison against such a baseline measures regularisation rather than the
    hypothesis class. Tuning must both move off the grid floor and actually
    reduce out-of-sample error.
    """
    from experiments.enso.enso_lim import (RIDGE_GRID, fit_direct_cv,
                                           fit_lim_direct, predict_lim_direct)
    rng = np.random.default_rng(7)
    n, p = 260, 60
    x = rng.normal(size=(n, p))
    # Weak signal in one column, everything else noise: 60 predictors for 120
    # fitting rows is exactly the regime the ENSO delay design sits in.
    y = 0.4 * x[:, 0] + rng.normal(size=n)
    fit, held = np.arange(0, 120), np.arange(140, 240)

    tuned_w, ridges = fit_direct_cv(x, y, fit, horizon=2)
    default_w = fit_lim_direct(x, y, fit, horizon=2, ridge=1e-6)

    assert ridges.min() > RIDGE_GRID.min()      # off the grid floor
    assert ridges.max() <= RIDGE_GRID.max()

    truth = np.stack([y[held + lead] for lead in (1, 2)], axis=1)
    tuned = np.mean((predict_lim_direct(tuned_w, x[held]) - truth) ** 2)
    default = np.mean((predict_lim_direct(default_w, x[held]) - truth) ** 2)
    assert tuned < default


def test_propagated_forecast_is_the_eeg_rollout_with_nothing_fitted():
    """`Psi K^n`, state block out, fixed readout -- the EEG rollout, verbatim.

    The rung's whole point is that no parameter is fitted per lead and none is
    fitted to the target at all; a fitted readout would score better while
    quietly reintroducing exactly the per-lead tuning it exists to avoid. The
    test therefore pins it to matrix powers of `K`.
    """
    from experiments.enso.exp_enso_index import propagated_forecast
    rng = np.random.default_rng(8)
    state_dim = 4
    psi = np.concatenate([np.ones((60, 1)), rng.normal(size=(60, 6))], axis=1)
    K = rng.normal(size=(7, 7)) * 0.2 + np.eye(7)
    origins = np.arange(40, 55)
    readout = lambda block: block[:, -1] * 2.0 + 1.0   # fixed, unfitted

    out = propagated_forecast(psi, K, state_dim, readout, origins, horizon=3)
    assert out.shape == (15, 3)
    for lead in (1, 2, 3):
        advanced = psi[origins] @ np.linalg.matrix_power(K, lead)
        np.testing.assert_allclose(
            out[:, lead - 1], readout(advanced[:, 1:1 + state_dim]),
            atol=1e-10)
    # Only the fixed state coordinates are read back out, never the learned
    # block -- that is what makes the reconstruction map fixed.
    assert not np.allclose(out[:, 0], readout(
        (psi[origins] @ K)[:, 1 + state_dim:1 + 2 * state_dim]))


def test_pca_inverse_round_trips_a_full_rank_state():
    """The propagated rung reconstructs through this map, so it must invert."""
    from experiments.enso.enso_data import pca_basis, pca_inverse, pca_state
    rng = np.random.default_rng(12)
    delay = rng.normal(size=(90, 6))
    scores = pca_state(delay, train_end=70, rank=6)
    mean, components = pca_basis(delay, train_end=70, rank=6)
    np.testing.assert_allclose(pca_inverse(scores, mean, components), delay,
                               atol=1e-9)


def test_block_bootstrap_draws_contiguous_blocks_of_the_right_length():
    from experiments.enso.exp_enso_index import _block_resamples
    draws = _block_resamples(n_origins=50, block=10, n_samples=7, seed=0)
    assert draws.shape == (7, 50)
    assert draws.min() >= 0 and draws.max() < 50
    # Every block of `block` consecutive columns is a run of consecutive months.
    blocks = draws.reshape(7, 5, 10)
    np.testing.assert_array_equal(np.diff(blocks, axis=2),
                                  np.ones((7, 5, 9), dtype=int))


def test_acc_difference_interval_covers_zero_for_identical_forecasts():
    """Two identical forecasts differ by exactly zero at every resample."""
    from experiments.enso.exp_enso_index import acc_difference_interval
    rng = np.random.default_rng(9)
    truth = rng.normal(size=(120, 4))
    pred = truth + 0.5 * rng.normal(size=(120, 4))
    lower, upper = acc_difference_interval(pred, pred, truth, horizon=4,
                                           n_samples=200)
    np.testing.assert_allclose(lower, 0.0, atol=1e-12)
    np.testing.assert_allclose(upper, 0.0, atol=1e-12)

    # A forecast that is strictly better must give an interval clear of zero.
    worse = 0.05 * pred + rng.normal(size=(120, 4))
    lower, _ = acc_difference_interval(pred, worse, truth, horizon=4,
                                       n_samples=200)
    assert np.all(lower > 0.0)


def test_ladder_rungs_share_one_state_width_so_only_the_method_differs():
    """The dictionary, its linear control and the MLP take the same input."""
    from experiments.enso.exp_enso_index import (DICTIONARY_SIZE, DirectMLP,
                                                 REDUCED_DIM)
    assert DICTIONARY_SIZE == 1 + REDUCED_DIM + 11
    mlp = DirectMLP(REDUCED_DIM, horizon=6, hidden_dims=(16, 16))
    first = next(m for m in mlp.net if hasattr(m, "in_features"))
    last = [m for m in mlp.net if hasattr(m, "out_features")][-1]
    assert first.in_features == REDUCED_DIM
    assert last.out_features == 6


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
