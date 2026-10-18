"""Small, deterministic checks for the SST physical-score definitions."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.sst.exp_sst import (
    _weighted_acc,
    _weighted_bias,
    _weighted_mae,
    _weighted_rmse,
    _weighted_std_ratio,
)


def test_weighted_rmse_and_mae_average_over_cases_and_spatial_weights():
    truth = np.zeros((2, 1, 2), dtype=np.float64)
    prediction = np.asarray(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([[0.25, 0.75]], dtype=np.float64)

    # Weighted squared-error sum is 17.5; weighted absolute-error sum is 5.5.
    assert _weighted_rmse(prediction, truth, weights) == pytest.approx(
        np.sqrt(17.5 / 2.0)
    )
    assert _weighted_mae(prediction, truth, weights) == pytest.approx(5.5 / 2.0)


def test_weighted_bias_retains_the_sign_and_uses_kelvin_offsets():
    truth = np.asarray(
        [
            [[10.0, 11.0]],
            [[12.0, 13.0]],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([[0.4, 0.6]], dtype=np.float64)

    warm_prediction = truth + 2.0
    cold_prediction = truth - 0.75

    assert _weighted_bias(warm_prediction, truth, weights) == pytest.approx(2.0)
    assert _weighted_bias(cold_prediction, truth, weights) == pytest.approx(-0.75)


def test_centered_acc_is_one_for_scaled_patterns_and_ignores_uniform_offset():
    climatology = np.zeros((2, 1, 2), dtype=np.float64)
    truth = np.asarray(
        [
            [[1.0, -1.0]],
            [[2.0, -2.0]],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([[0.5, 0.5]], dtype=np.float64)
    prediction = 2.0 * truth + 7.0

    assert _weighted_acc(prediction, truth, climatology, weights) == pytest.approx(
        1.0
    )
    assert _weighted_acc(-prediction, truth, climatology, weights) == pytest.approx(
        -1.0
    )


def test_centered_acc_is_undefined_for_a_climatology_forecast():
    climatology = np.zeros((2, 1, 2), dtype=np.float64)
    truth = np.asarray(
        [
            [[1.0, -1.0]],
            [[2.0, -2.0]],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([[0.5, 0.5]], dtype=np.float64)

    assert np.isnan(_weighted_acc(climatology, truth, climatology, weights))


def test_std_ratio_measures_uncentered_anomaly_rms_amplitude():
    climatology = np.zeros((2, 1, 2), dtype=np.float64)
    truth = np.asarray(
        [
            [[1.0, -1.0]],
            [[2.0, -2.0]],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([[0.5, 0.5]], dtype=np.float64)

    assert _weighted_std_ratio(
        2.5 * truth, truth, climatology, weights
    ) == pytest.approx(2.5)
