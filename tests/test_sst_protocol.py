"""Unit tests for the SST evaluation protocols.

These tests intentionally exercise only pure protocol helpers.  They do not
open the SST archive, load checkpoints, or require a CUDA device.
"""

from __future__ import annotations

import numpy as np

from experiments.sst import exp_sst as sst


SHORT_MEDIUM_PROFILE = "short-medium-90d"
STABILITY_PROFILE = "rollout-stability-540d"


def test_expected_evaluation_profiles_are_registered():
    assert SHORT_MEDIUM_PROFILE in sst.EVALUATION_PROFILES
    assert STABILITY_PROFILE in sst.EVALUATION_PROFILES


def test_short_medium_profile_uses_every_day_from_one_through_ninety():
    leads = np.asarray(
        sst.build_lead_grid(SHORT_MEDIUM_PROFILE), dtype=np.int64
    )

    np.testing.assert_array_equal(leads, np.arange(1, 91, dtype=np.int64))


def test_stability_profile_uses_the_sparse_ninety_six_lead_grid():
    expected = np.unique(
        np.concatenate(
            [
                np.arange(1, 31, dtype=np.int64),
                np.arange(35, 181, 5, dtype=np.int64),
                np.arange(190, 541, 10, dtype=np.int64),
            ]
        )
    )
    leads = np.asarray(
        sst.build_lead_grid(STABILITY_PROFILE), dtype=np.int64
    )

    assert expected.size == 96
    np.testing.assert_array_equal(leads, expected)


def test_selected_origins_are_strictly_increasing_and_do_not_cross_eval_end():
    train_end = 1_000
    evaluation_end = 2_096
    n_steps = 90
    origins = np.asarray(
        sst.select_origins(
            train_end=train_end,
            evaluation_end=evaluation_end,
            n_steps=n_steps,
            n_origins=48,
        ),
        dtype=np.int64,
    )

    assert origins.shape == (48,)
    assert origins[0] == 0
    assert np.all(np.diff(origins) > 0)
    absolute_origins = train_end + origins
    assert np.all(absolute_origins >= train_end)
    assert np.all(absolute_origins + n_steps < evaluation_end)


def test_origin_offset_is_respected_without_changing_boundary_safety():
    train_end = 200
    evaluation_end = 500
    n_steps = 60
    origin_offset = 7
    origins = np.asarray(
        sst.select_origins(
            train_end=train_end,
            evaluation_end=evaluation_end,
            n_steps=n_steps,
            n_origins=12,
            origin_offset=origin_offset,
        ),
        dtype=np.int64,
    )

    assert origins.shape == (12,)
    assert origins[0] == origin_offset
    assert np.all(np.diff(origins) > 0)
    assert np.all(train_end + origins + n_steps < evaluation_end)
