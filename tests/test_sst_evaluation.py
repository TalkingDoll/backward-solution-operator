"""CPU-only checks for the locked SST evaluation protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.sst import sst_evaluation as evaluation


def test_run_root_is_the_shared_repository() -> None:
    assert evaluation.REPO_ROOT.is_dir()
    assert evaluation.RUN_ROOT == (
        evaluation.REPO_ROOT
        / "results"
        / "sst"
        / "runs"
    )


def test_first_failure_is_one_based_and_preserves_never_failed() -> None:
    mask = np.asarray([
        [True, True, True],
        [False, True, True],
        [False, False, True],
    ])
    np.testing.assert_array_equal(
        evaluation._first_failure(mask), np.asarray([2, 3, -1])
    )


def test_safe_output_rejects_path_outside_run_root(tmp_path: Path) -> None:
    try:
        evaluation._safe_run_output(tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("outside output path was accepted")


def test_bso_nonfinite_only_keeps_large_finite_values_valid() -> None:
    prediction = np.asarray([
        [[[2000.0]]],
        [[[3000.0]]],
        [[[float("inf")]]],
    ])
    dataset = {
        "meta": {
            "channel_mean": np.asarray([0.0]),
            "channel_std": np.asarray([1.0]),
            "state_fields": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }
    }
    diagnostics = evaluation._bso_validity(
        prediction,
        dataset,
        np.asarray([0]),
        threshold=1000.0,
        invalidity_policy="nonfinite_only",
    )
    np.testing.assert_array_equal(
        diagnostics["first_threshold_exceedance"], np.asarray([1])
    )
    np.testing.assert_array_equal(
        diagnostics["first_nonfinite"], np.asarray([3])
    )
    np.testing.assert_array_equal(
        diagnostics["first_divergence"], np.asarray([3])
    )
    np.testing.assert_allclose(
        diagnostics["valid_fraction"], np.asarray([1.0, 1.0, 1.0, 0.0])
    )

    summary = evaluation._stability_method_summary(
        {
            "leads": np.asarray([1, 5, 10]),
            "rmse_bso": np.asarray([0.1, 2.0e6, np.nan]),
            "valid_fraction_bso": np.asarray([1.0, 1.0, 0.0]),
            "rollout_bso_first_nonfinite": np.asarray([7, -1]),
        },
        "bso",
    )
    assert summary == {
        "first_nonfinite_day": 7,
        "last_complete_cohort_lead": 5,
        "last_complete_cohort_rmse": 2.0e6,
    }
