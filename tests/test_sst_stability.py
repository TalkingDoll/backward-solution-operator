"""CPU-only checks for strict SST neural-operator rollout diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from experiments.sst import exp_sst as sst


class _Operator:
    def eval(self):
        return self


def _dataset() -> dict:
    return {
        "meta": {
            "train_end": 1,
            "state_fields": np.zeros((8, 1, 1, 1), dtype=np.float32),
        }
    }


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        grid=np.zeros((1,), dtype=np.float32),
        kind="grid2d_multichannel",
        operator_pure=_Operator(),
    )


def test_strict_rollout_keeps_finite_cases_valid(monkeypatch) -> None:
    monkeypatch.setattr(
        sst,
        "apply_operator",
        lambda operator, values, grid, kind: values + 0.1,
    )
    trajectory, diagnostics = sst.batch_state_operator_rollout(
        _result(), _dataset(), np.asarray([0, 1]), 3,
        divergence_threshold=10.0, return_diagnostics=True,
    )
    assert np.all(np.isfinite(trajectory))
    np.testing.assert_array_equal(
        diagnostics["first_divergence"], np.asarray([-1, -1])
    )
    np.testing.assert_allclose(diagnostics["valid_fraction"], 1.0)


def test_strict_rollout_marks_nonfinite_cases_without_rescue(monkeypatch) -> None:
    monkeypatch.setattr(
        sst,
        "apply_operator",
        lambda operator, values, grid, kind: torch.full_like(values, float("nan")),
    )
    trajectory, diagnostics = sst.batch_state_operator_rollout(
        _result(), _dataset(), np.asarray([0]), 3,
        return_diagnostics=True,
    )
    assert np.isnan(trajectory[1:]).all()
    np.testing.assert_array_equal(
        diagnostics["first_nonfinite"], np.asarray([1])
    )
    np.testing.assert_array_equal(
        diagnostics["first_divergence"], np.asarray([1])
    )
    np.testing.assert_allclose(
        diagnostics["valid_fraction"], np.asarray([1.0, 0.0, 0.0, 0.0])
    )


def test_strict_rollout_marks_threshold_exceedance(monkeypatch) -> None:
    monkeypatch.setattr(
        sst,
        "apply_operator",
        lambda operator, values, grid, kind: torch.full_like(values, 2.0),
    )
    _, diagnostics = sst.batch_state_operator_rollout(
        _result(), _dataset(), np.asarray([0]), 2,
        divergence_threshold=1.0, return_diagnostics=True,
    )
    np.testing.assert_array_equal(
        diagnostics["first_threshold_exceedance"], np.asarray([1])
    )
    np.testing.assert_array_equal(
        diagnostics["first_divergence"], np.asarray([1])
    )


def test_nonfinite_only_records_large_finite_values_without_stopping(
        monkeypatch) -> None:
    outputs = iter((2.0, 3.0, float("inf")))

    def next_output(operator, values, grid, kind):
        return torch.full_like(values, next(outputs))

    monkeypatch.setattr(
        sst,
        "apply_operator",
        next_output,
    )
    trajectory, diagnostics = sst.batch_state_operator_rollout(
        _result(), _dataset(), np.asarray([0]), 3,
        divergence_threshold=1.0,
        invalidity_policy="nonfinite_only",
        return_diagnostics=True,
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
    np.testing.assert_allclose(trajectory[1], 2.0)
    np.testing.assert_allclose(trajectory[2], 3.0)
    assert np.isnan(trajectory[3]).all()
