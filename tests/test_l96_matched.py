import numpy as np
import torch
from torch import nn

from experiments.lorenz96.exp_lorenz96 import STABILITY_RADIUS
from experiments.lorenz96.exp_lorenz96_data_length_matched import (
    DICT_SIZE, _fno_ensemble, build_matched_bso)

GRID = np.linspace(0.0, 1.0, 20, endpoint=False)


class _Scale(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = float(factor)

    def forward(self, fields, grid):
        return fields * self.factor


class _NonFinite(nn.Module):
    def forward(self, fields, grid):
        return torch.full_like(fields, float("inf"))


def test_matched_bso_is_MxM_and_stabilized():
    sampled = np.random.default_rng(0).normal(size=(200, 20))
    model = build_matched_bso(_Scale(0.5), GRID, "sensor1d", sampled)
    assert model is not None
    assert model["A"].shape == (DICT_SIZE, DICT_SIZE)      # 30 x 30
    radius = float(np.max(np.abs(np.linalg.eigvals(model["A"]))))
    assert radius <= STABILITY_RADIUS + 1e-6


def test_fno_ensemble_grows_and_stays_finite():
    seeds = np.random.default_rng(1).normal(size=(10, 20))
    ensemble = _fno_ensemble(_Scale(1.0), GRID, "sensor1d", seeds, target=100)
    assert ensemble.shape[1] == 20
    assert len(ensemble) > len(seeds)
    assert np.isfinite(ensemble).all()


def test_fno_ensemble_drops_nonfinite_states():
    seeds = np.random.default_rng(2).normal(size=(10, 20))
    ensemble = _fno_ensemble(_NonFinite(), GRID, "sensor1d", seeds, target=100)
    assert len(ensemble) == len(seeds)                     # only finite seeds
    assert np.isfinite(ensemble).all()
