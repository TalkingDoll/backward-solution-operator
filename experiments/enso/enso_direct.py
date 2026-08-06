"""Direct field-to-index forecaster for the ENSO experiments.

Every other ENSO method predicts the Indo-Pacific anomaly field and obtains
Niño-3.4 afterwards by area-averaging the decoded field. Those methods are
therefore optimised for basin-wide error while Niño-3.4 covers only ~4 % of the
ocean cells, so the index panels report a *derived* diagnostic rather than a
training objective.

This module supplies the missing comparator, following the construction used by
the older comparison suite (``comparison_experiments/run_enso.py``
``FieldToIndexFNO``): a neural operator emits one feature field per lead and a
**fixed** Niño-region area readout collapses each to an index value. The readout
is not learned, so the model remains a field-to-index map rather than an
unconstrained regressor, and it is the *same* weighting used by
``exp_enso._nino34`` -- pinned by a regression test so the two index definitions
cannot drift apart.

Two properties to keep in mind when reporting it:

* it emits all leads in one shot, so it is a lead-indexed supervised regressor,
  **not** an autonomous dynamical model that can be rolled out;
* it produces **no field forecast**, so it cannot appear in field-error panels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from edmd_bso_method import device, set_seed


class FieldToIndexFNO(nn.Module):
    """Operator features followed by a fixed Niño-region area readout."""

    def __init__(self, shape: tuple[int, int], horizon: int,
                 region_weights: np.ndarray, modes=(12, 24), width: int = 48,
                 layers: int = 4, in_channels: int = 1):
        super().__init__()
        from neuralop.layers.embeddings import GridEmbeddingND
        from neuralop.models import FNO

        self.shape = (int(shape[0]), int(shape[1]))
        self.horizon = int(horizon)
        self.in_channels = int(in_channels)
        self.core = FNO(
            n_modes=tuple(int(m) for m in modes), in_channels=self.in_channels,
            out_channels=self.horizon, hidden_channels=int(width),
            n_layers=int(layers),
            positional_embedding=GridEmbeddingND(
                in_channels=self.in_channels, dim=2,
                grid_boundaries=[[0.0, 1.0]] * 2))
        weights = np.asarray(region_weights, dtype=np.float64)
        self.register_buffer("region_weights",
                             torch.as_tensor(weights.reshape(1, 1, *self.shape),
                                             dtype=torch.float32))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        """``(B, C*H*W)`` anomaly fields -> ``(B, horizon)`` index values.

        ``C = 1`` is a single month; ``C = history`` supplies a month-history
        stack, matching the old suite's delay-embedded direct model.
        """
        x = fields.reshape(-1, self.in_channels, *self.shape)
        return torch.sum(self.core(x) * self.region_weights, dim=(-2, -1))


def region_readout_weights(data) -> np.ndarray:
    """Area weights restricted to Niño-3.4 and normalised to sum one.

    Identical to the weighting inside ``enso_data.area_mean``, so the learned
    model and the diagnostic used for every other method share one definition.
    """
    w = np.asarray(data.weights, dtype=np.float64) * data.nino_masks["nino34"]
    total = w.sum()
    if total <= 0:
        raise ValueError("Niño region has no valid ocean cells")
    return w / total


def _metadata(shape, horizon, modes, width, layers, epochs, train_end,
              in_channels) -> dict:
    return {"model": "field_to_index_fno", "shape": tuple(int(s) for s in shape),
            "horizon": int(horizon), "modes": tuple(int(m) for m in modes),
            "width": int(width), "layers": int(layers), "epochs": int(epochs),
            "train_end": int(train_end), "in_channels": int(in_channels)}


def train_direct(data, origins: np.ndarray, index: np.ndarray, horizon: int,
                 *, inputs: np.ndarray | None = None, in_channels: int = 1,
                 modes=(12, 24), width: int = 48, layers: int = 4,
                 epochs: int = 400, batch_size: int = 16, lr: float = 6e-4,
                 seed: int = 2026, checkpoint: Path | None = None):
    """Fit the direct model on training-period origins only.

    ``origins`` must lie strictly before ``data.train_end - horizon`` so no
    target month reaches the test period. The target is the Niño-3.4 index
    divided by its training-period standard deviation.
    """
    origins = np.asarray(origins, dtype=int)
    if origins.size and int(origins.max()) + horizon >= data.train_end:
        raise ValueError("direct-model targets must stay before train_end")
    scale = float(np.std(index[:data.train_end]))
    meta = _metadata(data.shape, horizon, modes, width, layers, epochs,
                     data.train_end, in_channels)

    set_seed(seed)
    model = FieldToIndexFNO(data.shape, horizon, region_readout_weights(data),
                            modes=modes, width=width, layers=layers,
                            in_channels=in_channels).to(device)
    if checkpoint is not None and Path(checkpoint).exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload.get("meta") == meta:
            # ``state_dict`` carries a ``_metadata`` attribute that survives a
            # save/load round trip as a stray key; drop it before loading.
            state = {k: v for k, v in payload["state"].items()
                     if k != "_metadata"}
            model.load_state_dict(state)
            print(f"[enso-direct] loaded checkpoint {checkpoint}")
            return model.eval(), scale

    features = data.anomalies[origins] if inputs is None else inputs
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    y = torch.as_tensor(
        np.stack([index[origins + lead] for lead in range(1, horizon + 1)],
                 axis=1) / max(scale, 1e-8), dtype=torch.float32)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(lr))
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(int(epochs)):
        order = rng.permutation(len(x))
        for start in range(0, len(order), int(batch_size)):
            idx = torch.as_tensor(order[start:start + int(batch_size)],
                                  dtype=torch.long)
            loss = torch.mean((model(x[idx].to(device)) - y[idx].to(device)) ** 2)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    model.eval()
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": dict(model.state_dict()), "meta": meta},
                   checkpoint)
        print(f"[enso-direct] saved checkpoint {checkpoint}")
    return model, scale


def predict_direct(model, fields: np.ndarray, scale: float,
                   batch_size: int = 64) -> np.ndarray:
    """``(n_origins, horizon)`` index predictions in physical units."""
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(fields), int(batch_size)):
            batch = torch.as_tensor(fields[start:start + int(batch_size)],
                                    dtype=torch.float32, device=device)
            out.append(model(batch).cpu().numpy().astype(np.float64))
    return np.concatenate(out) * float(scale)
