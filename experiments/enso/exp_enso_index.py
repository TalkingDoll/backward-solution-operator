"""Direct Niño-3.4 forecasting through the BSO framework.

Every other BSO in the ENSO suite (``exp_enso``, ``exp_enso_delay``) advances
the Indo-Pacific **field** and obtains Niño-3.4 afterwards by area-averaging the
decoded field. ``enso_direct.FieldToIndexFNO`` forecasts the index directly but
bypasses the BSO entirely: it has no dictionary and forms no Galerkin matrix.

This module supplies the missing object -- the direct-forecast construction
carried *inside* the BSO framework:

    y(t+tau) ~= sum_j c_j(tau) psi_j(z(t))

for a reduced state ``z(t)`` carrying 12 months of memory. The dictionary
``Psi`` is trained by the two-phase BSO recipe (a frozen neural operator, then a
Galerkin/EDMD-DL invariance objective) and the per-lead coefficients ``c(tau)``
are fitted by ridge least squares. That is structurally **exactly the KAF
construction** of Wang, Slawinska & Giannakis (2020),
``c_l(tau) = (1/N) sum_n phi_ln y_{n+q}``, with the BSO's learned dictionary in
place of their kernel eigenbasis.

The Galerkin matrix ``K`` from phase B is reported two ways: the per-lead fits
above do not use it (it only shapes the dictionary), while ``bso_prop`` forecasts
every lead by propagating it, exactly as the published EEG rollout does
(``eeg_operator_bso.rollout_operator_bso``). The second uses one operator
and a single readout where the first uses ``horizon`` independent regressions,
so the two together separate skill from parameter count.

``bso_prop`` is the rung the published figure draws, because it is the only one
with an equivalent in the EEG figure: the per-lead form is the KAF construction
and a different estimator, and showing it under the same name in two papers'
figures would conflate the two. ``PRESENTATION_RUNG`` names the choice.

Everything reverts to zero anomaly
----------------------------------
Niño-3.4 is an anomaly against a 1978-2007 base period, but the 1870-2007 fitting
record averages -0.387 °C against it. A state built by removing the record mean
therefore decays to -0.39 °C rather than to the climatology the index is defined
against, and the propagated rung carried a cold bias reaching -0.47 °C by lead 15
-- enough on its own to lose an RMSE comparison it was winning on error variance
(0.89 debiased against the LIM's 0.91). Both reduced states are consequently
*scaled but not centred*, and both readouts are offset-free; ``enso_lim.WangLIM``
makes the matching choice via Eq. 9. Only then does the error panel compare error
rather than offset.

Equal information, one varying factor
-------------------------------------
Every forecast rung below is fitted on the **same predictor state**, over the
same origins, with its ridge selected by the same protocol. That is the whole
point of the module: an earlier version gave the BSO a 12-month delay state
while its baselines saw a single month, so the reported margin measured memory,
not the dictionary. The ladder separates the factors instead:

===========  ======================================  ==========================
rung         predictors                              isolates
===========  ======================================  ==========================
``lim_prop`` propagated ``G(tau)`` on 20 PCs         the classical benchmark
``lim_wang`` Wang et al. ``tau0 = 2``                the published variant
``lim_inst`` ``[1, c(t)]``, 20 EOFs, one month       its per-lead counterpart
``lim_raw``  ``[1, X(t)]``, 12 x 20 delay            what memory alone buys
``linear_*`` ``[1, z(t)]``, the reduced state        what the compression buys
``bso_*``    ``[1, z, h_theta(z)]``                  **the learned dictionary**
``bso_prop`` ``Psi K^n``, state block, fixed map    the EEG rollout, verbatim
``mlp_*``    a plain MLP on ``z``                    dictionary vs. generic net
===========  ======================================  ==========================

``bso_* - linear_*`` is the claim, at identical input.

One asymmetry is deliberate. The three LIM rows are **not** ridge-tuned: a LIM
is ``G(tau) = C(tau) C(0)^-1`` on a truncated PC basis, the truncation is its
regularisation, and a cross-validated ridge on top would make it a
better-conditioned estimator that is no longer the published method. The
``linear_*`` and ``lim_raw`` rows *are* tuned, because they are controls rather
than named baselines and their entire job is to be fitted exactly as the BSO is.
Reporting both kinds is what keeps "fitted as published" and "fitted identically"
from having to be the same row.

Two reduced states are carried so the choice of state is visible rather than
assumed: ``eof`` (a training-fit PCA of the 12-month delay embedding of 20 EOF
coefficients) and ``index`` (the 12-month Niño-3.4 history alone).
``direct_fno`` is kept as a deliberately *unmatched* reference -- it sees the
whole field -- and is labelled as such wherever it appears.

Why the EEG BSO rather than ``run_bso_pipeline``
------------------------------------------------
The repository holds two different notions of "dictionary" and only one gives a
non-trivial direct forecast:

* ``run_bso_pipeline`` (``edmd_bso_method.build_basis``) builds columns from
  ``BasisNet(coords)`` -- functions of the *spatial coordinate*. A state enters
  only through ``c_j = <psi_j, u>_w``, which is **linear in the state**. A direct
  per-lead regression on it is therefore *identically* the ``linear_*`` rung:
  provably degenerate.
* ``eeg_edmddl.PaperEDMDDLDictionary`` builds ``Psi(z) = [1, z, h_theta(z)]``,
  an MLP of the *state*. This is the nonlinear hypothesis space KAF assumes, so
  it is the one used here.

Usage:
    python -m experiments.enso.exp_enso_index [--quick]
    python -m experiments.enso.exp_enso_index --ladder   # the table, no figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.dont_write_bytecode = True

from edmd_bso_method import device, set_seed
from experiments.eeg.eeg_edmddl import PaperEDMDDLDictionary
# Both helpers are generic in the state dimension -- they validate against their
# arguments, not the EEG constants -- so they are shared verbatim rather than
# reimplemented, and the two experiments cannot drift apart.
from experiments.eeg.eeg_operator_bso import (TemporalFNOStateOperator,
                                               _dictionary_values,
                                               _galerkin_transition)
from experiments.enso.enso_data import (fit_eofs, load_ersstv5, pca_basis,
                                         pca_inverse, pca_state)
from experiments.enso.enso_direct import predict_direct, train_direct
from experiments.enso.enso_lim import (RIDGE_GRID, fit_direct_cv,
                                        fit_lim_direct, fit_lim_wang,
                                        predict_lim_direct)
from experiments.enso.exp_enso import (DATA_PATH, HORIZON, LIM_MODES, _corr,
                                        _nino34, damped_persistence_gamma,
                                        encode, fit_lim, lim_forecast)
from experiments.enso.exp_enso_delay import HISTORY, delay_state
from figure_style import (DOUBLE_COLUMN_WIDTH, GRID_COLOR, LINEWIDTH,
                          MARKER_EDGE_WIDTH, MARKER_SIZE, add_figure_title,
                          add_legend, make_figure, publication_style,
                          save_publication_pair, style_axis)
from plotting_utils import print_figure_caption
from project_paths import FIGURES_DIR, RESULTS_DIR

# The reduced state every method is handed. ``REDUCED_DIM`` plays the role of
# ``OperatorBSOConfig.reduced_state_dimension`` (8 for EEG); 12 keeps one
# dimension per month of memory.
COMMON_MODES = LIM_MODES          # EOF coefficients entering the delay embedding
REDUCED_DIM = 12

# 1 constant + REDUCED_DIM fixed state coordinates + 11 learned functions,
# mirroring the EEG split of 1 + 8 + 11 (``eeg_operator_bso.OperatorBSOConfig``).
DICTIONARY_SIZE = 1 + REDUCED_DIM + 11
TRANSITION_RIDGE = 1e-6
# ``eeg_operator_bso.OperatorBSOConfig.dictionary_hidden_dims``, so the
# dictionary construction is identical to the EEG experiment's. The matched-input
# MLP reuses the same widths, so "learned dictionary vs. generic net" cannot be a
# capacity comparison in disguise.
HIDDEN_DIMS = (16, 16)

# Verification uncertainty. Monthly origins overlap heavily -- the record holds
# roughly one independent ENSO event per two years -- so a plain bootstrap over
# origins would be far too confident. Blocks of BOOTSTRAP_BLOCK months keep whole
# events together.
BOOTSTRAP_BLOCK = 24
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 2026


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def index_delay_states(index: np.ndarray, train_end: int,
                       history: int = HISTORY):
    """Scaled ``history``-month delay embedding of a scalar index.

    Returns ``(states, mean, scale)`` where row ``t`` of ``states`` holds months
    ``t-history+1 … t`` oldest-first (``exp_enso_delay.delay_state`` with a
    single mode), so its **newest** entry is the scaled index at ``t``. Rows
    without full context are NaN. ``scale`` uses training months only, so no test
    statistic leaks into the state definition.

    The index is divided by its training standard deviation but **not** centred,
    and ``mean`` is returned for reporting rather than applied. Niño-3.4 is
    already an anomaly, against a 1978-2007 base period that is much warmer than
    the 1870-2007 fitting record; subtracting the record mean would re-reference
    it to a period that is not its base period, and every forecast built on the
    state would then decay to -0.39 °C instead of to the climatology the index is
    defined against. Measured, that cost the propagated rung a cold bias reaching
    -0.47 °C by lead 15 -- enough to reverse an RMSE ranking on its own. Leaving
    the mean in makes the state a scaled anomaly, so an operator that drives it to
    the origin predicts zero anomaly, matching ``enso_lim.WangLIM``'s convention.

    ``scale`` stays the standard deviation rather than the RMS about zero: it is
    only a scale, and the asymptote is set by the centring, not by it.
    """
    index = np.asarray(index, dtype=np.float64)
    train = index[:int(train_end)]
    mean, scale = float(train.mean()), float(train.std())
    scale = scale if scale > 1e-12 else 1.0
    return delay_state(index[:, None] / scale, int(history)), mean, scale


def reduced_states(data, index: np.ndarray, patterns: np.ndarray) -> dict:
    """``name -> (state, readout)`` for the states the ladder is run on.

    ``eof``
        a training-fit PCA of the 12-month delay embedding of the leading
        ``COMMON_MODES`` EOF coefficients -- memory *and* the basin.
    ``index``
        the 12-month Niño-3.4 history alone -- memory without the basin.

    Returning both is what makes the choice of state visible instead of assumed:
    each gets its own dictionary, its own linear control and its own MLP, so a
    margin can be attributed to the dictionary rather than to the state it
    happened to be given.

    ``readout`` maps a state back to Niño-3.4 in °C with no fitted parameters,
    by inverting the state map and applying the standard area average. It is the
    ENSO counterpart of ``rollout_operator_bso``'s "reconstruct only through
    fixed inverse PCA", and it is what lets the propagated rung be the same
    construction as the EEG rollout rather than a regression wearing its name.

    Both states are **uncentred**: the score of a zero anomaly is zero, so a
    decaying operator predicts zero anomaly rather than the fitting record's mean.
    See :func:`index_delay_states` for why that is a correctness requirement here
    rather than a preference, and ``enso_data.pca_state``'s ``center`` argument
    for the same choice on the EOF side. The two readouts are correspondingly
    offset-free: no ``index_mean`` is added back, and ``pca_inverse`` is given a
    zero mean.
    """
    coeff = encode(patterns[:COMMON_MODES], data.weights, data.anomalies)
    delay = delay_state(coeff, HISTORY)
    _, components = pca_basis(delay, data.train_end, REDUCED_DIM)
    zero_mean = np.zeros(components.shape[1])

    def eof_readout(scores):
        newest = pca_inverse(scores, zero_mean, components)[:, -COMMON_MODES:]
        return _nino34(data, newest @ patterns[:COMMON_MODES])

    _, _, index_scale = index_delay_states(index, data.train_end)

    def index_readout(scores):
        return np.asarray(scores)[:, -1] * index_scale

    return {
        "eof": (pca_state(delay, data.train_end, REDUCED_DIM, center=False),
                eof_readout),
        "index": (index_delay_states(index, data.train_end)[0], index_readout),
    }


def _pair_ends(first: int, last: int) -> np.ndarray:
    """Ends ``t`` of one-month pairs ``z(t) -> z(t+1)`` with ``t+1 < last``."""
    return np.arange(int(first), int(last) - 1)


# ---------------------------------------------------------------------------
# Phase A: neural operator on the delay state
# ---------------------------------------------------------------------------

def _operator_metadata(state, state_dim, epochs, train_end, width, modes,
                       layers, batch_size, lr, weight_decay, seed,
                       n_pairs) -> dict:
    """Everything that changes the fitted weights, so a stale reload cannot.

    The earlier version recorded only architecture and epochs, which meant an
    edited learning rate, batch size or seed silently reused old weights -- and,
    worse, that swapping the reduced state kept the same tensor shapes and so
    matched the old checkpoint exactly.

    ``state_centering`` is the field that earned that warning. Moving both states
    to the uncentred (anomaly) convention shifts every input by a constant while
    changing no name, no dimension and no tensor shape -- both states are 12-wide
    -- so without it the operators trained against the *centred* states would have
    reloaded silently and ``K`` would have been recomputed from a mismatched pair.
    """
    return {"model": "reduced_state_temporal_fno", "state": str(state),
            "state_dim": int(state_dim), "epochs": int(epochs),
            "train_end": int(train_end), "width": int(width),
            "modes": int(modes), "layers": int(layers),
            "batch_size": int(batch_size), "lr": float(lr),
            "weight_decay": float(weight_decay), "seed": int(seed),
            "n_pairs": int(n_pairs), "history": int(HISTORY),
            "common_modes": int(COMMON_MODES),
            "state_centering": "anomaly"}


def train_phase_a(current: np.ndarray, target: np.ndarray, *, state: str,
                  train_end: int, width: int = 24, modes: int = 12,
                  layers: int = 2, epochs: int = 200, batch_size: int = 64,
                  lr: float = 5e-4, weight_decay: float = 2e-5,
                  seed: int = 2026, checkpoint: Path | None = None):
    """Fit and freeze the one-month flow map ``z(t) -> z(t+1)``.

    The state's coordinates are a fixed linear functional of the record, so this
    is one-step supervision on the reduced state itself. Unlike
    ``run_bso_pipeline``'s phase A there is no two-step (``sg_weight``) term --
    the EEG operator is trained one-step only, and that convention is kept.
    """
    state_dim = int(current.shape[1])
    meta = _operator_metadata(state, state_dim, epochs, train_end, width, modes,
                              layers, batch_size, lr, weight_decay, seed,
                              len(current))
    set_seed(seed)
    operator = TemporalFNOStateOperator(
        1, state_dim, hidden_channels=int(width), n_modes=int(modes),
        n_layers=int(layers)).to(device)

    if checkpoint is not None and Path(checkpoint).exists():
        payload = torch.load(checkpoint, map_location=device,
                             weights_only=False)
        if payload.get("meta") == meta:
            # ``state_dict`` carries a ``_metadata`` attribute that survives a
            # save/load round trip as a stray key; drop it before loading.
            operator.load_state_dict({k: v for k, v in payload["state"].items()
                                      if k != "_metadata"})
            print(f"[enso-index] loaded phase-A checkpoint {checkpoint}")
            return operator.eval(), meta

    x = torch.as_tensor(current[:, None, :], dtype=torch.float32)
    y = torch.as_tensor(target[:, None, :], dtype=torch.float32)
    optimiser = torch.optim.Adam(operator.parameters(), lr=float(lr),
                                 weight_decay=float(weight_decay))
    rng = np.random.default_rng(seed)
    operator.train()
    for _ in range(int(epochs)):
        order = rng.permutation(len(x))
        for start in range(0, len(order), int(batch_size)):
            idx = torch.as_tensor(order[start:start + int(batch_size)],
                                  dtype=torch.long)
            loss = torch.mean((operator(x[idx].to(device))
                               - y[idx].to(device)) ** 2)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    operator.eval()
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": dict(operator.state_dict()), "meta": meta},
                   checkpoint)
        print(f"[enso-index] saved phase-A checkpoint {checkpoint}")
    return operator, meta


def operator_image(operator, states: np.ndarray,
                   batch_size: int = 256) -> np.ndarray:
    """``(N, state_dim)`` one-month images of the frozen phase-A operator."""
    out = []
    operator.eval()
    with torch.no_grad():
        for start in range(0, len(states), int(batch_size)):
            batch = torch.as_tensor(states[start:start + int(batch_size),
                                           None, :],
                                    dtype=torch.float32, device=device)
            out.append(operator(batch)[:, 0].cpu().numpy().astype(np.float64))
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Phase B: Galerkin/EDMD-DL dictionary on the frozen operator
# ---------------------------------------------------------------------------

def train_phase_b(current_state: np.ndarray, operator_target_state: np.ndarray,
                  *, operator_meta: dict, dictionary_size: int | None = None,
                  hidden_dims=HIDDEN_DIMS, epochs: int = 200,
                  batch_size: int = 64, lr: float = 1e-3, seed: int = 2026,
                  print_every: int = 0, checkpoint: Path | None = None):
    """Train ``Psi(z) = [1, z, h_theta(z)]`` against the frozen operator image.

    Alternates the closed-form Galerkin solve for ``K`` with gradient steps on
    the invariance residual ``Psi(z) K - Psi(F(z))``, exactly the alternation in
    ``eeg_operator_bso._train_phase_b``; the loop is written out here rather
    than imported because that function is bound to ``OperatorBSOConfig``'s EEG
    defaults.

    ``operator_meta`` is phase A's metadata, stored inside this checkpoint's own.
    Without it a phase-A change invalidates ``phase_a.pt`` but leaves this
    checkpoint matching, so a dictionary fitted against the *previous* frozen
    operator reloads silently while ``K`` is recomputed from the new one -- a
    wrong answer that looks entirely plausible. The reduced states here have
    equal width, so the tensor shapes alone would not catch the swap.
    """
    state_dim = int(current_state.shape[1])
    dictionary_size = (1 + state_dim + 11 if dictionary_size is None
                       else int(dictionary_size))
    set_seed(seed)
    dictionary = PaperEDMDDLDictionary(
        state_dimension=state_dim, dictionary_size=dictionary_size,
        hidden_dims=tuple(int(h) for h in hidden_dims)).to(device)
    weights = np.full(len(current_state), 1.0 / len(current_state))
    meta = {"model": "reduced_state_edmddl_dictionary",
            "state_dimension": state_dim,
            "dictionary_size": dictionary_size,
            "hidden_dims": tuple(int(h) for h in hidden_dims),
            "epochs": int(epochs), "batch_size": int(batch_size),
            "lr": float(lr), "seed": int(seed),
            "ridge": float(TRANSITION_RIDGE),
            "n_pairs": int(len(current_state)),
            "operator": dict(operator_meta)}

    def _finalise():
        dictionary.eval()
        return _galerkin_transition(
            _dictionary_values(dictionary, current_state, device),
            _dictionary_values(dictionary, operator_target_state, device),
            sample_weights=weights, ridge=TRANSITION_RIDGE)

    if checkpoint is not None and Path(checkpoint).exists():
        payload = torch.load(checkpoint, map_location=device,
                             weights_only=False)
        if payload.get("meta") == meta:
            dictionary.load_state_dict(
                {k: v for k, v in payload["state"].items() if k != "_metadata"})
            print(f"[enso-index] loaded phase-B checkpoint {checkpoint}")
            return (dictionary, *_finalise())

    optimiser = torch.optim.Adam(dictionary.parameters(), lr=float(lr))
    current = torch.as_tensor(current_state, dtype=torch.float32)
    target = torch.as_tensor(operator_target_state, dtype=torch.float32)
    rng = np.random.default_rng(seed)

    transition = np.eye(int(dictionary_size))
    diagnostics: dict = {}
    for epoch in range(int(epochs)):
        transition, diagnostics = _galerkin_transition(
            _dictionary_values(dictionary, current_state, device),
            _dictionary_values(dictionary, operator_target_state, device),
            sample_weights=weights, ridge=TRANSITION_RIDGE)
        k = torch.as_tensor(transition, dtype=torch.float32, device=device)
        dictionary.train()
        order = rng.permutation(len(current_state))
        for start in range(0, len(order), int(batch_size)):
            idx = torch.as_tensor(order[start:start + int(batch_size)],
                                  dtype=torch.long)
            loss = torch.mean((dictionary(current[idx].to(device)) @ k
                               - dictionary(target[idx].to(device))) ** 2)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        if print_every and (epoch + 1) % int(print_every) == 0:
            print(f"[enso-index] phase B epoch {epoch + 1}: invariance="
                  f"{diagnostics['weighted_invariance_mse']:.4e}")

    transition, diagnostics = _finalise()
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": dict(dictionary.state_dict()), "meta": meta},
                   checkpoint)
        print(f"[enso-index] saved phase-B checkpoint {checkpoint}")
    return dictionary, transition, diagnostics


def dictionary_features(dictionary, states: np.ndarray) -> np.ndarray:
    """``Psi(z)`` for every row, NaN-safe: rows without full context stay NaN.

    ``delay_state`` marks incomplete rows NaN so missing context cannot be used
    by accident; the network is only evaluated on the complete ones.
    """
    states = np.asarray(states, dtype=np.float64)
    valid = np.all(np.isfinite(states), axis=1)
    out = np.full((len(states), dictionary.dictionary_size), np.nan)
    out[valid] = _dictionary_values(dictionary, states[valid], device)
    return out


def propagated_forecast(psi: np.ndarray, transition: np.ndarray,
                        state_dim: int, readout, origins: np.ndarray,
                        horizon: int) -> np.ndarray:
    """``Psi K^n``, state block out, fixed readout -- the EEG rollout, verbatim.

    This is ``eeg_operator_bso.rollout_operator_bso`` transposed to ENSO: push
    the dictionary forward with the Galerkin operator, slice the fixed state
    coordinates ``Psi[:, 1:1+state_dim]`` back out, and reconstruct through the
    *fixed* inverse state map. Nothing is fitted per lead, and nothing is fitted
    to the index at all -- the whole forecast is ``K`` plus a map that was
    determined before any forecasting began.

    That makes it the only rung here with an equivalent in the published EEG
    figure, and the reason to prefer it over a fitted linear readout of ``Psi``:
    the fitted version would score better while quietly reintroducing the
    per-lead tuning the rung exists to do without.

    ``psi`` must include its constant column, since ``K`` acts on the full
    dictionary under the ``Psi(z_next) ~= Psi(z) @ K`` convention.
    """
    features = np.asarray(psi, dtype=np.float64)[np.asarray(origins, dtype=int)]
    transition = np.asarray(transition, dtype=np.float64)
    stop = 1 + int(state_dim)
    out = []
    for _ in range(int(horizon)):
        features = features @ transition
        out.append(readout(features[:, 1:stop]))
    return np.stack(out, axis=1)


class DirectMLP(nn.Module):
    """Plain MLP from the reduced state to every lead at once.

    Deliberately the same widths and activation as
    ``PaperEDMDDLDictionary.learned_dictionary``, so "learned dictionary vs.
    generic network" is a comparison of *construction* -- Galerkin invariance
    against direct supervision -- and not of capacity.
    """

    def __init__(self, input_dim: int, horizon: int, hidden_dims=HIDDEN_DIMS):
        super().__init__()
        widths = (int(input_dim), *(int(h) for h in hidden_dims), int(horizon))
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(widths) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def train_direct_mlp(states: np.ndarray, index: np.ndarray,
                     origins: np.ndarray, horizon: int, *, state: str,
                     hidden_dims=HIDDEN_DIMS, epochs: int = 400,
                     batch_size: int = 64, lr: float = 1e-3,
                     weight_decay: float = 1e-4, seed: int = 2026,
                     checkpoint: Path | None = None):
    """Fit :class:`DirectMLP` on the same origins as every other rung."""
    origins = np.asarray(origins, dtype=int)
    scale = float(np.std(np.asarray(index)[origins]))
    scale = scale if scale > 1e-12 else 1.0
    # ``state_centering`` for the same reason as in ``_operator_metadata``: the
    # centred and uncentred states share every other field, including width.
    meta = {"model": "reduced_state_direct_mlp", "state": str(state),
            "state_dim": int(states.shape[1]), "horizon": int(horizon),
            "hidden_dims": tuple(int(h) for h in hidden_dims),
            "epochs": int(epochs), "batch_size": int(batch_size),
            "lr": float(lr), "weight_decay": float(weight_decay),
            "seed": int(seed), "n_origins": int(len(origins)),
            "first_origin": int(origins[0]),
            "state_centering": "anomaly"}

    set_seed(seed)
    model = DirectMLP(states.shape[1], horizon, hidden_dims).to(device)
    if checkpoint is not None and Path(checkpoint).exists():
        payload = torch.load(checkpoint, map_location=device,
                             weights_only=False)
        if payload.get("meta") == meta:
            model.load_state_dict({k: v for k, v in payload["state"].items()
                                   if k != "_metadata"})
            print(f"[enso-index] loaded MLP checkpoint {checkpoint}")
            return model.eval(), scale

    x = torch.as_tensor(states[origins], dtype=torch.float32)
    y = torch.as_tensor(
        np.stack([np.asarray(index)[origins + lead]
                  for lead in range(1, int(horizon) + 1)], axis=1) / scale,
        dtype=torch.float32)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(lr),
                                 weight_decay=float(weight_decay))
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(int(epochs)):
        order = rng.permutation(len(x))
        for start in range(0, len(order), int(batch_size)):
            idx = torch.as_tensor(order[start:start + int(batch_size)],
                                  dtype=torch.long)
            loss = torch.mean((model(x[idx].to(device))
                               - y[idx].to(device)) ** 2)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    model.eval()
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": dict(model.state_dict()), "meta": meta},
                   checkpoint)
        print(f"[enso-index] saved MLP checkpoint {checkpoint}")
    return model, scale


def predict_mlp(model, states: np.ndarray, origins: np.ndarray,
                scale: float) -> np.ndarray:
    """``(n_origins, horizon)`` index predictions in physical units."""
    model.eval()
    with torch.no_grad():
        batch = torch.as_tensor(states[np.asarray(origins, dtype=int)],
                                dtype=torch.float32, device=device)
        return model(batch).cpu().numpy().astype(np.float64) * float(scale)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def _direct_metrics(pred_n34, truth_n34, horizon, mask=None):
    """Per-lead RMSE and correlation, optionally over a ragged origin set.

    ``mask[i, l]`` marks origin ``i`` as having an observed target at lead
    ``l+1``. At the 12-month horizon every origin has every lead and ``mask`` is
    ``None``; the long-lead variant runs out of verifying months at the end of
    the record, so each lead is scored on whatever origins it has.
    """
    rmse, acc = np.empty(horizon), np.empty(horizon)
    for lead in range(horizon):
        keep = (slice(None) if mask is None
                else np.flatnonzero(mask[:, lead]))
        p, t = pred_n34[keep, lead], truth_n34[keep, lead]
        rmse[lead] = np.sqrt(np.mean((p - t) ** 2)) if len(p) else np.nan
        acc[lead] = _corr(p, t) if len(p) > 2 else np.nan
    return dict(nino34_rmse=rmse, nino34_acc=acc)


def _block_resamples(n_origins: int, block: int, n_samples: int, seed: int):
    """Indices for a moving-block bootstrap over ``n_origins`` launches.

    Origins are consecutive months, so they are nowhere near independent: the
    verification window holds ~17 years and ENSO decorrelates on a ~2-year
    scale, which is roughly one independent event per ``block`` origins. Drawing
    single origins with replacement would treat neighbouring months as fresh
    information and produce intervals several times too narrow.
    """
    rng = np.random.default_rng(int(seed))
    block = max(1, min(int(block), int(n_origins)))
    n_blocks = int(np.ceil(n_origins / block))
    starts = rng.integers(0, n_origins - block + 1,
                          size=(int(n_samples), n_blocks))
    offsets = np.arange(block)
    draws = (starts[:, :, None] + offsets[None, None, :])
    return draws.reshape(int(n_samples), -1)[:, :n_origins]


def acc_difference_interval(pred_a, pred_b, truth, horizon, mask=None,
                            block: int = BOOTSTRAP_BLOCK,
                            n_samples: int = BOOTSTRAP_SAMPLES,
                            seed: int = BOOTSTRAP_SEED, alpha: float = 0.05):
    """Percentile CI on the *paired* per-lead ``ACC(a) - ACC(b)``.

    Both forecasts are resampled on the same drawn origins, so the shared
    year-to-year variation of the verification window cancels and the interval
    speaks to the difference rather than to either curve's absolute level.
    """
    draws = _block_resamples(len(truth), block, n_samples, seed)
    lower = np.full(int(horizon), np.nan)
    upper = np.full(int(horizon), np.nan)
    for lead in range(int(horizon)):
        column = (draws if mask is None
                  else np.where(mask[draws, lead], draws, -1))
        deltas = np.empty(len(draws))
        for i, idx in enumerate(column):
            idx = idx[idx >= 0]
            if len(idx) < 3:
                deltas[i] = np.nan
                continue
            t = truth[idx, lead]
            deltas[i] = (_corr(pred_a[idx, lead], t)
                         - _corr(pred_b[idx, lead], t))
        if np.isfinite(deltas).sum() < 2:
            continue
        lower[lead], upper[lead] = np.nanpercentile(
            deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lower, upper


def run(quick: bool, horizon: int | None = None, ragged: bool = False) -> dict:
    """Fit every ladder rung on shared origins and score them identically.

    ``horizon`` overrides the 12-month default. ``ragged`` scores each lead on
    every origin that has an observed target rather than on one common origin
    set -- necessary past ~24 months, where a fixed origin set would collapse to
    a handful of launches. It is off at the default horizon so those numbers are
    unaffected.

    The classical LIM appears here as ``lim_inst``, the per-lead lagged
    regression on instantaneous EOF coefficients, rather than as the propagated
    lag-1 propagator used by ``exp_enso``. Two reasons: it is the stronger of the
    two on this record, so the baseline is not weakened for convenience; and it
    is fitted on ``fit_origins`` like everything else, whereas the propagated
    variant is estimated from every training month and so would not share the
    sample.
    """
    data = load_ersstv5(DATA_PATH)
    if horizon is None:
        horizon = 4 if quick else HORIZON
    horizon = int(horizon)
    ckpt_dir = None if quick else RESULTS_DIR / "enso_index"

    index = _nino34(data, data.anomalies)
    eof = fit_eofs(data, data.train_end, COMMON_MODES)
    coeff = encode(eof.patterns[:COMMON_MODES], data.weights, data.anomalies)
    states = reduced_states(data, index, eof.patterns)

    # Fixed origin set: every launch verifies every lead. Ragged: launch from
    # every post-training month and let each lead use the origins that still
    # have an observed target inside the test window. The reserve past
    # ``test_end`` stays untouched either way.
    if ragged:
        origins = np.arange(data.train_end, data.test_end - 1)
        targets = origins[:, None] + np.arange(1, horizon + 1)[None, :]
        mask = targets < data.test_end
        truth_n34 = index[np.clip(targets, 0, data.test_end - 1)]
    else:
        origins = np.arange(data.train_end, data.test_end - horizon)
        mask = None
        truth_n34 = np.stack([index[origins + lead]
                              for lead in range(1, horizon + 1)], axis=1)
    # One fitting origin set for every rung, so no method is fitted on more
    # months than another.
    fit_origins = np.arange(HISTORY - 1, data.train_end - horizon)

    metrics, ridges, predictions = {}, {}, {}

    def _score(name, prediction):
        predictions[name] = prediction
        metrics[name] = _direct_metrics(prediction, truth_n34, horizon, mask)

    def _regress(name, features):
        """One rung: blocked-CV ridge on ``features``, scored like every other.

        ``fit_direct_cv`` prepends an intercept, so a dictionary is passed
        without its constant column and that intercept *becomes* the constant.
        Every rung goes through this one call, so the only thing that differs
        between them is the feature matrix.
        """
        weights, ridges[name] = fit_direct_cv(features, index, fit_origins,
                                              horizon, grid=RIDGE_GRID)
        _score(name, predict_lim_direct(weights, features[origins]))

    # --- rungs on the reduced states: dictionary vs. its own controls -----
    diagnostics = {}
    for state_name, (state, readout) in states.items():
        ends = _pair_ends(HISTORY - 1, data.train_end)
        current, target = state[ends], state[ends + 1]

        # Phase A on training pairs only: the latest target month is
        # train_end - 1.
        operator, operator_meta = train_phase_a(
            current, target, state=state_name, train_end=data.train_end,
            width=8 if quick else 24, modes=6 if quick else 12,
            layers=2, epochs=20 if quick else 200,
            checkpoint=None if quick else
            ckpt_dir / f"phase_a_{state_name}.pt")

        # Phase B: the dictionary is fitted to the frozen operator, not to data.
        # HIDDEN_DIMS is the EEG BSO's own setting, so the construction here is
        # identical to that experiment's.
        image = operator_image(operator, current)
        dictionary, K, diag = train_phase_b(
            current, image, operator_meta=operator_meta,
            epochs=20 if quick else 200, print_every=0 if quick else 50,
            checkpoint=None if quick else
            ckpt_dir / f"phase_b_{state_name}.pt")

        psi = dictionary_features(dictionary, state)
        _regress(f"bso_{state_name}", psi[:, 1:])
        _regress(f"linear_{state_name}", state)
        _score(f"bso_prop_{state_name}",
               propagated_forecast(psi, K, state.shape[1], readout, origins,
                                   horizon))

        mlp, mlp_scale = train_direct_mlp(
            state, index, fit_origins, horizon, state=state_name,
            epochs=40 if quick else 400,
            checkpoint=None if quick else
            ckpt_dir / f"mlp_{state_name}_h{horizon}.pt")
        _score(f"mlp_{state_name}", predict_mlp(mlp, state, origins, mlp_scale))

        diagnostics[state_name] = dict(
            state_dim=int(state.shape[1]),
            dictionary_size=int(dictionary.dictionary_size),
            n_learned=int(dictionary.n_learned_observables),
            invariance_mse=float(diag["weighted_invariance_mse"]),
            transition_radius=float(diag["transition_spectral_radius"]),
            growth=[float(np.linalg.norm(np.linalg.matrix_power(K, n), 2))
                    for n in range(1, horizon + 1)])

    # --- the LIM baselines, fitted as the literature defines them ---------
    # No ridge selection here, deliberately. A LIM is
    # ``G(tau) = C(tau) C(0)^-1`` on a truncated PC basis: the truncation to
    # COMMON_MODES modes *is* its regularisation, and bolting cross-validated
    # ridge on top would produce a better-conditioned estimator that is no
    # longer the method being compared against. The rule applied throughout is
    # that a **named literature baseline** is fitted as published, while a
    # **control whose only job is to isolate the dictionary** is fitted exactly
    # like the BSO -- see ``linear_*`` above.
    G = fit_lim(coeff[:data.train_end], ridge=0.0)
    _score("lim_prop", np.stack(
        [_nino34(data, lim_forecast(G, coeff[origins], horizon)[:, l]
                 @ eof.patterns[:COMMON_MODES]) for l in range(horizon)],
        axis=1))
    wang = fit_lim_wang(coeff[:data.train_end], index[:data.train_end],
                        tau0=2, ridge=0.0)
    _score("lim_wang", wang.index_forecast(coeff[origins], horizon))
    _score("lim_inst", predict_lim_direct(
        fit_lim_direct(coeff, index, fit_origins, horizon, ridge=0.0),
        coeff[origins]))

    # A control, not a LIM: 241 predictors is the regime where regularisation
    # decides the answer, so it is tuned like the BSO and labelled accordingly.
    _regress("lim_raw", np.nan_to_num(delay_state(coeff, HISTORY)))

    # --- unmatched reference: this one sees the whole field ---------------
    # Its own checkpoint per horizon, since the head count is the horizon.
    direct_ckpt = ("direct.pt" if horizon == HORIZON
                   else f"direct_h{horizon}.pt")
    direct_model, direct_scale = train_direct(
        data, fit_origins, index, horizon,
        modes=(6, 10) if quick else (12, 24), width=16 if quick else 48,
        layers=2 if quick else 4, epochs=30 if quick else 400,
        checkpoint=None if quick else RESULTS_DIR / "enso" / direct_ckpt)
    _score("direct_fno",
           predict_direct(direct_model, data.anomalies[origins], direct_scale))

    # --- trivial references, legitimately state-free ----------------------
    gamma = damped_persistence_gamma(data, horizon)
    launch = index[origins][:, None]
    leads = np.arange(1, horizon + 1)
    _score("climatology", np.zeros_like(truth_n34))
    _score("persistence", np.repeat(launch, horizon, axis=1))
    _score("damped_persistence", launch * (gamma ** leads)[None])

    # The claim, with its uncertainty: the dictionary against its own matched
    # linear control, on the same drawn origins.
    intervals = {
        state_name: acc_difference_interval(
            predictions[f"bso_{state_name}"],
            predictions[f"linear_{state_name}"],
            truth_n34, horizon, mask)
        for state_name in states}

    # The *figure's* claim, which is a different pair: the plotted propagated rung
    # against the plotted LIM. Kept out of ``intervals`` deliberately --
    # ``print_ladder`` derives ``bso_<key>``/``linear_<key>`` metric names from
    # that dict's keys, so an extra entry there would look up a rung that does not
    # exist.
    figure_interval = acc_difference_interval(
        predictions[f"{PRESENTATION_RUNG}_{PRESENTATION_STATE}"],
        predictions[PRESENTATION_LIM], truth_n34, horizon, mask)

    return dict(data=data, horizon=horizon, metrics=metrics, gamma=gamma,
                predictions=predictions, ridges=ridges, intervals=intervals,
                figure_interval=figure_interval,
                diagnostics=diagnostics, n_origins=len(origins),
                ragged=bool(ragged), n_fit=len(fit_origins),
                origins_per_lead=(None if mask is None
                                  else mask.sum(axis=0).astype(int)))


# Training-seed sensitivity of the propagated rung, measured by ``--seed-spread``
# (``logs/enso_index_seed_spread.log``, 2026-08-10).
#
# This exists because the block bootstrap does **not** measure it. That bootstrap
# resamples verification origins, so it captures how much the score depends on
# which years were verified; it holds the fitted operator fixed and therefore says
# nothing about how much the score depends on the optimiser's random draw. For
# this rung the second variance is the larger one past lead 12, and reporting only
# the first would understate the uncertainty.
#
# Through lead SEED_SPREAD_STABLE_THROUGH the three runs agree to within
# SEED_SPREAD_TOLERANCE ACC. Past it they do not: lead 24 spans
# SEED_SPREAD_LONG_RANGE. The figure plots one seed, so the caption states this.
SEED_SPREAD_SEEDS = (2026, 7, 12345)
SEED_SPREAD_STABLE_THROUGH = 9
SEED_SPREAD_TOLERANCE = 0.03
SEED_SPREAD_LONG_RANGE = (0.14, 0.34)


def seed_spread(quick: bool, horizon: int | None = None,
                seeds=SEED_SPREAD_SEEDS) -> dict:
    """``state -> (seeds, acc)`` for the propagated rung, one row per seed.

    Retrains phases A and B at each seed with **no checkpointing**, so the
    published checkpoints are neither read nor overwritten and the spread is a
    property of the training procedure rather than of what happens to be on disk.
    Only the propagated rung is scored: it is the one the figure draws, and the
    per-lead rungs refit their coefficients anyway so their seed dependence is a
    different and much smaller thing.
    """
    data = load_ersstv5(DATA_PATH)
    horizon = LONG_HORIZON if horizon is None else int(horizon)
    index = _nino34(data, data.anomalies)
    eof = fit_eofs(data, data.train_end, COMMON_MODES)
    states = reduced_states(data, index, eof.patterns)
    ends = _pair_ends(HISTORY - 1, data.train_end)

    origins = np.arange(data.train_end, data.test_end - 1)
    targets = origins[:, None] + np.arange(1, horizon + 1)[None, :]
    mask = targets < data.test_end
    truth = index[np.clip(targets, 0, data.test_end - 1)]

    out = {}
    for name, (state, readout) in states.items():
        current = state[ends]
        rows = []
        for seed in seeds:
            operator, meta = train_phase_a(
                current, state[ends + 1], state=name,
                train_end=data.train_end, width=8 if quick else 24,
                modes=6 if quick else 12, layers=2,
                epochs=20 if quick else 200, seed=int(seed), checkpoint=None)
            dictionary, k, _ = train_phase_b(
                current, operator_image(operator, current), operator_meta=meta,
                epochs=20 if quick else 200, print_every=0, seed=int(seed),
                checkpoint=None)
            prediction = propagated_forecast(
                dictionary_features(dictionary, state), k, state.shape[1],
                readout, origins, horizon)
            rows.append(_direct_metrics(prediction, truth, horizon,
                                        mask)["nino34_acc"])
        out[name] = (tuple(int(s) for s in seeds), np.stack(rows))
    return dict(horizon=horizon, spread=out)


def print_seed_spread(result: dict) -> None:
    """One row per seed, then the per-lead range -- the figure's other error bar."""
    horizon = result["horizon"]
    picks = [i for i in (1, 3, 6, 9, 12, 18, 24) if i <= horizon]
    header = "".join(f"{i:>7d}" for i in picks)
    for name, (seeds, acc) in result["spread"].items():
        print(f"\n[enso-index] propagated rung, {name} state: ACC by training "
              f"seed at leads{header}")
        for seed, row in zip(seeds, acc):
            print(f"  seed {seed:<10d}" + "".join(f"{row[i - 1]:7.2f}"
                                                 for i in picks))
        span = acc.max(axis=0) - acc.min(axis=0)
        print("  range over seeds " + "".join(f"{span[i - 1]:7.2f}"
                                             for i in picks))
        stable = [i for i in range(1, horizon + 1)
                  if span[i - 1] <= SEED_SPREAD_TOLERANCE]
        run_end = 0
        for lead in range(1, horizon + 1):
            if lead in stable:
                run_end = lead
            else:
                break
        print(f"  agrees to {SEED_SPREAD_TOLERANCE:.2f} ACC through lead "
              f"{run_end}; lead {horizon} spans "
              f"{acc[:, horizon - 1].min():.2f}–{acc[:, horizon - 1].max():.2f}")


# Every rung in LADDER_ROWS, each property explicit. Colour is the method family
# and line style separates variants inside it, which makes the two readings the
# table supports visible directly:
#
# * **red and purple** are the matched pairs -- solid is the dictionary, dashed
#   its per-lead linear control on the *same* state. That gap, and only that gap,
#   is the learned dictionary.
# * **orange** and **brown** hold one curve per state (solid index, dashed EOF),
#   so the gap there is the state rather than the method.
# * **blue** is the three LIMs, weighted so the published one (solid, Wang et al.)
#   reads as the baseline and the two variants recede.
#
# The one curve carrying LINEWIDTH["main"] in each family is the one the paper
# figure draws or compares against.
DIAGNOSTIC_STYLE = {
    "bso_eof": dict(
        label="BSO (EOF state)", color="#d62728", linestyle="-",
        linewidth=LINEWIDTH["main"], marker="o", zorder=11),
    "linear_eof": dict(
        label="Linear (EOF state)", color="#d62728", linestyle="--",
        linewidth=LINEWIDTH["data"], marker="None", zorder=10),
    "bso_index": dict(
        label="BSO (index state)", color="#7b3294", linestyle="-",
        linewidth=LINEWIDTH["data"], marker="o", zorder=9),
    "linear_index": dict(
        label="Linear (index state)", color="#7b3294", linestyle="--",
        linewidth=LINEWIDTH["data"], marker="None", zorder=8),
    "bso_prop_index": dict(
        label="BSO propagated, index (Ψ Kⁿ)", color="#ff7f0e", linestyle="-",
        linewidth=LINEWIDTH["main"], marker="D", zorder=7),
    "bso_prop_eof": dict(
        label="BSO propagated, EOF (Ψ Kⁿ)", color="#ff7f0e", linestyle="--",
        linewidth=LINEWIDTH["data"], marker="D", zorder=7),
    "mlp_eof": dict(
        label="MLP (EOF state)", color="#8c564b", linestyle="-",
        linewidth=LINEWIDTH["data"], marker="v", zorder=6),
    "mlp_index": dict(
        label="MLP (index state)", color="#8c564b", linestyle="--",
        linewidth=LINEWIDTH["data"], marker="v", zorder=6),
    "lim_wang": dict(
        label="LIM (Wang et al. τ₀=2)", color="#1f77b4", linestyle="-",
        linewidth=LINEWIDTH["main"], marker="^", zorder=5),
    "lim_inst": dict(
        label="LIM (per-lead lagged)", color="#1f77b4", linestyle="--",
        linewidth=LINEWIDTH["data"], marker="None", zorder=5),
    "lim_prop": dict(
        label="LIM (propagated lag-1)", color="#1f77b4", linestyle=":",
        linewidth=LINEWIDTH["data"], marker="None", zorder=4),
    "lim_raw": dict(
        label="Linear control (raw delay)", color="#17becf", linestyle="-",
        linewidth=LINEWIDTH["data"], marker="None", zorder=4),
    "direct_fno": dict(
        label="FNO (full field, unmatched)", color="#7f7f7f", linestyle="-",
        linewidth=LINEWIDTH["data"], marker="s", zorder=3),
    "damped_persistence": dict(
        label="Damped persistence", color="#2ca02c", linestyle="--",
        linewidth=LINEWIDTH["data"], marker="None", zorder=2),
    "climatology": dict(
        label="Climatology", color="#000000", linestyle=(0, (6, 4)),
        linewidth=LINEWIDTH["reference"], marker="None", zorder=1),
}

# Every rung in LADDER_ROWS is drawn here, so the legend runs to five rows at
# three columns. Each extra row costs 0.20 in of bottom margin and 0.20 in of
# overall height, which keeps the panels at their physical size rather than
# squeezing them -- the same accounting as the four-row version this replaced.
LAYOUT_DIAGNOSTIC = {
    "figsize": (DOUBLE_COLUMN_WIDTH, 3.95), "nrows": 1, "ncols": 2,
    "left": 0.52, "right": 0.10, "bottom": 1.46, "top": 0.52,
    "wgap": 0.62, "hgap": 0.0,
    "title_y": 0.975, "legend_loc": "lower center",
    "legend_bbox": (0.5, 0.005), "legend_ncol": 3,
}


def _draw(ax, x, values, kw, *, with_label):
    """One curve with every property stated; markers only where declared."""
    return ax.plot(
        x, values, label=kw["label"] if with_label else None,
        color=kw["color"], linestyle=kw["linestyle"],
        linewidth=kw["linewidth"], marker=kw["marker"],
        markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
        alpha=1.0, zorder=kw["zorder"])


def plot(metrics, horizon, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    leads = np.arange(1, horizon + 1)
    with publication_style():
        fig, axes = make_figure(LAYOUT_DIAGNOSTIC)
        rmse_ax, acc_ax = axes[0, 0], axes[0, 1]
        for name, kw in DIAGNOSTIC_STYLE.items():
            m = metrics[name]
            # A zero-anomaly forecast makes the Niño-3.4 RMSE equal to the RMS
            # of the verifying observations, which drifts slightly because each
            # lead verifies a different set of target months. That is a property
            # of the verification window, not of the forecast, so climatology is
            # drawn as its lead-average.
            rmse = (np.full_like(m["nino34_rmse"],
                                 float(np.mean(m["nino34_rmse"])))
                    if name == "climatology" else m["nino34_rmse"])
            _draw(rmse_ax, leads, rmse, kw, with_label=True)
            _draw(acc_ax, leads, m["nino34_acc"], kw, with_label=False)

        style_axis(rmse_ax, xlabel="Lead (months)", ylabel="RMSE (°C)",
                   title="Niño-3.4 error", grid=True)
        style_axis(acc_ax, xlabel="Lead (months)",
                   ylabel="Anomaly correlation", title="Niño-3.4 skill",
                   grid=True)
        for level in (0.5, 0.6):
            acc_ax.axhline(level, color=GRID_COLOR, linestyle=":",
                           linewidth=LINEWIDTH["reference"], zorder=0)
        acc_ax.set_ylim(-0.2, 1.02)
        for ax in (rmse_ax, acc_ax):
            ax.set_xlim(1, horizon)
            ax.set_xticks(leads[1::2])

        add_figure_title(
            fig, "Direct Niño-3.4 forecasting through the BSO framework: "
                 f"matched-state ladder ({HISTORY}-month memory, ERSSTv5)",
            y=LAYOUT_DIAGNOSTIC["title_y"])
        add_legend(fig, *rmse_ax.get_legend_handles_labels(),
                   loc=LAYOUT_DIAGNOSTIC["legend_loc"],
                   bbox_to_anchor=LAYOUT_DIAGNOSTIC["legend_bbox"],
                   ncol=LAYOUT_DIAGNOSTIC["legend_ncol"])
        pdf, _ = save_publication_pair(
            fig, out_dir / "enso_index_prediction_skill.pdf")
        plt.close(fig)
    return pdf


# Which reduced state the published figure reports. Both are computed; this
# names the one shown, so the choice is a stated decision rather than whichever
# happened to be plotted. ``index`` is the state whose forecast target is itself
# a coordinate, so the propagated rung's readout is a coordinate slice -- the
# most literal transcription of ``rollout_operator_bso``, which reads its last
# ``lag`` samples straight out of the reconstructed history.
PRESENTATION_STATE = "index"
# Which BSO rung. ``bso_prop`` is the propagated Galerkin operator: the same
# construction as the published EEG rollout, with nothing fitted per lead and
# nothing fitted to the index at all. The per-lead ``bso_`` rung is the KAF
# variant; it is a different estimator and would not be the EEG figure's object,
# so naming the rung here keeps the two from being confused under one label.
PRESENTATION_RUNG = "bso_prop"
# Which LIM. ``lim_wang`` is the only one of the three that is the referenced
# paper (Wang, Slawinska & Giannakis 2020, Eqs. 6/7/11/12); ``lim_prop`` is a
# lag-1 variant and ``lim_inst`` is a per-lead regression that ``enso_lim``
# documents as not a LIM. It is also the structural match for a propagated BSO:
# both advance an operator and read the index out, neither fits per lead.
PRESENTATION_LIM = "lim_wang"

# Four curves: the BSO, the paper's LIM, and the two trivial references.
#
# The matched linear control, the per-lead dictionary rung, the MLP and the
# raw-delay control are all computed and all reported by ``print_ladder``; they
# are kept out of this figure deliberately. The consequence is that **this figure
# does not on its own distinguish the learned dictionary from the state it is
# given** -- that separation lives in the ladder table, and the caption reports
# the BSO-against-LIM interval, which is what the panels actually show.
#
# Every curve's colour, dash pattern, weight, marker and draw order is stated
# here, per the publication specification. The BSO carries LINEWIDTH["main"];
# climatology is a reference level rather than a competing forecast.
def presentation_style(state: str = PRESENTATION_STATE,
                       lim: str = PRESENTATION_LIM,
                       rung: str = PRESENTATION_RUNG) -> dict:
    return {
        f"{rung}_{state}": dict(
            label="BSO", color="#d62728", linestyle="-",
            linewidth=LINEWIDTH["main"], marker="o", markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH, alpha=1.0, zorder=4),
        lim: dict(
            label="LIM", color="#1f77b4", linestyle="-",
            linewidth=LINEWIDTH["data"], marker="^", markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH, alpha=1.0, zorder=3),
        "damped_persistence": dict(
            label="Damped persistence", color="#2ca02c", linestyle="--",
            linewidth=LINEWIDTH["data"], marker="None", markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH, alpha=1.0, zorder=2),
        "climatology": dict(
            label="Climatology", color="#4d4d4d", linestyle=(0, (6, 4)),
            linewidth=LINEWIDTH["reference"], marker="None",
            markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
            alpha=1.0, zorder=1),
    }


# 1x2 wide layout. Margins and gaps in inches; the bottom margin holds one
# legend row (four entries) plus the x-axis label.
LAYOUT_PRESENTATION = {
    "figsize": (DOUBLE_COLUMN_WIDTH, 2.95), "nrows": 1, "ncols": 2,
    "left": 0.52, "right": 0.10, "bottom": 0.66, "top": 0.42,
    "wgap": 0.62, "hgap": 0.0,
    "title_y": 0.965, "legend_loc": "lower center",
    "legend_bbox": (0.5, 0.005), "legend_ncol": 4,
}


# Extended-lead variant: the same figure carried to two years. 24 points per
# curve still read cleanly, so it needs no styling special cases -- only the tick
# spacing scales, and that is derived from the horizon below.
LONG_HORIZON = 24
PRESENTATION_FILENAME = "nino34_forecast_skill.pdf"
LONG_FILENAME = "nino34_forecast_skill_long.pdf"


def _significant_spans(lower: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous lead ranges (1-based, inclusive) where the CI clears zero."""
    ahead = np.asarray(lower) > 0.0
    spans, start = [], None
    for i, flag in enumerate(ahead):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start + 1, i))
            start = None
    if start is not None:
        spans.append((start + 1, len(ahead)))
    return spans


def plot_presentation(metrics, horizon, out_dir: Path, *,
                      filename: str = PRESENTATION_FILENAME,
                      state: str = PRESENTATION_STATE,
                      lim: str = PRESENTATION_LIM,
                      rung: str = PRESENTATION_RUNG) -> Path:
    """Niño-3.4 RMSE (°C) and anomaly correlation, all on one reduced state.

    ``filename`` is explicit rather than inferred from ``horizon``: two runs at
    different horizons write two figures, and deriving the path from a threshold
    let one silently overwrite the other.

    Four curves and one unlabelled ACC = 0.5 threshold; no significance
    shading. The bootstrap intervals -- both the plotted pair's and the
    dictionary-against-its-control one -- are still computed and still printed by
    :func:`print_ladder`; they are simply not drawn.

    Both forecast curves revert to zero anomaly, so the error panel compares
    error rather than offset. That parity is the point of the state and readout
    conventions in :func:`index_delay_states` and ``enso_lim.WangLIM``; without it
    the BSO carried a cold bias reaching -0.47 °C and lost the RMSE panel on bias
    alone while having the lower error variance.

    Two properties of this curve set are handled here rather than left to the
    caption, because both would otherwise read as plotting bugs:

    * **Climatology has no anomaly correlation.** A zero-anomaly forecast has
      zero variance, so Pearson correlation is undefined and ``_corr`` returns
      NaN by design. It is drawn in the RMSE panel only, and since the legend
      labels are deliberately short here, that fact lives in the printed caption
      rather than on the axes.
    * **Damped persistence's correlation is exactly persistence's**, since
      correlation is invariant to the positive per-lead scaling ``gamma**lead``.
      That curve is the lag-τ autocorrelation of Niño-3.4; damping shows up in
      RMSE alone. Drawn dashed to mark it as the trivial reference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    style = presentation_style(state, lim, rung)
    leads = np.arange(1, horizon + 1)
    # Roughly six labelled ticks whatever the horizon: every 2 months at a
    # one-year lead, every 4 at two years.
    step = max(1, int(round(horizon / 6)))

    with publication_style():
        fig, axes = make_figure(LAYOUT_PRESENTATION)
        rmse_ax, acc_ax = axes[0, 0], axes[0, 1]
        for name, kw in style.items():
            m = metrics[name]
            # Climatology's RMSE drifts across leads only because each lead
            # verifies a different set of target months -- a property of the
            # verification window, not the forecast -- so it is drawn as its
            # lead-average constant.
            rmse = (np.full_like(m["nino34_rmse"],
                                 float(np.nanmean(m["nino34_rmse"])))
                    if name == "climatology" else m["nino34_rmse"])
            rmse_ax.plot(leads, rmse, **kw)
            if name != "climatology":
                acc_ax.plot(leads, m["nino34_acc"], **{**kw, "label": None})

        style_axis(rmse_ax, xlabel="Forecast lead (months)",
                   ylabel="RMSE (°C)", title="Error", grid=True)
        style_axis(acc_ax, xlabel="Forecast lead (months)",
                   ylabel="Anomaly correlation", title="Skill", grid=True)
        # The conventional ACC = 0.5 skill threshold, drawn unlabelled: the
        # value is readable off the y-axis, and an in-axes label costs more
        # clutter than it earns.
        acc_ax.axhline(0.5, color=GRID_COLOR, linestyle=":",
                       linewidth=LINEWIDTH["reference"], zorder=0)
        # Follow the data downward rather than fixing the floor: at longer
        # horizons the trivial baselines go clearly negative, and a hard -0.1
        # limit clipped them mid-curve so they looked like they stopped. The
        # min() keeps the one-year figure at its existing -0.1.
        acc_min = min(float(np.nanmin(metrics[name]["nino34_acc"]))
                      for name in style if name != "climatology")
        acc_ax.set_ylim(min(-0.1, np.floor((acc_min - 0.05) * 10) / 10), 1.02)
        for ax in (rmse_ax, acc_ax):
            ax.set_xlim(1, horizon)
            ax.set_xticks(np.arange(step, horizon + 1, step))

        # The RMSE panel carries every method, so its handles are complete.
        add_figure_title(fig, "Niño-3.4 forecast skill (ERSSTv5)",
                         y=LAYOUT_PRESENTATION["title_y"])
        add_legend(fig, *rmse_ax.get_legend_handles_labels(),
                   loc=LAYOUT_PRESENTATION["legend_loc"],
                   bbox_to_anchor=LAYOUT_PRESENTATION["legend_bbox"],
                   ncol=LAYOUT_PRESENTATION["legend_ncol"])
        pdf, _ = save_publication_pair(fig, out_dir / filename)
        plt.close(fig)
    return pdf


LADDER_ROWS = [
    ("lim_prop", "LIM (propagated lag-1, OLS)"),
    ("lim_wang", "LIM (Wang et al. t0=2, OLS)"),
    ("lim_inst", "LIM (per-lead lagged, OLS)"),
    ("lim_raw", "Linear control (raw 12x20 delay)"),
    ("linear_index", "Linear (index state)"),
    ("bso_index", "BSO (index state)"),
    ("bso_prop_index", "BSO propagated rollout (index)"),
    ("mlp_index", "MLP (index state)"),
    ("linear_eof", "Linear (EOF state)"),
    ("bso_eof", "BSO (EOF state)"),
    ("bso_prop_eof", "BSO propagated rollout (EOF)"),
    ("mlp_eof", "MLP (EOF state)"),
    ("direct_fno", "FNO (full field, unmatched)"),
    ("damped_persistence", "Damped persistence"),
    ("climatology", "Climatology"),
]


def print_ladder(result: dict) -> None:
    """The decision table: every rung, both metrics, and the claim's CI.

    Printed rather than plotted because it is what the figure's design should be
    read off, and because a table can carry rungs a six-curve figure cannot.
    """
    horizon = result["horizon"]
    picks = [i for i in (1, 3, 6, 9, 12, 18, 24) if i <= horizon]
    header = "".join(f"{i:>7d}" for i in picks)
    counts = result["origins_per_lead"]

    print(f"\n[enso-index] {result['n_fit']} fitting origins, "
          f"{result['n_origins']} verification origins"
          + ("" if counts is None else
             f" ({counts[0]} at lead 1 falling to {counts[-1]} at lead "
             f"{horizon})"))
    for state_name, diag in result["diagnostics"].items():
        print(f"[enso-index] {state_name:5s} state: dim {diag['state_dim']}, "
              f"dictionary 1 + {diag['state_dim']} + {diag['n_learned']} = "
              f"{diag['dictionary_size']}, invariance MSE "
              f"{diag['invariance_mse']:.3e}, rho(K) "
              f"{diag['transition_radius']:.4f}")

    for metric, label in (("nino34_acc", "anomaly correlation"),
                          ("nino34_rmse", "RMSE (°C)")):
        print(f"\n[enso-index] Niño-3.4 {label} at leads{header}")
        for key, name in LADDER_ROWS:
            if key not in result["metrics"]:
                continue
            values = result["metrics"][key][metric]
            print(f"  {name:30s}"
                  + "".join(f"{values[i - 1]:7.2f}" for i in picks))

    # The claim itself: dictionary minus its own matched linear control, with a
    # block-bootstrap interval. A lead is only evidence for the dictionary if
    # the interval clears zero there.
    print(f"\n[enso-index] ACC(BSO) - ACC(linear control), same state, "
          f"95% block bootstrap at leads{header}")
    for state_name, (lower, upper) in result["intervals"].items():
        delta = (result["metrics"][f"bso_{state_name}"]["nino34_acc"]
                 - result["metrics"][f"linear_{state_name}"]["nino34_acc"])
        print(f"  {state_name:5s} delta                   "
              + "".join(f"{delta[i - 1]:7.2f}" for i in picks))
        print(f"  {state_name:5s} 95% CI lower            "
              + "".join(f"{lower[i - 1]:7.2f}" for i in picks))
        print(f"  {state_name:5s} 95% CI upper            "
              + "".join(f"{upper[i - 1]:7.2f}" for i in picks))
        spans = _significant_spans(lower)
        print(f"  {state_name:5s} dictionary is ahead of its control at leads "
              + (", ".join(f"{a}–{b}" for a, b in spans) if spans else "none"))

    # What the published figure actually shows, which is the other comparison:
    # the propagated operator against the paper's LIM.
    plotted = f"{PRESENTATION_RUNG}_{PRESENTATION_STATE}"
    lower, upper = result["figure_interval"]
    delta = (result["metrics"][plotted]["nino34_acc"]
             - result["metrics"][PRESENTATION_LIM]["nino34_acc"])
    print(f"\n[enso-index] figure claim: ACC({plotted}) - "
          f"ACC({PRESENTATION_LIM}), 95% block bootstrap at leads{header}")
    for label, row in (("delta", delta), ("95% CI lower", lower),
                       ("95% CI upper", upper)):
        print(f"  {label:24s}" + "".join(f"{row[i - 1]:7.2f}" for i in picks))
    spans = _significant_spans(lower)
    print("  BSO is ahead of the LIM at leads "
          + (", ".join(f"{a}–{b}" for a, b in spans) if spans else "none"))

    ridges = result["ridges"]
    print("\n[enso-index] selected ridge (min–max over leads), blocked CV:")
    for key, name in LADDER_ROWS:
        if key in ridges:
            print(f"  {name:30s} {ridges[key].min():.2e} – "
                  f"{ridges[key].max():.2e}")


def figure_caption(result: dict, state: str, lim: str,
                   rung: str = PRESENTATION_RUNG) -> str:
    """One caption for whichever horizon was run, built from that run's numbers.

    Shared by both figure branches: the earlier version emitted a caption only
    on the 12-month path, so the figure that actually goes into the paper had
    none and the crossover lead quoted in it came from a different run.

    Descriptive rather than argumentative on purpose. It states what is plotted,
    how each curve was fitted, and what the interval does and does not cover; it
    does not nominate a winner. Where the record does not separate two methods,
    the caption says so instead of picking one.

    Parameterised by ``state`` and ``rung`` rather than describing one hardcoded
    construction: an earlier version took ``state`` but spelled out the EOF
    delay-PCA state and the per-lead regression regardless, so changing either
    presentation constant silently produced a caption for a figure that was not
    drawn.
    """
    horizon = result["horizon"]
    lower, _ = result["figure_interval"]
    spans = _significant_spans(lower)
    # Two uncertainties, both stated, because they measure different things: the
    # bootstrap resamples verification origins with the operator held fixed, and
    # the seed range holds the origins fixed and refits. Past lead
    # SEED_SPREAD_STABLE_THROUGH the second is the larger of the two.
    ahead = (
        "The BSO is separated from it at leads "
        + ", ".join(f"{a}–{b}" for a, b in spans)
        + " months by a 95 % block bootstrap on the paired per-lead difference"
        if spans else
        "The two are not separated at any lead by a 95 % block bootstrap on the "
        "paired per-lead difference: on this record, with this information, the "
        "neural-operator forecast matches the classical benchmark rather than "
        "beating it")
    seed_note = (
        f" A single training seed is plotted; retraining changes long-lead "
        f"correlation appreciably while leaving leads 1–{SEED_SPREAD_STABLE_THROUGH} "
        f"within {SEED_SPREAD_TOLERANCE:.2f} ACC, with lead {horizon} spanning "
        f"{SEED_SPREAD_LONG_RANGE[0]:.2f}–{SEED_SPREAD_LONG_RANGE[1]:.2f} over "
        f"{len(SEED_SPREAD_SEEDS)} seeds, so the long-lead values should be read "
        "as indicative rather than as point estimates."
        if horizon > SEED_SPREAD_STABLE_THROUGH else "")

    predictor = ("the " if state == "index" else "a ") + (
        f"{HISTORY}-month Niño-3.4 history"
        if state == "index" else
        f"training-fit {REDUCED_DIM}-dimensional PCA of the {HISTORY}-month "
        f"delay embedding of {COMMON_MODES} weighted EOF coefficients")
    forecast = (
        "every lead is then obtained by propagating that operator as Ψ Kⁿ and "
        "reading the state coordinates back out through the fixed inverse state "
        "map, so nothing is fitted per lead and nothing is fitted to the index — "
        "the same construction as the published resting-EEG rollout"
        if rung == "bso_prop" else
        "per-lead coefficients are then regressed on the dictionary")

    return (
        f"Niño-3.4 forecast skill on ERSSTv5, leads 1–{horizon}. The BSO "
        f"forecasts the index directly: a reduced state — {predictor} — is "
        "advanced by a frozen neural operator, a dictionary Ψ(z) = [1, z, h(z)] "
        "is fitted against that operator by the Galerkin invariance objective, "
        f"and {forecast}. LIM is the linear inverse model of Wang, Slawinska and "
        f"Giannakis (2020) on {COMMON_MODES} weighted EOF coefficients — a "
        "propagator estimated at a two-month lag, other leads by its fractional "
        "powers — fitted by ordinary least squares, since its regularisation is "
        "the principal-component truncation. Both forecasts decay to zero "
        "anomaly, the climatology the index is defined against, so the error "
        f"panel compares error rather than offset. {ahead}.{seed_note} The full "
        "ladder, including a per-lead linear control on the same state that "
        "isolates the learned dictionary, is reported in the accompanying table — "
        "these panels do not by themselves separate the dictionary from the state "
        "it is given. "
        "Climatology has undefined anomaly correlation (a zero-anomaly forecast "
        "has zero variance) so it appears in the error panel alone, drawn as "
        "its lead-average. Damped persistence has exactly the correlation of "
        "plain persistence, because correlation is invariant to the positive "
        "per-lead scaling γ^τ; damping is visible in RMSE only.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--ladder", action="store_true",
        help="print the matched-state ladder and the bootstrap interval on the "
             "claim, and write no figure. This is the table the figure's design "
             "should be read off.")
    parser.add_argument(
        "--seed-spread", action="store_true",
        help="retrain the propagated rung at several seeds and print the ACC "
             "range, then exit. This is the uncertainty the block bootstrap does "
             "not measure: it resamples verification origins with the operator "
             "held fixed, so it cannot see how much the score moves with the "
             "optimiser's random draw. Writes no checkpoint and no figure.")
    parser.add_argument(
        "--long-only", action="store_true",
        help=f"write only the {LONG_HORIZON}-month figure ({LONG_FILENAME}) "
             f"and skip the {HORIZON}-month one. Both are produced by default. "
             "At the longer horizon each lead is verified on every origin that "
             "still has an observed target, so the sample shrinks with lead, "
             "and the direct FNO and the MLPs are retrained because their head "
             "count is the horizon.")
    args = parser.parse_args()

    if args.seed_spread:
        print_seed_spread(seed_spread(args.quick))
        return

    if args.ladder:
        print_ladder(run(args.quick, horizon=LONG_HORIZON, ragged=True))
        return

    # Both lead ranges, from two runs rather than one truncated to the other:
    # ``fit_origins`` reserves ``horizon`` months of target, so the two are
    # fitted on slightly different samples and neither is a slice of the other.
    plans = [(LONG_HORIZON, True, LONG_FILENAME)]
    if not args.long_only:
        plans.insert(0, (4 if args.quick else HORIZON, False,
                         PRESENTATION_FILENAME))

    for horizon, ragged, filename in plans:
        result = run(args.quick, horizon=horizon, ragged=ragged)
        horizon = result["horizon"]
        talk_pdf = plot_presentation(
            result["metrics"], horizon, FIGURES_DIR / "enso",
            filename=filename, state=PRESENTATION_STATE,
            lim=PRESENTATION_LIM, rung=PRESENTATION_RUNG)
        print_ladder(result)
        print(f"\n[enso-index] {horizon}-month figure written to {talk_pdf}")
        if filename == PRESENTATION_FILENAME:
            pdf = plot(result["metrics"], horizon, FIGURES_DIR / "enso")
            print(f"[enso-index] diagnostic figure written to {pdf}")
        print_figure_caption(figure_caption(
            result, PRESENTATION_STATE, PRESENTATION_LIM, PRESENTATION_RUNG))


if __name__ == "__main__":
    main()
