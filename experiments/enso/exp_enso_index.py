"""Direct Niño-3.4 forecasting through the BSO framework.

Every other BSO in the ENSO suite (``exp_enso``, ``exp_enso_delay``) advances
the Indo-Pacific **field** and obtains Niño-3.4 afterwards by area-averaging the
decoded field. ``enso_direct.FieldToIndexFNO`` forecasts the index directly but
bypasses the BSO entirely: it has no dictionary and forms no Galerkin matrix.

This module supplies the missing object -- the direct-forecast construction
carried *inside* the BSO framework:

    y(t+tau) ~= sum_j c_j(tau) psi_j(z(t)),        z(t) = [y(t-11) ... y(t)]

The dictionary ``Psi`` is trained by the two-phase BSO recipe (a frozen neural
operator, then a Galerkin/EDMD-DL invariance objective) and the per-lead
coefficients ``c(tau)`` are fitted by ridge least squares. That is structurally
**exactly the KAF construction** of Wang, Slawinska & Giannakis (2020),
``c_l(tau) = (1/N) sum_n phi_ln y_{n+q}``, with the BSO's learned dictionary in
place of their kernel eigenbasis. The Galerkin matrix ``K`` is still produced by
phase B -- it is what shapes the dictionary -- but it is *not* used to forecast,
which is the difference from the propagated ``bno`` rows.

Why the EEG BSO rather than ``run_bso_pipeline``
------------------------------------------------
The repository holds two different notions of "dictionary" and only one gives a
non-trivial direct forecast:

* ``run_bso_pipeline`` (``edmd_bso_method.build_basis``) builds columns from
  ``BasisNet(coords)`` -- functions of the *spatial coordinate*. A state enters
  only through ``c_j = <psi_j, u>_w``, which is **linear in the state**. With a
  12-point state the dictionary spans all of R^12 plus the constant, so a direct
  per-lead regression on it would be *identically* linear regression on
  ``[1, z]``: provably degenerate.
* ``eeg_edmddl.PaperEDMDDLDictionary`` builds ``Psi(z) = [1, z, h_theta(z)]``,
  an MLP of the *state*. This is the nonlinear hypothesis space KAF assumes, so
  it is the one used here.

The linear case is not discarded, it is kept as the control (``bso_direct_linear``
below), which is what makes the learned variant's margin interpretable.

Usage:
    python -m experiments.enso.exp_enso_index [--quick]
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

sys.dont_write_bytecode = True

from edmd_bso_method import device, set_seed
from experiments.eeg.eeg_edmddl import PaperEDMDDLDictionary
# Both helpers are generic in the state dimension -- they validate against their
# arguments, not the EEG constants -- so they are shared verbatim rather than
# reimplemented, and the two experiments cannot drift apart.
from experiments.eeg.eeg_operator_bso import (TemporalFNOStateOperator,
                                               _dictionary_values,
                                               _galerkin_transition)
from experiments.enso.enso_data import fit_eofs, load_ersstv5
from experiments.enso.enso_direct import predict_direct, train_direct
from experiments.enso.enso_lim import fit_lim_direct, predict_lim_direct
from experiments.enso.exp_enso import (DATA_PATH, HORIZON, LIM_MODES, _corr,
                                        _nino34, damped_persistence_gamma,
                                        encode, fit_lim, lim_forecast)
from experiments.enso.exp_enso_delay import HISTORY, delay_state
from figure_style import (DOUBLE_COLUMN_WIDTH, GRID_COLOR, LINEWIDTH,
                          MARKER_EDGE_WIDTH, MARKER_SIZE, TEXT_COLOR,
                          add_figure_title, add_legend, make_figure,
                          publication_style, save_publication_pair,
                          style_axis)
from plotting_utils import print_figure_caption
from project_paths import FIGURES_DIR, RESULTS_DIR

# 1 constant + HISTORY fixed state coordinates + 11 learned functions, mirroring
# the EEG split of 1 + 8 + 11 (``eeg_operator_bso.OperatorBSOConfig``).
DICTIONARY_SIZE = 1 + HISTORY + 11
TRANSITION_RIDGE = 1e-6
# ``eeg_operator_bso.OperatorBSOConfig.dictionary_hidden_dims``, so the
# dictionary construction is identical to the EEG experiment's. The wider fit is
# reported alongside it as a sensitivity check, not as an alternative.
HIDDEN_DIMS = (16, 16)
WIDE_HIDDEN_DIMS = (64, 64)


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def index_delay_states(index: np.ndarray, train_end: int,
                       history: int = HISTORY):
    """Standardised ``history``-month delay embedding of a scalar index.

    Returns ``(states, mean, scale)`` where row ``t`` of ``states`` holds months
    ``t-history+1 … t`` oldest-first (``exp_enso_delay.delay_state`` with a
    single mode), so its **newest** entry is the standardised index at ``t``.
    Rows without full context are NaN. Centring and scaling use training months
    only, so no test statistic leaks into the state definition.
    """
    index = np.asarray(index, dtype=np.float64)
    train = index[:int(train_end)]
    mean, scale = float(train.mean()), float(train.std())
    scale = scale if scale > 1e-12 else 1.0
    standard = (index - mean) / scale
    return delay_state(standard[:, None], int(history)), mean, scale


def _pair_ends(first: int, last: int) -> np.ndarray:
    """Ends ``t`` of one-month pairs ``z(t) -> z(t+1)`` with ``t+1 < last``."""
    return np.arange(int(first), int(last) - 1)


# ---------------------------------------------------------------------------
# Phase A: neural operator on the delay state
# ---------------------------------------------------------------------------

def _operator_metadata(history, epochs, train_end, width, modes,
                       layers) -> dict:
    return {"model": "index_delay_temporal_fno", "history": int(history),
            "epochs": int(epochs), "train_end": int(train_end),
            "width": int(width), "modes": int(modes), "layers": int(layers)}


def train_phase_a(current: np.ndarray, target: np.ndarray, *, history: int,
                  train_end: int, width: int = 24, modes: int = 12,
                  layers: int = 2, epochs: int = 200, batch_size: int = 64,
                  lr: float = 5e-4, weight_decay: float = 2e-5,
                  seed: int = 2026, checkpoint: Path | None = None):
    """Fit and freeze the one-month flow map ``z(t) -> z(t+1)``.

    Because every coordinate of the state is an index value, this *is* direct
    index supervision: the target's newest coordinate is ``y(t+1)``. Unlike
    ``run_bso_pipeline``'s phase A there is no two-step (``sg_weight``) term --
    the EEG operator is trained one-step only, and that convention is kept.
    """
    meta = _operator_metadata(history, epochs, train_end, width, modes, layers)
    set_seed(seed)
    operator = TemporalFNOStateOperator(
        1, int(history), hidden_channels=int(width), n_modes=int(modes),
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
            return operator.eval()

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
    return operator


def operator_image(operator, states: np.ndarray,
                   batch_size: int = 256) -> np.ndarray:
    """``(N, history)`` one-month images of the frozen phase-A operator."""
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
                  *, dictionary_size: int = DICTIONARY_SIZE,
                  hidden_dims=HIDDEN_DIMS, epochs: int = 200,
                  batch_size: int = 64, lr: float = 1e-3, seed: int = 2026,
                  print_every: int = 0, checkpoint: Path | None = None):
    """Train ``Psi(z) = [1, z, h_theta(z)]`` against the frozen operator image.

    Alternates the closed-form Galerkin solve for ``K`` with gradient steps on
    the invariance residual ``Psi(z) K - Psi(F(z))``, exactly the alternation in
    ``eeg_operator_bso._train_phase_b``; the loop is written out here rather
    than imported because that function is bound to ``OperatorBSOConfig``'s EEG
    defaults. ``K`` is returned for diagnostics only -- forecasting never uses
    it.
    """
    set_seed(seed)
    dictionary = PaperEDMDDLDictionary(
        state_dimension=int(current_state.shape[1]),
        dictionary_size=int(dictionary_size),
        hidden_dims=tuple(int(h) for h in hidden_dims)).to(device)
    weights = np.full(len(current_state), 1.0 / len(current_state))
    meta = {"model": "index_edmddl_dictionary",
            "state_dimension": int(current_state.shape[1]),
            "dictionary_size": int(dictionary_size),
            "hidden_dims": tuple(int(h) for h in hidden_dims),
            "epochs": int(epochs), "n_pairs": int(len(current_state))}

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


def run(quick: bool, horizon: int | None = None, ragged: bool = False) -> dict:
    """Fit everything and score it.

    ``horizon`` overrides the 12-month default. ``ragged`` scores each lead on
    every origin that has an observed target rather than on one common origin
    set -- necessary past ~24 months, where a fixed origin set would collapse to
    a handful of launches. It is off at the default horizon so those numbers are
    unaffected.
    """
    data = load_ersstv5(DATA_PATH)
    if horizon is None:
        horizon = 4 if quick else HORIZON
    horizon = int(horizon)
    ckpt_dir = None if quick else RESULTS_DIR / "enso_index"

    index = _nino34(data, data.anomalies)
    states, _, _ = index_delay_states(index, data.train_end)

    # Phase A on training pairs only: the latest target month is train_end - 1.
    ends = _pair_ends(HISTORY - 1, data.train_end)
    current, target = states[ends], states[ends + 1]
    operator = train_phase_a(
        current, target, history=HISTORY, train_end=data.train_end,
        width=8 if quick else 24, modes=6 if quick else 12,
        layers=2, epochs=20 if quick else 200,
        checkpoint=None if quick else ckpt_dir / "phase_a.pt")

    # Phase B: the dictionary is fitted to the frozen operator, not to data.
    # Fitted at both hidden widths. HIDDEN_DIMS is the EEG BSO's own setting
    # (``OperatorBSOConfig.dictionary_hidden_dims``) and is the headline row, so
    # the dictionary construction here matches the EEG experiment exactly; the
    # wide fit is carried alongside purely to show the choice does not drive the
    # result. Everything else about the two is identical.
    image = operator_image(operator, current)
    dictionaries, K, dict_diag = {}, None, None
    for key, hidden in (("", HIDDEN_DIMS), ("_w64", WIDE_HIDDEN_DIMS)):
        dictionaries[key], transition, diag = train_phase_b(
            current, image, hidden_dims=hidden,
            epochs=20 if quick else 200, print_every=0 if quick else 50,
            checkpoint=None if quick else
            ckpt_dir / f"phase_b_h{hidden[0]}.pt")
        if not key:                       # diagnostics report the headline fit
            K, dict_diag = transition, diag

    # --- the direct step -------------------------------------------------
    # ``fit_lim_direct`` prepends an intercept, so passing Psi without its
    # constant column makes that intercept *be* Psi's constant: the fit is
    # exactly the per-lead ridge least squares in the module docstring. Every
    # variant goes through the same call, so the only difference between them is
    # the dictionary itself.
    psi_linear = states                                   # Psi(z) = [1, z]

    # Fixed origin set: every launch verifies every lead. Ragged: launch from
    # every post-training month and let each lead use the origins that still
    # have an observed target inside the test window. The 18-month reserve past
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
    fit_origins = np.arange(HISTORY - 1, data.train_end - horizon)

    variants = [(f"bso_direct_learned{key}",
                 dictionary_features(d, states)[:, 1:])
                for key, d in dictionaries.items()]
    metrics = {}
    for name, features in variants + [("bso_direct_linear", psi_linear)]:
        weights = fit_lim_direct(features, index, fit_origins, horizon)
        metrics[name] = _direct_metrics(
            predict_lim_direct(weights, features[origins]), truth_n34, horizon,
            mask)

    # --- baselines, all index-space and all fitted training-only ---------
    eof = fit_eofs(data, data.train_end, LIM_MODES)
    coeff = encode(eof.patterns[:LIM_MODES], data.weights, data.anomalies)
    G = fit_lim(coeff[:data.train_end])
    lim_n34 = np.stack(
        [_nino34(data, lim_forecast(G, coeff[origins], horizon)[:, l]
                 @ eof.patterns[:LIM_MODES]) for l in range(horizon)], axis=1)
    metrics["lim"] = _direct_metrics(lim_n34, truth_n34, horizon, mask)

    lim_direct_w = fit_lim_direct(coeff, index, fit_origins, horizon)
    metrics["lim_direct"] = _direct_metrics(
        predict_lim_direct(lim_direct_w, coeff[origins]), truth_n34, horizon,
        mask)

    # At the default horizon this reuses the checkpoint written by ``exp_enso``,
    # so the shared comparator is the identical trained model rather than a
    # re-fit. A different horizon changes the number of output heads, so it gets
    # its own file instead of silently overwriting the 12-month one.
    direct_ckpt = ("direct.pt" if horizon == HORIZON
                   else f"direct_h{horizon}.pt")
    direct_model, direct_scale = train_direct(
        data, fit_origins, index, horizon,
        modes=(6, 10) if quick else (12, 24), width=16 if quick else 48,
        layers=2 if quick else 4, epochs=30 if quick else 400,
        checkpoint=None if quick else RESULTS_DIR / "enso" / direct_ckpt)
    metrics["direct_fno"] = _direct_metrics(
        predict_direct(direct_model, data.anomalies[origins], direct_scale),
        truth_n34, horizon, mask)

    gamma = damped_persistence_gamma(data, horizon)
    launch = index[origins][:, None]
    leads = np.arange(1, horizon + 1)
    metrics["climatology"] = _direct_metrics(
        np.zeros_like(truth_n34), truth_n34, horizon, mask)
    metrics["persistence"] = _direct_metrics(
        np.repeat(launch, horizon, axis=1), truth_n34, horizon, mask)
    metrics["damped_persistence"] = _direct_metrics(
        launch * (gamma ** leads)[None], truth_n34, horizon, mask)

    return dict(data=data, horizon=horizon, metrics=metrics, gamma=gamma,
                n_origins=len(origins), ragged=bool(ragged),
                origins_per_lead=(None if mask is None
                                  else mask.sum(axis=0).astype(int)), n_fit=len(fit_origins),
                dictionary_size=dictionaries[""].dictionary_size,
                n_learned=dictionaries[""].n_learned_observables,
                invariance_mse=float(dict_diag["weighted_invariance_mse"]),
                transition_radius=float(dict_diag["transition_spectral_radius"]),
                growth=[float(np.linalg.norm(np.linalg.matrix_power(K, n), 2))
                        for n in range(1, horizon + 1)])


# Nine curves, every property explicit. Dashed entries are the ones expected to
# lie on top of another curve: the wide-dictionary sensitivity check tracks the
# headline fit, and damped persistence has exactly persistence's correlation.
# Drawing them dashed rather than omitting them keeps a legend entry from
# reading as a missing line.
DIAGNOSTIC_STYLE = {
    "bso_direct_learned": dict(
        label="BSO direct (learned)", color="#d62728", linestyle="-",
        linewidth=LINEWIDTH["main"], marker="o", zorder=9),
    "bso_direct_learned_w64": dict(
        label="BSO direct (wide 64×64)", color="#e377c2", linestyle="--",
        linewidth=LINEWIDTH["data"], marker="None", zorder=8),
    "bso_direct_linear": dict(
        label="BSO direct (linear control)", color="#9467bd", linestyle="-",
        linewidth=LINEWIDTH["data"], marker="o", zorder=7),
    "direct_fno": dict(
        label="Direct FNO (index only)", color="#ff7f0e", linestyle="-",
        linewidth=LINEWIDTH["data"], marker="s", zorder=6),
    "lim_direct": dict(
        label="Direct lagged LIM", color="#8c564b", linestyle="-",
        linewidth=LINEWIDTH["data"], marker="^", zorder=5),
    "lim": dict(
        label="LIM (field rollout)", color="#1f77b4", linestyle="-",
        linewidth=LINEWIDTH["data"], marker="v", zorder=4),
    "climatology": dict(
        label="Climatology", color="#000000", linestyle=(0, (6, 4)),
        linewidth=LINEWIDTH["reference"], marker="None", zorder=1),
    "persistence": dict(
        label="Persistence", color="#7f7f7f", linestyle="-",
        linewidth=LINEWIDTH["data"], marker="None", zorder=2),
    "damped_persistence": dict(
        label="Damped persistence", color="#2ca02c", linestyle="--",
        linewidth=LINEWIDTH["data"], marker="None", zorder=3),
}

LAYOUT_DIAGNOSTIC = {
    "figsize": (DOUBLE_COLUMN_WIDTH, 3.55), "nrows": 1, "ncols": 2,
    "left": 0.52, "right": 0.10, "bottom": 1.06, "top": 0.52,
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
            fig, "Direct Niño-3.4 forecasting through the BSO framework "
                 f"({HISTORY}-month index delay state, ERSSTv5)",
            y=LAYOUT_DIAGNOSTIC["title_y"])
        add_legend(fig, *rmse_ax.get_legend_handles_labels(),
                   loc=LAYOUT_DIAGNOSTIC["legend_loc"],
                   bbox_to_anchor=LAYOUT_DIAGNOSTIC["legend_bbox"],
                   ncol=LAYOUT_DIAGNOSTIC["legend_ncol"])
        pdf, _ = save_publication_pair(
            fig, out_dir / "enso_index_prediction_skill.pdf")
        plt.close(fig)
    return pdf


# Pruned to five curves: the full nine-curve diagnostic above stays as the
# reference figure, and this one drops the linear control, the wide-dictionary
# sensitivity curve, the propagated LIM and plain persistence.
# Every curve's colour, dash pattern, weight, marker and draw order is stated
# here, per the publication specification -- nothing is left to a Matplotlib
# default. The BSO carries LINEWIDTH["main"]; climatology is a reference level
# rather than a competing forecast and carries LINEWIDTH["reference"].
# Direct FNO is purple rather than the diagnostic figure's orange: its curve
# nearly coincides with the BSO's and red/orange do not separate cleanly.
PRESENTATION_STYLE = {
    "bso_direct_learned": dict(
        label="BSO", color="#d62728", linestyle="-",
        linewidth=LINEWIDTH["main"], marker="o", markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH, alpha=1.0, zorder=5),
    "direct_fno": dict(
        label="FNO", color="#7b3294", linestyle="-",
        linewidth=LINEWIDTH["data"], marker="s", markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGE_WIDTH, alpha=1.0, zorder=4),
    "lim_direct": dict(
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
# legend row plus the x-axis label.
LAYOUT_PRESENTATION = {
    "figsize": (DOUBLE_COLUMN_WIDTH, 2.95), "nrows": 1, "ncols": 2,
    "left": 0.52, "right": 0.10, "bottom": 0.66, "top": 0.42,
    "wgap": 0.62, "hgap": 0.0,
    "title_y": 0.965, "legend_loc": "lower center",
    "legend_bbox": (0.5, 0.005), "legend_ncol": 5,
}


# Extended-lead variant: the same figure carried to two years. 24 points per
# curve still read cleanly, so it needs no styling special cases -- only the tick
# spacing scales, and that is derived from the horizon below.
LONG_HORIZON = 24
PRESENTATION_FILENAME = "nino34_forecast_skill.pdf"
LONG_FILENAME = "nino34_forecast_skill_long.pdf"


def plot_presentation(metrics, horizon, out_dir: Path, *,
                      filename: str = PRESENTATION_FILENAME) -> Path:
    """Niño-3.4 RMSE (°C) and anomaly correlation for five methods.

    ``filename`` is explicit rather than inferred from ``horizon``: two runs at
    different horizons write two figures, and deriving the path from a threshold
    let one silently overwrite the other.

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
    leads = np.arange(1, horizon + 1)
    # Roughly six labelled ticks whatever the horizon: every 2 months at a
    # one-year lead, every 4 at two years.
    step = max(1, int(round(horizon / 6)))

    with publication_style():
        fig, axes = make_figure(LAYOUT_PRESENTATION)
        rmse_ax, acc_ax = axes[0, 0], axes[0, 1]
        for name, kw in PRESENTATION_STYLE.items():
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
        acc_ax.axhline(0.5, color=GRID_COLOR, linestyle=":",
                       linewidth=LINEWIDTH["reference"], zorder=0)
        acc_ax.text(horizon, 0.52, "ACC = 0.5", ha="right", va="bottom",
                    fontsize=8, color=TEXT_COLOR)
        # Follow the data downward rather than fixing the floor: at longer
        # horizons the trivial baselines go clearly negative, and a hard -0.1
        # limit clipped them mid-curve so they looked like they stopped. The
        # min() keeps the one-year figure at its existing -0.1.
        acc_min = min(float(np.nanmin(metrics[name]["nino34_acc"]))
                      for name in PRESENTATION_STYLE if name != "climatology")
        acc_ax.set_ylim(min(-0.1, np.floor((acc_min - 0.05) * 10) / 10), 1.02)
        for ax in (rmse_ax, acc_ax):
            ax.set_xlim(1, horizon)
            ax.set_xticks(np.arange(step, horizon + 1, step))

        # The RMSE panel carries all five methods, so its handles are complete.
        add_figure_title(fig, "Niño-3.4 Forecast skill (ERSSTv5)",
                         y=LAYOUT_PRESENTATION["title_y"])
        add_legend(fig, *rmse_ax.get_legend_handles_labels(),
                   loc=LAYOUT_PRESENTATION["legend_loc"],
                   bbox_to_anchor=LAYOUT_PRESENTATION["legend_bbox"],
                   ncol=LAYOUT_PRESENTATION["legend_ncol"])
        pdf, _ = save_publication_pair(fig, out_dir / filename)
        plt.close(fig)
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--long", action="store_true",
        help=f"score out to {LONG_HORIZON} months instead of {HORIZON}, into a "
             f"separate figure ({LONG_FILENAME}). Each lead is verified on "
             "every origin that still has an observed target, so the sample "
             "shrinks with lead; the direct FNO is retrained because its head "
             "count is the horizon.")
    args = parser.parse_args()
    if args.long:
        result = run(args.quick, horizon=LONG_HORIZON, ragged=True)
        counts = result["origins_per_lead"]
        horizon = result["horizon"]
        pdf = plot_presentation(result["metrics"], horizon,
                                FIGURES_DIR / "enso", filename=LONG_FILENAME)
        print(f"\n[enso-index] {horizon}-month figure written to {pdf}")
        print(f"[enso-index] verifying origins fall from {counts[0]} at lead 1 "
              f"to {counts[-1]} at lead {horizon}")
        acc = result["metrics"]["bso_direct_learned"]["nino34_acc"]
        below = np.flatnonzero(acc < 0.5)
        print("[enso-index] BSO Niño-3.4 ACC drops below 0.5 at lead "
              f"{int(below[0]) + 1}" if len(below) else
              "[enso-index] BSO ACC stays above 0.5 across all leads")
        picks = [i for i in (1, 6, 12, 18, 24) if i <= horizon]
        print("[enso-index] Niño-3.4 ACC at leads "
              + ", ".join(str(i) for i in picks) + ":")
        for name, m in result["metrics"].items():
            print(f"  {name:24s} "
                  + " ".join(f"{m['nino34_acc'][i - 1]:6.2f}" for i in picks))
        return
    result = run(args.quick)
    pdf = plot(result["metrics"], result["horizon"], FIGURES_DIR / "enso")
    talk_pdf = plot_presentation(result["metrics"], result["horizon"],
                                 FIGURES_DIR / "enso")
    print(f"\n[enso-index] dictionary 1 + {HISTORY} + "
          f"{result['n_learned']} = {result['dictionary_size']} functions")
    print(f"[enso-index] phase-B invariance MSE = "
          f"{result['invariance_mse']:.4e}, rho(K) = "
          f"{result['transition_radius']:.4f}")
    print(f"[enso-index] {result['n_fit']} fitting origins, "
          f"{result['n_origins']} verification origins, "
          f"gamma = {result['gamma']:.3f}")
    print("[enso-index] Niño-3.4 anomaly correlation by method (lead 1..H):")
    for name, m in result["metrics"].items():
        acc = " ".join(f"{v:5.2f}" for v in m["nino34_acc"])
        print(f"  {name:22s} {acc}")
    print("[enso-index] Niño-3.4 RMSE (°C) by method (lead 1..H):")
    for name, m in result["metrics"].items():
        rmse = " ".join(f"{v:5.2f}" for v in m["nino34_rmse"])
        print(f"  {name:22s} {rmse}")
    # Whether the EEG-parity width drives the result, stated as a number rather
    # than left to the eye.
    narrow, wide = (result["metrics"]["bso_direct_learned"],
                    result["metrics"]["bso_direct_learned_w64"])
    print(f"[enso-index] dictionary width {HIDDEN_DIMS} (headline, = EEG) vs "
          f"{WIDE_HIDDEN_DIMS}: max |ΔACC| = "
          f"{np.nanmax(np.abs(narrow['nino34_acc'] - wide['nino34_acc'])):.3f}, "
          f"max |ΔRMSE| = "
          f"{np.nanmax(np.abs(narrow['nino34_rmse'] - wide['nino34_rmse'])):.3f} °C")
    print(f"[enso-index] figure written to {pdf}")
    print(f"[enso-index] presentation figure written to {talk_pdf}")
    # State the crossover from the numbers rather than by eye, so the caption
    # cannot drift out of step with a rerun.
    acc = result["metrics"]["bso_direct_learned"]["nino34_acc"]
    ref = result["metrics"]["lim_direct"]["nino34_acc"]
    ahead = np.where(acc > ref)[0]
    crossover = (f"is the stronger method through lead {int(ahead[0])}, after "
                 "which the BSO leads" if len(ahead)
                 else "is the stronger method at every lead shown")
    print_figure_caption(
        "Niño-3.4 forecast skill on ERSSTv5. The BSO forecasts the index "
        "directly: a 12-month Niño-3.4 delay state is advanced by a frozen "
        "neural operator, a dictionary is fitted to it by the Galerkin "
        "invariance objective, and per-lead coefficients are regressed on that "
        "dictionary. Direct lagged LIM is the matched linear counterpart — the "
        f"same per-lead regression on 20 EOF coefficients — and {crossover}. "
        "Climatology has undefined anomaly "
        "correlation (a zero-anomaly forecast has zero variance) so it appears "
        "in the error panel alone, drawn as its lead-average. Damped "
        "persistence has exactly the correlation of plain persistence, because "
        "correlation is invariant to the positive per-lead scaling γ^τ; damping "
        "is visible in RMSE only.")


if __name__ == "__main__":
    main()
