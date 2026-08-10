"""Shared increasing-data ("N < M") driver for the BSO forecasting systems.

The scientific question is identical across Lorenz-96, ENSO, and EEG: given a
fixed target operator dimension ``M`` and a growing number of observed training
transition pairs ``N``, how does forecast skill scale, and can each method even
*define* an M-dimensional model when ``N < M``?

Methods, all sharing one dictionary Psi so the comparison is exact:

* **BSO** -- ``edmd_bso_method.run_bso_pipeline`` supplies the dictionary Psi;
  ``run_sweep(bso_source=...)`` chooses where the Galerkin matrix ``K`` comes
  from. Either way ``K`` is ``M x M`` at every ``N``, including ``N < M``,
  because neither route needs more than the ``N`` observed states.

  ``"quadrature"`` (the default, used by ENSO and EEG)
      ``K`` is formed from the operator's action on Psi over the
      data-independent quadrature ``(basis_coords, weights)`` -- the operator is
      evaluated at the basis functions themselves. This is the textbook Galerkin
      projection wherever ``basis_coords`` are *state* coordinates, as for OU and
      the triple well: there the Psi_i genuinely are observables of the state.
  ``"ensemble"`` (used by Lorenz-96)
      ``K`` is the same weighted least squares, fit over states the operator
      generates from the observed ones (:func:`operator_ensemble`). Where
      ``basis_coords`` are *spatial* -- the Lorenz site ring, where the state is
      the whole field -- Psi is a function of position, so probing the operator
      at the basis functions evaluates it far from any state the system visits.
      Fitting over the ensemble instead cuts one-step error by 16-44% and makes
      the construction match ``exp_lorenz96.build_bso_model``, which fits over
      observed states advanced by the frozen operator.
* **Data EDMD (M)** -- the same Psi, transition fit from the ``N`` observed
  pairs projected onto Psi. Needs ``N >= M`` for a full-rank fit, so it is
  reported only where feasible (NaN otherwise).
* **Data EDMD (M = min(N, p))** -- the best a purely data-driven EDMD can do at a
  given ``N``: use the leading ``min(N, M)`` dictionary coordinates (``M`` is the
  dictionary size, so this is capped at ``M``; for Lorenz-96 the field dimension
  ``p`` equals ``M`` so the cap is exactly ``min(N, p)``). Always defined.

Two metrics are reported: **one-step generalization** (relative-L2 at lead 1)
and **valid-prediction-time** (the horizon a multi-step rollout stays below a
relative-L2 threshold). The neural operator is trained once per ``(N, seed)`` by
``run_bso_pipeline``; every method reuses that fit. Only the sweep, the
projected-EDMD least-squares fit, the rollouts/scoring, and the figures are new
here -- everything scientific comes from ``edmd_bso_method``.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import ScalarFormatter

from edmd_bso_method import apply_operator, device, run_bso_pipeline
from figure_style import (GRID_COLOR, LINEWIDTH, MARKER_EDGE_WIDTH, MARKER_SIZE,
                          SINGLE_COLUMN_WIDTH, TEXT_COLOR, add_legend,
                          make_figure, publication_style,
                          save_publication_pair, style_axis)


# Every property of every curve, stated once. BSO is the method under study, so
# it is the only "main" (1.7 pt) line; everything else is a 1.4 pt data line.
METHOD_STYLE = {
    "fno": dict(color="#1f77b4", label="FNO", linestyle="-",
                linewidth=LINEWIDTH["data"], marker="o", zorder=3),
    "bno": dict(color="#d62728", label="BSO", linestyle="-",
                linewidth=LINEWIDTH["main"], marker="o", zorder=5),
    "data_edmd": dict(color="#2ca02c", label="Data EDMD (M={M})",
                      linestyle="-", linewidth=LINEWIDTH["data"], marker="o",
                      zorder=4),
    "data_edmd_npairs": dict(color="#9467bd",
                             label="Data EDMD (M=min(N,p))", linestyle="-",
                             linewidth=LINEWIDTH["data"], marker="o",
                             zorder=4),
    # forced full-dimensional fits, drawn only where N < M (see plot helpers)
    "data_edmd_forced_ridge": dict(color="#8c564b",
                                   label="Data EDMD (M={M}, ridge)",
                                   linestyle="--",
                                   linewidth=LINEWIDTH["data"], marker="o",
                                   zorder=2),
    "data_edmd_forced_pinv": dict(color="#e377c2",
                                  label="Data EDMD (M={M}, min-norm)",
                                  linestyle=":",
                                  linewidth=LINEWIDTH["data"], marker="o",
                                  zorder=2),
}
FORCED_METHODS = ("data_edmd_forced_ridge", "data_edmd_forced_pinv")
VALID_TIME_THRESHOLD = 0.5

# One 1x1 single-column layout shared by all three data-length figures, in
# inches. The 1.22 in bottom holds the five-row legend (~0.76 in), the x tick
# labels and the axis label, plus a ~0.12 in breathing gap between the axis
# label and the first legend row; 0.66 in on the left holds a two-line rotated
# y label. These margins are what to adjust if an element collides -- never the
# font, line, marker, tick or legend sizes, which are physical constants of the
# paper.
LAYOUT_DATA_LENGTH = {
    "figsize": (SINGLE_COLUMN_WIDTH, 3.4), "nrows": 1, "ncols": 1,
    "left": 0.66, "right": 0.12, "bottom": 1.22, "top": 0.28,
    "wgap": 0.0, "hgap": 0.0,
    "legend_loc": "lower center", "legend_bbox": (0.5, 0.005),
    "legend_ncol": 1,
}


@dataclass
class DataLengthResult:
    lengths: np.ndarray                 # (n_length,) number of training pairs N
    seeds: np.ndarray                   # (n_seed,)
    one_step: dict[str, np.ndarray]     # method -> (n_seed, n_length) rel-L2
    valid_time: dict[str, np.ndarray]   # method -> (n_seed, n_length) horizon
    target_M: int
    step_time: float                    # physical time per forecast step (T)
    gram_rank: np.ndarray | None = None   # (n_seed, n_length) rank of G at M
    gram_cond: np.ndarray | None = None   # (n_seed, n_length) cond(G) at M


# ---------------------------------------------------------------------------
# Projected data-EDMD sharing the BSO dictionary
# ---------------------------------------------------------------------------

def _coordinates(psi: np.ndarray, weights: np.ndarray,
                 fields: np.ndarray) -> np.ndarray:
    """Weighted projection of fields (n, p) onto Psi (m, p): returns (m, n)."""
    return psi @ (weights[:, None] * np.asarray(fields, dtype=np.float64).T)


def gram_diagnostics(psi: np.ndarray, weights: np.ndarray, u0: np.ndarray,
                     dim: int) -> tuple[int, float]:
    """Numerical rank and condition number of the empirical Gram matrix.

    These are what make "rank deficient" a measured statement rather than an
    assertion: with ``n_s`` observed transitions the Gram of ``dim`` dictionary
    coordinates has rank at most ``min(n_s, dim)``.
    """
    c0 = _coordinates(psi[:dim], weights, u0)
    gram = c0 @ c0.T
    return int(np.linalg.matrix_rank(gram)), float(np.linalg.cond(gram))


def fit_projected_edmd(psi: np.ndarray, weights: np.ndarray, u0: np.ndarray,
                       uT: np.ndarray, dim: int, reg_alpha: float,
                       solver: str = "ridge") -> np.ndarray:
    """Least-squares EDMD in the leading ``dim`` dictionary coordinates.

    Uses the same weighted-Galerkin algebra as ``_final_spectral_analysis``
    (``K = (G + aI)^{-1} A``) but with the *observed* pairs instead of the
    operator image, so ``U Psi ~= K^T Psi`` on the data.

    ``solver`` selects how the normal equations are resolved:

    ``"ridge"``
        the default, ``(G + lam I)^{-1} A`` with ``lam`` scaled by
        ``trace(G)/dim``. Always well posed, so it returns a matrix even when
        ``G`` is rank deficient -- the regularizer, not the data, supplies the
        missing directions.
    ``"pinv"``
        the minimum-norm solution ``pinv(G) A`` with no regularization. When
        ``n_s < dim`` the Gram is singular and this is the honest
        rank-deficient least-squares answer.
    """
    if solver not in ("ridge", "pinv"):
        raise ValueError(f"solver must be 'ridge' or 'pinv', got {solver!r}")
    c0 = _coordinates(psi[:dim], weights, u0)     # (dim, N)
    c1 = _coordinates(psi[:dim], weights, uT)
    gram = c0 @ c0.T
    cross = c0 @ c1.T
    if solver == "pinv":
        transition = np.linalg.pinv(gram) @ cross
    else:
        reg = float(reg_alpha) * np.trace(gram) / max(gram.shape[0], 1)
        transition = np.linalg.solve(gram + reg * np.eye(gram.shape[0]), cross)
    transition[:, 0] = 0.0                         # pin the constant mode
    transition[0, 0] = 1.0
    return transition


def _one_step(psi: np.ndarray, transition: np.ndarray, weights: np.ndarray,
              test_u0: np.ndarray, test_u1: np.ndarray) -> float:
    """Weighted relative-L2 of a single dictionary step on held-out pairs."""
    dim = transition.shape[0]
    c0 = _coordinates(psi[:dim], weights, test_u0)        # (dim, n_test)
    c1 = transition.T @ c0
    prediction = (psi[:dim].T @ c1).T                     # (n_test, p)
    num = np.sqrt(np.sum(weights[None] * (prediction - test_u1) ** 2, axis=1))
    den = np.sqrt(np.sum(weights[None] * test_u1 ** 2, axis=1)) + 1e-12
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# Multi-step rollouts and metrics
# ---------------------------------------------------------------------------

def dictionary_rollout(psi: np.ndarray, transition: np.ndarray,
                       weights: np.ndarray, u0: np.ndarray,
                       horizon: int) -> np.ndarray:
    """Roll ``c <- K^T c`` and decode: returns (horizon, n, p)."""
    dim = transition.shape[0]
    c = _coordinates(psi[:dim], weights, u0)              # (dim, n)
    out = []
    for _ in range(int(horizon)):
        c = transition.T @ c
        out.append((psi[:dim].T @ c).T)                   # (n, p)
    return np.stack(out)


def fno_rollout(result, kind: str, u0: np.ndarray, horizon: int) -> np.ndarray:
    """Autoregressive rollout of the frozen operator: returns (horizon, n, p)."""
    grid = torch.as_tensor(result.grid, dtype=torch.float32, device=device)
    u = torch.as_tensor(np.asarray(u0, dtype=np.float64),
                        dtype=torch.float32, device=device)
    result.operator_pure.eval()
    out = []
    with torch.no_grad():
        for _ in range(int(horizon)):
            u = apply_operator(result.operator_pure, u, grid, kind)
            out.append(u.detach().cpu().numpy().astype(np.float64))
    return np.stack(out)


def operator_ensemble(result, kind: str, seed_states: np.ndarray,
                      weights: np.ndarray, *, seed: int,
                      target_seeds: int = 256, sigma: float = 0.1,
                      depth: int = 2,
                      rms_bounds: tuple[float, float] = (0.5, 1.5)
                      ) -> np.ndarray:
    """States on which to probe the operator: the observed ones, spread out.

    The Galerkin fit needs states, and at small ``N`` there are not enough of
    them. Each default here was measured rather than guessed, on Lorenz-96 with
    ``M = 40``:

    ``target_seeds``/``sigma``
        the ``N`` observed states are padded with Gaussian perturbations until
        there are at least ``target_seeds``. Without this, ``N = 10`` leaves far
        fewer states than the 40 dictionary coordinates and the fit is
        under-determined.
    ``depth``
        only two operator steps. Rolling further is how
        ``exp_lorenz96_data_length_matched._fno_ensemble`` reaches its target
        count (``ceil(512/N)``, so 52 steps at ``N = 10``), but an operator
        trained on 10 pairs is unstable under autoregression: the ensemble's
        weighted RMS climbs from 1.04 to 11.16 over those steps and the
        resulting fit is worse than no ensemble at all. Two steps hold it at
        1.0.
    ``rms_bounds``
        drop diverged states. The attractor's weighted RMS is 1.000, so states
        outside [0.5, 1.5] are rollout artefacts, not dynamics.

    ``seed`` makes the perturbations reproducible per ``(N, seed)`` cell.
    """
    seed_states = np.asarray(seed_states, dtype=np.float64)
    if seed_states.ndim != 2:
        raise ValueError(f"seed_states must be (n, p), got {seed_states.shape}")
    rng = np.random.default_rng(int(seed))
    repeats = max(1, math.ceil(int(target_seeds) / max(len(seed_states), 1)))
    padded = np.concatenate(
        [seed_states]
        + [seed_states + float(sigma) * rng.standard_normal(seed_states.shape)
           for _ in range(repeats)])
    states = np.concatenate(
        [padded, *fno_rollout(result, kind, padded, int(depth))])
    states = states[np.isfinite(states).all(axis=1)]
    rms = np.sqrt(np.sum(np.asarray(weights)[None] * states ** 2, axis=1))
    low, high = rms_bounds
    return states[(rms > float(low)) & (rms < float(high))]


def _rel_l2_by_lead(pred: np.ndarray, truth: np.ndarray,
                    weights: np.ndarray) -> np.ndarray:
    """Weighted relative-L2 per (lead, origin) from (horizon, n, p) arrays."""
    w = weights[None, None]
    num = np.sqrt(np.sum(w * (pred - truth) ** 2, axis=2))
    den = np.sqrt(np.sum(w * truth ** 2, axis=2)) + 1e-12
    return num / den                                       # (horizon, n)


def _valid_time(rel_by_lead: np.ndarray, step_time: float,
                threshold: float) -> np.ndarray:
    """Per-origin lead time (physical units) until relative-L2 first exceeds."""
    horizon, n = rel_by_lead.shape
    exceed = ~np.isfinite(rel_by_lead) | (rel_by_lead > float(threshold))
    out = np.full(n, horizon * float(step_time))
    for j in range(n):
        idx = np.where(exceed[:, j])[0]
        if len(idx):
            out[j] = float(idx[0]) * float(step_time)
    return out


def _score(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray,
           step_time: float, threshold: float) -> tuple[float, float]:
    rel = _rel_l2_by_lead(pred, truth, weights)
    one_step = float(np.mean(rel[0]))
    valid = float(np.median(_valid_time(rel, step_time, threshold)))
    return one_step, valid


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    *,
    system: str,
    make_dataset_n: Callable[[int, int], dict[str, Any]],
    test_traj: np.ndarray,
    weights: np.ndarray,
    kind: str,
    cfg_builder: Callable[[int, int], dict[str, Any]],
    target_M: int,
    lengths: tuple[int, ...],
    seeds: tuple[int, ...],
    checkpoint_root: Path | None,
    reg_alpha: float,
    threshold: float = VALID_TIME_THRESHOLD,
    bso_source: str = "quadrature",
) -> DataLengthResult:
    """Train BSO and score all methods over the (length, seed) grid.

    ``bso_source`` selects where the BSO's Galerkin matrix comes from; see the
    module docstring. ``"quadrature"`` keeps ``run_bso_pipeline``'s own ``K``,
    which is what the ENSO and EEG drivers use. ``"ensemble"`` refits it over
    operator-generated states, which is correct where ``basis_coords`` are
    spatial rather than state coordinates.

    ``make_dataset_n(N, seed)`` returns a dataset dict whose ``u0/uT/u2T`` are
    the first ``N`` training pairs (the measure ``basis_coords/weights`` are
    held fixed). ``cfg_builder(N, seed)`` returns the ``run_bso_pipeline`` config
    (with ``n_basis == target_M``). ``test_traj`` has shape ``(n_origins, H+1,
    p)``: consecutive held-out states from each forecast origin. One-step skill
    uses leads 0->1; valid-prediction-time uses the full horizon ``H``.
    """
    if bso_source not in ("quadrature", "ensemble"):
        raise ValueError("bso_source must be 'quadrature' or 'ensemble', got "
                         f"{bso_source!r}")
    test_traj = np.asarray(test_traj, dtype=np.float64)
    test_traj = test_traj.reshape(test_traj.shape[0], test_traj.shape[1], -1)
    test_u0 = test_traj[:, 0]
    truth = np.moveaxis(test_traj[:, 1:], 0, 1)            # (H, n, p)
    horizon = truth.shape[0]
    weights = np.asarray(weights, dtype=np.float64)
    methods = ("fno", "bno", "data_edmd", "data_edmd_npairs", *FORCED_METHODS)
    nan = lambda: {m: np.full((len(seeds), len(lengths)), np.nan)
                   for m in methods}
    one_step, valid_time = nan(), nan()
    gram_rank = np.full((len(seeds), len(lengths)), np.nan)
    gram_cond = np.full((len(seeds), len(lengths)), np.nan)
    step_time = 1.0

    for si, seed in enumerate(seeds):
        for li, length in enumerate(lengths):
            dataset_n = make_dataset_n(int(length), int(seed))
            u0_flat = np.asarray(dataset_n["u0"], dtype=np.float64)
            u0_flat = u0_flat.reshape(u0_flat.shape[0], -1)
            uT_flat = np.asarray(dataset_n["uT"], dtype=np.float64)
            uT_flat = uT_flat.reshape(uT_flat.shape[0], -1)
            n_pairs = u0_flat.shape[0]
            cfg = cfg_builder(int(length), int(seed))
            if int(cfg["n_basis"]) != int(target_M):
                raise ValueError("cfg_builder must set n_basis == target_M")
            ckpt = (None if checkpoint_root is None
                    else checkpoint_root / f"n{length:05d}_seed{seed}")
            result = run_bso_pipeline(dataset_n, cfg, checkpoint_dir=ckpt)
            psi = np.asarray(result.psi, dtype=np.float64)
            step_time = float(result.T)

            if bso_source == "quadrature":
                k_bso = np.asarray(result.K, dtype=np.float64)
            else:
                # Same dictionary, same least squares -- only the states the
                # operator is probed at change, from the basis functions to
                # points the system actually visits.
                ens = operator_ensemble(result, kind, u0_flat, weights,
                                        seed=int(seed))
                k_bso = fit_projected_edmd(
                    psi, weights, ens, fno_rollout(result, kind, ens, 1)[0],
                    target_M, reg_alpha)
            rollouts = {
                "fno": fno_rollout(result, kind, test_u0, horizon),
                "bno": dictionary_rollout(psi, k_bso, weights, test_u0,
                                          horizon),
            }
            if n_pairs >= target_M:
                k_data = fit_projected_edmd(
                    psi, weights, u0_flat, uT_flat, target_M, reg_alpha)
                rollouts["data_edmd"] = dictionary_rollout(
                    psi, k_data, weights, test_u0, horizon)
            # Forced full-dimensional fits: no feasibility gate, so they also
            # cover N < M where the Gram is rank deficient. Recording the rank
            # makes that deficiency measured rather than asserted.
            gram_rank[si, li], gram_cond[si, li] = gram_diagnostics(
                psi, weights, u0_flat, target_M)
            for method, solver in zip(FORCED_METHODS, ("ridge", "pinv")):
                k_forced = fit_projected_edmd(
                    psi, weights, u0_flat, uT_flat, target_M, reg_alpha,
                    solver=solver)
                rollouts[method] = dictionary_rollout(
                    psi, k_forced, weights, test_u0, horizon)
            dim_np = int(min(n_pairs, target_M))
            if dim_np >= 2:
                k_np = fit_projected_edmd(
                    psi, weights, u0_flat, uT_flat, dim_np, reg_alpha)
                rollouts["data_edmd_npairs"] = dictionary_rollout(
                    psi[:dim_np], k_np, weights, test_u0, horizon)

            for method, pred in rollouts.items():
                os_val, vt_val = _score(pred, truth, weights, step_time,
                                        threshold)
                one_step[method][si, li] = os_val
                valid_time[method][si, li] = vt_val
            print(f"[{system}:data-length] N={length} seed={seed} "
                  f"pairs={n_pairs} M={target_M} "
                  f"rank(G)={int(gram_rank[si, li])}: "
                  f"one-step fno={one_step['fno'][si, li]:.4f} "
                  f"bso={one_step['bno'][si, li]:.4f} "
                  f"data@min(N,p)={one_step['data_edmd_npairs'][si, li]:.4f} "
                  f"forced[ridge]="
                  f"{one_step['data_edmd_forced_ridge'][si, li]:.4f} "
                  f"forced[pinv]="
                  f"{one_step['data_edmd_forced_pinv'][si, li]:.4f} | "
                  f"valid-time bso={valid_time['bno'][si, li]:.3g} "
                  f"fno={valid_time['fno'][si, li]:.3g}", flush=True)

    return DataLengthResult(np.asarray(lengths), np.asarray(seeds), one_step,
                            valid_time, int(target_M), step_time,
                            gram_rank=gram_rank, gram_cond=gram_cond)


# ---------------------------------------------------------------------------
# Figures and persistence
# ---------------------------------------------------------------------------

def _curve(ax, lengths, per_seed, method, target_M, mask=None, *, bands=True):
    """Median (+IQR) curve for one method; ``mask`` restricts the N range.

    ``bands=False`` drops the 25-75 percentile shading and draws medians only,
    for the variant of each figure that stays legible when several curves
    overlap. The medians are computed identically either way.
    """
    style = METHOD_STYLE[method]
    label = style["label"].format(M=target_M)
    lengths = np.asarray(lengths)
    values = np.asarray(per_seed, dtype=np.float64)
    if mask is not None:
        lengths, values = lengths[mask], values[:, mask]
        if not len(lengths):
            return
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        if np.all(np.isnan(median)):
            return
        ax.plot(lengths, median, label=label, color=style["color"],
                linestyle=style["linestyle"], linewidth=style["linewidth"],
                marker=style["marker"], markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGE_WIDTH, alpha=1.0,
                zorder=style["zorder"])
        if bands and values.shape[0] > 1:
            finite = np.sum(np.isfinite(values), axis=0) > 1
            if finite.any():
                lo, hi = np.nanpercentile(values, [25, 75], axis=0)
                ax.fill_between(lengths, lo, hi, where=finite,
                                color=style["color"], alpha=0.15,
                                linewidth=0.0, zorder=style["zorder"] - 1)


def _forced_curves(ax, result: DataLengthResult, *, bands: bool = True) -> None:
    """Draw the forced full-dimensional EDMD fits, only where ``N < M``.

    Above ``M`` the ridge-forced fit is the *same* computation as
    ``data_edmd``, so plotting it there would hide one curve beneath another;
    below ``M`` is the rank-deficient regime these curves exist to expose.
    """
    mask = np.asarray(result.lengths) < result.target_M
    if not mask.any():
        return
    for method in FORCED_METHODS:
        _curve(ax, result.lengths, result.one_step[method], method,
               result.target_M, mask=mask, bands=bands)


def _mark_M(ax, target_M):
    ax.axvline(target_M, color=GRID_COLOR, linestyle="--",
               linewidth=LINEWIDTH["reference"], zorder=1)
    ax.annotate(f"M={target_M}", (target_M, 0.98),
                xycoords=("data", "axes fraction"), ha="center", va="top",
                fontsize=8, color=TEXT_COLOR, zorder=6)


def _finish_axis(ax, lengths, *, ylabel, title):
    """The log-N axis every data-length figure shares, stated explicitly.

    The log locator would label only the decades (10, 100) and render them as
    powers; ``ScalarFormatter`` keeps them as plain integers, which reads better
    against an "N" axis whose sampled values are 10..400.
    """
    ax.set_xscale("log")
    ax.set_xlim(float(np.min(lengths)) * 0.85, float(np.max(lengths)) * 1.18)
    ax.set_xticks([10, 100])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    style_axis(ax, xlabel="Number of training pairs N", ylabel=ylabel,
               title=title, grid=True)


def _band_tag(bands: bool) -> str:
    """Filename marker keeping the median-only variant beside the banded one."""
    return "" if bands else "_nobands"


def plot_two_versions(result: DataLengthResult, figure_dir: Path, system: str,
                      title_suffix: str = "", *,
                      bands: bool = True) -> tuple[Path, ...]:
    """One-step skill vs. N, in two variants.

    The variant qualifier ("typical comparisons" / "best-feasible data EDMD,
    M=min(N,p)") belongs in the LaTeX caption rather than on the figure, so the
    panel title stays a single 9 pt line at single-column width.

    ``bands=False`` writes a ``_nobands`` companion showing medians only.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" {title_suffix}".rstrip()
    slug = re.sub(r"[^0-9a-z]+", "_", system.lower()).strip("_")
    outputs = []
    for tag, methods in (
        ("", ("fno", "bno", "data_edmd")),                  # typical comparisons
        ("_npairs", ("fno", "bno", "data_edmd_npairs")),    # M=min(N,p)
    ):
        with publication_style():
            fig, axes = make_figure(LAYOUT_DATA_LENGTH)
            ax = axes[0, 0]
            for method in methods:
                _curve(ax, result.lengths, result.one_step[method], method,
                       result.target_M, bands=bands)
            _forced_curves(ax, result, bands=bands)
            _mark_M(ax, result.target_M)
            _finish_axis(ax, result.lengths,
                         ylabel="One-step generalization\n(relative $L^2$)",
                         title=f"{system}{suffix}")
            add_legend(fig, *ax.get_legend_handles_labels(),
                       loc=LAYOUT_DATA_LENGTH["legend_loc"],
                       bbox_to_anchor=LAYOUT_DATA_LENGTH["legend_bbox"],
                       ncol=LAYOUT_DATA_LENGTH["legend_ncol"])
            pdf, _ = save_publication_pair(
                fig, figure_dir / f"{slug}_one_step_vs_pairs{tag}"
                                  f"{_band_tag(bands)}.pdf")
            plt.close(fig)
        outputs.append(pdf)
    return tuple(outputs)


def plot_valid_time(result: DataLengthResult, figure_dir: Path, system: str,
                    title_suffix: str = "", *, bands: bool = True) -> Path:
    """Valid-prediction-time vs. N -- where BSO's advantage over FNO shows."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" {title_suffix}".rstrip()
    slug = re.sub(r"[^0-9a-z]+", "_", system.lower()).strip("_")
    with publication_style():
        fig, axes = make_figure(LAYOUT_DATA_LENGTH)
        ax = axes[0, 0]
        for method in ("fno", "bno", "data_edmd_npairs"):
            _curve(ax, result.lengths, result.valid_time[method], method,
                   result.target_M, bands=bands)
        mask = np.asarray(result.lengths) < result.target_M
        if mask.any():
            for method in FORCED_METHODS:
                _curve(ax, result.lengths, result.valid_time[method], method,
                       result.target_M, mask=mask, bands=bands)
        _mark_M(ax, result.target_M)
        _finish_axis(ax, result.lengths,
                     ylabel="Valid-prediction-time\n"
                            f"(rel. $L^2$ < {VALID_TIME_THRESHOLD})",
                     title=f"{system}{suffix}")
        add_legend(fig, *ax.get_legend_handles_labels(),
                   loc=LAYOUT_DATA_LENGTH["legend_loc"],
                   bbox_to_anchor=LAYOUT_DATA_LENGTH["legend_bbox"],
                   ncol=LAYOUT_DATA_LENGTH["legend_ncol"])
        pdf, _ = save_publication_pair(
            fig, figure_dir / f"{slug}_valid_time_vs_pairs"
                              f"{_band_tag(bands)}.pdf")
        plt.close(fig)
    return pdf


def save_result(result: DataLengthResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"one_step_{m}": v for m, v in result.one_step.items()}
    arrays.update({f"valid_time_{m}": v for m, v in result.valid_time.items()})
    if result.gram_rank is not None:
        arrays["gram_rank"] = result.gram_rank
        arrays["gram_cond"] = result.gram_cond
    np.savez(path, lengths=result.lengths, seeds=result.seeds,
             target_M=result.target_M, step_time=result.step_time, **arrays)


def load_result(path: Path) -> DataLengthResult:
    """Inverse of :func:`save_result`, so figures can be restyled without a rerun.

    Replotting from the stored arrays is the only way to change presentation
    with a guarantee that the numbers did not move: this path cannot retrain.
    """
    with np.load(path) as data:
        one_step = {k[len("one_step_"):]: data[k] for k in data.files
                    if k.startswith("one_step_")}
        valid_time = {k[len("valid_time_"):]: data[k] for k in data.files
                      if k.startswith("valid_time_")}
        return DataLengthResult(
            lengths=data["lengths"], seeds=data["seeds"], one_step=one_step,
            valid_time=valid_time, target_M=int(data["target_M"]),
            step_time=float(data["step_time"]),
            gram_rank=data["gram_rank"] if "gram_rank" in data else None,
            gram_cond=data["gram_cond"] if "gram_cond" in data else None)
