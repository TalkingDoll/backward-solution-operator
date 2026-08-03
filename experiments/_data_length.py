"""Shared increasing-data ("N < M") driver for the BSO forecasting systems.

The scientific question is identical across Lorenz-96, ENSO, and EEG: given a
fixed target operator dimension ``M`` and a growing number of observed training
transition pairs ``N``, how does forecast skill scale, and can each method even
*define* an M-dimensional model when ``N < M``?

Methods, all sharing one dictionary Psi so the comparison is exact:

* **BSO** -- ``edmd_bso_method.run_bso_pipeline``. The Galerkin matrix ``K`` is
  formed from the neural operator's action on Psi over the data-independent
  quadrature ``(basis_coords, weights)``. ``K`` is therefore ``M x M`` at every
  ``N``, including ``N < M``.
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

from edmd_bso_method import apply_operator, device, run_bso_pipeline
from plotting_utils import save_figure_pair


METHOD_STYLE = {
    "fno": ("#1f77b4", "FNO"),
    "bno": ("#d62728", "BSO"),
    "data_edmd": ("#2ca02c", "Data EDMD (M={M})"),
    "data_edmd_npairs": ("#9467bd", "Data EDMD (M=min(N,p))"),
}
VALID_TIME_THRESHOLD = 0.5


@dataclass
class DataLengthResult:
    lengths: np.ndarray                 # (n_length,) number of training pairs N
    seeds: np.ndarray                   # (n_seed,)
    one_step: dict[str, np.ndarray]     # method -> (n_seed, n_length) rel-L2
    valid_time: dict[str, np.ndarray]   # method -> (n_seed, n_length) horizon
    target_M: int
    step_time: float                    # physical time per forecast step (T)


# ---------------------------------------------------------------------------
# Projected data-EDMD sharing the BSO dictionary
# ---------------------------------------------------------------------------

def _coordinates(psi: np.ndarray, weights: np.ndarray,
                 fields: np.ndarray) -> np.ndarray:
    """Weighted projection of fields (n, p) onto Psi (m, p): returns (m, n)."""
    return psi @ (weights[:, None] * np.asarray(fields, dtype=np.float64).T)


def fit_projected_edmd(psi: np.ndarray, weights: np.ndarray, u0: np.ndarray,
                       uT: np.ndarray, dim: int,
                       reg_alpha: float) -> np.ndarray:
    """Least-squares EDMD in the leading ``dim`` dictionary coordinates.

    Uses the same weighted-Galerkin algebra as ``_final_spectral_analysis``
    (``K = (G + aI)^{-1} A``) but with the *observed* pairs instead of the
    operator image, so ``U Psi ~= K^T Psi`` on the data.
    """
    c0 = _coordinates(psi[:dim], weights, u0)     # (dim, N)
    c1 = _coordinates(psi[:dim], weights, uT)
    gram = c0 @ c0.T
    cross = c0 @ c1.T
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
) -> DataLengthResult:
    """Train BSO and score all methods over the (length, seed) grid.

    ``make_dataset_n(N, seed)`` returns a dataset dict whose ``u0/uT/u2T`` are
    the first ``N`` training pairs (the measure ``basis_coords/weights`` are
    held fixed). ``cfg_builder(N, seed)`` returns the ``run_bso_pipeline`` config
    (with ``n_basis == target_M``). ``test_traj`` has shape ``(n_origins, H+1,
    p)``: consecutive held-out states from each forecast origin. One-step skill
    uses leads 0->1; valid-prediction-time uses the full horizon ``H``.
    """
    test_traj = np.asarray(test_traj, dtype=np.float64)
    test_traj = test_traj.reshape(test_traj.shape[0], test_traj.shape[1], -1)
    test_u0 = test_traj[:, 0]
    truth = np.moveaxis(test_traj[:, 1:], 0, 1)            # (H, n, p)
    horizon = truth.shape[0]
    weights = np.asarray(weights, dtype=np.float64)
    methods = ("fno", "bno", "data_edmd", "data_edmd_npairs")
    nan = lambda: {m: np.full((len(seeds), len(lengths)), np.nan)
                   for m in methods}
    one_step, valid_time = nan(), nan()
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

            rollouts = {
                "fno": fno_rollout(result, kind, test_u0, horizon),
                "bno": dictionary_rollout(
                    psi, np.asarray(result.K, dtype=np.float64), weights,
                    test_u0, horizon),
            }
            if n_pairs >= target_M:
                k_data = fit_projected_edmd(
                    psi, weights, u0_flat, uT_flat, target_M, reg_alpha)
                rollouts["data_edmd"] = dictionary_rollout(
                    psi, k_data, weights, test_u0, horizon)
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
                  f"pairs={n_pairs} M={target_M}: "
                  f"one-step fno={one_step['fno'][si, li]:.4f} "
                  f"bso={one_step['bno'][si, li]:.4f} "
                  f"data@min(N,p)={one_step['data_edmd_npairs'][si, li]:.4f} | "
                  f"valid-time bso={valid_time['bno'][si, li]:.3g} "
                  f"fno={valid_time['fno'][si, li]:.3g}", flush=True)

    return DataLengthResult(np.asarray(lengths), np.asarray(seeds), one_step,
                            valid_time, int(target_M), step_time)


# ---------------------------------------------------------------------------
# Figures and persistence
# ---------------------------------------------------------------------------

def _curve(ax, lengths, per_seed, method, target_M):
    color, label = METHOD_STYLE[method]
    label = label.format(M=target_M)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(per_seed, axis=0)
        if np.all(np.isnan(median)):
            return
        ax.plot(lengths, median, color=color, marker="o", lw=2, label=label)
        if per_seed.shape[0] > 1:
            finite = np.sum(np.isfinite(per_seed), axis=0) > 1
            if finite.any():
                lo, hi = np.nanpercentile(per_seed, [25, 75], axis=0)
                ax.fill_between(lengths, lo, hi, color=color, alpha=0.15,
                                where=finite)


def _mark_M(ax, target_M):
    ax.axvline(target_M, color="0.5", ls="--", lw=1)
    ax.annotate(f"M={target_M}", (target_M, 0.98),
                xycoords=("data", "axes fraction"), ha="center", va="top",
                fontsize=9, color="0.4")


def plot_two_versions(result: DataLengthResult, figure_dir: Path, system: str,
                      title_suffix: str = "") -> tuple[Path, ...]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" {title_suffix}".rstrip()
    slug = re.sub(r"[^0-9a-z]+", "_", system.lower()).strip("_")
    outputs = []
    for tag, methods, version in (
        ("", ("fno", "bno", "data_edmd"), "typical comparisons"),
        ("_npairs", ("fno", "bno", "data_edmd_npairs"),
         "best-feasible data EDMD, M=min(N,p)"),
    ):
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for method in methods:
            _curve(ax, result.lengths, result.one_step[method], method,
                   result.target_M)
        _mark_M(ax, result.target_M)
        ax.set(xscale="log", xlabel="Number of training pairs N",
               ylabel="One-step generalization (relative $L^2$)",
               title=f"{system}: one-step vs. training pairs{suffix}\n"
                     f"({version})")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        pdf, _ = save_figure_pair(
            fig, figure_dir / f"{slug}_one_step_vs_pairs{tag}.pdf")
        plt.close(fig)
        outputs.append(pdf)
    return tuple(outputs)


def plot_valid_time(result: DataLengthResult, figure_dir: Path, system: str,
                    title_suffix: str = "") -> Path:
    """Valid-prediction-time vs. N -- where BSO's advantage over FNO shows."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    suffix = f" {title_suffix}".rstrip()
    slug = re.sub(r"[^0-9a-z]+", "_", system.lower()).strip("_")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for method in ("fno", "bno", "data_edmd_npairs"):
        _curve(ax, result.lengths, result.valid_time[method], method,
               result.target_M)
    _mark_M(ax, result.target_M)
    ax.set(xscale="log", xlabel="Number of training pairs N",
           ylabel=f"Valid-prediction-time (rel. $L^2$ < {VALID_TIME_THRESHOLD})",
           title=f"{system}: valid-prediction-time vs. training pairs{suffix}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    pdf, _ = save_figure_pair(fig, figure_dir / f"{slug}_valid_time_vs_pairs.pdf")
    plt.close(fig)
    return pdf


def save_result(result: DataLengthResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"one_step_{m}": v for m, v in result.one_step.items()}
    arrays.update({f"valid_time_{m}": v for m, v in result.valid_time.items()})
    np.savez(path, lengths=result.lengths, seeds=result.seeds,
             target_M=result.target_M, step_time=result.step_time, **arrays)
