"""Lorenz-96 increasing-data study, matched to the original l96_prediction BSO.

Unlike ``exp_lorenz96_data_length`` (a linear Galerkin scored one-step), this
variant reproduces the construction of the finalized ``exp_lorenz96`` /
``l96_prediction.pdf``: the standard ``K=20`` nonlinear-sensor family, a fixed
**polynomial dictionary** (standardized curve + quadratic monomials of its
leading 4 PCA coordinates, ``M=30`` features), a **ridge decoder**, and
**eigenvalue stabilization** (``|mu| <= 0.99``), scored with the **multi-step**
weighted relative-L2 metric.

To keep the ``N < M`` data-efficiency story, the matched **BSO** fits its PCA,
decoder, and Galerkin map over an **FNO-generated ensemble** (seeded from the N
observed states and rolled forward), so the M-dimensional operator is defined at
every N. The **Data EDMD** counterpart fits the *same* polynomial dictionary on
the N *observed* states, so it is defined only for ``N >= M``.

Everything scientific is reused from ``experiments.lorenz96.exp_lorenz96`` and
``edmd_bso_method``; only the sweep, the ensemble, and the figures are new.

Usage:
    python -m experiments.lorenz96.exp_lorenz96_data_length_matched [--quick]
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

sys.dont_write_bytecode = True

from edmd_bso_method import apply_operator, device, run_bso_pipeline
from experiments._data_length import FORCED_METHODS, METHOD_STYLE, _curve
from experiments.lorenz96.exp_lorenz96 import (
    DECODER_RIDGE_RELATIVE, FAMILY, FNO_KWARGS, N_ORIGINS, N_STEPS,
    PCA_COORDINATES, POLYNOMIAL_DEGREE, STABILITY_RADIUS,
    TRANSITION_RIDGE_RELATIVE, ForecastContext, _lift_dictionary,
    _polynomial_exponents, _raw_dictionary, _relative_errors, _ridge_map,
    _stabilize_transition, batch_operator_rollout, bso_rollout)
from kbe_observation_data import generate_kbe_dataset
from plotting_utils import save_figure_pair
from project_paths import FIGURES_DIR, RESULTS_DIR

SYSTEM = "lorenz96_matched"
K_SITES = 20
FORCING = 8.0
ENSEMBLE_TARGET = 512
DICT_SIZE = K_SITES + len(_polynomial_exponents(PCA_COORDINATES,
                                                POLYNOMIAL_DEGREE))   # 30
VALID_TIME_THRESHOLD = 0.5

LENGTHS = (10, 20, 30, 50, 100, 200, 400)
SEEDS = (2026, 2027, 2028, 2029, 2030)
QUICK_LENGTHS = (20, 50)
QUICK_SEEDS = (2026,)


def _fno_model_kwargs(quick: bool) -> dict:
    return dict(modes=8, width=32, n_layers=2) if quick else dict(FNO_KWARGS)


def _train_kwargs(quick: bool) -> dict:
    return dict(epochs=20 if quick else 400, lr=6e-4, bs=24,
                weight_decay=7.5e-5, sg_weight=0.25, val_fraction=0.15,
                print_every=0)


def build_full_dataset(quick: bool, seed: int = 2026) -> dict:
    long_traj = 6_000 if quick else 30_000
    cache = None if quick else str(RESULTS_DIR / "l96_data_length_matched"
                                   / "dataset_k20.pkl")
    return generate_kbe_dataset(SYSTEM.split("_")[0], n_samples=0, seed=seed,
                                cache_path=cache, K=K_SITES, F=FORCING,
                                long_traj_steps=long_traj)


# ---------------------------------------------------------------------------
# FNO ensemble and matched polynomial BSO (reusing the original helpers)
# ---------------------------------------------------------------------------

def _fno_ensemble(operator, grid_np: np.ndarray, kind: str,
                  seed_states: np.ndarray,
                  target: int = ENSEMBLE_TARGET) -> np.ndarray:
    """Data-independent measure: FNO rollouts from the observed seed states.

    Consecutive autoregressive states are collected; non-finite states are
    dropped. Returns ``(<=(steps+1)*n_seed, K)`` finite states.
    """
    seed_states = np.asarray(seed_states, dtype=np.float64)
    steps = max(1, math.ceil(int(target) / max(len(seed_states), 1)))
    grid = torch.as_tensor(grid_np, dtype=torch.float32, device=device)
    states = [seed_states]
    operator.eval()
    with torch.no_grad():
        for _ in range(steps):
            current = torch.as_tensor(states[-1], dtype=torch.float32,
                                      device=device)
            nxt = apply_operator(operator, current, grid, kind)
            states.append(nxt.cpu().numpy().astype(np.float64))
    stacked = np.concatenate(states, axis=0)
    finite = np.isfinite(stacked).all(axis=1)
    return stacked[finite]


def build_matched_bso(operator, grid_np: np.ndarray, kind: str,
                      sampled: np.ndarray,
                      solver: str = "ridge") -> dict | None:
    """Original polynomial-dictionary Galerkin BSO fit over ``sampled`` states.

    Mirrors ``exp_lorenz96.build_bso_model`` but takes the fitting states
    directly (an FNO ensemble or the observed states). Returns ``None`` if the
    stabilized transition is too ill-conditioned to trust.

    ``solver`` selects how the transition is resolved: ``"ridge"`` is the
    original relative-ridge map, ``"pinv"`` is the minimum-norm least-squares
    solution with no regularization -- the honest answer when ``sampled``
    carries fewer states than the dictionary has columns. The decoder always
    uses the original ridge.
    """
    if solver not in ("ridge", "pinv"):
        raise ValueError(f"solver must be 'ridge' or 'pinv', got {solver!r}")
    sampled = np.asarray(sampled, dtype=np.float64)
    grid = torch.as_tensor(grid_np, dtype=torch.float32, device=device)
    operator.eval()
    advanced_chunks = []
    with torch.no_grad():
        for start in range(0, len(sampled), 4096):
            fields = torch.as_tensor(sampled[start:start + 4096],
                                     dtype=torch.float32, device=device)
            advanced_chunks.append(
                apply_operator(operator, fields, grid, kind)
                .cpu().numpy().astype(np.float64))
    advanced = np.concatenate(advanced_chunks)

    field_mean = sampled.mean(axis=0)
    centered = sampled - field_mean
    _, vectors = np.linalg.eigh(centered.T @ centered)
    pca = vectors[:, ::-1][:, :PCA_COORDINATES]
    model = {
        "field_mean": field_mean,
        "field_scale": centered.std(axis=0) + 1e-12,
        "pca": pca,
        "score_scale": (centered @ pca).std(axis=0) + 1e-12,
        "exponents": _polynomial_exponents(PCA_COORDINATES, POLYNOMIAL_DEGREE),
    }
    raw = _raw_dictionary(sampled, model)
    model["feature_mean"] = raw.mean(axis=0)
    model["feature_scale"] = raw.std(axis=0) + 1e-12
    source = _lift_dictionary(sampled, model)
    target = _lift_dictionary(advanced, model)
    model["decoder"] = _ridge_map(source, sampled - field_mean,
                                  DECODER_RIDGE_RELATIVE)
    if solver == "pinv":
        transition_row = np.linalg.pinv(source) @ target
    else:
        transition_row = _ridge_map(source, target, TRANSITION_RIDGE_RELATIVE)
    try:
        transition_row, raw_radius, condition = _stabilize_transition(
            transition_row, STABILITY_RADIUS)
    except np.linalg.LinAlgError:
        return None
    model["A"] = transition_row.T
    model["raw_spectral_radius"] = np.asarray(raw_radius)
    model["dictionary_rank"] = np.asarray(np.linalg.matrix_rank(source))
    return model


# ---------------------------------------------------------------------------
# Multi-step metrics (matched to the original pooled relative-L2)
# ---------------------------------------------------------------------------

def _origins(readings: np.ndarray, horizon: int, n_origins: int) -> np.ndarray:
    last = readings.shape[1] - int(horizon) - 2
    if last < 0:
        raise ValueError("held-out window too short for the horizon")
    return np.unique(np.linspace(0, last, int(n_origins), dtype=int))


def _truth(readings: np.ndarray, origins: np.ndarray,
           horizon: int) -> np.ndarray:
    return np.stack([readings[:, origins + s].T
                     for s in range(int(horizon) + 1)])


def _bands(times: np.ndarray) -> dict[str, np.ndarray]:
    return {"short": times <= 0.3 + 1e-12,
            "mid": (times > 0.3) & (times <= 1.0 + 1e-12),
            "long": times > 1.0}


def _valid_time(errors: np.ndarray, times: np.ndarray,
                threshold: float) -> float:
    over = np.where(~np.isfinite(errors) | (errors > float(threshold)))[0]
    return float(times[over[0]]) if len(over) else float(times[-1])


# ---------------------------------------------------------------------------
# Sweep and figures
# ---------------------------------------------------------------------------

def run_sweep(quick: bool) -> dict:
    lengths = QUICK_LENGTHS if quick else LENGTHS
    seeds = QUICK_SEEDS if quick else SEEDS
    horizon = 20 if quick else N_STEPS
    n_origins = 8 if quick else N_ORIGINS
    dataset = build_full_dataset(quick)
    tau = float(dataset["T"])
    weights = np.asarray(dataset["weights"], dtype=np.float64)
    train = np.asarray(dataset["meta"]["train_readings"][FAMILY],
                       dtype=np.float64)
    test = np.asarray(dataset["meta"]["test_readings"][FAMILY],
                      dtype=np.float64)
    times = tau * np.arange(horizon + 1)
    bands = _bands(times)
    origins = _origins(test, horizon, n_origins)
    truth = _truth(test, origins, horizon)
    ckpt_root = (None if quick
                 else RESULTS_DIR / "l96_data_length_matched" / "checkpoints")

    methods = ("fno", "bno", "data_edmd", *FORCED_METHODS)
    band_err = {b: {m: np.full((len(seeds), len(lengths)), np.nan)
                    for m in methods} for b in bands}
    valid = {m: np.full((len(seeds), len(lengths)), np.nan) for m in methods}
    dict_rank = np.full((len(seeds), len(lengths)), np.nan)

    def record(method, si, li, trajectory):
        errors = _relative_errors(trajectory, truth, weights)
        for b, mask in bands.items():
            band_err[b][method][si, li] = float(np.mean(errors[mask]))
        valid[method][si, li] = _valid_time(errors, times, VALID_TIME_THRESHOLD)

    for si, seed in enumerate(seeds):
        for li, length in enumerate(lengths):
            n_available = train.shape[1] - 2
            starts = np.unique(np.linspace(0, n_available - 1, int(length),
                                           dtype=int))
            u0 = train[:, starts].T.copy()
            dataset_n = dict(dataset, u0=u0, uT=train[:, starts + 1].T.copy(),
                             u2T=train[:, starts + 2].T.copy())
            cfg = dict(seed=int(seed), operator="fno",
                       model_kwargs=_fno_model_kwargs(quick),
                       train_kwargs=_train_kwargs(quick), n_basis=2,
                       joint_kwargs=dict(epochs=1, model_lr=0.0, basis_lr=2e-4,
                                         basis_steps=1, print_every=0))
            ckpt = None if ckpt_root is None else ckpt_root / f"n{length:05d}_seed{seed}"
            result = run_bso_pipeline(dataset_n, cfg, checkpoint_dir=ckpt)
            operator = result.operator_pure
            grid_np = np.asarray(result.grid, dtype=np.float64)
            kind = result.kind
            context = ForecastContext(
                operator=operator, operator_pure=operator, grid=grid_np,
                kind=kind, weights=weights, T=tau,
                operator_val_rel_l2=float(result.operator_val_rel_l2))

            record("fno", si, li,
                   batch_operator_rollout(context, truth[0], horizon))
            ensemble = _fno_ensemble(operator, grid_np, kind, u0)
            bso = (build_matched_bso(operator, grid_np, kind, ensemble)
                   if len(ensemble) > DICT_SIZE else None)
            if bso is not None:
                record("bno", si, li, bso_rollout(bso, truth[0], horizon))
            if int(length) >= DICT_SIZE:
                data = build_matched_bso(operator, grid_np, kind, u0)
                if data is not None:
                    record("data_edmd", si, li,
                           bso_rollout(data, truth[0], horizon))
            # Forced fits on the observed states, ungated: below DICT_SIZE the
            # observed-state PCA and the lifted design matrix are rank
            # deficient, which is exactly the regime these curves expose.
            for method, solver in zip(FORCED_METHODS, ("ridge", "pinv")):
                forced = build_matched_bso(operator, grid_np, kind, u0,
                                           solver=solver)
                if forced is None:
                    continue
                dict_rank[si, li] = float(forced["dictionary_rank"])
                record(method, si, li, bso_rollout(forced, truth[0], horizon))
            print(f"[{SYSTEM}:data-length] N={length} seed={seed} M={DICT_SIZE} "
                  f"rank(Psi)={dict_rank[si, li]:.0f}: "
                  f"mid rel-L2 fno={band_err['mid']['fno'][si, li]:.3f} "
                  f"bso={band_err['mid']['bno'][si, li]:.3f} "
                  f"data={band_err['mid']['data_edmd'][si, li]:.3f} "
                  f"forced[ridge]="
                  f"{band_err['mid']['data_edmd_forced_ridge'][si, li]:.3f} "
                  f"forced[pinv]="
                  f"{band_err['mid']['data_edmd_forced_pinv'][si, li]:.3f} | "
                  f"valid-time bso={valid['bno'][si, li]:.3g} "
                  f"fno={valid['fno'][si, li]:.3g}", flush=True)

    return dict(lengths=np.asarray(lengths), band_err=band_err, valid=valid,
                target_M=DICT_SIZE, dict_rank=dict_rank)


def plot(results: dict, quick: bool) -> list:
    out = FIGURES_DIR / "lorenz96"
    out.mkdir(parents=True, exist_ok=True)
    lengths, M = results["lengths"], results["target_M"]
    import matplotlib.pyplot as plt

    # forced fits are drawn only in the rank-deficient regime N < M; above M
    # the ridge-forced fit is the same computation as data_edmd
    forced_mask = np.asarray(lengths) < M

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    for ax, band in zip(axes, ("mid", "long")):
        for method in ("fno", "bno", "data_edmd"):
            _curve(ax, lengths, results["band_err"][band][method], method, M)
        if forced_mask.any():
            for method in FORCED_METHODS:
                _curve(ax, lengths, results["band_err"][band][method], method,
                       M, mask=forced_mask)
        ax.axvline(M, color="0.5", ls="--", lw=1)
        ax.set(xscale="log", xlabel="Number of training pairs N",
               ylabel="Relative $L^2$",
               title=f"{'0.3<t≤1' if band == 'mid' else 't>1'} band")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Lorenz-96 (matched BSO, K=20, M=30): multi-step error vs. "
                 "training pairs")
    fig.tight_layout()
    err_pdf, _ = save_figure_pair(
        fig, out / "lorenz_96_matched_error_vs_pairs.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for method in ("fno", "bno", "data_edmd"):
        _curve(ax, lengths, results["valid"][method], method, M)
    if forced_mask.any():
        for method in FORCED_METHODS:
            _curve(ax, lengths, results["valid"][method], method, M,
                   mask=forced_mask)
    ax.axvline(M, color="0.5", ls="--", lw=1)
    ax.set(xscale="log", xlabel="Number of training pairs N",
           ylabel=f"Valid-prediction-time (rel. $L^2$ < {VALID_TIME_THRESHOLD})",
           title="Lorenz-96 (matched BSO): valid-prediction-time vs. training pairs")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    vt_pdf, _ = save_figure_pair(
        fig, out / "lorenz_96_matched_valid_time_vs_pairs.pdf")
    plt.close(fig)
    return [err_pdf, vt_pdf]


def save(results: dict) -> None:
    """Persist the sweep numbers so figures can be redrawn without refitting."""
    path = (RESULTS_DIR / "l96_data_length_matched"
            / "l96_data_length_matched_results.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"lengths": results["lengths"], "target_M": results["target_M"],
              "dictionary_rank": results["dict_rank"]}
    for band, methods in results["band_err"].items():
        for method, values in methods.items():
            arrays[f"band_err_{band}_{method}"] = values
    for method, values in results["valid"].items():
        arrays[f"valid_time_{method}"] = values
    np.savez(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    results = run_sweep(args.quick)
    save(results)
    pdfs = plot(results, args.quick)
    print(f"[{SYSTEM}:data-length] figures: {[str(p) for p in pdfs]}")


if __name__ == "__main__":
    main()
