"""ENSO increasing-data study, matched to the ENSO prediction setup.

``exp_enso_data_length`` scores an *unstabilized* BSO one-step. This variant
matches ``exp_enso`` instead: the BSO carries **eigenvalue stabilization**
(``stabilize_max_radius=0.999``) and is scored **multi-step** against the same
baselines used for prediction -- **LIM**, **climatology**, and a projected
**Data EDMD (M=40)** -- as a function of the number of observed training months.

The dictionary is the fixed 40 leading weighted EOFs of the development record,
evaluated over the area measure, so the M=40 BSO exists at every ``N`` (no FNO
ensemble is needed, unlike the Lorenz-96 matched variant). Data EDMD instead
needs ``N >= 40`` monthly transition pairs.

Metrics per lead band (short 1-3 months, long 9-12 months): weighted field
relative-L2 and Niño-3.4 RMSE / anomaly correlation, plus the **Niño-3.4 skill
horizon** (leads while ACC >= 0.5 -- the standard ENSO predictability measure,
and the reference level drawn in ``exp_enso``). A field relative-L2 threshold is
not usable here: climatology scores exactly 1.0 by construction, so every method
sits near that scale from lead 1.

LIM is itself rank-limited: a 20-EOF propagator needs more than 21 months, so it
is reported only where feasible -- the same kind of small-N limit as Data EDMD.

Usage:
    python -m experiments.enso.exp_enso_data_length_matched [--quick]
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True

from edmd_bso_method import run_bso_pipeline
from experiments._data_length import (FORCED_METHODS, METHOD_STYLE, _curve,
                                       dictionary_rollout, fit_projected_edmd,
                                       fno_rollout, gram_diagnostics)
from experiments.enso.enso_data import (build_grid_dataset, fit_eofs,
                                         load_ersstv5)
from experiments.enso.exp_enso import (LIM_MODES, _corr, _field_rel_l2,
                                        _nino34, decode, encode, fit_lim,
                                        lim_forecast)
from plotting_utils import save_figure_pair
from project_paths import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR

SYSTEM = "enso_matched"
TARGET_M = 40
REG_ALPHA = 1e-4
STABILITY_RADIUS = 0.999
HORIZON = 12
QUICK_HORIZON = 4
N_TEST_ORIGINS = 120
SKILL_ACC_THRESHOLD = 0.5      # standard ENSO Niño-3.4 predictability level
DATA_PATH = PROJECT_ROOT.parent / "sst.mnmean.nc"

LENGTHS = (10, 20, 30, 40, 80, 160, 320)
SEEDS = (2026, 2027, 2028, 2029, 2030)
QUICK_LENGTHS = (20, 80)
QUICK_SEEDS = (2026,)

METHODS = ("fno", "bno", "data_edmd", "lim", "climatology", *FORCED_METHODS)
# LIM and climatology are reference baselines; reuse the prediction figure's
# colors so the two ENSO figures read consistently.
EXTRA_STYLE = {"lim": ("#1f77b4", "LIM"),
               "climatology": ("#000000", "Climatology")}


def _model_kwargs(quick: bool) -> dict:
    return (dict(modes=6, width=16, n_layers=2) if quick
            else dict(modes=12, width=48, n_layers=4))


def _train_kwargs(quick: bool) -> dict:
    return dict(epochs=30 if quick else 400, lr=6e-4, bs=16,
                weight_decay=7.5e-5, sg_weight=0.25, val_fraction=0.15,
                print_every=0)


def lead_bands(horizon: int) -> dict[str, np.ndarray]:
    """Lead-month masks: short = 1-3, long = the final quarter of the horizon."""
    leads = np.arange(1, int(horizon) + 1)
    return {"short": leads <= 3,
            "long": leads > max(int(horizon) - 3, int(horizon) * 3 // 4)}


def band_means(per_lead: np.ndarray, bands: dict) -> dict[str, float]:
    values = np.asarray(per_lead, dtype=np.float64)
    return {name: float(np.nanmean(values[mask])) if mask.any() else np.nan
            for name, mask in bands.items()}


def skill_horizon_months(acc_by_lead: np.ndarray,
                         threshold: float = SKILL_ACC_THRESHOLD) -> float:
    """Leads (months) sustained before Niño-3.4 ACC first drops below threshold."""
    values = np.asarray(acc_by_lead, dtype=np.float64)
    below = np.where(~np.isfinite(values) | (values < float(threshold)))[0]
    return float(below[0]) if len(below) else float(len(values))


def run_sweep(quick: bool) -> dict:
    lengths = QUICK_LENGTHS if quick else LENGTHS
    seeds = QUICK_SEEDS if quick else SEEDS
    horizon = QUICK_HORIZON if quick else HORIZON
    bands = lead_bands(horizon)
    data = load_ersstv5(DATA_PATH)
    eof = fit_eofs(data, data.train_end, TARGET_M)
    patterns, fixed_columns = eof.patterns, eof.fixed_columns()

    last = data.test_end - horizon - 1
    origins = np.unique(np.linspace(data.train_end, last, N_TEST_ORIGINS,
                                    dtype=int))
    fields0 = data.anomalies[origins]
    truth = np.stack([data.anomalies[origins + lead]
                      for lead in range(1, horizon + 1)], axis=1)   # (n,H,p)
    truth_n34 = np.stack([_nino34(data, data.anomalies[origins + lead])
                          for lead in range(1, horizon + 1)], axis=1)

    shape = (len(seeds), len(lengths))
    field_err = {b: {m: np.full(shape, np.nan) for m in METHODS} for b in bands}
    n34_acc = {b: {m: np.full(shape, np.nan) for m in METHODS} for b in bands}
    n34_rmse = {b: {m: np.full(shape, np.nan) for m in METHODS} for b in bands}
    horizon_months = {m: np.full(shape, np.nan) for m in METHODS}
    radii = np.full(shape, np.nan)
    gram_rank = np.full(shape, np.nan)
    gram_cond = np.full(shape, np.nan)
    ckpt_root = (None if quick else
                 RESULTS_DIR / "enso_data_length_matched" / "checkpoints")

    def record(method, si, li, pred_fields):
        """pred_fields: (n_origins, horizon, p) anomaly fields."""
        rel = np.array([_field_rel_l2(pred_fields[:, l], truth[:, l],
                                      data.weights) for l in range(horizon)])
        n34 = np.stack([_nino34(data, pred_fields[:, l])
                        for l in range(horizon)], axis=1)
        acc = np.array([_corr(n34[:, l], truth_n34[:, l])
                        for l in range(horizon)])
        rmse = np.sqrt(np.mean((n34 - truth_n34) ** 2, axis=0))
        for b, means in band_means(rel, bands).items():
            field_err[b][method][si, li] = means
        for b, means in band_means(acc, bands).items():
            n34_acc[b][method][si, li] = means
        for b, means in band_means(rmse, bands).items():
            n34_rmse[b][method][si, li] = means
        horizon_months[method][si, li] = skill_horizon_months(acc)

    for si, seed in enumerate(seeds):
        for li, length in enumerate(lengths):
            start = max(data.train_end - 2 - int(length), 0)
            dataset_n = build_grid_dataset(data, start, data.train_end - 2)
            cfg = dict(seed=int(seed), operator="fno",
                       model_kwargs=_model_kwargs(quick),
                       train_kwargs=_train_kwargs(quick), n_basis=TARGET_M,
                       fixed_basis=fixed_columns,
                       joint_kwargs=dict(epochs=1, model_lr=0.0, basis_lr=2e-4,
                                         basis_steps=1, print_every=0),
                       reg_alpha=1e-5,
                       stabilize_max_radius=STABILITY_RADIUS)
            ckpt = (None if ckpt_root is None
                    else ckpt_root / f"n{length:05d}_seed{seed}")
            result = run_bso_pipeline(dataset_n, cfg, checkpoint_dir=ckpt)
            psi = np.asarray(result.psi, dtype=np.float64)
            K = np.asarray(result.K, dtype=np.float64)
            radii[si, li] = float(np.max(np.abs(np.linalg.eigvals(K))))

            # FNO: autoregressive field rollout -> (H, n, p) -> (n, H, p)
            record("fno", si, li,
                   np.moveaxis(fno_rollout(result, "grid2d", fields0, horizon),
                               0, 1))
            # BSO: stabilized dictionary rollout.
            record("bno", si, li,
                   np.moveaxis(dictionary_rollout(psi, K.T, data.weights,
                                                  fields0, horizon), 0, 1))
            # Data EDMD (M=40) on the observed months: needs N >= M.
            u0 = np.asarray(dataset_n["u0"], dtype=np.float64)
            u0 = u0.reshape(u0.shape[0], -1)
            uT = np.asarray(dataset_n["uT"], dtype=np.float64)
            uT = uT.reshape(uT.shape[0], -1)
            if len(u0) >= TARGET_M:
                k_data = fit_projected_edmd(psi, data.weights, u0, uT,
                                            TARGET_M, REG_ALPHA)
                record("data_edmd", si, li,
                       np.moveaxis(dictionary_rollout(psi, k_data, data.weights,
                                                      fields0, horizon), 0, 1))
            # Forced full-dimensional fits, ungated, so they also cover N < M
            # where the Gram is rank deficient.
            gram_rank[si, li], gram_cond[si, li] = gram_diagnostics(
                psi, data.weights, u0, TARGET_M)
            for method, solver in zip(FORCED_METHODS, ("ridge", "pinv")):
                k_forced = fit_projected_edmd(psi, data.weights, u0, uT,
                                              TARGET_M, REG_ALPHA,
                                              solver=solver)
                record(method, si, li,
                       np.moveaxis(dictionary_rollout(psi, k_forced,
                                                      data.weights, fields0,
                                                      horizon), 0, 1))
            # LIM on the same N months of 20-EOF coefficients.
            coeff_n = encode(patterns[:LIM_MODES], data.weights,
                             data.anomalies[start:data.train_end - 2])
            if len(coeff_n) > LIM_MODES + 1:
                G = fit_lim(coeff_n)
                c0 = encode(patterns[:LIM_MODES], data.weights, fields0)
                record("lim", si, li,
                       decode(patterns[:LIM_MODES],
                              lim_forecast(G, c0, horizon)))
            record("climatology", si, li, np.zeros_like(truth))

            print(f"[{SYSTEM}:data-length] N={length} seed={seed} M={TARGET_M}: "
                  f"|mu|max={radii[si, li]:.4f} | long-band ACC "
                  f"bso={n34_acc['long']['bno'][si, li]:+.2f} "
                  f"fno={n34_acc['long']['fno'][si, li]:+.2f} "
                  f"lim={n34_acc['long']['lim'][si, li]:+.2f} "
                  f"data={n34_acc['long']['data_edmd'][si, li]:+.2f}",
                  flush=True)

    return dict(lengths=np.asarray(lengths), seeds=np.asarray(seeds),
                field_err=field_err, n34_acc=n34_acc, n34_rmse=n34_rmse,
                skill_horizon=horizon_months, radii=radii, target_M=TARGET_M,
                horizon=horizon, gram_rank=gram_rank, gram_cond=gram_cond)


def _styled_curve(ax, lengths, per_seed, method, target_M):
    """_curve for suite methods; LIM/climatology use their own style."""
    if method in METHOD_STYLE:
        _curve(ax, lengths, per_seed, method, target_M)
        return
    color, label = EXTRA_STYLE[method]
    median = np.nanmedian(per_seed, axis=0)
    if np.all(np.isnan(median)):
        return
    ax.plot(lengths, median, color=color, marker="o", lw=2, label=label)


def plot(results: dict) -> list:
    out = FIGURES_DIR / "enso"
    out.mkdir(parents=True, exist_ok=True)
    lengths, M = results["lengths"], results["target_M"]
    horizon = results["horizon"]
    # forced fits are shown only where N < M (the rank-deficient regime); above
    # M the ridge-forced fit is the same computation as Data EDMD (M)
    forced_mask = np.asarray(lengths) < M
    long_label = f"leads {max(horizon - 2, 1)}-{horizon} months"
    outputs = []

    panels = (
        ("field_err", "Field relative $L^2$",
         "enso_matched_field_error_vs_pairs.pdf",
         "ENSO (matched BSO, M=40 EOFs): field error vs. training months"),
        ("n34_acc", "Niño-3.4 anomaly correlation",
         "enso_matched_nino34_vs_pairs.pdf",
         "ENSO (matched BSO, M=40 EOFs): Niño-3.4 skill vs. training months"),
    )
    for key, ylabel, filename, suptitle in panels:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
        for ax, band in zip(axes, ("short", "long")):
            for method in METHODS:
                if method in FORCED_METHODS:
                    continue
                _styled_curve(ax, lengths, results[key][band][method], method, M)
            if forced_mask.any():
                for method in FORCED_METHODS:
                    _curve(ax, lengths, results[key][band][method], method, M,
                           mask=forced_mask)
            ax.axvline(M, color="0.5", ls="--", lw=1)
            ax.set(xscale="log", xlabel="Number of training months N",
                   ylabel=ylabel,
                   title="leads 1-3 months" if band == "short" else long_label)
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(suptitle)
        fig.tight_layout()
        pdf, _ = save_figure_pair(fig, out / filename)
        plt.close(fig)
        outputs.append(pdf)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for method in METHODS:
        if method in FORCED_METHODS:
            continue
        _styled_curve(ax, lengths, results["skill_horizon"][method], method, M)
    if forced_mask.any():
        for method in FORCED_METHODS:
            _curve(ax, lengths, results["skill_horizon"][method], method, M,
                   mask=forced_mask)
    ax.axvline(M, color="0.5", ls="--", lw=1)
    ax.set(xscale="log", xlabel="Number of training months N",
           ylabel=f"Niño-3.4 skill horizon (months, ACC ≥ "
                  f"{SKILL_ACC_THRESHOLD})",
           title="ENSO (matched BSO): Niño-3.4 skill horizon vs. training months")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    pdf, _ = save_figure_pair(
        fig, out / "enso_matched_skill_horizon_vs_pairs.pdf")
    plt.close(fig)
    outputs.append(pdf)
    return outputs


def save(results: dict) -> None:
    path = (RESULTS_DIR / "enso_data_length_matched"
            / "enso_data_length_matched_results.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"lengths": results["lengths"], "seeds": results["seeds"],
              "target_M": results["target_M"], "horizon": results["horizon"],
              "spectral_radii": results["radii"],
              "gram_rank": results["gram_rank"],
              "gram_cond": results["gram_cond"]}
    for key in ("field_err", "n34_acc", "n34_rmse"):
        for band, methods in results[key].items():
            for method, values in methods.items():
                arrays[f"{key}_{band}_{method}"] = values
    for method, values in results["skill_horizon"].items():
        arrays[f"skill_horizon_{method}"] = values
    np.savez(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    results = run_sweep(args.quick)
    save(results)
    pdfs = plot(results)
    print(f"[{SYSTEM}:data-length] figures: {[str(p) for p in pdfs]}")


if __name__ == "__main__":
    main()
