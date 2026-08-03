"""ENSO prediction: BNO vs. LIM and climatology on monthly ERSSTv5.

The BNO (``run_bso_pipeline``) advances the Indo-Pacific SST-anomaly field with
an FNO and projects it onto ``M=40`` fixed weighted EOFs. It is compared with

* **LIM** -- a linear inverse model, a lag-1 propagator on the leading 20 EOFs;
* **Climatology** -- the zero-anomaly forecast;
* **Persistence** -- the initial anomaly held fixed.

All methods are scored with the same metrics used across the repository:
weighted relative-L2 field error and Niño-3.4 RMSE / anomaly correlation per
lead, plus the valid-prediction-time horizon.

Usage:
    python -m experiments.enso.exp_enso [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.dont_write_bytecode = True

from edmd_bso_method import run_bso_pipeline
from experiments.enso.enso_data import (area_mean, build_grid_dataset,
                                         fit_eofs, load_ersstv5)
from plotting_utils import save_figure_pair
from project_paths import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR

TARGET_M = 40
LIM_MODES = 20
HORIZON = 12
DATA_PATH = PROJECT_ROOT.parent / "sst.mnmean.nc"


# ---------------------------------------------------------------------------
# EOF encode/decode and the minimal LIM baseline
# ---------------------------------------------------------------------------

def encode(patterns, weights, fields):
    return (np.asarray(fields) * weights[None]) @ patterns.T


def decode(patterns, coeff):
    return np.asarray(coeff) @ patterns


def fit_lim(coeff, ridge=1e-6):
    """Lag-1 EOF-space propagator: c_{t+1} ~= c_t @ G^T."""
    x, y = coeff[:-1], coeff[1:]
    gram = x.T @ x
    reg = ridge * np.trace(gram) / gram.shape[0]
    return np.linalg.solve(gram + reg * np.eye(gram.shape[0]), x.T @ y).T


def lim_forecast(G, c0, horizon):
    out, c = [], np.asarray(c0, dtype=np.float64)
    for _ in range(horizon):
        c = c @ G.T
        out.append(c)
    return np.stack(out, axis=1)                       # (n_orig, horizon, r)


def bno_forecast(result, patterns, weights, fields0, horizon):
    """Batched linear rollout in the BSO dictionary, decoded to EOF space."""
    psi, w, K = result.psi, result.weights, result.K
    C = psi @ (w[:, None] * np.asarray(fields0, dtype=np.float64).T)   # (M, n)
    out = []
    for _ in range(horizon):
        C = K @ C
        field = (psi.T @ C).T                                          # (n, p)
        out.append(encode(patterns, weights, field))
    return np.stack(out, axis=1)                       # (n_orig, horizon, r)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _field_rel_l2(pred_field, truth_field, weights):
    num = np.sqrt(np.sum(weights[None] * (pred_field - truth_field) ** 2, axis=1))
    den = np.sqrt(np.sum(weights[None] * truth_field ** 2, axis=1)) + 1e-12
    return float(np.mean(num / den))


def _nino34(data, fields):
    return area_mean(np.asarray(fields) * data.scale, data.weights,
                     data.nino_masks["nino34"])


def _corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _valid_time(rel_l2_by_lead, threshold=0.6):
    over = np.where(np.asarray(rel_l2_by_lead) > threshold)[0]
    return int(over[0]) if len(over) else len(rel_l2_by_lead)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def _model_kwargs(quick):
    return (dict(modes=6, width=16, n_layers=2) if quick
            else dict(modes=12, width=48, n_layers=4))


def run(quick: bool) -> dict:
    data = load_ersstv5(DATA_PATH)
    eof = fit_eofs(data, data.dev_end, TARGET_M)
    patterns = eof.patterns
    horizon = 4 if quick else HORIZON

    # BNO: FNO + fixed EOF dictionary, trained on the development record.
    dataset = build_grid_dataset(data, 12, data.dev_end - 2)
    cfg = dict(seed=2026, operator="fno", model_kwargs=_model_kwargs(quick),
               train_kwargs=dict(epochs=30 if quick else 400, lr=6e-4, bs=16,
                                 weight_decay=7.5e-5, sg_weight=0.25,
                                 val_fraction=0.15, print_every=0),
               n_basis=TARGET_M, fixed_basis=eof.fixed_columns(),
               joint_kwargs=dict(epochs=1, model_lr=0.0, basis_lr=2e-4,
                                 basis_steps=1, print_every=0),
               reg_alpha=1e-5, stabilize_max_radius=0.999)
    ckpt = None if quick else RESULTS_DIR / "enso"
    result = run_bso_pipeline(dataset, cfg, checkpoint_dir=ckpt)

    # LIM on the development EOF coefficients.
    coeff_dev = encode(patterns[:LIM_MODES], data.weights,
                       data.anomalies[:data.dev_end])
    G = fit_lim(coeff_dev)

    # Primary verification origins.
    lo, hi = data.dev_end, data.primary_end - horizon
    origins = np.arange(lo, hi)
    fields0 = data.anomalies[origins]
    truth = np.stack([data.anomalies[origins + lead]
                      for lead in range(1, horizon + 1)], axis=1)   # (n,H,p)
    truth_n34 = np.stack([_nino34(data, data.anomalies[origins + lead])
                          for lead in range(1, horizon + 1)], axis=1)

    bno = bno_forecast(result, patterns, data.weights, fields0, horizon)
    c0_lim = encode(patterns[:LIM_MODES], data.weights, fields0)
    lim = lim_forecast(G, c0_lim, horizon)

    methods = {}
    methods["bno"] = decode(patterns, bno)
    methods["lim"] = decode(patterns[:LIM_MODES], lim)
    methods["climatology"] = np.zeros_like(truth)
    methods["persistence"] = np.repeat(fields0[:, None], horizon, axis=1)

    metrics = {}
    for name, pred in methods.items():
        rel = np.array([_field_rel_l2(pred[:, l], truth[:, l], data.weights)
                        for l in range(horizon)])
        n34 = np.stack([_nino34(data, pred[:, l]) for l in range(horizon)],
                       axis=1)                                        # (n,H)
        rmse = np.sqrt(np.mean((n34 - truth_n34) ** 2, axis=0))
        acc = np.array([_corr(n34[:, l], truth_n34[:, l])
                        for l in range(horizon)])
        metrics[name] = dict(rel_l2=rel, nino34_rmse=rmse, nino34_acc=acc,
                             valid_time=_valid_time(rel))
    return dict(data=data, horizon=horizon, metrics=metrics,
                operator_val_rel_l2=float(result.operator_val_rel_l2))


def plot(metrics, horizon, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    leads = np.arange(1, horizon + 1)
    style = {"bno": ("#d62728", "BNO"), "lim": ("#1f77b4", "LIM"),
             "climatology": ("#000000", "Climatology"),
             "persistence": ("#7f7f7f", "Persistence")}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for name, (color, label) in style.items():
        m = metrics[name]
        axes[0].plot(leads, m["rel_l2"], color=color, marker="o", label=label)
        axes[1].plot(leads, m["nino34_rmse"], color=color, marker="o",
                     label=label)
        axes[2].plot(leads, m["nino34_acc"], color=color, marker="o",
                     label=label)
    axes[0].set(xlabel="Lead (months)", ylabel="Field relative $L^2$",
                title="Indo-Pacific field error")
    axes[1].set(xlabel="Lead (months)", ylabel="RMSE (°C)",
                title="Niño-3.4 error")
    axes[2].set(xlabel="Lead (months)", ylabel="Anomaly correlation",
                title="Niño-3.4 skill", ylim=(-0.2, 1.02))
    for level in (0.5, 0.6):
        axes[2].axhline(level, color="0.6", ls="--", lw=1)
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("ENSO prediction: BNO vs. LIM and climatology (ERSSTv5)")
    fig.tight_layout()
    pdf, _ = save_figure_pair(fig, out_dir / "enso_prediction_skill.pdf")
    plt.close(fig)
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run(args.quick)
    pdf = plot(result["metrics"], result["horizon"], FIGURES_DIR / "enso")
    print(f"\n[enso] operator val rel-L2 = {result['operator_val_rel_l2']:.4f}")
    print("[enso] Niño-3.4 anomaly correlation by method (lead 1..H):")
    for name, m in result["metrics"].items():
        acc = " ".join(f"{v:+.2f}" for v in m["nino34_acc"])
        print(f"  {name:12s} valid_time={m['valid_time']:2d}  ACC: {acc}")
    print(f"[enso] figure: {pdf}")


if __name__ == "__main__":
    main()
