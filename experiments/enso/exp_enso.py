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
import torch

sys.dont_write_bytecode = True

from edmd_bso_method import apply_operator, device, run_bso_pipeline
from experiments.enso.enso_data import (area_mean, build_grid_dataset,
                                         fit_eofs, load_ersstv5)
from experiments.enso.enso_direct import (predict_direct, train_direct)
from experiments.enso.enso_lim import (fit_lim_direct, fit_lim_wang,
                                        predict_lim_direct)
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


def damped_persistence_gamma(data, horizon: int,
                             grid: np.ndarray | None = None) -> float:
    """Per-month decay factor for damped persistence, chosen training-only.

    The anomaly field is already climatology-centred, so damped persistence is
    ``gamma**lead * anomaly(t)``; ``gamma = 1`` is plain persistence and
    ``gamma = 0`` is climatology. The factor is selected by minimising
    Niño-3.4 RMSE over leads ``1..horizon`` on origins strictly inside the
    training period, mirroring the SST experiment's fitting-period-only rule.
    """
    grid = np.linspace(0.5, 1.0, 51) if grid is None else np.asarray(grid)
    origins = np.arange(12, data.train_end - horizon)
    index = _nino34(data, data.anomalies)
    launch = index[origins]
    truth = np.stack([index[origins + lead]
                      for lead in range(1, int(horizon) + 1)], axis=1)
    leads = np.arange(1, int(horizon) + 1)
    errors = [np.sqrt(np.mean((g ** leads * launch[:, None] - truth) ** 2))
              for g in grid]
    return float(grid[int(np.argmin(errors))])


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def _model_kwargs(quick):
    return (dict(modes=6, width=16, n_layers=2) if quick
            else dict(modes=12, width=48, n_layers=4))


def run(quick: bool) -> dict:
    data = load_ersstv5(DATA_PATH)
    eof = fit_eofs(data, data.train_end, TARGET_M)
    patterns = eof.patterns
    horizon = 4 if quick else HORIZON

    # BNO: FNO + fixed EOF dictionary, trained on the development record.
    dataset = build_grid_dataset(data, 12, data.train_end - 2)
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
                       data.anomalies[:data.train_end])
    G = fit_lim(coeff_dev)

    # Wang et al. (2020) benchmark LIM (Eqs. 6, 7, 11, 12) and the direct
    # lagged regression -- the linear counterpart of the direct FNO below.
    index = _nino34(data, data.anomalies)
    lim_wang = fit_lim_wang(coeff_dev, index[:data.train_end], tau0=2)
    direct_origins = np.arange(12, data.train_end - horizon)
    lim_direct_w = fit_lim_direct(
        encode(patterns[:LIM_MODES], data.weights, data.anomalies),
        index, direct_origins, horizon)

    # Direct field->index comparator: the same input the BSO sees, but the
    # Niño-3.4 sequence itself is the training target, so the comparison
    # isolates the training objective rather than the input information.
    direct_model, direct_scale = train_direct(
        data, direct_origins, index, horizon,
        modes=(6, 10) if quick else (12, 24), width=16 if quick else 48,
        layers=2 if quick else 4, epochs=30 if quick else 400,
        checkpoint=None if quick else RESULTS_DIR / "enso" / "direct.pt")

    lo, hi = data.train_end, data.test_end - horizon
    origins = np.arange(lo, hi)
    fields0 = data.anomalies[origins]
    truth = np.stack([data.anomalies[origins + lead]
                      for lead in range(1, horizon + 1)], axis=1)   # (n,H,p)
    truth_n34 = np.stack([_nino34(data, data.anomalies[origins + lead])
                          for lead in range(1, horizon + 1)], axis=1)

    bno = bno_forecast(result, patterns, data.weights, fields0, horizon)
    c0_lim = encode(patterns[:LIM_MODES], data.weights, fields0)
    lim = lim_forecast(G, c0_lim, horizon)

    gamma = damped_persistence_gamma(data, horizon)
    decay = gamma ** np.arange(1, horizon + 1)

    methods = {}
    methods["bno"] = decode(patterns, bno)
    methods["lim"] = decode(patterns[:LIM_MODES], lim)
    methods["lim_wang"] = decode(
        patterns[:LIM_MODES],
        lim_wang.coefficient_forecast(c0_lim, horizon))
    methods["climatology"] = np.zeros_like(truth)
    methods["persistence"] = np.repeat(fields0[:, None], horizon, axis=1)
    methods["damped_persistence"] = (fields0[:, None]
                                     * decay[None, :, None])

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

    # The direct model yields only the index, so its field error is undefined;
    # NaN keeps it out of the field panel explicitly rather than silently.
    lim_direct_n34 = predict_lim_direct(
        lim_direct_w, encode(patterns[:LIM_MODES], data.weights, fields0))
    metrics["lim_direct"] = dict(
        rel_l2=np.full(horizon, np.nan),           # index only: no field
        nino34_rmse=np.sqrt(np.mean((lim_direct_n34 - truth_n34) ** 2, axis=0)),
        nino34_acc=np.array([_corr(lim_direct_n34[:, l], truth_n34[:, l])
                             for l in range(horizon)]),
        valid_time=-1)

    direct_n34 = predict_direct(direct_model, fields0, direct_scale)
    metrics["direct_fno"] = dict(
        rel_l2=np.full(horizon, np.nan),
        nino34_rmse=np.sqrt(np.mean((direct_n34 - truth_n34) ** 2, axis=0)),
        nino34_acc=np.array([_corr(direct_n34[:, l], truth_n34[:, l])
                             for l in range(horizon)]),
        valid_time=-1)

    K = np.asarray(result.K, dtype=np.float64)
    growth = {
        "bso": [float(np.linalg.norm(np.linalg.matrix_power(K, n), 2))
                for n in range(1, horizon + 1)],
        "lim": [float(np.linalg.norm(np.linalg.matrix_power(G, n), 2))
                for n in range(1, horizon + 1)],
    }
    return dict(data=data, horizon=horizon, metrics=metrics, gamma=gamma,
                operator_val_rel_l2=float(result.operator_val_rel_l2),
                growth=growth,
                one_step=one_step_diagnostic(data, result, G, patterns,
                                             origins, fields0))


def one_step_diagnostic(data, result, G, patterns, origins, fields0) -> dict:
    """One-month field error of the raw operator, LIM and persistence.

    BSO compresses the learned map, so its lead-1 error cannot be better than
    the map itself; this isolates whether the deficit originates in the
    operator or in the Galerkin compression.
    """
    target = data.anomalies[origins + 1]
    grid = torch.as_tensor(result.grid, dtype=torch.float32, device=device)
    shaped = fields0.reshape(len(fields0), *data.shape)
    with torch.no_grad():
        fno = apply_operator(
            result.operator_pure,
            torch.as_tensor(shaped.reshape(len(shaped), -1),
                            dtype=torch.float32, device=device),
            grid, "grid2d").cpu().numpy().astype(np.float64)
    lim_one = decode(patterns[:LIM_MODES],
                     encode(patterns[:LIM_MODES], data.weights, fields0) @ G.T)
    return {
        "fno_map": _field_rel_l2(fno, target, data.weights),
        "lim": _field_rel_l2(lim_one, target, data.weights),
        "persistence": _field_rel_l2(fields0, target, data.weights),
    }


def plot(metrics, horizon, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    leads = np.arange(1, horizon + 1)
    style = {"bno": ("#d62728", "BSO (field rollout)"),
             "lim": ("#1f77b4", "LIM (field rollout)"),
             "climatology": ("#000000", "Climatology"),
             "persistence": ("#7f7f7f", "Persistence"),
             "damped_persistence": ("#2ca02c", "Damped persistence"),
             "lim_wang": ("#17becf", "LIM (Wang et al., τ₀=2)"),
             "lim_direct": ("#8c564b", "Direct lagged LIM (index only)"),
             "direct_fno": ("#ff7f0e", "Direct field→index FNO (index only)")}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for name, (color, label) in style.items():
        m = metrics[name]
        # A zero-anomaly forecast makes the Niño-3.4 RMSE equal to the RMS of
        # the verifying observations, which drifts slightly because each lead
        # verifies a different set of target months. That is a property of the
        # verification window, not of the forecast, so climatology is drawn as
        # its lead-average.
        rmse = (np.full_like(m["nino34_rmse"], float(np.mean(m["nino34_rmse"])))
                if name == "climatology" else m["nino34_rmse"])
        if np.isfinite(m["rel_l2"]).any():          # index-only methods skip
            axes[0].plot(leads, m["rel_l2"], color=color, marker="o",
                         label=label)
        axes[1].plot(leads, rmse, color=color, marker="o", label=label)
        # Damped persistence scales persistence by gamma**lead, and correlation
        # is invariant to a positive per-lead scaling, so its ACC coincides
        # exactly with persistence's. Drawn dashed on top rather than omitted,
        # so a legend entry does not look like a missing curve.
        if name == "damped_persistence":
            axes[2].plot(leads, m["nino34_acc"], color=color, ls="--", lw=1.6,
                         label=f"{label} (= persistence)")
        else:
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
    # One shared legend under all three panels: eight entries inside an axis
    # covered the curves. Panel 2 carries every method, so take its handles.
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center",
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle("ENSO prediction on ERSSTv5: field-rollout vs. direct "
                 "Niño-3.4 forecasts, learned and linear")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, top=0.86)
    pdf, _ = save_figure_pair(fig, out_dir / "enso_prediction_skill.pdf")
    plt.close(fig)
    return pdf


def radius_sweep(quick: bool,
                 radii=(0.999, 0.99, 0.95)) -> None:
    """Diagnostic only: how BSO responds to stricter eigenvalue clipping.

    Reuses the cached FNO checkpoint, so only the (cheap) Galerkin projection is
    recomputed. Changes no default.
    """
    data = load_ersstv5(DATA_PATH)
    eof = fit_eofs(data, data.train_end, TARGET_M)
    patterns = eof.patterns
    horizon = 4 if quick else HORIZON
    dataset = build_grid_dataset(data, 12, data.train_end - 2)
    lo, hi = data.train_end, data.test_end - horizon
    origins = np.arange(lo, hi)
    fields0 = data.anomalies[origins]
    truth = np.stack([data.anomalies[origins + lead]
                      for lead in range(1, horizon + 1)], axis=1)
    truth_n34 = np.stack([_nino34(data, data.anomalies[origins + lead])
                          for lead in range(1, horizon + 1)], axis=1)
    print("\n[enso] stabilisation-radius sweep (diagnostic; defaults unchanged)")
    print(f"  {'radius':>8} {'||K^H||':>9} {'relL2@1':>9} {'relL2@H':>9} "
          f"{'ACC@1':>7} {'ACC@H':>7}")
    for radius in radii:
        cfg = dict(seed=2026, operator="fno", model_kwargs=_model_kwargs(quick),
                   train_kwargs=dict(epochs=30 if quick else 400, lr=6e-4,
                                     bs=16, weight_decay=7.5e-5,
                                     sg_weight=0.25, val_fraction=0.15,
                                     print_every=0),
                   n_basis=TARGET_M, fixed_basis=eof.fixed_columns(),
                   joint_kwargs=dict(epochs=1, model_lr=0.0, basis_lr=2e-4,
                                     basis_steps=1, print_every=0),
                   reg_alpha=1e-5, stabilize_max_radius=radius)
        result = run_bso_pipeline(dataset, cfg,
                                  checkpoint_dir=None if quick
                                  else RESULTS_DIR / "enso")
        pred = decode(patterns, bno_forecast(result, patterns, data.weights,
                                             fields0, horizon))
        rel = np.array([_field_rel_l2(pred[:, l], truth[:, l], data.weights)
                        for l in range(horizon)])
        n34 = np.stack([_nino34(data, pred[:, l]) for l in range(horizon)],
                       axis=1)
        acc = np.array([_corr(n34[:, l], truth_n34[:, l])
                        for l in range(horizon)])
        K = np.asarray(result.K, dtype=np.float64)
        growth = np.linalg.norm(np.linalg.matrix_power(K, horizon), 2)
        print(f"  {radius:>8.3f} {growth:>9.3f} {rel[0]:>9.3f} {rel[-1]:>9.3f} "
              f"{acc[0]:>+7.2f} {acc[-1]:>+7.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--radius-sweep", action="store_true",
                        help="diagnostic sweep over stabilize_max_radius")
    args = parser.parse_args()
    if args.radius_sweep:
        radius_sweep(args.quick)
        return
    result = run(args.quick)
    pdf = plot(result["metrics"], result["horizon"], FIGURES_DIR / "enso")
    print(f"\n[enso] operator val rel-L2 = {result['operator_val_rel_l2']:.4f}")
    print(f"[enso] damped-persistence gamma = {result['gamma']:.3f} "
          f"(training-period selection)")
    one = result["one_step"]
    print("[enso] one-month field relative L2 (isolates the operator):")
    for name in ("fno_map", "lim", "persistence"):
        print(f"  {name:12s} {one[name]:.4f}")
    growth = result["growth"]
    print(f"[enso] transient growth ||A^n|| at n=1,H: "
          f"BSO {growth['bso'][0]:.3f},{growth['bso'][-1]:.3f}  "
          f"LIM {growth['lim'][0]:.3f},{growth['lim'][-1]:.3f}")
    print("[enso] Niño-3.4 anomaly correlation by method (lead 1..H):")
    for name, m in result["metrics"].items():
        acc = " ".join(f"{v:+.2f}" for v in m["nino34_acc"])
        print(f"  {name:20s} valid_time={m['valid_time']:2d}  ACC: {acc}")
    print(f"[enso] figure: {pdf}")


if __name__ == "__main__":
    main()
