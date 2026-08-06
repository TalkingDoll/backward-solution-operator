"""ENSO prediction on a delay-embedded state, with a memory-matched LIM.

``exp_enso`` forecasts from an *instantaneous* monthly SST anomaly field. ENSO is
a delayed/recharge oscillator whose tendency is not determined by that snapshot
(the thermocline memory is not in SST), so a one-month map has little to
propagate. This variant instead carries a **12-month delay embedding** of the
leading EOF coefficients -- the construction used by the older comparison suite
(``comparison_experiments/enso_koopman.py``), where it was the source of that
suite's long-lead skill.

Crucially, the old suite gave its EDMD 12 months of memory while leaving LIM a
single snapshot, so part of that apparent win was extra state information rather
than a better operator. Here LIM is reported **both ways**:

``lim``
    the classical instantaneous lag-1 propagator on the leading EOFs;
``lim_delay``
    the *same* estimator applied to the *same* 12-month delay state as the BSO.

The three-way gap separates the two effects: ``lim -> lim_delay`` is what the
memory alone buys, and ``lim_delay -> BSO`` is what the learned operator adds on
top of it.

Usage:
    python -m experiments.enso.exp_enso_delay [--quick]
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
from experiments.enso.enso_data import fit_eofs, load_ersstv5
from experiments.enso.enso_direct import predict_direct, train_direct
from experiments.enso.enso_lim import (fit_lim_direct, fit_lim_wang,
                                        predict_lim_direct)
from experiments.enso.exp_enso import (DATA_PATH, HORIZON, LIM_MODES, TARGET_M,
                                        _corr, _field_rel_l2, _nino34,
                                        _valid_time, damped_persistence_gamma,
                                        decode, encode, fit_lim, lim_forecast)
from plotting_utils import save_figure_pair
from project_paths import FIGURES_DIR, RESULTS_DIR

HISTORY = 12          # months of memory, matching the old comparison suite


def delay_state(coefficients: np.ndarray, history: int = HISTORY) -> np.ndarray:
    """Chronological delay blocks, oldest first, newest last.

    Mirrors ``comparison_experiments/enso_koopman._delay_coefficients``: row
    ``t`` holds months ``t-history+1 … t`` flattened, and rows without full
    context are NaN so accidental use of missing data is obvious.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    history = int(history)
    n_modes = coefficients.shape[1]
    out = np.full((len(coefficients), history * n_modes), np.nan)
    for end in range(history - 1, len(coefficients)):
        out[end] = coefficients[end - history + 1:end + 1].reshape(-1)
    return out


def newest_block(delay: np.ndarray, n_modes: int) -> np.ndarray:
    """The most recent coefficient block of a delay vector."""
    return np.asarray(delay)[..., -int(n_modes):]


def build_delay_dataset(states: np.ndarray, first: int, last: int) -> dict:
    """A ``sensor1d`` BSO dataset of consecutive delay vectors in [first, last)."""
    ends = np.arange(int(first), int(last))
    p = states.shape[1]
    grid = np.linspace(0.0, 1.0, p, endpoint=False)
    return dict(
        system="enso_delay", kind="sensor1d", T=1.0,
        u0=states[ends].copy(), uT=states[ends + 1].copy(),
        u2T=states[ends + 2].copy(),
        fno_grid=grid, basis_coords=grid[:, None],
        weights=np.full(p, 1.0 / p),
        meta=dict(history=HISTORY))


def delay_pca_columns(states: np.ndarray, n_columns: int) -> np.ndarray:
    """Leading PCA directions of the training delay states, as dictionary columns.

    Returns ``(p, n_columns)``; ``build_basis`` prepends the constant, so these
    are the analogue of the EOF columns in the instantaneous experiment.
    """
    centred = states - states.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[:int(n_columns)].T.copy()


def fit_lim_delay(states: np.ndarray):
    """LIM given the same memory as the BSO: the estimator is unchanged.

    ``fit_lim`` is applied verbatim to the delay-embedded coefficients, so this
    differs from the instantaneous ``lim`` only by the state it sees. With
    ``history = 1`` the two coincide exactly.
    """
    return fit_lim(states)


def _model_kwargs(quick):
    return (dict(modes=8, width=32, n_layers=2) if quick
            else dict(modes=24, width=64, n_layers=4))


def run(quick: bool) -> dict:
    data = load_ersstv5(DATA_PATH)
    eof = fit_eofs(data, data.train_end, TARGET_M)
    patterns = eof.patterns
    horizon = 4 if quick else HORIZON

    # Delay states for the BSO (TARGET_M EOFs) and for the matched LIM.
    coeff = encode(patterns, data.weights, data.anomalies)
    states = delay_state(coeff, HISTORY)
    lim_states = delay_state(coeff[:, :LIM_MODES], HISTORY)

    first = HISTORY - 1
    dataset = build_delay_dataset(states, first, data.train_end - 2)
    fixed = delay_pca_columns(states[first:data.train_end - 2], TARGET_M - 1)
    cfg = dict(seed=2026, operator="fno", model_kwargs=_model_kwargs(quick),
               train_kwargs=dict(epochs=30 if quick else 400, lr=6e-4, bs=16,
                                 weight_decay=7.5e-5, sg_weight=0.25,
                                 val_fraction=0.15, print_every=0),
               n_basis=TARGET_M, fixed_basis=fixed,
               joint_kwargs=dict(epochs=1, model_lr=0.0, basis_lr=2e-4,
                                 basis_steps=1, print_every=0),
               reg_alpha=1e-5, stabilize_max_radius=0.999)
    ckpt = None if quick else RESULTS_DIR / "enso_delay"
    result = run_bso_pipeline(dataset, cfg, checkpoint_dir=ckpt)

    # Baselines, all fitted on the training period only.
    G = fit_lim(encode(patterns[:LIM_MODES], data.weights,
                       data.anomalies[:data.train_end]))
    G_delay = fit_lim_delay(lim_states[first:data.train_end])

    # Wang et al. (2020) benchmark LIM on the instantaneous PCs, and the direct
    # lagged regression on the *delay* state (linear twin of the direct FNO).
    index = _nino34(data, data.anomalies)
    coeff_lim = encode(patterns[:LIM_MODES], data.weights, data.anomalies)
    lim_wang = fit_lim_wang(coeff_lim[:data.train_end],
                            index[:data.train_end], tau0=2)

    # Direct field->index comparator, given the same 12 months of memory as
    # the BSO here: the input is a stack of history fields, but the training
    # target is the Niño-3.4 sequence itself.
    hist_fields = lambda ends: np.stack(
        [data.anomalies[e - HISTORY + 1:e + 1].reshape(-1) for e in ends])
    direct_origins = np.arange(HISTORY - 1, data.train_end - horizon)
    lim_direct_w = fit_lim_direct(np.nan_to_num(lim_states), index,
                                  direct_origins, horizon)
    direct_model, direct_scale = train_direct(
        data, direct_origins, index, horizon,
        inputs=hist_fields(direct_origins), in_channels=HISTORY,
        modes=(6, 10) if quick else (12, 24), width=16 if quick else 48,
        layers=2 if quick else 4, epochs=30 if quick else 400,
        checkpoint=None if quick else RESULTS_DIR / "enso_delay" / "direct.pt")

    lo, hi = data.train_end, data.test_end - horizon
    origins = np.arange(lo, hi)
    fields0 = data.anomalies[origins]
    truth = np.stack([data.anomalies[origins + lead]
                      for lead in range(1, horizon + 1)], axis=1)
    truth_n34 = np.stack([_nino34(data, data.anomalies[origins + lead])
                          for lead in range(1, horizon + 1)], axis=1)

    # BSO: roll the delay vector, read the newest block, decode to the field.
    psi, w, K = result.psi, result.weights, np.asarray(result.K)
    C = psi @ (w[:, None] * states[origins].T)
    bso = []
    for _ in range(horizon):
        C = K @ C
        bso.append(decode(patterns, newest_block((psi.T @ C).T, TARGET_M)))
    bso = np.stack(bso, axis=1)

    lim_inst = decode(patterns[:LIM_MODES],
                      lim_forecast(G, encode(patterns[:LIM_MODES],
                                             data.weights, fields0), horizon))
    delay_roll = lim_forecast(G_delay, lim_states[origins], horizon)
    lim_mem = decode(patterns[:LIM_MODES],
                     newest_block(delay_roll, LIM_MODES))

    gamma = damped_persistence_gamma(data, horizon)
    decay = gamma ** np.arange(1, horizon + 1)
    methods = {
        "bno": bso,
        "lim_delay": lim_mem,
        "lim": lim_inst,
        "lim_wang": decode(patterns[:LIM_MODES],
                           lim_wang.coefficient_forecast(
                               encode(patterns[:LIM_MODES], data.weights,
                                      fields0), horizon)),
        "climatology": np.zeros_like(truth),
        "persistence": np.repeat(fields0[:, None], horizon, axis=1),
        "damped_persistence": fields0[:, None] * decay[None, :, None],
    }

    metrics = {}
    for name, pred in methods.items():
        rel = np.array([_field_rel_l2(pred[:, l], truth[:, l], data.weights)
                        for l in range(horizon)])
        n34 = np.stack([_nino34(data, pred[:, l]) for l in range(horizon)],
                       axis=1)
        metrics[name] = dict(
            rel_l2=rel,
            nino34_rmse=np.sqrt(np.mean((n34 - truth_n34) ** 2, axis=0)),
            nino34_acc=np.array([_corr(n34[:, l], truth_n34[:, l])
                                 for l in range(horizon)]),
            valid_time=_valid_time(rel))

    lim_direct_n34 = predict_lim_direct(lim_direct_w, lim_states[origins])
    metrics["lim_direct"] = dict(
        rel_l2=np.full(horizon, np.nan),           # index only: no field
        nino34_rmse=np.sqrt(np.mean((lim_direct_n34 - truth_n34) ** 2, axis=0)),
        nino34_acc=np.array([_corr(lim_direct_n34[:, l], truth_n34[:, l])
                             for l in range(horizon)]),
        valid_time=-1)

    direct_n34 = predict_direct(direct_model, hist_fields(origins),
                                direct_scale)
    metrics["direct_fno"] = dict(
        rel_l2=np.full(horizon, np.nan),           # index only: no field
        nino34_rmse=np.sqrt(np.mean((direct_n34 - truth_n34) ** 2, axis=0)),
        nino34_acc=np.array([_corr(direct_n34[:, l], truth_n34[:, l])
                             for l in range(horizon)]),
        valid_time=-1)

    growth = [float(np.linalg.norm(np.linalg.matrix_power(K, n), 2))
              for n in range(1, horizon + 1)]
    return dict(horizon=horizon, metrics=metrics, gamma=gamma, growth=growth,
                operator_val_rel_l2=float(result.operator_val_rel_l2),
                one_step=_one_step(data, result, states, origins, patterns,
                                   fields0))


def _one_step(data, result, states, origins, patterns, fields0) -> dict:
    """One-month field error of the delay operator vs persistence.

    The instantaneous variant's operator lost to persistence (0.671 vs 0.460);
    this is the same measurement with 12 months of memory available.
    """
    import torch

    from edmd_bso_method import apply_operator, device

    target = data.anomalies[origins + 1]
    grid = torch.as_tensor(result.grid, dtype=torch.float32, device=device)
    with torch.no_grad():
        advanced = apply_operator(
            result.operator_pure,
            torch.as_tensor(states[origins], dtype=torch.float32,
                            device=device),
            grid, "sensor1d").cpu().numpy().astype(np.float64)
    return {
        "fno_map": _field_rel_l2(decode(patterns,
                                        newest_block(advanced, TARGET_M)),
                                 target, data.weights),
        "persistence": _field_rel_l2(fields0, target, data.weights),
    }


def plot(metrics, horizon, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    leads = np.arange(1, horizon + 1)
    style = {"bno": ("#d62728", "BSO (delay, field rollout)"),
             "lim_delay": ("#9467bd", "LIM (delay, matched memory)"),
             "lim": ("#1f77b4", "LIM (instantaneous, field rollout)"),
             "climatology": ("#000000", "Climatology"),
             "persistence": ("#7f7f7f", "Persistence"),
             "damped_persistence": ("#2ca02c", "Damped persistence"),
             "lim_wang": ("#17becf", "LIM (Wang et al., τ₀=2)"),
             "lim_direct": ("#8c564b", "Direct lagged LIM (index only)"),
             "direct_fno": ("#ff7f0e", "Direct field→index FNO (index only)")}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for name, (color, label) in style.items():
        m = metrics[name]
        rmse = (np.full_like(m["nino34_rmse"], float(np.mean(m["nino34_rmse"])))
                if name == "climatology" else m["nino34_rmse"])
        if np.isfinite(m["rel_l2"]).any():          # index-only methods skip
            axes[0].plot(leads, m["rel_l2"], color=color, marker="o",
                         label=label)
        axes[1].plot(leads, rmse, color=color, marker="o", label=label)
        if name == "damped_persistence":
            # Correlation is invariant to the positive per-lead scaling
            # gamma**lead, so this coincides exactly with persistence. Drawn
            # dashed on top of it: omitting it instead makes a legend entry
            # look like a missing curve.
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
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center",
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"ENSO prediction with a {HISTORY}-month delay embedding "
                 "(ERSSTv5): field-rollout vs. direct Niño-3.4 forecasts")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, top=0.86)
    pdf, _ = save_figure_pair(fig, out_dir / "enso_delay_prediction_skill.pdf")
    plt.close(fig)
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run(args.quick)
    pdf = plot(result["metrics"], result["horizon"], FIGURES_DIR / "enso")
    print(f"\n[enso-delay] operator val rel-L2 = "
          f"{result['operator_val_rel_l2']:.4f}")
    print(f"[enso-delay] damped-persistence gamma = {result['gamma']:.3f}")
    one = result["one_step"]
    print(f"[enso-delay] one-month field relative L2: "
          f"delay FNO map {one['fno_map']:.4f}  persistence "
          f"{one['persistence']:.4f}")
    print(f"[enso-delay] transient growth ||K^n|| at n=1,H: "
          f"{result['growth'][0]:.3f},{result['growth'][-1]:.3f}")
    print("[enso-delay] Niño-3.4 anomaly correlation by method (lead 1..H):")
    for name, m in result["metrics"].items():
        acc = " ".join(f"{v:+.2f}" for v in m["nino34_acc"])
        print(f"  {name:20s} valid_time={m['valid_time']:2d}  ACC: {acc}")
    print(f"[enso-delay] figure: {pdf}")


if __name__ == "__main__":
    main()
