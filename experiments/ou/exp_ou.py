"""OU BSO validation against the analytic spectrum lambda_n = -n.

Usage:
    python -m experiments.ou.exp_ou [--quick] [--solver solver_edmd.py|solver_resdmd.py]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite_e import hermeval

from kbe_observation_data import generate_kbe_dataset
from edmd_bso_method import run_bso_pipeline
from plotting_utils import (
    PUBLICATION_STYLE,
    apply_physical_layout,
    apply_publication_rcparams,
    save_figure_pair,
    style_axis,
)
from project_paths import FIGURES_DIR, RESULTS_DIR


OU_EVALUATION_FIGURE = {
    "figsize": (7.0, 3.0),
    "nrows": 1,
    "ncols": 2,
    "left_in": 0.52,
    "right_in": 0.10,
    "bottom_in": 0.48,
    "top_in": 0.30,
    "column_gap_in": 0.55,
    "row_gap_in": 0.0,
    "legend_ncol": 1,
}


def hermite_fixed_columns(dataset, k=11):
    """Enveloped Hermite dictionary columns He_n(x/sigma)*env, unit L2(rho) norm."""
    meta = dataset["meta"]
    x = meta["x"]
    sigma = np.sqrt(meta["params"]["D"])
    env = meta["envelope"]
    w = dataset["weights"]
    cols = []
    for n_deg in range(1, k + 1):
        c = np.zeros(n_deg + 1)
        c[-1] = 1.0
        h = hermeval(x / sigma, c) * env
        cols.append(h / (np.sqrt(np.sum(w * h * h)) + 1e-300))
    return np.stack(cols, axis=1)


def build_deflation_config(quick: bool) -> dict:
    """Second OU experiment: deflation-trained dictionary (sequential per-mode).

    Slow modes are learned one at a time: each stage trains a single function
    in the rho-orthogonal complement of the previously accepted columns,
    maximizing the Rayleigh quotient under an eigen-residual penalty, a
    semigroup non-expansiveness cap, and a consistency guard built from the
    training pairs. Stages pass or fail on data-driven residual tolerances;
    the dictionary is then filled to the same 20 columns with the joint
    recipe. Every training run uses 1000 epochs. Cached in
    galerkin.pt.
    """
    cfg = build_config(quick)
    epochs = 100 if quick else 1000
    cfg["fixed_basis"] = None
    cfg["n_basis"] = 20
    cfg["basis_kwargs"] = dict(hidden=224, depth=4, num_freq=16)
    cfg["joint_kwargs"] = dict(
        epochs=epochs, model_lr=0.0, basis_lr=2e-4, basis_steps=2,
        inv_weight=2.5, data_weight=0.0, print_every=epochs // 5,
        smooth_weight=0.05, symmetrize_target=True)
    cfg["deflation"] = dict(
        n_stages=5, epochs=epochs, lr=3e-4, res_weight=5.0, cons_weight=5.0,
        cap_weight=10.0, smooth_weight=0.05, res_tol=2e-3, cons_tol=5e-3,
        seed0=2033)
    cfg["symmetrize_K"] = True
    cfg["seed"] = 2033
    cfg["bso_file"] = "galerkin.pt"
    return cfg


def build_config(quick: bool) -> dict:
    if quick:
        return dict(
            model_kwargs=dict(modes=24, width=64, n_layers=4),
            train_kwargs=dict(epochs=500, lr=6e-4, bs=24, weight_decay=7.5e-5,
                              sg_weight=0.25, val_fraction=0.10, print_every=100,
                              const_weight=0.15, weighted_loss=True),
            n_basis=20,
            joint_kwargs=dict(epochs=1, model_lr=0.0, basis_lr=2e-4, basis_steps=1,
                              inv_weight=1.0, data_weight=0.0, print_every=0),
        )
    return dict(
        model_kwargs=dict(modes=24, width=96, n_layers=4),
        train_kwargs=dict(epochs=3000, lr=6e-4, bs=24, weight_decay=7.5e-5,
                          sg_weight=0.25, val_fraction=0.10, print_every=300,
                          const_weight=0.15, weighted_loss=True),
        n_basis=20,
        joint_kwargs=dict(epochs=1, model_lr=0.0, basis_lr=2e-4, basis_steps=1,
                          inv_weight=1.0, data_weight=0.0, print_every=0),
    )


def evaluate(result, dataset, out_dir: Path):
    analytic = dataset["meta"]["analytic_lambdas"]          # 0, -1, -2, ...
    lam = result.lam
    n_check = 6
    print("\n=== OU BSO evaluation (target lambda_n = -n) ===")
    errors = []
    for n in range(n_check):
        target = analytic[n]
        err = np.min(np.abs(lam - target))
        nearest = lam[np.argmin(np.abs(lam - target))]
        errors.append(err)
        print(f"  mode {n}: target {target:+.3f}  nearest {nearest.real:+.6f}{nearest.imag:+.4f}j  |err| {err:.4e}")
    print(f"  head-{n_check} max abs err = {max(errors):.4e}")
    print(f"  r_eff = {result.r_eff}/{result.psi.shape[0]}, cond(G) = {result.cond_G:.3e}")

    apply_publication_rcparams()
    style = PUBLICATION_STYLE
    x = dataset["meta"]["x"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=OU_EVALUATION_FIGURE["figsize"],
        dpi=style["dpi"],
        squeeze=True,
        sharex=False,
        sharey=False,
        layout=None,
        facecolor=style["figure_face_color"],
    )
    fig.set_layout_engine(None)
    axes[0].scatter(
        lam.real,
        lam.imag,
        marker="x",
        s=2.0 * style["marker_size"] ** 2,
        color="tab:blue",
        linewidths=style["marker_edge_width"],
        alpha=1.0,
        zorder=3,
        label="learned",
    )
    axes[0].scatter(
        analytic[:n_check],
        np.zeros(n_check),
        marker="o",
        s=2.0 * style["marker_size"] ** 2,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=style["marker_edge_width"],
        alpha=1.0,
        zorder=2,
        label=r"analytic $-n$",
    )
    axes[0].set_xlabel(
        r"Re($\lambda$)",
        fontsize=style["axis_label_size"],
        fontweight=style["font_weight"],
        labelpad=style["axis_label_pad"],
        color=style["text_color"],
    )
    axes[0].set_ylabel(
        r"Im($\lambda$)",
        fontsize=style["axis_label_size"],
        fontweight=style["font_weight"],
        labelpad=style["axis_label_pad"],
        color=style["text_color"],
    )
    axes[0].set_xlim(-8.5, 0.5)
    axes[0].set_title(
        "OU generator spectrum",
        fontsize=style["panel_title_size"],
        fontweight=style["title_weight"],
        pad=style["panel_title_pad"],
        color=style["text_color"],
    )
    for j in range(min(4, result.eigenfunctions.shape[0])):
        ef = result.eigenfunctions[j].real
        ef = ef / (np.max(np.abs(ef)) + 1e-12)
        axes[1].plot(
            x,
            ef,
            color=f"C{j}",
            linestyle="-",
            linewidth=style["line_width"],
            marker=None,
            alpha=1.0,
            zorder=2,
            label=rf"$\varphi_{j}$",
        )
    axes[1].set_xlabel(
        r"$x$",
        fontsize=style["axis_label_size"],
        fontweight=style["font_weight"],
        labelpad=style["axis_label_pad"],
        color=style["text_color"],
    )
    axes[1].set_title(
        "Leading BSO eigenfunctions",
        fontsize=style["panel_title_size"],
        fontweight=style["title_weight"],
        pad=style["panel_title_pad"],
        color=style["text_color"],
    )
    for axis in axes:
        style_axis(axis, grid_axis="both", grid_visible=True, minor_ticks=False)
        axis.legend(
            loc="best",
            ncol=OU_EVALUATION_FIGURE["legend_ncol"],
            frameon=style["legend_frameon"],
            fancybox=style["legend_fancybox"],
            framealpha=style["legend_framealpha"],
            borderaxespad=style["legend_borderaxespad"],
            borderpad=style["legend_borderpad"],
            labelspacing=style["legend_labelspacing"],
            handlelength=style["legend_handlelength"],
            handleheight=style["legend_handleheight"],
            handletextpad=style["legend_handletextpad"],
            columnspacing=style["legend_columnspacing"],
            fontsize=style["legend_size"],
        )
    apply_physical_layout(fig, OU_EVALUATION_FIGURE)
    save_figure_pair(
        fig,
        out_dir / f"ou_spectrum_{result.solver_name}.pdf",
        dpi=style["dpi"],
    )
    plt.close(fig)

    np.savez(out_dir / f"ou_result_{result.solver_name}.npz",
             lam=lam, mu=result.mu, K=result.K, errors=np.asarray(errors))
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--solver", default="solver_edmd.py")
    ap.add_argument("--operator", default="fno", help="fno|sfno|pino|deeponet|kernel")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    plt.switch_backend("Agg")

    out_dir = FIGURES_DIR / "ou"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = None if args.quick else RESULTS_DIR / "ou"
    dataset = generate_kbe_dataset(
        "ou", n_samples=96 if args.quick else 384, seed=args.seed,
        cache_path=None if ckpt_dir is None else str(ckpt_dir / "dataset.pkl"))
    cfg = build_config(args.quick)
    cfg["fixed_basis"] = hermite_fixed_columns(dataset, k=cfg["n_basis"] - 1)
    result = run_bso_pipeline(dataset, cfg, solver_file=args.solver, operator=args.operator,
                                  checkpoint_dir=ckpt_dir)
    evaluate(result, dataset, out_dir)


if __name__ == "__main__":
    main()
