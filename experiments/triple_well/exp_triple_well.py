"""Triple-well experiment: spectrum vs a finite-difference reference.

Usage:
    python -m experiments.triple_well.exp_triple_well [--quick] [--solver solver_edmd.py|solver_resdmd.py]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from kbe_observation_data import generate_kbe_dataset, compute_triple_well_fd_reference_lambdas
from edmd_bso_method import run_bso_pipeline
from project_paths import RESULTS_DIR


def match_reference_spectrum(
    lam: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match the constant and nonconstant reference modes one-to-one."""
    lam = np.asarray(lam)
    reference = np.asarray(reference)
    constant_idx = int(np.argmin(np.abs(lam)))
    nonconstant_idx = np.flatnonzero(np.abs(lam) > 1.0e-8)
    if len(nonconstant_idx) < len(reference) - 1:
        raise ValueError("not enough nonconstant learned modes for matching")
    cost = np.abs(
        reference[1:, None] - lam[nonconstant_idx][None, :]
    )
    ref_rows, learned_cols = linear_sum_assignment(cost)
    matched_by_ref = {
        int(row + 1): int(nonconstant_idx[col])
        for row, col in zip(ref_rows, learned_cols)
    }
    matched_idx = np.asarray(
        [constant_idx]
        + [matched_by_ref[n] for n in range(1, len(reference))],
        dtype=np.int64,
    )
    errors = np.abs(lam[matched_idx] - reference)
    return matched_idx, errors


def build_config(quick: bool) -> dict:
    """Triple-well configuration: FNO with 16x16
    modes, width 96, 5 layers, trained 500 epochs (lr 1e-3, bs 16, sg 0.30);
    dictionary of 49 functions (constant + 48 learned by an MLP with hidden
    256, depth 5, num_freq 4) refined JOINTLY with the operator for 250
    epochs (model_lr 1.5e-5, basis_lr 5e-5, 10 basis steps per epoch,
    inv_weight 10) using the plain EDMD invariance loss. The final spectrum
    uses the frozen Phase-A operator over three steps, reversible
    symmetrization, and uniform L2(dx) geometry. Checkpoints:
    <operator>.pt + galerkin.pt under results/triple_well.
    """
    if quick:
        return dict(
            model_kwargs=dict(modes=10, width=48, n_layers=4),
            train_kwargs=dict(epochs=200, lr=1e-3, bs=8, weight_decay=8e-5,
                              sg_weight=0.30, val_fraction=0.15, print_every=40),
            n_basis=24,
            reg_alpha=5e-5,
            projection_steps=1,
            training_projection_steps=1,
            projection_operator="joint",
            symmetrize_K=False,
            basis_kwargs=dict(hidden=160, depth=4, num_freq=4),
            joint_kwargs=dict(epochs=50, model_lr=1.5e-5, basis_lr=5e-5,
                              basis_steps=10, inv_weight=10.0, data_weight=1.0,
                              print_every=10),
            bso_file="galerkin.pt",
        )
    return dict(
        model_kwargs=dict(modes=16, width=96, n_layers=5),
        train_kwargs=dict(epochs=500, lr=1e-3, bs=16, weight_decay=8e-5,
                          sg_weight=0.30, val_fraction=0.15, print_every=100),
        n_basis=49,
        reg_alpha=1e-8,
        projection_steps=3,
        training_projection_steps=1,
        projection_operator="phase_a",
        symmetrize_K=True,
        basis_kwargs=dict(hidden=256, depth=5, num_freq=4),
        joint_kwargs=dict(epochs=250, model_lr=1.5e-5, basis_lr=5e-5,
                          basis_steps=10, inv_weight=10.0, data_weight=1.0,
                          print_every=25),
        bso_file="galerkin.pt",
    )


def evaluate(result, dataset, result_dir: Path | None, quick: bool):
    n_ref = 4 if quick else 6
    print(f"\n[triple_well] computing FD reference spectrum (n_eigs={n_ref}) ...")
    ref = compute_triple_well_fd_reference_lambdas(
        n_eigs=n_ref, N_ref=96 if quick else 128, tol=1e-8, maxiter=8000)
    lam = result.lam

    print("=== Triple-well BSO evaluation (vs FD reference) ===")
    matched_idx, errors = match_reference_spectrum(lam, ref)
    print(
        f"  mode 0: ref {ref[0].real:+.6f}  "
        f"matched {lam[matched_idx[0]].real:+.6f}"
        f"{lam[matched_idx[0]].imag:+.4f}j  |err| {errors[0]:.4e}"
    )
    for n in range(1, len(ref)):
        j = matched_idx[n]
        print(
            f"  mode {n}: ref {ref[n].real:+.6f}  "
            f"matched {lam[j].real:+.6f}{lam[j].imag:+.4f}j  "
            f"|err| {errors[n]:.4e}"
        )
    print(
        f"  head-{len(ref)} MAE = {np.mean(errors):.4e}, "
        f"max = {np.max(errors):.4e}"
    )
    print(f"  r_eff = {result.r_eff}/{result.psi.shape[0]}, cond(G) = {result.cond_G:.3e}")
    if result_dir is not None:
        np.savez(
            result_dir / "spectrum_metrics.npz",
            reference_lambdas=ref,
            matched_lambdas=lam[matched_idx],
            matched_indices=matched_idx,
            absolute_errors=np.asarray(errors),
            mean_absolute_error=float(np.mean(errors)),
            maximum_absolute_error=float(np.max(errors)),
            effective_horizon=(
                int(build_config(False)["projection_steps"])
                * float(dataset["T"])
            ),
        )
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--solver", default="solver_edmd.py")
    ap.add_argument("--operator", default="fno", help="fno|sfno|pino|deeponet|kernel")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    ckpt_dir = None if args.quick else RESULTS_DIR / "triple_well"
    overrides = dict(N=48) if args.quick else {}
    dataset = generate_kbe_dataset(
        "triple_well", n_samples=32 if args.quick else 64, seed=args.seed,
        cache_path=None if ckpt_dir is None else str(ckpt_dir / "dataset.pkl"),
        **overrides)
    result = run_bso_pipeline(dataset, build_config(args.quick), solver_file=args.solver,
                                  operator=args.operator, checkpoint_dir=ckpt_dir)
    evaluate(result, dataset, ckpt_dir, args.quick)


if __name__ == "__main__":
    main()
