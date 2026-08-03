"""ENSO increasing-data experiment: BNO vs. Data EDMD as N crosses M=40 EOFs.

The operator dimension is ``M = 40`` leading weighted EOFs of the development
record (a fixed measurement definition, like the Lorenz-96 sites). The BNO
(``run_bso_pipeline``) forms its 40x40 Galerkin matrix from a monthly-field FNO
acting on the fixed EOF dictionary over the area measure, so it is defined at
every ``N``; a plain data EDMD needs at least 40 monthly transition pairs.

Usage:
    python -m experiments.enso.exp_enso_data_length [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

from experiments._data_length import (plot_two_versions, plot_valid_time,
                                       run_sweep, save_result)
from experiments.enso.enso_data import (build_grid_dataset, fit_eofs,
                                         load_ersstv5)
from project_paths import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR

SYSTEM = "enso"
TARGET_M = 40
REG_ALPHA = 1e-4
HORIZON = 12              # months for the valid-prediction-time rollout
QUICK_HORIZON = 4
N_TEST_ORIGINS = 120
DATA_PATH = PROJECT_ROOT.parent / "sst.mnmean.nc"

LENGTHS = (10, 20, 30, 40, 80, 160, 320)
SEEDS = (2026, 2027, 2028, 2029, 2030)
QUICK_LENGTHS = (20, 80)
QUICK_SEEDS = (2026,)


def _model_kwargs(quick: bool) -> dict:
    if quick:
        return dict(modes=6, width=16, n_layers=2)
    return dict(modes=12, width=48, n_layers=4)


def _train_kwargs(quick: bool) -> dict:
    return dict(epochs=30 if quick else 400, lr=6e-4, bs=16,
                weight_decay=7.5e-5, sg_weight=0.25, val_fraction=0.15,
                print_every=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    lengths = QUICK_LENGTHS if args.quick else LENGTHS
    seeds = QUICK_SEEDS if args.quick else SEEDS
    horizon = QUICK_HORIZON if args.quick else HORIZON
    data = load_ersstv5(DATA_PATH)
    eof = fit_eofs(data, data.dev_end, TARGET_M)
    fixed_columns = eof.fixed_columns()

    # Held-out (n_origins, horizon+1, p) trajectories over the primary window.
    last = data.primary_end - horizon - 1
    origins = np.unique(np.linspace(data.dev_end, last, N_TEST_ORIGINS,
                                    dtype=int))
    test_traj = np.stack([data.anomalies[origins + s]
                          for s in range(horizon + 1)], axis=1)

    def make_dataset_n(n_pairs: int, seed: int) -> dict:
        start = data.dev_end - 2 - int(n_pairs)
        return build_grid_dataset(data, max(start, 0), data.dev_end - 2)

    def cfg_builder(n_pairs: int, seed: int) -> dict:
        return dict(
            seed=int(seed), operator="fno",
            model_kwargs=_model_kwargs(args.quick),
            train_kwargs=_train_kwargs(args.quick), n_basis=TARGET_M,
            fixed_basis=fixed_columns,
            joint_kwargs=dict(epochs=1, model_lr=0.0, basis_lr=2e-4,
                              basis_steps=1, print_every=0),
            reg_alpha=1e-5)

    ckpt_root = (None if args.quick
                 else RESULTS_DIR / "enso_data_length" / "checkpoints")
    result = run_sweep(
        system=SYSTEM, make_dataset_n=make_dataset_n, test_traj=test_traj,
        weights=data.weights, kind="grid2d", cfg_builder=cfg_builder,
        target_M=TARGET_M, lengths=lengths, seeds=seeds,
        checkpoint_root=ckpt_root, reg_alpha=REG_ALPHA)

    save_result(result, RESULTS_DIR / "enso_data_length"
                / "enso_data_length_results.npz")
    pdfs = plot_two_versions(result, FIGURES_DIR / "enso", "ENSO",
                             title_suffix="(M=40 EOFs)")
    vt = plot_valid_time(result, FIGURES_DIR / "enso", "ENSO",
                         title_suffix="(M=40 EOFs)")
    print(f"[enso:data-length] figures: {[str(p) for p in (*pdfs, vt)]}")


if __name__ == "__main__":
    main()
