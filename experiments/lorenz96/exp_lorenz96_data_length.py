"""Lorenz-96 increasing-data experiment: BNO vs. Data EDMD as N crosses M.

Standard chaotic Lorenz-96 (``K=40`` sites, ``F=8``). The target operator
dimension is ``M=40`` -- the full observed state -- so a plain data EDMD needs
at least 40 transition pairs, while the BNO (``run_bso_pipeline``) forms its
40x40 Galerkin matrix from the frozen FNO acting on the dictionary over the
sampling measure and is defined at every ``N``.

Usage:
    python -m experiments.lorenz96.exp_lorenz96_data_length [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

from experiments._data_length import plot_two_versions, run_sweep, save_result
from kbe_observation_data import generate_kbe_dataset
from project_paths import FIGURES_DIR, RESULTS_DIR

SYSTEM = "lorenz96"
FAMILY = "nonlinear_sensor"
TARGET_M = 40
K_SITES = 40
FORCING = 8.0
N_TEST_ORIGINS = 128
REG_ALPHA = 1e-4
HORIZON = 150             # tau-steps for the valid-prediction-time rollout
QUICK_HORIZON = 20

LENGTHS = (10, 20, 30, 40, 60, 100, 200, 400)
SEEDS = (2026, 2027, 2028, 2029, 2030)
QUICK_LENGTHS = (20, 60)
QUICK_SEEDS = (2026,)


def _model_kwargs(quick: bool) -> dict:
    if quick:
        return dict(modes=8, width=32, n_layers=2)
    return dict(modes=12, width=64, n_layers=4)


def _train_kwargs(quick: bool) -> dict:
    common = dict(lr=6e-4, bs=24, weight_decay=7.5e-5, sg_weight=0.25,
                  val_fraction=0.15, print_every=0)
    common["epochs"] = 30 if quick else 500
    return common


def _joint_kwargs(quick: bool) -> dict:
    return dict(epochs=5 if quick else 40, model_lr=0.0, basis_lr=2e-4,
                basis_steps=2, inv_weight=2.0, data_weight=1.0, print_every=0)


def build_full_dataset(quick: bool, seed: int = 2026) -> dict:
    long_traj = 12_000 if quick else 30_000
    cache = None if quick else str(RESULTS_DIR / "l96_data_length"
                                   / "dataset_k40.pkl")
    return generate_kbe_dataset(
        SYSTEM, n_samples=0, seed=seed, cache_path=cache,
        K=K_SITES, F=FORCING, long_traj_steps=long_traj)


def _training_pairs(readings: np.ndarray, n_pairs: int) -> tuple:
    """First ``n_pairs`` pairs spread across the training window."""
    n_available = readings.shape[1] - 2
    starts = np.unique(np.linspace(0, n_available - 1, int(n_pairs),
                                   dtype=int))
    return (readings[:, starts].T.copy(),
            readings[:, starts + 1].T.copy(),
            readings[:, starts + 2].T.copy())


def _test_traj(readings: np.ndarray, n_origins: int, horizon: int) -> np.ndarray:
    """Held-out (n_origins, horizon+1, K) consecutive-state trajectories."""
    last = readings.shape[1] - int(horizon) - 1
    origins = np.unique(np.linspace(0, last, int(n_origins), dtype=int))
    return np.stack([readings[:, origins + s].T
                     for s in range(int(horizon) + 1)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    lengths = QUICK_LENGTHS if args.quick else LENGTHS
    seeds = QUICK_SEEDS if args.quick else SEEDS
    horizon = QUICK_HORIZON if args.quick else HORIZON
    dataset = build_full_dataset(args.quick)
    train_readings = np.asarray(dataset["meta"]["train_readings"][FAMILY],
                                dtype=np.float64)
    test_readings = np.asarray(dataset["meta"]["test_readings"][FAMILY],
                               dtype=np.float64)
    test_traj = _test_traj(test_readings, N_TEST_ORIGINS, horizon)

    def make_dataset_n(n_pairs: int, seed: int) -> dict:
        u0, uT, u2T = _training_pairs(train_readings, n_pairs)
        return dict(dataset, u0=u0, uT=uT, u2T=u2T)

    def cfg_builder(n_pairs: int, seed: int) -> dict:
        return dict(
            seed=int(seed), operator="fno",
            model_kwargs=_model_kwargs(args.quick),
            train_kwargs=_train_kwargs(args.quick), n_basis=TARGET_M,
            joint_kwargs=_joint_kwargs(args.quick), reg_alpha=1e-5)

    ckpt_root = (None if args.quick
                 else RESULTS_DIR / "l96_data_length" / "checkpoints")
    result = run_sweep(
        system=SYSTEM, make_dataset_n=make_dataset_n, test_traj=test_traj,
        weights=np.asarray(dataset["weights"], dtype=np.float64),
        kind=str(dataset["kind"]), cfg_builder=cfg_builder,
        target_M=TARGET_M, lengths=lengths, seeds=seeds,
        checkpoint_root=ckpt_root, reg_alpha=REG_ALPHA)

    out_results = RESULTS_DIR / "l96_data_length"
    out_figures = FIGURES_DIR / "lorenz96"
    save_result(result, out_results / "l96_data_length_results.npz")
    # Valid-prediction-time is not plotted here: at tau=0.02 every method's
    # horizon collapses into {0.00 ... 0.12}, six steps of resolution, so the
    # metric cannot separate them. ``plot_valid_time`` is still used by the
    # ENSO and EEG drivers, whose step sizes give it room.
    pdfs = plot_two_versions(result, out_figures, "Lorenz-96",
                             title_suffix="(K=40, F=8)")
    print(f"[lorenz96:data-length] figures: {[str(p) for p in pdfs]}")


if __name__ == "__main__":
    main()
