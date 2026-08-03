"""EEG increasing-data experiment: BNO vs. Data EDMD as N crosses M.

Resting-state EEGMMIDB record (subject S029, run R02), band-passed 8-30 Hz with
a causal common-average reference by ``eeg_resting_data`` -- the same
preprocessing decisions as the rest of the EEG code. The forecast state is the
instantaneous 64-electrode vector advanced by ``LAG_SAMPLES``.

The BNO is ``edmd_bso_method.run_bso_pipeline`` (the measure-based Galerkin
projection), used identically to the Lorenz-96 and ENSO studies so that the
``N < M`` conclusion is comparable across systems. Note this differs from
``eeg_operator_bso``, whose Galerkin transition is built from the observed
states and is therefore rank-limited by ``N`` like a plain data EDMD -- it
cannot form an ``M``-dimensional operator when ``N < M``.

Target dimensions ``M in {20, 64}`` (a 20-mode reduction and the full electrode
state). Requires ``mne`` to read/download the EDF on first use.

Usage:
    python -m experiments.eeg.exp_eeg_data_length [--quick] [--subject N --run R]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

from experiments._data_length import (plot_two_versions, plot_valid_time,
                                       run_sweep, save_result)
from experiments.eeg.eeg_resting_data import (HISTORY_SAMPLES, LAG_SAMPLES,
                                               SAMPLING_RATE_HZ,
                                               load_resting_eeg_run)
from project_paths import DATA_DIR, FIGURES_DIR, RESULTS_DIR

# On-disk EEGMMIDB records live under data/physionet.org/files/...; missing
# subjects (e.g. S029) are downloaded here by mne on first use.
EEG_DATA_ROOT = DATA_DIR / "physionet.org"

SYSTEM = "eeg"
N_ELECTRODES = 64
TARGET_DIMS = (20, 64)
REG_ALPHA = 1e-4
TRAIN_SECONDS = 15.0
HORIZON = 40             # lag-steps for the valid-prediction-time rollout
QUICK_HORIZON = 10

LENGTHS = (10, 20, 30, 40, 60, 100, 200)
SEEDS = (2026, 2027, 2028, 2029, 2030)
QUICK_LENGTHS = (20, 80)
QUICK_SEEDS = (2026,)


def _model_kwargs(quick: bool) -> dict:
    if quick:
        return dict(modes=6, width=16, n_layers=2)
    return dict(modes=12, width=64, n_layers=4)


def _train_kwargs(quick: bool) -> dict:
    return dict(epochs=30 if quick else 500, lr=6e-4, bs=24,
                weight_decay=7.5e-5, sg_weight=0.25, val_fraction=0.15,
                print_every=0)


def _prepare(subject: int, run: int, horizon: int) -> dict:
    resting = load_resting_eeg_run(subject, run, data_root=EEG_DATA_ROOT,
                                   download=True)
    signal = np.asarray(resting.data, dtype=np.float64)          # (64, T)
    split = int(TRAIN_SECONDS * SAMPLING_RATE_HZ)
    train = signal[:, HISTORY_SAMPLES:split]
    mean = train.mean(axis=1, keepdims=True)
    scale = train.std(axis=1, keepdims=True) + 1e-6
    normalized = (signal - mean) / scale
    lag = int(LAG_SAMPLES)
    train_origins = np.arange(HISTORY_SAMPLES, split - 2 * lag)
    test_origins = np.arange(split, signal.shape[1] - (int(horizon) + 1) * lag)
    return dict(normalized=normalized, lag=lag, train_origins=train_origins,
                test_origins=test_origins)


def _dataset_from_origins(normalized, origins, lag) -> dict:
    grid = np.linspace(0.0, 1.0, N_ELECTRODES, endpoint=False)
    return dict(
        system="eeg", kind="sensor1d", T=float(lag / SAMPLING_RATE_HZ),
        u0=normalized[:, origins].T.copy(),
        uT=normalized[:, origins + lag].T.copy(),
        u2T=normalized[:, origins + 2 * lag].T.copy(),
        fno_grid=grid, basis_coords=grid[:, None],
        weights=np.full(N_ELECTRODES, 1.0 / N_ELECTRODES),
        meta=dict(n_electrodes=N_ELECTRODES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--subject", type=int, default=29)
    parser.add_argument("--run", type=int, default=2)
    args = parser.parse_args()

    lengths = QUICK_LENGTHS if args.quick else LENGTHS
    seeds = QUICK_SEEDS if args.quick else SEEDS
    horizon = QUICK_HORIZON if args.quick else HORIZON
    prep = _prepare(args.subject, args.run, horizon)
    normalized, lag = prep["normalized"], prep["lag"]
    train_origins, test_origins = prep["train_origins"], prep["test_origins"]
    weights = np.full(N_ELECTRODES, 1.0 / N_ELECTRODES)
    n_test = 256 if not args.quick else 64
    test_sel = np.unique(np.linspace(0, len(test_origins) - 1, n_test,
                                     dtype=int))
    test_o = test_origins[test_sel]
    test_traj = np.stack([normalized[:, test_o + s * lag].T
                          for s in range(horizon + 1)], axis=1)

    def make_dataset_n(n_pairs: int, seed: int) -> dict:
        sel = np.unique(np.linspace(0, len(train_origins) - 1, int(n_pairs),
                                    dtype=int))
        return _dataset_from_origins(normalized, train_origins[sel], lag)

    for target_M in TARGET_DIMS:
        def cfg_builder(n_pairs: int, seed: int, _M=target_M) -> dict:
            return dict(
                seed=int(seed), operator="fno",
                model_kwargs=_model_kwargs(args.quick),
                train_kwargs=_train_kwargs(args.quick), n_basis=int(_M),
                joint_kwargs=dict(epochs=5 if args.quick else 40,
                                  model_lr=0.0, basis_lr=2e-4, basis_steps=2,
                                  inv_weight=2.0, data_weight=1.0,
                                  print_every=0),
                reg_alpha=1e-5)

        ckpt_root = (None if args.quick else RESULTS_DIR / "eeg_data_length"
                     / f"M{target_M}" / "checkpoints")
        result = run_sweep(
            system=f"{SYSTEM}_M{target_M}", make_dataset_n=make_dataset_n,
            test_traj=test_traj, weights=weights, kind="sensor1d",
            cfg_builder=cfg_builder, target_M=target_M, lengths=lengths,
            seeds=seeds, checkpoint_root=ckpt_root, reg_alpha=REG_ALPHA)
        save_result(result, RESULTS_DIR / "eeg_data_length"
                    / f"eeg_data_length_M{target_M}.npz")
        label = f"EEG (M={target_M})"
        suffix = f"S{args.subject:03d}R{args.run:02d}"
        pdfs = plot_two_versions(result, FIGURES_DIR / "eeg", label,
                                 title_suffix=suffix)
        vt = plot_valid_time(result, FIGURES_DIR / "eeg", label,
                             title_suffix=suffix)
        print(f"[eeg:data-length] M={target_M} figures: "
              f"{[str(p) for p in (*pdfs, vt)]}")


if __name__ == "__main__":
    main()
