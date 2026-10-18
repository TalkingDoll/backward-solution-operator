"""Multivariate SST forecasting with SST-only verification.

The learned daily state contains SST plus ten contemporaneous ocean/air
fields.  All channels are advanced autonomously, so forecasts never read
future observations of the auxiliary variables.  The reported target remains
SST throughout:

The SST experiment keeps the raw one-day FNO input domain unchanged, but
fits the BSO in a training-only structured anomaly representation:

    multichannel raw state
      -> empirical calendar climatology / auxiliary harmonics
      -> per-channel EOF coefficients (64 SST, 4 per auxiliary channel)
      -> rank-96 joint projection
      -> observed/FNO one-step blended target (alpha_data = 0.6)
      -> real-Schur stabilized BSO recurrence (radius <= 0.9975)
      -> SST-only decoding and verification

The reusable model implementation is in
``experiments/sst/sst_model.py``. This is the sole command entry point: use
``python -m experiments.sst.exp_sst evaluation ...`` for the reproducible
workflow, or its regular options for raw FNO/SFNO preparation and training.

The archive spans 1993-01-01 through 2022-12-31.  Validation fits through
2016-12-31 and evaluates 2017--2019; the reported evaluation fits through
2019-12-31 and evaluates 2020--2022.  Running this file without ``--train``
only validates and summarizes prepared data.  Training is deliberately
opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# SST preparation and training use BLAS. Default to half the logical CPUs
# (at most eight) so the experiment does not monopolize the workstation.
# An explicitly supplied environment setting always takes precedence.
_CPU_THREAD_CAP = max(1, min(8, (os.cpu_count() or 4) // 2))
_configured_thread_caps = []
for _thread_env in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS"):
    _existing = os.environ.get(_thread_env)
    try:
        _limit = (
            _CPU_THREAD_CAP if _existing is None
            else max(1, min(_CPU_THREAD_CAP, int(_existing))))
    except ValueError:
        _limit = _CPU_THREAD_CAP
    os.environ[_thread_env] = str(_limit)
    _configured_thread_caps.append(_limit)

import numpy as np
import torch

_TORCH_THREAD_CAP = min(_configured_thread_caps)
torch.set_num_threads(_TORCH_THREAD_CAP)
try:
    torch.set_num_interop_threads(max(1, min(2, _TORCH_THREAD_CAP)))
except RuntimeError:
    # Harmless when an embedding process initialized the inter-op pool first.
    pass

from edmd_bso_method import (
    _pace_cuda,
    _pacing_start,
    apply_operator,
    build_operator,
    device,
    set_seed,
    train_operator,
)
from kbe_observation_data import (
    calendar_day_index,
    generate_kbe_dataset,
)
from project_paths import PROJECT_ROOT
from experiments.sst.sst_model import _operator_fingerprint


MODULE_ROOT = PROJECT_ROOT
RECORD_START = "1993-01-01"
RECORD_END = "2022-12-31"
VALIDATION_FIT_END = "2017-01-01"
EVALUATION_FIT_END = "2020-01-01"
VALIDATION_EVALUATION_END = "2020-01-01"
VALIDATION_TRAIN_PAIRS = 8764
EVALUATION_TRAIN_PAIRS = 9859

# Forecast skill and the longer numerical-stability stress test use separate
# lead grids and interpretation rules.
DEFAULT_EVALUATION_PROFILE = "short-medium-90d"
STABILITY_EVALUATION_PROFILE = "rollout-stability-540d"
EVALUATION_PROFILES = {
    DEFAULT_EVALUATION_PROFILE: {
        "purpose": "forecast_skill",
        "n_steps": 90,
        "n_origins": 96,
        "lead_grid": "daily",
        "evaluation_end_margin": 1,
        "summary_bands": (
            ("short", 1, 10),
            ("medium", 11, 30),
            ("extended", 31, 90),
        ),
    },
    STABILITY_EVALUATION_PROFILE: {
        "purpose": "rollout_stability",
        "n_steps": 540,
        "n_origins": 48,
        "lead_grid": "sparse",
        "evaluation_end_margin": 2,
        "summary_bands": (),
    },
}


def build_lead_grid(profile, n_steps=None):
    """Return the exact lead grid for an SST evaluation profile."""
    if profile not in EVALUATION_PROFILES:
        raise ValueError(
            f"unknown SST evaluation profile {profile!r}; expected one of "
            f"{sorted(EVALUATION_PROFILES)}")
    spec = EVALUATION_PROFILES[profile]
    horizon = int(spec["n_steps"] if n_steps is None else n_steps)
    if horizon <= 0:
        raise ValueError("n_steps must be positive")
    if spec["lead_grid"] == "daily":
        return np.arange(1, horizon + 1, dtype=int)
    if spec["lead_grid"] == "sparse":
        return np.unique(np.concatenate([
            np.arange(1, min(30, horizon) + 1, dtype=int),
            np.arange(35, min(180, horizon) + 1, 5, dtype=int),
            np.arange(190, horizon + 1, 10, dtype=int),
        ]))
    raise ValueError(f"unsupported lead grid {spec['lead_grid']!r}")


def select_origins(
        train_end, evaluation_end, n_steps, n_origins, origin_offset=0,
        evaluation_end_margin=1):
    """Choose uniform held-out origins without crossing the evaluation end."""
    train_end = int(train_end)
    evaluation_end = int(evaluation_end)
    n_steps = int(n_steps)
    n_origins = int(n_origins)
    origin_offset = int(origin_offset)
    evaluation_end_margin = int(evaluation_end_margin)
    if n_steps <= 0 or n_origins <= 0:
        raise ValueError("n_steps and n_origins must be positive")
    if evaluation_end_margin <= 0:
        raise ValueError("evaluation_end_margin must be positive")
    if not train_end < evaluation_end:
        raise ValueError("evaluation_end must be after train_end")
    last_origin = (
        evaluation_end - train_end - n_steps - evaluation_end_margin)
    if origin_offset < 0 or last_origin < origin_offset:
        raise ValueError("evaluation window is too short for the requested horizon")
    if n_origins == 1:
        return np.asarray([origin_offset], dtype=int)
    if last_origin - origin_offset + 1 < n_origins:
        raise ValueError(
            "not enough distinct launch dates for the requested n_origins")
    origins = np.linspace(
        origin_offset, last_origin, n_origins, dtype=int)
    if np.any(np.diff(origins) <= 0):
        raise ValueError("origin selection did not produce distinct launch dates")
    return origins


def _clim_for_times(dataset, times):
    """Return training-only month/day SST climatology in physical units."""
    meta = dataset["meta"]
    if meta.get("climatology_calendar") != "month-day-leap-v1":
        raise ValueError("SST climatology calendar metadata is incompatible")
    day = calendar_day_index(times)
    return np.asarray(dataset["meta"]["climatology"], dtype=np.float64)[day]


def _weighted_rmse(pred_phys, truth_phys, weights_2d):
    """Latitude/ocean-weighted RMSE in kelvin."""
    w = np.asarray(weights_2d, dtype=np.float64)[None, :, :]
    return float(np.sqrt(
        np.sum(w * (pred_phys - truth_phys) ** 2) / pred_phys.shape[0]))


def _weighted_mae(pred_phys, truth_phys, weights_2d):
    """Latitude/ocean-weighted MAE in kelvin."""
    w = np.asarray(weights_2d, dtype=np.float64)[None, :, :]
    return float(
        np.sum(w * np.abs(pred_phys - truth_phys)) / pred_phys.shape[0])


def _weighted_acc(pred_phys, truth_phys, clim_phys, weights_2d):
    """Latitude/ocean-weighted centered anomaly correlation coefficient."""
    w = np.asarray(weights_2d, dtype=np.float64)[None, :, :]
    pa = pred_phys - clim_phys
    ta = truth_phys - clim_phys
    pa = pa - np.sum(w * pa, axis=(1, 2), keepdims=True)
    ta = ta - np.sum(w * ta, axis=(1, 2), keepdims=True)
    num = float(np.sum(w * pa * ta))
    pred_energy = float(np.sum(w * pa ** 2))
    truth_energy = float(np.sum(w * ta ** 2))
    if pred_energy <= 1e-24 or truth_energy <= 1e-24:
        return float("nan")
    den = float(np.sqrt(pred_energy * truth_energy))
    return float(np.clip(num / den, -1.0, 1.0))


def _weighted_bias(pred_phys, truth_phys, weights_2d):
    """Latitude/ocean-weighted mean error in kelvin."""
    w = np.asarray(weights_2d, dtype=np.float64)[None, :, :]
    return float(np.sum(w * (pred_phys - truth_phys)) / pred_phys.shape[0])


def _weighted_std_ratio(pred_phys, truth_phys, clim_phys, weights_2d):
    """Forecast/observed SST-anomaly standard-deviation ratio."""
    w = np.asarray(weights_2d, dtype=np.float64)[None, :, :]
    pa = pred_phys - clim_phys
    ta = truth_phys - clim_phys
    sp = float(np.sqrt(np.sum(w * pa ** 2) / pred_phys.shape[0]))
    st = float(np.sqrt(np.sum(w * ta ** 2) / truth_phys.shape[0]))
    return sp / (st + 1e-12)


def damped_persistence_gamma(dataset, leads=(7, 14, 30),
                             tail_days=400, n_cases=64):
    """Choose persistence decay on the tail of the fitting period only."""
    meta = dataset["meta"]
    train_end = int(meta["train_end"])
    fields = np.asarray(meta["fields"], dtype=np.float64)
    times = meta["times"]
    mean, std = float(meta["mean"]), float(meta["std"])
    weights_2d = np.asarray(meta["weights_2d"], dtype=np.float64)
    lo = max(0, train_end - int(tail_days))
    starts = np.linspace(
        lo, train_end - max(leads) - 2, int(n_cases), dtype=int)
    phys0 = fields[starts] * std + mean
    anom0 = phys0 - _clim_for_times(dataset, times[starts])
    best_gamma, best_error = 1.0, np.inf
    for gamma in np.linspace(0.90, 1.00, 21):
        total = 0.0
        for lead in leads:
            climatology = _clim_for_times(dataset, times[starts + int(lead)])
            prediction = climatology + float(gamma) ** int(lead) * anom0
            truth = fields[starts + int(lead)] * std + mean
            total += _weighted_rmse(
                prediction, truth, weights_2d)
        if total < best_error:
            best_error, best_gamma = total, float(gamma)
    return best_gamma


def horizon_summary(leads, metrics, bands):
    """Return metric means for explicit inclusive lead bands."""
    leads = np.asarray(leads, dtype=int)
    selected_bands = tuple(
        (str(name), (leads >= int(first)) & (leads <= int(last)))
        for name, first, last in bands)
    rows = []
    for band, mask in selected_bands:
        if not np.any(mask):
            continue
        for method in metrics["rmse"]:
            rows.append(dict(
                band=band,
                method=method,
                rmse=float(metrics["rmse"][method][mask].mean()),
                mae=(
                    float(metrics["mae"][method][mask].mean())
                    if "mae" in metrics else np.nan),
                acc=float(metrics["acc"][method][mask].mean()),
                abs_bias=float(np.abs(metrics["bias"][method][mask]).mean()),
                std_ratio=float(metrics["std_ratio"][method][mask].mean()),
            ))
    return rows


def rmse_skill_scores(metrics, baselines=("raw_persistence", "climatology")):
    """Return lead-wise RMSE skill, ``1 - model_rmse / baseline_rmse``."""
    rmse = metrics["rmse"]
    result = {}
    for baseline in baselines:
        if baseline not in rmse:
            continue
        reference = np.asarray(rmse[baseline], dtype=np.float64)
        for method, values in rmse.items():
            if method == baseline:
                continue
            result[f"{method}_vs_{baseline}"] = (
                1.0 - np.asarray(values, dtype=np.float64)
                / np.maximum(reference, 1e-12))
    return result


def build_config(quick: bool) -> dict:
    """FNO/SFNO training configuration."""
    if quick:
        return dict(
            model_kwargs=dict(modes=(8, 16), width=48, n_layers=4),
            train_kwargs=dict(
                epochs=40, lr=6e-4, bs=12, weight_decay=1e-5,
                sg_weight=0.25, val_fraction=0.10, print_every=10,
                weighted_loss=True, max_duty_cycle=0.8),
        )
    return dict(
        model_kwargs=dict(modes=(12, 24), width=96, n_layers=4),
        train_kwargs=dict(
            epochs=180, lr=5e-4, bs=12, weight_decay=1e-5,
            sg_weight=0.25, val_fraction=0.10, print_every=25,
            weighted_loss=True, mix_augment=0.5, max_duty_cycle=0.8),
    )


def train_or_load_baseline(
        dataset, operator_kind, model_kwargs, train_kwargs,
        checkpoint_path=None, force_retrain=False, seed=2026,
        training_stop_path=None):
    """Train or load a neural-operator baseline on identical daily pairs."""
    set_seed(int(seed))
    kind = dataset["kind"]
    u0 = torch.as_tensor(
        np.asarray(dataset["u0"]).reshape(len(dataset["u0"]), -1),
        dtype=torch.float32, device=device)
    uT = torch.as_tensor(
        np.asarray(dataset["uT"]).reshape(len(dataset["uT"]), -1),
        dtype=torch.float32, device=device)
    u2T = torch.as_tensor(
        np.asarray(dataset["u2T"]).reshape(len(dataset["u2T"]), -1),
        dtype=torch.float32, device=device)
    grid = torch.as_tensor(
        dataset["fno_grid"], dtype=torch.float32, device=device)
    weights = torch.as_tensor(
        dataset["weights"], dtype=torch.float32, device=device)
    model = build_operator(operator_kind, dataset, model_kwargs).to(device)
    metadata = dict(
        operator=str(operator_kind),
        model_kwargs=dict(model_kwargs),
        system=dataset.get("system"),
        n_points=int(u0.shape[1]),
        state_shape=tuple(
            int(value) for value in np.asarray(dataset["u0"]).shape[1:]
        ),
        channels=tuple(dataset["meta"]["channel_names"]),
        data_fingerprint=dataset["meta"]["data_fingerprint"],
        training_fingerprint=dataset["meta"]["training_fingerprint"],
        seed=int(seed),
        train_kwargs=dict(train_kwargs),
    )
    progress_path = (
        None if checkpoint_path is None
        else Path(checkpoint_path).with_name(
            f".{str(operator_kind).lower()}.training.pt"))
    if force_retrain and progress_path is not None:
        progress_path.unlink(missing_ok=True)
        progress_path.with_suffix(
            progress_path.suffix + ".tmp").unlink(missing_ok=True)
    if (
            checkpoint_path is not None
            and Path(checkpoint_path).exists()
            and not force_retrain):
        payload = torch.load(
            checkpoint_path, map_location=device, weights_only=False)
        if payload.get("meta") == metadata:
            # neuraloperator may serialize a non-parameter ``_metadata`` key.
            model.load_state_dict(payload["state"], strict=False)
            model.eval()
            if progress_path is not None:
                progress_path.unlink(missing_ok=True)
            print(
                f"[sst] loaded {operator_kind} checkpoint "
                f"{checkpoint_path}")
            return model
        print(f"[sst] {operator_kind} checkpoint differs; retraining")

    kwargs = dict(train_kwargs)
    loss_weights = weights if kwargs.pop("weighted_loss", False) else None
    print(
        f"[sst] training {operator_kind} on {u0.shape[0]} daily samples ...")
    information = train_operator(
        model, u0, uT, u2T, grid, kind,
        seed=int(seed),
        loss_weights=loss_weights,
        progress_path=progress_path,
        progress_metadata=metadata,
        training_stop_path=training_stop_path,
        **kwargs)
    model.eval()
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_checkpoint = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".tmp")
        torch.save(
            dict(
                state={
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                meta=metadata,
                best_val=information.get("best_val"),
                completed_epochs=information.get("completed_epochs"),
                stopped_early=information.get("stopped_early", False),
            ),
            temporary_checkpoint,
        )
        temporary_checkpoint.replace(checkpoint_path)
        if progress_path is not None:
            progress_path.unlink(missing_ok=True)
        print(
            f"[sst] saved {operator_kind} checkpoint to {checkpoint_path}")
    return model


def batch_state_operator_rollout(
        result, dataset, origins, n_steps, operator=None, bs=12,
        origin_base=None, max_duty_cycle=0.8,
        divergence_threshold=1e3,
        invalidity_policy="threshold_or_nonfinite",
        return_diagnostics=False):
    """Roll out the full internal state but retain only SST for scoring.

    ``threshold_or_nonfinite`` invalidates a case at its first nonfinite value
    or large-state threshold exceedance. ``nonfinite_only`` records threshold
    exceedances diagnostically but keeps every finite state in the rollout.
    """
    divergence_threshold = float(divergence_threshold)
    if divergence_threshold <= 0:
        raise ValueError("divergence_threshold must be positive")
    allowed_policies = {"threshold_or_nonfinite", "nonfinite_only"}
    if invalidity_policy not in allowed_policies:
        raise ValueError(
            f"invalidity_policy must be one of {sorted(allowed_policies)}")
    meta = dataset["meta"]
    train_end = (
        int(meta["train_end"]) if origin_base is None else int(origin_base))
    states = np.asarray(meta["state_fields"], dtype=np.float32)
    C, H, W = states.shape[1:]
    absolute = train_end + np.asarray(origins, dtype=int)
    current = torch.as_tensor(
        states[absolute].reshape(len(absolute), -1),
        dtype=torch.float32, device=device)
    grid = torch.as_tensor(result.grid, dtype=torch.float32, device=device)
    operator = (
        result.operator_pure if operator is None else operator).eval()
    traj = [states[absolute, 0].reshape(len(absolute), -1).astype(np.float64)]
    n_cases = len(absolute)
    active = torch.ones(n_cases, dtype=torch.bool, device=device)
    first_nonfinite = np.full(n_cases, -1, dtype=int)
    first_threshold_exceedance = np.full(n_cases, -1, dtype=int)
    first_divergence = np.full(n_cases, -1, dtype=int)
    valid_fraction = [1.0]
    max_abs_state = [
        float(torch.max(torch.abs(current)).detach().cpu())]
    with torch.no_grad():
        for step in range(1, int(n_steps) + 1):
            operator_input = torch.where(
                active[:, None], current, torch.zeros_like(current))
            nxt = []
            for start in range(0, operator_input.shape[0], int(bs)):
                pacing_started = _pacing_start(max_duty_cycle)
                nxt.append(apply_operator(
                    operator, operator_input[start:start + int(bs)],
                    grid, result.kind))
                _pace_cuda(pacing_started, max_duty_cycle)
            raw_next = torch.cat(nxt)
            raw_finite = torch.isfinite(raw_next).all(dim=1)
            finite_abs = torch.nan_to_num(
                torch.abs(raw_next), nan=float("inf"),
                posinf=float("inf"), neginf=float("inf"))
            raw_peak = finite_abs.amax(dim=1)
            max_abs_state.append(float(raw_peak.max().detach().cpu()))

            threshold_ok = raw_peak <= divergence_threshold
            nonfinite_now = active & ~raw_finite
            threshold_now = active & raw_finite & ~threshold_ok
            invalid_now = (
                nonfinite_now
                if invalidity_policy == "nonfinite_only"
                else nonfinite_now | threshold_now)
            nonfinite_np = nonfinite_now.detach().cpu().numpy()
            threshold_np = threshold_now.detach().cpu().numpy()
            invalid_np = invalid_now.detach().cpu().numpy()
            first_nonfinite[
                nonfinite_np & (first_nonfinite < 0)] = step
            first_threshold_exceedance[
                threshold_np & (first_threshold_exceedance < 0)] = step
            first_divergence[
                invalid_np & (first_divergence < 0)] = step
            active = active & ~invalid_now
            current = torch.where(
                active[:, None], raw_next, torch.zeros_like(raw_next))

            sst = (
                current.reshape(-1, C, H, W)[:, 0]
                .reshape(current.shape[0], -1).cpu().numpy().astype(np.float64))
            sst[~active.detach().cpu().numpy()] = np.nan
            traj.append(sst)
            valid_fraction.append(
                float(active.float().mean().detach().cpu()))

    diagnostics = {
        "numerical_policy": str(invalidity_policy),
        "divergence_threshold": divergence_threshold,
        "first_nonfinite": first_nonfinite,
        "first_threshold_exceedance": first_threshold_exceedance,
        "first_divergence": first_divergence,
        "valid_fraction": np.asarray(valid_fraction, dtype=np.float64),
        "max_abs_state": np.asarray(max_abs_state, dtype=np.float64),
    }
    trajectory = np.asarray(traj)
    if return_diagnostics:
        return trajectory, diagnostics
    return trajectory


def _sst_dataset_view(dataset):
    """SST-only view used by climatology and persistence baselines."""
    view = dict(dataset)
    meta = dict(dataset["meta"])
    meta.update(
        fields=np.asarray(meta["sst_fields"]),
        mean=float(meta["channel_mean"][0]),
        std=float(meta["channel_std"][0]),
    )
    view["meta"] = meta
    return view


def _restore_normalized_sst(dataset, values, absolute_indices):
    """Reshape normalized SST values to the physical grid."""
    meta = dataset["meta"]
    height, width = np.asarray(meta["weights_2d"]).shape
    return np.asarray(values, dtype=np.float64).reshape(
        len(absolute_indices), height, width)


def baseline_prediction_errors(
        result, dataset, sfno_model=None, n_steps=None, n_origins=None,
        origin_offset=0, leads=None,
        evaluation_end=None, max_duty_cycle=0.8,
        evaluation_profile=DEFAULT_EVALUATION_PROFILE,
        divergence_threshold=1e3,
        return_origins=False, return_diagnostics=False):
    """Evaluate neural-operator and statistical SST reference methods."""
    if evaluation_profile not in EVALUATION_PROFILES:
        raise ValueError(f"unknown evaluation profile {evaluation_profile!r}")
    spec = EVALUATION_PROFILES[evaluation_profile]
    n_steps = int(spec["n_steps"] if n_steps is None else n_steps)
    n_origins = int(spec["n_origins"] if n_origins is None else n_origins)
    meta = dataset["meta"]
    train_end, n_time = int(meta["train_end"]), int(meta["n_time"])
    H, W = meta["weights_2d"].shape
    evaluation_end = n_time if evaluation_end is None else int(evaluation_end)
    if not train_end < evaluation_end <= n_time:
        raise ValueError(
            f"evaluation_end={evaluation_end} must lie in "
            f"({train_end}, {n_time}]")
    origins = select_origins(
        train_end, evaluation_end, n_steps, n_origins,
        origin_offset=origin_offset,
        evaluation_end_margin=spec["evaluation_end_margin"])
    absolute = train_end + origins
    if leads is None:
        leads = build_lead_grid(evaluation_profile, n_steps=n_steps)
    leads = np.asarray([v for v in leads if v <= n_steps], dtype=int)

    rollout_diagnostics = {}
    fno_output = batch_state_operator_rollout(
        result, dataset, origins, n_steps, origin_base=train_end,
        max_duty_cycle=max_duty_cycle,
        divergence_threshold=divergence_threshold,
        return_diagnostics=return_diagnostics)
    if return_diagnostics:
        fno_traj, rollout_diagnostics["fno"] = fno_output
    else:
        fno_traj = fno_output
    sfno_traj = None
    if sfno_model is not None:
        sfno_output = batch_state_operator_rollout(
            result, dataset, origins, n_steps, operator=sfno_model,
            origin_base=train_end, max_duty_cycle=max_duty_cycle,
            divergence_threshold=divergence_threshold,
            return_diagnostics=return_diagnostics)
        if return_diagnostics:
            sfno_traj, rollout_diagnostics["sfno"] = sfno_output
        else:
            sfno_traj = sfno_output
    sst_view = _sst_dataset_view(dataset)
    gamma = damped_persistence_gamma(sst_view)
    fields = np.asarray(
        sst_view["meta"]["fields"], dtype=np.float64)
    mean, std = float(meta["channel_mean"][0]), float(meta["channel_std"][0])
    times = meta["times"]
    weights_2d = np.asarray(meta["weights_2d"], dtype=np.float64)
    phys0 = fields[absolute] * std + mean
    anom0 = phys0 - _clim_for_times(sst_view, times[absolute])

    methods = [
        "fno", "damped_persistence", "raw_persistence", "climatology"
    ]
    if sfno_traj is not None:
        methods.insert(1, "sfno")
    metrics = {
        metric: {method: np.zeros(len(leads)) for method in methods}
        for metric in ("rmse", "mae", "acc", "bias", "std_ratio")
    }
    for li, lead in enumerate(leads):
        truth = fields[absolute + lead] * std + mean
        clim = _clim_for_times(sst_view, times[absolute + lead])
        preds = {
            "fno": _restore_normalized_sst(
                dataset, fno_traj[lead], absolute + lead) * std + mean,
            "damped_persistence":
                clim + (gamma ** int(lead)) * anom0,
            "raw_persistence":
                np.broadcast_to(phys0, truth.shape),
            "climatology": clim,
        }
        if sfno_traj is not None:
            preds["sfno"] = (
                _restore_normalized_sst(
                    dataset, sfno_traj[lead], absolute + lead) * std + mean)
        for name, pred in preds.items():
            metrics["rmse"][name][li] = _weighted_rmse(
                pred, truth, weights_2d)
            metrics["mae"][name][li] = _weighted_mae(
                pred, truth, weights_2d)
            metrics["acc"][name][li] = _weighted_acc(
                pred, truth, clim, weights_2d)
            metrics["bias"][name][li] = _weighted_bias(
                pred, truth, weights_2d)
            metrics["std_ratio"][name][li] = _weighted_std_ratio(
                pred, truth, clim, weights_2d)
    if return_origins and return_diagnostics:
        return leads, metrics, origins, rollout_diagnostics
    if return_origins:
        return leads, metrics, origins
    if return_diagnostics:
        return leads, metrics, rollout_diagnostics
    return leads, metrics


def baseline_rollout_stability(
        result, dataset, sfno_model=None, n_steps=540, n_origins=48,
        origin_offset=0, leads=None,
        evaluation_end=None, max_duty_cycle=0.8,
        divergence_threshold=1e3,
        invalidity_policy="nonfinite_only"):
    """Run the FNO/SFNO portion of the 540-day stability stress test.

    BSO is evaluated by the structured model runner and merged into the same
    artifact. No climatology or persistence scores are produced here.
    """
    meta = dataset["meta"]
    train_end, n_time = int(meta["train_end"]), int(meta["n_time"])
    evaluation_end = n_time if evaluation_end is None else int(evaluation_end)
    origins = select_origins(
        train_end, evaluation_end, n_steps, n_origins,
        origin_offset=origin_offset,
        evaluation_end_margin=EVALUATION_PROFILES[
            STABILITY_EVALUATION_PROFILE]["evaluation_end_margin"])
    absolute = train_end + origins
    if leads is None:
        leads = build_lead_grid(
            STABILITY_EVALUATION_PROFILE, n_steps=n_steps)
    leads = np.asarray([v for v in leads if 0 < v <= n_steps], dtype=int)

    fno_traj, fno_diagnostics = batch_state_operator_rollout(
        result, dataset, origins, n_steps, origin_base=train_end,
        max_duty_cycle=max_duty_cycle,
        divergence_threshold=divergence_threshold,
        invalidity_policy=invalidity_policy, return_diagnostics=True)
    trajectories = {"fno": fno_traj}
    diagnostics = {"fno": fno_diagnostics}
    if sfno_model is not None:
        sfno_traj, sfno_diagnostics = batch_state_operator_rollout(
            result, dataset, origins, n_steps, operator=sfno_model,
            origin_base=train_end, max_duty_cycle=max_duty_cycle,
            divergence_threshold=divergence_threshold,
            invalidity_policy=invalidity_policy,
            return_diagnostics=True)
        trajectories["sfno"] = sfno_traj
        diagnostics["sfno"] = sfno_diagnostics

    sst_view = _sst_dataset_view(dataset)
    fields = np.asarray(sst_view["meta"]["fields"], dtype=np.float64)
    mean, std = float(meta["channel_mean"][0]), float(meta["channel_std"][0])
    weights_2d = np.asarray(meta["weights_2d"], dtype=np.float64)
    height, width = weights_2d.shape
    methods = tuple(trajectories)
    stability = {
        metric: {method: np.full(len(leads), np.nan, dtype=np.float64)
                 for method in methods}
        for metric in (
            "rmse", "rmse_valid", "valid_fraction", "max_abs_sst")
    }

    for li, lead in enumerate(leads):
        truth = fields[absolute + lead] * std + mean
        for method, trajectory in trajectories.items():
            pred = _restore_normalized_sst(
                dataset, trajectory[lead], absolute + lead) * std + mean
            pred = np.asarray(pred, dtype=np.float64).reshape(
                len(origins), height, width)
            case_valid = np.all(np.isfinite(pred), axis=(1, 2))
            case_valid &= (
                diagnostics[method]["first_divergence"] < 0) | (
                diagnostics[method]["first_divergence"] > lead)

            n_valid = int(np.sum(case_valid))
            stability["valid_fraction"][method][li] = (
                n_valid / float(len(origins)))
            if n_valid:
                stability["rmse_valid"][method][li] = _weighted_rmse(
                    pred[case_valid], truth[case_valid], weights_2d)
                stability["max_abs_sst"][method][li] = float(
                    np.max(np.abs(pred[case_valid])))
            if n_valid == len(origins):
                stability["rmse"][method][li] = _weighted_rmse(
                    pred, truth, weights_2d)

    return leads, stability, origins, diagnostics


def prepare_references():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true",
        help="Opt in to training. Without this flag only data readiness is checked.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--evaluation-profile",
        choices=(DEFAULT_EVALUATION_PROFILE,),
        default=DEFAULT_EVALUATION_PROFILE,
        help=(
            "Forecast-skill protocol evaluating every lead from 1--90 days."))
    parser.add_argument(
        "--n-origins", type=int,
        help="Override the forecast-skill profile's number of launch dates.")
    parser.add_argument(
        "--skip-stability", action="store_true",
        help="Skip the FNO/SFNO portion of the 540-day stability stress test.")
    parser.add_argument(
        "--stability-origins", type=int,
        help="Override the 540-day stability profile's number of launch dates.")
    parser.add_argument(
        "--divergence-threshold", type=float, default=1e3,
        help=(
            "Absolute normalized-state large-value threshold; the 1--90-day "
            "protocol invalidates a rollout at this threshold, while the "
            "540-day protocol records it diagnostically and terminates only "
            "at NaN/Inf; default 1000."))
    parser.add_argument(
        "--checkpoint-source", type=Path,
        help=(
            "Reuse compatible FNO/SFNO checkpoints from this directory while "
            "writing reference metrics to a new run."))
    parser.add_argument(
        "--validation", action="store_true",
        help=(
            "Fit 1993--2016 and evaluate 2017--2019. The 2020--2022 "
            "test period remains held out."))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-name")
    parser.add_argument("--train-pairs", type=int)
    parser.add_argument("--sst-weight", type=float, default=0.5)
    parser.add_argument("--eof-sst", type=int, default=64)
    parser.add_argument("--eof-aux", type=int, default=4)
    parser.add_argument("--width", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--modes-lat", type=int)
    parser.add_argument("--modes-lon", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sg-weight", type=float)
    parser.add_argument("--mix-augment", type=float)
    parser.add_argument(
        "--gpu-duty-cycle", type=float, default=0.8,
        help="Average CUDA duty-cycle cap in (0,0.8]; default 0.8.")
    parser.add_argument(
        "--skip-sfno", action="store_true",
        help="Skip the matched SFNO comparison; included by default.")
    parser.add_argument(
        "--force-sfno-retrain", action="store_true",
        help="Ignore an existing matching SFNO checkpoint.")
    parser.add_argument(
        "--training-stop-file", type=Path,
        help=(
            "Cooperative stop marker. At the end of the current epoch, "
            "save the best validation state and continue to evaluation."))
    args = parser.parse_args()

    if not 0.0 < args.gpu_duty_cycle <= 0.8:
        parser.error("--gpu-duty-cycle must lie in (0, 0.8]")
    if not 0.0 < args.sst_weight < 1.0:
        parser.error("--sst-weight must lie in (0, 1)")
    for flag, value in (
            ("--train-pairs", args.train_pairs),
            ("--eof-sst", args.eof_sst),
            ("--eof-aux", args.eof_aux),
            ("--width", args.width),
            ("--layers", args.layers),
            ("--modes-lat", args.modes_lat),
            ("--modes-lon", args.modes_lon),
            ("--epochs", args.epochs),
            ("--batch-size", args.batch_size),
            ("--n-origins", args.n_origins),
            ("--stability-origins", args.stability_origins)):
        if value is not None and int(value) <= 0:
            parser.error(f"{flag} must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.sg_weight is not None and args.sg_weight < 0:
        parser.error("--sg-weight must be nonnegative")
    if args.mix_augment is not None and not 0.0 <= args.mix_augment <= 1.0:
        parser.error("--mix-augment must lie in [0, 1]")
    if args.divergence_threshold <= 0:
        parser.error("--divergence-threshold must be positive")
    if args.run_name is not None:
        allowed = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        if not args.run_name or any(ch not in allowed for ch in args.run_name):
            parser.error("--run-name may contain only letters, digits, '-', '_' and '.'")

    train_pairs = (
        int(args.train_pairs) if args.train_pairs is not None
        else (
            4096 if args.quick
            else (
                VALIDATION_TRAIN_PAIRS
                if args.validation else EVALUATION_TRAIN_PAIRS)))
    fit_end = VALIDATION_FIT_END if args.validation else EVALUATION_FIT_END
    overrides = dict(
        record_start=RECORD_START,
        record_end=RECORD_END,
        train_end_date=fit_end,
        n_train_times=train_pairs,
        sst_channel_weight=float(args.sst_weight),
        n_eof_sst=int(args.eof_sst),
        n_eof_aux=int(args.eof_aux),
    )
    dataset = generate_kbe_dataset(
        "sst",
        seed=args.seed,
        # The source Zarr is already the persistent data cache.  Avoid a
        # dataset pickle here: it cannot safely represent different date
        # boundaries or sample counts used by validation and evaluation fits.
        cache_path=None,
        **overrides)
    meta = dataset["meta"]
    protocol = "validation" if args.validation else "evaluation"
    print(
        f"[sst] ready: {meta['n_time']} daily states, "
        f"{meta['n_channels']} channels {meta['channel_names']}, "
        f"target='sst', protocol={protocol}, "
        f"fingerprint={meta['data_fingerprint']}")
    if not args.train:
        print("[sst] data-only readiness check complete; no training started.")
        return

    config = build_config(args.quick)
    config["seed"] = int(args.seed)
    model_kwargs = config["model_kwargs"]
    default_modes = tuple(model_kwargs["modes"])
    model_kwargs["modes"] = (
        int(args.modes_lat) if args.modes_lat is not None else default_modes[0],
        int(args.modes_lon) if args.modes_lon is not None else default_modes[1],
    )
    if args.width is not None:
        model_kwargs["width"] = int(args.width)
    if args.layers is not None:
        model_kwargs["n_layers"] = int(args.layers)
    train_kwargs = config["train_kwargs"]
    if args.epochs is not None:
        train_kwargs["epochs"] = int(args.epochs)
    if args.learning_rate is not None:
        train_kwargs["lr"] = float(args.learning_rate)
    if args.batch_size is not None:
        train_kwargs["bs"] = int(args.batch_size)
    if args.sg_weight is not None:
        train_kwargs["sg_weight"] = float(args.sg_weight)
    if args.mix_augment is not None:
        train_kwargs["mix_augment"] = float(args.mix_augment)
    train_kwargs["max_duty_cycle"] = float(args.gpu_duty_cycle)

    forecast_spec = EVALUATION_PROFILES[args.evaluation_profile]
    forecast_steps = (
        min(30, int(forecast_spec["n_steps"]))
        if args.quick else int(forecast_spec["n_steps"]))
    forecast_origins = (
        int(args.n_origins) if args.n_origins is not None
        else (
            min(48, int(forecast_spec["n_origins"]))
            if args.quick else int(forecast_spec["n_origins"])))
    stability_enabled = not bool(args.skip_stability)
    stability_spec = EVALUATION_PROFILES[STABILITY_EVALUATION_PROFILE]
    stability_steps = (
        180 if args.quick else int(stability_spec["n_steps"]))
    stability_origins = (
        int(args.stability_origins)
        if args.stability_origins is not None
        else (
            min(24, int(stability_spec["n_origins"]))
            if args.quick else int(stability_spec["n_origins"])))
    origin_offset = 0
    times_daily = np.asarray(meta["times"]).astype("datetime64[D]")
    evaluation_end_date = (
        VALIDATION_EVALUATION_END if args.validation else "2023-01-01")
    evaluation_end = int(np.searchsorted(
        times_daily, np.datetime64(evaluation_end_date), side="left"))
    if evaluation_end != (
            9861 if args.validation else int(meta["n_time"])):
        raise ValueError(
            f"unexpected evaluation boundary {evaluation_end} for "
            f"{evaluation_end_date}")
    run_manifest = dict(
        protocol=protocol,
        quick=bool(args.quick),
        seed=int(args.seed),
        purpose="neural_operator_and_statistical_references",
        neural_operators=(
            ["fno"] if args.skip_sfno else ["fno", "sfno"]
        ),
        dataset_overrides=overrides,
        model_kwargs=config["model_kwargs"],
        train_kwargs=config["train_kwargs"],
        state_representation="training-window-channel-zscore",
        training_fingerprint=meta["training_fingerprint"],
        artifact_schema="sst-reference-preparation",
        evaluation=dict(
            forecast_skill=dict(
                profile=args.evaluation_profile,
                purpose=forecast_spec["purpose"],
                n_steps=forecast_steps,
                n_origins=forecast_origins,
                origin_offset=origin_offset,
                evaluation_end_margin=forecast_spec[
                    "evaluation_end_margin"],
                evaluation_end=evaluation_end,
                leads=build_lead_grid(
                    args.evaluation_profile,
                    n_steps=forecast_steps).tolist(),
                summary_bands=forecast_spec["summary_bands"],
                compared_methods=(
                    [
                        "fno", "damped_persistence",
                        "raw_persistence", "climatology",
                    ]
                    if args.skip_sfno
                    else [
                        "fno", "sfno", "damped_persistence",
                        "raw_persistence", "climatology",
                    ]
                ),
                neural_numerical_policy="strict",
                invalidity_policy="threshold_or_nonfinite",
                divergence_threshold=float(args.divergence_threshold)),
            rollout_stability=dict(
                enabled=stability_enabled,
                profile=STABILITY_EVALUATION_PROFILE,
                purpose=stability_spec["purpose"],
                n_steps=stability_steps,
                n_origins=stability_origins,
                origin_offset=origin_offset,
                evaluation_end_margin=stability_spec[
                    "evaluation_end_margin"],
                evaluation_end=evaluation_end,
                leads=(
                    build_lead_grid(
                        STABILITY_EVALUATION_PROFILE,
                        n_steps=stability_steps).tolist()
                    if stability_enabled else []),
                compared_methods=(
                    ["fno"] if args.skip_sfno else ["fno", "sfno"]
                ),
                metric="strict_full_cohort_physical_sst_rmse",
                invalidity_policy="nonfinite_only",
                large_state_diagnostic_threshold=float(
                    args.divergence_threshold))),
    )
    checkpoint_source = None
    if args.checkpoint_source is not None:
        checkpoint_source = args.checkpoint_source.expanduser().resolve()
        if not checkpoint_source.is_dir():
            parser.error(
                f"--checkpoint-source is not a directory: {checkpoint_source}")
        if args.force_sfno_retrain:
            parser.error(
                "--force-sfno-retrain cannot be combined with "
                "--checkpoint-source")
        required_checkpoints = ["fno.pt"]
        if not args.skip_sfno:
            required_checkpoints.append("sfno.pt")
        missing = [
            name for name in required_checkpoints
            if not (checkpoint_source / name).is_file()]
        if missing:
            parser.error(
                "--checkpoint-source is missing required checkpoint(s): "
                + ", ".join(missing))
        run_manifest["checkpoint_source"] = str(checkpoint_source)
    manifest_hash_payload = {
        key: value for key, value in run_manifest.items()
        if key != "checkpoint_source"}
    manifest_text = json.dumps(
        manifest_hash_payload, sort_keys=True, separators=(",", ":"))
    run_id = args.run_name or hashlib.sha256(
        manifest_text.encode("utf-8")).hexdigest()[:16]
    cache_dir = (
        MODULE_ROOT / "results" / "sst" / "references"
        / f"{protocol}-{run_id}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir = checkpoint_source or cache_dir
    print(f"[sst] run id: {run_id}")
    if checkpoint_source is not None:
        print(f"[sst] reusing compatible checkpoints from {model_dir}")

    fno_model = train_or_load_baseline(
        dataset,
        "fno",
        config["model_kwargs"],
        config["train_kwargs"],
        checkpoint_path=model_dir / "fno.pt",
        seed=int(args.seed),
        training_stop_path=args.training_stop_file,
    )
    result = SimpleNamespace(
        operator_pure=fno_model,
        grid=np.asarray(dataset["fno_grid"], dtype=np.float64),
        kind=dataset["kind"],
    )
    sfno_model = None
    if not args.skip_sfno:
        sfno_model = train_or_load_baseline(
            dataset,
            "sfno",
            config["model_kwargs"],
            config["train_kwargs"],
            checkpoint_path=model_dir / "sfno.pt",
            force_retrain=bool(args.force_sfno_retrain),
            seed=int(args.seed),
            training_stop_path=args.training_stop_file,
        )
    reference_fingerprint = _operator_fingerprint(fno_model)
    leads, metrics, skill_origins, skill_diagnostics = baseline_prediction_errors(
        result, dataset, sfno_model=sfno_model,
        n_steps=forecast_steps,
        n_origins=forecast_origins,
        origin_offset=origin_offset,
        evaluation_end=evaluation_end,
        max_duty_cycle=args.gpu_duty_cycle,
        evaluation_profile=args.evaluation_profile,
        divergence_threshold=args.divergence_threshold,
        return_origins=True,
        return_diagnostics=True)
    skill_scores = rmse_skill_scores(metrics)
    skill_absolute = int(meta["train_end"]) + skill_origins
    skill_diagnostic_arrays = {
        f"rollout_{method}_{name}": value
        for method, diagnostic in skill_diagnostics.items()
        for name, value in diagnostic.items()
        if name not in {"numerical_policy"}
    }
    np.savez(
        cache_dir / "forecast_metrics.npz",
        leads=leads,
        fingerprint=reference_fingerprint,
        artifact_schema="sst-forecast-references",
        evaluation_profile=args.evaluation_profile,
        protocol=protocol,
        train_end=int(meta["train_end"]),
        evaluation_end=evaluation_end,
        origin_offset=origin_offset,
        origins=skill_origins,
        origin_dates=times_daily[skill_absolute],
        last_target_dates=times_daily[skill_absolute + forecast_steps],
        n_origins=forecast_origins,
        n_steps=forecast_steps,
        summary_bands_json=json.dumps(
            forecast_spec["summary_bands"], separators=(",", ":")),
        all_metrics_finite=bool(all(
            np.all(np.isfinite(values))
            for by_method in metrics.values()
            for values in by_method.values())),
        **{
            f"rmse_skill_{name}": values
            for name, values in skill_scores.items()
        },
        **skill_diagnostic_arrays,
        **{
            f"{metric}_{method}": values
            for metric, by_method in metrics.items()
            for method, values in by_method.items()
        },
    )

    stability_summary = None
    if stability_enabled:
        (stability_leads, stability_metrics, stability_launches,
         stability_diagnostics) = baseline_rollout_stability(
            result, dataset, sfno_model=sfno_model,
            n_steps=stability_steps,
            n_origins=stability_origins,
            origin_offset=origin_offset,
            evaluation_end=evaluation_end,
            max_duty_cycle=args.gpu_duty_cycle,
            divergence_threshold=args.divergence_threshold,
            invalidity_policy="nonfinite_only")
        stability_absolute = int(meta["train_end"]) + stability_launches
        stability_diagnostic_arrays = {
            f"rollout_{method}_{name}": value
            for method, diagnostic in stability_diagnostics.items()
            for name, value in diagnostic.items()
            if name not in {"numerical_policy"}
        }
        np.savez(
            cache_dir / "stability_metrics.npz",
            leads=stability_leads,
            fingerprint=reference_fingerprint,
            artifact_schema="sst-rollout-stability-references",
            evaluation_profile=STABILITY_EVALUATION_PROFILE,
            protocol=protocol,
            train_end=int(meta["train_end"]),
            evaluation_end=evaluation_end,
            origin_offset=origin_offset,
            origins=stability_launches,
            origin_dates=times_daily[stability_absolute],
            last_target_dates=times_daily[
                stability_absolute + stability_steps],
            n_origins=stability_origins,
            n_steps=stability_steps,
            divergence_threshold=float(args.divergence_threshold),
            invalidity_policy="nonfinite_only",
            **stability_diagnostic_arrays,
            **{
                f"{metric}_{method}": values
                for metric, by_method in stability_metrics.items()
                for method, values in by_method.items()
            },
        )
        stability_summary = (
            stability_leads, stability_metrics, stability_diagnostics)

    (cache_dir / "run_manifest.json").write_text(
        json.dumps(
            {**run_manifest, "run_id": run_id,
             "fno_operator_fingerprint": reference_fingerprint},
            indent=2, sort_keys=True),
        encoding="utf-8")
    print("[sst] SST reference forecast-skill summary")
    for row in horizon_summary(
            leads, metrics, bands=forecast_spec["summary_bands"]):
        print(
            f"  {row['band']:<8s} {row['method']:<15s} "
            f"RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} "
            f"ACC={row['acc']:.3f} "
            f"|bias|={row['abs_bias']:.4f} std-ratio={row['std_ratio']:.3f}")
    if stability_summary is not None:
        stability_leads, stability_metrics, stability_diagnostics = (
            stability_summary)
        print(
            "[sst] 540-day nonfinite-only neural-operator "
            "rollout-stability summary")
        for method in stability_metrics["rmse"]:
            rmse = np.asarray(
                stability_metrics["rmse"][method], dtype=np.float64)
            valid_fraction = np.asarray(
                stability_metrics["valid_fraction"][method],
                dtype=np.float64)
            complete = np.isfinite(rmse) & np.isclose(valid_fraction, 1.0)
            complete_indices = np.flatnonzero(complete)
            if complete_indices.size:
                last_index = int(complete_indices[-1])
                last_rmse = float(rmse[last_index])
                rmse_text = (
                    f"{last_rmse:.4e}"
                    if last_rmse >= 1.0e4 else f"{last_rmse:.4f}"
                )
                last_text = (
                    f"day {int(stability_leads[last_index])}, "
                    f"RMSE={rmse_text}"
                )
            else:
                last_text = "none"
            first = np.asarray(
                stability_diagnostics[method]["first_nonfinite"], dtype=int)
            first = first[first >= 0]
            first_text = "none" if not len(first) else str(int(first.min()))
            print(
                f"  {method:<15s} last complete-cohort point="
                f"{last_text}, "
                f"first-nonfinite={first_text}")


def main() -> None:
    """Dispatch reference preparation or the reproducible SST workflow."""
    if len(sys.argv) > 1 and sys.argv[1] == "evaluation":
        from experiments.sst.sst_evaluation import run_evaluation

        run_evaluation(sys.argv[2:])
        return
    prepare_references()


if __name__ == "__main__":
    main()
