"""Locked SST fitting and evaluation protocol.

The formal forecast protocol is daily 1--90 days.  The separate 540-day
protocol is a learned-model stability stress test only.  Pinned FNO, SFNO,
persistence, and climatology arrays are reused; BSO is evaluated on the
held-out 2020--2022 period.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from edmd_bso_method import build_operator, device
from project_paths import PROJECT_ROOT, RESULTS_DIR
from experiments.sst.exp_sst import (
    EVALUATION_FIT_END,
    EVALUATION_TRAIN_PAIRS,
    RECORD_END,
    RECORD_START,
    _clim_for_times,
    build_config,
)
from kbe_observation_data import generate_kbe_dataset
from experiments.sst.sst_model import (
    MODEL_VERSION,
    _operator_fingerprint,
    build_structured_bso,
    rollout_structured_bso,
    validate_fit_window_model_state,
    validate_model_state_integrity,
)


REPO_ROOT = PROJECT_ROOT
RUN_ROOT = REPO_ROOT / "results" / "sst" / "runs"
EVALUATION_DATA_FINGERPRINT = "13c88b85d7836d658475"
EVALUATION_TRAINING_FINGERPRINT = "326c250ccbccb19daf0c"
EVALUATION_FNO_SHA256 = (
    "6c0545ed03dcdc88d6fe1cd2bb3d38edcfe3538b232542dc93792bf2285cee32"
)
EVALUATION_FNO_OPERATOR_FINGERPRINT = "bada83be03e5482edcb8"
BASELINE_FORECAST_SHA256 = (
    "886f6b760095f3925420d3c113eddf2e44b195d7ab0f3f2841439675794b25b7"
)
BASELINE_STABILITY_SHA256 = (
    "f95ca6be7c6675eb43447a0513576115313f134d0b5dd61f131b557ef81bb43a"
)
FORECAST_PROFILE = "short-medium-90d"
STABILITY_PROFILE = "rollout-stability-540d"


class RunLog:
    """Small JSON-lines logger for the reproducible evaluation runner."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        payload = {"event": event, "created_at_utc": _utc_now(), **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _strict_parameter_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state must be a mapping")
    if not all(isinstance(key, str) for key in state):
        raise TypeError("checkpoint state keys must be strings")
    return dict(state)


def _rollout_bso_selected_leads(
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    absolute_origins: np.ndarray,
    leads: np.ndarray,
    *,
    operator_key: str,
    residual_mode: str,
) -> np.ndarray:
    selected = np.asarray(leads, dtype=int)
    if selected.ndim != 1 or len(selected) == 0 or np.any(selected < 1):
        raise ValueError("leads must be a nonempty one-dimensional positive grid")
    rollout = rollout_structured_bso(
        dataset,
        model,
        np.asarray(absolute_origins, dtype=int),
        int(selected.max()),
        operator_key=operator_key,
        residual_mode=residual_mode,  # type: ignore[arg-type]
        origin_base=0,
        output_units="kelvin",
    )
    return np.asarray(rollout[selected], dtype=np.float64)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    """Return a portable repository-relative artifact path."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _require_sha256(path: Path, expected: str, label: str) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _load_json_pinned(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    _require_sha256(path, expected_sha256, "selection lock")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_status") != "locked":
        raise ValueError("selection lock is not locked")
    return payload


def _safe_run_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = RUN_ROOT.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"output must be below the SST run root {root}: {resolved}"
        ) from exc
    if relative == Path("."):
        raise ValueError("output must be a child of the run root")
    return resolved


def _build_evaluation_dataset(recipe: Mapping[str, Any]) -> dict[str, Any]:
    dataset = generate_kbe_dataset(
        "sst",
        seed=2026,
        cache_path=None,
        record_start=RECORD_START,
        record_end=RECORD_END,
        train_end_date=EVALUATION_FIT_END,
        n_train_times=EVALUATION_TRAIN_PAIRS,
        sst_channel_weight=0.5,
        n_eof_sst=int(recipe["eof_sst"]),
        n_eof_aux=int(recipe["eof_aux"]),
    )
    meta = dataset["meta"]
    checks = {
        "data fingerprint": (
            str(meta["data_fingerprint"]) == EVALUATION_DATA_FINGERPRINT
        ),
        "training fingerprint": (
            str(meta["training_fingerprint"])
            == EVALUATION_TRAINING_FINGERPRINT
        ),
        "train_end": int(meta["train_end"]) == 9861,
        "training pairs": (
            int(meta["params"]["n_train_times"]) == EVALUATION_TRAIN_PAIRS
        ),
        "raw FNO domain": not bool(meta.get("seasonal_anomaly", False)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"evaluation dataset mismatch: {', '.join(failed)}")
    return dataset


def _expected_fno_metadata(dataset: Mapping[str, Any]) -> dict[str, Any]:
    config = build_config(False)
    state_shape = tuple(
        int(value) for value in np.asarray(dataset["u0"]).shape[1:]
    )
    return {
        "operator": "fno",
        "model_kwargs": dict(config["model_kwargs"]),
        "system": dataset.get("system"),
        "n_points": int(np.prod(state_shape)),
        "state_shape": state_shape,
        "channels": tuple(dataset["meta"]["channel_names"]),
        "data_fingerprint": dataset["meta"]["data_fingerprint"],
        "training_fingerprint": dataset["meta"]["training_fingerprint"],
        "seed": 2026,
        "train_kwargs": dict(config["train_kwargs"]),
    }


def _load_evaluation_fno(
    dataset: Mapping[str, Any],
    checkpoint: Path,
    *,
    logger: RunLog,
) -> tuple[SimpleNamespace, dict[str, Any]]:
    actual_sha = _require_sha256(
        checkpoint, EVALUATION_FNO_SHA256, "evaluation FNO checkpoint"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {
        "state", "meta", "best_val", "completed_epochs", "stopped_early"
    }
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise ValueError("evaluation FNO checkpoint is malformed")
    expected = _expected_fno_metadata(dataset)
    if payload["meta"] != expected:
        differing = sorted(
            key
            for key in set(payload["meta"]) | set(expected)
            if payload["meta"].get(key) != expected.get(key)
        )
        raise ValueError(f"evaluation FNO metadata mismatch in {differing}")
    if int(payload["completed_epochs"]) != 180 or bool(
        payload["stopped_early"]
    ):
        raise ValueError("evaluation FNO did not complete the 180 epochs")
    if not np.isfinite(float(payload["best_val"])):
        raise ValueError("evaluation FNO best_val is nonfinite")
    operator = build_operator("fno", dataset, expected["model_kwargs"])
    operator.load_state_dict(
        _strict_parameter_state(payload["state"]), strict=True
    )
    operator = operator.to(device).eval()
    fingerprint = _operator_fingerprint(operator)
    if fingerprint != EVALUATION_FNO_OPERATOR_FINGERPRINT:
        raise ValueError(
            "evaluation FNO operator fingerprint mismatch: "
            f"{fingerprint} != {EVALUATION_FNO_OPERATOR_FINGERPRINT}"
        )
    logger.write(
        "loaded evaluation raw FNO",
        path=str(checkpoint.resolve()),
        sha256=actual_sha,
        operator_fingerprint=fingerprint,
    )
    return (
        SimpleNamespace(
            operator_pure=operator,
            grid=np.asarray(dataset["fno_grid"], dtype=np.float64),
            kind=dataset["kind"],
        ),
        {
            "path": str(checkpoint.resolve()),
            "sha256": actual_sha,
            "operator_fingerprint": fingerprint,
            "best_val": float(payload["best_val"]),
            "completed_epochs": int(payload["completed_epochs"]),
        },
    )


def _model_recipe_checks(
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> None:
    if model.get("version") != MODEL_VERSION:
        raise ValueError("model is not a structured BSO")
    validate_model_state_integrity(model)
    validate_fit_window_model_state(model, int(dataset["meta"]["train_end"]))
    eof_counts = (
        int(recipe["eof_sst"]),
        *(
            [int(recipe["eof_aux"])]
            * (int(dataset["meta"]["n_channels"]) - 1)
        ),
    )
    operator_key = str(recipe["operator_key"])
    checks = {
        "rank": int(model["rank"]) == int(recipe["rank"]),
        "ridge": np.isclose(
            float(model["ridge"]), float(recipe["ridge"]), rtol=0, atol=0
        ),
        "EOF counts": tuple(model["eof_counts"]) == eof_counts,
        "affine": bool(model["fit_affine"]) == bool(recipe["fit_affine"]),
        "radius": model["stabilize_max_radius"]
        == float(recipe["stabilize_max_radius"]),
        "operator key": operator_key in model["operators"],
        "alpha": (
            operator_key in model["operators"]
            and np.isclose(
                float(model["operators"][operator_key]["alpha_data"]),
                float(recipe["alpha_data"]),
                rtol=0,
                atol=0,
            )
        ),
        "raw FNO fingerprint": (
            model["raw_operator_fingerprint"]
            == EVALUATION_FNO_OPERATOR_FINGERPRINT
        ),
        "training fingerprint": (
            model["raw_operator_training_fingerprint"]
            == EVALUATION_TRAINING_FINGERPRINT
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"evaluation BSO model recipe mismatch: {', '.join(failed)}"
        )


def _load_npz_pinned(
    path: Path, expected_sha256: str, label: str
) -> dict[str, np.ndarray]:
    _require_sha256(path, expected_sha256, label)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _weighted_rmse(
    prediction: np.ndarray, truth: np.ndarray, weights_2d: np.ndarray
) -> float:
    weights = np.asarray(weights_2d, dtype=np.float64)[None]
    return float(np.sqrt(
        np.sum(weights * (prediction - truth) ** 2) / prediction.shape[0]
    ))


def _weighted_mae(
    prediction: np.ndarray, truth: np.ndarray, weights_2d: np.ndarray
) -> float:
    weights = np.asarray(weights_2d, dtype=np.float64)[None]
    return float(
        np.sum(weights * np.abs(prediction - truth)) / prediction.shape[0]
    )


def _weighted_acc(
    prediction: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    weights_2d: np.ndarray,
) -> float:
    weights = np.asarray(weights_2d, dtype=np.float64)[None]
    pred_anomaly = prediction - climatology
    truth_anomaly = truth - climatology
    pred_anomaly -= np.sum(
        weights * pred_anomaly, axis=(1, 2), keepdims=True
    )
    truth_anomaly -= np.sum(
        weights * truth_anomaly, axis=(1, 2), keepdims=True
    )
    numerator = float(np.sum(weights * pred_anomaly * truth_anomaly))
    pred_energy = float(np.sum(weights * pred_anomaly ** 2))
    truth_energy = float(np.sum(weights * truth_anomaly ** 2))
    if pred_energy <= 1e-24 or truth_energy <= 1e-24:
        return float("nan")
    return float(np.clip(
        numerator / np.sqrt(pred_energy * truth_energy), -1.0, 1.0
    ))


def _weighted_bias(
    prediction: np.ndarray, truth: np.ndarray, weights_2d: np.ndarray
) -> float:
    weights = np.asarray(weights_2d, dtype=np.float64)[None]
    return float(
        np.sum(weights * (prediction - truth)) / prediction.shape[0]
    )


def _weighted_std_ratio(
    prediction: np.ndarray,
    truth: np.ndarray,
    climatology: np.ndarray,
    weights_2d: np.ndarray,
) -> float:
    weights = np.asarray(weights_2d, dtype=np.float64)[None]
    pred_anomaly = prediction - climatology
    truth_anomaly = truth - climatology
    pred_rms = float(np.sqrt(
        np.sum(weights * pred_anomaly ** 2) / prediction.shape[0]
    ))
    truth_rms = float(np.sqrt(
        np.sum(weights * truth_anomaly ** 2) / truth.shape[0]
    ))
    return pred_rms / (truth_rms + 1e-12)


def _first_failure(mask: np.ndarray) -> np.ndarray:
    failures = ~np.asarray(mask, dtype=bool)
    result = np.full(failures.shape[1], -1, dtype=np.int32)
    any_failure = failures.any(axis=0)
    result[any_failure] = failures[:, any_failure].argmax(axis=0) + 1
    return result


def _bso_validity(
    prediction_kelvin: np.ndarray,
    dataset: Mapping[str, Any],
    absolute_origins: np.ndarray,
    *,
    threshold: float = 1000.0,
    invalidity_policy: str = "threshold_or_nonfinite",
) -> dict[str, np.ndarray]:
    allowed_policies = {"threshold_or_nonfinite", "nonfinite_only"}
    if invalidity_policy not in allowed_policies:
        raise ValueError(
            f"invalidity_policy must be one of {sorted(allowed_policies)}")
    meta = dataset["meta"]
    mean = float(np.asarray(meta["channel_mean"])[0])
    std = float(np.asarray(meta["channel_std"])[0])
    finite = np.isfinite(prediction_kelvin).all(axis=(2, 3))
    peak = np.empty(finite.shape, dtype=np.float64)
    for day in range(len(prediction_kelvin)):
        peak[day] = np.nanmax(
            np.abs((prediction_kelvin[day] - mean) / std), axis=(1, 2)
        )
    threshold_ok = peak <= float(threshold)
    valid_now = (
        finite
        if invalidity_policy == "nonfinite_only"
        else finite & threshold_ok
    )
    valid_through = np.logical_and.accumulate(valid_now, axis=0)
    launch_peak = float(np.max(np.abs(
        np.asarray(meta["state_fields"], dtype=np.float32)[
            absolute_origins, 0
        ]
    )))
    return {
        "first_nonfinite": _first_failure(finite),
        "first_threshold_exceedance": _first_failure(threshold_ok),
        "first_divergence": _first_failure(valid_now),
        "valid_through": valid_through,
        "valid_fraction": np.concatenate(
            ([1.0], valid_through.mean(axis=1).astype(np.float64))
        ),
        "max_abs_state": np.concatenate(
            ([launch_peak], np.max(peak, axis=1))
        ),
    }


def _verify_baseline_origins(
    arrays: Mapping[str, np.ndarray],
    dataset: Mapping[str, Any],
    *,
    n_steps: int,
    n_origins: int,
) -> tuple[np.ndarray, np.ndarray]:
    meta = dataset["meta"]
    if int(arrays["n_steps"]) != n_steps or int(
        arrays["n_origins"]
    ) != n_origins:
        raise ValueError("baseline artifact protocol shape mismatch")
    origins = np.asarray(arrays["origins"], dtype=int)
    if len(origins) != n_origins:
        raise ValueError("baseline origin count mismatch")
    absolute = int(meta["train_end"]) + origins
    times = np.asarray(meta["times"]).astype("datetime64[D]")
    if not np.array_equal(
        times[absolute], np.asarray(arrays["origin_dates"]).astype("datetime64[D]")
    ):
        raise ValueError("baseline origin dates differ from evaluation dataset")
    if not np.array_equal(
        times[absolute + n_steps],
        np.asarray(arrays["last_target_dates"]).astype("datetime64[D]"),
    ):
        raise ValueError("baseline target dates differ from evaluation dataset")
    return origins, absolute


def _score_forecast_bso(
    prediction: np.ndarray,
    dataset: Mapping[str, Any],
    absolute: np.ndarray,
    leads: np.ndarray,
) -> dict[str, np.ndarray]:
    meta = dataset["meta"]
    mean = float(np.asarray(meta["channel_mean"])[0])
    std = float(np.asarray(meta["channel_std"])[0])
    raw_sst = np.asarray(meta["sst_fields"], dtype=np.float64)
    times = np.asarray(meta["times"])
    weights = np.asarray(meta["weights_2d"], dtype=np.float64)
    result = {
        name: np.empty(len(leads), dtype=np.float64)
        for name in ("rmse", "mae", "acc", "bias", "std_ratio")
    }
    for slot, lead in enumerate(leads):
        truth = raw_sst[absolute + int(lead)] * std + mean
        climatology = _clim_for_times(
            dataset, times[absolute + int(lead)]
        )
        pred = prediction[slot]
        result["rmse"][slot] = _weighted_rmse(pred, truth, weights)
        result["mae"][slot] = _weighted_mae(pred, truth, weights)
        result["acc"][slot] = _weighted_acc(
            pred, truth, climatology, weights
        )
        result["bias"][slot] = _weighted_bias(pred, truth, weights)
        result["std_ratio"][slot] = _weighted_std_ratio(
            pred, truth, climatology, weights
        )
    return result


def _assemble_forecast_metrics(
    reference: dict[str, np.ndarray],
    model: Mapping[str, Any],
    metrics: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, np.ndarray],
    *,
    operator_key: str,
) -> dict[str, np.ndarray]:
    arrays = {
        key: value
        for key, value in reference.items()
        if (
            "bso" not in key
            and key not in {
                "fingerprint",
                "spectral_radius",
                "raw_spectral_radius",
                "stabilized_eigenvalues",
                "matrix_condition",
                "all_metrics_finite",
            }
        )
    }
    arrays["fingerprint"] = np.asarray(str(model["fingerprint"]))
    arrays["artifact_schema"] = np.asarray("sst-forecast-skill")
    selected = model["operators"][operator_key]
    arrays["spectral_radius"] = np.asarray(
        float(selected["spectral_radius"])
    )
    arrays["raw_spectral_radius"] = np.asarray(
        float(selected["raw_spectral_radius"])
    )
    arrays["stabilized_eigenvalues"] = np.asarray(
        int(selected["clipped_modes"]), dtype=np.int32
    )
    arrays["matrix_condition"] = np.asarray(
        float(selected["matrix_condition"])
    )
    for metric, values in metrics.items():
        arrays[f"{metric}_bso"] = np.asarray(values)
    arrays["rollout_bso_divergence_threshold"] = np.asarray(1000.0)
    for key in (
        "first_nonfinite",
        "first_threshold_exceedance",
        "first_divergence",
        "valid_fraction",
        "max_abs_state",
    ):
        arrays[f"rollout_bso_{key}"] = np.asarray(diagnostics[key])
    for baseline_name in ("raw_persistence", "climatology"):
        arrays[f"rmse_skill_bso_vs_{baseline_name}"] = (
            1.0
            - arrays["rmse_bso"]
            / np.maximum(arrays[f"rmse_{baseline_name}"], 1e-12)
        )
    arrays["all_metrics_finite"] = np.asarray(bool(all(
        np.all(np.isfinite(arrays[f"{metric}_{method}"]))
        for metric in ("rmse", "mae", "acc", "bias", "std_ratio")
        for method in (
            "fno", "sfno", "bso", "damped_persistence",
            "raw_persistence", "climatology"
        )
        if f"{metric}_{method}" in arrays
    )))
    return arrays


def _score_stability_bso(
    prediction: np.ndarray,
    dataset: Mapping[str, Any],
    absolute: np.ndarray,
    leads: np.ndarray,
    diagnostics: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    meta = dataset["meta"]
    mean = float(np.asarray(meta["channel_mean"])[0])
    std = float(np.asarray(meta["channel_std"])[0])
    raw_sst = np.asarray(meta["sst_fields"], dtype=np.float64)
    weights = np.asarray(meta["weights_2d"], dtype=np.float64)
    indices = leads - 1
    valid_through = diagnostics["valid_through"]
    strict = np.full(len(leads), np.nan, dtype=np.float64)
    survivor = np.full(len(leads), np.nan, dtype=np.float64)
    valid_fraction = valid_through[indices].mean(axis=1)
    max_abs_sst = np.full(len(leads), np.nan, dtype=np.float64)
    for slot, (lead, day_index) in enumerate(zip(leads, indices)):
        truth = raw_sst[absolute + int(lead)] * std + mean
        valid = valid_through[day_index]
        if np.any(valid):
            survivor[slot] = _weighted_rmse(
                prediction[day_index, valid], truth[valid], weights
            )
            max_abs_sst[slot] = float(np.max(
                np.abs(prediction[day_index, valid])
            ))
        if np.all(valid):
            strict[slot] = _weighted_rmse(
                prediction[day_index], truth, weights
            )
    return {
        "rmse": strict,
        "rmse_valid": survivor,
        "valid_fraction": valid_fraction,
        "max_abs_sst": max_abs_sst,
    }


def _assemble_stability_metrics(
    reference: dict[str, np.ndarray],
    model: Mapping[str, Any],
    scores: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    arrays = {
        key: value
        for key, value in reference.items()
        if "bso" not in key and key != "fingerprint"
    }
    arrays["fingerprint"] = np.asarray(str(model["fingerprint"]))
    arrays["artifact_schema"] = np.asarray("sst-rollout-stability")
    arrays["invalidity_policy"] = np.asarray("nonfinite_only")
    arrays["large_state_diagnostic_threshold"] = np.asarray(1000.0)
    arrays["rollout_bso_divergence_threshold"] = np.asarray(1000.0)
    for key in (
        "first_nonfinite",
        "first_threshold_exceedance",
        "first_divergence",
        "valid_fraction",
        "max_abs_state",
    ):
        arrays[f"rollout_bso_{key}"] = np.asarray(diagnostics[key])
    for metric, values in scores.items():
        arrays[f"{metric}_bso"] = np.asarray(values)
    arrays["all_48_bso_valid_through_540"] = np.asarray(
        bool(np.all(diagnostics["valid_through"][-1]))
    )
    return arrays


def _stability_method_summary(
    arrays: Mapping[str, np.ndarray],
    method: str,
) -> dict[str, int | float | None]:
    leads = np.asarray(arrays["leads"], dtype=int)
    rmse = np.asarray(arrays[f"rmse_{method}"], dtype=np.float64)
    valid_fraction = np.asarray(
        arrays[f"valid_fraction_{method}"], dtype=np.float64
    )
    complete = np.isfinite(rmse) & np.isclose(valid_fraction, 1.0)
    complete_indices = np.flatnonzero(complete)
    if complete_indices.size:
        last_index = int(complete_indices[-1])
        last_lead: int | None = int(leads[last_index])
        last_rmse: float | None = float(rmse[last_index])
    else:
        last_lead = None
        last_rmse = None

    first_nonfinite = np.asarray(
        arrays[f"rollout_{method}_first_nonfinite"], dtype=int
    )
    recorded = first_nonfinite[first_nonfinite >= 0]
    return {
        "first_nonfinite_day": (
            None if recorded.size == 0 else int(recorded.min())
        ),
        "last_complete_cohort_lead": last_lead,
        "last_complete_cohort_rmse": last_rmse,
    }


def fit(args: argparse.Namespace) -> None:
    output = _safe_run_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"fit output is not empty: {output}")
    lock = _load_json_pinned(
        args.selection_lock, args.expected_selection_lock_sha256
    )
    recipe = lock["selected_recipe"]
    dataset = _build_evaluation_dataset(recipe)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLog(output / "fit.jsonl")
    result, fno_info = _load_evaluation_fno(
        dataset, args.fno_checkpoint, logger=logger
    )
    eof_counts = (
        int(recipe["eof_sst"]),
        *(
            [int(recipe["eof_aux"])]
            * (int(dataset["meta"]["n_channels"]) - 1)
        ),
    )
    logger.write("building locked BSO", recipe=recipe)
    model = build_structured_bso(
        result,
        dataset,
        rank=int(recipe["rank"]),
        ridge=float(recipe["ridge"]),
        eof_counts=eof_counts,
        alphas=(float(recipe["alpha_data"]),),
        fit_affine=bool(recipe["fit_affine"]),
        aux_harmonics=int(recipe["aux_harmonics"]),
        stabilize_max_radius=float(recipe["stabilize_max_radius"]),
        residual_max_abs_gamma=float(recipe["residual_max_abs_gamma"]),
        batch_size=24,
        max_duty_cycle=0.8,
    )
    _model_recipe_checks(model, dataset, recipe)
    model_path = output / "sst_model.pt"
    _atomic_torch_save(model_path, model)
    model_sha = _sha256_file(model_path)
    frozen_recipe = {
        "schema": "sst-bso-frozen-recipe",
        "created_at_utc": _utc_now(),
        "selection_lock": {
            "path": str(args.selection_lock.resolve()),
            "sha256": args.expected_selection_lock_sha256.lower(),
        },
        "selected_recipe": recipe,
        "dataset": {
            "data_fingerprint": EVALUATION_DATA_FINGERPRINT,
            "training_fingerprint": EVALUATION_TRAINING_FINGERPRINT,
            "train_end": int(dataset["meta"]["train_end"]),
        },
        "fno": fno_info,
        "model": {
            "path": str(model_path),
            "sha256": model_sha,
            "fingerprint": str(model["fingerprint"]),
            "full_state_integrity_sha256": str(
                model["full_state_integrity_sha256"]
            ),
        },
        "evaluation": lock["evaluation_policy"],
    }
    recipe_path = output / "frozen_recipe.json"
    _atomic_json(recipe_path, frozen_recipe)
    print(json.dumps({
        "model": str(model_path),
        "model_sha256": model_sha,
        "model_fingerprint": model["fingerprint"],
        "frozen_recipe": str(recipe_path),
        "frozen_recipe_sha256": _sha256_file(recipe_path),
    }, indent=2, sort_keys=True))


def preflight_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output = _safe_run_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {output}")
    frozen = _load_json_pinned_frozen(
        args.frozen_recipe, args.expected_frozen_recipe_sha256
    )
    recipe = frozen["selected_recipe"]
    dataset = _build_evaluation_dataset(recipe)
    _require_sha256(args.model, args.expected_model_sha256, "BSO model")
    model = torch.load(args.model, map_location="cpu", weights_only=False)
    _model_recipe_checks(model, dataset, recipe)
    if str(model["fingerprint"]) != frozen["model"]["fingerprint"]:
        raise ValueError("frozen recipe model fingerprint mismatch")
    forecast = _load_npz_pinned(
        args.baseline_forecast,
        BASELINE_FORECAST_SHA256,
        "baseline forecast artifact",
    )
    stability = _load_npz_pinned(
        args.baseline_stability,
        BASELINE_STABILITY_SHA256,
        "baseline stability artifact",
    )
    forecast_origins, _ = _verify_baseline_origins(
        forecast, dataset, n_steps=90, n_origins=96
    )
    stability_origins, _ = _verify_baseline_origins(
        stability, dataset, n_steps=540, n_origins=48
    )
    if not np.array_equal(
        np.asarray(forecast["leads"], dtype=int), np.arange(1, 91)
    ):
        raise ValueError("baseline forecast lead grid is not daily 1--90")
    expected_stability_leads = np.unique(np.concatenate([
        np.arange(1, 31),
        np.arange(35, 181, 5),
        np.arange(190, 541, 10),
    ]))
    if not np.array_equal(
        np.asarray(stability["leads"], dtype=int), expected_stability_leads
    ):
        raise ValueError("baseline stability lead grid mismatch")
    return {
        "output": output,
        "frozen": frozen,
        "recipe": recipe,
        "dataset": dataset,
        "model": model,
        "forecast": forecast,
        "stability": stability,
        "forecast_origins": forecast_origins,
        "stability_origins": stability_origins,
    }


def _load_json_pinned_frozen(
    path: Path, expected_sha256: str
) -> Mapping[str, Any]:
    _require_sha256(path, expected_sha256, "frozen recipe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "sst-bso-frozen-recipe":
        raise ValueError("unexpected frozen recipe schema")
    return payload


def evaluate(args: argparse.Namespace) -> None:
    state = preflight_evaluate(args)
    output: Path = state["output"]
    output.mkdir(parents=True, exist_ok=False)
    marker = output / "EVALUATION_STARTED"
    marker.write_text(
        json.dumps({
            "started_at_utc": _utc_now(),
            "frozen_recipe_sha256": args.expected_frozen_recipe_sha256,
            "model_sha256": args.expected_model_sha256,
            "pid": os.getpid(),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger = RunLog(output / "evaluate.jsonl")
    dataset = state["dataset"]
    model = state["model"]
    recipe = state["recipe"]
    meta = dataset["meta"]
    train_end = int(meta["train_end"])

    forecast_reference = state["forecast"]
    forecast_leads = np.asarray(forecast_reference["leads"], dtype=int)
    forecast_absolute = train_end + state["forecast_origins"]
    logger.write("running BSO 90-day forecast")
    forecast_prediction = _rollout_bso_selected_leads(
        dataset,
        model,
        forecast_absolute,
        forecast_leads,
        operator_key=str(recipe["operator_key"]),
        residual_mode=str(recipe["residual_mode"]),
    )
    forecast_diagnostics = _bso_validity(
        forecast_prediction, dataset, forecast_absolute,
        invalidity_policy="threshold_or_nonfinite",
    )
    forecast_metrics = _score_forecast_bso(
        forecast_prediction, dataset, forecast_absolute, forecast_leads
    )
    forecast_arrays = _assemble_forecast_metrics(
        forecast_reference,
        model,
        forecast_metrics,
        forecast_diagnostics,
        operator_key=str(recipe["operator_key"]),
    )

    stability_reference = state["stability"]
    stability_leads = np.asarray(stability_reference["leads"], dtype=int)
    stability_absolute = train_end + state["stability_origins"]
    logger.write("running BSO 540-day daily stability")
    daily = np.arange(1, 541, dtype=int)
    stability_prediction = _rollout_bso_selected_leads(
        dataset,
        model,
        stability_absolute,
        daily,
        operator_key=str(recipe["operator_key"]),
        residual_mode=str(recipe["residual_mode"]),
    )
    stability_diagnostics = _bso_validity(
        stability_prediction, dataset, stability_absolute,
        invalidity_policy="nonfinite_only",
    )
    stability_scores = _score_stability_bso(
        stability_prediction,
        dataset,
        stability_absolute,
        stability_leads,
        stability_diagnostics,
    )
    stability_arrays = _assemble_stability_metrics(
        stability_reference, model, stability_scores, stability_diagnostics
    )

    forecast_path = output / "forecast_metrics.npz"
    stability_path = output / "stability_metrics.npz"
    _atomic_savez(forecast_path, forecast_arrays)
    _atomic_savez(stability_path, stability_arrays)
    summary = {
        "schema": "sst-bso-evaluation",
        "created_at_utc": _utc_now(),
        "protocol": "evaluation_2020_2022",
        "forecast_profile": FORECAST_PROFILE,
        "stability_profile": STABILITY_PROFILE,
        "method_definitions": {
            "bso": {
                "display_name": "BSO",
                "kind": "structured_bso",
                "operator_key": str(recipe["operator_key"]),
                "target_kind": "observation_fno_blend",
                "alpha_data": float(recipe["alpha_data"]),
                "residual_mode": str(recipe["residual_mode"]),
            }
        },
        "baseline_reuse": {
            "forecast_path": str(args.baseline_forecast.resolve()),
            "forecast_sha256": BASELINE_FORECAST_SHA256,
            "stability_path": str(args.baseline_stability.resolve()),
            "stability_sha256": BASELINE_STABILITY_SHA256,
            "invariant": (
                "Pinned FNO, SFNO, persistence, and climatology arrays are "
                "carried into the common evaluation artifact."
            ),
        },
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.expected_model_sha256.lower(),
            "fingerprint": str(model["fingerprint"]),
        },
        "stability": {
            "invalidity_policy": "nonfinite_only",
            "large_state_diagnostic_threshold": 1000.0,
            "termination_condition": "NaN_or_infinity",
            "methods": {
                method: _stability_method_summary(
                    stability_arrays, method
                )
                for method in ("bso", "fno", "sfno")
            },
        },
        "outputs": {
            "forecast_metrics": str(forecast_path),
            "forecast_metrics_sha256": _sha256_file(forecast_path),
            "stability_metrics": str(stability_path),
            "stability_metrics_sha256": _sha256_file(stability_path),
        },
    }
    summary_path = output / "summary.json"
    _atomic_json(summary_path, summary)
    manifest = {
        "schema": "sst-bso-current-result",
        "status": "evaluation_completed",
        "created_at_utc": _utc_now(),
        "frozen_recipe": {
            "path": str(args.frozen_recipe.resolve()),
            "sha256": args.expected_frozen_recipe_sha256.lower(),
        },
        "model": summary["model"],
        "evaluation": {
            "forecast": {
                "profile": FORECAST_PROFILE,
                "n_steps": 90,
                "n_origins": 96,
                "bands": ["1-10", "11-30", "31-90"],
            },
            "stability": {
                "profile": STABILITY_PROFILE,
                "n_steps": 540,
                "n_origins": 48,
                "learned_methods_only": True,
                "no_statistical_baseline_ranking": True,
                "invalidity_policy": "nonfinite_only",
                "large_state_diagnostic_threshold": 1000.0,
                "termination_condition": "NaN_or_infinity",
            },
        },
        "outputs": {
            **summary["outputs"],
            "summary": str(summary_path),
            "summary_sha256": _sha256_file(summary_path),
        },
    }
    manifest_path = output / "run_manifest.json"
    _atomic_json(manifest_path, manifest)
    latest_path = RESULTS_DIR / "sst" / "latest_result.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(latest_path, {
        "schema": "sst-result-index",
        "status": "evaluation_completed",
        "manifest": _project_relative(manifest_path),
        "metrics": _project_relative(forecast_path),
        "stability_metrics": _project_relative(stability_path),
        "model": _project_relative(args.model),
        "run_dir": _project_relative(output),
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_evaluation(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser.add_argument("--selection-lock", type=Path, required=True)
    fit_parser.add_argument("--expected-selection-lock-sha256", required=True)
    fit_parser.add_argument("--fno-checkpoint", type=Path, required=True)

    def add_evaluate_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--output-dir", type=Path, required=True)
        target.add_argument("--frozen-recipe", type=Path, required=True)
        target.add_argument("--expected-frozen-recipe-sha256", required=True)
        target.add_argument("--model", type=Path, required=True)
        target.add_argument("--expected-model-sha256", required=True)
        target.add_argument("--baseline-forecast", type=Path, required=True)
        target.add_argument("--baseline-stability", type=Path, required=True)

    preflight_parser = subparsers.add_parser("preflight-evaluate")
    add_evaluate_arguments(preflight_parser)
    evaluate_parser = subparsers.add_parser("evaluate")
    add_evaluate_arguments(evaluate_parser)
    args = parser.parse_args(argv)

    if args.command == "fit":
        fit(args)
    elif args.command == "preflight-evaluate":
        state = preflight_evaluate(args)
        print(json.dumps({
            "status": "preflight_passed_no_predictions_run",
            "model_fingerprint": state["model"]["fingerprint"],
            "forecast_origins": len(state["forecast_origins"]),
            "stability_origins": len(state["stability_origins"]),
        }, indent=2, sort_keys=True))
    else:
        evaluate(args)
