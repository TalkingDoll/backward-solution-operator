"""Lorenz-63 BSO forecasting with a frozen neural operator.

The experiment advances 20 sensor curves with a cached FNO. A fixed
instantaneous dictionary combines the standardized curve with low-order
monomials of its leading PCA coordinates. A Galerkin least-squares map
propagates the lifted coefficients for the BSO forecast. All reported
forecasts use the held-out trajectory tail.

Usage:
    python -m experiments.lorenz63.exp_lorenz63
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from edmd_bso_method import apply_operator, build_operator, device
from kbe_observation_data import generate_kbe_dataset
from plotting_utils import DEFAULT_LAYOUT, apply_layout, save_figure_pair
from project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
RESULTS_DIR = ROOT / "results" / "l63"
FIGURE_PATH = ROOT / "figures" / "lorenz63" / "l63_prediction.pdf"

FAMILY = "nonlinear_sensor"
SEED = 2026
FNO_KWARGS = {"modes": 10, "width": 128, "n_layers": 5}
PCA_COORDINATES = 5
POLYNOMIAL_DEGREE = 4
TRANSITION_RIDGE_RELATIVE = 1.0e-3
DECODER_RIDGE_RELATIVE = 1.0e-10
STABILITY_RADIUS = 0.97
SAMPLE_STRIDE = 1
N_STEPS = 110
N_ORIGINS = 64
ORIGIN_START = 25
PREDICTION_FIGSIZE = (15.0, 4.3)
PREDICTION_LAYOUT = dict(
    DEFAULT_LAYOUT,
    left=0.065,
    right=0.985,
    bottom=0.16,
    top=0.72,
    wspace=0.20,
    suptitle_y=0.985,
    legend_y=0.925,
)


@dataclass
class ForecastContext:
    operator: Any
    operator_pure: Any
    grid: np.ndarray
    kind: str
    weights: np.ndarray
    T: float
    operator_val_rel_l2: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprints() -> dict[str, dict[str, Any]]:
    names = ("dataset.pkl", "fno.pt")
    missing = [name for name in names if not (RESULTS_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing L63 source assets: {missing}")
    return {
        name: {
            "sha256": _sha256(RESULTS_DIR / name),
            "size": (RESULTS_DIR / name).stat().st_size,
            "mtime_ns": (RESULTS_DIR / name).stat().st_mtime_ns,
        }
        for name in names
    }


def _flatten(fields: Any) -> np.ndarray:
    array = np.asarray(fields)
    return array.reshape(array.shape[0], -1)


def _polynomial_exponents(
    n_coordinates: int,
    maximum_degree: int,
) -> np.ndarray:
    """Enumerate degree-2 through degree-p monomials deterministically."""

    exponents: list[np.ndarray] = []
    for total_degree in range(2, int(maximum_degree) + 1):
        for indices in itertools.combinations_with_replacement(
            range(int(n_coordinates)), total_degree
        ):
            exponent = np.zeros(int(n_coordinates), dtype=np.int16)
            for index in indices:
                exponent[index] += 1
            exponents.append(exponent)
    return np.asarray(exponents, dtype=np.int16)


def _raw_dictionary(
    fields: np.ndarray,
    model: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Evaluate the fixed instantaneous polynomial observable dictionary."""

    centered = np.asarray(fields, dtype=np.float64) - np.asarray(
        model["field_mean"], dtype=np.float64
    )
    linear = centered / np.asarray(model["field_scale"], dtype=np.float64)
    scores = (
        centered @ np.asarray(model["pca"], dtype=np.float64)
    ) / np.asarray(model["score_scale"], dtype=np.float64)
    exponents = np.asarray(model["exponents"], dtype=np.int16)
    if exponents.size == 0:
        return linear
    polynomial = np.prod(
        scores[:, None, :] ** exponents[None, :, :], axis=2
    )
    return np.concatenate([linear, polynomial], axis=1)


def _lift_dictionary(
    fields: np.ndarray,
    model: Mapping[str, np.ndarray],
) -> np.ndarray:
    raw = _raw_dictionary(fields, model)
    return (
        raw - np.asarray(model["feature_mean"], dtype=np.float64)
    ) / np.asarray(model["feature_scale"], dtype=np.float64)


def _ridge_map(
    source: np.ndarray,
    target: np.ndarray,
    ridge_relative: float,
) -> np.ndarray:
    """Fit a scale-normalized ridge map in row-vector convention."""

    gram = source.T @ source
    scale = float(np.trace(gram) / gram.shape[0])
    ridge = float(ridge_relative) * scale
    return np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0], dtype=np.float64),
        source.T @ target,
    )


def _stabilize_transition(
    transition: np.ndarray,
    maximum_radius: float,
) -> tuple[np.ndarray, float, float]:
    """Radially clip unstable eigenvalues without changing stable modes."""

    eigenvalues, eigenvectors = np.linalg.eig(transition)
    raw_radius = float(np.max(np.abs(eigenvalues)))
    condition = float(np.linalg.cond(eigenvectors))
    if not np.isfinite(condition) or condition > 1.0e10:
        raise np.linalg.LinAlgError(
            f"ill-conditioned BSO eigenvectors: {condition:.6e}"
        )
    if raw_radius <= float(maximum_radius):
        return np.asarray(transition, dtype=np.float64).copy(), raw_radius, condition

    magnitudes = np.abs(eigenvalues)
    clipped = eigenvalues.copy()
    mask = magnitudes > float(maximum_radius)
    clipped[mask] *= float(maximum_radius) / magnitudes[mask]
    stable = eigenvectors @ np.diag(clipped) @ np.linalg.inv(eigenvectors)
    if np.iscomplexobj(stable):
        relative_imaginary = float(
            np.linalg.norm(stable.imag)
            / (np.linalg.norm(stable.real) + np.finfo(np.float64).eps)
        )
        if relative_imaginary > 1.0e-7:
            raise np.linalg.LinAlgError(
                "stabilized BSO matrix has a non-negligible imaginary part"
            )
        stable = stable.real
    return np.asarray(stable, dtype=np.float64), raw_radius, condition


def load_dataset() -> dict[str, Any]:
    """Load the cached Lorenz-63 sensor data without regenerating it."""

    cache_path = RESULTS_DIR / "dataset.pkl"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing cached dataset: {cache_path}")
    return generate_kbe_dataset(
        "lorenz63",
        n_samples=12,
        seed=SEED,
        cache_path=str(cache_path),
    )


def load_frozen_operator(dataset: Mapping[str, Any]) -> ForecastContext:
    """Restore the cached FNO and return only the pieces needed for rollout."""

    checkpoint_path = RESULTS_DIR / "fno.pt"
    model = build_operator("fno", dataset, dict(FNO_KWARGS)).to(device)
    expected_meta = {
        "operator": "fno",
        "model_kwargs": dict(FNO_KWARGS),
        "system": dataset.get("system"),
        "n_points": int(_flatten(dataset["u0"]).shape[1]),
    }
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("meta") != expected_meta:
        raise RuntimeError("The cached FNO checkpoint does not match this experiment.")
    model.load_state_dict(payload["state"], strict=False)
    model = model.to(device).eval()

    grid = torch.as_tensor(dataset["fno_grid"], dtype=torch.float32, device=device)
    u0 = torch.as_tensor(_flatten(dataset["u0"]), dtype=torch.float32, device=device)
    uT = torch.as_tensor(_flatten(dataset["uT"]), dtype=torch.float32, device=device)
    with torch.no_grad():
        prediction = apply_operator(model, u0, grid, dataset["kind"])
        validation = torch.mean(
            torch.linalg.vector_norm(prediction - uT, dim=-1)
            / (torch.linalg.vector_norm(uT, dim=-1) + 1.0e-10)
        )

    return ForecastContext(
        operator=model,
        operator_pure=copy.deepcopy(model).eval(),
        grid=np.asarray(dataset["fno_grid"], dtype=np.float64),
        kind=str(dataset["kind"]),
        weights=np.asarray(dataset["weights"], dtype=np.float64),
        T=float(dataset["T"]),
        operator_val_rel_l2=float(validation),
    )


def batch_operator_rollout(
    context: ForecastContext,
    initial_fields: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Advance a batch of sensor curves with the frozen FNO."""

    grid = torch.as_tensor(context.grid, dtype=torch.float32, device=device)
    fields = torch.as_tensor(initial_fields, dtype=torch.float32, device=device)
    trajectory = [np.asarray(initial_fields, dtype=np.float64).copy()]
    context.operator_pure.eval()
    with torch.no_grad():
        for _ in range(int(n_steps)):
            fields = apply_operator(context.operator_pure, fields, grid, context.kind)
            trajectory.append(fields.detach().cpu().numpy().astype(np.float64))
    return np.asarray(trajectory)


def build_bso_model(
    context: ForecastContext,
    dataset: Mapping[str, Any],
    pca_coordinates: int = PCA_COORDINATES,
    polynomial_degree: int = POLYNOMIAL_DEGREE,
    stride: int = SAMPLE_STRIDE,
    transition_ridge_relative: float = TRANSITION_RIDGE_RELATIVE,
    decoder_ridge_relative: float = DECODER_RIDGE_RELATIVE,
    stability_radius: float = STABILITY_RADIUS,
) -> dict[str, np.ndarray]:
    """Fit the instantaneous polynomial-dictionary Galerkin BSO."""

    readings = np.asarray(dataset["meta"]["train_readings"][FAMILY], dtype=np.float64)
    sampled = readings[:, np.arange(0, readings.shape[1], int(stride))].T
    grid = torch.as_tensor(context.grid, dtype=torch.float32, device=device)
    advanced_chunks = []
    context.operator_pure.eval()
    with torch.no_grad():
        for start in range(0, len(sampled), 4096):
            fields = torch.as_tensor(sampled[start:start + 4096], dtype=torch.float32, device=device)
            advanced_chunks.append(
                apply_operator(context.operator_pure, fields, grid, context.kind)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    advanced = np.concatenate(advanced_chunks)

    field_mean = sampled.mean(axis=0)
    centered = sampled - field_mean
    _, vectors = np.linalg.eigh(centered.T @ centered)
    pca = vectors[:, ::-1][:, : int(pca_coordinates)]
    score_scale = (centered @ pca).std(axis=0) + 1.0e-12
    field_scale = centered.std(axis=0) + 1.0e-12
    exponents = _polynomial_exponents(
        int(pca_coordinates), int(polynomial_degree)
    )
    model: dict[str, np.ndarray] = {
        "field_mean": field_mean,
        "field_scale": field_scale,
        "pca": pca,
        "score_scale": score_scale,
        "exponents": exponents,
    }
    raw = _raw_dictionary(sampled, model)
    feature_mean = raw.mean(axis=0)
    feature_scale = raw.std(axis=0) + 1.0e-12
    model["feature_mean"] = feature_mean
    model["feature_scale"] = feature_scale

    source = _lift_dictionary(sampled, model)
    target = _lift_dictionary(advanced, model)
    decoder = _ridge_map(
        source,
        sampled - field_mean,
        float(decoder_ridge_relative),
    )
    transition_row = _ridge_map(
        source,
        target,
        float(transition_ridge_relative),
    )
    transition_row, raw_radius, eigenvector_condition = _stabilize_transition(
        transition_row, float(stability_radius)
    )
    model.update(
        {
            "A": transition_row.T,
            "decoder": decoder,
            "raw_spectral_radius": np.asarray(raw_radius),
            "eigenvector_condition": np.asarray(eigenvector_condition),
        }
    )
    return model


def _forecast_origins(readings: np.ndarray) -> np.ndarray:
    last = readings.shape[1] - N_STEPS - 2
    if ORIGIN_START >= last:
        raise ValueError("The held-out test window is too short for the forecast protocol.")
    return np.linspace(ORIGIN_START, last, N_ORIGINS, dtype=int)


def _truth_trajectory(readings: np.ndarray, origins: np.ndarray) -> np.ndarray:
    return np.stack([readings[:, origins + step].T for step in range(N_STEPS + 1)])


def _relative_errors(
    trajectory: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    field_weights = weights.reshape(1, 1, -1)
    numerator = np.sqrt(np.sum(field_weights * (trajectory - truth) ** 2, axis=(1, 2)))
    denominator = np.sqrt(np.sum(field_weights * truth**2, axis=(1, 2))) + 1.0e-12
    return numerator / denominator


def bso_rollout(
    model: Mapping[str, np.ndarray],
    initial_fields: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Advance lifted coefficients analytically through the Galerkin map."""

    transition = np.asarray(model["A"], dtype=np.float64)
    decoder = np.asarray(model["decoder"], dtype=np.float64)
    field_mean = np.asarray(model["field_mean"], dtype=np.float64)
    coefficients = _lift_dictionary(initial_fields, model).T
    eigenvalues, eigenvectors = np.linalg.eig(transition)
    amplitudes = np.linalg.lstsq(eigenvectors, coefficients.astype(complex), rcond=None)[0]
    trajectory = []
    for step in range(int(n_steps) + 1):
        evolved = eigenvectors @ (eigenvalues[:, None] ** step * amplitudes)
        trajectory.append(evolved.T.real @ decoder + field_mean)
    return np.asarray(trajectory)


def forecast_errors(
    context: ForecastContext,
    dataset: Mapping[str, Any],
    model: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate direct and BSO forecasts on the held-out trajectory tail."""

    readings = np.asarray(dataset["meta"]["test_readings"][FAMILY], dtype=np.float64)
    origins = _forecast_origins(readings)
    truth = _truth_trajectory(readings, origins)
    operator_trajectory = batch_operator_rollout(context, truth[0], N_STEPS)
    spectral_trajectory = bso_rollout(model, truth[0], N_STEPS)
    times = context.T * np.arange(N_STEPS + 1)
    return (
        times,
        origins,
        _relative_errors(operator_trajectory, truth, context.weights),
        _relative_errors(spectral_trajectory, truth, context.weights),
    )


def _band_means(times: np.ndarray, errors: np.ndarray) -> dict[str, float]:
    bands = {
        "short_t_le_0_5": times <= 0.5 + 1.0e-12,
        "mid_0_5_lt_t_le_2": (times > 0.5) & (times <= 2.0 + 1.0e-12),
        "long_2_lt_t_le_11": times > 2.0,
    }
    return {name: float(np.mean(errors[mask])) for name, mask in bands.items()}


def plot_forecasts(
    path: Path,
    times: np.ndarray,
    operator_errors: np.ndarray,
    spectral_errors: np.ndarray,
) -> None:
    """Create a legible short-, mid-, and long-range prediction figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=PREDICTION_FIGSIZE)
    panels = ((0.0, 0.5, "short term"), (0.5, 2.0, "mid term"), (2.0, 11.0, "long term"))
    for ax, (t_lo, t_hi, name) in zip(axes, panels):
        mask = (times >= t_lo - 1.0e-12) & (times <= t_hi + 1.0e-12)
        operator_y = np.maximum(operator_errors[mask], 1.0e-8)
        spectral_y = np.maximum(spectral_errors[mask], 1.0e-8)
        ax.plot(times[mask], operator_y, "-o", ms=3.5, c="tab:blue", label="FNO")
        ax.plot(times[mask], spectral_y, "-s", ms=3.5, c="tab:red", label="BSO")

        values = np.concatenate([operator_y, spectral_y])
        if t_lo == 0:
            ax.set_yscale("log")
            log_values = np.log10(values)
            log_span = max(float(log_values.max() - log_values.min()), 0.06)
            y_lo = 10.0 ** (float(log_values.min()) - 0.12 * log_span)
            y_hi = 10.0 ** (float(log_values.max()) + 0.12 * log_span)
        else:
            y_min, y_max = float(values.min()), float(values.max())
            y_span = max(y_max - y_min, 0.05)
            y_lo = max(0.0, y_min - 0.10 * y_span)
            y_hi = y_max + 0.10 * y_span
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r"$t$", fontsize=20)
        interval = (
            rf"$t \leq {t_hi:g}$"
            if t_lo == 0
            else rf"${t_lo:g} < t \leq {t_hi:g}$"
        )
        ax.set_title(f"{name} ({interval})", fontsize=20)
        ax.tick_params(labelsize=18)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, PREDICTION_LAYOUT["legend_y"]),
        ncol=2,
        frameon=False,
        fontsize=18,
    )
    fig.suptitle(r"Lorenz-63 prediction (relative $L^2$ error)", fontsize=22,
                 y=PREDICTION_LAYOUT["suptitle_y"])
    apply_layout(fig, PREDICTION_LAYOUT)
    save_figure_pair(fig, path, dpi=PREDICTION_LAYOUT["dpi"])
    plt.close(fig)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _readme(metrics: Mapping[str, Any]) -> str:
    selected = metrics["selected_leads"]
    means = metrics["mean_relative_l2"]
    operator_bands = metrics["operator_band_means"]
    bso_bands = metrics["spectral_band_means"]
    protocol = metrics["protocol"]
    method = metrics["method"]
    return f"""# Lorenz-63 BSO forecast

## Method

The cached FNO advances the 20 sensor curves. The instantaneous BSO dictionary
contains the standardized curve and degree-2 through degree-{method['polynomial_degree']}
monomials of its leading {method['pca_coordinates']} PCA coordinates, for
{method['dictionary_size']} features in total. A ridge-regularized Galerkin
map evolves those coefficients autonomously.

## Held-out evaluation

The evaluation uses {protocol['n_origins']} uniformly spaced forecast origins
from the held-out trajectory tail, {protocol['n_steps']} forecast steps, and
the weighted relative $L^2$ metric over all sensor sites.

| Horizon | FNO | BSO |
|---|---:|---:|
| Short, $t \\leq 0.5$ | {operator_bands['short_t_le_0_5']:.10f} | {bso_bands['short_t_le_0_5']:.10f} |
| Middle, $0.5 < t \\leq 2$ | {operator_bands['mid_0_5_lt_t_le_2']:.10f} | {bso_bands['mid_0_5_lt_t_le_2']:.10f} |
| Long, $2 < t \\leq 11$ | {operator_bands['long_2_lt_t_le_11']:.10f} | {bso_bands['long_2_lt_t_le_11']:.10f} |
| All steps | {means['operator']:.10f} | {means['spectral']:.10f} |

Reported BSO errors are {selected['t0.1']:.10f} at $t=0.1$,
{selected['t0.5']:.10f} at $t=0.5$, {selected['t2']:.10f} at $t=2$,
and {selected['t11']:.10f} at $t=11$.

## Files

- `dataset.pkl`: cached Lorenz-63 sensor data.
- `fno.pt`: frozen FNO checkpoint.
- `spectral_model.npz`: BSO dictionary transforms, decoder, Galerkin map, and eigenvalues.
- `forecast_metrics.npz`: time grid, forecast origins, and both error curves.
- `manifest.json`: method settings, metrics, and source-asset hashes.

Reproduce with `python -m experiments.lorenz63.exp_lorenz63`.
"""


def run(output_dir: Path = RESULTS_DIR, figure_path: Path = FIGURE_PATH) -> dict[str, Any]:
    """Recreate the regular Lorenz-63 forecast artifacts from cached inputs."""

    output_dir = output_dir.expanduser().resolve()
    figure_path = figure_path.expanduser().resolve()
    if output_dir == ROOT.resolve():
        raise ValueError("The repository root cannot be an output directory.")
    before = _source_fingerprints()
    dataset = load_dataset()
    context = load_frozen_operator(dataset)
    model = build_bso_model(context, dataset)
    times, origins, operator_errors, spectral_errors = forecast_errors(context, dataset, model)
    after = _source_fingerprints()
    if before != after:
        raise AssertionError("The cached dataset or FNO checkpoint changed during evaluation.")

    eigenvalues = np.linalg.eigvals(model["A"])
    metrics: dict[str, Any] = {
        "created_at": _utc_now(),
        "method": {
            "operator": "frozen FNO",
            "training_family": FAMILY,
            "dictionary": "standardized field plus PCA-coordinate monomials",
            "pca_coordinates": PCA_COORDINATES,
            "polynomial_degree": POLYNOMIAL_DEGREE,
            "dictionary_size": int(model["A"].shape[0]),
            "sampling_stride": SAMPLE_STRIDE,
            "transition_ridge_relative": TRANSITION_RIDGE_RELATIVE,
            "decoder_ridge_relative": DECODER_RIDGE_RELATIVE,
            "stability_radius": STABILITY_RADIUS,
            "projection": "ridge-regularized Galerkin least squares",
            "target": "frozen FNO one-step image",
        },
        "protocol": {
            "n_steps": N_STEPS,
            "n_origins": N_ORIGINS,
            "origin_start_test_index": int(origins[0]),
            "origin_end_test_index": int(origins[-1]),
            "tau": float(context.T),
            "metric": "weighted relative L2 over forecast origins and 20 sensor sites",
        },
        "mean_relative_l2": {
            "operator": float(np.mean(operator_errors)),
            "spectral": float(np.mean(spectral_errors)),
        },
        "operator_band_means": _band_means(times, operator_errors),
        "spectral_band_means": _band_means(times, spectral_errors),
        "selected_leads": {
            "t0": float(spectral_errors[0]),
            "t0.1": float(spectral_errors[1]),
            "t0.2": float(spectral_errors[2]),
            "t0.5": float(spectral_errors[5]),
            "t2": float(spectral_errors[20]),
            "t11": float(spectral_errors[-1]),
        },
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "raw_spectral_radius": float(model["raw_spectral_radius"]),
        "eigenvector_condition": float(model["eigenvector_condition"]),
        "operator_validation_relative_l2": float(context.operator_val_rel_l2),
        "source_assets": before,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        output_dir / "spectral_model.npz",
        A=model["A"],
        decoder=model["decoder"],
        field_mean=model["field_mean"],
        field_scale=model["field_scale"],
        pca=model["pca"],
        score_scale=model["score_scale"],
        exponents=model["exponents"],
        feature_mean=model["feature_mean"],
        feature_scale=model["feature_scale"],
        eigenvalues=eigenvalues,
        pca_coordinates=np.asarray(PCA_COORDINATES),
        polynomial_degree=np.asarray(POLYNOMIAL_DEGREE),
        transition_ridge_relative=np.asarray(TRANSITION_RIDGE_RELATIVE),
        decoder_ridge_relative=np.asarray(DECODER_RIDGE_RELATIVE),
        stability_radius=np.asarray(STABILITY_RADIUS),
        stride=np.asarray(SAMPLE_STRIDE),
    )
    _atomic_npz(
        output_dir / "forecast_metrics.npz",
        times=times,
        origins=origins,
        operator_errors=operator_errors,
        spectral_errors=spectral_errors,
    )
    _atomic_json(output_dir / "manifest.json", metrics)
    (output_dir / "README.md").write_text(_readme(metrics), encoding="utf-8")
    plot_forecasts(figure_path, times, operator_errors, spectral_errors)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recreate the Lorenz-63 BSO forecast artifacts.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = run(args.output_dir, args.figure)
    means = metrics["mean_relative_l2"]
    print(f"FNO mean relative L2: {means['operator']:.10f}")
    print(f"BSO mean relative L2: {means['spectral']:.10f}")
    print(f"artifacts: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
