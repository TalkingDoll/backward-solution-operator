"""Reproduce the fixed R02 resting-EEG rollout experiment.

The experiment uses subject 29, run 2. It fits on 0--15 s and scores one
uninterrupted boundary rollout from 15 s through 19 s. At 160 Hz with a
16-sample lag, the four-second endpoint contains 40 rollout steps.

The FNO-BSO path trains the Phase-A field operator, freezes it, and then
learns a 25-function dictionary on a four-dimensional fixed PCA state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from project_paths import DATA_DIR, FIGURES_DIR, RESULTS_DIR
from experiments.eeg.eeg_baselines import fit_kernel_edmd, fit_pca_var
from experiments.eeg.eeg_edmddl import (
    EDMDDLTrainingConfig,
    FixedPCAStateMap,
    rollout_paper_edmd_dl,
    train_paper_edmd_dl,
)
from experiments.eeg.eeg_operator_bso import (
    OperatorBSOConfig,
    fit_operator_bso,
    project_frozen_operator_bso,
    rollout_operator_bso,
    rollout_phase_a_operator,
)
from experiments.eeg.eeg_resting_data import (
    LAG_SAMPLES,
    RestingRolloutSplit,
    load_resting_eeg_run,
    make_resting_rollout_split,
    restore_resting_signal_units,
)


SUBJECT = 29
RUN = 2
SAMPLING_RATE_HZ = 160
ROLLOUT_STEPS = 40
STEP_SECONDS = LAG_SAMPLES / SAMPLING_RATE_HZ
HORIZON_SECONDS = ROLLOUT_STEPS * STEP_SECONDS
SOURCE_SHA256 = (
    "7eff9783f20f4d3521ae877f26d7b2c87c609a920e16d12e5479072c2cb65671"
)
METHODS = (
    "FNO-BSO",
    "FNO",
    "EDMD-DL",
    "Kernel EDMD",
    "PCA-VAR",
)
REPRODUCTION_SCHEMA = "eeg-resting-S029-R02-rollout-v1"
REFERENCE_SCORES = {
    "FNO-BSO": 0.7494409016952366,
    "FNO": 0.8736726336035538,
    "EDMD-DL": 0.9409873902990185,
    "Kernel EDMD": 0.944762508644189,
    "PCA-VAR": 0.9632015237957963,
}
DEFAULT_OUTPUT_DIR = (
    RESULTS_DIR
    / "eeg"
    / "resting-S029-R02-rollout"
)
DEFAULT_FIGURE_PDF = (
    FIGURES_DIR / "eeg" / "eeg_rollout_comparison.pdf"
)
EEG_FIGURE = {
    "figsize": (7.0, 3.0),
    "nrows": 1,
    "ncols": 1,
    "left_in": 0.48,
    "right_in": 0.10,
    "bottom_in": 0.50,
    "top_in": 0.80,
    "column_gap_in": 0.0,
    "row_gap_in": 0.0,
    "suptitle_top_in": 0.05,
    "legend_top_in": 0.38,
    "legend_ncol": 5,
}


def locked_hyperparameters() -> dict[str, dict[str, Any]]:
    """Return the fixed configurations used by the reference experiment."""

    return {
        "FNO-BSO": {
            "phase_a": {
                "dictionary_size": 10,
                "reduced_state_dimension": 8,
                "dictionary_hidden_dims": (16, 16),
                "operator_epochs": 100,
                "dictionary_epochs": 1,
                "batch_size": 32,
                "operator_learning_rate": 7.0e-4,
                "dictionary_learning_rate": 7.0e-4,
                "operator_weight_decay": 2.0e-5,
                "transition_ridge": 3.0e-6,
                "gradient_clip": 5.0,
                "seed": 2026,
                "print_every": 0,
                "fno_hidden_channels": 48,
                "fno_modes": 64,
                "fno_layers": 2,
            },
            "phase_b": {
                "dictionary_size": 25,
                "reduced_state_dimension": 4,
                "dictionary_epochs": 300,
            },
        },
        "FNO": {
            "dictionary_size": 4,
            "reduced_state_dimension": 2,
            "dictionary_hidden_dims": (4,),
            "operator_epochs": 160,
            "dictionary_epochs": 1,
            "batch_size": 32,
            "operator_learning_rate": 7.0e-4,
            "dictionary_learning_rate": 1.0e-3,
            "operator_weight_decay": 2.0e-5,
            "transition_ridge": 1.0e-3,
            "gradient_clip": 5.0,
            "seed": 2026,
            "print_every": 0,
            "fno_hidden_channels": 48,
            "fno_modes": 32,
            "fno_layers": 2,
        },
        "EDMD-DL": {
            "dictionary_size": 25,
            "reduced_state_dimension": 20,
            "hidden_dims": (16, 16),
            "epochs": 120,
            "batch_size": 32,
            "learning_rate": 1.0e-3,
            "transition_ridge": 0.1,
            "seed": 2026,
            "print_every": 0,
        },
        "Kernel EDMD": {
            "reduced_state_dimension": 8,
            "bandwidth_multiplier": 0.2,
            "ridge": 1.0e-10,
        },
        "PCA-VAR": {
            "reduced_state_dimension": 10,
            "ridge": 1.0e-6,
        },
    }


def validate_reproduction_result(result: Mapping[str, Any]) -> None:
    """Validate a cached result before it is reused for plotting."""

    if result.get("schema") != REPRODUCTION_SCHEMA:
        raise ValueError("cached EEG result uses an unsupported schema")
    if result.get("subject") != SUBJECT or result.get("run") != RUN:
        raise ValueError("cached EEG result uses a different record")
    if result.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("cached EEG result uses a different EDF source")

    protocol = result.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("cached EEG result has no protocol")
    expected_protocol = {
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "lag_samples": LAG_SAMPLES,
        "rollout_steps": ROLLOUT_STEPS,
        "step_seconds": STEP_SECONDS,
        "scored_horizon_seconds": HORIZON_SECONDS,
        "n_training_pairs": 104,
        "scored_interval_seconds": [15.0, 19.0],
        "metric": "cumulative relative L2",
    }
    for name, expected in expected_protocol.items():
        if protocol.get(name) != expected:
            raise ValueError(
                f"cached EEG result has an incompatible protocol field {name!r}"
            )

    cached_parameters = json.loads(
        json.dumps(result.get("hyperparameters"), sort_keys=True)
    )
    locked_parameters = json.loads(
        json.dumps(locked_hyperparameters(), sort_keys=True)
    )
    if cached_parameters != locked_parameters:
        raise ValueError("cached EEG result uses different hyperparameters")

    methods = result.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(METHODS):
        raise ValueError("cached EEG result has an incompatible method set")
    for method in METHODS:
        row = methods[method]
        if not isinstance(row, Mapping):
            raise ValueError(f"cached EEG result row is invalid for {method}")
        curve = np.asarray(row.get("relative_l2_by_step"), dtype=float)
        if curve.shape != (ROLLOUT_STEPS,) or not np.all(np.isfinite(curve)):
            raise ValueError(f"cached EEG curve is invalid for {method}")
        endpoint = float(row.get("reproduced_score_relative_l2_4s"))
        if not np.isfinite(endpoint) or not np.isclose(
            endpoint, curve[-1], rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(f"cached EEG endpoint is invalid for {method}")


def _shared_state_map(
    batch: Any,
    *,
    state_dimension: int,
) -> FixedPCAStateMap:
    """Fit the common fixed-PCA state map from training histories only."""

    histories = np.asarray(batch.history, dtype=np.float64)
    return FixedPCAStateMap.fit(
        histories.reshape(len(histories), -1),
        state_dimension=int(state_dimension),
        sample_weights=batch.sample_weights,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_l2_curve(
    prediction: Any,
    split: RestingRolloutSplit,
) -> list[float]:
    restored_prediction = restore_resting_signal_units(
        prediction, split.normalizer
    )
    restored_truth = restore_resting_signal_units(
        split.rollout.future, split.normalizer
    )
    curve: list[float] = []
    for step in range(1, ROLLOUT_STEPS + 1):
        predicted = np.transpose(
            restored_prediction[:, :step], (0, 2, 1, 3)
        ).reshape(len(restored_prediction), 64, step * LAG_SAMPLES)
        target = restored_truth[:, :, : step * LAG_SAMPLES]
        difference = predicted - target
        numerator = np.linalg.norm(difference.reshape(len(difference), -1), axis=1)
        denominator = np.linalg.norm(target.reshape(len(target), -1), axis=1)
        relative_l2 = numerator / np.maximum(denominator, np.finfo(np.float64).eps)
        curve.append(float(relative_l2[0]))
    return curve


def _fit_fno_bso(
    split: RestingRolloutSplit,
    coordinates: np.ndarray,
    *,
    device: str | None,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    phase_a_config = OperatorBSOConfig(**dict(parameters["phase_a"]))
    phase_a_state = _shared_state_map(
        split.batch,
        state_dimension=phase_a_config.reduced_state_dimension,
    )
    phase_a_fit = fit_operator_bso(
        "FNO-BSO",
        split.batch,
        coordinates,
        phase_a_config,
        state_map=phase_a_state,
        device=device,
    )
    phase_b_values = dict(parameters["phase_b"])
    phase_b_config = replace(phase_a_config, **phase_b_values)
    phase_b_state = _shared_state_map(
        split.batch,
        state_dimension=phase_b_config.reduced_state_dimension,
    )
    fit = project_frozen_operator_bso(
        phase_a_fit,
        split.batch,
        phase_b_config,
        state_map=phase_b_state,
        device=device,
    )
    prediction = rollout_operator_bso(
        fit,
        split.rollout.initial_history,
        lag_samples=LAG_SAMPLES,
        n_steps=ROLLOUT_STEPS,
        device=device,
    )
    diagnostics = {
        "phase_a_config": asdict(phase_a_config),
        "phase_b_config": asdict(phase_b_config),
        "phase_a": phase_a_fit.diagnostics,
        "phase_b": fit.diagnostics,
    }
    return np.asarray(prediction, dtype=np.float32), diagnostics


def _fit_fno(
    split: RestingRolloutSplit,
    coordinates: np.ndarray,
    *,
    device: str | None,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    config = OperatorBSOConfig(**dict(parameters))
    state_map = _shared_state_map(
        split.batch,
        state_dimension=config.reduced_state_dimension,
    )
    fit = fit_operator_bso(
        "FNO-BSO",
        split.batch,
        coordinates,
        config,
        state_map=state_map,
        device=device,
    )
    prediction = rollout_phase_a_operator(
        fit,
        split.rollout.initial_history,
        lag_samples=LAG_SAMPLES,
        n_steps=ROLLOUT_STEPS,
        device=device,
    )
    fit_diagnostics = dict(fit.diagnostics)
    fit_diagnostics["fit_container_method"] = fit_diagnostics.get("method")
    fit_diagnostics["fit_container_algorithm"] = fit_diagnostics.get(
        "algorithm"
    )
    fit_diagnostics["method"] = "FNO"
    fit_diagnostics["algorithm"] = (
        "direct Phase-A FNO autoregressive rollout; the fitted dictionary "
        "projection is not used for prediction"
    )
    fit_diagnostics["prediction_path"] = "rollout_phase_a_operator"
    fit_diagnostics["phase_b_used_for_prediction"] = False
    return (
        np.asarray(prediction, dtype=np.float32),
        {"config": asdict(config), "fit": fit_diagnostics},
    )


def _fit_edmd_dl(
    split: RestingRolloutSplit,
    *,
    device: str | None,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    config = EDMDDLTrainingConfig(**dict(parameters))
    state_map = _shared_state_map(
        split.batch,
        state_dimension=int(config.reduced_state_dimension),
    )
    fit = train_paper_edmd_dl(
        split.batch,
        config,
        state_map=state_map,
        device=device,
    )
    prediction = rollout_paper_edmd_dl(
        fit,
        split.rollout.initial_history,
        n_steps=ROLLOUT_STEPS,
        lag_samples=LAG_SAMPLES,
        device=device,
    )
    return (
        np.asarray(prediction, dtype=np.float32),
        {"config": asdict(config), "fit": fit.diagnostics},
    )


def _fit_kernel_edmd(
    split: RestingRolloutSplit,
    *,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    q = int(parameters["reduced_state_dimension"])
    state_map = _shared_state_map(split.batch, state_dimension=q)
    median_fit = fit_kernel_edmd(
        split.batch.history,
        split.batch.next_history,
        state_map=state_map,
        ridge=1.0e-4,
        bandwidth=None,
        sample_weights=split.batch.sample_weights,
    )
    bandwidth = float(median_fit.bandwidth) * float(
        parameters["bandwidth_multiplier"]
    )
    fit = fit_kernel_edmd(
        split.batch.history,
        split.batch.next_history,
        state_map=state_map,
        ridge=float(parameters["ridge"]),
        bandwidth=bandwidth,
        sample_weights=split.batch.sample_weights,
    )
    prediction = fit.rollout(
        split.rollout.initial_history,
        ROLLOUT_STEPS,
        lag_samples=LAG_SAMPLES,
    )
    return (
        np.asarray(prediction, dtype=np.float32),
        {
            "reduced_state_dimension": q,
            "median_bandwidth": float(median_fit.bandwidth),
            "bandwidth_multiplier": float(
                parameters["bandwidth_multiplier"]
            ),
            "bandwidth": bandwidth,
            "ridge": float(parameters["ridge"]),
            "fit": fit.diagnostics,
        },
    )


def _fit_pca_var(
    split: RestingRolloutSplit,
    *,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    q = int(parameters["reduced_state_dimension"])
    state_map = _shared_state_map(split.batch, state_dimension=q)
    fit = fit_pca_var(
        split.batch.history,
        split.batch.next_history,
        state_map=state_map,
        ridge=float(parameters["ridge"]),
        sample_weights=split.batch.sample_weights,
    )
    prediction = fit.rollout(
        split.rollout.initial_history,
        ROLLOUT_STEPS,
        lag_samples=LAG_SAMPLES,
    )
    return (
        np.asarray(prediction, dtype=np.float32),
        {
            "reduced_state_dimension": q,
            "ridge": float(parameters["ridge"]),
            "fit": fit.diagnostics,
        },
    )


def _plot_metric_reproduction(
    result: Mapping[str, Any],
    pdf_path: str | Path,
    *,
    metric_key: str,
    title: str,
    y_max: float,
) -> tuple[Path, Path]:
    """Plot one 40-step metric with the established EEG visual style."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plotting_utils import (
        PUBLICATION_STYLE,
        apply_physical_layout,
        apply_publication_rcparams,
        save_figure_pair,
        style_axis,
        y_from_top,
    )

    colors = {
        "FNO-BSO": "#D1495B",
        "FNO": "#F28E2B",
        "PCA-VAR": "#59A14F",
        "EDMD-DL": "#4E79A7",
        "Kernel EDMD": "#79706E",
    }
    markers = {
        "FNO-BSO": "o",
        "FNO": "s",
        "PCA-VAR": "^",
        "EDMD-DL": "D",
        "Kernel EDMD": "P",
    }
    display_labels = {
        "FNO-BSO": "BSO",
        "FNO": "FNO",
        "PCA-VAR": "PCA-VAR",
        "EDMD-DL": "EDMD-DL",
        "Kernel EDMD": "Kernel EDMD",
    }
    apply_publication_rcparams()
    style = PUBLICATION_STYLE
    steps = np.arange(1, ROLLOUT_STEPS + 1, dtype=int)
    figure, axis = plt.subplots(
        1,
        1,
        figsize=EEG_FIGURE["figsize"],
        dpi=style["dpi"],
        squeeze=True,
        sharex=False,
        sharey=False,
        layout=None,
        facecolor=style["figure_face_color"],
    )
    figure.set_layout_engine(None)
    for method in METHODS:
        values = np.asarray(
            result["methods"][method][metric_key],
            dtype=float,
        )
        if values.shape != (ROLLOUT_STEPS,):
            raise ValueError(
                f"{method} must contain 40 values for {metric_key}"
            )
        axis.plot(
            steps,
            values,
            color=colors[method],
            linestyle="-",
            marker=markers[method],
            markevery=np.arange(4, ROLLOUT_STEPS, 5, dtype=int),
            markersize=style["marker_size"],
            markeredgewidth=style["marker_edge_width"],
            linewidth=(
                style["highlight_line_width"]
                if method == "FNO-BSO"
                else style["line_width"]
            ),
            label=display_labels[method],
            alpha=1.0,
            zorder=3 if method == "FNO-BSO" else 2,
        )

    axis.set_xlim(0.8, ROLLOUT_STEPS + 0.2)
    axis.set_ylim(0.0, float(y_max))
    axis.set_xticks(
        np.asarray((1, 5, 10, 15, 20, 25, 30, 35, 40), dtype=int)
    )
    axis.set_xticks(steps, minor=True)
    axis.set_xlabel(
        f"Rollout step ({STEP_SECONDS:g} s per step)",
        fontsize=style["axis_label_size"],
        fontweight=style["font_weight"],
        labelpad=style["axis_label_pad"],
        color=style["text_color"],
    )
    style_axis(axis, grid_axis="y", grid_visible=True, minor_ticks=True)
    axis.legend(
        frameon=style["legend_frameon"],
        fancybox=style["legend_fancybox"],
        framealpha=style["legend_framealpha"],
        ncol=EEG_FIGURE["legend_ncol"],
        loc="upper center",
        bbox_to_anchor=(0.5, y_from_top(figure, EEG_FIGURE["legend_top_in"])),
        bbox_transform=figure.transFigure,
        borderaxespad=style["legend_borderaxespad"],
        borderpad=style["legend_borderpad"],
        labelspacing=style["legend_labelspacing"],
        columnspacing=style["legend_columnspacing"],
        handlelength=style["legend_handlelength"],
        handleheight=style["legend_handleheight"],
        handletextpad=style["legend_handletextpad"],
        fontsize=style["legend_size"],
    )
    figure.suptitle(
        title,
        fontsize=style["suptitle_size"],
        fontweight=style["title_weight"],
        color=style["text_color"],
        x=0.5,
        y=y_from_top(figure, EEG_FIGURE["suptitle_top_in"]),
        horizontalalignment="center",
        verticalalignment="top",
    )
    apply_physical_layout(figure, EEG_FIGURE)
    pdf, png = save_figure_pair(figure, pdf_path, dpi=style["dpi"])
    plt.close(figure)
    return pdf, png


def plot_reproduction(
    result: Mapping[str, Any],
    pdf_path: str | Path = DEFAULT_FIGURE_PDF,
) -> tuple[Path, Path]:
    """Plot cumulative relative L2 error through each rollout step."""

    validate_reproduction_result(result)
    return _plot_metric_reproduction(
        result,
        pdf_path,
        metric_key="relative_l2_by_step",
        title=r"Multi-step EEG Rollout Forecasting (cumulative relative $L^2$ error)",
        y_max=1.1,
    )


def reproduce(
    *,
    data_root: str | Path = DATA_DIR / "eegmmidb",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    figure_pdf: str | Path = DEFAULT_FIGURE_PDF,
    device: str | None = None,
    download: bool = False,
) -> dict[str, Any]:
    """Fit all five locked methods, save their predictions, and plot them."""

    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run = load_resting_eeg_run(
        SUBJECT,
        RUN,
        data_root=Path(data_root),
        download=download,
    )
    source_path = Path(run.source)
    source_sha256 = _sha256(source_path)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            "R02 EDF hash differs from the fixed source checksum"
        )
    split = make_resting_rollout_split(
        run,
        lag_samples=LAG_SAMPLES,
        train_stride_samples=20,
    )
    parameters = locked_hyperparameters()
    predictions: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}

    predictions["FNO-BSO"], diagnostics["FNO-BSO"] = _fit_fno_bso(
        split,
        run.coordinates,
        device=device,
        parameters=parameters["FNO-BSO"],
    )
    predictions["FNO"], diagnostics["FNO"] = _fit_fno(
        split,
        run.coordinates,
        device=device,
        parameters=parameters["FNO"],
    )
    predictions["EDMD-DL"], diagnostics["EDMD-DL"] = _fit_edmd_dl(
        split,
        device=device,
        parameters=parameters["EDMD-DL"],
    )
    predictions["Kernel EDMD"], diagnostics["Kernel EDMD"] = (
        _fit_kernel_edmd(
            split,
            parameters=parameters["Kernel EDMD"],
        )
    )
    predictions["PCA-VAR"], diagnostics["PCA-VAR"] = _fit_pca_var(
        split,
        parameters=parameters["PCA-VAR"],
    )

    method_results: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        curve = _relative_l2_curve(predictions[method], split)
        reproduced_score = float(curve[-1])
        reference_score = float(REFERENCE_SCORES[method])
        method_results[method] = {
            "reference_score_relative_l2_4s": reference_score,
            "reproduced_score_relative_l2_4s": reproduced_score,
            "absolute_reference_difference": abs(
                reproduced_score - reference_score
            ),
            "relative_l2_by_step": curve,
            "diagnostics": diagnostics[method],
        }

    bso_score = float(
        method_results["FNO-BSO"]["reproduced_score_relative_l2_4s"]
    )
    reductions = {
        method: (
            0.0
            if method == "FNO-BSO"
            else 100.0
            * (
                float(
                    method_results[method][
                        "reproduced_score_relative_l2_4s"
                    ]
                )
                - bso_score
            )
            / float(
                method_results[method][
                    "reproduced_score_relative_l2_4s"
                ]
            )
        )
        for method in METHODS
    }
    best_non_bso = min(
        METHODS[1:],
        key=lambda method: float(
            method_results[method]["reproduced_score_relative_l2_4s"]
        ),
    )
    result = {
        "schema": REPRODUCTION_SCHEMA,
        "subject": SUBJECT,
        "run": RUN,
        "source": str(source_path),
        "source_sha256": source_sha256,
        "protocol": {
            "training_interval_seconds": [0.0, 15.0],
            "training_interval_role": (
                "chronological split envelope; actual lag-pair support and "
                "origins are reported separately"
            ),
            "normalizer_fit_interval_seconds": [1.0, 15.0],
            "train_origin_start_sample": int(
                split.train_origin_samples[0]
            ),
            "train_origin_end_sample": int(
                split.train_origin_samples[-1]
            ),
            "train_origin_stride_samples": 20,
            "train_origin_interval_seconds": [
                float(split.train_origin_samples[0] / SAMPLING_RATE_HZ),
                float(split.train_origin_samples[-1] / SAMPLING_RATE_HZ),
            ],
            "effective_training_pair_support_seconds": [1.0, 14.975],
            "n_training_pairs": int(len(split.train_origin_samples)),
            "scored_interval_seconds": [15.0, 19.0],
            "continuous_boundary_rollout": True,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "lag_samples": LAG_SAMPLES,
            "step_seconds": STEP_SECONDS,
            "rollout_steps": ROLLOUT_STEPS,
            "scored_horizon_seconds": HORIZON_SECONDS,
            "metric": "cumulative relative L2",
        },
        "hyperparameters": parameters,
        "methods": method_results,
        "bso_reduction_percent": reductions,
        "best_non_bso_method": best_non_bso,
        "bso_reduction_percent_vs_best_non_bso": reductions[best_non_bso],
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = output / "reproduction.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "predictions.npz",
        truth=np.asarray(split.rollout.future, dtype=np.float32),
        **{
            method.lower().replace(" ", "_").replace("-", "_"): prediction
            for method, prediction in predictions.items()
        },
    )
    figure_paths = plot_reproduction(result, figure_pdf)
    result["artifacts"] = {
        "result_json": str(result_path),
        "predictions_npz": str(output / "predictions.npz"),
        "figure_pdf": str(figure_paths[0]),
        "figure_png": str(figure_paths[1]),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_DIR / "eegmmidb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--figure-pdf",
        type=Path,
        default=DEFAULT_FIGURE_PDF,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download S029R02 through MNE when the EDF is not present",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    payload = reproduce(
        data_root=arguments.data_root,
        output_dir=arguments.output_dir,
        figure_pdf=arguments.figure_pdf,
        device=arguments.device,
        download=arguments.download,
    )
    for method in METHODS:
        row = payload["methods"][method]
        score = row["reproduced_score_relative_l2_4s"]
        reduction = payload["bso_reduction_percent"][method]
        print(f"{method:12s} {score:.9f}  BSO reduction {reduction:.3f}%")
