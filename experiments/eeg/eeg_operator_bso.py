"""Autonomous two-phase neural-operator BSO for EEG delay states.

The implementation follows the project's operator-then-projection structure:

Phase A
    an FNO learns the full field map
    ``(64, 160) history -> (64, 160) shifted next history``;

Phase B
    the operator is frozen, its full-field predictions are mapped through the
    common fixed PCA state definition, and a dictionary
    ``Psi(z) = [1, z, h_theta(z)]`` is trained by the autonomous Galerkin/EDMD
    invariance objective.

The total dictionary size and fixed PCA state dimension are shared with the
EDMD-DL baseline.  Multi-step forecasts read the fixed ``z`` coordinate block
from ``Psi K^n`` and use only the fixed inverse PCA map.  There is no control
matrix, autoencoder, learned decoder/readout, whitening, forecast loss,
anti-collapse penalty, or spectral clipping.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .eeg_edmddl import (
    FULL_STATE_DIMENSION,
    FixedPCAStateMap,
    PaperEDMDDLDictionary,
)
from .eeg_resting_data import HISTORY_SAMPLES


BSO_METHODS = ("FNO-BSO",)
N_ELECTRODES = 64
HISTORY_LENGTH = HISTORY_SAMPLES


@dataclass
class OperatorBSOConfig:
    """Settings for the two independently trained BSO phases."""

    # Shared with EDMD-DL: 1 constant + q fixed coordinates + m learned.
    dictionary_size: int = 20
    reduced_state_dimension: int = 8
    dictionary_hidden_dims: tuple[int, ...] = (16, 16)

    operator_epochs: int = 120
    dictionary_epochs: int = 200
    batch_size: int = 64
    operator_learning_rate: float = 5.0e-4
    dictionary_learning_rate: float = 1.0e-3
    operator_weight_decay: float = 2.0e-5
    transition_ridge: float = 1.0e-6
    gradient_clip: float = 5.0
    seed: int = 2026
    print_every: int = 25

    # FNO spectral weights are complex, so both tensor-element and real
    # scalar parameter counts are reported in the fit diagnostics.
    fno_hidden_channels: int = 24
    fno_modes: int = 12
    fno_layers: int = 2


class TemporalFNOStateOperator(nn.Module):
    """One-dimensional FNO over time with electrodes as field channels."""

    def __init__(
        self,
        n_electrodes: int,
        history_length: int,
        *,
        hidden_channels: int,
        n_modes: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        if min(
            int(n_electrodes),
            int(history_length),
            int(hidden_channels),
            int(n_modes),
            int(n_layers),
        ) < 1:
            raise ValueError("all FNO dimensions must be positive")
        from neuralop.models import FNO

        self.n_electrodes = int(n_electrodes)
        self.history_length = int(history_length)
        self.core = FNO(
            n_modes=(
                min(int(n_modes), self.history_length // 2 + 1),
            ),
            in_channels=self.n_electrodes,
            out_channels=self.n_electrodes,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
            positional_embedding="grid",
        )

    def forward(self, field: Tensor) -> Tensor:
        _require_field_tensor(
            field, self.n_electrodes, self.history_length, "field"
        )
        return self.core(field)


@dataclass
class OperatorBSOFit:
    """Fitted BSO; intentionally has no decoder or control fields."""

    method: str
    operator: nn.Module
    dictionary: PaperEDMDDLDictionary
    state_map: FixedPCAStateMap
    transition: np.ndarray
    training_history: dict[str, list[dict[str, float]]]
    diagnostics: dict[str, Any]
    config: dict[str, Any]
    device: str = "cpu"

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "eeg-autonomous-neural-operator-bso-v1",
            "method": self.method,
            "operator_state": {
                name: value.detach().cpu()
                for name, value in self.operator.state_dict().items()
            },
            "dictionary_state": {
                name: value.detach().cpu()
                for name, value in self.dictionary.state_dict().items()
            },
            "state_map": self.state_map.to_payload(),
            "transition": np.asarray(self.transition, dtype=np.float64),
            "training_history": copy.deepcopy(self.training_history),
            "diagnostics": copy.deepcopy(self.diagnostics),
            "config": copy.deepcopy(self.config),
        }
        torch.save(payload, output)
        return output


def _set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _require_field_tensor(
    values: Tensor,
    n_electrodes: int,
    history_length: int,
    name: str,
) -> None:
    if values.ndim != 3 or tuple(values.shape[1:]) != (
        int(n_electrodes),
        int(history_length),
    ):
        raise ValueError(
            f"{name} must have shape "
            f"(batch,{n_electrodes},{history_length})"
        )
    if not torch.is_floating_point(values) or not torch.isfinite(
        values
    ).all():
        raise ValueError(f"{name} must be a finite floating-point tensor")


def _normalized_weights(values: Any, n_samples: int) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        weights.shape != (int(n_samples),)
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError(
            "sample_weights must be finite, nonnegative, and have one "
            "entry per training pair"
        )
    return weights / float(weights.sum())


def _validated_pair_data(
    batch: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = np.asarray(batch.history, dtype=np.float64)
    target = np.asarray(batch.next_history, dtype=np.float64)
    if current.ndim != 3 or current.shape[1:] != (
        N_ELECTRODES,
        HISTORY_LENGTH,
    ):
        raise ValueError(
            "history must have shape "
            f"(N,{N_ELECTRODES},{HISTORY_LENGTH})"
        )
    if (
        target.shape != current.shape
        or len(current) < 3
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError(
            "next_history must be finite, match history, and contain at "
            "least three training pairs"
        )
    weights = _normalized_weights(batch.sample_weights, len(current))
    return current, target, weights


def _validate_config(
    config: OperatorBSOConfig,
    n_training_samples: int,
) -> None:
    integer_positive = (
        "dictionary_size",
        "reduced_state_dimension",
        "operator_epochs",
        "dictionary_epochs",
        "batch_size",
        "fno_hidden_channels",
        "fno_modes",
        "fno_layers",
    )
    for name in integer_positive:
        if int(getattr(config, name)) < 1:
            raise ValueError(f"{name} must be positive")
    q = int(config.reduced_state_dimension)
    total = int(config.dictionary_size)
    if q > int(n_training_samples) - 1:
        raise ValueError(
            "reduced_state_dimension exceeds the rank available from "
            "current training histories"
        )
    if total < q + 2:
        raise ValueError(
            "dictionary_size must contain one constant, all q fixed "
            "state coordinates, and at least one learned function"
        )
    if any(int(width) < 1 for width in config.dictionary_hidden_dims):
        raise ValueError(
            "dictionary_hidden_dims must contain positive widths"
        )
    for name in (
        "operator_learning_rate",
        "dictionary_learning_rate",
    ):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name in (
        "operator_weight_decay",
        "transition_ridge",
        "gradient_clip",
    ):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")


def build_state_operator(
    method: str,
    *,
    coordinates: Any,
    config: OperatorBSOConfig,
) -> nn.Module:
    """Build an autonomous full-field operator with no cue inputs."""

    if method == "FNO-BSO":
        return TemporalFNOStateOperator(
            N_ELECTRODES,
            HISTORY_LENGTH,
            hidden_channels=int(config.fno_hidden_channels),
            n_modes=int(config.fno_modes),
            n_layers=int(config.fno_layers),
        )
    raise ValueError(f"unknown BSO method {method!r}; expected {BSO_METHODS}")


def _weighted_relative_field_loss(
    prediction: Tensor,
    target: Tensor,
    sample_weights: Tensor,
) -> Tensor:
    difference = (prediction - target).flatten(1)
    target_flat = target.flatten(1)
    relative = (
        torch.sum(difference.square(), dim=1)
        / torch.sum(target_flat.square(), dim=1).clamp_min(1.0e-8)
    )
    normalized = sample_weights / sample_weights.sum().clamp_min(1.0e-12)
    return torch.sum(normalized * relative)


def _train_phase_a(
    operator: nn.Module,
    current: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    config: OperatorBSOConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    current_tensor = torch.as_tensor(current, dtype=torch.float32)
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    weight_tensor = torch.as_tensor(sample_weights, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        operator.parameters(),
        lr=float(config.operator_learning_rate),
        weight_decay=float(config.operator_weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config.operator_epochs)
    )
    rng = np.random.default_rng(int(config.seed))
    history: list[dict[str, float]] = []
    for epoch in range(int(config.operator_epochs)):
        operator.train()
        order = rng.permutation(len(current))
        total = 0.0
        seen = 0
        for start in range(0, len(order), int(config.batch_size)):
            indices = order[start : start + int(config.batch_size)]
            index = torch.as_tensor(indices, dtype=torch.long)
            inputs = current_tensor[index].to(device)
            targets = target_tensor[index].to(device)
            weights = weight_tensor[index].to(device)
            prediction = operator(inputs)
            loss = _weighted_relative_field_loss(
                prediction, targets, weights
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(config.gradient_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    operator.parameters(), float(config.gradient_clip)
                )
            optimizer.step()
            count = len(indices)
            total += float(loss.detach()) * count
            seen += count
        scheduler.step()
        record = {
            "epoch": float(epoch + 1),
            "weighted_relative_field_mse": total / max(seen, 1),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)
        if config.print_every and (
            epoch == 0
            or epoch + 1 == int(config.operator_epochs)
            or (epoch + 1) % int(config.print_every) == 0
        ):
            print(
                f"[{operator.__class__.__name__}:Phase A] "
                f"epoch {epoch + 1:4d}/{config.operator_epochs}: "
                f"field={record['weighted_relative_field_mse']:.6e}"
            )
    return history


def _operator_predictions(
    operator: nn.Module,
    histories: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    operator.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(histories), int(batch_size)):
            values = torch.as_tensor(
                histories[start : start + int(batch_size)],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(
                operator(values).detach().cpu().numpy().astype(np.float64)
            )
    return np.concatenate(outputs, axis=0)


def _dictionary_values(
    dictionary: PaperEDMDDLDictionary,
    states: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    was_training = dictionary.training
    dictionary.eval()
    with torch.no_grad():
        values = (
            dictionary(
                torch.as_tensor(
                    states, dtype=torch.float32, device=device
                )
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    dictionary.train(was_training)
    values[:, 0] = 1.0
    return values


def _galerkin_transition(
    current: np.ndarray,
    target: np.ndarray,
    *,
    sample_weights: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if current.ndim != 2 or current.shape != target.shape:
        raise ValueError("current and target dictionaries must match")
    if not (
        np.array_equal(current[:, 0], np.ones(len(current)))
        and np.array_equal(target[:, 0], np.ones(len(target)))
    ):
        raise ValueError("the leading dictionary function must be exact one")
    regularization = float(ridge)
    weights = _normalized_weights(sample_weights, len(current))
    gram = current.T @ (current * weights[:, None])
    # Use the declared Galerkin/EDMD projection without an extra eigenvalue
    # constraint or post-processing.
    cross = current.T @ (target * weights[:, None])
    system = gram + regularization * np.eye(current.shape[1])
    try:
        transition = np.linalg.solve(system, cross)
        solver = "solve"
    except np.linalg.LinAlgError:
        transition = np.linalg.pinv(system) @ cross
        solver = "pinv"
    residual = current @ transition - target
    residual_squared_sum = float(
        np.sum(weights[:, None] * residual**2)
    )
    return np.asarray(transition, dtype=np.float64), {
        "convention": "Psi(z_operator_next) ~= Psi(z) @ K",
        "autonomous": True,
        "constant_dictionary_function_is_fixed": True,
        "constant_transition_column_is_constrained": False,
        "ridge": regularization,
        "linear_solver": solver,
        "weighted_invariance_squared_sum": residual_squared_sum,
        "weighted_invariance_mse": (
            residual_squared_sum / current.shape[1]
        ),
        "ridge_term_full_matrix": (
            regularization * float(np.sum(transition**2))
        ),
        "gram_condition": float(np.linalg.cond(gram)),
        "gram_rank": int(np.linalg.matrix_rank(gram)),
        "n_training_pairs": int(len(current)),
        "dictionary_size": int(current.shape[1]),
        "transition_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(transition)))
        ),
        "regularized_system_condition": float(np.linalg.cond(system)),
        "spectral_clipping": False,
    }


def _train_phase_b(
    dictionary: PaperEDMDDLDictionary,
    current_state: np.ndarray,
    operator_target_state: np.ndarray,
    sample_weights: np.ndarray,
    config: OperatorBSOConfig,
    device: torch.device,
) -> tuple[
    np.ndarray,
    list[dict[str, float]],
    dict[str, Any],
]:
    optimizer = torch.optim.Adam(
        dictionary.parameters(),
        lr=float(config.dictionary_learning_rate),
    )
    current_tensor = torch.as_tensor(
        current_state, dtype=torch.float32
    )
    target_tensor = torch.as_tensor(
        operator_target_state, dtype=torch.float32
    )
    weight_tensor = torch.as_tensor(
        sample_weights, dtype=torch.float32
    )
    rng = np.random.default_rng(int(config.seed))
    history: list[dict[str, float]] = []
    transition_diagnostics: dict[str, Any] = {}
    transition = np.eye(int(config.dictionary_size), dtype=np.float64)
    for epoch in range(int(config.dictionary_epochs)):
        phi_current = _dictionary_values(
            dictionary, current_state, device
        )
        phi_target = _dictionary_values(
            dictionary, operator_target_state, device
        )
        transition, transition_diagnostics = _galerkin_transition(
            phi_current,
            phi_target,
            sample_weights=sample_weights,
            ridge=float(config.transition_ridge),
        )
        transition_tensor = torch.as_tensor(
            transition, dtype=torch.float32, device=device
        )

        dictionary.train()
        order = rng.permutation(len(current_state))
        residual_sum = 0.0
        residual_weight = 0.0
        for start in range(0, len(order), int(config.batch_size)):
            indices = order[start : start + int(config.batch_size)]
            index = torch.as_tensor(indices, dtype=torch.long)
            state = current_tensor[index].to(device)
            target = target_tensor[index].to(device)
            weights = weight_tensor[index].to(device)
            current_dictionary = dictionary(state)
            target_dictionary = dictionary(target)
            residual_per_sample = torch.mean(
                (
                    current_dictionary @ transition_tensor
                    - target_dictionary
                ).square(),
                dim=1,
            )
            loss = torch.sum(
                weights * residual_per_sample
            ) / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            residual_sum += float(
                torch.sum(
                    weights * residual_per_sample.detach()
                ).cpu()
            )
            residual_weight += float(weights.sum().detach().cpu())
        record = {
            "epoch": float(epoch + 1),
            "weighted_invariance_mse": (
                residual_sum / max(residual_weight, 1.0e-15)
            ),
        }
        history.append(record)
        if config.print_every and (
            epoch == 0
            or epoch + 1 == int(config.dictionary_epochs)
            or (epoch + 1) % int(config.print_every) == 0
        ):
            print(
                f"[Phase B] epoch {epoch + 1:4d}/"
                f"{config.dictionary_epochs}: "
                f"invariance="
                f"{record['weighted_invariance_mse']:.6e}"
            )

    dictionary.eval()
    transition, transition_diagnostics = _galerkin_transition(
        _dictionary_values(dictionary, current_state, device),
        _dictionary_values(dictionary, operator_target_state, device),
        sample_weights=sample_weights,
        ridge=float(config.transition_ridge),
    )
    return transition, history, transition_diagnostics


def _parameter_count(module: nn.Module) -> int:
    return int(
        sum(parameter.numel() for parameter in module.parameters())
    )


def _real_scalar_parameter_count(module: nn.Module) -> int:
    """Count real scalar degrees of freedom, expanding complex tensors."""

    return int(
        sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in module.parameters()
        )
    )


def _training_pair_fingerprint(
    current: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("current", current),
        ("target", target),
        ("weights", sample_weights),
    ):
        array = np.ascontiguousarray(values)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _operator_state_fingerprint(operator: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(operator.state_dict().items()):
        digest.update(name.encode("utf-8"))
        if isinstance(value, Tensor):
            array = np.ascontiguousarray(
                value.detach().cpu().numpy()
            )
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(array.tobytes())
        else:
            # neuraloperator stores constructor metadata in ``_extra_state``.
            # It is immutable configuration rather than a trainable tensor.
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def fit_operator_bso(
    method: str,
    batch: Any,
    electrode_coordinates: Any,
    config: OperatorBSOConfig | None = None,
    *,
    state_map: FixedPCAStateMap | None = None,
    device: str | torch.device | None = None,
) -> OperatorBSOFit:
    """Fit Phase A, freeze it, then fit the finite Galerkin projection."""

    if method not in BSO_METHODS:
        raise ValueError(f"method must be one of {BSO_METHODS}")
    cfg = copy.deepcopy(config or OperatorBSOConfig())
    current, target, sample_weights = _validated_pair_data(batch)
    _validate_config(cfg, len(current))
    coordinates = np.asarray(electrode_coordinates, dtype=np.float32)
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] != N_ELECTRODES
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError(
            "electrode_coordinates must be finite with one row per channel"
        )
    target_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _set_seed(int(cfg.seed))

    if state_map is None:
        state_map = FixedPCAStateMap.fit(
            current.reshape(len(current), FULL_STATE_DIMENSION),
            state_dimension=int(cfg.reduced_state_dimension),
            sample_weights=sample_weights,
        )
        state_map_source = "fit_from_current_training_histories"
    else:
        if not isinstance(state_map, FixedPCAStateMap):
            raise TypeError("state_map must be a FixedPCAStateMap")
        state_map.validate()
        if state_map.full_dimension != FULL_STATE_DIMENSION:
            raise ValueError(
                "provided state_map has the wrong full state dimension"
            )
        if state_map.state_dimension != int(
            cfg.reduced_state_dimension
        ):
            raise ValueError(
                "provided state_map dimension differs from the BSO config"
            )
        state_map_source = "shared_training_state_map"
    current_state = state_map.transform(current)

    operator = build_state_operator(
        method, coordinates=coordinates, config=cfg
    ).to(target_device)
    phase_a_history = _train_phase_a(
        operator,
        current,
        target,
        sample_weights,
        cfg,
        target_device,
    )
    operator.eval()
    for parameter in operator.parameters():
        parameter.requires_grad_(False)
    phase_a_state_sha256 = _operator_state_fingerprint(operator)
    operator_target_history = _operator_predictions(
        operator,
        current,
        batch_size=int(cfg.batch_size),
        device=target_device,
    )
    operator_target_state = state_map.transform(
        operator_target_history
    )

    # Reset before Phase B so EDMD-DL and FNO-BSO start the common dictionary
    # architecture from the same initialization. Phase A has already been
    # frozen, so this cannot alter the learned field operator.
    _set_seed(int(cfg.seed))
    dictionary = PaperEDMDDLDictionary(
        state_dimension=int(cfg.reduced_state_dimension),
        dictionary_size=int(cfg.dictionary_size),
        hidden_dims=tuple(cfg.dictionary_hidden_dims),
    ).to(target_device)
    (
        transition,
        phase_b_history,
        transition_diagnostics,
    ) = _train_phase_b(
        dictionary,
        current_state,
        operator_target_state,
        sample_weights,
        cfg,
        target_device,
    )
    dictionary.eval()
    true_next_state = state_map.transform(target)
    operator_state_relative_error = float(
        np.linalg.norm(operator_target_state - true_next_state)
        / max(np.linalg.norm(true_next_state), 1.0e-15)
    )
    q = int(cfg.reduced_state_dimension)
    m = int(cfg.dictionary_size) - 1 - q
    diagnostics: dict[str, Any] = {
        "method": method,
        "algorithm": (
            "Phase A full-field neural operator; frozen Phase B "
            "Galerkin/EDMD dictionary projection"
        ),
        "autonomous": True,
        "phase_a_map": (
            "64x160 current history -> shifted 64x160 next history"
        ),
        "phase_b_target": "z(F_operator(history))",
        "operator_frozen_before_projection": True,
        "training_pair_fingerprint": _training_pair_fingerprint(
            current, target, sample_weights
        ),
        "phase_a_operator_state_sha256": phase_a_state_sha256,
        "phase_a_reused": False,
        "uses_control_input": False,
        "has_control_matrix": False,
        "uses_autoencoder": False,
        "uses_trainable_decoder": False,
        "uses_readout_regression": False,
        "uses_reconstruction_loss": False,
        "uses_forecast_loss": False,
        "uses_multistep_loss": False,
        "uses_anti_collapse": False,
        "uses_whitening": False,
        "uses_spectral_clipping": False,
        "full_delay_state_dimension": FULL_STATE_DIMENSION,
        "state_definition": (
            "fixed weighted PCA of flattened 64x160 current training "
            "histories; identical definition to EDMD-DL"
        ),
        "state_map_source": state_map_source,
        "reduced_state_dimension": q,
        "dictionary_size": int(cfg.dictionary_size),
        "fixed_constant_functions": 1,
        "fixed_state_coordinate_functions": q,
        "learned_dictionary_functions": m,
        "state_inverse": (
            "fixed inverse PCA map; no fitted or trainable decoder"
        ),
        "operator_parameters": _parameter_count(operator),
        "operator_real_scalar_parameters": (
            _real_scalar_parameter_count(operator)
        ),
        "dictionary_trainable_parameters": _parameter_count(dictionary),
        "operator_target_state_relative_l2": (
            operator_state_relative_error
        ),
        "transition": transition_diagnostics,
    }
    return OperatorBSOFit(
        method=method,
        operator=operator.eval(),
        dictionary=dictionary.eval(),
        state_map=state_map,
        transition=transition,
        training_history={
            "phase_a_operator": phase_a_history,
            "phase_b_dictionary": phase_b_history,
        },
        diagnostics=diagnostics,
        config=asdict(cfg),
        device=str(target_device),
    )


def project_frozen_operator_bso(
    phase_a_fit: OperatorBSOFit,
    batch: Any,
    config: OperatorBSOConfig,
    *,
    state_map: FixedPCAStateMap | None = None,
    device: str | torch.device | None = None,
) -> OperatorBSOFit:
    """Fit Phase B while keeping the supplied Phase-A map fixed."""

    if not isinstance(phase_a_fit, OperatorBSOFit):
        raise TypeError("phase_a_fit must be an OperatorBSOFit")
    assert_operator_bso_fit(phase_a_fit)
    cfg = copy.deepcopy(config)
    current, target, sample_weights = _validated_pair_data(batch)
    _validate_config(cfg, len(current))
    pair_fingerprint = _training_pair_fingerprint(
        current, target, sample_weights
    )
    if pair_fingerprint != phase_a_fit.diagnostics.get(
        "training_pair_fingerprint"
    ):
        raise ValueError(
            "the frozen Phase-A operator may be reused only with the "
            "identical training pairs and weights"
        )
    target_device = torch.device(
        device if device is not None else phase_a_fit.device
    )
    operator = phase_a_fit.operator.to(target_device).eval()
    if any(parameter.requires_grad for parameter in operator.parameters()):
        raise ValueError(
            "phase_a_fit operator must already be frozen before reuse"
        )
    operator_state_sha256 = _operator_state_fingerprint(operator)
    if operator_state_sha256 != phase_a_fit.diagnostics.get(
        "phase_a_operator_state_sha256"
    ):
        raise ValueError("the frozen Phase-A operator state has changed")

    selected_state_map = (
        phase_a_fit.state_map if state_map is None else state_map
    )
    if not isinstance(selected_state_map, FixedPCAStateMap):
        raise TypeError("state_map must be a FixedPCAStateMap")
    selected_state_map.validate()
    if (
        selected_state_map.full_dimension != FULL_STATE_DIMENSION
        or selected_state_map.state_dimension
        != int(cfg.reduced_state_dimension)
    ):
        raise ValueError(
            "state_map dimensions do not match the reused BSO config"
        )

    current_state = selected_state_map.transform(current)
    operator_target_history = _operator_predictions(
        operator,
        current,
        batch_size=int(cfg.batch_size),
        device=target_device,
    )
    operator_target_state = selected_state_map.transform(
        operator_target_history
    )

    _set_seed(int(cfg.seed))
    dictionary = PaperEDMDDLDictionary(
        state_dimension=int(cfg.reduced_state_dimension),
        dictionary_size=int(cfg.dictionary_size),
        hidden_dims=tuple(cfg.dictionary_hidden_dims),
    ).to(target_device)
    (
        transition,
        phase_b_history,
        transition_diagnostics,
    ) = _train_phase_b(
        dictionary,
        current_state,
        operator_target_state,
        sample_weights,
        cfg,
        target_device,
    )
    dictionary.eval()
    true_next_state = selected_state_map.transform(target)
    operator_state_relative_error = float(
        np.linalg.norm(operator_target_state - true_next_state)
        / max(np.linalg.norm(true_next_state), 1.0e-15)
    )
    q = int(cfg.reduced_state_dimension)
    m = int(cfg.dictionary_size) - 1 - q
    diagnostics = copy.deepcopy(phase_a_fit.diagnostics)
    diagnostics.update(
        {
            "phase_a_reused": True,
            "phase_a_source_dictionary_size": int(
                phase_a_fit.diagnostics["dictionary_size"]
            ),
            "operator_frozen_before_projection": True,
            "state_map_source": "shared_training_state_map",
            "reduced_state_dimension": q,
            "dictionary_size": int(cfg.dictionary_size),
            "fixed_constant_functions": 1,
            "fixed_state_coordinate_functions": q,
            "learned_dictionary_functions": m,
            "dictionary_trainable_parameters": _parameter_count(
                dictionary
            ),
            "operator_target_state_relative_l2": (
                operator_state_relative_error
            ),
            "transition": transition_diagnostics,
        }
    )
    if _operator_state_fingerprint(operator) != operator_state_sha256:
        raise RuntimeError(
            "Phase-B dictionary training mutated the frozen operator"
        )
    return OperatorBSOFit(
        method=phase_a_fit.method,
        operator=operator,
        dictionary=dictionary,
        state_map=selected_state_map,
        transition=transition,
        training_history={
            "phase_a_operator": [],
            "phase_b_dictionary": phase_b_history,
        },
        diagnostics=diagnostics,
        config=asdict(cfg),
        device=str(target_device),
    )


def _validated_initial_histories(values: Any) -> np.ndarray:
    histories = np.asarray(values, dtype=np.float64)
    if (
        histories.ndim != 3
        or histories.shape[1:] != (N_ELECTRODES, HISTORY_LENGTH)
        or len(histories) == 0
        or not np.all(np.isfinite(histories))
    ):
        raise ValueError(
            "initial_history must be a finite array shaped "
            f"(trials,{N_ELECTRODES},{HISTORY_LENGTH})"
        )
    return histories


def _validated_steps(n_steps: Any) -> int:
    try:
        steps = int(n_steps)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("n_steps must be a nonnegative integer") from error
    if steps < 0 or (
        isinstance(n_steps, (float, np.floating))
        and float(n_steps) != steps
    ):
        raise ValueError("n_steps must be a nonnegative integer")
    return steps


def _validated_lag_samples(lag_samples: Any) -> int:
    try:
        lag = int(lag_samples)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "lag_samples must be a positive history length"
        ) from error
    if lag < 1 or lag > HISTORY_LENGTH or (
        isinstance(lag_samples, (float, np.floating))
        and float(lag_samples) != lag
    ):
        raise ValueError("lag_samples must be a positive history length")
    return lag


def rollout_operator_bso(
    fit: OperatorBSOFit,
    initial_history: Any,
    *,
    n_steps: int,
    lag_samples: int,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Roll out ``Psi K^n`` and reconstruct only through fixed inverse PCA."""

    histories = _validated_initial_histories(initial_history)
    steps = _validated_steps(n_steps)
    lag = _validated_lag_samples(lag_samples)
    if steps == 0:
        return np.empty(
            (len(histories), 0, N_ELECTRODES, lag),
            dtype=np.float64,
        )
    target_device = torch.device(
        device if device is not None else fit.device
    )
    state = fit.state_map.transform(histories)
    features = _dictionary_values(
        fit.dictionary.to(target_device), state, target_device
    )
    state_start = 1
    state_stop = 1 + fit.state_map.state_dimension
    outputs: list[np.ndarray] = []
    for _ in range(steps):
        features = features @ np.asarray(
            fit.transition, dtype=np.float64
        )
        predicted_state = features[:, state_start:state_stop]
        predicted_history = fit.state_map.inverse_transform(
            predicted_state
        )
        outputs.append(predicted_history[:, :, -lag:].copy())
    return np.stack(outputs, axis=1)


def rollout_phase_a_operator(
    fit: OperatorBSOFit,
    initial_history: Any,
    *,
    n_steps: int,
    lag_samples: int,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Autoregressive rollout of the frozen full-field operator."""

    histories = _validated_initial_histories(initial_history)
    steps = _validated_steps(n_steps)
    lag = _validated_lag_samples(lag_samples)
    if steps == 0:
        return np.empty(
            (len(histories), 0, N_ELECTRODES, lag),
            dtype=np.float32,
        )
    target_device = torch.device(
        device if device is not None else fit.device
    )
    current = torch.as_tensor(
        histories, dtype=torch.float32, device=target_device
    )
    fit.operator.to(target_device).eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(steps):
            current = fit.operator(current)
            outputs.append(
                current[:, :, -lag:].detach().cpu().numpy()
            )
    return np.stack(outputs, axis=1).astype(np.float32)


def assert_operator_bso_fit(fit: OperatorBSOFit) -> None:
    """Validate the structural contract of an operator-BSO fit."""

    required_false = (
        "uses_control_input",
        "has_control_matrix",
        "uses_autoencoder",
        "uses_trainable_decoder",
        "uses_readout_regression",
        "uses_reconstruction_loss",
        "uses_forecast_loss",
        "uses_multistep_loss",
        "uses_anti_collapse",
        "uses_whitening",
        "uses_spectral_clipping",
    )
    for name in required_false:
        if bool(fit.diagnostics.get(name, True)):
            raise ValueError(f"BSO contract field {name!r} must be false")
    if not bool(
        fit.diagnostics.get("operator_frozen_before_projection", False)
    ):
        raise ValueError("BSO operator was not frozen before projection")
    expected = (
        1
        + int(fit.diagnostics["fixed_state_coordinate_functions"])
        + int(fit.diagnostics["learned_dictionary_functions"])
    )
    if expected != int(fit.diagnostics["dictionary_size"]):
        raise ValueError("BSO dictionary accounting is inconsistent")
    if fit.transition.shape != (expected, expected):
        raise ValueError("BSO transition dimension is inconsistent")
    if hasattr(fit, "decoder") or hasattr(fit, "control_matrix"):
        raise ValueError("BSO fit unexpectedly exposes decoder/control state")


__all__ = [
    "BSO_METHODS",
    "OperatorBSOConfig",
    "OperatorBSOFit",
    "TemporalFNOStateOperator",
    "assert_operator_bso_fit",
    "build_state_operator",
    "fit_operator_bso",
    "project_frozen_operator_bso",
    "rollout_operator_bso",
    "rollout_phase_a_operator",
]
