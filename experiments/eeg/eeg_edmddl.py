"""Autonomous EDMD with dictionary learning for the EEG benchmark.

This module implements the alternating EDMD-DL objective of Li et al.,
arXiv:1707.00225, on an explicitly defined reduced EEG delay state:

    z = P (x - mean),
    Psi(z; theta) = [1, z_1, ..., z_q, psi_1(z; theta), ..., psi_m(z; theta)],
    min_{K, theta} ||Psi(z_next) - Psi(z) K||_W^2
                   + lambda ||K||_F^2.

The 64-by-160 delay state has 10,240 coordinates.  A literal application of
the paper would therefore require at least 10,241 dictionary functions before
adding a single learned function.  The default experiment instead defines a
fixed weighted-PCA state ``z`` and applies EDMD-DL to that reduced state.  This
is an explicit modelling choice, not a learned autoencoder:

* the PCA map is fitted once from current training states only;
* the constant and every reduced-state coordinate are fixed functions;
* only the additional dictionary functions are trained;
* there is no control input, neural decoder, reconstruction loss, forecast
  loss, multistep loss, whitening, anti-collapse term, or spectral clipping;
* full EEG histories are recovered only by the fixed inverse PCA state map.

Consequently, results from this module must be described as EDMD-DL on a
PCA-reduced delay state, not as full 10,240-dimensional EDMD-DL.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .eeg_resting_data import CANONICAL_CHANNELS, HISTORY_SAMPLES


N_ELECTRODES = len(CANONICAL_CHANNELS)
HISTORY_LENGTH = HISTORY_SAMPLES
FULL_STATE_DIMENSION = N_ELECTRODES * HISTORY_LENGTH
EDMD_DL_NAME = "EDMD-DL"
EDMD_DL_LONG_NAME = "EDMD with dictionary learning"
EDMD_DL_VARIANT = "EDMD-DL on training-PCA state"
EDMD_DL_REFERENCE = "https://arxiv.org/abs/1707.00225"


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _normalized_weights(values: Any, n_samples: int) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (int(n_samples),):
        raise ValueError(
            f"sample_weights must have shape ({n_samples},), "
            f"got {weights.shape}"
        )
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError(
            "sample_weights must be finite, nonnegative, and have "
            "positive sum"
        )
    return weights / float(weights.sum())


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    """Remove the arbitrary sign choice of each SVD component."""

    result = np.asarray(components, dtype=np.float64).copy()
    for row in result:
        anchor = int(np.argmax(np.abs(row)))
        if row[anchor] < 0.0:
            row *= -1.0
    return result


@dataclass(frozen=True)
class FixedPCAStateMap:
    """Fixed affine state definition used before EDMD-DL.

    ``components`` has shape ``(q, 10240)`` and contains orthonormal rows.
    This object has no trainable parameters and is never optimized against a
    reconstruction or forecasting target.
    """

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    total_variance: float
    training_weight_sum: float

    @property
    def full_dimension(self) -> int:
        return int(self.components.shape[1])

    @property
    def state_dimension(self) -> int:
        return int(self.components.shape[0])

    def validate(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        components = np.asarray(self.components, dtype=np.float64)
        explained = np.asarray(
            self.explained_variance, dtype=np.float64
        )
        if mean.shape != (FULL_STATE_DIMENSION,):
            raise ValueError(
                "PCA state mean must have shape "
                f"({FULL_STATE_DIMENSION},)"
            )
        if (
            components.ndim != 2
            or components.shape[1] != FULL_STATE_DIMENSION
            or components.shape[0] < 1
        ):
            raise ValueError(
                "PCA components must have shape "
                f"(q,{FULL_STATE_DIMENSION})"
            )
        if explained.shape != (components.shape[0],):
            raise ValueError(
                "explained_variance must have one value per PCA state "
                "coordinate"
            )
        if not (
            np.all(np.isfinite(mean))
            and np.all(np.isfinite(components))
            and np.all(np.isfinite(explained))
            and np.isfinite(float(self.total_variance))
            and np.isfinite(float(self.training_weight_sum))
        ):
            raise ValueError("PCA state-map values must be finite")
        if (
            np.any(explained < 0.0)
            or float(self.total_variance) < 0.0
            or float(self.training_weight_sum) <= 0.0
        ):
            raise ValueError(
                "PCA variances must be nonnegative and training weight "
                "sum must be positive"
            )
        identity = components @ components.T
        if not np.allclose(
            identity,
            np.eye(components.shape[0]),
            rtol=1.0e-8,
            atol=1.0e-10,
        ):
            raise ValueError("PCA state components must have orthonormal rows")

    @classmethod
    def fit(
        cls,
        states: Any,
        *,
        state_dimension: int,
        sample_weights: Any,
    ) -> "FixedPCAStateMap":
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("states must have shape (samples, coordinates)")
        if values.shape[1] != FULL_STATE_DIMENSION:
            raise ValueError(
                f"states must have {FULL_STATE_DIMENSION} coordinates, "
                f"got {values.shape[1]}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("states must contain only finite values")
        weights = _normalized_weights(sample_weights, len(values))
        dimension = int(state_dimension)
        maximum_rank = min(len(values) - 1, values.shape[1])
        if not 1 <= dimension <= maximum_rank:
            raise ValueError(
                "state_dimension must lie in "
                f"[1,{maximum_rank}] for this training set"
            )

        mean = np.sum(values * weights[:, None], axis=0)
        centered = values - mean
        weighted = centered * np.sqrt(weights[:, None])
        _, singular_values, right_vectors = np.linalg.svd(
            weighted, full_matrices=False
        )
        components = _canonicalize_component_signs(
            right_vectors[:dimension]
        )
        explained_variance = singular_values[:dimension] ** 2
        result = cls(
            mean=np.asarray(mean, dtype=np.float64),
            components=components,
            explained_variance=np.asarray(
                explained_variance, dtype=np.float64
            ),
            total_variance=float(np.sum(singular_values**2)),
            training_weight_sum=float(np.asarray(sample_weights).sum()),
        )
        result.validate()
        return result

    def transform(self, histories: Any) -> np.ndarray:
        self.validate()
        values = np.asarray(histories, dtype=np.float64)
        original_shape = values.shape
        if values.ndim == 3:
            if values.shape[1:] != (N_ELECTRODES, HISTORY_LENGTH):
                raise ValueError(
                    "histories must have shape "
                    f"(N,{N_ELECTRODES},{HISTORY_LENGTH})"
                )
            values = values.reshape(len(values), -1)
        elif values.ndim != 2 or values.shape[1] != self.full_dimension:
            raise ValueError(
                "histories must be flattened states or EEG delay states"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("histories must contain only finite values")
        result = (values - self.mean) @ self.components.T
        if original_shape[0] != result.shape[0]:
            raise RuntimeError("state transform changed the sample count")
        return result

    def inverse_transform(self, states: Any) -> np.ndarray:
        self.validate()
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.state_dimension:
            raise ValueError(
                "reduced states must have shape "
                f"(N,{self.state_dimension})"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("reduced states must contain only finite values")
        flat = values @ self.components + self.mean
        return flat.reshape(-1, N_ELECTRODES, HISTORY_LENGTH)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "fixed_weighted_pca_delay_state",
            "mean": np.asarray(self.mean, dtype=np.float64),
            "components": np.asarray(self.components, dtype=np.float64),
            "explained_variance": np.asarray(
                self.explained_variance, dtype=np.float64
            ),
            "total_variance": float(self.total_variance),
            "training_weight_sum": float(self.training_weight_sum),
        }


class PaperEDMDDLDictionary(nn.Module):
    """Paper-style dictionary with fixed 1 and fixed state coordinates."""

    def __init__(
        self,
        *,
        state_dimension: int,
        dictionary_size: int,
        hidden_dims: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.state_dimension = int(state_dimension)
        self.dictionary_size = int(dictionary_size)
        self.n_learned_observables = (
            self.dictionary_size - 1 - self.state_dimension
        )
        if self.state_dimension < 1:
            raise ValueError("state_dimension must be positive")
        if self.n_learned_observables < 1:
            raise ValueError(
                "dictionary_size must leave room for the constant, all "
                "state coordinates, and at least one learned function"
            )
        if any(int(width) < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain only positive widths")

        widths = (
            self.state_dimension,
            *(int(width) for width in hidden_dims),
            self.n_learned_observables,
        )
        layers: list[nn.Module] = []
        for index, (input_width, output_width) in enumerate(
            zip(widths[:-1], widths[1:])
        ):
            layers.append(nn.Linear(input_width, output_width))
            if index < len(widths) - 2:
                layers.append(nn.Tanh())
        self.learned_dictionary = nn.Sequential(*layers)

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dimension:
            raise ValueError(
                "state must have shape "
                f"(batch,{self.state_dimension})"
            )
        if not torch.is_floating_point(state):
            raise TypeError("state must be a floating-point tensor")
        learned = self.learned_dictionary(state)
        constant = state.new_ones(len(state), 1)
        return torch.cat((constant, state, learned), dim=1)


@dataclass
class EDMDDLTrainingConfig:
    """Only parameters present in the autonomous EDMD-DL training path."""

    # Fair default shared with BSO: 1 constant + 8 fixed PCA-state
    # coordinates + 11 learned functions.
    dictionary_size: int = 20
    # ``None`` divides the nonconstant dictionary evenly between fixed state
    # coordinates and learned functions.
    reduced_state_dimension: int | None = 8
    hidden_dims: tuple[int, ...] = (16, 16)
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    transition_ridge: float = 1.0e-6
    seed: int = 2026
    print_every: int = 25


@dataclass
class EDMDDLFit:
    """Fitted autonomous EDMD-DL model with no control or decoder."""

    dictionary: PaperEDMDDLDictionary
    state_map: FixedPCAStateMap
    transition: np.ndarray
    training_history: list[dict[str, float]]
    diagnostics: dict[str, Any]
    config: dict[str, Any]
    device: str = "cpu"

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "eeg-edmd-dl-li-2017-v1",
            "method": EDMD_DL_NAME,
            "dictionary_state": {
                key: value.detach().cpu()
                for key, value in self.dictionary.state_dict().items()
            },
            "state_map": self.state_map.to_payload(),
            "transition": np.asarray(self.transition, dtype=np.float64),
            "training_history": copy.deepcopy(self.training_history),
            "diagnostics": copy.deepcopy(self.diagnostics),
            "config": copy.deepcopy(self.config),
        }
        torch.save(payload, output)
        return output


def _paper_transition(
    current: np.ndarray,
    target: np.ndarray,
    *,
    sample_weights: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve Eq. (10) of Li et al. in row-feature convention."""

    if current.shape != target.shape or current.ndim != 2:
        raise ValueError("current and target dictionaries must match")
    if not (
        np.array_equal(current[:, 0], np.ones(len(current)))
        and np.array_equal(target[:, 0], np.ones(len(target)))
    ):
        raise ValueError("the leading dictionary function must be exact one")
    regularization = float(ridge)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("transition_ridge must be finite and nonnegative")

    weights = _normalized_weights(sample_weights, len(current))
    gram = current.T @ (current * weights[:, None])
    # Keep the full-matrix ridge solve from Eq. (10) / Algorithm 1 of
    # Li et al.  The constant is a fixed dictionary function, but the paper
    # does not additionally constrain the first column of the ridge solution.
    cross = current.T @ (target * weights[:, None])
    system = gram + regularization * np.eye(
        current.shape[1], dtype=np.float64
    )
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
    ridge_term = regularization * float(np.sum(transition**2))
    return np.asarray(transition, dtype=np.float64), {
        "convention": "Psi(z_next) ~= Psi(z) @ K",
        "autonomous": True,
        "constant_dictionary_function_is_fixed": True,
        "constant_transition_column_is_constrained": False,
        "ridge": regularization,
        "linear_solver": solver,
        "weighted_invariance_squared_sum": residual_squared_sum,
        "weighted_invariance_mse": (
            residual_squared_sum / current.shape[1]
        ),
        "ridge_term_full_matrix": ridge_term,
        "objective_squared_sum_convention": (
            residual_squared_sum + ridge_term
        ),
        "gram_condition": float(np.linalg.cond(gram)),
        "gram_rank": int(np.linalg.matrix_rank(gram)),
        "n_training_pairs": int(len(current)),
        "dictionary_size": int(current.shape[1]),
        "transition_spectral_radius": float(
            np.max(np.abs(np.linalg.eigvals(transition)))
        ),
        "regularized_system_condition": float(np.linalg.cond(system)),
        "constant_output_max_abs_error": float(
            np.max(np.abs(current @ transition[:, 0] - 1.0))
        ),
        "spectral_clipping": False,
    }


def _dictionary_numpy(
    model: PaperEDMDDLDictionary,
    states: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        result = (
            model(
                torch.as_tensor(
                    states, dtype=torch.float32, device=device
                )
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    model.train(was_training)
    # Recreate this fixed feature in float64 to make its exactness explicit.
    result[:, 0] = 1.0
    return result


def _validate_training_batch(
    batch: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories = np.asarray(batch.history, dtype=np.float64)
    next_histories = np.asarray(batch.next_history, dtype=np.float64)
    if histories.ndim != 3 or histories.shape[1:] != (
        N_ELECTRODES,
        HISTORY_LENGTH,
    ):
        raise ValueError(
            "history must have shape "
            f"(N,{N_ELECTRODES},{HISTORY_LENGTH})"
        )
    if next_histories.shape != histories.shape:
        raise ValueError("next_history must have the same shape as history")
    if len(histories) < 3:
        raise ValueError("EDMD-DL requires at least three transition pairs")
    if not (
        np.all(np.isfinite(histories))
        and np.all(np.isfinite(next_histories))
    ):
        raise ValueError("history arrays must contain only finite values")
    weights = _normalized_weights(batch.sample_weights, len(histories))
    return histories, next_histories, weights


def train_paper_edmd_dl(
    batch: Any,
    config: EDMDDLTrainingConfig | None = None,
    *,
    state_map: FixedPCAStateMap | None = None,
    device: torch.device | str | None = None,
) -> EDMDDLFit:
    """Fit autonomous EDMD-DL on a declared PCA-reduced delay state."""

    cfg = copy.deepcopy(config or EDMDDLTrainingConfig())
    dictionary_size = int(cfg.dictionary_size)
    if dictionary_size < 3:
        raise ValueError("dictionary_size must be at least three")
    state_dimension = (
        max(1, (dictionary_size - 1) // 2)
        if cfg.reduced_state_dimension is None
        else int(cfg.reduced_state_dimension)
    )
    if not 1 <= state_dimension <= dictionary_size - 2:
        raise ValueError(
            "reduced_state_dimension must leave at least one learned "
            "dictionary function"
        )
    if int(cfg.epochs) < 1 or int(cfg.batch_size) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if (
        not np.isfinite(float(cfg.learning_rate))
        or float(cfg.learning_rate) <= 0.0
    ):
        raise ValueError("learning_rate must be finite and positive")

    _set_seed(int(cfg.seed))
    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    histories, next_histories, weights = _validate_training_batch(batch)
    flat_histories = histories.reshape(len(histories), -1)
    if state_map is None:
        fitted_state_map = FixedPCAStateMap.fit(
            flat_histories,
            state_dimension=state_dimension,
            sample_weights=weights,
        )
        state_map_source = "fit_from_current_training_histories"
    else:
        if not isinstance(state_map, FixedPCAStateMap):
            raise TypeError("state_map must be a FixedPCAStateMap")
        state_map.validate()
        if state_map.full_dimension != FULL_STATE_DIMENSION:
            raise ValueError(
                "shared state_map has the wrong full delay-state dimension"
            )
        if state_map.state_dimension != state_dimension:
            raise ValueError(
                "shared state_map dimension does not match "
                "reduced_state_dimension"
            )
        fitted_state_map = state_map
        state_map_source = "shared_training_state_map"
    state_map = fitted_state_map
    current_state = state_map.transform(histories)
    next_state = state_map.transform(next_histories)

    dictionary = PaperEDMDDLDictionary(
        state_dimension=state_dimension,
        dictionary_size=dictionary_size,
        hidden_dims=tuple(cfg.hidden_dims),
    ).to(target_device)
    optimizer = torch.optim.Adam(
        dictionary.parameters(), lr=float(cfg.learning_rate)
    )
    rng = np.random.default_rng(int(cfg.seed))
    current_tensor = torch.as_tensor(current_state, dtype=torch.float32)
    target_tensor = torch.as_tensor(next_state, dtype=torch.float32)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
    history: list[dict[str, float]] = []
    transition_diagnostics: dict[str, Any] = {}

    for epoch in range(int(cfg.epochs)):
        phi_current = _dictionary_numpy(
            dictionary, current_state, target_device
        )
        phi_target = _dictionary_numpy(
            dictionary, next_state, target_device
        )
        transition, transition_diagnostics = _paper_transition(
            phi_current,
            phi_target,
            sample_weights=weights,
            ridge=float(cfg.transition_ridge),
        )
        transition_tensor = torch.as_tensor(
            transition, dtype=torch.float32, device=target_device
        )

        dictionary.train()
        permutation = rng.permutation(len(histories))
        residual_sum = 0.0
        residual_weight = 0.0
        for start in range(0, len(permutation), int(cfg.batch_size)):
            indices = permutation[start : start + int(cfg.batch_size)]
            index = torch.as_tensor(indices, dtype=torch.long)
            state = current_tensor[index].to(target_device)
            next_value = target_tensor[index].to(target_device)
            batch_weights = weight_tensor[index].to(target_device)
            optimizer.zero_grad(set_to_none=True)
            current_dictionary = dictionary(state)
            next_dictionary = dictionary(next_value)
            residual_per_sample = torch.mean(
                (
                    current_dictionary @ transition_tensor
                    - next_dictionary
                ).square(),
                dim=1,
            )
            weight_sum = batch_weights.sum()
            loss = torch.sum(
                batch_weights * residual_per_sample
            ) / weight_sum
            loss.backward()
            optimizer.step()
            residual_sum += float(
                torch.sum(
                    batch_weights * residual_per_sample.detach()
                ).cpu()
            )
            residual_weight += float(weight_sum.detach().cpu())

        invariance_mse = residual_sum / residual_weight
        # Dividing the complete ridge objective by the fixed dictionary
        # dimension leaves the Eq. (10) minimizer unchanged and matches the
        # feature-mean invariance loss reported here.
        ridge_term = (
            float(cfg.transition_ridge)
            * float(np.sum(transition**2))
            / float(dictionary_size)
        )
        record = {
            "epoch": float(epoch + 1),
            "weighted_invariance_mse": float(invariance_mse),
            "koopman_ridge_term": float(ridge_term),
            "objective": float(invariance_mse + ridge_term),
        }
        history.append(record)
        if cfg.print_every and (
            epoch == 0
            or (epoch + 1) % int(cfg.print_every) == 0
            or epoch + 1 == int(cfg.epochs)
        ):
            print(
                f"[{EDMD_DL_NAME}] epoch {epoch + 1:4d}/{cfg.epochs}: "
                f"invariance={invariance_mse:.6e}"
            )

    dictionary = dictionary.eval()
    final_current = _dictionary_numpy(
        dictionary, current_state, target_device
    )
    final_target = _dictionary_numpy(
        dictionary, next_state, target_device
    )
    transition, transition_diagnostics = _paper_transition(
        final_current,
        final_target,
        sample_weights=weights,
        ridge=float(cfg.transition_ridge),
    )
    trainable_parameters = int(
        sum(parameter.numel() for parameter in dictionary.parameters())
    )
    explained_total = float(np.sum(state_map.explained_variance))
    explained_fraction = (
        explained_total / float(state_map.total_variance)
        if float(state_map.total_variance) > 0.0
        else 0.0
    )
    diagnostics: dict[str, Any] = {
        "method": EDMD_DL_NAME,
        "long_name": EDMD_DL_LONG_NAME,
        "method_variant": EDMD_DL_VARIANT,
        "reference": EDMD_DL_REFERENCE,
        "paper_objective": (
            "min_{K,theta} ||Psi(z_next;theta)-"
            "Psi(z;theta)K||_W^2 + lambda||K||_F^2"
        ),
        "autonomous": True,
        "uses_control_input": False,
        "has_control_matrix": False,
        "uses_trainable_decoder": False,
        "uses_reconstruction_loss": False,
        "uses_forecast_loss": False,
        "uses_multistep_loss": False,
        "uses_anti_collapse": False,
        "uses_whitening": False,
        "uses_spectral_clipping": False,
        "full_delay_state_dimension": FULL_STATE_DIMENSION,
        "state_definition": (
            "fixed weighted PCA of flattened 64x160 current training "
            "histories"
        ),
        "state_map_source": state_map_source,
        "reduced_state_dimension": state_dimension,
        "scope_warning": (
            "This is Li et al. EDMD-DL on a PCA-reduced delay state, "
            "not full 10240-dimensional EDMD-DL."
        ),
        "full_state_dictionary_lower_bound": FULL_STATE_DIMENSION + 1,
        "dictionary_size": dictionary_size,
        "fixed_constant_functions": 1,
        "fixed_state_coordinate_functions": state_dimension,
        "learned_dictionary_functions": (
            dictionary_size - 1 - state_dimension
        ),
        "dictionary_trainable_parameters": trainable_parameters,
        "state_inverse": (
            "fixed inverse PCA map; no fitted or trainable decoder"
        ),
        "pca_retained_training_variance": explained_total,
        "pca_retained_training_variance_fraction": explained_fraction,
        "transition": transition_diagnostics,
    }
    return EDMDDLFit(
        dictionary=dictionary,
        state_map=state_map,
        transition=transition,
        training_history=history,
        diagnostics=diagnostics,
        config=asdict(cfg),
        device=str(target_device),
    )


def rollout_paper_edmd_dl(
    fit: EDMDDLFit,
    initial_history: Any,
    *,
    n_steps: int,
    lag_samples: int,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Roll out the autonomous finite-dimensional Koopman matrix."""

    histories = np.asarray(initial_history, dtype=np.float64)
    if histories.ndim != 3 or histories.shape[1:] != (
        N_ELECTRODES,
        HISTORY_LENGTH,
    ):
        raise ValueError(
            "initial_history must have shape "
            f"(N,{N_ELECTRODES},{HISTORY_LENGTH})"
        )
    try:
        steps = int(n_steps)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("n_steps must be a nonnegative integer") from error
    if (
        isinstance(n_steps, (float, np.floating))
        and float(n_steps) != steps
    ) or steps < 0:
        raise ValueError("n_steps must be a nonnegative integer")
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
    if steps == 0:
        return np.empty(
            (len(histories), 0, N_ELECTRODES, lag),
            dtype=np.float64,
        )

    target_device = (
        torch.device(device)
        if device is not None
        else torch.device(fit.device)
    )
    state = fit.state_map.transform(histories)
    features = _dictionary_numpy(
        fit.dictionary.to(target_device), state, target_device
    )
    outputs: list[np.ndarray] = []
    state_start = 1
    state_stop = 1 + fit.state_map.state_dimension
    for _ in range(steps):
        features = features @ np.asarray(
            fit.transition, dtype=np.float64
        )
        predicted_state = features[:, state_start:state_stop]
        predicted_history = fit.state_map.inverse_transform(
            predicted_state
        )
        outputs.append(
            predicted_history[:, :, -lag:].copy()
        )
    return np.stack(outputs, axis=1)


__all__ = [
    "EDMD_DL_LONG_NAME",
    "EDMD_DL_NAME",
    "EDMD_DL_REFERENCE",
    "EDMD_DL_VARIANT",
    "EDMDDLFit",
    "EDMDDLTrainingConfig",
    "FULL_STATE_DIMENSION",
    "FixedPCAStateMap",
    "PaperEDMDDLDictionary",
    "rollout_paper_edmd_dl",
    "train_paper_edmd_dl",
]
