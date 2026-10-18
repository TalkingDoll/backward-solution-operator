"""Small autonomous reference baselines for the EEG forecasting experiment.

Both baselines operate on the *same externally fitted*
:class:`eeg_edmddl.FixedPCAStateMap`.  This makes the reduced state definition
explicit and prevents a baseline from learning a separate autoencoder or
output decoder.

The models are intentionally standard and autonomous:

* ``PCA-VAR`` fits one affine VAR(1) step on the fixed PCA delay state;
* ``Kernel EDMD`` uses one Gaussian kernel section at every training snapshot,
  augmented by ``[1, z]`` so EEG coordinates are read out without a decoder.

There are no labels, cue phase, controls, trainable decoders, reconstruction
losses, spectral-radius constraints, or spectral clipping in this module.
Full EEG histories are obtained only through the fixed inverse PCA map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .eeg_resting_data import HISTORY_SAMPLES
from .eeg_edmddl import FixedPCAStateMap


N_ELECTRODES = 64


def _history_batch(
    values: Any,
    *,
    name: str,
) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    single = array.ndim == 2
    if single:
        array = array[None]
    expected = (N_ELECTRODES, HISTORY_SAMPLES)
    if array.ndim != 3 or array.shape[1:] != expected:
        raise ValueError(
            f"{name} must have shape {expected} or (N,{expected[0]},"
            f"{expected[1]})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array, single


def _training_pairs(
    histories: Any,
    targets: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate current and next complete EEG delay states."""

    current, single = _history_batch(histories, name="histories")
    if single:
        raise ValueError("fitting requires a batch of training histories")
    target = np.asarray(targets, dtype=np.float64)
    if target.shape != current.shape:
        raise ValueError(
            "targets must be complete next histories with shape "
            f"{current.shape}"
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("targets must contain only finite values")
    return current, target


def _normalized_weights(
    sample_weights: Any | None,
    n_samples: int,
) -> np.ndarray:
    if sample_weights is None:
        return np.full(n_samples, 1.0 / n_samples, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if weights.shape != (n_samples,):
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


def _validate_state_map(
    state_map: FixedPCAStateMap,
) -> FixedPCAStateMap:
    if not isinstance(state_map, FixedPCAStateMap):
        raise TypeError("state_map must be a FixedPCAStateMap")
    if state_map.full_dimension != N_ELECTRODES * HISTORY_SAMPLES:
        raise ValueError("state_map does not describe the EEG delay state")
    if not (
        np.all(np.isfinite(state_map.mean))
        and np.all(np.isfinite(state_map.components))
    ):
        raise ValueError("state_map must contain only finite values")
    return state_map


def _fit_transition(
    current_features: np.ndarray,
    next_features: np.ndarray,
    *,
    sample_weights: np.ndarray,
    ridge: float,
    pin_constant: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit an autonomous row-convention EDMD matrix.

    Kernel EDMD uses the ordinary full-matrix weighted ridge solution.  The
    affine PCA-VAR reference may pin its structural intercept coordinate.
    No eigenvalue or spectral-radius post-processing is applied.
    """

    current = np.asarray(current_features, dtype=np.float64)
    target = np.asarray(next_features, dtype=np.float64)
    if current.ndim != 2 or current.shape != target.shape:
        raise ValueError("feature arrays must be matching two-dimensional")
    if not (
        np.array_equal(current[:, 0], np.ones(len(current)))
        and np.array_equal(target[:, 0], np.ones(len(target)))
    ):
        raise ValueError("feature zero must be the exact constant one")
    regularization = float(ridge)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("ridge must be finite and nonnegative")

    gram = current.T @ (current * sample_weights[:, None])
    target_columns = target[:, 1:] if pin_constant else target
    cross = current.T @ (
        target_columns * sample_weights[:, None]
    )
    system = gram + regularization * np.eye(
        current.shape[1], dtype=np.float64
    )
    try:
        solved = np.linalg.solve(system, cross)
        solver = "solve"
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(system) @ cross
        solver = "pinv"

    if pin_constant:
        transition = np.empty(
            (current.shape[1], current.shape[1]), dtype=np.float64
        )
        transition[:, 0] = 0.0
        transition[0, 0] = 1.0
        transition[:, 1:] = solved
    else:
        transition = np.asarray(solved, dtype=np.float64)
    residual = current @ transition - target
    weighted_mse = float(
        np.sum(sample_weights[:, None] * residual**2)
        / current.shape[1]
    )
    return transition, {
        "convention": "Psi(next) ~= Psi(current) @ K",
        "autonomous": True,
        "ridge": regularization,
        "linear_solver": solver,
        "weighted_invariance_mse": weighted_mse,
        "gram_condition": float(np.linalg.cond(gram)),
        "regularized_system_condition": float(np.linalg.cond(system)),
        "constant_dictionary_function_is_fixed": True,
        "constant_transition_column_is_constrained": bool(pin_constant),
        "spectral_clipping": False,
    }


def _tail_from_reduced_state(
    state_map: FixedPCAStateMap,
    reduced_state: np.ndarray,
    *,
    lag_samples: int,
) -> np.ndarray:
    lag = int(lag_samples)
    if lag < 1 or lag > HISTORY_SAMPLES:
        raise ValueError("lag_samples must lie in 1..HISTORY_SAMPLES")
    histories = state_map.inverse_transform(reduced_state)
    return histories[:, :, -lag:]


@dataclass(frozen=True)
class PCAVARModel:
    """Autonomous affine VAR(1) on a fixed PCA delay state."""

    state_map: FixedPCAStateMap
    K: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def state_dimension(self) -> int:
        return self.state_map.state_dimension

    def transform(self, histories: Any) -> np.ndarray:
        values, single = _history_batch(histories, name="histories")
        reduced = self.state_map.transform(values)
        features = np.column_stack(
            (np.ones(len(reduced), dtype=np.float64), reduced)
        )
        return features[0] if single else features

    def predict_tail(self, histories: Any) -> np.ndarray:
        values, single = _history_batch(histories, name="histories")
        features = np.asarray(self.transform(values), dtype=np.float64)
        next_features = features @ self.K
        tails = _tail_from_reduced_state(
            self.state_map, next_features[:, 1:]
        )
        return tails[0] if single else tails

    def rollout(
        self,
        histories: Any,
        n_steps: int,
        *,
        lag_samples: int,
    ) -> np.ndarray:
        """Return autonomous forecasts with a declared output lag."""

        values, _ = _history_batch(histories, name="histories")
        steps = int(n_steps)
        if steps < 0:
            raise ValueError("n_steps must be nonnegative")
        lag = int(lag_samples)
        if lag < 1 or lag > HISTORY_SAMPLES:
            raise ValueError("lag_samples must lie in 1..HISTORY_SAMPLES")
        features = np.asarray(self.transform(values), dtype=np.float64)
        result = np.empty(
            (len(values), steps, N_ELECTRODES, lag),
            dtype=np.float64,
        )
        for index in range(steps):
            features = features @ self.K
            result[:, index] = _tail_from_reduced_state(
                self.state_map,
                features[:, 1:],
                lag_samples=lag,
            )
        return result


@dataclass(frozen=True)
class KernelEDMDModel:
    """Autonomous EDMD with fixed coordinates and landmark RBF features."""

    state_map: FixedPCAStateMap
    landmarks: np.ndarray
    bandwidth: float
    K: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def state_dimension(self) -> int:
        return self.state_map.state_dimension

    @property
    def n_rbf_features(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def dictionary_size(self) -> int:
        return int(self.K.shape[0])

    def _features_from_state(self, reduced: np.ndarray) -> np.ndarray:
        squared_distance = _pairwise_squared_distance(
            reduced, self.landmarks
        )
        rbf = np.exp(
            -squared_distance / (2.0 * self.bandwidth**2)
        )
        return np.column_stack(
            (np.ones(len(reduced), dtype=np.float64), reduced, rbf)
        )

    def transform(self, histories: Any) -> np.ndarray:
        values, single = _history_batch(histories, name="histories")
        features = self._features_from_state(
            self.state_map.transform(values)
        )
        return features[0] if single else features

    def predict_tail(self, histories: Any) -> np.ndarray:
        values, single = _history_batch(histories, name="histories")
        features = np.asarray(self.transform(values), dtype=np.float64)
        next_features = features @ self.K
        start = 1
        stop = start + self.state_dimension
        tails = _tail_from_reduced_state(
            self.state_map, next_features[:, start:stop]
        )
        return tails[0] if single else tails

    def rollout(
        self,
        histories: Any,
        n_steps: int,
        *,
        lag_samples: int,
    ) -> np.ndarray:
        """Return autonomous forecasts with a declared output lag."""

        values, _ = _history_batch(histories, name="histories")
        steps = int(n_steps)
        if steps < 0:
            raise ValueError("n_steps must be nonnegative")
        lag = int(lag_samples)
        if lag < 1 or lag > HISTORY_SAMPLES:
            raise ValueError("lag_samples must lie in 1..HISTORY_SAMPLES")
        features = np.asarray(self.transform(values), dtype=np.float64)
        result = np.empty(
            (len(values), steps, N_ELECTRODES, lag),
            dtype=np.float64,
        )
        state_stop = 1 + self.state_dimension
        for index in range(steps):
            features = features @ self.K
            result[:, index] = _tail_from_reduced_state(
                self.state_map,
                features[:, 1:state_stop],
                lag_samples=lag,
            )
        return result


def _pairwise_squared_distance(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    squared = (
        np.sum(left_values**2, axis=1, keepdims=True)
        + np.sum(right_values**2, axis=1)[None]
        - 2.0 * left_values @ right_values.T
    )
    return np.maximum(squared, 0.0)


def _median_bandwidth(
    states: np.ndarray,
    landmarks: np.ndarray,
) -> float:
    distances = _pairwise_squared_distance(states, landmarks)
    positive = distances[distances > np.finfo(np.float64).eps]
    if positive.size:
        bandwidth = float(np.sqrt(np.median(positive)))
    else:
        bandwidth = 1.0
    return max(bandwidth, np.sqrt(np.finfo(np.float64).eps))


def fit_pca_var(
    histories: Any,
    targets: Any,
    *,
    state_map: FixedPCAStateMap,
    ridge: float = 1.0e-6,
    sample_weights: Any | None = None,
) -> PCAVARModel:
    """Fit autonomous affine VAR(1) on the passed fixed PCA state."""

    current, next_histories = _training_pairs(histories, targets)
    fixed_map = _validate_state_map(state_map)
    weights = _normalized_weights(sample_weights, len(current))
    current_state = fixed_map.transform(current)
    next_state = fixed_map.transform(next_histories)
    current_features = np.column_stack(
        (np.ones(len(current_state)), current_state)
    )
    next_features = np.column_stack(
        (np.ones(len(next_state)), next_state)
    )
    transition, transition_diagnostics = _fit_transition(
        current_features,
        next_features,
        sample_weights=weights,
        ridge=ridge,
        pin_constant=True,
    )
    diagnostics = {
        "method": "PCA-VAR",
        "model": "autonomous affine VAR(1) on fixed PCA delay state",
        "n_training_pairs": int(len(current)),
        "reduced_state_dimension": int(fixed_map.state_dimension),
        "dictionary_size": int(1 + fixed_map.state_dimension),
        "fixed_constant_functions": 1,
        "fixed_state_coordinate_functions": int(
            fixed_map.state_dimension
        ),
        "learned_dictionary_functions": 0,
        "uses_control_input": False,
        "has_control_matrix": False,
        "uses_trainable_decoder": False,
        "state_reconstruction": "fixed PCA inverse only",
        "uses_spectral_clipping": False,
        "transition": transition_diagnostics,
    }
    return PCAVARModel(
        state_map=fixed_map,
        K=transition,
        diagnostics=diagnostics,
    )


def fit_kernel_edmd(
    histories: Any,
    targets: Any,
    *,
    state_map: FixedPCAStateMap,
    ridge: float = 1.0e-6,
    bandwidth: float | None = None,
    sample_weights: Any | None = None,
) -> KernelEDMDModel:
    """Fit sample-scaled Gaussian-kernel EDMD on the fixed PCA state.

    With ``N`` training pairs and a ``q``-coordinate state, the explicit
    dictionary is ``[1, z, k(z,x_1), ..., k(z,x_N)]`` and the Koopman matrix
    has dimension ``1 + q + N``.  Including ``z`` permits direct state
    recovery without fitting a decoder or preimage map.
    """

    current, next_histories = _training_pairs(histories, targets)
    fixed_map = _validate_state_map(state_map)
    weights = _normalized_weights(sample_weights, len(current))
    current_state = fixed_map.transform(current)
    next_state = fixed_map.transform(next_histories)
    # Standard sample-scaled kernel dictionary: every current snapshot is a
    # kernel centre.  There is no random landmark truncation or Nyström map.
    landmarks = np.asarray(current_state, dtype=np.float64).copy()
    total_features = 1 + fixed_map.state_dimension + len(landmarks)
    if bandwidth is None:
        fitted_bandwidth = _median_bandwidth(
            current_state, landmarks
        )
        bandwidth_source = "training states median distance"
    else:
        fitted_bandwidth = float(bandwidth)
        if not np.isfinite(fitted_bandwidth) or fitted_bandwidth <= 0.0:
            raise ValueError("bandwidth must be finite and positive")
        bandwidth_source = "caller"

    provisional = KernelEDMDModel(
        state_map=fixed_map,
        landmarks=landmarks,
        bandwidth=fitted_bandwidth,
        K=np.empty((total_features, total_features)),
        diagnostics={},
    )
    current_features = provisional._features_from_state(current_state)
    next_features = provisional._features_from_state(next_state)
    transition, transition_diagnostics = _fit_transition(
        current_features,
        next_features,
        sample_weights=weights,
        ridge=ridge,
        pin_constant=False,
    )
    diagnostics = {
        "method": "Kernel EDMD",
        "model": (
            "sample-scaled autonomous Gaussian-kernel EDMD with explicit "
            "fixed PCA coordinates"
        ),
        "feature_definition": (
            f"1 + {fixed_map.state_dimension} fixed PCA coordinates + "
            f"{len(landmarks)} Gaussian kernel sections"
        ),
        "n_training_pairs": int(len(current)),
        "reduced_state_dimension": int(fixed_map.state_dimension),
        "dictionary_size": total_features,
        "fixed_constant_functions": 1,
        "fixed_state_coordinate_functions": int(
            fixed_map.state_dimension
        ),
        "kernel_dictionary_functions": int(len(landmarks)),
        "koopman_matrix_dimension": int(total_features),
        "kernel_centres": "all current training states",
        "kernel_fit_source": "training current states only",
        "bandwidth": fitted_bandwidth,
        "bandwidth_source": bandwidth_source,
        "uses_nystrom_whitening": False,
        "uses_control_input": False,
        "has_control_matrix": False,
        "uses_trainable_decoder": False,
        "state_reconstruction": "fixed PCA inverse only",
        "uses_spectral_clipping": False,
        "transition": transition_diagnostics,
    }
    return KernelEDMDModel(
        state_map=fixed_map,
        landmarks=landmarks,
        bandwidth=fitted_bandwidth,
        K=transition,
        diagnostics=diagnostics,
    )


__all__ = [
    "HISTORY_SAMPLES",
    "KernelEDMDModel",
    "PCAVARModel",
    "fit_kernel_edmd",
    "fit_pca_var",
]
