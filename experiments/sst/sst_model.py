"""Core utilities for a structured, one-step-Markov SST BSO.

This module deliberately does not train or alter the raw FNO.  The raw FNO
continues to consume the original multichannel, training-window z-score state
from :mod:`kbe_observation_data`.  Only its one-day output is transformed into
the structured anomaly coordinates used by the BSO:

``raw normalized state -> calendar anomaly -> EOFs -> joint PCA -> q``.

The SST calendar term is the exact empirical month/day climatology used by the
forecast scoring baseline.  Auxiliary-channel calendar terms are K-harmonic
least-squares fits.  Every fitted quantity uses indices strictly before
``meta["train_end"]``.  The reduced dynamics remain first-order Markov:

``q[t + 1] = A @ q[t] + b``.

No delay coordinates, past-state concatenation, or memory kernels are
implemented here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
from scipy.linalg import schur

from edmd_bso_method import (
    _pace_cuda,
    _pacing_start,
    apply_operator,
    device,
)
from kbe_observation_data import calendar_day_index, calendar_phase


TRANSFORM_VERSION = "sst-empirical-aux-harmonic-structured-anomaly"
MODEL_VERSION = "sst-bso-one-step-markov"
MODEL_FINGERPRINT_VERSION = "sst-bso-full-state-sha256"
ResidualMode = Literal["none", "scalar_ar1"]
OutputUnits = Literal["structured", "raw_normalized", "kelvin"]


def _channel_spatial_weights(
    dataset: Mapping[str, Any], channel: int
) -> np.ndarray:
    meta = dataset["meta"]
    n_channels = int(meta["n_channels"])
    height, width = np.asarray(meta["weights_2d"]).shape
    weights = np.asarray(dataset["weights"], dtype=np.float64).reshape(
        n_channels, height, width
    )[int(channel)]
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"channel {channel} has zero spatial weight")
    return (weights / total).reshape(-1)


def _leading_weighted_eofs(
    weighted: Any, requested: int, max_duty_cycle: float = 0.8
) -> np.ndarray:
    """Return the leading spatial right-singular vectors."""
    matrix = np.asarray(weighted, dtype=np.float32)
    rank = min(int(requested), matrix.shape[0], matrix.shape[1])
    if torch.cuda.is_available():
        pacing_started = _pacing_start(max_duty_cycle)
        tensor = torch.as_tensor(
            matrix, dtype=torch.float32, device=device
        )
        covariance = tensor.T @ tensor
        covariance = 0.5 * (covariance + covariance.T)
        _, vectors = torch.linalg.eigh(covariance)
        basis = vectors[:, -rank:].flip(1).T
        pivot = torch.argmax(torch.abs(basis), dim=1)
        signs = torch.sign(
            basis[torch.arange(rank, device=basis.device), pivot]
        )
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        basis = basis * signs[:, None]
        result = basis.detach().cpu().numpy().astype(np.float64)
        del tensor, covariance, vectors, basis
        _pace_cuda(pacing_started, max_duty_cycle)
        return result
    _, _, right = np.linalg.svd(
        np.asarray(weighted, dtype=np.float64), full_matrices=False
    )
    return right[:rank]


def encode_state(
    dataset: Mapping[str, Any],
    dictionary: Mapping[str, Any],
    states: Any,
) -> np.ndarray:
    """Encode normalized states into the fitted per-channel EOF basis."""
    array = np.asarray(states, dtype=np.float64)
    if array.ndim == 3:
        array = array[None]
    blocks = []
    scales = np.asarray(
        dictionary["coefficient_scales"], dtype=np.float64
    )
    for channel, (field_mean, basis) in enumerate(
        zip(dictionary["field_means"], dictionary["eof_bases"])
    ):
        fields = array[:, channel].reshape(array.shape[0], -1)
        weights = _channel_spatial_weights(dataset, channel)
        blocks.append(
            scales[channel]
            * (
                (fields - np.asarray(field_mean, dtype=np.float64))
                @ (
                    np.asarray(basis, dtype=np.float64)
                    * weights[None, :]
                ).T
            )
        )
    return np.concatenate(blocks, axis=1)


def decode_sst(
    dictionary: Mapping[str, Any], coefficients: Any
) -> np.ndarray:
    """Decode the SST block from fitted EOF coefficients."""
    low, high = dictionary["coeff_slices"][0]
    basis = np.asarray(dictionary["eof_bases"][0], dtype=np.float64)
    field_mean = np.asarray(
        dictionary["field_means"][0], dtype=np.float64
    )
    scale = float(np.asarray(dictionary["coefficient_scales"])[0])
    selected = (
        np.asarray(coefficients, dtype=np.float64)[:, low:high] / scale
    )
    return selected @ basis + field_mean


def _operator_fingerprint(operator: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(operator.state_dict().items()):
        digest.update(str(name).encode("utf-8"))
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(
                np.asarray(array.shape, dtype=np.int64).tobytes()
            )
            digest.update(memoryview(array).cast("B"))
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()[:20]


def _require_mapping_keys(
        mapping: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"{name} is missing required keys: {missing}")


def _hash_array(digest: Any, name: str, values: Any) -> None:
    array = np.asarray(values)
    shape = array.shape
    array = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))


def _hash_model_value(digest: Any, name: str, value: Any) -> None:
    """Hash an arbitrary persisted model value with explicit type framing."""
    encoded_name = str(name).encode("utf-8")
    digest.update(b"N")
    digest.update(len(encoded_name).to_bytes(8, "little"))
    digest.update(encoded_name)
    if torch.is_tensor(value):
        digest.update(b"T")
        _hash_array(digest, "tensor", value.detach().cpu().numpy())
        return
    if isinstance(value, np.ndarray):
        digest.update(b"A")
        _hash_array(digest, "array", value)
        return
    if isinstance(value, np.generic):
        _hash_model_value(digest, "numpy_scalar", value.item())
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("BSO model mapping keys must all be strings")
        digest.update(b"M")
        keys = sorted(value)
        digest.update(len(keys).to_bytes(8, "little"))
        for key in keys:
            _hash_model_value(digest, "key", str(key))
            _hash_model_value(digest, str(key), value[key])
        return
    if isinstance(value, tuple):
        digest.update(b"U")
        digest.update(len(value).to_bytes(8, "little"))
        for index, item in enumerate(value):
            _hash_model_value(digest, str(index), item)
        return
    if isinstance(value, list):
        digest.update(b"L")
        digest.update(len(value).to_bytes(8, "little"))
        for index, item in enumerate(value):
            _hash_model_value(digest, str(index), item)
        return
    if value is None:
        digest.update(b"0")
        return
    if isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
        return
    if isinstance(value, int):
        payload = str(value).encode("ascii")
        digest.update(b"I")
    elif isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"model fingerprint cannot hash nonfinite {name}")
        payload = float(value).hex().encode("ascii")
        digest.update(b"F")
    elif isinstance(value, str):
        payload = value.encode("utf-8")
        digest.update(b"S")
    else:
        raise TypeError(
            f"unsupported persisted model value for fingerprint at {name}: "
            f"{type(value).__name__}"
        )
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def model_state_integrity_sha256(model: Mapping[str, Any]) -> str:
    """Return a full-state digest for every persisted scientific model field.

    The short ``fingerprint`` is convenient for reports, but cache
    integrity must cover the complete encoder, decoder, reduced coordinates,
    operators, residual closure, fit-window coefficients, and provenance.
    Self-referential digest fields are the only exclusions.
    """
    if not isinstance(model, Mapping):
        raise TypeError("BSO model must be a mapping")
    payload = {
        key: value
        for key, value in model.items()
        if key not in {
            "fingerprint",
            "fingerprint_schema",
            "full_state_integrity_sha256",
        }
    }
    digest = hashlib.sha256()
    digest.update(MODEL_FINGERPRINT_VERSION.encode("ascii"))
    _hash_model_value(digest, "model", payload)
    return digest.hexdigest()


def validate_model_state_integrity(model: Mapping[str, Any]) -> str:
    """Recompute and verify the persisted scientific-model state digest."""
    if model.get("fingerprint_schema") != MODEL_FINGERPRINT_VERSION:
        raise ValueError("BSO model fingerprint schema mismatch")
    expected = str(model.get("full_state_integrity_sha256", ""))
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ValueError("BSO model full-state integrity SHA-256 mismatch")
    if model.get("fingerprint") != expected[:20]:
        raise ValueError("BSO model full-state integrity SHA-256 mismatch")
    actual = model_state_integrity_sha256(model)
    if actual != expected:
        raise ValueError("BSO model full-state integrity SHA-256 mismatch")
    return expected


def validate_fit_window_model_state(
        model: Mapping[str, Any], expected_train_end: int) -> None:
    """Reject checkpoints that persist observation-derived held-out states."""
    train_end = int(expected_train_end)
    if train_end <= 1:
        raise ValueError("expected_train_end must exceed one")
    coefficients = np.asarray(model.get("coeffs"))
    if coefficients.ndim != 2 or coefficients.shape[0] != train_end:
        raise ValueError(
            "BSO coefficient state must contain exactly the fitting window"
        )
    if model.get("coefficient_state_scope") != "fit_window_only":
        raise ValueError("BSO coefficient state scope is not fit-window-only")
    if int(model.get("maximum_observation_index_encoded", -1)) != train_end - 1:
        raise ValueError(
            "BSO checkpoint persisted an observation index outside the fit window"
        )
    transform = model.get("transform")
    if not isinstance(transform, Mapping) or int(
        transform.get("train_end", -1)
    ) != train_end:
        raise ValueError("BSO transform fit boundary differs")
    coordinates = model.get("fit_coordinates")
    if not isinstance(coordinates, Mapping):
        raise ValueError("BSO model lacks fitting-window reduced coordinates")
    expected_rows = train_end - 1
    for name in ("q_source", "q_data", "q_fno"):
        values = np.asarray(coordinates.get(name))
        if values.ndim != 2 or values.shape[0] != expected_rows:
            raise ValueError(
                f"BSO reduced coordinate {name} is not fit-window-only"
            )


def _harmonic_design(phases: Any, harmonics: int) -> np.ndarray:
    """Return ``[1, sin(k theta), cos(k theta)]`` calendar features."""
    harmonics = int(harmonics)
    if harmonics <= 0:
        raise ValueError("auxiliary harmonics must be positive")
    theta = 2.0 * np.pi * np.asarray(phases, dtype=np.float64)
    columns = [np.ones_like(theta)]
    for order in range(1, harmonics + 1):
        columns.extend((
            np.sin(order * theta),
            np.cos(order * theta),
        ))
    return np.stack(columns, axis=1)


def _validate_raw_dataset(
        dataset: Mapping[str, Any],
        *,
        check_finite: bool = False,
) -> tuple[Mapping[str, Any], np.ndarray, int, int, int, int]:
    """Validate and return the immutable raw-state geometry.

    ``state_fields`` must be the original dimensionless channel z-scores used
    to train the FNO. Alternate transformed inputs are rejected because they
    would not match the FNO input domain.
    """
    _require_mapping_keys(dataset, ("kind", "weights", "meta"), "dataset")
    meta = dataset["meta"]
    if not isinstance(meta, Mapping):
        raise TypeError("dataset['meta'] must be a mapping")
    _require_mapping_keys(
        meta,
        (
            "state_fields",
            "times",
            "train_end",
            "n_channels",
            "channel_mean",
            "channel_std",
            "channel_valid_masks",
            "channel_weights",
            "weights_2d",
            "climatology",
            "training_fingerprint",
            "data_fingerprint",
        ),
        "dataset['meta']",
    )
    if dataset["kind"] != "grid2d_multichannel":
        raise ValueError(
            "SST BSO requires kind='grid2d_multichannel', got "
            f"{dataset['kind']!r}")
    if bool(meta.get("seasonal_anomaly", False)):
        raise ValueError(
            "SST BSO requires the raw FNO z-score dataset")
    if bool(meta.get("structured_anomaly", False)):
        raise ValueError("a structured anomaly view cannot be fitted twice")

    fields = np.asarray(meta["state_fields"])
    if fields.ndim != 4:
        raise ValueError(
            "meta['state_fields'] must have shape (time, channel, lat, lon)")
    n_time, n_channels, height, width = map(int, fields.shape)
    if n_channels != int(meta["n_channels"]):
        raise ValueError("state_fields channel count disagrees with metadata")
    if int(meta.get("sst_index", 0)) != 0:
        raise ValueError("SST BSO currently requires SST at channel index 0")
    train_end = int(meta["train_end"])
    if not 2 < train_end <= n_time:
        raise ValueError("train_end must leave at least two one-day pairs")
    if len(np.asarray(meta["times"])) != n_time:
        raise ValueError("time coordinate length disagrees with state_fields")
    if check_finite and not np.all(np.isfinite(fields[:train_end])):
        raise ValueError(
            "fitting-window raw normalized state_fields must be finite"
        )
    valid_masks = np.asarray(meta["channel_valid_masks"], dtype=bool)
    if valid_masks.shape != (n_channels, height, width):
        raise ValueError("channel_valid_masks has an incompatible shape")
    return meta, fields, train_end, n_channels, height, width


def _empirical_sst_climatology_from_training(
        fields: np.ndarray,
        meta: Mapping[str, Any],
        day_index: np.ndarray,
        train_end: int,
) -> np.ndarray:
    """Recompute the training-only SST climatology in kelvin.

    This is used as a leakage guard: the recomputed climatology must match the
    scoring climatology already attached to the raw dataset.
    """
    # Mirror kbe_observation_data._generate_sst exactly.  In particular, do
    # not cast the cached float32 normalized fields/statistics before the
    # long temporal means: doing so changes the day-0 fallback by O(1e-2 K)
    # even though day 1..366 and the source fingerprint are compatible.
    channel_mean = np.asarray(meta["channel_mean"])[0]
    channel_std = np.asarray(meta["channel_std"])[0]
    sst_kelvin = np.asarray(fields[:, 0]) * channel_std + channel_mean
    fallback = sst_kelvin[:train_end].mean(axis=0)
    result = np.tile(
        fallback[None], (367, 1, 1)).astype(np.float32)
    for day in range(1, 367):
        selected = np.flatnonzero(day_index[:train_end] == day)
        if selected.size:
            result[day] = sst_kelvin[selected].mean(axis=0)
    return result


def fit_structured_anomaly_transform(
        dataset: Mapping[str, Any],
        *,
        aux_harmonics: int = 6,
        verify_sst_climatology: bool = True,
        sst_climatology_atol_kelvin: float = 1e-6,
        scale_floor: float = 1e-6,
        batch_size: int = 256,
) -> dict[str, Any]:
    """Fit the structured calendar transform on the training window only.

    Parameters
    ----------
    dataset:
        Raw SST dataset.  ``meta["state_fields"]`` is in the dimensionless
        z-score units seen by the frozen raw FNO.
    aux_harmonics:
        Number of smooth annual harmonics for channels 1..C-1.  SST never uses
        this approximation; its term is the empirical month/day climatology
        used by the scoring baseline.
    verify_sst_climatology:
        Recompute SST month/day means from training data and require agreement
        with ``meta["climatology"]``.  Disabling this guard is not recommended.
    sst_climatology_atol_kelvin:
        Absolute tolerance for that verification.  Recalculation mirrors the
        source float32 arithmetic, so matching datasets agree byte-for-byte;
        the 1e-6 K default only allows harmless serialization roundoff.
    scale_floor:
        Lower bound for each training anomaly standard deviation, expressed in
        raw FNO z-score units.
    batch_size:
        Temporal batch size used while estimating anomaly variances.

    Returns
    -------
    dict
        A serializable transform.  ``climatology_raw_normalized`` and
        ``anomaly_scale_raw_normalized`` implement
        ``z=(x-C(day))/s``.  Its fingerprint hashes all fitted arrays and the
        source training fingerprint.
    """
    meta, fields, train_end, n_channels, height, width = (
        _validate_raw_dataset(dataset, check_finite=True))
    aux_harmonics = int(aux_harmonics)
    if aux_harmonics <= 0:
        raise ValueError("aux_harmonics must be positive")
    if not np.isfinite(scale_floor) or float(scale_floor) <= 0.0:
        raise ValueError("scale_floor must be finite and positive")
    sst_climatology_atol_kelvin = float(sst_climatology_atol_kelvin)
    if (
            not np.isfinite(sst_climatology_atol_kelvin)
            or sst_climatology_atol_kelvin < 0.0):
        raise ValueError(
            "sst_climatology_atol_kelvin must be finite and nonnegative")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    times = np.asarray(meta["times"])
    computed_day = np.asarray(calendar_day_index(times), dtype=np.int16)
    supplied_day = np.asarray(
        meta.get("calendar_day_index", computed_day), dtype=np.int16)
    if supplied_day.shape != computed_day.shape or not np.array_equal(
            supplied_day, computed_day):
        raise ValueError("calendar_day_index is inconsistent with meta['times']")
    if np.any((computed_day < 1) | (computed_day > 366)):
        raise ValueError("calendar day indices must lie in [1, 366]")

    channel_mean = np.asarray(meta["channel_mean"], dtype=np.float64)
    channel_std = np.asarray(meta["channel_std"], dtype=np.float64)
    if channel_mean.shape != (n_channels,) or channel_std.shape != (n_channels,):
        raise ValueError("channel_mean/channel_std must have one value per channel")
    if np.any(~np.isfinite(channel_std)) or np.any(channel_std <= 0.0):
        raise ValueError("raw channel standard deviations must be positive")

    supplied_sst_kelvin = np.asarray(meta["climatology"], dtype=np.float64)
    if supplied_sst_kelvin.shape != (367, height, width):
        raise ValueError(
            "the SST scoring climatology must have shape (367, lat, lon)")
    recomputed = _empirical_sst_climatology_from_training(
        fields, meta, computed_day, train_end)
    sst_climatology_max_error_kelvin = float(np.max(np.abs(
        supplied_sst_kelvin - recomputed)))
    sst_climatology_verified = bool(
        sst_climatology_max_error_kelvin
        <= sst_climatology_atol_kelvin)
    if verify_sst_climatology and not sst_climatology_verified:
        raise ValueError(
            "meta['climatology'] is not compatible with the training-only "
            "empirical SST month/day climatology reconstructed through the "
            "float32 normalized cache "
            f"(max discrepancy {sst_climatology_max_error_kelvin:.3e} K, "
            f"allowed {sst_climatology_atol_kelvin:.3e} K)")

    climatology = np.zeros(
        (367, n_channels, height, width), dtype=np.float32)
    climatology[:, 0] = (
        (supplied_sst_kelvin - channel_mean[0]) / channel_std[0]
    ).astype(np.float32)

    training_design = _harmonic_design(
        calendar_phase(times[:train_end]), aux_harmonics)
    gram = training_design.T @ training_design
    reference_times = np.arange(
        np.datetime64("2000-01-01"),
        np.datetime64("2001-01-01"),
        dtype="datetime64[D]",
    )
    reference_day = np.asarray(calendar_day_index(reference_times), dtype=int)
    reference_design = _harmonic_design(
        calendar_phase(reference_times), aux_harmonics)
    n_features = int(training_design.shape[1])
    aux_coefficients = np.zeros(
        (max(0, n_channels - 1), n_features, height, width),
        dtype=np.float32,
    )
    valid_masks = np.asarray(meta["channel_valid_masks"], dtype=bool)
    for channel in range(1, n_channels):
        targets = np.asarray(
            fields[:train_end, channel], dtype=np.float64).reshape(
                train_end, -1)
        rhs = training_design.T @ targets
        coefficients = np.linalg.solve(gram, rhs)
        mask_flat = valid_masks[channel].reshape(-1)
        coefficients[:, ~mask_flat] = 0.0
        aux_coefficients[channel - 1] = coefficients.reshape(
            n_features, height, width).astype(np.float32)
        evaluated = reference_design @ coefficients
        climatology[reference_day, channel] = evaluated.reshape(
            len(reference_day), height, width).astype(np.float32)
        climatology[0, channel] = climatology[1, channel]

    # Invalid cells are identically zero in the raw FNO state.  Keeping both
    # their calendar term and anomaly at zero prevents land-fill artifacts.
    climatology *= valid_masks[None]

    scales = np.empty(n_channels, dtype=np.float64)
    for channel in range(n_channels):
        mask = valid_masks[channel]
        count = 0
        total = 0.0
        total_sq = 0.0
        for start in range(0, train_end, int(batch_size)):
            stop = min(train_end, start + int(batch_size))
            anomaly = (
                np.asarray(fields[start:stop, channel], dtype=np.float64)
                - np.asarray(
                    climatology[computed_day[start:stop], channel],
                    dtype=np.float64,
                )
            )
            values = anomaly[:, mask]
            count += int(values.size)
            total += float(np.sum(values, dtype=np.float64))
            total_sq += float(np.sum(values * values, dtype=np.float64))
        if count == 0:
            raise ValueError(f"channel {channel} has no valid training cells")
        mean_anomaly = total / count
        variance = max(total_sq / count - mean_anomaly ** 2, 0.0)
        scales[channel] = max(np.sqrt(variance), float(scale_floor))
    scales = scales.astype(np.float32)

    digest = hashlib.sha256()
    digest.update(TRANSFORM_VERSION.encode("ascii"))
    digest.update(str(meta["data_fingerprint"]).encode("utf-8"))
    digest.update(str(meta["training_fingerprint"]).encode("utf-8"))
    digest.update(np.asarray(
        [train_end, n_channels, height, width, aux_harmonics],
        dtype=np.int64,
    ).tobytes())
    digest.update(
        f"verify_sst={bool(verify_sst_climatology)}|"
        f"sst_atol_K={sst_climatology_atol_kelvin:.17g}".encode("ascii"))
    _hash_array(digest, "calendar_day_index", computed_day)
    _hash_array(digest, "climatology_raw_normalized", climatology)
    _hash_array(digest, "aux_harmonic_coefficients", aux_coefficients)
    _hash_array(digest, "anomaly_scale_raw_normalized", scales)
    _hash_array(digest, "channel_valid_masks", valid_masks)
    fingerprint = digest.hexdigest()[:20]

    return {
        "version": TRANSFORM_VERSION,
        "transform_fingerprint": fingerprint,
        "source_data_fingerprint": str(meta["data_fingerprint"]),
        "source_training_fingerprint": str(meta["training_fingerprint"]),
        "train_end": train_end,
        "n_channels": n_channels,
        "spatial_shape": (height, width),
        "aux_harmonics": aux_harmonics,
        "calendar_day_index": computed_day,
        "climatology_raw_normalized": climatology,
        "sst_climatology_kelvin": supplied_sst_kelvin.astype(np.float32),
        "sst_climatology_verification_enabled": bool(
            verify_sst_climatology),
        "sst_climatology_roundtrip_max_abs_error_kelvin": (
            sst_climatology_max_error_kelvin),
        "sst_climatology_verification_atol_kelvin": (
            sst_climatology_atol_kelvin),
        "sst_climatology_verification_passed": sst_climatology_verified,
        "aux_harmonic_coefficients_raw_normalized": aux_coefficients,
        "anomaly_scale_raw_normalized": scales,
        "anomaly_scale_physical": (
            scales.astype(np.float64) * channel_std).astype(np.float32),
        "raw_channel_mean": channel_mean.astype(np.float32),
        "raw_channel_std": channel_std.astype(np.float32),
        "channel_valid_masks": valid_masks,
        "fit_time_start": str(times[0])[:10],
        "fit_time_end_inclusive": str(times[train_end - 1])[:10],
    }


def _validate_transform_source(
        dataset: Mapping[str, Any], transform: Mapping[str, Any]) -> None:
    meta, _, train_end, n_channels, height, width = _validate_raw_dataset(dataset)
    _require_mapping_keys(
        transform,
        (
            "transform_fingerprint",
            "source_data_fingerprint",
            "source_training_fingerprint",
            "train_end",
            "n_channels",
            "spatial_shape",
            "calendar_day_index",
            "climatology_raw_normalized",
            "anomaly_scale_raw_normalized",
            "channel_valid_masks",
        ),
        "transform",
    )
    if str(transform["source_data_fingerprint"]) != str(
            meta["data_fingerprint"]):
        raise ValueError("transform data fingerprint does not match dataset")
    if str(transform["source_training_fingerprint"]) != str(
            meta["training_fingerprint"]):
        raise ValueError("transform training fingerprint does not match dataset")
    if int(transform["train_end"]) != train_end:
        raise ValueError("transform train_end does not match dataset")
    if int(transform["n_channels"]) != n_channels:
        raise ValueError("transform channel count does not match dataset")
    if tuple(transform["spatial_shape"]) != (height, width):
        raise ValueError("transform spatial shape does not match dataset")


def transform_raw_states(
        dataset: Mapping[str, Any],
        transform: Mapping[str, Any],
        raw_states: Any,
        absolute_indices: Any,
) -> np.ndarray:
    """Convert raw FNO z-score states to structured anomalies.

    Inputs and the returned array are dimensionless.  ``raw_states`` must have
    shape ``(B,C,H,W)`` (or ``(C,H,W)`` for one state).  ``absolute_indices``
    identifies each state's calendar date; no observed value at those dates is
    read by this function.
    """
    _validate_transform_source(dataset, transform)
    array = np.asarray(raw_states, dtype=np.float64)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError("raw_states must have shape (B,C,H,W) or (C,H,W)")
    expected = (
        int(transform["n_channels"]),
        *tuple(int(v) for v in transform["spatial_shape"]),
    )
    if tuple(array.shape[1:]) != expected:
        raise ValueError(
            f"raw state shape {tuple(array.shape[1:])} does not match {expected}")
    indices = np.asarray(absolute_indices, dtype=int).reshape(-1)
    if len(indices) != len(array):
        raise ValueError("absolute_indices must have one entry per raw state")
    calendar = np.asarray(transform["calendar_day_index"], dtype=int)
    if np.any(indices < 0) or np.any(indices >= len(calendar)):
        raise IndexError("absolute state index lies outside the dataset calendar")
    centers = np.asarray(
        transform["climatology_raw_normalized"], dtype=np.float64)[
            calendar[indices]]
    scales = np.asarray(
        transform["anomaly_scale_raw_normalized"], dtype=np.float64)
    if scales.shape != (expected[0],):
        raise ValueError("transform anomaly scales have an incompatible shape")
    result = (array - centers) / scales[None, :, None, None]
    result *= np.asarray(
        transform["channel_valid_masks"], dtype=bool)[None]
    return result


def inverse_transform_sst(
        dataset: Mapping[str, Any],
        transform: Mapping[str, Any],
        structured_sst: Any,
        absolute_indices: Any,
        *,
        output_units: OutputUnits = "raw_normalized",
) -> np.ndarray:
    """Decode structured SST anomalies into requested units.

    ``raw_normalized`` is the original FNO channel z-score.  ``kelvin`` applies
    the training-window raw channel mean/std after restoring the empirical
    calendar climatology.  ``structured`` returns a copy of the input.
    """
    _validate_transform_source(dataset, transform)
    height, width = tuple(int(v) for v in transform["spatial_shape"])
    array = np.asarray(structured_sst, dtype=np.float64)
    if array.ndim == 1:
        array = array[None]
    if array.ndim == 2 and array.shape[1] == height * width:
        array = array.reshape(-1, height, width)
    if array.ndim == 2 and array.shape == (height, width):
        array = array[None]
    if array.ndim != 3 or tuple(array.shape[1:]) != (height, width):
        raise ValueError(
            "structured_sst must have shape (B,H,W), (B,H*W), or (H,W)")
    indices = np.asarray(absolute_indices, dtype=int).reshape(-1)
    if len(indices) != len(array):
        raise ValueError("absolute_indices must have one entry per SST field")
    if output_units == "structured":
        return array.copy()
    calendar = np.asarray(transform["calendar_day_index"], dtype=int)
    if np.any(indices < 0) or np.any(indices >= len(calendar)):
        raise IndexError("absolute state index lies outside the dataset calendar")
    center = np.asarray(
        transform["climatology_raw_normalized"], dtype=np.float64)[
            calendar[indices], 0]
    scale = float(np.asarray(
        transform["anomaly_scale_raw_normalized"], dtype=np.float64)[0])
    raw_normalized = center + scale * array
    if output_units == "raw_normalized":
        return raw_normalized
    if output_units == "kelvin":
        meta = dataset["meta"]
        return (
            raw_normalized
            * float(np.asarray(meta["channel_std"], dtype=np.float64)[0])
            + float(np.asarray(meta["channel_mean"], dtype=np.float64)[0])
        )
    raise ValueError(
        "output_units must be 'structured', 'raw_normalized', or 'kelvin'")


def _structured_channel(
        fields: np.ndarray,
        transform: Mapping[str, Any],
        channel: int,
        end_exclusive: int,
) -> np.ndarray:
    """Materialize a fit-window channel of the structured view in float32."""
    end_exclusive = int(end_exclusive)
    if end_exclusive < 1 or end_exclusive > len(fields):
        raise ValueError("structured-channel end is outside the archive")
    calendar = np.asarray(
        transform["calendar_day_index"], dtype=int)[:end_exclusive]
    center = np.asarray(
        transform["climatology_raw_normalized"], dtype=np.float32)[
            calendar, channel]
    scale = float(np.asarray(
        transform["anomaly_scale_raw_normalized"], dtype=np.float64)[channel])
    result = (
        (
            np.asarray(
                fields[:end_exclusive, channel], dtype=np.float32
            )
            - center
        )
        / scale
    ).astype(np.float32)
    result *= np.asarray(
        transform["channel_valid_masks"], dtype=bool)[channel][None]
    return result


def fit_structured_snapshot_dictionary(
        dataset: Mapping[str, Any],
        transform: Mapping[str, Any],
        *,
        eof_counts: Sequence[int] | None = None,
        max_duty_cycle: float = 0.8,
) -> dict[str, Any]:
    """Fit per-channel weighted EOFs in structured anomaly coordinates.

    Only fitting-window coefficients are computed or stored. Held-out launch
    states are encoded on demand by :func:`encode_state`, so building
    an evaluation artifact never reads a held-out state. Only one
    transformed channel is materialized at a time, avoiding a second full
    ~1 GiB 11-channel archive.
    """
    _validate_transform_source(dataset, transform)
    meta, fields, train_end, n_channels, _, _ = _validate_raw_dataset(dataset)
    if eof_counts is None:
        params = meta.get("params", {})
        eof_counts = (
            int(params.get("n_eof_sst", 64)),
            *([int(params.get("n_eof_aux", 8))] * (n_channels - 1)),
        )
    eof_counts = tuple(int(value) for value in eof_counts)
    if len(eof_counts) != n_channels:
        raise ValueError(
            f"expected {n_channels} EOF counts, got {len(eof_counts)}")
    if any(value <= 0 for value in eof_counts):
        raise ValueError("all EOF counts must be positive")

    coefficient_scales = np.sqrt(
        np.asarray(meta["channel_weights"], dtype=np.float64))
    means: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    coefficient_blocks: list[np.ndarray] = []
    slices: list[tuple[int, int]] = []
    offset = 0
    for channel, requested in enumerate(eof_counts):
        structured = _structured_channel(
            fields, transform, channel, train_end)
        matrix = np.asarray(
            structured.reshape(train_end, -1), dtype=np.float64)
        mean_channel = matrix.mean(axis=0)
        weights = _channel_spatial_weights(dataset, channel)
        sqrt_weights = np.sqrt(np.clip(weights, 1e-12, None))
        weighted = (
            matrix - mean_channel) * sqrt_weights[None]
        leading = _leading_weighted_eofs(
            weighted, requested, max_duty_cycle=max_duty_cycle)
        rank_channel = int(leading.shape[0])
        basis_channel = leading / sqrt_weights[None]
        basis_channel[:, weights <= 0.0] = 0.0
        coefficients = (
            (matrix - mean_channel)
            @ (basis_channel * weights[None]).T
        )
        means.append(mean_channel.astype(np.float32))
        bases.append(basis_channel.astype(np.float32))
        coefficient_blocks.append(
            (coefficient_scales[channel] * coefficients).astype(np.float32))
        slices.append((offset, offset + rank_channel))
        offset += rank_channel

    return {
        "field_means": means,
        "eof_bases": bases,
        "coeffs": np.concatenate(coefficient_blocks, axis=1),
        "coefficient_state_scope": "fit_window_only",
        "maximum_observation_index_encoded": train_end - 1,
        "coeff_slices": slices,
        "eof_counts": tuple(int(basis.shape[0]) for basis in bases),
        "coefficient_scales": coefficient_scales.astype(np.float32),
        "transform_fingerprint": str(transform["transform_fingerprint"]),
    }


def _fit_joint_projection(
        coefficients: np.ndarray,
        train_end: int,
        rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one joint PCA and return ``mean, P, q_source, q_data``."""
    source = np.asarray(coefficients[:train_end - 1], dtype=np.float64)
    observed_target = np.asarray(
        coefficients[1:train_end], dtype=np.float64)
    mean = source.mean(axis=0)
    centered = source - mean
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    rank = min(int(rank), right.shape[0], right.shape[1])
    if rank <= 0:
        raise ValueError("joint rank must be positive")
    projection = right[:rank].T
    # Fix arbitrary PCA signs to make fingerprints backend-stable.
    pivots = np.argmax(np.abs(projection), axis=0)
    signs = np.sign(projection[pivots, np.arange(rank)])
    signs[signs == 0.0] = 1.0
    projection *= signs[None]
    q_source = centered @ projection
    q_data = (observed_target - mean) @ projection
    return mean, projection, q_source, q_data


def encode_frozen_raw_fno_targets(
        result: Any,
        dataset: Mapping[str, Any],
        transform: Mapping[str, Any],
        dictionary: Mapping[str, Any],
        *,
        batch_size: int = 24,
        max_duty_cycle: float = 0.8,
) -> tuple[np.ndarray, str]:
    """Apply the frozen raw FNO and encode target-date structured anomalies.

    The operator input is *always* ``meta["state_fields"][t]`` in the original
    raw z-score domain.  Its output is transformed using the known calendar
    term for ``t+1`` before EOF encoding.  Operator parameters are fingerprinted
    before and after inference and must remain byte-identical.
    """
    _validate_transform_source(dataset, transform)
    meta, fields, train_end, n_channels, height, width = (
        _validate_raw_dataset(dataset))
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if not hasattr(result, "operator_pure") or not hasattr(result, "grid"):
        raise TypeError("result must expose operator_pure and grid")
    if str(getattr(result, "kind", dataset["kind"])) != str(dataset["kind"]):
        raise ValueError("FNO result kind does not match the raw dataset")

    operator = result.operator_pure
    before_fingerprint = _operator_fingerprint(operator)
    operator.eval()
    try:
        operator_device = next(operator.parameters()).device
    except StopIteration:
        operator_device = device
    grid = torch.as_tensor(
        result.grid, dtype=torch.float32, device=operator_device)
    encoded: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, train_end - 1, int(batch_size)):
            stop = min(start + int(batch_size), train_end - 1)
            pacing_started = _pacing_start(max_duty_cycle)
            raw_input = torch.as_tensor(
                fields[start:stop].reshape(-1, n_channels * height * width),
                dtype=torch.float32,
                device=operator_device,
            )
            raw_prediction = apply_operator(
                operator, raw_input, grid, dataset["kind"]).reshape(
                    -1, n_channels, height, width).cpu().numpy()
            _pace_cuda(pacing_started, max_duty_cycle)
            target_indices = np.arange(start + 1, stop + 1, dtype=int)
            structured_prediction = transform_raw_states(
                dataset, transform, raw_prediction, target_indices)
            encoded.append(encode_state(
                dataset, dictionary, structured_prediction))
    after_fingerprint = _operator_fingerprint(operator)
    if after_fingerprint != before_fingerprint:
        raise RuntimeError("raw FNO parameters changed during frozen inference")
    coefficients = np.concatenate(encoded, axis=0)
    if coefficients.shape[0] != train_end - 1:
        raise RuntimeError("frozen FNO target encoding has the wrong length")
    return coefficients, before_fingerprint


def fit_affine_ridge_map(
        source: Any,
        target: Any,
        *,
        ridge: float = 3e-4,
        fit_affine: bool = False,
        sample_weights: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit ``target ~= source @ A.T + b`` in reduced coordinates.

    ``ridge`` is dimensionless and scaled by ``trace(X.T W X)/rank``.  When
    ``fit_affine`` is true, the intercept is not regularized.
    """
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("source and target must be finite 2-D arrays of equal shape")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("source and target coordinates must be finite")
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    n_samples, rank = x.shape
    if n_samples == 0 or rank == 0:
        raise ValueError("source and target cannot be empty")
    if sample_weights is None:
        weights = np.ones(n_samples, dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if len(weights) != n_samples:
            raise ValueError("sample_weights must match the number of pairs")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weights must be finite and nonnegative")
        if float(weights.sum()) <= 0.0:
            raise ValueError("sample_weights must have positive total mass")
    weights = weights * (n_samples / weights.sum())

    if fit_affine:
        x_mean = np.average(x, axis=0, weights=weights)
        y_mean = np.average(y, axis=0, weights=weights)
    else:
        x_mean = np.zeros(rank, dtype=np.float64)
        y_mean = np.zeros(rank, dtype=np.float64)
    xc = x - x_mean
    yc = y - y_mean
    weighted_x = xc * weights[:, None]
    gram = xc.T @ weighted_x
    ridge_lambda = (
        ridge * float(np.trace(gram)) / rank if ridge > 0.0 else 0.0)
    rhs = xc.T @ (yc * weights[:, None])
    regularized = gram + ridge_lambda * np.eye(rank)
    try:
        matrix = np.linalg.solve(regularized, rhs).T
    except np.linalg.LinAlgError:
        matrix = np.linalg.lstsq(regularized, rhs, rcond=None)[0].T
    intercept = y_mean - matrix @ x_mean if fit_affine else y_mean
    fitted = x @ matrix.T + intercept
    residual = fitted - y
    weighted_mse = float(np.sum(
        weights[:, None] * residual ** 2) / (weights.sum() * rank))
    diagnostics = {
        "ridge": ridge,
        "ridge_lambda": float(ridge_lambda),
        "fit_affine": bool(fit_affine),
        "reduced_target_rmse": float(np.sqrt(weighted_mse)),
        "raw_spectral_radius": float(np.max(np.abs(
            np.linalg.eigvals(matrix)))),
        "raw_matrix_condition": float(np.linalg.cond(matrix)),
    }
    return matrix, intercept, diagnostics


def materialize_stable_matrix(
        matrix: Any,
        *,
        max_radius: float | None = 0.999,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Materialize a real-Schur spectral-radius-clipped matrix.

    Clipping is applied once and the returned real matrix itself is used by
    every recurrence.  Real 1x1 and conjugate
    2x2 Schur diagonal blocks are radially scaled; upper-block couplings are
    kept. This controls eigenvalue radii but intentionally does not claim
    to remove non-normal transient amplification.
    """
    raw = np.asarray(matrix, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(raw)):
        raise ValueError("matrix must be finite")
    raw_eigenvalues = np.linalg.eigvals(raw)
    raw_radius = float(np.max(np.abs(raw_eigenvalues)))
    if max_radius is None:
        stable = raw.copy()
        stable_eigenvalues = raw_eigenvalues
        clipped_modes = 0
        method = "none"
        reconstruction_error = 0.0
    else:
        max_radius = float(max_radius)
        if not np.isfinite(max_radius) or not 0.0 < max_radius <= 1.0:
            raise ValueError("max_radius must lie in (0, 1] or be None")
        triangular, vectors = schur(raw, output="real")
        reconstructed = vectors @ triangular @ vectors.T
        reconstruction_error = float(
            np.linalg.norm(reconstructed - raw)
            / max(np.linalg.norm(raw), 1e-15))
        clipped = triangular.copy()
        tolerance = (
            np.finfo(np.float64).eps
            * max(1.0, float(np.linalg.norm(triangular, ord=np.inf)))
            * max(1, raw.shape[0])
            * 10.0
        )
        index = 0
        while index < raw.shape[0]:
            is_pair = (
                index + 1 < raw.shape[0]
                and abs(triangular[index + 1, index]) > tolerance
            )
            block_size = 2 if is_pair else 1
            block = triangular[
                index:index + block_size, index:index + block_size]
            block_radius = float(np.max(np.abs(np.linalg.eigvals(block))))
            if block_radius > max_radius:
                clipped[
                    index:index + block_size,
                    index:index + block_size,
                ] *= max_radius / block_radius
            index += block_size
        stable = np.asarray(vectors @ clipped @ vectors.T, dtype=np.float64)
        stable_eigenvalues = np.linalg.eigvals(stable)
        stable_radius = float(np.max(np.abs(stable_eigenvalues)))
        numerical_slack = 5e-10 * max(1.0, max_radius)
        numerical_guard_scale = 1.0
        if stable_radius > max_radius + numerical_slack:
            # Reconstructing a highly non-normal matrix from an orthogonal
            # Schur basis can perturb sensitive eigenvalues a few ulps beyond
            # the block radius.  Apply only a tightly bounded, recorded
            # numerical guard.  A material correction would hide a genuinely
            # ill-conditioned stabilization and is therefore rejected.
            target_radius = max_radius * (1.0 - 1e-7)
            for _ in range(4):
                correction = target_radius / stable_radius
                numerical_guard_scale *= correction
                if numerical_guard_scale < 0.9999:
                    raise RuntimeError(
                        "Schur reconstruction would require a material global "
                        "contraction to respect the requested radius: "
                        f"scale={numerical_guard_scale:.12g}")
                stable *= correction
                stable_eigenvalues = np.linalg.eigvals(stable)
                stable_radius = float(np.max(np.abs(stable_eigenvalues)))
                if stable_radius <= max_radius + numerical_slack:
                    break
            if stable_radius > max_radius + numerical_slack:
                raise RuntimeError(
                    "materialized Schur-clipped matrix exceeds requested "
                    f"radius after numerical guard: {stable_radius:.12g} > "
                    f"{max_radius:.12g}")
        clipped_modes = int(np.count_nonzero(
            np.abs(raw_eigenvalues) > max_radius))
        method = "real_schur_block_radial_clip"

    stable_radius = float(np.max(np.abs(stable_eigenvalues)))
    info = {
        "method": method,
        "max_radius": max_radius,
        "raw_spectral_radius": raw_radius,
        "spectral_radius": stable_radius,
        "clipped_modes": clipped_modes,
        "raw_eigenvalues": raw_eigenvalues,
        "stable_eigenvalues": stable_eigenvalues,
        "raw_matrix_condition": float(np.linalg.cond(raw)),
        "matrix_condition": float(np.linalg.cond(stable)),
        "schur_reconstruction_relative_error": reconstruction_error,
        "numerical_guard_scale": (
            1.0 if max_radius is None else numerical_guard_scale),
    }
    return stable, info


def joint_decoder_residuals(
        dataset: Mapping[str, Any],
        transform: Mapping[str, Any],
        model: Mapping[str, Any],
        absolute_indices: Any,
) -> np.ndarray:
    """Return residuals of the joint-PCA SST decoder.

    The returned dimensionless structured-anomaly residual is

    ``e_t = z_sst_t - decode_sst(mu + P P.T (c_t - mu))``.

    It therefore includes both SST-EOF truncation and joint-PCA truncation,
    matching the actual BSO decoder.
    """
    _validate_transform_source(dataset, transform)
    _require_mapping_keys(
        model, ("coeffs", "coeff_mean", "P", "eof_bases", "coeff_slices"),
        "model")
    indices = np.asarray(absolute_indices, dtype=int).reshape(-1)
    coefficients = np.asarray(model["coeffs"])
    if np.any(indices < 0) or np.any(indices >= len(coefficients)):
        raise IndexError("residual index lies outside encoded state coefficients")
    mean = np.asarray(model["coeff_mean"], dtype=np.float64)
    projection = np.asarray(model["P"], dtype=np.float64)
    selected = np.asarray(coefficients[indices], dtype=np.float64)
    q = (selected - mean) @ projection
    projected = mean + q @ projection.T
    decoded = decode_sst(model, projected)

    raw_fields = np.asarray(dataset["meta"]["state_fields"])
    raw_sst = np.asarray(raw_fields[indices, 0], dtype=np.float64)
    calendar = np.asarray(transform["calendar_day_index"], dtype=int)
    center = np.asarray(
        transform["climatology_raw_normalized"], dtype=np.float64)[
            calendar[indices], 0]
    scale = float(np.asarray(
        transform["anomaly_scale_raw_normalized"], dtype=np.float64)[0])
    actual = (raw_sst - center) / scale
    actual *= np.asarray(
        transform["channel_valid_masks"], dtype=bool)[0][None]
    return actual.reshape(len(indices), -1) - decoded


def fit_decoder_residual_closures(
        dataset: Mapping[str, Any],
        transform: Mapping[str, Any],
        model: Mapping[str, Any],
        *,
        max_abs_gamma: float = 0.999,
        batch_size: int = 256,
) -> dict[str, dict[str, Any]]:
    """Fit no-carry and scalar one-step AR(1) residual closures.

    The scalar gamma uses only consecutive residuals inside the fitting window
    and latitude/ocean weights.  It is clipped to a strictly stable interval.
    No higher-order lag is used.
    """
    _validate_transform_source(dataset, transform)
    train_end = int(transform["train_end"])
    if int(batch_size) <= 1:
        raise ValueError("batch_size must exceed one")
    max_abs_gamma = float(max_abs_gamma)
    if not np.isfinite(max_abs_gamma) or not 0.0 <= max_abs_gamma < 1.0:
        raise ValueError("max_abs_gamma must lie in [0, 1)")
    weights = np.asarray(
        dataset["meta"]["weights_2d"], dtype=np.float64).reshape(1, -1)

    numerator = 0.0
    denominator = 0.0
    residual_energy = 0.0
    last_residual: np.ndarray | None = None
    for start in range(0, train_end, int(batch_size)):
        stop = min(train_end, start + int(batch_size))
        residual = joint_decoder_residuals(
            dataset, transform, model, np.arange(start, stop))
        residual_energy += float(np.sum(weights * residual ** 2))
        if last_residual is not None:
            numerator += float(
                np.sum(weights * last_residual * residual[:1])
            )
            denominator += float(
                np.sum(weights * last_residual ** 2)
            )
        if len(residual) > 1:
            numerator += float(np.sum(
                weights * residual[:-1] * residual[1:]))
            denominator += float(np.sum(weights * residual[:-1] ** 2))
        last_residual = residual[-1:]
    raw_gamma = numerator / max(denominator, 1e-15)
    gamma = float(np.clip(raw_gamma, -max_abs_gamma, max_abs_gamma))
    weighted_rmse = float(np.sqrt(residual_energy / train_end))
    common = {
        "definition": "joint_pca_sst_decoder_residual",
        "fit_train_end_exclusive": train_end,
        "training_weighted_residual_rmse_structured": weighted_rmse,
    }
    return {
        "none": {
            **common,
            "mode": "none",
            "gamma": None,
        },
        "scalar_ar1": {
            **common,
            "mode": "scalar_ar1",
            "gamma": gamma,
            "raw_gamma": float(raw_gamma),
            "max_abs_gamma": max_abs_gamma,
            "clipped": bool(not np.isclose(gamma, raw_gamma)),
        },
    }


def blend_operator_key(alpha: float) -> str:
    """Return the canonical key for an observation/FNO target blend."""
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if np.isclose(alpha, 0.0, rtol=0.0, atol=1e-15):
        return "A_FNO"
    if np.isclose(alpha, 1.0, rtol=0.0, atol=1e-15):
        return "A_data"
    return f"A_blend_alpha={alpha:.12g}"


def _refresh_model_fingerprint(model: dict[str, Any]) -> str:
    """Refresh both the full-state digest and its short reporting token."""
    integrity = model_state_integrity_sha256(model)
    model["fingerprint_schema"] = MODEL_FINGERPRINT_VERSION
    model["full_state_integrity_sha256"] = integrity
    model["fingerprint"] = integrity[:20]
    return str(model["fingerprint"])


def build_structured_bso(
        result: Any,
        dataset: Mapping[str, Any],
        *,
        rank: int = 128,
        ridge: float = 3e-4,
        eof_counts: Sequence[int] | None = None,
        alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        fit_affine: bool = False,
        aux_harmonics: int = 6,
        stabilize_max_radius: float | None = 0.999,
        residual_max_abs_gamma: float = 0.999,
        sample_weights: Any | None = None,
        batch_size: int = 24,
        max_duty_cycle: float = 0.8,
) -> dict[str, Any]:
    """Build FNO, data, and blended one-step BSO matrices in one coordinate set.

    The expensive EOF/joint-PCA/FNO projection is shared across all target
    choices.  For each ``alpha``,

    ``q_target = alpha*q_data + (1-alpha)*q_FNO``.

    ``A_data`` is explicitly labelled as the EOF-LIM/DMD comparator.  Every
    operator stores both ``A_raw`` and the actually rolled-out, materialized
    stable matrix ``A``.
    """
    if int(rank) <= 0:
        raise ValueError("rank must be positive")
    alpha_values = tuple(sorted({float(value) for value in alphas}))
    if not alpha_values:
        raise ValueError("alphas cannot be empty")
    for alpha in alpha_values:
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("all alphas must lie in [0, 1]")

    transform = fit_structured_anomaly_transform(
        dataset, aux_harmonics=aux_harmonics)
    dictionary = fit_structured_snapshot_dictionary(
        dataset,
        transform,
        eof_counts=eof_counts,
        max_duty_cycle=max_duty_cycle,
    )
    train_end = int(transform["train_end"])
    mean, projection, q_source, q_data = _fit_joint_projection(
        np.asarray(dictionary["coeffs"]), train_end, int(rank))
    fno_coefficients, operator_fingerprint = encode_frozen_raw_fno_targets(
        result,
        dataset,
        transform,
        dictionary,
        batch_size=batch_size,
        max_duty_cycle=max_duty_cycle,
    )
    q_fno = (
        np.asarray(fno_coefficients, dtype=np.float64) - mean) @ projection
    if q_fno.shape != q_data.shape:
        raise RuntimeError("FNO and observed reduced targets have different shapes")

    model: dict[str, Any] = dict(dictionary)
    model.update({
        "version": MODEL_VERSION,
        "transform": transform,
        "coeff_mean": mean,
        "P": projection,
        "rank": int(projection.shape[1]),
        "ridge": float(ridge),
        "fit_affine": bool(fit_affine),
        "stabilize_max_radius": stabilize_max_radius,
        "raw_operator_fingerprint": operator_fingerprint,
        "raw_operator_training_fingerprint": str(
            dataset["meta"]["training_fingerprint"]),
        "operators": {},
        "alpha_grid": alpha_values,
        # Compact reduced coordinates allow the BSO recurrence to be built
        # without applying the frozen FNO to every fitting day again.
        "fit_coordinates": {
            "q_source": np.asarray(q_source, dtype=np.float32),
            "q_data": np.asarray(q_data, dtype=np.float32),
            "q_fno": np.asarray(q_fno, dtype=np.float32),
        },
    })

    def fit_one(
            key: str,
            target: np.ndarray,
            target_kind: str,
            alpha: float | None,
    ) -> None:
        raw_matrix, intercept, fit_info = fit_affine_ridge_map(
            q_source,
            target,
            ridge=ridge,
            fit_affine=fit_affine,
            sample_weights=sample_weights,
        )
        stable_matrix, stable_info = materialize_stable_matrix(
            raw_matrix, max_radius=stabilize_max_radius)
        model["operators"][key] = {
            "key": key,
            "target_kind": target_kind,
            "alpha_data": alpha,
            "A_raw": raw_matrix,
            # A is deliberately the materialized matrix used by recurrence.
            "A": stable_matrix,
            "b": intercept,
            **fit_info,
            **stable_info,
        }

    # Include the two endpoint maps together with the configured blend.
    fit_one("A_FNO", q_fno, "frozen_raw_fno", 0.0)
    fit_one("A_data", q_data, "observed_eof_lim_dmd", 1.0)
    for alpha in alpha_values:
        key = blend_operator_key(alpha)
        if key in model["operators"]:
            continue
        target = alpha * q_data + (1.0 - alpha) * q_fno
        fit_one(key, target, "observation_fno_blend", alpha)

    model["residual_closures"] = fit_decoder_residual_closures(
        dataset,
        transform,
        model,
        max_abs_gamma=residual_max_abs_gamma,
    )
    _refresh_model_fingerprint(model)
    return model


def refit_structured_bso_operators(
        model: Mapping[str, Any],
        *,
        ridge: float,
        fit_affine: bool,
        alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        stabilize_max_radius: float | None = 0.999,
        sample_weights: Any | None = None,
) -> dict[str, Any]:
    """Refit only reduced maps while reusing the expensive frozen encoding.

    EOFs, the joint projection, the transformed raw-FNO targets, and residual
    closure remain fixed.  The returned dictionary is a shallow structural
    copy: large immutable representation arrays are shared with ``model``,
    while the operator mapping and fingerprint are new.
    """
    _require_mapping_keys(
        model,
        (
            "version",
            "transform",
            "rank",
            "fit_coordinates",
            "residual_closures",
            "raw_operator_fingerprint",
        ),
        "model",
    )
    if model["version"] != MODEL_VERSION:
        raise ValueError(f"unsupported BSO model version {model['version']!r}")
    coordinates = model["fit_coordinates"]
    _require_mapping_keys(
        coordinates, ("q_source", "q_data", "q_fno"),
        "model['fit_coordinates']")
    q_source = np.asarray(coordinates["q_source"], dtype=np.float64)
    q_data = np.asarray(coordinates["q_data"], dtype=np.float64)
    q_fno = np.asarray(coordinates["q_fno"], dtype=np.float64)
    if q_source.shape != q_data.shape or q_source.shape != q_fno.shape:
        raise ValueError("stored reduced fitting coordinates have inconsistent shapes")

    alpha_values = tuple(sorted({float(value) for value in alphas}))
    if not alpha_values:
        raise ValueError("alphas cannot be empty")
    for alpha in alpha_values:
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("all alphas must lie in [0, 1]")

    updated = dict(model)
    updated.update({
        "ridge": float(ridge),
        "fit_affine": bool(fit_affine),
        "stabilize_max_radius": stabilize_max_radius,
        "alpha_grid": alpha_values,
        "operators": {},
    })

    def fit_one(
            key: str,
            target: np.ndarray,
            target_kind: str,
            alpha: float | None,
    ) -> None:
        raw_matrix, intercept, fit_info = fit_affine_ridge_map(
            q_source,
            target,
            ridge=ridge,
            fit_affine=fit_affine,
            sample_weights=sample_weights,
        )
        stable_matrix, stable_info = materialize_stable_matrix(
            raw_matrix, max_radius=stabilize_max_radius)
        updated["operators"][key] = {
            "key": key,
            "target_kind": target_kind,
            "alpha_data": alpha,
            "A_raw": raw_matrix,
            "A": stable_matrix,
            "b": intercept,
            **fit_info,
            **stable_info,
        }

    fit_one("A_FNO", q_fno, "frozen_raw_fno", 0.0)
    fit_one("A_data", q_data, "observed_eof_lim_dmd", 1.0)
    for alpha in alpha_values:
        key = blend_operator_key(alpha)
        if key in updated["operators"]:
            continue
        target = alpha * q_data + (1.0 - alpha) * q_fno
        fit_one(key, target, "observation_fno_blend", alpha)
    _refresh_model_fingerprint(updated)
    return updated


def rollout_structured_bso(
        dataset: Mapping[str, Any],
        model: Mapping[str, Any],
        origins: Any,
        n_steps: int,
        *,
        operator_key: str = "A_FNO",
        residual_mode: ResidualMode = "scalar_ar1",
        origin_base: int | None = None,
        output_units: OutputUnits = "raw_normalized",
) -> np.ndarray:
    """Autonomously roll out a materialized BSO matrix.

    Only the multichannel launch state is observed.  Later dates contribute
    their deterministic calendar term but no future SST or auxiliary value.
    The result has shape ``(n_steps+1, n_origins, H, W)``.
    """
    _require_mapping_keys(
        model,
        (
            "version",
            "transform",
            "coeff_mean",
            "P",
            "operators",
            "residual_closures",
        ),
        "model",
    )
    if model["version"] != MODEL_VERSION:
        raise ValueError(f"unsupported BSO model version {model['version']!r}")
    transform = model["transform"]
    _validate_transform_source(dataset, transform)
    if operator_key not in model["operators"]:
        raise KeyError(
            f"unknown operator {operator_key!r}; choices are "
            f"{sorted(model['operators'])}")
    if residual_mode not in ("none", "scalar_ar1"):
        raise ValueError("residual_mode must be 'none' or 'scalar_ar1'")
    if residual_mode not in model["residual_closures"]:
        raise KeyError(f"model does not contain residual closure {residual_mode!r}")
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")

    meta, fields, train_end, _, height, width = _validate_raw_dataset(dataset)
    base = train_end if origin_base is None else int(origin_base)
    origin_offsets = np.asarray(origins, dtype=int).reshape(-1)
    absolute = base + origin_offsets
    if len(absolute) == 0:
        raise ValueError("origins cannot be empty")
    if np.any(absolute < 0) or np.any(absolute + n_steps >= len(fields)):
        raise IndexError("origin/lead lies outside the available calendar")

    initial_structured = transform_raw_states(
        dataset, transform, fields[absolute], absolute)
    initial_coefficients = encode_state(
        dataset, model, initial_structured)
    mean = np.asarray(model["coeff_mean"], dtype=np.float64)
    projection = np.asarray(model["P"], dtype=np.float64)
    q = (initial_coefficients - mean) @ projection

    operator = model["operators"][operator_key]
    matrix = np.asarray(operator["A"], dtype=np.float64)
    intercept = np.asarray(operator["b"], dtype=np.float64)
    if matrix.shape != (q.shape[1], q.shape[1]):
        raise ValueError("operator matrix shape does not match joint coordinates")
    if intercept.shape != (q.shape[1],):
        raise ValueError("operator intercept shape does not match joint coordinates")

    initial_projected = mean + q @ projection.T
    initial_decoded = decode_sst(model, initial_projected)
    initial_residual = (
        initial_structured[:, 0].reshape(len(absolute), -1)
        - initial_decoded
    )
    closure = model["residual_closures"][residual_mode]
    gamma = (
        None if residual_mode == "none" else float(closure["gamma"]))

    trajectory: list[np.ndarray] = []
    for lead in range(n_steps + 1):
        coefficients = mean + q @ projection.T
        structured_sst = decode_sst(model, coefficients)
        if gamma is not None:
            structured_sst = (
                structured_sst + (gamma ** lead) * initial_residual)
        target_indices = absolute + lead
        decoded = inverse_transform_sst(
            dataset,
            transform,
            structured_sst.reshape(-1, height, width),
            target_indices,
            output_units=output_units,
        )
        trajectory.append(decoded)
        if lead < n_steps:
            q = q @ matrix.T + intercept
    return np.asarray(trajectory)


__all__ = [
    "MODEL_FINGERPRINT_VERSION",
    "MODEL_VERSION",
    "TRANSFORM_VERSION",
    "blend_operator_key",
    "build_structured_bso",
    "encode_frozen_raw_fno_targets",
    "joint_decoder_residuals",
    "fit_affine_ridge_map",
    "fit_decoder_residual_closures",
    "fit_structured_anomaly_transform",
    "fit_structured_snapshot_dictionary",
    "inverse_transform_sst",
    "materialize_stable_matrix",
    "model_state_integrity_sha256",
    "refit_structured_bso_operators",
    "rollout_structured_bso",
    "transform_raw_states",
    "validate_fit_window_model_state",
]
