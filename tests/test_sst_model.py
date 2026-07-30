from types import SimpleNamespace

import copy

import numpy as np
import pytest
import torch

from edmd_bso_method import device
from kbe_observation_data import calendar_day_index
from experiments.sst.sst_model import (
    build_structured_bso,
    fit_structured_anomaly_transform,
    fit_structured_snapshot_dictionary,
    inverse_transform_sst,
    materialize_stable_matrix,
    model_state_integrity_sha256,
    refit_structured_bso_operators,
    rollout_structured_bso,
    transform_raw_states,
    validate_fit_window_model_state,
    validate_model_state_integrity,
)


class _FrozenLinearFNO(torch.nn.Module):
    """Small parameterized stand-in exercising the raw-FNO inference path."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = int(n_channels)
        self.gain = torch.nn.Parameter(
            torch.tensor(0.98), requires_grad=False)

    def forward(self, fields, grid):
        del grid
        return self.gain * fields


def _synthetic_raw_dataset():
    rng = np.random.default_rng(2026)
    n_time, n_channels, height, width = 760, 2, 2, 3
    train_end = 730
    times = np.arange(
        np.datetime64("2000-01-01"),
        np.datetime64("2000-01-01") + n_time,
        dtype="datetime64[D]",
    )
    day = calendar_day_index(times)
    time_index = np.arange(n_time, dtype=np.float64)
    spatial = np.linspace(-0.3, 0.3, height * width).reshape(height, width)
    fields = np.empty(
        (n_time, n_channels, height, width), dtype=np.float32)
    fields[:, 0] = (
        0.7 * np.sin(2.0 * np.pi * time_index / 365.0)[:, None, None]
        + spatial
        + 0.08 * rng.normal(size=(n_time, height, width))
    ).astype(np.float32)
    fields[:, 1] = (
        0.4 * np.cos(2.0 * np.pi * time_index / 365.0)[:, None, None]
        - 0.2 * spatial
        + 0.06 * rng.normal(size=(n_time, height, width))
    ).astype(np.float32)

    channel_mean = np.asarray([280.0, 4.0], dtype=np.float32)
    channel_std = np.asarray([2.0, 3.0], dtype=np.float32)
    # Match the production data builder's float32 climatology arithmetic.
    sst_kelvin = fields[:, 0] * channel_std[0] + channel_mean[0]
    fallback = sst_kelvin[:train_end].mean(axis=0)
    climatology = np.tile(
        fallback[None], (367, 1, 1)).astype(np.float32)
    for calendar_day in range(1, 367):
        selected = np.flatnonzero(day[:train_end] == calendar_day)
        if selected.size:
            climatology[calendar_day] = sst_kelvin[selected].mean(axis=0)

    channel_weights = np.asarray([0.6, 0.4], dtype=np.float32)
    weights_by_channel = np.stack([
        np.full(
            (height, width),
            channel_weights[channel] / (height * width),
            dtype=np.float64,
        )
        for channel in range(n_channels)
    ])
    valid_masks = np.ones(
        (n_channels, height, width), dtype=bool)
    dataset = {
        "system": "sst",
        "kind": "grid2d_multichannel",
        "weights": weights_by_channel.reshape(-1),
        "meta": {
            "state_fields": fields,
            "sst_fields": fields[:, 0],
            "times": times,
            "train_end": train_end,
            "n_channels": n_channels,
            "sst_index": 0,
            "channel_names": ("sst", "aux"),
            "channel_mean": channel_mean,
            "channel_std": channel_std,
            "channel_valid_masks": valid_masks,
            "channel_weights": channel_weights,
            "weights_2d": np.full(
                (height, width), 1.0 / (height * width)),
            "climatology": climatology,
            "climatology_calendar": "month-day-leap-v1",
            "calendar_day_index": day,
            "training_fingerprint": "synthetic-training",
            "data_fingerprint": "synthetic-data",
            "train_pair_indices": np.arange(train_end - 2),
            "params": {
                "n_eof_sst": 4,
                "n_eof_aux": 3,
            },
        },
    }
    return dataset


@pytest.fixture(scope="module")
def built_case():
    dataset = _synthetic_raw_dataset()
    _, n_channels, height, width = (
        dataset["meta"]["state_fields"].shape)
    operator = _FrozenLinearFNO(n_channels).to(device).eval()
    result = SimpleNamespace(
        operator_pure=operator,
        grid=np.zeros((height, width, 2), dtype=np.float32),
        kind="grid2d_multichannel",
    )
    state_before = {
        name: value.detach().cpu().clone()
        for name, value in operator.state_dict().items()
    }
    model = build_structured_bso(
        result,
        dataset,
        rank=5,
        ridge=1e-4,
        eof_counts=(4, 3),
        alphas=(0.0, 0.25, 0.5, 1.0),
        fit_affine=True,
        aux_harmonics=2,
        batch_size=64,
    )
    return dataset, operator, state_before, model


def test_transform_inverse_and_climatology_verification_metadata():
    dataset = _synthetic_raw_dataset()
    transform = fit_structured_anomaly_transform(
        dataset, aux_harmonics=2)
    indices = np.asarray([0, 366, 729])
    raw = dataset["meta"]["state_fields"][indices]
    structured = transform_raw_states(
        dataset, transform, raw, indices)
    restored = inverse_transform_sst(
        dataset,
        transform,
        structured[:, 0],
        indices,
        output_units="raw_normalized",
    )
    np.testing.assert_allclose(
        restored, raw[:, 0], rtol=0.0, atol=2e-7)
    assert transform["sst_climatology_verification_passed"]
    assert (
        transform["sst_climatology_roundtrip_max_abs_error_kelvin"]
        <= transform["sst_climatology_verification_atol_kelvin"])
    assert transform["sst_climatology_verification_atol_kelvin"] == 1e-6
    assert (
        transform["sst_climatology_roundtrip_max_abs_error_kelvin"] == 0.0)


def test_fit_artifacts_do_not_inspect_or_store_held_out_states():
    dataset = _synthetic_raw_dataset()
    train_end = int(dataset["meta"]["train_end"])
    fields = np.asarray(dataset["meta"]["state_fields"]).copy()
    fields[train_end:] = np.nan
    dataset["meta"]["state_fields"] = fields
    dataset["meta"]["sst_fields"] = fields[:, 0]

    transform = fit_structured_anomaly_transform(
        dataset, aux_harmonics=2)
    dictionary = fit_structured_snapshot_dictionary(
        dataset,
        transform,
        eof_counts=(4, 3),
    )
    assert len(transform["calendar_day_index"]) == len(fields)
    assert len(dictionary["coeffs"]) == train_end
    assert dictionary["coefficient_state_scope"] == "fit_window_only"
    assert dictionary["maximum_observation_index_encoded"] == train_end - 1
    assert np.all(np.isfinite(dictionary["coeffs"]))


def test_full_state_fingerprint_covers_rollout_and_fit_state(built_case):
    _, _, _, model = built_case
    baseline = model_state_integrity_sha256(model)
    assert model["full_state_integrity_sha256"] == baseline
    assert model["fingerprint"] == baseline[:20]

    mutations = []

    changed = copy.deepcopy(model)
    changed["transform"]["climatology_raw_normalized"][0, 0, 0, 0] += 0.25
    mutations.append(changed)

    changed = copy.deepcopy(model)
    changed["eof_bases"][0][0, 0] += 0.25
    mutations.append(changed)

    changed = copy.deepcopy(model)
    changed["field_means"][0][0] += 0.25
    mutations.append(changed)

    changed = copy.deepcopy(model)
    changed["coeffs"][0, 0] += 0.25
    mutations.append(changed)

    changed = copy.deepcopy(model)
    changed["fit_coordinates"]["q_fno"][0, 0] += 0.25
    mutations.append(changed)

    changed = copy.deepcopy(model)
    changed["residual_closures"]["scalar_ar1"]["gamma"] += 0.01
    mutations.append(changed)

    for mutated in mutations:
        assert model_state_integrity_sha256(mutated) != baseline
        with pytest.raises(ValueError, match="full-state integrity"):
            validate_model_state_integrity(mutated)


def test_full_state_encoding_preserves_array_shape_and_key_types(built_case):
    scalar = {"value": np.asarray(2.5, dtype=np.float64)}
    vector = {"value": np.asarray([2.5], dtype=np.float64)}
    assert model_state_integrity_sha256(scalar) != (
        model_state_integrity_sha256(vector))
    with pytest.raises(TypeError, match="keys must all be strings"):
        model_state_integrity_sha256({1: "value"})

    _, _, _, model = built_case
    validate_model_state_integrity(model)
    changed = copy.deepcopy(model)
    changed["full_state_integrity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="full-state integrity"):
        validate_model_state_integrity(changed)


def test_fit_window_model_guard_rejects_post_fit_state(built_case):
    dataset, _, _, model = built_case
    train_end = int(dataset["meta"]["train_end"])
    validate_fit_window_model_state(model, train_end)

    leaked = copy.deepcopy(model)
    leaked["coeffs"] = np.concatenate(
        [leaked["coeffs"], leaked["coeffs"][-1:]], axis=0)
    leaked["maximum_observation_index_encoded"] = train_end
    with pytest.raises(ValueError, match="exactly the fitting window"):
        validate_fit_window_model_state(leaked, train_end)


def test_blended_raw_operator_is_linear_in_alpha(built_case):
    _, _, _, model = built_case
    fno = model["operators"]["A_FNO"]
    data = model["operators"]["A_data"]
    for alpha in (0.25, 0.5):
        blend = model["operators"][f"A_blend_alpha={alpha}"]
        np.testing.assert_allclose(
            blend["A_raw"],
            alpha * data["A_raw"] + (1.0 - alpha) * fno["A_raw"],
            rtol=2e-11,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            blend["b"],
            alpha * data["b"] + (1.0 - alpha) * fno["b"],
            rtol=2e-11,
            atol=2e-12,
        )


def test_reduced_operators_can_be_refit_without_raw_fno_inference(built_case):
    _, operator, _, model = built_case
    calls = {"count": 0}

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        calls["count"] += 1
        raise AssertionError("the frozen raw FNO must not run during refit")

    original_forward = operator.forward
    operator.forward = fail_if_called
    try:
        refitted = refit_structured_bso_operators(
            model,
            ridge=7e-4,
            fit_affine=False,
            alphas=(0.0, 0.5, 1.0),
            stabilize_max_radius=0.97,
        )
    finally:
        operator.forward = original_forward

    assert calls["count"] == 0
    assert refitted["fingerprint"] != model["fingerprint"]
    assert refitted["ridge"] == 7e-4
    assert not refitted["fit_affine"]
    assert set(refitted["operators"]) == {
        "A_FNO", "A_blend_alpha=0.5", "A_data"}
    for operator_record in refitted["operators"].values():
        assert np.max(np.abs(np.linalg.eigvals(
            operator_record["A"]))) <= 0.97 + 1e-10


def test_materialized_real_schur_matrix_respects_radius():
    raw = np.asarray([
        [1.2, 8.0, 0.0],
        [-0.2, 1.2, 3.0],
        [0.0, 0.0, 1.4],
    ])
    stable, diagnostics = materialize_stable_matrix(
        raw, max_radius=0.91)
    assert np.isrealobj(stable)
    assert np.max(np.abs(np.linalg.eigvals(stable))) <= 0.91 + 1e-10
    assert diagnostics["clipped_modes"] == 3
    assert diagnostics["method"] == "real_schur_block_radial_clip"


def test_decoder_residual_restores_launch_sst(built_case):
    dataset, _, _, model = built_case
    origins = np.asarray([0, 3])
    train_end = int(dataset["meta"]["train_end"])
    trajectory = rollout_structured_bso(
        dataset,
        model,
        origins,
        2,
        operator_key="A_blend_alpha=0.5",
        residual_mode="scalar_ar1",
        origin_base=train_end,
        output_units="raw_normalized",
    )
    expected = dataset["meta"]["state_fields"][
        train_end + origins, 0]
    np.testing.assert_allclose(
        trajectory[0], expected, rtol=0.0, atol=2e-7)


def test_frozen_raw_fno_parameters_are_unchanged(built_case):
    dataset, operator, state_before, model = built_case
    state_after = operator.state_dict()
    assert state_before.keys() == state_after.keys()
    for name, before in state_before.items():
        assert torch.equal(before, state_after[name].detach().cpu())
    assert len(model["coeffs"]) == int(dataset["meta"]["train_end"])
