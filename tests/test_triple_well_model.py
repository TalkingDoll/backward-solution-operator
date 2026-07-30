import numpy as np

from experiments.triple_well.exp_triple_well import build_config, match_reference_spectrum
from kbe_observation_data import TRIPLE_WELL_PARAMS


def test_triple_well_noise_amplitude_is_fixed() -> None:
    diffusion = float(TRIPLE_WELL_PARAMS["D"])
    assert np.isclose(np.sqrt(2.0 * diffusion), 1.09)


def test_triple_well_final_projection_contract() -> None:
    config = build_config(False)
    assert config["training_projection_steps"] == 1
    assert config["projection_steps"] == 3
    assert config["projection_operator"] == "phase_a"
    assert config["symmetrize_K"] is True
    assert np.isclose(config["reg_alpha"], 1.0e-8)


def test_triple_well_matching_is_one_to_one() -> None:
    learned = np.asarray([0.0, -1.0, -10.0], dtype=complex)
    reference = np.asarray([0.0, -0.9, -1.1], dtype=complex)
    matched_idx, errors = match_reference_spectrum(learned, reference)
    assert len(np.unique(matched_idx)) == len(reference)
    assert np.array_equal(matched_idx, np.asarray([0, 1, 2]))
    assert np.allclose(errors, [0.0, 0.1, 8.9])
