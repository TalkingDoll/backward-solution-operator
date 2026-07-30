"""Minimal data pipeline for the retained EEGMMIDB S029 R02 experiment."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from project_paths import DATA_DIR


SAMPLING_RATE_HZ = 160.0
HISTORY_SAMPLES = 160
LAG_SAMPLES = 16
RESTING_RUNS = (1, 2)
RESTING_EFFECTIVE_SECONDS = 60.0
RESTING_EFFECTIVE_SAMPLES = int(RESTING_EFFECTIVE_SECONDS * SAMPLING_RATE_HZ)
RESTING_FILTER_WARMUP_SAMPLES = HISTORY_SAMPLES

CANONICAL_CHANNELS = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
)


def _require_mne():
    try:
        return importlib.import_module("mne")
    except ImportError as error:
        raise RuntimeError("Install mne to download and read EEGMMIDB EDF files") from error


def _channel_token(name: str) -> str:
    return "".join(character for character in str(name) if character.isalnum()).upper()


_CANONICAL_BY_TOKEN = {
    _channel_token(channel): channel for channel in CANONICAL_CHANNELS
}


def reorder_to_canonical(data: np.ndarray, channel_names: Sequence[str]) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim != 2 or values.shape[0] != len(channel_names):
        raise ValueError("EEG data and channel_names must have matching channel axes")
    located: dict[str, int] = {}
    for index, raw_name in enumerate(channel_names):
        canonical = _CANONICAL_BY_TOKEN.get(_channel_token(raw_name))
        if canonical is None:
            continue
        if canonical in located:
            raise ValueError(f"duplicate EEG channel: {canonical}")
        located[canonical] = index
    missing = [channel for channel in CANONICAL_CHANNELS if channel not in located]
    if missing:
        raise ValueError(f"missing canonical EEG channels: {missing}")
    return np.ascontiguousarray(values[[located[channel] for channel in CANONICAL_CHANNELS]])


def standard_montage_coordinates() -> np.ndarray:
    montage = _require_mne().channels.make_standard_montage("standard_1005")
    positions = {
        _channel_token(name): np.asarray(value, dtype=np.float64)
        for name, value in montage.get_positions()["ch_pos"].items()
    }
    coordinates = np.stack([positions[_channel_token(name)] for name in CANONICAL_CHANNELS])
    if coordinates.shape != (64, 3) or not np.all(np.isfinite(coordinates)):
        raise RuntimeError("invalid standard-1005 coordinates")
    return coordinates


def _find_local_edf(root: Path, subject: int, run: int) -> Path | None:
    subject_dir = f"S{subject:03d}"
    filename = f"S{subject:03d}R{run:02d}.edf"
    candidates = (
        root / subject_dir / filename,
        root / "files" / "eegmmidb" / "1.0.0" / subject_dir / filename,
        root / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / subject_dir / filename,
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def causal_car_bandpass(data: np.ndarray, *, sfreq: float = SAMPLING_RATE_HZ) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 64 or not np.all(np.isfinite(values)):
        raise ValueError("continuous EEG must be finite with shape (64,time)")
    car = values - values.mean(axis=0, keepdims=True)
    sos = butter(4, (8.0, 30.0), btype="bandpass", fs=float(sfreq), output="sos")
    return np.ascontiguousarray(sosfilt(sos, car, axis=-1), dtype=np.float32)


@dataclass(frozen=True)
class ChannelNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    epsilon: float

    def __post_init__(self) -> None:
        mean = np.ascontiguousarray(self.mean, dtype=np.float64)
        scale = np.ascontiguousarray(self.scale, dtype=np.float64)
        if mean.shape != (64, 1) or scale.shape != (64, 1):
            raise ValueError("normalizer arrays must have shape (64,1)")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("normalizer statistics must be finite and positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "epsilon", float(self.epsilon))

    def transform(self, data: np.ndarray) -> np.ndarray:
        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != 64:
            raise ValueError("normalizer input must have shape (64,time)")
        return np.ascontiguousarray((values - self.mean) / self.scale, dtype=np.float32)


@dataclass(frozen=True)
class ForecastBatch:
    history: np.ndarray
    next_history: np.ndarray
    sample_weights: np.ndarray


@dataclass(frozen=True)
class RestingEEGRun:
    subject: int
    run: int
    data: np.ndarray
    coordinates: np.ndarray
    source: str

    def __post_init__(self) -> None:
        data = np.ascontiguousarray(self.data, dtype=np.float32)
        coordinates = np.ascontiguousarray(self.coordinates, dtype=np.float64)
        if not 1 <= int(self.subject) <= 109 or int(self.run) not in RESTING_RUNS:
            raise ValueError("invalid EEGMMIDB subject or resting run")
        if data.shape != (64, RESTING_EFFECTIVE_SAMPLES) or not np.all(np.isfinite(data)):
            raise ValueError("resting EEG must be a finite 64 x 9600 record")
        if coordinates.shape != (64, 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must have shape (64,3)")
        object.__setattr__(self, "subject", int(self.subject))
        object.__setattr__(self, "run", int(self.run))
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "source", str(self.source))


@dataclass(frozen=True)
class RestingRolloutBatch:
    initial_history: np.ndarray
    future: np.ndarray
    lag_samples: int


@dataclass(frozen=True)
class RestingRolloutSplit:
    subject: int
    run: int
    split_sample: int
    evaluation_end_sample: int
    lag_samples: int
    train_origin_samples: tuple[int, ...]
    normalizer: ChannelNormalizer
    batch: ForecastBatch
    rollout: RestingRolloutBatch


def load_resting_eeg_run(
    subject: int,
    run: int,
    data_root: str | Path = DATA_DIR / "eegmmidb",
    *,
    download: bool = False,
) -> RestingEEGRun:
    subject, run = int(subject), int(run)
    if not 1 <= subject <= 109 or run not in RESTING_RUNS:
        raise ValueError("subject must be in 1..109 and run must be R01 or R02")
    root = Path(data_root).expanduser()
    path = _find_local_edf(root, subject, run)
    if path is None:
        if not download:
            raise FileNotFoundError(f"missing S{subject:03d}R{run:02d}.edf under {root}")
        root.mkdir(parents=True, exist_ok=True)
        fetched = _require_mne().datasets.eegbci.load_data(
            subject=subject,
            runs=[run],
            path=str(root),
            update_path=False,
            verbose="ERROR",
        )
        if len(fetched) != 1:
            raise FileNotFoundError("MNE did not return one resting EDF")
        path = Path(fetched[0]).resolve()

    mne = _require_mne()
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    try:
        sfreq = float(raw.info["sfreq"])
        if not np.isclose(sfreq, SAMPLING_RATE_HZ):
            raise ValueError(f"expected 160 Hz, found {sfreq:g} Hz")
        descriptions = tuple(str(value) for value in raw.annotations.description)
        onsets = np.asarray(raw.annotations.onset, dtype=np.float64)
        durations = np.asarray(raw.annotations.duration, dtype=np.float64)
        if descriptions != ("T0",) or onsets.shape != (1,) or durations.shape != (1,):
            raise ValueError("EDF is not one continuous T0 resting baseline")
        if not np.isclose(onsets[0], 0.0) or durations[0] < RESTING_EFFECTIVE_SECONDS:
            raise ValueError("resting annotation does not cover 60 seconds")
        canonical = reorder_to_canonical(raw.get_data(), raw.ch_names)
        if canonical.shape[-1] < RESTING_EFFECTIVE_SAMPLES:
            raise ValueError("resting EDF is shorter than 60 seconds")
        filtered = causal_car_bandpass(canonical[:, :RESTING_EFFECTIVE_SAMPLES], sfreq=sfreq)
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()
    return RestingEEGRun(subject, run, filtered, standard_montage_coordinates(), str(path))


def make_resting_rollout_split(
    resting_run: RestingEEGRun,
    *,
    lag_samples: int,
    train_stride_samples: int = 20,
    evaluation_horizon_samples: int = 640,
    train_origin_start_sample: int = 320,
) -> RestingRolloutSplit:
    """Build the fixed 0--15 s fit and continuous boundary rollout."""

    lag = int(lag_samples)
    stride = int(train_stride_samples)
    if lag < 1 or HISTORY_SAMPLES % lag != 0 or stride < 1:
        raise ValueError("lag must divide 160 and stride must be positive")
    if int(evaluation_horizon_samples) < lag or int(evaluation_horizon_samples) % lag:
        raise ValueError("evaluation horizon must be a positive lag multiple")
    split_sample = int(15 * SAMPLING_RATE_HZ)
    evaluation_end_sample = split_sample + int(evaluation_horizon_samples)
    if evaluation_end_sample > resting_run.data.shape[1]:
        raise ValueError("evaluation horizon exceeds the resting record")
    epsilon = 1.0e-6
    training = np.asarray(resting_run.data[:, RESTING_FILTER_WARMUP_SAMPLES:split_sample], dtype=np.float64)
    mean = training.mean(axis=1, keepdims=True)
    raw_scale = training.std(axis=1, keepdims=True)
    normalizer = ChannelNormalizer(
        mean,
        np.where(raw_scale >= epsilon, raw_scale, 1.0),
        epsilon,
    )
    normalized = normalizer.transform(resting_run.data)
    origins = tuple(range(int(train_origin_start_sample), split_sample - lag + 1, stride))
    histories = np.stack([normalized[:, origin - HISTORY_SAMPLES:origin] for origin in origins])
    next_histories = np.stack([
        normalized[:, origin - HISTORY_SAMPLES + lag:origin + lag] for origin in origins
    ])
    n_pairs = len(origins)
    batch = ForecastBatch(
        history=np.ascontiguousarray(histories, dtype=np.float32),
        next_history=np.ascontiguousarray(next_histories, dtype=np.float32),
        sample_weights=np.full(n_pairs, 1.0 / n_pairs, dtype=np.float64),
    )
    rollout = RestingRolloutBatch(
        initial_history=np.ascontiguousarray(
            normalized[
                None,
                :,
                split_sample - HISTORY_SAMPLES:split_sample,
            ],
            dtype=np.float32,
        ),
        future=np.ascontiguousarray(
            normalized[None, :, split_sample:evaluation_end_sample],
            dtype=np.float32,
        ),
        lag_samples=lag,
    )
    return RestingRolloutSplit(
        subject=resting_run.subject,
        run=resting_run.run,
        split_sample=split_sample,
        evaluation_end_sample=evaluation_end_sample,
        lag_samples=lag,
        train_origin_samples=origins,
        normalizer=normalizer,
        batch=batch,
        rollout=rollout,
    )


def restore_resting_signal_units(values: Any, normalizer: ChannelNormalizer) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 4:
        return array * normalizer.scale[None, None, :, :] + normalizer.mean[None, None, :, :]
    if array.ndim == 3:
        return array * normalizer.scale[None, :, :] + normalizer.mean[None, :, :]
    raise ValueError("values must have trial/step/channel or trial/channel axes")


__all__ = [
    "CANONICAL_CHANNELS",
    "HISTORY_SAMPLES",
    "LAG_SAMPLES",
    "SAMPLING_RATE_HZ",
    "ChannelNormalizer",
    "ForecastBatch",
    "RestingRolloutSplit",
    "RestingEEGRun",
    "RestingRolloutBatch",
    "load_resting_eeg_run",
    "make_resting_rollout_split",
    "restore_resting_signal_units",
]
