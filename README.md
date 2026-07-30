# Neural Operators Meet Kolmogorov Backward Operators

This repository studies neural approximations of backward solution operators
and their finite-dimensional BSO projections for forecasting and spectral
analysis. The examples span stochastic systems, chaotic dynamics,
geophysical fields, and real EEG observations.

## Method overview

[![BSO computational framework](bso_schematic.png)](bso_schematic.pdf)

Click the schematic to open the full-resolution PDF.

## Quick start

Run notebooks and module commands from the repository root. Paths to
`data/`, `results/`, and `figures/` are resolved by `project_paths.py`, so no
machine-specific path edit is required.

A tested Python 3.11 environment can be installed with:

```text
python -m pip install jupyter numpy==1.26.4 scipy==1.14.1 matplotlib==3.10.1 mne==1.8.0 torch==2.5.1 neuraloperator==1.0.2 pytest
```

CUDA is optional. Additional dataset-specific dependencies and complete
environment details are documented in
[experimental_detail.md](experimental_detail.md).

## Experiments

| Notebook | Experiment | Clean-checkout behavior |
|---|---|---|
| `nb_ou.ipynb` | Ornstein--Uhlenbeck spectrum | Generates data, fits the models, and creates the figures. |
| `nb_triple_well.ipynb` | Triple-well metastable spectrum | Generates data, fits the models, and creates the figures. |
| `nb_lorenz63.ipynb` | Lorenz-63 rollout | Uses versioned reference output when local training artifacts are unavailable. |
| `nb_lorenz96.ipynb` | Lorenz-96 rollout | Uses versioned reference output when local training artifacts are unavailable. |
| `nb_sst.ipynb` | Global SST rollout | Uses versioned reference output when locked evaluation artifacts are unavailable. |
| `nb_eeg.ipynb` | Resting-state EEG rollout | Downloads the required public record and reruns the experiment. |

## Reproducibility

All six notebooks run from a clean checkout without path changes; the first
EEG run requires network access for its public data download. The
Ornstein--Uhlenbeck, triple-well, and EEG notebooks perform their computations
directly. The Lorenz-63, Lorenz-96, and SST notebooks display committed
reference results when their large, ignored datasets or checkpoints are not
available.

Complete dataset definitions, preprocessing, model configurations,
hyperparameters, evaluation metrics, numerical results, and interpretation
limits are recorded in
[experimental_detail.md](experimental_detail.md).

## Repository layout and tests

- `experiments/<name>/`: experiment-specific implementations.
- `nb_*.ipynb`: notebook entry points.
- `figures/`: versioned publication figures.
- `edmd_bso_method.py`, `kbe_observation_data.py`: shared learning and data utilities.
- `solver_edmd.py`, `solver_resdmd.py`: finite-dimensional spectral solvers.
- `plotting_utils.py`, `project_paths.py`: shared plotting and path utilities.
- `tests/`: model, metric, protocol, and stability tests.

Run the test suite with:

```text
python -m pytest tests -q
```

Raw data, checkpoints, and runtime results under `data/` and `results/` are
ignored by Git.
