"""LIM baselines following Wang, Slawinska & Giannakis (2020).

Reference: *Extended-range statistical ENSO prediction through operator-theoretic
techniques for nonlinear dynamics*, Sci. Rep. **10**:2636, Methods Eqs. (5)-(12).

Two estimators are provided, and they are genuinely different objects:

``fit_lim_wang``
    the paper's benchmark LIM -- whitened principal components (Eq. 6), a
    propagator estimated at ``tau0 = 2`` months (Eq. 11), other leads by the
    fractional power ``G(tau) = G(tau0)^(tau/tau0)`` (Eq. 12), and the Niño-3.4
    readout of Eq. 7.
``fit_lim_direct``
    a direct lagged linear regression, fitting each lead independently with no
    propagation at all. This is the *linear counterpart of the direct
    field-to-index FNO*, not a LIM.

``fit_direct_cv`` is the same estimator with its ridge selected by blocked
cross-validation instead of left at a default. It exists so that a comparison
against a linear baseline measures the hypothesis class rather than how well
each side happened to be regularised.

Three facts about the paper's construction that are easy to get wrong, all
checked by tests:

* **The Eq. 7 index readout is not an independent modelling choice.** Their
  Eq. 9 shows it is exactly equivalent to propagating the full field and
  area-averaging over the Niño box, because Niño-3.4 is a linear functional of
  the field. A "direct LIM" therefore cannot differ from a propagated LIM by
  readout alone.
* **Whitening does not change the forecast, but centring does.** Rescaling
  ``psi -> D psi`` by a diagonal ``D`` sends ``G -> D G D^-1`` and
  ``z -> D^-1 z``, so ``z^T G^n psi`` is invariant. Whitening is kept because
  Eq. 7 assumes it (with unit-variance PCs the covariance ``<psi_l y>`` *is* the
  regression coefficient), not because it alters predictions. Subtracting the
  mean is a different matter: ``<psi_l y>`` is a covariance only once *both*
  factors are centred, and there is no intercept in Eq. 7 to absorb a nonzero
  ``<y>``. ``encode`` projects raw anomalies without removing the mean that
  ``fit_eofs`` subtracted before its SVD, so on this record the PC mean carries
  a third of the signal and an uncentred Eq. 7 readout is badly wrong -- at
  ``tau = 0``, where there is no propagation error at all, it scores 0.897 °C
  RMSE against the 0.056 °C reached by ordinary least squares on the same 20
  features. Centred, it reaches 0.0556 °C.
* **The forecast must decay to the climatology the index is defined against.**
  Niño-3.4 is already an anomaly, against a base period much warmer than the
  full fitting record, so the record mean is *not* zero (-0.387 °C here). The
  propagator is therefore fitted on centred whitened PCs but applied to
  uncentred ones, with no intercept added back, so ``G(tau) -> 0`` sends the
  forecast to zero anomaly rather than to the record mean. Verified equal to
  Eq. 9's decode-and-area-average form to 4e-14, which is what makes this the
  faithful reading rather than a convenience: Eq. 9 proves the two are the same
  object, and the decode form needs no fitted intercept in the first place.

The substantive differences from a lag-1 iterated LIM remain ``tau0 = 2`` and
the fractional powers. ``exp_enso.fit_lim`` (the ``lim_prop`` baseline) is left
uncentred on purpose: the record mean is a near-fixed-point of its ``G`` (2 %
residual), so centring would change only its asymptote, and its
decode-and-area-average readout already reverts to zero anomaly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


def _ridge_solve(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    gram = x.T @ x
    reg = float(ridge) * np.trace(gram) / max(gram.shape[0], 1)
    return np.linalg.solve(gram + reg * np.eye(gram.shape[0]), x.T @ y)


@dataclass
class WangLIM:
    """Paper-faithful LIM: whitened PCs, ``G(tau0)``, fractional powers.

    ``pc_mean`` and ``index_mean`` are the training means the fit was centred on.
    Neither enters a forecast -- the forecast propagates *uncentred* whitened PCs
    so it decays to zero anomaly -- but they are kept because their consistency is
    the sharpest available check on the readout: Eq. 9 implies
    ``(pc_mean / pc_scale) @ readout == index_mean``, which holds to 0.6 % on the
    ENSO record and is violated grossly by an uncentred fit.
    """

    propagator: np.ndarray        # G(tau0), acting on whitened PCs
    readout: np.ndarray           # z_l of Eq. 7, a genuine covariance
    pc_scale: np.ndarray          # whitening factors (training PC std)
    tau0: int
    pc_mean: np.ndarray           # training PC mean, for the Eq. 9 check
    index_mean: float             # training index mean, likewise

    def _power(self, lead: int) -> np.ndarray:
        """``G(tau) = G(tau0)^(tau/tau0)`` (Eq. 12).

        Fractional powers of a real matrix with complex eigenvalues can acquire
        a small imaginary part; it is discarded only after checking it is
        negligible, so a genuinely complex result is not silently hidden.
        """
        if int(lead) % int(self.tau0) == 0:
            return np.linalg.matrix_power(self.propagator,
                                          int(lead) // int(self.tau0))
        g = scipy.linalg.fractional_matrix_power(self.propagator,
                                                 float(lead) / float(self.tau0))
        if np.iscomplexobj(g):
            relative = (np.linalg.norm(g.imag)
                        / (np.linalg.norm(g.real) + np.finfo(float).eps))
            if relative > 1e-7:
                raise np.linalg.LinAlgError(
                    f"fractional LIM power at lead {lead} is complex "
                    f"(relative imaginary part {relative:.2e})")
            g = g.real
        return g

    def index_forecast(self, coeff0: np.ndarray, horizon: int) -> np.ndarray:
        """``(n_origins, horizon)`` Niño-3.4 predictions via Eq. 7.

        The whitened PC is *not* re-centred and no intercept is added, so as
        ``G(tau)`` decays the forecast goes to zero anomaly -- the climatology
        Niño-3.4 is defined against -- rather than to the fitting record's mean.
        """
        psi = np.asarray(coeff0, dtype=np.float64) / self.pc_scale
        return np.stack([(psi @ self._power(lead).T) @ self.readout
                         for lead in range(1, int(horizon) + 1)], axis=1)

    def coefficient_forecast(self, coeff0: np.ndarray,
                             horizon: int) -> np.ndarray:
        """``(n_origins, horizon, L)`` EOF coefficients, for the field panel.

        Equation 9: decoding these and area-averaging reproduces
        ``index_forecast`` exactly. Same convention as
        :meth:`index_forecast` -- uncentred in, nothing added back.
        """
        psi = np.asarray(coeff0, dtype=np.float64) / self.pc_scale
        return np.stack([(psi @ self._power(lead).T) * self.pc_scale
                         for lead in range(1, int(horizon) + 1)], axis=1)


def fit_lim_wang(coeff: np.ndarray, index: np.ndarray, tau0: int = 2,
                 ridge: float = 1e-8) -> WangLIM:
    """Fit the benchmark LIM of Wang et al. (2020) on training coefficients.

    ``coeff`` are EOF coefficients over the training period and ``index`` the
    matching Niño-3.4 values. ``tau0 = 1`` reduces this to a lag-1 iterated LIM
    *on centred coefficients*; it does not reduce to ``exp_enso.fit_lim``, which
    is deliberately uncentred (see the module docstring).

    The propagator and the readout are both estimated on **centred** whitened
    PCs, because Eq. 7's ``<psi_l y>`` is the regression coefficient it is meant
    to be only when both factors have zero mean. The forecast then propagates
    uncentred PCs so it decays to zero anomaly.
    """
    coeff = np.asarray(coeff, dtype=np.float64)
    index = np.asarray(index, dtype=np.float64)
    tau0 = int(tau0)
    pc_mean = coeff.mean(axis=0)
    scale = coeff.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)      # Eq. 6 whitening
    psi = (coeff - pc_mean) / scale
    index_mean = float(index.mean())
    g_t = _ridge_solve(psi[:-tau0], psi[tau0:], ridge)          # Eq. 11
    readout = (psi * (index - index_mean)[:, None]).mean(axis=0)  # Eq. 7
    return WangLIM(propagator=g_t.T, readout=readout, pc_scale=scale,
                   tau0=tau0, pc_mean=pc_mean, index_mean=index_mean)


def fit_lim_direct(predictors: np.ndarray, index: np.ndarray,
                   origins: np.ndarray, horizon: int,
                   ridge: float = 1e-6) -> np.ndarray:
    """Direct lagged linear regression, one independent fit per lead.

    Returns ``(horizon, n_features + 1)`` coefficients including an intercept.
    No propagation is involved: this is the linear counterpart of the direct
    field-to-index FNO, so comparing the two isolates linearity rather than the
    training target.
    """
    origins = np.asarray(origins, dtype=int)
    x = np.asarray(predictors, dtype=np.float64)[origins]
    x = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    return np.stack([_ridge_solve(x, np.asarray(index)[origins + lead], ridge)
                     for lead in range(1, int(horizon) + 1)])


RIDGE_GRID = np.logspace(-8.0, 2.0, 21)


def fit_direct_cv(predictors: np.ndarray, index: np.ndarray,
                  origins: np.ndarray, horizon: int, grid=None,
                  n_folds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """``fit_lim_direct`` with a per-lead ridge chosen by blocked CV.

    Returns ``(weights, ridges)``; ``weights`` has the same shape and meaning as
    ``fit_lim_direct``'s. Selection uses ``origins`` only, so it stays inside the
    training period wherever the caller's origins do.

    The folds are **contiguous blocks** of ``origins``, not random subsets.
    Consecutive months of a geophysical index are strongly autocorrelated, so a
    random split places near-duplicate rows on both sides of every fold, makes
    the held-out error look like training error, and selects a ridge far too
    small. That failure mode is exactly what a fixed default ridge already risks,
    and it is the reason a high-dimensional predictor set can appear to lose to a
    low-dimensional one when it has merely been left unregularised.

    Every lead is solved together for a given fold and ridge: the Gram matrix
    depends only on the predictors, so the cost is ``n_folds * len(grid)``
    solves rather than one per lead.
    """
    origins = np.asarray(origins, dtype=int)
    grid = RIDGE_GRID if grid is None else np.asarray(grid, dtype=np.float64)
    x = np.asarray(predictors, dtype=np.float64)
    y = np.asarray(index, dtype=np.float64)
    horizon = int(horizon)

    design = np.concatenate([np.ones((len(origins), 1)), x[origins]], axis=1)
    targets = np.stack([y[origins + lead] for lead in range(1, horizon + 1)],
                       axis=1)

    errors = np.zeros((len(grid), horizon))
    for fold in np.array_split(np.arange(len(origins)), int(n_folds)):
        held = np.zeros(len(origins), dtype=bool)
        held[fold] = True
        xt, xe = design[~held], design[held]
        gram = xt.T @ xt
        unit = np.trace(gram) / max(gram.shape[0], 1)
        cross = xt.T @ targets[~held]
        for i, ridge in enumerate(grid):
            w = np.linalg.solve(
                gram + float(ridge) * unit * np.eye(gram.shape[0]), cross)
            errors[i] += np.sum((xe @ w - targets[held]) ** 2, axis=0)

    ridges = grid[np.argmin(errors, axis=0)]
    weights = np.stack([_ridge_solve(design, targets[:, lead], ridges[lead])
                        for lead in range(horizon)])
    return weights, ridges


def predict_lim_direct(weights: np.ndarray,
                       predictors: np.ndarray) -> np.ndarray:
    """``(n_origins, horizon)`` predictions from ``fit_lim_direct`` weights."""
    x = np.asarray(predictors, dtype=np.float64)
    x = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    return np.stack([x @ w for w in weights], axis=1)
