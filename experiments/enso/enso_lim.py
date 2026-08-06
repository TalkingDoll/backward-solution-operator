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

Two facts about the paper's construction that are easy to get wrong, both
checked by tests:

* **The Eq. 7 index readout is not an independent modelling choice.** Their
  Eq. 9 shows it is exactly equivalent to propagating the full field and
  area-averaging over the Niño box, because Niño-3.4 is a linear functional of
  the field. A "direct LIM" therefore cannot differ from a propagated LIM by
  readout alone.
* **Whitening does not change the forecast.** Rescaling ``psi -> D psi`` by a
  diagonal ``D`` sends ``G -> D G D^-1`` and ``z -> D^-1 z``, so ``z^T G^n psi``
  is invariant. Whitening is kept because Eq. 7 assumes it (with unit-variance
  PCs the covariance ``<psi_l y>`` *is* the regression coefficient), not because
  it alters predictions. The substantive differences from a lag-1 iterated LIM
  are ``tau0 = 2`` and the fractional powers.
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
    """Paper-faithful LIM: whitened PCs, ``G(tau0)``, fractional powers."""

    propagator: np.ndarray        # G(tau0), acting on whitened PCs
    readout: np.ndarray           # z_l of Eq. 7
    pc_scale: np.ndarray          # whitening factors (training PC std)
    tau0: int

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
        """``(n_origins, horizon)`` Niño-3.4 predictions via Eq. 7."""
        psi = np.asarray(coeff0, dtype=np.float64) / self.pc_scale
        return np.stack([(psi @ self._power(lead).T) @ self.readout
                         for lead in range(1, int(horizon) + 1)], axis=1)

    def coefficient_forecast(self, coeff0: np.ndarray,
                             horizon: int) -> np.ndarray:
        """``(n_origins, horizon, L)`` EOF coefficients, for the field panel.

        Equation 9: decoding these and area-averaging reproduces
        ``index_forecast`` exactly.
        """
        psi = np.asarray(coeff0, dtype=np.float64) / self.pc_scale
        return np.stack([(psi @ self._power(lead).T) * self.pc_scale
                         for lead in range(1, int(horizon) + 1)], axis=1)


def fit_lim_wang(coeff: np.ndarray, index: np.ndarray, tau0: int = 2,
                 ridge: float = 1e-8) -> WangLIM:
    """Fit the benchmark LIM of Wang et al. (2020) on training coefficients.

    ``coeff`` are EOF coefficients over the training period and ``index`` the
    matching Niño-3.4 values. ``tau0 = 1`` reduces this to a lag-1 iterated LIM.
    """
    coeff = np.asarray(coeff, dtype=np.float64)
    index = np.asarray(index, dtype=np.float64)
    tau0 = int(tau0)
    scale = coeff.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)      # Eq. 6 whitening
    psi = coeff / scale
    g_t = _ridge_solve(psi[:-tau0], psi[tau0:], ridge)   # Eq. 11
    readout = (psi * index[:, None]).mean(axis=0)        # Eq. 7
    return WangLIM(propagator=g_t.T, readout=readout, pc_scale=scale,
                   tau0=tau0)


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


def predict_lim_direct(weights: np.ndarray,
                       predictors: np.ndarray) -> np.ndarray:
    """``(n_origins, horizon)`` predictions from ``fit_lim_direct`` weights."""
    x = np.asarray(predictors, dtype=np.float64)
    x = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    return np.stack([x @ w for w in weights], axis=1)
